# bike_agent.py

import os
import threading
from functools import lru_cache
from typing import List, Dict, Tuple

import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# =========================
# Environment / paths
# =========================

PRED_CSV = os.getenv(
    "PRED_CSV",
    "/tmp/predictions_with_lag_next_hour_all_stations_ensemble.csv",
)

INDEX_DIR = os.getenv(
    "INDEX_DIR",
    "/tmp/rag_index",
)

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

FEATURE6_MODULE = os.getenv(
    "FEATURE6_MODULE",
    "feature6_predict_next_hour_with_lag_all_stations",
)

# =========================
# Global state
# =========================

_lock = threading.Lock()
_state = {
    "index": None,   # faiss.Index
    "docs": None,    # List[str]
    "metas": None,   # List[dict]
}

# =========================
# System prompt
# =========================

SYSTEM_PROMPT = """You are a Citi Bike intelligent assistant.
You have access to:
- Current bike availability at each station
- Current empty dock counts
- Model-predicted bike demand for the next hour
You must answer based on the provided station information.
Prefer recommending 3–5 stations and explain why.
Keep answers concise and clear.
"""

# =========================
# Fixed output when tool is NOT used
# =========================

TOOL_EXPLANATION_TEXT = """I am a Citi Bike station analysis and demand forecasting tool.
I can help you with:
- Finding stations with many available bikes for renting
- Recommending stations with enough empty docks for returning bikes
- Predicting which stations may be busiest in the next hour
You can ask me questions like:
- "Where is the easiest place to rent a bike right now?"
- "Predict which stations will be busiest in the next hour"
- "I am near a station, where should I return my bike?"
If your question is not related to Citi Bike stations, renting, returning,
or short-term demand prediction, I will not call any data or prediction tools.
"""

# =========================
# Cached helpers
# =========================

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAI(api_key=api_key)

# =========================
# Tool: refresh predictions + rebuild RAG
# =========================

def refresh_predictions_and_rag() -> None:
    """
    High-cost world-model tool:
    - Run feature6 prediction pipeline
    - Rebuild FAISS RAG index
    """
    with _lock:
        print("[TOOL] refresh_predictions_and_rag called")

        # 1) Run feature6 prediction
        try:
            mod = __import__(FEATURE6_MODULE, fromlist=["main"])
            mod.main()
        except Exception as e:
            raise RuntimeError(f"feature6 run failed: {e}")

        # 2) Build RAG index
        try:
            from build_rag_index import build_rag_index
            index, docs, metas = build_rag_index(
                pred_csv=PRED_CSV,
                index_dir=INDEX_DIR,
                model_name=EMBED_MODEL_NAME,
                save=True,
            )
        except Exception as e:
            raise RuntimeError(f"build_rag_index failed: {e}")

        _state["index"] = index
        _state["docs"] = docs
        _state["metas"] = metas

# =========================
# RAG retrieval
# =========================

def retrieve_context(query: str, k: int = 5) -> Tuple[List[str], List[Dict]]:
    if _state["index"] is None or _state["docs"] is None:
        refresh_predictions_and_rag()

    embedder = get_embedder()
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = _state["index"].search(q_emb, k)

    idxs = I[0].tolist()
    docs = [_state["docs"][i] for i in idxs]
    metas = [_state["metas"][i] for i in idxs]
    return docs, metas

# =========================
# Agent router (keyword-based)
# =========================

def should_use_tool(user_query: str) -> bool:
    """
    Decide whether to trigger the Citi Bike world-model tool
    using an LLM-based router.
    The LLM must decide whether the question requires:
    - Citi Bike station data
    - bike availability analysis
    - short-term (next-hour) demand prediction
    The model must answer ONLY with YES or NO.
    """

    if not user_query or not user_query.strip():
        return False

    client = get_openai_client()

    router_prompt = f"""
You are an agent router.
Decide whether the following user question requires
Citi Bike station data or short-term bike demand prediction.
Examples of questions that REQUIRE data/tools:
- Where can I rent a bike right now?
- 现在纽约哪里最方便借车？
- Predict which stations will be busiest in the next hour
- 我现在在某个站附近，哪里还车最稳妥？
Examples of questions that do NOT require data/tools:
- What is 1 + 1?
- 你是谁？
- 给我讲个笑话
- How does a bike work?
If the question requires Citi Bike data or prediction, answer YES.
Otherwise, answer NO.
User question:
{user_query}
Answer with only YES or NO.
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": router_prompt}
        ],
        temperature=0,
        max_tokens=5,
    )

    decision = completion.choices[0].message.content.strip().upper()
    return decision.startswith("YES")

# =========================
# Main agent entry
# =========================

def answer_query(user_query: str) -> str:
    """
    Main agent entry point.
    This is the ONLY place where tool usage is decided.
    """

    # 1) Agent decision
    if not should_use_tool(user_query):
        return TOOL_EXPLANATION_TEXT

    # 2) Trigger tool only when needed
    refresh_predictions_and_rag()

    # 3) Retrieve relevant stations
    docs, metas = retrieve_context(user_query, k=5)
    context_text = "\n\n".join(
        f"[Station {i+1}] {doc}" for i, doc in enumerate(docs)
    )

    # 4) Ask LLM to answer
    user_prompt = (
        "The following station information may be relevant:\n"
        f"{context_text}\n\n"
        f"User question: {user_query}\n\n"
        "Please answer based ONLY on the station information above. "
        "If the information is insufficient, clearly explain the uncertainty."
    )

    client = get_openai_client()
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=450,
    )

    return completion.choices[0].message.content

