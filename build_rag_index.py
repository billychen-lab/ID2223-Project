# build_rag_index.py
import os
import pickle
from functools import lru_cache
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


# HF Spaces recommendation: write to /tmp
DEFAULT_PRED_CSV = os.getenv(
    "PRED_CSV",
    "/tmp/predictions_with_lag_next_hour_all_stations_ensemble.csv",
)
DEFAULT_INDEX_DIR = os.getenv(
    "INDEX_DIR",
    "/tmp/rag_index",
)

DEFAULT_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


def _to_int(x, default=0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def _to_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


@lru_cache(maxsize=1)
def _get_embed_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Cache the embedding model with lru_cache to avoid re-downloading/re-loading on every rebuild.
    This is very important for HF Spaces / Streamlit.
    """
    return SentenceTransformer(model_name)


def df_to_docs(df: pd.DataFrame) -> Tuple[List[str], List[Dict]]:
    """
    Convert each station row into a RAG document text + metadata.
    """
    required_cols = ["start_station_id"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    docs: List[str] = []
    metadatas: List[Dict] = []

    for _, row in df.iterrows():
        sid = str(row.get("start_station_id", ""))
        name = row.get("start_station_name", "") or ""

        bikes = _to_int(row.get("num_bikes_available", 0))
        docks = _to_int(row.get("num_docks_available", 0))

        lag1 = _to_float(row.get("lag_1h", 0))
        lag24 = _to_float(row.get("lag_24h", 0))

        y_hist = _to_float(row.get("predicted_trips_hist", 0))
        y_rt = _to_float(row.get("predicted_trips_realtime", 0))
        y_ens = _to_float(row.get("predicted_trips_ensemble", 0))

        text = (
            f"Station {sid} ({name}) currently has {bikes} bikes available and {docks} empty docks. "
            f"In the previous hour, {lag1:.1f} trips were made; "
            f"at the same time 24 hours ago, {lag24:.1f} trips were made. "
            f"The historical model predicts {y_hist:.1f} trips in the next hour; "
            f"the real-time occupancy-based model predicts {y_rt:.1f} trips; "
            f"the ensemble prediction is {y_ens:.1f} trips."
        )

        docs.append(text)
        metadatas.append(
            {
                "start_station_id": sid,
                "start_station_name": name,
                "num_bikes_available": bikes,
                "num_docks_available": docks,
                "predicted_trips_ensemble": y_ens,
            }
        )

    return docs, metadatas


def build_rag_index(
    pred_csv: str = DEFAULT_PRED_CSV,
    index_dir: str = DEFAULT_INDEX_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    save: bool = True,
) -> Tuple[faiss.Index, List[str], List[Dict]]:
    """
    Read the prediction CSV -> build docs -> compute embeddings -> build FAISS index.
    Returns: index, docs, metadatas
    When save=True, index/docs/metadatas are saved to index_dir (recommended: /tmp/rag_index).
    """
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

    df = pd.read_csv(pred_csv)
    docs, metadatas = df_to_docs(df)

    model = _get_embed_model(model_name)
    embeddings = model.encode(docs, convert_to_numpy=True, show_progress_bar=False)

    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise RuntimeError("Embedding output is not a 2D numpy array.")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    if save:
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "docs.pkl"), "wb") as f:
            pickle.dump(docs, f)
        with open(os.path.join(index_dir, "metadatas.pkl"), "wb") as f:
            pickle.dump(metadatas, f)

    return index, docs, metadatas


def load_rag_index(index_dir: str = DEFAULT_INDEX_DIR) -> Tuple[faiss.Index, List[str], List[Dict]]:
    """
    Load index/docs/metadatas from disk.
    """
    index_path = os.path.join(index_dir, "faiss.index")
    docs_path = os.path.join(index_dir, "docs.pkl")
    metas_path = os.path.join(index_dir, "metadatas.pkl")

    if not (os.path.exists(index_path) and os.path.exists(docs_path) and os.path.exists(metas_path)):
        raise FileNotFoundError(f"RAG index files not found in: {index_dir}")

    index = faiss.read_index(index_path)
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)
    with open(metas_path, "rb") as f:
        metadatas = pickle.load(f)

    return index, docs, metadatas


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", default=DEFAULT_PRED_CSV)
    parser.add_argument("--index_dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()

    idx, docs, metas = build_rag_index(
        pred_csv=args.pred_csv,
        index_dir=args.index_dir,
        model_name=args.model_name,
        save=True,
    )
    print(f"Prepared {len(docs)} documents.")
    print(f"FAISS index built and saved to: {args.index_dir}")

