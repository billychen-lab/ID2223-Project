# app.py
import streamlit as st

from bike_agent import answer_query

st.set_page_config(page_title="Citibike Smart Assistant", page_icon="🚲")

st.title("🚲 Citibike Smart Assistant")
st.write("An assistant powered by real-time prediction (Hopsworks) + RAG (FAISS) + OpenAI.")

st.markdown("""
You can ask me:
- Where is it easiest to rent a bike in NYC in the next hour?
- If I'm near a station, how many people will likely rent a bike in the next hour?
- Which stations will be the busiest?
""")

# ----------------------------
# Sidebar: Tips & Controls
# ----------------------------
with st.sidebar:
    st.header("Run Mode")
    st.caption("Goal: refresh the prediction CSV and rebuild the RAG index on every query.")
    st.session_state["refresh_each_query"] = st.toggle(
        "Refresh prediction & index on every query (slower but freshest)",
        value=True,
    )
    st.divider()
    st.caption("Tip: If the Space is slow on first launch, it is usually downloading models/dependencies.")

# ----------------------------
# Chat history
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Refreshing prediction + building index + thinking..."):
            try:
                # bike_agent.py will run: feature6 -> build_rag_index -> retrieve -> OpenAI
                # If you want to pass refresh_each_query into bike_agent, bike_agent must support this parameter.
                answer = answer_query(user_input)
            except Exception as e:
                answer = f"Runtime error: {e}"

            st.markdown(answer)

    st.session_state.history.append(("assistant", answer))
