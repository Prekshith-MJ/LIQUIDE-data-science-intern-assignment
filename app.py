# app.py
import streamlit as st
from agent import query_agent   # ✅ correct import

st.set_page_config(page_title="Research Agent", layout="wide")
st.title("Research Agent (RAG with OpenRouter)")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.info("RAG from ingested PDF first, web fallback. Run ingest.py once.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter query:"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            try:
                response = query_agent(prompt)  # ✅ vectorstore first, then web fallback
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})