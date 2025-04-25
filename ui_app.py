# ui_app.py

import streamlit as st
import sys
import os
import logging
from dotenv import load_dotenv
from typing import List, Dict

# --- Add project root to path ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# --- Import necessary components ---
try:
    from orchestrator.main_app import is_retriever_ready, query_rag_stream
    # --- Remove extract_final_answer import ---
    # from orchestrator.prompt_templates import extract_final_answer
except ImportError as e:
    st.error(f"Error importing orchestrator modules: {e}.")
    st.stop()

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
CHAT_HISTORY_LENGTH = 3

# --- Page Configuration ---
st.set_page_config(
    page_title="NutiSmart RAG Assistant",
    page_icon="🍎",
    layout="wide"
)

# --- Initialization Check ---
retriever_is_ready_flag = is_retriever_ready()

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        logger.info("Chat history cleared by user.")
        st.session_state.messages.append({
           "role": "assistant",
           "content": "Chat history cleared. How can I help you?"
        })
        st.rerun()
    st.markdown("---")
    st.subheader("Status")
    if retriever_is_ready_flag: st.success("Retriever Initialized")
    else: st.error("Retriever FAILED to Initialize")
    st.markdown("---")
    st.info("Ask questions about nutrition and fitness based on the loaded documents.")

# --- Main Chat Interface ---
st.title("🍎 NutiSmart RAG Assistant")
st.caption("Your AI assistant for Nutrition and Fitness questions.")

if not retriever_is_ready_flag:
    st.error("FATAL ERROR: Backend retriever not ready. Cannot process queries.")
    st.stop()

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! How can I help you with nutrition and fitness questions today?"
    })

# --- Display Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("Ask your question here..."):
    logger.info(f"User input received: {prompt}")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history_to_send: List[Dict[str, str]] = st.session_state.messages[-(CHAT_HISTORY_LENGTH * 2):-1]
    logger.info(f"Sending last {len(history_to_send)} messages as history.")

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        logger.info("Calling query_rag_stream function with history...")
        try:
            response_stream = query_rag_stream(prompt, chat_history=history_to_send)
            if response_stream is None:
                 st.error("Could not get response stream from the backend.")
                 full_response = "Error: Failed to get response stream."
            else:
                # write_stream displays the output and returns the concatenated string
                full_response = st.write_stream(response_stream)
                logger.info("Finished streaming response.")
        except Exception as e:
            logger.exception(f"Error during RAG stream processing: {e}", exc_info=True)
            error_message = f"Sorry, an unexpected error occurred: {e}"
            st.error(error_message)
            full_response = error_message

    # --- Store the RAW full_response in history ---
    # No extraction needed anymore
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    # --- End history update ---

    # st.rerun() # Optional
