# ui_app.py (Place this file in the project root directory)

import streamlit as st
import sys
import os
import logging
from dotenv import load_dotenv
from typing import List, Dict # Import Dict

# --- Add project root to path ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# --- Import necessary components ---
try:
    from orchestrator.main_app import retriever_ready, query_rag_stream
    from orchestrator.prompt_templates import extract_final_answer
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
CHAT_HISTORY_LENGTH = 3 # Number of turns (user + assistant = 1 turn) to send to LLM

# --- Page Configuration ---
st.set_page_config(
    page_title="NutiSmart RAG Assistant",
    page_icon="🍎",
    layout="wide"
)

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        logger.info("Chat history cleared by user.")
        # Add a default message back after clearing
        st.session_state.messages.append({
           "role": "assistant",
           "content": "Chat history cleared. How can I help you?"
        })
        st.rerun()

    st.markdown("---")
    st.subheader("Status")
    if retriever_ready:
        st.success("Retriever Initialized")
    else:
        st.error("Retriever FAILED to Initialize")
    st.markdown("---")
    st.info("Ask questions about nutrition and fitness based on the loaded documents.")


# --- Main Chat Interface ---
st.title("🍎 NutiSmart RAG Assistant")
st.caption("Your AI assistant for Nutrition and Fitness questions.")

# --- Initialization Check ---
if not retriever_ready:
    st.error("FATAL ERROR: Backend retriever not ready. Cannot process queries.")
    st.warning("Check console logs for details.")
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
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Prepare chat history to send to backend ---
    # Get the last N turns (N pairs of user/assistant messages)
    # Each turn has 2 messages, so we take last N*2 messages
    history_to_send: List[Dict[str, str]] = st.session_state.messages[-(CHAT_HISTORY_LENGTH * 2):-1] # Exclude the latest user prompt itself
    logger.info(f"Sending last {len(history_to_send)} messages as history.")

    # Generate and display assistant's response using streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        logger.info("Calling query_rag_stream function with history...")
        try:
            # --- Pass chat_history to the backend function ---
            response_stream = query_rag_stream(prompt, chat_history=history_to_send)

            if response_stream is None:
                 st.error("Could not get response stream from the backend.")
                 full_response = "Error: Failed to get response stream."
            else:
                # Display the stream
                full_response = st.write_stream(response_stream)
                logger.info("Finished streaming response.")

        except Exception as e:
            logger.exception(f"Error during RAG stream processing: {e}", exc_info=True)
            error_message = f"Sorry, an unexpected error occurred: {e}"
            st.error(error_message)
            full_response = error_message

    # Add the *complete* assistant response to chat history AFTER streaming
    final_answer_for_history = extract_final_answer(full_response)
    st.session_state.messages.append({"role": "assistant", "content": final_answer_for_history})

    # Optional: Rerun to ensure the latest message is added before next input
    # st.rerun() # Can sometimes cause minor UI flicker, use if needed

