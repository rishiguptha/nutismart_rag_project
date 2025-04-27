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
    # Import the readiness check and the STREAMING pipeline function
    from orchestrator.main_app import is_retriever_ready, query_rag_stream
    # No extraction function needed for this prompt style
except ImportError as e:
    st.error(f"Error importing orchestrator modules: {e}. Check paths and file structure.")
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
    page_title="NutriSmart Assistant",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Initialization Check ---
retriever_is_ready_flag = is_retriever_ready()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        logger.info("Chat history cleared by user.")
        st.session_state.messages.append({
           "role": "assistant",
           "content": "Chat history cleared. How can I help?"
        })
        st.rerun()
    st.markdown("---")
    st.subheader("Status")
    if retriever_is_ready_flag: st.success("✅ Backend Ready")
    else: st.error("❌ Backend Not Initialized")
    st.markdown("---")
    st.caption("NutiSmart RAG v1.0 (Citations)") # Updated version


# --- Main Chat Interface ---
st.title("🍎 NutriSmart Assistant")
st.caption("Ask about Nutrition & Fitness - Powered by local documents & AI.")

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
        # Display the full content including citations and Sources list
        st.markdown(message["content"])

# --- Handle User Input ---
if user_prompt := st.chat_input("Ask your question here..."):
    logger.info(f"User input: {user_prompt}")

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    history_to_send: List[Dict[str, str]] = st.session_state.messages[-(CHAT_HISTORY_LENGTH * 2):-1]
    logger.info(f"Sending last {len(history_to_send)} messages as history.")

    # Generate and display assistant's response using streaming
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                logger.info("Calling query_rag_stream function...")
                # Call the streaming function
                response_stream = query_rag_stream(user_prompt, chat_history=history_to_send)

                if response_stream is None:
                    st.error("Error: Could not get response stream from the backend.")
                    full_response = "Error: Failed to get response stream."
                else:
                    # Use write_stream - it displays the stream and returns the full string
                    full_response = st.write_stream(response_stream)
                    logger.info("Finished streaming response.")

            except Exception as e:
                logger.exception(f"Error during RAG stream processing: {e}", exc_info=True)
                error_message = f"Sorry, an unexpected error occurred: {e}"
                st.error(error_message)
                full_response = error_message

    # Add the complete assistant response (including citations/Sources) to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.rerun() # Rerun to display the latest message immediately
