# ui_app.py (Place this file in the project root directory)

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
    # If using ReAct style prompts that need extraction:
    # from orchestrator.prompt_templates import extract_final_answer
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
CHAT_HISTORY_LENGTH = 3 # How many turns (user + assistant) to send to LLM

# --- Page Configuration ---
st.set_page_config(
    page_title="NutiSmart Assistant", # Slightly shorter title
    page_icon="🍎",
    layout="centered", # Use centered layout for minimalism
    initial_sidebar_state="collapsed" # Start with sidebar collapsed
)

# --- Initialization Check ---
# Perform this check early
retriever_is_ready_flag = is_retriever_ready()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        logger.info("Chat history cleared by user.")
        # Add a gentle confirmation message back
        st.session_state.messages.append({
           "role": "assistant",
           "content": "Chat history cleared. Ready for new questions!"
        })
        st.rerun() # Rerun to update the main chat display

    st.markdown("---") # Divider
    st.subheader("Status")
    if retriever_is_ready_flag:
        st.success("✅ Backend Ready")
    else:
        st.error("❌ Backend Not Initialized")
        st.warning("Please check console logs and ensure indexing script ran.")
    st.markdown("---")
    st.caption("NutiSmart RAG v1.0") # Simple version indicator


# --- Main Chat Interface ---
st.title("🍎 NutiSmart Assistant")
st.caption("Ask about Nutrition & Fitness - Powered by local documents & AI.")

# --- Initialization Check Display ---
# Show error in main area if backend isn't ready
if not retriever_is_ready_flag:
    st.error("FATAL ERROR: Backend retriever not ready. Cannot process queries.")
    st.warning("Check console logs for details (e.g., FAISS index path issues).")
    st.stop() # Halt the app

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! How can I help you with nutrition and fitness questions today?"
    })

# --- Display Chat Messages ---
# Use st.container() for better control if needed, but direct loop is fine
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) # Render content as Markdown

# --- Handle User Input ---
if user_prompt := st.chat_input("Ask your question here..."):
    logger.info(f"User input: {user_prompt}")

    # Append and display user message immediately
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Prepare history for the backend call
    history_to_send: List[Dict[str, str]] = st.session_state.messages[-(CHAT_HISTORY_LENGTH * 2):-1]
    logger.info(f"Sending last {len(history_to_send)} messages as history.")

    # Generate and display assistant's response using streaming
    with st.chat_message("assistant"):
        # Use a placeholder with a spinner for better "thinking" indication
        with st.spinner("Thinking..."):
            try:
                logger.info("Calling query_rag_stream function...")
                response_stream = query_rag_stream(user_prompt, chat_history=history_to_send)

                if response_stream is None:
                    st.error("Error: Could not get response stream from the backend.")
                    full_response = "Error: Failed to get response stream."
                else:
                    # Use write_stream to display the generator output chunk by chunk
                    full_response = st.write_stream(response_stream)
                    logger.info("Finished streaming response.")

            except Exception as e:
                logger.exception(f"Error during RAG stream processing: {e}", exc_info=True)
                error_message = f"Sorry, an unexpected error occurred: {e}"
                st.error(error_message) # Display error clearly in chat
                full_response = error_message

    # Add the complete assistant response (or error) to history
    # Assumes the simplified prompt is used and no extraction needed
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Optional: Automatically rerun might feel slightly smoother after response
    # st.rerun()

# --- End of Streamlit App ---