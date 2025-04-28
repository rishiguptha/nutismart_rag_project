# ui_app.py

import streamlit as st
import sys
import os
import logging
from dotenv import load_dotenv # Keep load_dotenv for API key loading consistency
from typing import List, Dict, Any # Add Any for session state flexibility
import datetime # Import for potential file logging

# --- Add project root to path ---
# Assumes ui_app.py is in the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Addition ---

# --- Import necessary components from orchestrator ---
# This needs the correct sys.path set above
try:
    from orchestrator.main_app import is_retriever_ready, query_rag_stream
except ImportError as e:
    st.error(f"Error importing orchestrator modules: {e}. Check project structure and sys.path.")
    st.stop() # Stop the app if core components can't be imported

# --- Basic Logging Setup ---
# Configure logging (optional for UI, but helpful for debugging)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

# --- Configuration ---
# Keep loading from .env for consistency, even if config.py isn't used for this step
load_dotenv()
# Use a default if not set, or retrieve from config if you decide to use it later
CHAT_HISTORY_LENGTH = int(os.environ.get("CHAT_HISTORY_LENGTH", 3)) # Example loading

# --- Helper Function for Feedback ---
def log_feedback(message_index: int, feedback_type: str, query: str, response: str):
    """Logs feedback to the console and updates session state."""
    log_message = f"Feedback Received:\n  Message Index: {message_index}\n  Type: {feedback_type}\n  Query: '{query[:100]}...'\n  Response: '{response[:100]}...'"
    logger.info(log_message)
    # Update session state to mark feedback as given for this index
    st.session_state.feedback_given[message_index] = True
    st.toast(f"Feedback ({feedback_type}) recorded!", icon="✅" if feedback_type == "👍 Positive" else "❌")
    # Rerun needed here to make buttons disappear immediately after click
    st.rerun()


# --- Page Configuration (Set early) ---
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
    if st.button("Clear Chat History", key="clear_chat", help="Clears the current conversation history."):
        st.session_state.messages = []
        # Also clear the feedback state when history is cleared
        st.session_state.feedback_given = {}
        logger.info("Chat history and feedback state cleared by user.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Chat history cleared. How can I help?"
        })
        st.rerun() # Rerun is appropriate here to clear the main display

    st.markdown("---")
    st.subheader("Status")
    if retriever_is_ready_flag:
        st.success("✅ Backend Ready")
        st.caption("Retriever and models initialized.")
    else:
        st.error("❌ Backend Not Initialized")
        st.caption("Retriever failed to load. Check logs.")
    st.markdown("---")
    st.caption("NutriSmart RAG v1.0 (Feedback Added)")


# --- Main Chat Interface ---
st.title("🍎 NutriSmart Assistant")
st.caption("Ask about Nutrition & Fitness - Powered by local documents & AI.")

if not retriever_is_ready_flag:
    # Display a more prominent error if the backend isn't ready on startup
    st.error("🔴 **Initialization Error:** The backend components (retriever/models) failed to load. The assistant cannot function. Please check the application logs for details.")
    st.stop() # Stop execution

# --- Initialize Chat History and Feedback State in Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! How can I help you with nutrition and fitness questions today?"
    })
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {} # Stores {message_index: True}

# --- Display Chat Messages ---
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Add feedback buttons ONLY for assistant messages,
        # and only if it's not the initial welcome message (index > 0)
        # and only if feedback hasn't been given for this message yet.
        if message["role"] == "assistant" and index > 0 :
            feedback_key_base = f"feedback_{index}"
            if not st.session_state.feedback_given.get(index, False):
                col1, col2, _ = st.columns([1, 1, 10])
                with col1:
                    if st.button("👍", key=f"{feedback_key_base}_up", help="Mark response as helpful"):
                        user_query = "Unknown"
                        if index > 0 and st.session_state.messages[index-1]["role"] == "user":
                            user_query = st.session_state.messages[index-1]["content"]
                        log_feedback(index, "👍 Positive", user_query, message["content"])
                        # Rerun is now called inside log_feedback

                with col2:
                    if st.button("👎", key=f"{feedback_key_base}_down", help="Mark response as not helpful"):
                        user_query = "Unknown"
                        if index > 0 and st.session_state.messages[index-1]["role"] == "user":
                            user_query = st.session_state.messages[index-1]["content"]
                        log_feedback(index, "👎 Negative", user_query, message["content"])
                        # Rerun is now called inside log_feedback


# --- Handle User Input ---
if user_prompt := st.chat_input("Ask your question here..."):
    logger.info(f"User input received: {user_prompt}")

    # Add user message immediately to history and display it
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    # Rerun to show the user message instantly before processing
    st.rerun()

# --- Process the latest message if it's from the user and hasn't been processed ---
# Check if the last message is from the user and potentially needs a response
# Add a simple flag to prevent reprocessing after an error or completion
needs_processing = False
if st.session_state.messages:
    last_message = st.session_state.messages[-1]
    if last_message["role"] == "user" and not last_message.get("processed", False):
         needs_processing = True

if needs_processing:
    # Mark the user message as being processed to avoid reprocessing on rerun
    st.session_state.messages[-1]["processed"] = True
    user_message_content = st.session_state.messages[-1]["content"] # Get the actual user prompt

    history_to_send: List[Dict[str, str]] = []
    # History should include messages *before* the current user prompt
    start_index = max(0, len(st.session_state.messages) - (CHAT_HISTORY_LENGTH * 2) - 2) # Go back further
    for msg in st.session_state.messages[start_index:-1]: # Exclude the last (current user) message
        history_to_send.append({"role": msg["role"], "content": msg["content"]})

    logger.info(f"Sending last {len(history_to_send)} messages as history context for query: '{user_message_content[:50]}...'")

    # Generate and display assistant's response using streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        error_occurred = False
        try:
            # Show spinner while processing
            with st.spinner("Thinking..."):
                logger.info("Calling backend: query_rag_stream...")
                response_stream = query_rag_stream(user_message_content, chat_history=history_to_send)

                if response_stream is None:
                    # This case might be less likely if query_rag_stream always yields something
                    logger.error("Backend function query_rag_stream returned None.")
                    full_response = "Sorry, I encountered a problem connecting to the backend. Please try again later."
                    message_placeholder.error(full_response) # Use st.error for UI
                    error_occurred = True
                else:
                    for chunk in response_stream:
                        # Check for specific system error messages from the stream
                        if chunk.startswith("[SYSTEM:"):
                            logger.warning(f"Received system message from stream: {chunk}")
                            if "[SYSTEM: Retriever not ready]" in chunk:
                                full_response = "Sorry, the document retrieval system is not available right now. Please try again later."
                            elif "[SYSTEM: Error communicating with Language Model]" in chunk or "[SYSTEM: Error during LLM communication:" in chunk :
                                full_response = "Sorry, I had trouble communicating with the language model. Please try again."
                            elif "[SYSTEM: Response blocked due to:" in chunk or "[SYSTEM: Response may be blocked due to:" in chunk:
                                full_response = "Sorry, the response could not be generated due to content restrictions."
                            else: # Generic fallback for other system messages
                                full_response = "Sorry, an unexpected issue occurred while generating the response. Please try again."
                            message_placeholder.error(full_response) # Show error in UI
                            error_occurred = True
                            break # Stop processing the stream on error
                        else:
                            # Append normal chunks
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌") # Append cursor

                    if not error_occurred:
                        message_placeholder.markdown(full_response) # Show final response
                        logger.info("Finished streaming response to UI.")

        except Exception as e:
            # Catch unexpected errors during the streaming process itself
            logger.exception(f"Error during RAG stream processing in UI: {e}", exc_info=True)
            full_response = f"Sorry, an unexpected application error occurred. Please report this issue."
            message_placeholder.error(full_response) # Show error in UI
            error_occurred = True

    # Add the complete assistant response (or error message) to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Rerun AFTER adding the assistant message to history to display it correctly
    st.rerun()

# --- End of Streamlit App ---
