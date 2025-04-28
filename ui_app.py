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

# --- Helper Function to Set Input Text ---
# We'll use session state to manage the chat input value
def set_chat_input(text):
    st.session_state.chat_input_value = text


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
        st.session_state.feedback_given = {}
        st.session_state.chat_input_value = "" # Clear input field on history clear
        logger.info("Chat history and feedback state cleared by user.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Chat history cleared. How can I help?"
        })
        st.rerun()

    st.markdown("---")
    st.subheader("Status")
    if retriever_is_ready_flag:
        st.success("✅ Backend Ready")
        st.caption("Retriever and models initialized.")
    else:
        st.error("❌ Backend Not Initialized")
        st.caption("Retriever failed to load. Check logs.")
    st.markdown("---")
    st.caption("NutriSmart RAG v1.0 (Examples Added)") # Update version/note


# --- Main Chat Interface ---
st.title("🍎 NutriSmart Assistant")
st.caption("Ask about Nutrition & Fitness - Powered by local documents & AI.")

# --- Example Questions ---
st.markdown("##### Try asking:")
examples = [
    "What are the benefits of protein?",
    "How is body composition measured?",
    "Explain SMART goals for fitness.",
    "Examples of moderate-intensity activities?",
]
# Use columns for a cleaner layout of example buttons
cols = st.columns(len(examples))
for i, example in enumerate(examples):
    with cols[i]:
        # When button is clicked, call set_chat_input to update the state variable
        st.button(example, key=f"ex{i}", on_click=set_chat_input, args=(example,))


# --- Initialization Check After UI Elements ---
if not retriever_is_ready_flag:
    st.error("🔴 **Initialization Error:** The backend components (retriever/models) failed to load. The assistant cannot function. Please check the application logs for details.")
    st.stop()

# --- Initialize Session State Variables ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! How can I help you with nutrition and fitness questions today?"
    })
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}
# Initialize the chat input value in session state if it doesn't exist
if "chat_input_value" not in st.session_state:
    st.session_state.chat_input_value = ""

# --- Display Chat Messages ---
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Feedback buttons logic (as before)
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
                with col2:
                    if st.button("👎", key=f"{feedback_key_base}_down", help="Mark response as not helpful"):
                        user_query = "Unknown"
                        if index > 0 and st.session_state.messages[index-1]["role"] == "user":
                            user_query = st.session_state.messages[index-1]["content"]
                        log_feedback(index, "👎 Negative", user_query, message["content"])


# --- Handle User Input ---
# Use the value from session state, controlled by the input widget AND the example buttons
user_prompt = st.chat_input(
    "Ask your question here...",
    key="chat_input_widget", # Give the widget a key
    # on_change callback can be used, but managing state directly is often simpler
)

# Check if the input came from the text input OR if an example button set the state
prompt_to_process = None
if user_prompt: # Input came directly from chat_input widget
    prompt_to_process = user_prompt
    st.session_state.chat_input_value = "" # Clear state after processing direct input
elif st.session_state.chat_input_value: # Input came from an example button setting the state
    prompt_to_process = st.session_state.chat_input_value
    st.session_state.chat_input_value = "" # Clear state after processing button input

if prompt_to_process:
    logger.info(f"User input received: {prompt_to_process}")

    # Add user message immediately to history
    st.session_state.messages.append({"role": "user", "content": prompt_to_process})
    # Mark it as needing processing (add a flag or check the last message role)
    st.session_state.messages[-1]["needs_processing"] = True
    # Rerun to display the user message and trigger processing logic below
    st.rerun()

# --- Process the latest message if it's from the user and needs processing ---
needs_processing_flag = False
if st.session_state.messages:
    last_message = st.session_state.messages[-1]
    if last_message["role"] == "user" and last_message.get("needs_processing", False):
         needs_processing_flag = True

if needs_processing_flag:
    # Mark the user message as processed
    st.session_state.messages[-1]["needs_processing"] = False
    user_message_content = st.session_state.messages[-1]["content"]

    history_to_send: List[Dict[str, str]] = []
    start_index = max(0, len(st.session_state.messages) - (CHAT_HISTORY_LENGTH * 2) - 2)
    for msg in st.session_state.messages[start_index:-1]:
        history_to_send.append({"role": msg["role"], "content": msg["content"]})

    logger.info(f"Sending last {len(history_to_send)} messages as history context for query: '{user_message_content[:50]}...'")

    # Generate and display assistant's response using streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        error_occurred = False
        try:
            with st.spinner("Thinking..."):
                logger.info("Calling backend: query_rag_stream...")
                response_stream = query_rag_stream(user_message_content, chat_history=history_to_send)

                if response_stream is None:
                    logger.error("Backend function query_rag_stream returned None.")
                    full_response = "Sorry, I encountered a problem connecting to the backend. Please try again later."
                    message_placeholder.error(full_response)
                    error_occurred = True
                else:
                    for chunk in response_stream:
                        if chunk.startswith("[SYSTEM:"):
                            logger.warning(f"Received system message from stream: {chunk}")
                            # Simplified error mapping
                            if "Retriever not ready" in chunk:
                                full_response = "Sorry, the document retrieval system is not available right now."
                            elif "Error communicating" in chunk or "Error during LLM communication" in chunk:
                                full_response = "Sorry, I had trouble communicating with the language model."
                            elif "Response blocked" in chunk or "Response may be blocked" in chunk:
                                full_response = "Sorry, the response could not be generated due to content restrictions."
                            else:
                                full_response = "Sorry, an unexpected issue occurred while generating the response."
                            message_placeholder.error(full_response)
                            error_occurred = True
                            break
                        else:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")

                    if not error_occurred:
                        message_placeholder.markdown(full_response)
                        logger.info("Finished streaming response to UI.")

        except Exception as e:
            logger.exception(f"Error during RAG stream processing in UI: {e}", exc_info=True)
            full_response = f"Sorry, an unexpected application error occurred. Please report this issue."
            message_placeholder.error(full_response)
            error_occurred = True

    # Add the complete assistant response (or error message) to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Rerun AFTER adding the assistant message to history
    # This ensures the message loop displays the latest assistant message
    # and its associated feedback buttons correctly.
    st.rerun()

# --- End of Streamlit App ---
