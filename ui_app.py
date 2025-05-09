# ui_app.py

import streamlit as st
import sys
import os
import logging
import re # Import regex for splitting
from dotenv import load_dotenv # Keep load_dotenv for API key loading consistency
from typing import List, Dict, Any # Add Any for session state flexibility
import datetime # Import for potential file logging
import json
import streamlit.components.v1 as components
import uuid

def setup_logging():
    """Setup logging with timestamped log files in the logs directory."""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'nutrismart_streamlit_{timestamp}.log'
    log_path = os.path.join(logs_dir, log_filename)
    
    # Setup logging
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Console handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(log_formatter)
        logger.addHandler(ch)
    
    # File handler
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(log_formatter)
        logger.addHandler(fh)
    
    return logger

# Initialize logger
logger = setup_logging()

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
    logger.error(f"Error importing orchestrator modules: {e}. Check project structure and sys.path.")
    st.error(f"Error importing orchestrator modules: {e}. Check project structure and sys.path.")
    st.stop() # Stop the app if core components can't be imported

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

# Add a session state variable for expand/collapse all sources
if "expand_all_sources" not in st.session_state:
    st.session_state.expand_all_sources = False

def render_sources(sources_part, source_chunks=None, msg_index=0):
    st.markdown("**Sources:**")
    # Show/Hide Details button above the list
    if st.button(
        "Hide Details" if st.session_state.expand_all_sources else "Show Details",
        key=f"toggle_sources_expand_{msg_index}"
    ):
        st.session_state.expand_all_sources = not st.session_state.expand_all_sources
    for index, line in enumerate(sources_part.splitlines()):
        line = line.strip()
        if not line or line.lower().startswith("**sources:**"):
            continue
        if line.startswith("["):
            parts = line.split("]", 1)
            if len(parts) == 2:
                num = parts[0][1:]
                rest = parts[1].strip()
                chunk_content = None
                title = None
                snippet = None
                if source_chunks:
                    for chunk in source_chunks:
                        if str(chunk.get("number")) == num:
                            chunk_content = chunk.get("content")
                            title = chunk.get("title")
                            snippet = chunk.get("snippet")
                            break
                # Icon logic
                if rest.startswith("http://") or rest.startswith("https://"):
                    icon = "🔗"
                else:
                    icon = "📄"
                st.markdown(f"{icon} **[{num}]** {rest}", unsafe_allow_html=True)
                if title:
                    st.markdown(f"**Title:** {title}", unsafe_allow_html=True)
                if snippet:
                    st.markdown(f"<span style='color:gray;font-size:0.95em;'><i>{snippet}</i></span>", unsafe_allow_html=True)
                if st.session_state.expand_all_sources and chunk_content:
                    st.info(chunk_content)
        else:
            st.markdown(line)

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
    # Confirmation dialog for clearing chat history
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False
    if st.session_state.confirm_clear:
        st.warning("Are you sure you want to clear the chat history?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, clear", key="confirm_clear_yes"):
                st.session_state.messages = []
                st.session_state.feedback_given = {}
                st.session_state.chat_input_value = ""
                st.session_state.confirm_clear = False
                logger.info("Chat history and feedback state cleared by user.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Chat history cleared. How can I help?"
                })
                st.rerun()
        with col2:
            if st.button("Cancel", key="confirm_clear_no"):
                st.session_state.confirm_clear = False
    else:
        if st.button("Clear Chat History", key="clear_chat", help="Clears the current conversation history."):
            st.session_state.confirm_clear = True

    st.markdown("---")
    st.subheader("Status")
    if retriever_is_ready_flag:
        st.success("✅ Backend Ready")
        st.caption("Retriever and models initialized.")
    else:
        st.error("❌ Backend Not Initialized")
        st.caption("Retriever failed to load. Check logs.")
    st.markdown("---")
    st.caption("NutriSmart RAG v1.0 (UI Tweaks)") # Update version/note

    import json
    chat_json = json.dumps(st.session_state.get("messages", []), indent=2)
    st.download_button("Download Chat History", chat_json, file_name="chat_history.json", mime="application/json")

# --- Main Chat Interface ---
st.title("🍎 NutriSmart Assistant")
st.caption("Ask about Nutrition & Fitness - Powered by local documents & AI.")

# --- Example Questions ---
st.markdown("---") # Add a divider
st.markdown("##### Try asking:")
examples = [
    "What are the benefits of protein?",
    "How is body composition measured?",
    "Explain SMART goals for fitness.",
    "Examples of moderate-intensity activities?",
]
cols = st.columns(len(examples))
for i, example in enumerate(examples):
    with cols[i]:
        st.button(example, key=f"ex{i}", on_click=set_chat_input, args=(example,))
st.markdown("---") # Add a divider


# --- Initialization Check After UI Elements ---
if not retriever_is_ready_flag:
    logger.error("🔴 Initialization Error: The backend components (retriever/models) failed to load. The assistant cannot function. Please check the application logs for details.")
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
if "chat_input_value" not in st.session_state:
    st.session_state.chat_input_value = ""

# Add a session state flag for processing
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# --- Helper function to highlight citations ---
def highlight_citations(text):
    # Highlight [number] citations in blue and bold
    return re.sub(r'(\[\d+\])', r'<span style="color:#1976D2;font-weight:bold;">\1</span>', text)
# --- Display Chat Messages ---
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        message_content = message["content"]
        answer_part = message_content
        sources_part = None
        sources_match = re.search(r"(\n\n|^)\*\*Sources:\*\*\n", message_content, re.IGNORECASE)
        if sources_match:
            split_index = sources_match.start()
            answer_part = message_content[:split_index].strip()
            sources_part = message_content[split_index:].strip()
        ts = message.get("timestamp")
        if ts:
            st.markdown(f"<span style='color:gray;font-size:0.8em;'>{ts}</span>", unsafe_allow_html=True)
        if message["role"] == "assistant":
            st.markdown(highlight_citations(answer_part), unsafe_allow_html=True)
            # --- Show response time if available ---
            if message.get("response_time"):
                st.markdown(f"<span style='color:gray;font-size:0.8em;'>⏱️ {message['response_time']} seconds</span>", unsafe_allow_html=True)
            if index == len(st.session_state.messages) - 1:
                if st.button("🔄 Regenerate", key="regenerate_btn"):
                    if index > 0 and st.session_state.messages[index-1]["role"] == "user":
                        st.session_state.messages.pop()
                        st.session_state.messages[-1]["needs_processing"] = True
                        st.session_state.is_processing = True
                        st.rerun()
        else:
            st.markdown(answer_part)
        if sources_part:
            with st.expander("View Sources"):
                st.session_state.current_message_index = index
                render_sources(sources_part, message.get("source_chunks"), msg_index=index)
        # Feedback buttons logic - place AFTER the main answer content
        if message["role"] == "assistant" and index > 0 :
            feedback_key_base = f"feedback_{index}"
            if not st.session_state.feedback_given.get(index, False):
                col1, col2, _ = st.columns([1, 1, 10]) # Use smaller columns for buttons
                with col1:
                    if st.button("👍", key=f"{feedback_key_base}_up", help="Mark response as helpful"):
                        user_query = "Unknown"
                        if index > 0 and st.session_state.messages[index-1]["role"] == "user":
                            user_query = st.session_state.messages[index-1]["content"]
                        log_feedback(index, "👍 Positive", user_query, message_content)
                with col2:
                    if st.button("👎", key=f"{feedback_key_base}_down", help="Mark response as not helpful"):
                        user_query = "Unknown"
                        if index > 0 and st.session_state.messages[index-1]["role"] == "user":
                            user_query = st.session_state.messages[index-1]["content"]
                        log_feedback(index, "👎 Negative", user_query, message_content)


# --- Handle User Input ---
# Use the value from session state, controlled by the input widget AND the example buttons
user_prompt = st.chat_input(
    "Ask your question here...",
    key="chat_input_widget",
    disabled=st.session_state.get("is_processing", False)
)

prompt_to_process = None
if user_prompt:
    prompt_to_process = user_prompt
    st.session_state.chat_input_value = ""
    logger.info(f"User input received: {prompt_to_process}")
elif st.session_state.chat_input_value:
    prompt_to_process = st.session_state.chat_input_value
    st.session_state.chat_input_value = ""
    logger.info(f"User input received (from example button): {prompt_to_process}")

if prompt_to_process:
    st.session_state.is_processing = True
    st.session_state.messages.append({
        "role": "user",
        "content": prompt_to_process,
        "needs_processing": True,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    st.rerun()

# --- Process the latest message if it's from the user and needs processing ---
needs_processing_flag = False
if st.session_state.messages:
    last_message = st.session_state.messages[-1]
    if last_message["role"] == "user" and last_message.get("needs_processing", False):
        needs_processing_flag = True

if needs_processing_flag:
    st.session_state.messages[-1]["needs_processing"] = False
    user_message_content = st.session_state.messages[-1]["content"]

    history_to_send: List[Dict[str, str]] = []
    start_index = max(0, len(st.session_state.messages) - (CHAT_HISTORY_LENGTH * 2) - 2)
    for msg in st.session_state.messages[start_index:-1]:
        history_to_send.append({"role": msg["role"], "content": msg["content"]})

    logger.info(f"Sending last {len(history_to_send)} messages as history context for query: '{user_message_content[:50]}...'")

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        error_occurred = False
        source_chunks = None
        import time
        start_time = time.time()
        try:
            with st.spinner("🔄 Retrieving context and generating answer..."):
                progress_bar = st.progress(0, text="Generating answer...")
                logger.info("Calling backend: query_rag_stream...")
                response_stream = query_rag_stream(user_message_content, chat_history=history_to_send)

                if response_stream is None:
                    logger.error("Backend function query_rag_stream returned None.")
                    full_response = "Sorry, I encountered a problem connecting to the backend. Please try again later."
                    message_placeholder.error(full_response)
                    error_occurred = True
                else:
                    progress_val = 0
                    for chunk in response_stream:
                        if chunk.startswith("[SOURCE_CHUNKS]") and chunk.endswith("[/SOURCE_CHUNKS]"):
                            try:
                                source_chunks = json.loads(chunk[len("[SOURCE_CHUNKS]"):-len("[/SOURCE_CHUNKS]")])
                            except Exception as e:
                                logger.error(f"Failed to parse source_chunks: {e}")
                                source_chunks = None
                            continue
                        if chunk.startswith("[SYSTEM:"):
                            logger.warning(f"Received system message from stream: {chunk}")
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
                            progress_val = min(progress_val + 5, 100)
                            progress_bar.progress(progress_val, text="Generating answer...")
                            message_placeholder.markdown(full_response + "▌")

                    if not error_occurred:
                        message_placeholder.markdown(full_response)
                        logger.info("Finished streaming response to UI.")
        except Exception as e:
            logger.exception(f"Error during RAG stream processing in UI: {e}", exc_info=True)
            full_response = f"Sorry, an unexpected application error occurred. Please report this issue."
            message_placeholder.error(full_response)
            error_occurred = True
        end_time = time.time()
        response_time = round(end_time - start_time, 2)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "source_chunks": source_chunks,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "response_time": response_time
    })
    st.session_state.is_processing = False
    st.rerun()

# --- End of Streamlit App ---