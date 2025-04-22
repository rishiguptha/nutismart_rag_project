# ```python
# ui_app.py (Place this file in the project root directory)

import streamlit as st
import sys
import os
import logging
from dotenv import load_dotenv
import time # For simulating streaming if needed for testing

# --- Add project root to path ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# --- Import necessary components ---
try:
    from orchestrator.main_app import retriever_ready
    # Import the STREAMING RAG function
    from orchestrator.main_app import query_rag_stream
    # We might need the final answer extractor if the stream includes "Thought:"
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

# --- Page Configuration ---
st.set_page_config(
    page_title="NutiSmart RAG Assistant", # Renamed
    page_icon="🍎",
    layout="wide" # Use wide layout for more space
)

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")
    # Add a button to clear the chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = [] # Reset the messages list
        logger.info("Chat history cleared by user.")
        st.rerun() # Rerun the app to reflect the cleared state

    st.markdown("---")
    st.subheader("Status")
    if retriever_ready:
        st.success("Retriever Initialized")
        # Optionally display index info if available and needed
        # try:
        #     from orchestrator.main_app import vector_db
        #     st.write(f"Index Vectors: {vector_db.index.ntotal}")
        # except: pass # Ignore if vector_db not imported/available
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
    # Add initial welcome message
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

    # Generate and display assistant's response using streaming
    with st.chat_message("assistant"):
        try:
            logger.info("Calling query_rag_stream function...")
            # Use st.write_stream which consumes the generator from query_rag_stream
            response_stream = query_rag_stream(prompt)

            # Check if the stream function returned None (e.g., LLM client failed init)
            if response_stream is None:
                 st.error("Could not get response stream from the backend.")
                 full_response = "Error: Failed to get response stream."
            else:
                # Display the stream - st.write_stream handles iterating and displaying
                # It also returns the full concatenated response once done.
                full_response = st.write_stream(response_stream)
                logger.info("Finished streaming response.")

        except Exception as e:
            logger.exception(f"Error during RAG stream processing: {e}", exc_info=True)
            error_message = f"Sorry, an unexpected error occurred: {e}"
            st.error(error_message)
            full_response = error_message # Store error message

    # Add the *complete* assistant response to chat history AFTER streaming
    # Note: The ReAct prompt might still include "Thought:" and "Final Answer:"
    # We might want to extract the final answer *after* streaming for history.
    final_answer_for_history = extract_final_answer(full_response) # Extract for cleaner history
    st.session_state.messages.append({"role": "assistant", "content": final_answer_for_history})

# ```
#