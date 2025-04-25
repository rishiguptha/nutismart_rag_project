# orchestrator/main_app.py

# Keep imports for logging, sys, os, dotenv, FAISS, SentenceTransformerEmbeddings, Generator, List, Dict, Document
import logging
import sys
import os
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Generator, List, Dict, Tuple
from langchain_core.documents import Document

# Import functions from other orchestrator files
# Import BOTH streaming and non-streaming LLM functions now
from .llm_client import stream_llm_response, get_llm_response
from .prompt_templates import format_react_style_rag_prompt, extract_final_answer

# (Keep Load Environment Variables, Configuration, Logging Setup, Retriever Initialization, is_retriever_ready, retrieve_context_local, query_rag_stream functions as they are)
# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store/faiss_index')
FAISS_INDEX_NAME = "nutrition_fitness_index"
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_TOP_K = 5

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# --- Initialize Retriever Components Directly (FAISS Version) ---
vector_db = None
embeddings = None
retriever_ready = False
try:
    logger.info("Initializing retriever components locally (FAISS)...")
    abs_index_path = os.path.abspath(FAISS_INDEX_PATH)
    faiss_file = os.path.join(abs_index_path, f"{FAISS_INDEX_NAME}.faiss")
    pkl_file = os.path.join(abs_index_path, f"{FAISS_INDEX_NAME}.pkl")
    logger.info(f"Attempting to load FAISS index from: {abs_index_path}")
    if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
        logger.error(f"FAISS index files not found.")
        raise FileNotFoundError(f"FAISS index files not found")
    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_db = FAISS.load_local(
        folder_path=abs_index_path,
        embeddings=embeddings,
        index_name=FAISS_INDEX_NAME,
        allow_dangerous_deserialization=True
    )
    logger.info(f"FAISS retriever components initialized. Index count: {vector_db.index.ntotal}")
    retriever_ready = True
except FileNotFoundError as fnf_error:
     logger.error(f"Initialization failed: {fnf_error}")
except Exception as e:
    logger.exception(f"Failed to initialize FAISS retriever components: {e}", exc_info=True)


def is_retriever_ready() -> bool:
    """Checks if the retriever components were initialized successfully."""
    return retriever_ready

# --- Retrieval Function (Local FAISS Version - Unchanged) ---
def retrieve_context_local(query: str, k: int = DEFAULT_TOP_K) -> List[Document]:
    """Embeds query and retrieves top_k relevant Document objects locally using FAISS."""
    if not retriever_ready or vector_db is None:
         logger.error("Local FAISS retriever is not ready.")
         return []
    if not query or not query.strip():
        logger.warning("Received empty query for retrieval.")
        return []
    try:
        logger.info(f"Querying local FAISS index for top {k} results: '{query[:50]}...'")
        results: List[Document] = vector_db.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(results)} Document objects locally.")
        return results
    except Exception as e:
        logger.exception(f"Error during local FAISS query: {e}", exc_info=True)
        return []

# --- Main RAG Query Function for Streaming (Unchanged) ---
def query_rag_stream(
    user_query: str,
    chat_history: List[Dict[str, str]] = None,
    top_k: int = DEFAULT_TOP_K
    ) -> Generator[str, None, None]:
    """
    Performs the RAG pipeline using local retrieval, considering chat history,
    and streams the response from the external LLM. Yields text chunks.
    """
    logger.info(f"Retrieving context locally for query: '{user_query[:50]}...'")
    context_docs: List[Document] = retrieve_context_local(user_query, k=top_k)
    logger.info(f"Retrieved {len(context_docs)} context documents locally.")

    prompt = format_react_style_rag_prompt(user_query, context_docs, chat_history)

    logger.info("Streaming prompt to external LLM...")
    response_generator = stream_llm_response(prompt)

    if response_generator is None:
        yield "Sorry, the Language Model client is not available."
        return

    yield from response_generator


# --- ADD THIS NEW FUNCTION FOR EVALUATION ---
def run_rag_for_evaluation(
    user_query: str,
    top_k: int = DEFAULT_TOP_K
    ) -> Dict:
    """
    Runs the RAG pipeline for a single query and returns results for evaluation.
    Uses non-streaming LLM call.
    """
    if not is_retriever_ready():
        logger.error("[Evaluation] Retriever not ready. Cannot run evaluation.")
        # Return a dictionary indicating failure
        return {
            "query": user_query,
            "retrieved_context": [],
            "generated_answer": "ERROR: Retriever not initialized.",
            "error": "Retriever not initialized."
        }

    logger.info(f"[Evaluation] Processing query: '{user_query[:50]}...'")
    final_answer = "ERROR: Processing failed." # Default error answer
    context_texts = []
    error_msg = None

    try:
        # 1. Retrieve Context
        context_docs: List[Document] = retrieve_context_local(user_query, k=top_k)
        # Extract text and metadata for evaluation output
        context_texts = [doc.page_content for doc in context_docs]
        context_metadata = [doc.metadata for doc in context_docs] # Keep metadata
        logger.info(f"[Evaluation] Retrieved {len(context_docs)} context documents.")

        # 2. Format Prompt (No history for isolated eval questions)
        prompt = format_react_style_rag_prompt(user_query, context_docs, chat_history=None)

        # 3. Call LLM (Non-Streaming)
        logger.info("[Evaluation] Sending prompt to external LLM (non-streaming)...")
        raw_llm_response = get_llm_response(prompt) # Use non-streaming version

        if raw_llm_response is None:
            logger.error("[Evaluation] Failed to get response from LLM.")
            final_answer = "ERROR: LLM call failed."
            error_msg = "LLM call failed."
        elif "[SYSTEM:" in raw_llm_response: # Check for system errors returned by LLM client
             logger.warning(f"[Evaluation] LLM response indicates an issue: {raw_llm_response}")
             final_answer = raw_llm_response # Return the system message
             error_msg = final_answer
        else:
            # 4. Extract Final Answer
            final_answer = extract_final_answer(raw_llm_response)
            logger.info(f"[Evaluation] Generated answer length: {len(final_answer)}")

    except Exception as e:
        logger.exception(f"[Evaluation] Unexpected error during evaluation run for query '{user_query[:50]}...': {e}", exc_info=True)
        final_answer = f"ERROR: Unexpected error during processing - {e}"
        error_msg = str(e)

    # 5. Return results dictionary
    return {
        "query": user_query,
        "retrieved_context": context_texts,
        "retrieved_metadata": context_metadata, # Include metadata
        "generated_answer": final_answer,
        "error": error_msg # Include error message if any occurred
    }
# --- END OF NEW FUNCTION ---

# --- Main Execution Block (Command-line version - Needs update to handle history if used) ---
# This block is mainly for testing the backend logic, not the conversational flow.
# The Streamlit UI is the primary way to test the conversation.
if __name__ == "__main__":
    if not retriever_ready:
        print("\nFATAL ERROR: Failed to initialize local FAISS retrieval components.")
        sys.exit(1)

    print("\nNutrition & Fitness RAG Assistant (FAISS Version - Streaming Test)")
    print("Enter your query below. Type 'quit' to exit.")
    print("-" * 30)

    # Simple history for command-line testing (won't persist between runs)
    cli_history = []
    HISTORY_LENGTH = 3 # How many turns (user + assistant = 1 turn) to keep for CLI

    while True:
        try:
            query = input("Query: ")
            if query.lower() == 'quit': break
            if not query.strip(): continue

            print("\nStreaming Answer:")
            full_response = ""
            # Pass the recent history to the stream function
            recent_history = cli_history[-(HISTORY_LENGTH * 2):] # Get last N pairs
            response_gen = query_rag_stream(query, chat_history=recent_history)

            for chunk in response_gen:
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n" + "-" * 30 + "\n")

            # Update CLI history (simple version)
            # Extract final answer for cleaner history storage
            final_answer_hist = extract_final_answer(full_response)
            cli_history.append({"role": "user", "content": query})
            cli_history.append({"role": "assistant", "content": final_answer_hist})


        except EOFError: print("\nExiting."); break
        except KeyboardInterrupt: print("\nExiting."); break
