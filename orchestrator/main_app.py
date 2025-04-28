# orchestrator/main_app.py

import logging
import sys
import os
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Generator, List, Dict, Tuple, Union # Keep Union if needed elsewhere
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import pprint

# Import functions from other orchestrator files
# Use the reverted llm_client functions
from .llm_client import stream_llm_response, get_llm_response, transform_query_with_history
# Use the reverted prompt formatter
from .prompt_templates import format_rag_prompt # This should be the stricter grounding one now

# (Keep Load Environment Variables, Configuration, Logging Setup, Retriever Initialization, is_retriever_ready, retrieve_context_local functions as they are)
# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store/faiss_index')
FAISS_INDEX_NAME = "nutrition_fitness_index"
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL_NAME = os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
INITIAL_RETRIEVAL_K = 20
FINAL_TOP_K = 10

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# --- Initialize Retriever Components ---
vector_db = None
embeddings = None
cross_encoder = None
retriever_ready = False
# (Keep the try-except block for initialization as is)
try:
    logger.info("Initializing retriever components locally (FAISS & CrossEncoder)...")
    abs_index_path = os.path.abspath(FAISS_INDEX_PATH)
    faiss_file = os.path.join(abs_index_path, f"{FAISS_INDEX_NAME}.faiss")
    pkl_file = os.path.join(abs_index_path, f"{FAISS_INDEX_NAME}.pkl")
    logger.info(f"Attempting to load FAISS index from: {abs_index_path}")
    if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
        logger.error(f"FAISS index files not found.")
        raise FileNotFoundError(f"FAISS index files not found")
    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True}
    )
    vector_db = FAISS.load_local(
        folder_path=abs_index_path, embeddings=embeddings, index_name=FAISS_INDEX_NAME, allow_dangerous_deserialization=True
    )
    logger.info(f"FAISS index loaded. Index count: {vector_db.index.ntotal}")
    logger.info(f"Initializing CrossEncoder model: {CROSS_ENCODER_MODEL_NAME}...")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME, max_length=512, device='cpu')
    logger.info("CrossEncoder model initialized.")
    retriever_ready = True
except FileNotFoundError as fnf_error:
     logger.error(f"Initialization failed: {fnf_error}")
except Exception as e:
    logger.exception(f"Failed to initialize retriever components: {e}", exc_info=True)


# --- Readiness Check Function ---
def is_retriever_ready() -> bool:
    return retriever_ready and vector_db is not None and cross_encoder is not None and embeddings is not None

# --- Retrieval Function with Re-ranking (Unchanged) ---
def retrieve_context_local(query: str, initial_k: int = INITIAL_RETRIEVAL_K, final_k: int = FINAL_TOP_K) -> List[Document]:
    # (Keep the existing retrieve_context_local function as is)
    if not is_retriever_ready():
         logger.error("Local retriever components are not ready.")
         return []
    if not query or not query.strip():
        logger.warning("Received empty query for retrieval.")
        return []
    try:
        logger.info(f"Querying FAISS for initial top {initial_k} results: '{query[:50]}...'")
        initial_results: List[Document] = vector_db.similarity_search(query, k=initial_k)
        logger.info(f"Retrieved {len(initial_results)} initial Document objects.")
        if not initial_results: return []
        logger.info(f"Re-ranking {len(initial_results)} documents...")
        cross_inp = [[query, doc.page_content] for doc in initial_results]
        cross_scores = cross_encoder.predict(cross_inp, show_progress_bar=False)
        logger.info(f"CrossEncoder scores calculated.")
        doc_scores = list(zip(initial_results, cross_scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in doc_scores[:final_k]]
        logger.info(f"Returning top {len(reranked_docs)} re-ranked documents.")
        return reranked_docs
    except Exception as e:
        logger.exception(f"Error during retrieval or re-ranking: {e}", exc_info=True)
        return []

# --- Main RAG Query Function for Streaming (Reverted) ---
def query_rag_stream(
    user_query: str,
    chat_history: List[Dict[str, str]] = None
    ) -> Generator[str, None, None]:
    """
    Performs the RAG pipeline: Query Transformation -> Retrieval -> Re-ranking -> LLM Stream.
    """
    if not is_retriever_ready():
        yield "[SYSTEM: Retriever not ready]"
        return

    # 1. Transform Query
    if chat_history:
        retrieval_query = transform_query_with_history(user_query, chat_history)
    else:
        retrieval_query = user_query
    logger.info(f"Using query for retrieval: '{retrieval_query[:50]}...'")

    # 2. Retrieve and Re-rank Context
    context_docs: List[Document] = retrieve_context_local(retrieval_query) # Uses defaults
    logger.info(f"Retrieved {len(context_docs)} re-ranked context documents locally.")

    # --- DEBUG Print (Optional) ---
    # print("\n" + "="*10 + " DEBUG: Final Context Sent to LLM " + "="*10)
    # ... (debug print logic) ...
    # print("="*10 + " End DEBUG Context " + "="*10 + "\n")
    # --- END DEBUG ---

    # 3. Format Prompt using the numerical citation version
    prompt = format_rag_prompt(user_query, context_docs, chat_history)

    # 4. Stream prompt to LLM
    logger.info("Streaming prompt to external LLM...")
    response_generator = stream_llm_response(prompt) # Use the streaming function

    if response_generator is None:
        yield "Sorry, the Language Model client is not available."
        return

    # Yield the raw chunks - the LLM output should now be the direct answer + sources
    yield from response_generator


# --- Evaluation Function (Reverted to use get_llm_response) ---
def run_rag_for_evaluation(
    user_query: str,
    top_k: int = FINAL_TOP_K
    ) -> Dict:
    """
    Runs the RAG pipeline for a single query and returns results for evaluation.
    Uses non-streaming LLM call.
    """
    if not is_retriever_ready():
        logger.error("[Evaluation] Retriever not ready.")
        return {
            "query": user_query, "retrieved_context": [], "retrieved_metadata": [],
            "generated_answer": "ERROR: Retriever not initialized.", "error": "Retriever not initialized."
        }

    logger.info(f"[Evaluation] Processing query: '{user_query[:50]}...' with top_k={top_k}")
    final_answer = "ERROR: Processing failed."
    context_texts = []
    context_metadata = []
    error_msg = None

    try:
        # 1. Retrieve Context (No query transform for isolated eval questions)
        context_docs: List[Document] = retrieve_context_local(user_query, final_k=top_k)
        context_texts = [doc.page_content for doc in context_docs]
        context_metadata = [doc.metadata for doc in context_docs]
        logger.info(f"[Evaluation] Retrieved {len(context_docs)} context documents.")

        # 2. Format Prompt (No history for isolated eval questions)
        prompt = format_rag_prompt(user_query, context_docs, chat_history=None)

        # 3. Call LLM (Non-Streaming)
        logger.info("[Evaluation] Sending prompt to external LLM (non-streaming)...")
        generated_answer = get_llm_response(prompt) # Use non-streaming version

        if generated_answer is None:
            final_answer = "ERROR: LLM call failed."
            error_msg = "LLM call failed."
        elif "[SYSTEM:" in generated_answer:
             final_answer = generated_answer
             error_msg = final_answer
        else:
            # Use the direct answer (includes citations and Sources section)
            final_answer = generated_answer
            logger.info(f"[Evaluation] Generated answer length: {len(final_answer)}")
            error_msg = None # Clear error on success

    except Exception as e:
        logger.exception(f"[Evaluation] Unexpected error: {e}", exc_info=True)
        final_answer = f"ERROR: Unexpected error - {e}"
        error_msg = str(e)

    # 5. Return results dictionary
    return {
        "query": user_query,
        "retrieved_context": context_texts,
        "retrieved_metadata": context_metadata,
        "generated_answer": final_answer,
        "error": error_msg
    }
# --- END OF EVALUATION FUNCTION ---


# --- Main Execution Block (Reverted to use query_rag_stream) ---
if __name__ == "__main__":
    if not is_retriever_ready():
        print("\nFATAL ERROR: Failed to initialize local retrieval components.")
        sys.exit(1)

    print("\nNutrition & Fitness RAG Assistant (FAISS + Re-ranking + Query Transform)")
    print("Enter your query below. Type 'quit' to exit.")
    print("-" * 30)
    cli_history = []
    HISTORY_LENGTH = 3
    while True:
        try:
            query = input("Query: ")
            if query.lower() == 'quit': break
            if not query.strip(): continue
            print("\nStreaming Answer:")
            full_response = ""
            recent_history = cli_history[-(HISTORY_LENGTH * 2):]
            # Call the streaming function for interactive mode
            response_gen = query_rag_stream(query, chat_history=recent_history)
            for chunk in response_gen:
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n" + "-" * 30 + "\n")
            # Store the raw response (which includes citations and Sources)
            cli_history.append({"role": "user", "content": query})
            cli_history.append({"role": "assistant", "content": full_response})
        except EOFError: print("\nExiting."); break
        except KeyboardInterrupt: print("\nExiting."); break
        except Exception as e:
            print(f"\nError: {e}")
            logger.exception(f"Unexpected error in CLI: {e}", exc_info=True)
            continue
    print("Goodbye!")
