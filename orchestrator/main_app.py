# orchestrator/main_app.py

import logging
import sys
import os
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Generator, List, Dict, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder # <--- Import CrossEncoder

# Import functions from other orchestrator files
from .llm_client import stream_llm_response, get_llm_response # Ensure get_llm_response exists if used by evaluate
from .prompt_templates import format_react_style_rag_prompt, extract_final_answer

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store/faiss_index')
FAISS_INDEX_NAME = "nutrition_fitness_index"
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# Choose a cross-encoder model. ms-marco-MiniLM-L-6-v2 is small and fast.
CROSS_ENCODER_MODEL_NAME = os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
INITIAL_RETRIEVAL_K = 20 # Retrieve more initially for re-ranking
FINAL_TOP_K = 5 # Select the best N after re-ranking

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
cross_encoder = None # Initialize cross_encoder variable
retriever_ready = False
try:
    logger.info("Initializing retriever components locally (FAISS & CrossEncoder)...")
    abs_index_path = os.path.abspath(FAISS_INDEX_PATH)
    faiss_file = os.path.join(abs_index_path, f"{FAISS_INDEX_NAME}.faiss")
    pkl_file = os.path.join(abs_index_path, f"{FAISS_INDEX_NAME}.pkl")
    logger.info(f"Attempting to load FAISS index from: {abs_index_path}")
    if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
        logger.error(f"FAISS index files not found.")
        raise FileNotFoundError(f"FAISS index files not found")

    # Initialize embedding model (for FAISS load/query)
    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    # Load FAISS index
    vector_db = FAISS.load_local(
        folder_path=abs_index_path,
        embeddings=embeddings,
        index_name=FAISS_INDEX_NAME,
        allow_dangerous_deserialization=True
    )
    logger.info(f"FAISS index loaded. Index count: {vector_db.index.ntotal}")

    # Initialize Cross-Encoder model
    logger.info(f"Initializing CrossEncoder model: {CROSS_ENCODER_MODEL_NAME}...")
    # You might want to specify device='mps' for Mac M-series if torch supports it well
    # or device='cuda' for NVIDIA, default is usually CPU.
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME, max_length=512, device='cpu')
    logger.info("CrossEncoder model initialized.")

    retriever_ready = True # Set ready only if all components load

except FileNotFoundError as fnf_error:
     logger.error(f"Initialization failed: {fnf_error}")
except Exception as e:
    logger.exception(f"Failed to initialize retriever components: {e}", exc_info=True)

# --- Readiness Check Function ---
def is_retriever_ready() -> bool:
    """Checks if the retriever components (FAISS & CrossEncoder) initialized successfully."""
    # Check all necessary components
    return retriever_ready and vector_db is not None and cross_encoder is not None and embeddings is not None

# --- Retrieval Function with Re-ranking ---
def retrieve_context_local(query: str, initial_k: int = INITIAL_RETRIEVAL_K, final_k: int = FINAL_TOP_K) -> List[Document]:
    """
    Retrieves initial documents using FAISS, then re-ranks them using a CrossEncoder.
    Returns the top 'final_k' re-ranked Document objects.
    """
    if not is_retriever_ready():
         logger.error("Local retriever components (FAISS or CrossEncoder) are not ready.")
         return []
    if not query or not query.strip():
        logger.warning("Received empty query for retrieval.")
        return []

    try:
        # 1. Initial Retrieval (Semantic Search from FAISS)
        logger.info(f"Querying FAISS for initial top {initial_k} results: '{query[:50]}...'")
        initial_results: List[Document] = vector_db.similarity_search(query, k=initial_k)
        logger.info(f"Retrieved {len(initial_results)} initial Document objects.")

        if not initial_results:
            logger.info("No initial results found from FAISS.")
            return [] # No need to re-rank if nothing was found

        # 2. Re-ranking with CrossEncoder
        logger.info(f"Re-ranking {len(initial_results)} documents using CrossEncoder: {CROSS_ENCODER_MODEL_NAME}...")
        # Create pairs of [query, document_content] for the cross-encoder input
        cross_inp = [[query, doc.page_content] for doc in initial_results]

        # Predict relevance scores (higher score = more relevant)
        # This can be computationally intensive depending on the model and list size
        cross_scores = cross_encoder.predict(cross_inp, show_progress_bar=False) # Set True for progress bar
        logger.info(f"CrossEncoder scores calculated.")

        # Combine initial documents with their new relevance scores
        doc_scores = list(zip(initial_results, cross_scores))

        # Sort the combined list by score in descending order
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # 3. Select Top K Re-ranked Documents
        reranked_docs = [doc for doc, score in doc_scores[:final_k]] # Get top 'final_k' documents
        logger.info(f"Returning top {len(reranked_docs)} re-ranked documents.")

        # Optional: Log scores for debugging
        # for i, (doc, score) in enumerate(doc_scores[:final_k]):
        #     logger.debug(f"  Rank {i+1} Score: {score:.4f} - Doc: {doc.page_content[:100]}...")

        return reranked_docs

    except Exception as e:
        logger.exception(f"Error during retrieval or re-ranking: {e}", exc_info=True)
        return []

# --- Main RAG Query Function (Uses re-ranked context) ---
def query_rag_stream(
    user_query: str,
    chat_history: List[Dict[str, str]] = None
    # top_k is now handled internally by retrieve_context_local
    ) -> Generator[str, None, None]:
    """
    Performs the RAG pipeline using local re-ranked retrieval, considering chat history,
    and streams the response from the external LLM. Yields text chunks.
    """
    logger.info(f"Retrieving and re-ranking context locally for query: '{user_query[:50]}...'")
    # Call retrieve_context_local, which now performs re-ranking
    # It uses INITIAL_RETRIEVAL_K and FINAL_TOP_K defined above
    context_docs: List[Document] = retrieve_context_local(user_query)
    logger.info(f"Retrieved {len(context_docs)} re-ranked context documents locally.")

    # Pass the re-ranked documents to the prompt formatter
    prompt = format_react_style_rag_prompt(user_query, context_docs, chat_history)

    logger.info("Streaming prompt to external LLM...")
    response_generator = stream_llm_response(prompt)

    if response_generator is None:
        yield "Sorry, the Language Model client is not available."
        return

    yield from response_generator


# --- Main Execution Block (Unchanged - for CLI testing) ---
if __name__ == "__main__":
    if not is_retriever_ready(): # Use updated check
        print("\nFATAL ERROR: Failed to initialize local retrieval components (FAISS or CrossEncoder).")
        sys.exit(1)

    print("\nNutrition & Fitness RAG Assistant (FAISS + Re-ranking - Streaming Test)")
    print("Enter your query below. Type 'quit' to exit.")
    print("-" * 30)
    # (Rest of __main__ block remains the same)
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
            response_gen = query_rag_stream(query, chat_history=recent_history)

            for chunk in response_gen:
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n" + "-" * 30 + "\n")

            final_answer_hist = extract_final_answer(full_response)
            cli_history.append({"role": "user", "content": query})
            cli_history.append({"role": "assistant", "content": final_answer_hist})

        except EOFError: print("\nExiting."); break
        except KeyboardInterrupt: print("\nExiting."); break

