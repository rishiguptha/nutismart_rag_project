# Keep imports for logging, sys, os, dotenv, llm_client, prompt_templates
import logging
import sys
import os
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS # <--- Import FAISS

# Import functions from other orchestrator files
from .llm_client import get_llm_response
from .prompt_templates import format_react_style_rag_prompt, extract_final_answer # Import NEW functions

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
# Define folder path for FAISS index files relative to this script's parent dir
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store/faiss_index')
FAISS_INDEX_NAME = "nutrition_fitness_index" # Must match index_data.py
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_TOP_K = 7

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# --- Initialize Retriever Components Directly (FAISS Version) ---
vector_db = None # Use vector_db instead of collection
embeddings = None
retriever_ready = False
try:
    logger.info("Initializing retriever components locally (FAISS)...")
    abs_index_path = os.path.abspath(FAISS_INDEX_PATH)
    faiss_file = os.path.join(abs_index_path, f"{FAISS_INDEX_NAME}.faiss")
    pkl_file = os.path.join(abs_index_path, f"{FAISS_INDEX_NAME}.pkl")

    logger.info(f"Attempting to load FAISS index from: {abs_index_path} (Index name: {FAISS_INDEX_NAME})")

    if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
        logger.error(f"FAISS index files (.faiss and .pkl) not found in {abs_index_path}. Run the indexing script first.")
        raise FileNotFoundError(f"FAISS index files not found")

    # Initialize components needed for retrieval
    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, # Use CPU for wider compatibility
        encode_kwargs={'normalize_embeddings': True}
    )

    # Load the FAISS index from local files
    vector_db = FAISS.load_local(
        folder_path=abs_index_path,
        embeddings=embeddings,
        index_name=FAISS_INDEX_NAME,
        allow_dangerous_deserialization=True # Often required for SentenceTransformerEmbeddings
    )
    logger.info(f"FAISS retriever components initialized. Index contains {vector_db.index.ntotal} vectors.")
    retriever_ready = True

except FileNotFoundError as fnf_error:
     logger.error(f"Initialization failed: {fnf_error}")
except Exception as e:
    logger.exception(f"Failed to initialize FAISS retriever components: {e}", exc_info=True)

# --- Retrieval Function (Local FAISS Version) ---
def retrieve_context_local(query: str, k: int = DEFAULT_TOP_K) -> list[str]:
    """Embeds query and retrieves top_k relevant document chunks locally using FAISS."""
    if not retriever_ready or vector_db is None:
         logger.error("Local FAISS retriever is not ready.")
         return []

    if not query or not query.strip():
        logger.warning("Received empty query for retrieval.")
        return []

    try:
        logger.info(f"Querying local FAISS index for top {k} results: '{query[:50]}...'")
        # FAISS similarity_search returns Document objects by default based on relevance score (typically L2 distance for normalized embeddings)
        results = vector_db.similarity_search(query, k=k)
        # Extract just the page content for the context
        retrieved_docs = [doc.page_content for doc in results]
        logger.info(f"Retrieved {len(retrieved_docs)} chunks locally.")
        return retrieved_docs
    except Exception as e:
        logger.exception(f"Error during local FAISS query: {e}", exc_info=True)
        return []

# --- Main RAG Query Function (Unchanged internally, calls new retrieve_context_local) ---
def query_rag(user_query: str, top_k: int = DEFAULT_TOP_K):
    """Performs the full RAG pipeline using local retrieval and external LLM."""
    logger.info(f"Retrieving context locally for query: '{user_query[:50]}...'")
    context_chunks = retrieve_context_local(user_query, k=top_k)
    logger.info(f"Retrieved {len(context_chunks)} context chunks locally.")

    prompt = format_react_style_rag_prompt(user_query, context_chunks) # Call NEW function

    # 3. Call External LLM
    logger.info("Sending prompt to external LLM...")
    raw_llm_response = get_llm_response(prompt) # Get the full output including Thought

    if raw_llm_response is None:
        # Error logged within get_llm_response
        return "Sorry, encountered an error getting the response from the language model. Please check logs."

    # 4. Extract Final Answer from LLM's full output
    final_answer = extract_final_answer(raw_llm_response) # Add this extraction step

    # 5. Return Result
    return final_answer # Return the extracted answer

# --- Main Execution Block (Unchanged) ---
if __name__ == "__main__":
    if not retriever_ready:
        print("\nFATAL ERROR: Failed to initialize local FAISS retrieval components.")
        print("Please check the logs above for details (e.g., FAISS index path).")
        print("Make sure you have run the 'scripts/index_data.py' script successfully after modifying it for FAISS.")
        sys.exit(1)

    print("\nNutrition & Fitness RAG Assistant (FAISS Version)")
    print("Enter your query below. Type 'quit' to exit.")
    print("-" * 30)

    while True:
        try:
            query = input("Query: ")
            if query.lower() == 'quit': break
            if not query.strip(): continue
            print("Thinking...")
            answer = query_rag(query)
            print("\nAnswer:")
            print(answer)
            print("-" * 30 + "\n")
        except EOFError: print("\nExiting."); break
        except KeyboardInterrupt: print("\nExiting."); break