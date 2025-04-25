# orchestrator/main_app.py

# Keep imports for logging, sys, os, dotenv, FAISS, SentenceTransformerEmbeddings
import logging
import sys
import os
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Generator, List # Import List
from langchain_core.documents import Document # Import Document type

# Import functions from other orchestrator files
from .llm_client import stream_llm_response
from .prompt_templates import format_react_style_rag_prompt, extract_final_answer

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

# --- Retrieval Function (Local FAISS Version) ---
# --- MODIFIED TO RETURN DOCUMENTS ---
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
        # FAISS similarity_search returns Document objects directly
        results: List[Document] = vector_db.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(results)} Document objects locally.")
        return results # Return the full Document objects
    except Exception as e:
        logger.exception(f"Error during local FAISS query: {e}", exc_info=True)
        return []

# --- Main RAG Query Function - Now passes Documents to prompt formatter ---
def query_rag_stream(user_query: str, top_k: int = DEFAULT_TOP_K) -> Generator[str, None, None]:
    """
    Performs the RAG pipeline using local retrieval and streams the response
    from the external LLM. Yields text chunks.
    """
    logger.info(f"Retrieving context locally for query: '{user_query[:50]}...'")
    # Now gets a list of Document objects
    context_docs: List[Document] = retrieve_context_local(user_query, k=top_k)
    logger.info(f"Retrieved {len(context_docs)} context documents locally.")

    # Pass the list of Document objects to the formatter
    prompt = format_react_style_rag_prompt(user_query, context_docs)

    logger.info("Streaming prompt to external LLM...")
    response_generator = stream_llm_response(prompt)

    if response_generator is None:
        yield "Sorry, the Language Model client is not available."
        return

    yield from response_generator

# --- Main Execution Block (Unchanged) ---
if __name__ == "__main__":
    if not retriever_ready:
        print("\nFATAL ERROR: Failed to initialize local FAISS retrieval components.")
        sys.exit(1)

    print("\nNutrition & Fitness RAG Assistant (FAISS Version - Streaming Test)")
    print("Enter your query below. Type 'quit' to exit.")
    print("-" * 30)

    while True:
        try:
            query = input("Query: ")
            if query.lower() == 'quit': break
            if not query.strip(): continue

            print("\nStreaming Answer:")
            full_response = ""
            for chunk in query_rag_stream(query):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n" + "-" * 30 + "\n")

        except EOFError: print("\nExiting."); break
        except KeyboardInterrupt: print("\nExiting."); break
