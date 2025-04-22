import chromadb
from langchain_community.embeddings import SentenceTransformerEmbeddings
import logging
import sys
import os
from dotenv import load_dotenv

# Import functions from other orchestrator files
from .llm_client import get_llm_response
from .prompt_templates import format_rag_prompt

# --- Load Environment Variables ---
# Looks for .env in current working directory or parent directories
load_dotenv()

# --- Configuration ---
# Ensure paths are relative to the project root where this script is likely run from
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), '..', 'vector_store')
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "nutrition_fitness_docs")
DEFAULT_TOP_K = 3 # Default number of chunks to retrieve

# Setup logging configuration for the main application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Ensure logs go to console
    ]
)
logger = logging.getLogger(__name__)


# --- Initialize Retriever Components Directly ---
embeddings = None
collection = None
retriever_ready = False
try:
    logger.info("Initializing retriever components locally...")
    abs_persist_dir = os.path.abspath(PERSIST_DIRECTORY) # Get absolute path for clarity
    logger.info(f"Attempting to load vector store from: {abs_persist_dir}")

    if not os.path.exists(abs_persist_dir):
         logger.error(f"Vector store directory not found at: {abs_persist_dir}. Run the indexing script first.")
         raise FileNotFoundError(f"Vector store directory not found: {abs_persist_dir}")

    # Initialize components needed for retrieval
    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, # Use CPU for wider compatibility
        encode_kwargs={'normalize_embeddings': True}
    )
    client = chromadb.PersistentClient(path=abs_persist_dir)
    collection = client.get_collection(name=COLLECTION_NAME) # Raises exception if not found

    collection_count = collection.count() # Check count early
    if collection_count == 0:
        logger.warning(f"Collection '{COLLECTION_NAME}' exists but is empty. Did indexing produce any valid chunks?")
        # Decide if this is an error state or not - for now, allow it but warn.
    logger.info(f"Retriever components initialized. Collection '{COLLECTION_NAME}' count: {collection_count}")
    retriever_ready = True

except FileNotFoundError as fnf_error:
     logger.error(f"Initialization failed: {fnf_error}")
except Exception as e:
    # Catch specific Chroma/SentenceTransformer errors if needed, otherwise log general
    logger.exception(f"Failed to initialize retriever components: {e}", exc_info=True)

# --- Retrieval Function (Local) ---
def retrieve_context_local(query: str, k: int = DEFAULT_TOP_K) -> list[str]:
    """Embeds query and retrieves top_k relevant document chunks locally."""
    if not retriever_ready:
         logger.error("Local retriever is not ready. Cannot retrieve context.")
         return [] # Return empty list if retriever failed initialization

    if not query or not query.strip():
        logger.warning("Received empty query for retrieval.")
        return [] # Handle empty query

    try:
        logger.info(f"Embedding query locally: '{query[:50]}...'")
        query_embedding = embeddings.embed_query(query) # Use the initialized embedder

        logger.info(f"Querying local ChromaDB collection '{COLLECTION_NAME}' for top {k} results.")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents'] # We only need the text content for the context
        )

        # Safely extract documents
        retrieved_docs = results.get('documents', [[]])[0] # Default to list with empty list if key missing

        if retrieved_docs:
            logger.info(f"Retrieved {len(retrieved_docs)} chunks locally.")
            return retrieved_docs
        else:
            logger.info("No relevant documents found locally in ChromaDB.")
            return []
    except Exception as e:
        logger.exception(f"Error during local ChromaDB query: {e}", exc_info=True)
        return [] # Return empty list on error

# --- Main RAG Query Function ---
def query_rag(user_query: str, top_k: int = DEFAULT_TOP_K):
    """Performs the full RAG pipeline using local retrieval and external LLM."""

    # 1. Retrieve Context Locally
    logger.info(f"Retrieving context locally for query: '{user_query[:50]}...'")
    context_chunks = retrieve_context_local(user_query, k=top_k)
    # No need to check for None here as retrieve_context_local returns []

    logger.info(f"Retrieved {len(context_chunks)} context chunks locally.")

    # 2. Format Prompt
    prompt = format_rag_prompt(user_query, context_chunks)

    # 3. Call External LLM
    logger.info("Sending prompt to external LLM...")
    final_answer = get_llm_response(prompt) # llm_client handles None case

    if final_answer is None:
        # Error logged within get_llm_response, provide user feedback
        return "Sorry, encountered an error getting the response from the language model. Please check logs."

    # 4. Return Result
    return final_answer

# --- Main Execution Block ---
if __name__ == "__main__":
    # Check if retriever is ready before starting the loop
    if not retriever_ready:
        print("\nFATAL ERROR: Failed to initialize local retrieval components (Embedder or Vector Store).")
        print("Please check the logs above for details (e.g., vector_store path, collection name).")
        print("Make sure you have run the 'scripts/index_data.py' script successfully.")
        sys.exit(1) # Exit if retrieval isn't working

    print("\nNutrition & Fitness RAG Assistant")
    print("Enter your query below. Type 'quit' to exit.")
    print("-" * 30)

    # Interactive loop
    while True:
        try:
            query = input("Query: ")
            if query.lower() == 'quit':
                break
            if not query.strip(): # Skip empty input
                continue

            print("Thinking...")
            # Call the main RAG function
            answer = query_rag(query)

            print("\nAnswer:")
            print(answer)
            print("-" * 30 + "\n")

        except EOFError: # Handle Ctrl+D
            print("\nExiting.")
            break
        except KeyboardInterrupt: # Handle Ctrl+C
            print("\nExiting.")
            break