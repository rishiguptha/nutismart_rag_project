import chromadb
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
import logging # Use standard logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
PERSIST_DIRECTORY = os.environ.get("CHROMADB_PATH", "/app/vector_store") # Check Dockerfile ENV
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "nutrition_fitness_docs")

# --- Initialize Components ---
# Load model/DB connection once at module level (FastAPI manages lifecycle)
embeddings = None
collection = None
is_initialized = False

try:
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, # Explicit CPU for containers often safer
        encode_kwargs={'normalize_embeddings': True} # Essential for cosine/L2 equivalence
    )
    logger.info("Embedding model initialized.")

    logger.info(f"Initializing ChromaDB client from path: {PERSIST_DIRECTORY}...")
    if not os.path.exists(PERSIST_DIRECTORY):
        # This check is crucial before initializing the client
        logger.error(f"ChromaDB persist directory not found at: {PERSIST_DIRECTORY}. Did indexing run and is the volume mounted correctly?")
        raise FileNotFoundError(f"Persist directory not found: {PERSIST_DIRECTORY}")

    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

    logger.info(f"Getting collection: {COLLECTION_NAME}...")
    # Use get_collection; it's the retriever's job to read, not create.
    # The indexing script should have created it. Error if not found.
    collection = client.get_collection(name=COLLECTION_NAME)
    logger.info(f"Connected to collection '{COLLECTION_NAME}' with {collection.count()} documents.")
    is_initialized = True

except FileNotFoundError as fnf_error:
    logger.error(f"Initialization failed: {fnf_error}")
    # Specific handling for missing directory
except Exception as e:
    # Catch other potential errors during init (model download, DB connection issues)
    logger.exception(f"Unexpected error during retriever initialization: {e}", exc_info=True)
    # Components remain None, is_initialized remains False

logger.info(f"Retriever initialization status: {'Complete' if is_initialized else 'Failed'}")

def get_relevant_chunks(query: str, k: int) -> list[str]:
    """Embeds query and retrieves top_k relevant chunks."""
    if not is_initialized:
        logger.error("Retriever is not initialized. Cannot process query.")
        # This should ideally be caught by the health check / startup event in FastAPI
        return [] # Return empty or raise an exception

    if not query or not query.strip():
         logger.warning("Received empty query.")
         return [] # Handle empty query gracefully

    logger.info(f"Embedding query (first 50 chars): '{query[:50]}...'")
    try:
        query_embedding = embeddings.embed_query(query)
        logger.info(f"Querying collection '{COLLECTION_NAME}' for top {k} results.")

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents'] # Request only documents for now
        )

        if results and results.get('documents') and results['documents'][0]:
            retrieved_docs = results['documents'][0]
            logger.info(f"Retrieved {len(retrieved_docs)} chunks.")
            return retrieved_docs
        else:
            logger.info("No relevant documents found in ChromaDB for the query.")
            return []
    except Exception as e:
        logger.exception(f"Error during ChromaDB query execution: {e}", exc_info=True)
        return [] # Return empty list on error