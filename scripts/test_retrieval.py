from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS # <--- Import FAISS
import pprint
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration (Should match index_data.py) ---
# Define folder path for FAISS index files
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store/faiss_index')
FAISS_INDEX_NAME = "nutrition_fitness_index" # Name used during saving
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    logger.info("--- Testing FAISS Embedding Retrieval ---")

    # --- 1. Initialize the SAME Embedding Model ---
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        embeddings = SentenceTransformerEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}, # Match indexing setting
            encode_kwargs={'normalize_embeddings': True} # Match indexing setting
        )
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        sys.exit(1)

    # --- 2. Load FAISS Index ---
    abs_index_path = os.path.abspath(FAISS_INDEX_PATH)
    faiss_file = os.path.join(abs_index_path, f"{FAISS_INDEX_NAME}.faiss")
    pkl_file = os.path.join(abs_index_path, f"{FAISS_INDEX_NAME}.pkl")

    logger.info(f"Attempting to load FAISS index from: {abs_index_path} (Index name: {FAISS_INDEX_NAME})")
    if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
         logger.error(f"FAISS index files (.faiss and .pkl) not found in {abs_index_path}. Did you run the indexing script successfully?")
         sys.exit(1)
    try:
        # FAISS requires the embedding function during load
        vector_db = FAISS.load_local(
            folder_path=abs_index_path,
            embeddings=embeddings,
            index_name=FAISS_INDEX_NAME,
            # Allow deserialization of SentenceTransformerEmbeddings if needed
            allow_dangerous_deserialization=True
        )
        logger.info(f"FAISS index loaded successfully. Index contains {vector_db.index.ntotal} vectors.")
    except Exception as e:
        logger.exception(f"Error loading FAISS index: {e}", exc_info=True)
        sys.exit(1)

    # --- 3. Define Test Queries ---
    test_queries = [
        "What are the benefits of protein?",
        "How much water should I drink daily?",
        "Recommend beginner exercises for strength training.",
        "What is the role of carbohydrates?",
        "Importance of sleep for fitness recovery",
    ]

    # --- 4. Perform Queries and Evaluate ---
    logger.info("\n--- Running Test Queries ---")
    for query in test_queries:
        logger.info(f"\n[QUERY]: {query}")
        try:
            # FAISS similarity_search_with_score returns Documents and L2 distance scores
            # Lower L2 distance means higher similarity for normalized embeddings
            results_with_scores = vector_db.similarity_search_with_score(query, k=3)

            print("[RESULTS]:") # Use print for direct user output here
            if results_with_scores:
                 for i, (doc, score) in enumerate(results_with_scores):
                     print(f"  --- Result {i+1} (L2 Distance: {score:.4f}) ---") # Score is L2 distance
                     print(f"  Metadata:")
                     pprint.pprint(doc.metadata, indent=4)
                     print(f"  Content: {doc.page_content[:500]}...")
            else:
                print("  No relevant documents found.")
        except Exception as e:
             logger.exception(f"  Error during FAISS query execution for '{query[:50]}...': {e}", exc_info=True)
             print("  Error during query execution.")


if __name__ == "__main__":
    main()