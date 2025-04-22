import os
import re
import logging
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS # <--- Import FAISS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_PATH = "data"
# Define folder path for FAISS index files
FAISS_INDEX_PATH = "vector_store/faiss_index"
FAISS_INDEX_NAME = "nutrition_fitness_index" # Name for the index files (.faiss, .pkl)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK_LENGTH = 50

def clean_whitespace(text):
    """Removes leading/trailing whitespace and collapses internal whitespace."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def is_potentially_irrelevant(text, min_length=MIN_CHUNK_LENGTH):
    """Checks if a chunk is likely irrelevant based on length or simple patterns."""
    if len(text) < min_length: return True
    lower_text = text.lower()
    irrelevant_patterns = [
        r"^\s*table of contents\s*$", r"^\s*references\s*$", r"^\s*bibliography\s*$",
        r"^\s*acknowledgements\s*$", r"^\s*author contributions\s*$",
        r"^\s*conflict of interest\s*$", r"^\s*funding\s*$",
        r"^\s*figure \d+:", r"^\s*table \d+:",
    ]
    for pattern in irrelevant_patterns:
        if re.search(pattern, lower_text): return True
    return False

def main():
    logger.info("Starting data indexing process for FAISS...")

    # --- 1. Load Documents ---
    logger.info(f"Loading documents from: {DATA_PATH}")
    loader = PyPDFDirectoryLoader(DATA_PATH, extract_images=False)
    try:
        documents = loader.load()
    except Exception as e:
        logger.error(f"Failed to load documents from {DATA_PATH}: {e}")
        return
    if not documents:
        logger.warning("No documents found. Exiting.")
        return
    logger.info(f"Loaded {len(documents)} document pages.")

    # --- 2. Split Documents into Chunks ---
    logger.info(f"Splitting documents into chunks (size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        length_function=len, add_start_index=True,
    )
    docs_split = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(docs_split)} initial chunks.")

    # --- 2.5 Clean and Filter Chunks ---
    logger.info("Cleaning and filtering chunks...")
    cleaned_docs = []
    filtered_count = 0
    for i, doc in enumerate(docs_split):
        cleaned_text = clean_whitespace(doc.page_content)
        if cleaned_text and not is_potentially_irrelevant(cleaned_text):
            doc.page_content = cleaned_text
            doc.metadata['chunk_id'] = f"doc_{doc.metadata.get('source', 'unknown').split('/')[-1]}_page_{doc.metadata.get('page', 'N/A')}_chunk_{i}"
            cleaned_docs.append(doc)
        else:
            filtered_count += 1
    if not cleaned_docs:
        logger.error("No valid chunks remaining after cleaning/filtering. Exiting.")
        return
    logger.info(f"Filtered out {filtered_count} chunks.")
    logger.info(f"Retained {len(cleaned_docs)} chunks after cleaning and filtering.")

    # --- 3. Initialize Embedding Model ---
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        embeddings = SentenceTransformerEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
         logger.error(f"Failed to initialize embedding model: {e}")
         return

    # --- 4. Create FAISS Index and Save ---
    try:
        logger.info("Creating FAISS index from documents (this may take a while)...")
        # Create index in memory first
        vector_db = FAISS.from_documents(cleaned_docs, embeddings)
        logger.info("FAISS index created in memory.")

        # Ensure the target directory exists
        abs_index_path = os.path.abspath(FAISS_INDEX_PATH)
        logger.info(f"Ensuring directory exists: {abs_index_path}")
        os.makedirs(abs_index_path, exist_ok=True)

        logger.info(f"Saving FAISS index locally to: {abs_index_path} with index name: {FAISS_INDEX_NAME}")
        # Save the index files to the specified folder
        vector_db.save_local(folder_path=abs_index_path, index_name=FAISS_INDEX_NAME)

        logger.info(f"FAISS index saved successfully. Index contains {vector_db.index.ntotal} vectors.")
        logger.info("Indexing process finished.")

    except Exception as e:
        logger.exception(f"Error during FAISS index creation or saving: {e}", exc_info=True)

if __name__ == "__main__":
    main()