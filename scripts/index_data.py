# scripts/index_data.py

import os
import re
import logging
import time # Import time for delays
import sys
import pickle
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader # Add WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
# Optional: Import BeautifulSoup if using advanced bs_kwargs
# import bs4

# --- Add project root to path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# --- Improved Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Console handler
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(log_formatter)
    logger.addHandler(ch)
# File handler
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    fh = logging.FileHandler('app.log', mode='a')
    fh.setFormatter(log_formatter)
    logger.addHandler(fh)

# --- Configuration ---
DATA_PATH = "data" # For local PDFs
FAISS_INDEX_PATH = "vector_store/faiss_index" # Relative to project root now
FAISS_INDEX_NAME = "nutrition_fitness_index"
DOCS_PKL_NAME = "nutrition_fitness_docs.pkl"  # New separate file for documents
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 384
CHUNK_OVERLAP = 50
MIN_CHUNK_LENGTH = 50
WEB_REQUEST_DELAY_SECONDS = 20 # Delay between loading web pages

# --- Updated URLs to Ingest (Based on Search Results - April 2025) ---
CDC_NUTRITION_URLS = [
    "https://www.cdc.gov/nutrition/features/healthy-eating-tips.html", # General Tips
    "https://www.cdc.gov/healthy-weight-growth/healthy-eating/index.html", # Healthy Eating Overview 
    # Specific macro pages seem less prominent, rely on general pages or add if found later
    "https://www.cdc.gov/nutrition/features/micronutrient-facts.html", # Micronutrient Facts
    "https://www.cdc.gov/nutrition/features/why-micronutrients-matter.html", # Why Micronutrients Matter
    "https://www.cdc.gov/healthy-weight-growth/water-healthy-drinks/index.html", # Water/Drinks
]

CDC_ACTIVITY_URLS = [
    "https://www.cdc.gov/physical-activity-basics/guidelines/index.html", # Guidelines Overview
    "https://www.cdc.gov/physical-activity-basics/guidelines/adults.html", # Adults specific (Note: .html)
    "https://www.cdc.gov/physical-activity-basics/measuring/index.html", # Measuring intensity (Note: .html)
    "https://www.cdc.gov/physical-activity-basics/adding-adults/index.html", # Adding PA (Note: .html)
    "https://www.cdc.gov/active-people-healthy-nation/php/why-be-active/index.html", # Benefits Overview
    "https://www.cdc.gov/physical-activity/features/10-reasons-to-get-moving.html" # 10 Reasons List
]

ALL_WEB_URLS = CDC_NUTRITION_URLS + CDC_ACTIVITY_URLS

# --- Helper Functions ---
def clean_whitespace(text):
    """Removes leading/trailing whitespace and collapses internal whitespace."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text) # Replace multiple whitespace chars with single space
    return text

def is_potentially_irrelevant(text, min_length=MIN_CHUNK_LENGTH):
    """Checks if a chunk is likely irrelevant based on length or simple patterns."""
    if len(text) < min_length: return True
    lower_text = text.lower()
    # More aggressive filtering for web boilerplate
    irrelevant_patterns = [
        r"^\s*table of contents\s*$", r"^\s*references\s*$", r"^\s*bibliography\s*$",
        r"^\s*acknowledgements\s*$", r"^\s*author contributions\s*$",
        r"^\s*conflict of interest\s*$", r"^\s*funding\s*$",
        r"^\s*figure \d+:", r"^\s*table \d+:",
        # CDC specific / common web patterns
        r"file size:", r"page last reviewed:", r"page last updated:", r"content source:",
        r"related pages", r"on this page", r"get email updates", r"recommend",
        r"facebook", r"twitter", r"linkedin", r"syndicate", r"cdc twenty four seven",
        r"saving lives, protecting people", r"centers for disease control and prevention",
        r"u\.s\. department of health & human services", r"usa\.gov",
        r"^\s*related topics\s*$", r"^\s*key points\s*$", r"^\s*resources\s*$",
        r"^\s*print page\s*$", r"^\s*email page\s*$",
    ]
    # Check common phrases often found in headers/footers/sidebars
    common_boilerplate = [
        "skip directly to main content", "skip directly to page options",
        "skip directly to a-z link", "site map", "policies", "disclaimers",
        "foia", "no fear act", "oig", "accessibility", "privacy", "contact us",
        "file formats help", "language assistance",
    ]

    for pattern in irrelevant_patterns:
        if re.search(pattern, lower_text, re.IGNORECASE):
            # logger.debug(f"Filtering chunk due to pattern: {pattern}")
            return True

    for phrase in common_boilerplate:
        if phrase in lower_text:
            # logger.debug(f"Filtering chunk due to boilerplate phrase: {phrase}")
            return True

    # Filter if very few alphabetic characters (likely code snippets or symbols)
    alpha_chars = sum(c.isalpha() for c in text)
    if len(text) > 20 and alpha_chars / len(text) < 0.5: # Less than 50% alpha chars
        # logger.debug(f"Filtering chunk due to low alpha char ratio: {text[:100]}...")
        return True

    return False

def process_and_clean_docs(documents, source_type="PDF"):
    """Splits, cleans, and filters documents."""
    logger.info(f"Processing {len(documents)} raw {source_type} documents...")
    if not documents: return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        length_function=len, add_start_index=True,
    )
    docs_split = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(docs_split)} initial chunks.")

    logger.info("Cleaning and filtering chunks...")
    cleaned_docs = []
    filtered_count = 0
    for i, doc in enumerate(docs_split):
        # Ensure metadata exists and add chunk_id
        if not hasattr(doc, 'metadata') or doc.metadata is None:
             doc.metadata = {}
        source_name = doc.metadata.get('source', f'unknown_{source_type}_{i}')
        # Use URL for web, filename for PDF
        doc_id_source = source_name.split('/')[-1].split('\\')[-1] if source_type == "PDF" else source_name
        page_num = doc.metadata.get('page', 'Web') # Use 'Web' if no page number
        doc.metadata['chunk_id'] = f"doc_{doc_id_source}_page_{page_num}_chunk_{i}"

        # Clean whitespace first
        cleaned_text = clean_whitespace(doc.page_content)

        # Filter based on cleaned text
        if cleaned_text and not is_potentially_irrelevant(cleaned_text):
            doc.page_content = cleaned_text # Update with cleaned text
            cleaned_docs.append(doc)
        else:
            filtered_count += 1
            # logger.debug(f"Filtered out {source_type} chunk {i} (length {len(cleaned_text)}): {cleaned_text[:100]}...")

    logger.info(f"Filtered out {filtered_count} {source_type} chunks.")
    logger.info(f"Retained {len(cleaned_docs)} cleaned {source_type} chunks.")
    return cleaned_docs


def main():
    logger.info("--- Starting Combined PDF & Web Data Indexing for FAISS ---")
    all_cleaned_docs = []
    abs_index_path = os.path.abspath(FAISS_INDEX_PATH)

    # --- Delete existing index directory ---
    if os.path.exists(abs_index_path):
        logger.warning(f"Existing index found at {abs_index_path}. Deleting before re-indexing.")
        import shutil
        try:
            shutil.rmtree(abs_index_path)
            logger.info("Deleted existing index.")
        except Exception as e:
            logger.error(f"Could not delete existing index directory: {e}. Please delete it manually.")
            return

    # --- 1. Process Local PDFs ---
    logger.info(f"Loading local PDF documents from: {DATA_PATH}")
    abs_data_path = os.path.abspath(DATA_PATH)
    if not os.path.isdir(abs_data_path):
        logger.warning(f"Data directory '{abs_data_path}' not found. Skipping PDF processing.")
    else:
        pdf_loader = PyPDFDirectoryLoader(abs_data_path, extract_images=False)
        try:
            pdf_documents = pdf_loader.load()
            if pdf_documents:
                logger.info(f"Loaded {len(pdf_documents)} PDF document pages.")
                pdf_cleaned_docs = process_and_clean_docs(pdf_documents, source_type="PDF")
                all_cleaned_docs.extend(pdf_cleaned_docs)
            else:
                logger.info("No PDF documents found in the data directory.")
        except Exception as e:
            logger.exception(f"Failed to load or process PDFs from {abs_data_path}: {e}", exc_info=True)

    # --- 2. Process Web URLs ---
    logger.info(f"Loading and processing {len(ALL_WEB_URLS)} web URLs...")
    web_docs_processed = []
    # Be a good bot citizen - identify your bot
    headers = {'User-Agent': 'NutiSmartBot/1.0 (Educational RAG Project; contact: your_email@example.com)'} # Replace with real contact if deploying

    for i, url in enumerate(ALL_WEB_URLS):
        logger.info(f"Loading URL ({i+1}/{len(ALL_WEB_URLS)}): {url}")
        try:
            loader = WebBaseLoader(web_paths=[url], header_template=headers)
            docs = loader.load()
            if docs:
                 # Add URL to metadata if not already present by loader
                 for doc in docs:
                     if 'source' not in doc.metadata:
                         doc.metadata['source'] = url
                 web_docs_processed.extend(process_and_clean_docs(docs, source_type="Web"))
            else:
                 logger.warning(f"No content loaded from URL: {url}")
        except Exception as e:
            logger.error(f"Failed to load or process URL {url}: {e}", exc_info=False)

        logger.info(f"Waiting {WEB_REQUEST_DELAY_SECONDS} seconds before next request...")
        time.sleep(WEB_REQUEST_DELAY_SECONDS)

    all_cleaned_docs.extend(web_docs_processed)
    logger.info(f"Total cleaned documents from all sources: {len(all_cleaned_docs)}")

    if not all_cleaned_docs:
        logger.error("No documents processed from any source. Cannot create index. Exiting.")
        return

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

    # --- 4. Create FAISS Index and Save Everything ---
    try:
        logger.info(f"Creating FAISS index from {len(all_cleaned_docs)} total documents (this may take a while)...")
        vector_db = FAISS.from_documents(all_cleaned_docs, embeddings)
        logger.info("FAISS index created in memory.")

        logger.info(f"Ensuring directory exists: {abs_index_path}")
        os.makedirs(abs_index_path, exist_ok=True)

        logger.info(f"Saving FAISS index locally to: {abs_index_path} with index name: {FAISS_INDEX_NAME}")
        vector_db.save_local(folder_path=abs_index_path, index_name=FAISS_INDEX_NAME)

        # Save documents separately
        docs_path = os.path.join(abs_index_path, DOCS_PKL_NAME)
        logger.info(f"Saving {len(all_cleaned_docs)} documents to: {docs_path}")
        with open(docs_path, 'wb') as f:
            pickle.dump(all_cleaned_docs, f)
        logger.info("Documents saved successfully.")

        # Verify the saves
        logger.info("Verifying saved files...")
        with open(docs_path, 'rb') as f:
            loaded_docs = pickle.load(f)
        loaded_db = FAISS.load_local(folder_path=abs_index_path, embeddings=embeddings, index_name=FAISS_INDEX_NAME)
        
        logger.info(f"Verification complete:")
        logger.info(f"  - Documents file contains {len(loaded_docs)} documents")
        logger.info(f"  - FAISS index contains {loaded_db.index.ntotal} vectors")
        
        if len(loaded_docs) != loaded_db.index.ntotal:
            logger.warning("WARNING: Number of documents does not match number of vectors!")
        else:
            logger.info("SUCCESS: Number of documents matches number of vectors.")

        logger.info("Combined Indexing process finished.")

    except Exception as e:
        logger.exception(f"Error during FAISS index creation or saving: {e}", exc_info=True)

if __name__ == "__main__":
    main()
