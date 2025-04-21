import os
import re # Import regex library
import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
DATA_PATH = "data"
PERSIST_DIRECTORY = "vector_store"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
COLLECTION_NAME = "nutrition_fitness_docs"
# --- Add Cleaning Config ---
MIN_CHUNK_LENGTH = 50 # Minimum number of characters for a chunk to be kept

def clean_whitespace(text):
    """Removes leading/trailing whitespace and collapses internal whitespace."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text) # Replace multiple whitespace chars with single space
    # Optional: replace multiple newlines if desired, but sometimes structure matters
    # text = re.sub(r'\n+', '\n', text)
    return text

def is_potentially_irrelevant(text, min_length=MIN_CHUNK_LENGTH):
    """Checks if a chunk is likely irrelevant based on length or simple patterns."""
    # 1. Check length
    if len(text) < min_length:
        return True

    # 2. Check for patterns indicative of irrelevant content (case-insensitive)
    # Customize this list based on your documents!
    lower_text = text.lower()
    irrelevant_patterns = [
        r"^\s*table of contents\s*$",
        r"^\s*references\s*$",
        r"^\s*bibliography\s*$",
        r"^\s*acknowledgements\s*$",
        r"^\s*author contributions\s*$",
        r"^\s*conflict of interest\s*$",
        r"^\s*funding\s*$",
        r"^\s*figure \d+:", # Start of a figure caption
        r"^\s*table \d+:",  # Start of a table caption
        # Consider adding patterns for headers/footers if they are very consistent
        # e.g., r"^\s*page \d+\s*$" - but this might be too aggressive
    ]
    for pattern in irrelevant_patterns:
        if re.search(pattern, lower_text):
            return True # Found an irrelevant pattern

    # 3. Add more checks if needed (e.g., very low alphabetic character ratio)

    return False # Chunk seems okay

def main():
    print("Starting data indexing process...")

    # --- 1. Load Documents ---
    print(f"Loading documents from: {DATA_PATH}")
    loader = PyPDFDirectoryLoader(DATA_PATH, extract_images=False) # ensure images aren't attempted
    documents = loader.load()
    if not documents:
        print("No documents found. Exiting.")
        return
    print(f"Loaded {len(documents)} document pages.")

    # --- 2. Split Documents into Chunks ---
    print(f"Splitting documents into chunks (size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True, # Add start index to metadata, useful for context
    )
    docs_split = text_splitter.split_documents(documents)
    print(f"Split into {len(docs_split)} initial chunks.")

    # --- 2.5 Clean and Filter Chunks ---
    print("Cleaning and filtering chunks...")
    cleaned_docs = []
    filtered_count = 0
    for i, doc in enumerate(docs_split):
        cleaned_text = clean_whitespace(doc.page_content)
        if cleaned_text and not is_potentially_irrelevant(cleaned_text):
            # Update the page content with the cleaned version
            doc.page_content = cleaned_text
            # Add a unique ID within the metadata might be useful
            doc.metadata['chunk_id'] = f"doc_{doc.metadata.get('source', 'unknown').split('/')[-1]}_page_{doc.metadata.get('page', 'N/A')}_chunk_{i}"
            cleaned_docs.append(doc)
        else:
            filtered_count += 1
            # Optional: Log filtered chunks for review
            # print(f"Filtered out chunk {i} (length {len(cleaned_text)}): {cleaned_text[:100]}...")

    if not cleaned_docs:
        print("No valid chunks remaining after cleaning/filtering. Exiting.")
        return
    print(f"Filtered out {filtered_count} chunks.")
    print(f"Retained {len(cleaned_docs)} chunks after cleaning and filtering.")

    # --- 3. Initialize Embedding Model ---
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, # Explicitly use CPU to save GPU RAM if needed elsewhere
        encode_kwargs={'normalize_embeddings': True} # Normalize for cosine similarity
        )

    # --- 4. Initialize ChromaDB and Store Embeddings ---
    print(f"Initializing ChromaDB persistent client at: {PERSIST_DIRECTORY}")
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

    print(f"Getting or creating collection: {COLLECTION_NAME}")
    # Let Chroma handle embedding using its default SentenceTransformer integration,
    # ensuring it uses the same model. Or explicitly pass the LangChain embedder.
    # For consistency, using the LangChain wrapper with Chroma is often clearer.
    

    print("Creating/updating ChromaDB collection with documents (this may take a while)...")
    # This will handle embedding the cleaned_docs using the provided embedding function
    vector_db = Chroma.from_documents(
        documents=cleaned_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME
        # ids=[doc.metadata['chunk_id'] for doc in cleaned_docs] # Providing IDs is good practice
    )

    # Optional: Persist explicitly if needed, though from_documents often does.
    # vector_db.persist()

    # Can't easily get the count from the LangChain wrapper creation method directly.
    # We can instantiate the client again to check count.
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' now contains {collection.count()} documents.")
    print("Indexing process finished.")

if __name__ == "__main__":
    main()