import chromadb
from langchain_community.embeddings import SentenceTransformerEmbeddings
import pprint # For nicely printing results

# --- Configuration (Should match index_data.py) ---
PERSIST_DIRECTORY = "vector_store" # Path relative to project root
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "nutrition_fitness_docs"

def main():
    print("--- Testing Embedding Retrieval ---")

    # --- 1. Initialize ChromaDB Client ---
    print(f"Connecting to ChromaDB at: {PERSIST_DIRECTORY}")
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

    # --- 2. Initialize the SAME Embedding Model ---
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    # It's crucial to use the exact same model & settings as during indexing
    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, # Match indexing setting
        encode_kwargs={'normalize_embeddings': True} # Match indexing setting
    )

    # --- 3. Get the ChromaDB Collection ---
    try:
        print(f"Getting collection: {COLLECTION_NAME}")
        # Get embedding function from LangChain wrapper to ensure compatibility with Chroma query
        lc_chroma_embed_func = embeddings.embed_query # LangChain's embedding function interface
        collection = client.get_collection(
            name=COLLECTION_NAME
            # If you didn't use the default embedding function during indexing,
            # you might need to pass the *exact same* embedding function instance here too,
            # but usually get_collection just needs the name. The query needs the embedding.
            )
        print(f"Collection '{COLLECTION_NAME}' found with {collection.count()} documents.")
    except Exception as e:
        print(f"Error getting collection: {e}")
        print("Did you run the indexing script successfully?")
        return

    # --- 4. Define Test Queries ---
    test_queries = [
        "What are the benefits of protein?",
        "How much water should I drink daily?",
        "Recommend beginner exercises for strength training.",
        "What is the role of carbohydrates?",
        "Importance of sleep for fitness recovery",
        # Add more queries specific to content you expect in your documents
    ]

    # --- 5. Perform Queries and Evaluate ---
    print("\n--- Running Test Queries ---")
    for query in test_queries:
        print(f"\n[QUERY]: {query}")

        # Embed the query using the LangChain wrapper's method
        query_embedding = embeddings.embed_query(query)

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding], # Chroma expects a list of embeddings
            n_results=3, # Number of results to retrieve
            include=['documents', 'metadatas', 'distances'] # Ask for text, metadata, and distance
        )

        print("[RESULTS]:")
        if results and results.get('documents') and results['documents'][0]:
             # results is structured like {'ids': [[id1, id2,..]], 'documents': [[doc1, doc2,..]], ...} for batched queries
             # Since we query one at a time, we access the first element (index 0) of each list.
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i]
                metadata = results['metadatas'][0][i]
                print(f"  --- Result {i+1} (Distance: {distance:.4f}) ---")
                print(f"  Metadata: {metadata}")
                print(f"  Content: {doc[:500]}...") # Print start of the chunk
        else:
            print("  No relevant documents found.")

if __name__ == "__main__":
    main()