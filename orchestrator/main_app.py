# orchestrator/main_app.py

# (Keep all imports and initialization code as is)
import logging
import sys
import os
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from typing import Generator, List, Dict, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import pprint # Import pprint for nice printing

from .llm_client import stream_llm_response, get_llm_response, transform_query_with_history
from .prompt_templates import format_rag_prompt

load_dotenv()
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store/faiss_index')
FAISS_INDEX_NAME = "nutrition_fitness_index"
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL_NAME = os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
INITIAL_RETRIEVAL_K = 20
FINAL_TOP_K = 10
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

vector_db = None
embeddings = None
cross_encoder = None
retriever_ready = False
# (Keep the try-except block for initialization as is)
try:
    logger.info("Initializing retriever components locally (FAISS & CrossEncoder)...")
    abs_index_path = os.path.abspath(FAISS_INDEX_PATH)
    faiss_file = os.path.join(abs_index_path, f"{FAISS_INDEX_NAME}.faiss")
    pkl_file = os.path.join(abs_index_path, f"{FAISS_INDEX_NAME}.pkl")
    logger.info(f"Attempting to load FAISS index from: {abs_index_path}")
    if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
        logger.error(f"FAISS index files not found.")
        raise FileNotFoundError(f"FAISS index files not found")
    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True}
    )
    vector_db = FAISS.load_local(
        folder_path=abs_index_path, embeddings=embeddings, index_name=FAISS_INDEX_NAME, allow_dangerous_deserialization=True
    )
    logger.info(f"FAISS index loaded. Index count: {vector_db.index.ntotal}")
    logger.info(f"Initializing CrossEncoder model: {CROSS_ENCODER_MODEL_NAME}...")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME, max_length=512, device='cpu')
    logger.info("CrossEncoder model initialized.")
    retriever_ready = True
except FileNotFoundError as fnf_error:
     logger.error(f"Initialization failed: {fnf_error}")
except Exception as e:
    logger.exception(f"Failed to initialize retriever components: {e}", exc_info=True)


def is_retriever_ready() -> bool:
    return retriever_ready and vector_db is not None and cross_encoder is not None and embeddings is not None

# (Keep retrieve_context_local function as is)
def retrieve_context_local(query: str, initial_k: int = INITIAL_RETRIEVAL_K, final_k: int = FINAL_TOP_K) -> List[Document]:
    if not is_retriever_ready():
         logger.error("Local retriever components are not ready.")
         return []
    if not query or not query.strip():
        logger.warning("Received empty query for retrieval.")
        return []
    try:
        logger.info(f"Querying FAISS for initial top {initial_k} results: '{query[:50]}...'")
        initial_results: List[Document] = vector_db.similarity_search(query, k=initial_k)
        logger.info(f"Retrieved {len(initial_results)} initial Document objects.")
        if not initial_results: return []
        logger.info(f"Re-ranking {len(initial_results)} documents...")
        cross_inp = [[query, doc.page_content] for doc in initial_results]
        cross_scores = cross_encoder.predict(cross_inp, show_progress_bar=False)
        logger.info(f"CrossEncoder scores calculated.")
        doc_scores = list(zip(initial_results, cross_scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in doc_scores[:final_k]]
        logger.info(f"Returning top {len(reranked_docs)} re-ranked documents.")
        return reranked_docs
    except Exception as e:
        logger.exception(f"Error during retrieval or re-ranking: {e}", exc_info=True)
        return []


# --- Main RAG Query Function - ADD DEBUG PRINT ---
def query_rag_stream(
    user_query: str,
    chat_history: List[Dict[str, str]] = None
    ) -> Generator[str, None, None]:
    """
    Performs the RAG pipeline: Query Transformation -> Retrieval -> Re-ranking -> LLM Stream.
    """
    # 1. Transform Query
    if chat_history:
        retrieval_query = transform_query_with_history(user_query, chat_history)
    else:
        retrieval_query = user_query
    logger.info(f"Using query for retrieval: '{retrieval_query[:50]}...'")

    # 2. Retrieve and Re-rank Context
    context_docs: List[Document] = retrieve_context_local(retrieval_query)
    logger.info(f"Retrieved {len(context_docs)} re-ranked context documents locally.")

    # --- DEBUG: Print the final context being sent to LLM ---
    print("\n" + "="*10 + " DEBUG: Final Context Sent to LLM " + "="*10)
    if context_docs:
        for i, doc in enumerate(context_docs):
            print(f"--- Context Doc {i+1} ---")
            print(f"Metadata:")
            pprint.pprint(doc.metadata, indent=2)
            print(f"Content Preview:\n{doc.page_content[:500]}...") # Show start of content
            print("-" * 20)
    else:
        print("  (No context documents were retrieved)")
    print("="*10 + " End DEBUG Context " + "="*10 + "\n")
    # --- END DEBUG ---

    # 3. Format Prompt
    prompt = format_rag_prompt(user_query, context_docs, chat_history)

    # 4. Stream prompt to LLM
    logger.info("Streaming prompt to external LLM...")
    response_generator = stream_llm_response(prompt)

    if response_generator is None:
        yield "Sorry, the Language Model client is not available."
        return

    yield from response_generator


# --- Main Execution Block (Unchanged) ---
if __name__ == "__main__":
    # (Keep the existing __main__ block)
    if not is_retriever_ready():
        print("\nFATAL ERROR: Failed to initialize local retrieval components.")
        sys.exit(1)
    print("\nNutrition & Fitness RAG Assistant (FAISS + Re-ranking + Query Transform)")
    print("Enter your query below. Type 'quit' to exit.")
    print("-" * 30)
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
            cli_history.append({"role": "user", "content": query})
            cli_history.append({"role": "assistant", "content": full_response})
        except EOFError: print("\nExiting."); break
        except KeyboardInterrupt: print("\nExiting."); break
