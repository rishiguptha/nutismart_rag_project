from fastapi import FastAPI, HTTPException, status # Import status codes
from .models import QueryRequest, ContextResponse
from . import retriever # Import your retriever logic module
import logging

# Use the same logger configured in retriever or setup here
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


app = FastAPI(
    title="RAG Retrieval Service",
    description="Provides semantic search over indexed documents.",
    version="1.0.0"
)

service_ready = False # Flag to track readiness

@app.on_event("startup")
async def startup_event():
    global service_ready
    logger.info("FastAPI startup sequence initiated...")
    if retriever.is_initialized:
        logger.info("Retriever components successfully initialized.")
        service_ready = True
    else:
        logger.error("Retriever components failed to initialize. Service will not be ready.")
        service_ready = False
    logger.info(f"Service ready status: {service_ready}")


@app.post("/retrieve",
          response_model=ContextResponse,
          summary="Retrieve relevant document chunks for a given query",
          tags=["Retrieval"])
async def retrieve_context(request: QueryRequest):
    """
    Accepts a user query and returns a list of the `top_k` most relevant
    document chunks based on semantic similarity.
    """
    if not service_ready:
        logger.error("Service not ready, rejecting request.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retriever service is not initialized or ready. Check server logs."
        )

    logger.info(f"Received retrieval request: query='{request.query[:50]}...', top_k={request.top_k}")
    try:
        # Input validation is handled by Pydantic models (QueryRequest)
        chunks = retriever.get_relevant_chunks(request.query, k=request.top_k)

        # Decide how to handle "no results found" - 200 OK with empty list is often fine
        # Alternatively, raise 404:
        # if not chunks:
        #     raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No relevant context found.")

        logger.info(f"Returning {len(chunks)} chunks for query '{request.query[:50]}...'")
        return ContextResponse(context=chunks)

    except HTTPException as http_exc:
         # Re-raise HTTP exceptions (like validation errors from Pydantic)
         raise http_exc
    except Exception as e:
        logger.exception(f"Unexpected error processing retrieval request for query '{request.query[:50]}...': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during retrieval."
        )

@app.get("/health",
         summary="Health check endpoint",
         tags=["Health"])
async def health_check():
    """
    Checks if the service is initialized and ready to accept requests.
    Verifies connection to the vector database collection.
    """
    if not service_ready or retriever.collection is None:
        logger.warning(f"/health check failed: Service not ready (service_ready={service_ready}, collection_exists={retriever.collection is not None})")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retriever service is not initialized or collection not found."
        )
    try:
        # Perform a cheap check on the collection
        count = retriever.collection.count()
        logger.info(f"/health check successful: Service ready, collection count={count}")
        return {"status": "ok", "collection_count": count}
    except Exception as e:
        logger.exception(f"Error during health check while accessing collection: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service potentially unhealthy: Error accessing collection - {e}"
        )