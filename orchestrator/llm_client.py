# orchestrator/llm_client.py

import os
import google.generativeai as genai
import logging
from dotenv import load_dotenv
from typing import Generator, Optional, List, Dict

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

API_KEY = os.environ.get("GOOGLE_API_KEY")
model = None
MODEL_NAME = os.environ.get("GENERATIVE_MODEL_NAME", "gemini-2.0-flash")

# --- Configure the Google AI Client ---
try:
    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=API_KEY)
    # Initialize model without tools
    model = genai.GenerativeModel(MODEL_NAME)
    logger.info(f"Google AI Client configured successfully for model: {MODEL_NAME}")
except Exception as e:
    logger.exception(f"Failed to configure Google AI Client: {e}", exc_info=True)
    # Model remains None

# --- Streaming function ---
def stream_llm_response(prompt: str) -> Generator[str, None, None] | None:
    """Streams response chunks."""
    if model is None:
        logger.error("LLM Client not initialized.")
        yield "[SYSTEM: LLM Client not initialized]"
        return
    try:
        logger.info(f"Streaming prompt to {MODEL_NAME} (length: {len(prompt)} chars)...")
        response_stream = model.generate_content(
            prompt, stream=True,
            generation_config=genai.types.GenerationConfig(temperature=0.7),
        )
        # Handle stream chunks and potential blocks
        for chunk in response_stream:
             if not chunk.parts:
                 if response_stream.prompt_feedback.block_reason:
                     block_reason = response_stream.prompt_feedback.block_reason
                     logger.warning(f"LLM stream potentially blocked during generation. Reason: {block_reason}")
                     yield f"[SYSTEM: Response may be blocked due to: {block_reason}]"
                 else:
                     # It's normal to get empty chunks sometimes, just continue
                     # logger.warning("LLM stream returned a chunk with no parts and no block reason.")
                     pass # Continue if just an empty chunk without block
                 continue # Skip empty/blocked chunk unless it's the final block reason
             if chunk.text: yield chunk.text # Yield valid text

        # Check for block reason after stream ends (if nothing was yielded)
        if response_stream.prompt_feedback.block_reason:
             # This check might be redundant if already yielded above, but safe fallback
             block_reason = response_stream.prompt_feedback.block_reason
             logger.warning(f"LLM stream blocked after generation. Reason: {block_reason}")
             # Check if anything was yielded before this block, if not, yield the reason
             # This requires tracking if anything was yielded, simplified here
             yield f"[SYSTEM: Response blocked due to: {block_reason}]"

    except Exception as e:
        logger.exception(f"Error streaming from Google AI API ({MODEL_NAME}): {e}", exc_info=True)
        yield f"[SYSTEM: Error during LLM communication: {e}]"

# --- Non-streaming function ---
def get_llm_response(prompt: str) -> str | None:
    """Gets the complete response text."""
    if model is None:
        logger.error("LLM Client not initialized.")
        return None
    try:
        logger.info(f"Sending prompt to {MODEL_NAME} (length: {len(prompt)} chars) for full response...")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.7),
        )
        # Handle response parts and potential blocks
        if response.parts:
            result_text = response.text
            logger.info(f"Received full response from {MODEL_NAME} (length: {len(result_text)} chars).")
            return result_text
        elif response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            logger.warning(f"LLM call blocked for prompt. Reason: {block_reason}")
            return f"[SYSTEM: Response blocked due to: {block_reason}]"
        else:
            # Check finish reason if no parts
            finish_reason = response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN"
            logger.warning(f"LLM returned no response parts. Finish Reason: {finish_reason}. Prompt feedback: {response.prompt_feedback}")
            return "[SYSTEM: LLM returned no content.]"
    except Exception as e:
        logger.exception(f"Error calling Google AI API ({MODEL_NAME}): {e}", exc_info=True)
        return None

# --- Query Transformation function ---
def transform_query_with_history(original_query: str, chat_history: List[Dict[str, str]]) -> str:
    """Uses the LLM to rewrite the query based on chat history."""
    if not chat_history:
        logger.info("No chat history provided, using original query for retrieval.")
        return original_query

    # Define prompt string directly
    history_str = ""
    for turn in chat_history:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "")
        content_preview = (content[:300] + '...') if len(content) > 300 else content
        history_str += f"{role}: {content_preview}\n"

    transform_prompt = f"""Given the following chat history and the latest user query, rewrite the latest user query to be a standalone question that incorporates the necessary context from the history. Only output the rewritten query, nothing else.

    **Chat History:**
    {history_str}
    **Latest User Query:** {original_query}

    **Standalone Query:**"""

    logger.info("Sending query transformation request to LLM...")
    if model is None:
         logger.error("LLM Client not initialized. Cannot transform query.")
         return original_query # Fallback

    try:
        # Use the non-streaming function for this short task
        # Use lower temperature for less creative rewriting
        response = model.generate_content(
             transform_prompt,
             generation_config=genai.types.GenerationConfig(temperature=0.1)
        )

        if response.parts and response.text:
            transformed_query = response.text.strip()
            # Add basic checks to ensure it's not just repeating instructions or empty
            if transformed_query and len(transformed_query) > 5 and transformed_query != original_query:
                logger.info(f"Transformed query: '{transformed_query}'")
                return transformed_query
            else:
                 logger.info("Query transformation resulted in original or invalid query.")
                 return original_query
        else:
            logger.warning(f"Query transformation failed or returned no text. Reason: {response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'}. Using original query.")
            return original_query # Fallback

    except Exception as e:
        logger.exception(f"Error during query transformation LLM call: {e}", exc_info=True)
        return original_query # Fallback to original query on error
