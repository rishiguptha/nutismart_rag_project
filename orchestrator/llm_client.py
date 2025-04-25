# orchestrator/llm_client.py

import os
import google.generativeai as genai
import logging
from dotenv import load_dotenv
# --- ADD List and Dict to the import ---
from typing import Generator, Optional, List, Dict

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

API_KEY = os.environ.get("GOOGLE_API_KEY")
model = None
MODEL_NAME = os.environ.get("GENERATIVE_MODEL_NAME", "gemini-1.5-flash-latest")

# --- Configure the Google AI Client ---
try:
    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    logger.info(f"Google AI Client configured successfully for model: {MODEL_NAME}")
except Exception as e:
    logger.exception(f"Failed to configure Google AI Client: {e}", exc_info=True)
    # Model remains None

# --- Streaming function (Unchanged) ---
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
        for chunk in response_stream:
             if not chunk.parts:
                 if response_stream.prompt_feedback.block_reason:
                     block_reason = response_stream.prompt_feedback.block_reason
                     logger.warning(f"LLM stream potentially blocked during generation. Reason: {block_reason}")
                     yield f"[SYSTEM: Response may be blocked due to: {block_reason}]"
                 else:
                     logger.warning("LLM stream returned a chunk with no parts and no block reason.")
                 continue
             if chunk.text: yield chunk.text
        if response_stream.prompt_feedback.block_reason:
             block_reason = response_stream.prompt_feedback.block_reason
             logger.warning(f"LLM stream blocked after generation. Reason: {block_reason}")
             yield f"[SYSTEM: Response blocked due to: {block_reason}]"
    except Exception as e:
        logger.exception(f"Error streaming from Google AI API ({MODEL_NAME}): {e}", exc_info=True)
        yield f"[SYSTEM: Error during LLM communication: {e}]"

# --- Non-streaming function (Unchanged) ---
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
        if response.parts:
            result_text = response.text
            logger.info(f"Received full response from {MODEL_NAME} (length: {len(result_text)} chars).")
            return result_text
        elif response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            logger.warning(f"LLM call blocked for prompt. Reason: {block_reason}")
            return f"[SYSTEM: Response blocked due to: {block_reason}]"
        else:
            logger.warning(f"LLM returned no response parts. Prompt feedback: {response.prompt_feedback}")
            return "[SYSTEM: LLM returned no content.]"
    except Exception as e:
        logger.exception(f"Error calling Google AI API ({MODEL_NAME}): {e}", exc_info=True)
        return None

# --- Query Transformation function (Unchanged logic, just needs the import fixed) ---
def transform_query_with_history(original_query: str, chat_history: List[Dict[str, str]]) -> str:
    """Uses the LLM to rewrite the query based on chat history."""
    if not chat_history:
        logger.info("No chat history provided, using original query for retrieval.")
        return original_query

    # Import the formatter here to avoid circular dependency if placed at top level
    # Ensure prompt_templates.py also imports List, Dict if needed
    from orchestrator.prompt_templates import format_query_transform_prompt

    transform_prompt = format_query_transform_prompt(original_query, chat_history)

    logger.info("Sending query transformation request to LLM...")
    if model is None:
         logger.error("LLM Client not initialized. Cannot transform query.")
         return original_query # Fallback

    try:
        response = model.generate_content(
             transform_prompt,
             generation_config=genai.types.GenerationConfig(temperature=0.1) # Low temp
        )

        if response.parts and response.text:
            transformed_query = response.text.strip()
            if transformed_query and transformed_query != original_query:
                logger.info(f"Transformed query: '{transformed_query}'")
                return transformed_query
            elif transformed_query == original_query:
                 logger.info("Query transformation resulted in the original query.")
                 return original_query
            else:
                 logger.warning("Query transformation returned empty result. Using original query.")
                 return original_query
        else:
            logger.warning(f"Query transformation failed or returned no text. Reason: {response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'}. Using original query.")
            return original_query # Fallback

    except Exception as e:
        logger.exception(f"Error during query transformation LLM call: {e}", exc_info=True)
        return original_query # Fallback to original query on error
