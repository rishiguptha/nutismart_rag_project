# orchestrator/llm_client.py

import os
import google.generativeai as genai
import logging
from dotenv import load_dotenv
from typing import Generator, Optional

# (Keep existing imports and setup code: load_dotenv, logger, API_KEY, MODEL_NAME, model initialization)
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

# --- Streaming function (Keep as is) ---
def stream_llm_response(prompt: str) -> Generator[str, None, None] | None:
    """
    Sends prompt to the configured Google AI model and streams the response chunks.
    Yields text chunks as they arrive. Returns None if initialization failed.
    """
    if model is None:
        logger.error("LLM Client not initialized. Cannot get response.")
        yield "[SYSTEM: LLM Client not initialized]"
        return

    try:
        logger.info(f"Streaming prompt to {MODEL_NAME} (length: {len(prompt)} chars)...")
        response_stream = model.generate_content(
            prompt,
            stream=True,
            generation_config=genai.types.GenerationConfig(temperature=0.7),
        )
        # (Keep the rest of the streaming logic from the previous version)
        for chunk in response_stream:
            if not chunk.parts:
                 if response_stream.prompt_feedback.block_reason:
                     block_reason = response_stream.prompt_feedback.block_reason
                     logger.warning(f"LLM stream potentially blocked during generation. Reason: {block_reason}")
                     yield f"[SYSTEM: Response may be blocked due to: {block_reason}]"
                 else:
                     logger.warning("LLM stream returned a chunk with no parts and no block reason.")
                 continue

            if chunk.text:
                yield chunk.text

        if response_stream.prompt_feedback.block_reason:
             block_reason = response_stream.prompt_feedback.block_reason
             logger.warning(f"LLM stream blocked after generation. Reason: {block_reason}")
             yield f"[SYSTEM: Response blocked due to: {block_reason}]"

    except Exception as e:
        logger.exception(f"Error streaming from Google AI API ({MODEL_NAME}): {e}", exc_info=True)
        yield f"[SYSTEM: Error during LLM communication: {e}]"

# --- ADD THIS NON-STREAMING FUNCTION ---
def get_llm_response(prompt: str) -> str | None:
    """
    Sends prompt to the configured Google AI model and returns the complete response text.
    Returns None if initialization failed or an error occurred.
    """
    if model is None:
        logger.error("LLM Client not initialized. Cannot get response.")
        return None
    try:
        logger.info(f"Sending prompt to {MODEL_NAME} (length: {len(prompt)} chars) for full response...")
        # Use stream=False (or omit it, as False is default)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.7),
            # safety_settings=[...] # Optional
        )

        # Enhanced error/response handling (same as before)
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
        return None # Indicate failure
