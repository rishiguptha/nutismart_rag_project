import os
import google.generativeai as genai
import logging
from dotenv import load_dotenv
from typing import Generator, Optional # Import Generator

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

API_KEY = os.environ.get("GOOGLE_API_KEY")
model = None
MODEL_NAME = os.environ.get("GENERATIVE_MODEL_NAME", "gemini-1.5-flash-latest") # Ensure this model supports streaming

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

# --- Updated function to STREAM responses ---
def stream_llm_response(prompt: str) -> Generator[str, None, None] | None:
    """
    Sends prompt to the configured Google AI model and streams the response chunks.
    Yields text chunks as they arrive. Returns None if initialization failed.
    """
    if model is None:
        logger.error("LLM Client not initialized. Cannot get response.")
        yield "[SYSTEM: LLM Client not initialized]" # Yield an error message
        return # Use return instead of yielding None directly from generator

    try:
        logger.info(f"Streaming prompt to {MODEL_NAME} (length: {len(prompt)} chars)...")
        # Use stream=True
        response_stream = model.generate_content(
            prompt,
            stream=True, # Enable streaming
            generation_config=genai.types.GenerationConfig(
                temperature=0.7
                # max_output_tokens=1024 # Max tokens can still be useful
            ),
            # safety_settings=[...] # Optional
        )

        # Iterate through the stream and yield text parts
        for chunk in response_stream:
            # Check for blocked content in the stream
            if not chunk.parts:
                 # Check if the block occurred *during* generation
                 if response_stream.prompt_feedback.block_reason:
                     block_reason = response_stream.prompt_feedback.block_reason
                     logger.warning(f"LLM stream potentially blocked during generation. Reason: {block_reason}")
                     yield f"[SYSTEM: Response may be blocked due to: {block_reason}]"
                 else:
                     logger.warning("LLM stream returned a chunk with no parts and no block reason.")
                 # Continue to next chunk or break if needed, depends on desired behavior
                 continue # Skip empty chunks unless blocked

            # Yield the text content of the chunk
            if chunk.text:
                # logger.debug(f"Yielding chunk: {chunk.text}") # Very verbose
                yield chunk.text

        # Check for blocking *after* generation is complete (if nothing was yielded)
        # This handles cases where the *entire* response is blocked from the start
        if response_stream.prompt_feedback.block_reason:
             block_reason = response_stream.prompt_feedback.block_reason
             logger.warning(f"LLM stream blocked after generation. Reason: {block_reason}")
             # Check if anything was yielded before this block
             # If nothing yielded, send the block message
             # This logic might need refinement based on API behavior
             yield f"[SYSTEM: Response blocked due to: {block_reason}]"


    except Exception as e:
        logger.exception(f"Error streaming from Google AI API ({MODEL_NAME}): {e}", exc_info=True)
        yield f"[SYSTEM: Error during LLM communication: {e}]" # Yield error message

# Keep the non-streaming version for potential internal use or testing? Or remove it


# import os
# import google.generativeai as genai
# import logging
# from dotenv import load_dotenv

# # Load environment variables from .env file in the project root
# # Assumes this script is run relative to the project root where .env lives
# # or python -m orchestrator... handles pathing correctly.
# load_dotenv()

# logger = logging.getLogger(__name__)
# # Basic config, gets refined in main_app if run via -m
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# API_KEY = os.environ.get("GOOGLE_API_KEY")
# model = None
# # Consider making MODEL_NAME configurable via .env too
# MODEL_NAME = os.environ.get("GENERATIVE_MODEL_NAME", "gemini-2.0-flash")

# # --- Configure the Google AI Client ---
# try:
#     if not API_KEY:
#         raise ValueError("GOOGLE_API_KEY not found in environment variables. Did you create a .env file?")
#     genai.configure(api_key=API_KEY)
#     model = genai.GenerativeModel(MODEL_NAME)
#     logger.info(f"Google AI Client configured successfully for model: {MODEL_NAME}")
# except Exception as e:
#     logger.exception(f"Failed to configure Google AI Client: {e}", exc_info=True)
#     # Model remains None

# def get_llm_response(prompt: str) -> str | None:
#     """Sends prompt to the configured Google AI model and returns the response text."""
#     if model is None:
#         logger.error("LLM Client not initialized. Cannot get response.")
#         return None
#     try:
#         logger.info(f"Sending prompt to {MODEL_NAME} (length: {len(prompt)} chars)...")
#         # Reference: https://ai.google.dev/api/python/google/generativeai/GenerativeModel#generate_content
#         response = model.generate_content(
#             prompt,
#             generation_config=genai.types.GenerationConfig(
#                 # candidate_count=1, # Default is 1
#                 # stop_sequences=['\n\n\n'],
#                 # max_output_tokens=1024, # Good practice to set a limit
#                 temperature=0.7 # Adjust for creativity vs factualness
#              ),
#             # Optional: Add safety_settings if needed
#             # safety_settings=[...]
#          )

#         # Enhanced error/response handling
#         if response.parts:
#             result_text = response.text
#             logger.info(f"Received response from {MODEL_NAME} (length: {len(result_text)} chars).")
#             return result_text
#         elif response.prompt_feedback.block_reason:
#             block_reason = response.prompt_feedback.block_reason
#             logger.warning(f"LLM call blocked for prompt. Reason: {block_reason}")
#             # Consider how to represent this to the user
#             return f"[SYSTEM: Response blocked due to: {block_reason}]"
#         else:
#             # This case might occur if generation finishes abruptly or yields no candidates
#             logger.warning(f"LLM returned no response parts. Prompt feedback: {response.prompt_feedback}")
#             return "[SYSTEM: LLM returned no content.]"

#     except Exception as e:
#         # Catch potential API errors, network issues etc.
#         logger.exception(f"Error calling Google AI API ({MODEL_NAME}): {e}", exc_info=True)
#         return None # Indicate failure

