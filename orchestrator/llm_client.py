import os
import google.generativeai as genai
import logging
from dotenv import load_dotenv

# Load environment variables from .env file in the project root
# Assumes this script is run relative to the project root where .env lives
# or python -m orchestrator... handles pathing correctly.
load_dotenv()

logger = logging.getLogger(__name__)
# Basic config, gets refined in main_app if run via -m
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

API_KEY = os.environ.get("GOOGLE_API_KEY")
model = None
# Consider making MODEL_NAME configurable via .env too
MODEL_NAME = os.environ.get("GENERATIVE_MODEL_NAME", "gemini-2.0-flash")

# --- Configure the Google AI Client ---
try:
    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Did you create a .env file?")
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    logger.info(f"Google AI Client configured successfully for model: {MODEL_NAME}")
except Exception as e:
    logger.exception(f"Failed to configure Google AI Client: {e}", exc_info=True)
    # Model remains None

def get_llm_response(prompt: str) -> str | None:
    """Sends prompt to the configured Google AI model and returns the response text."""
    if model is None:
        logger.error("LLM Client not initialized. Cannot get response.")
        return None
    try:
        logger.info(f"Sending prompt to {MODEL_NAME} (length: {len(prompt)} chars)...")
        # Reference: https://ai.google.dev/api/python/google/generativeai/GenerativeModel#generate_content
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                # candidate_count=1, # Default is 1
                # stop_sequences=['\n\n\n'],
                # max_output_tokens=1024, # Good practice to set a limit
                temperature=0.7 # Adjust for creativity vs factualness
             ),
            # Optional: Add safety_settings if needed
            # safety_settings=[...]
         )

        # Enhanced error/response handling
        if response.parts:
            result_text = response.text
            logger.info(f"Received response from {MODEL_NAME} (length: {len(result_text)} chars).")
            return result_text
        elif response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            logger.warning(f"LLM call blocked for prompt. Reason: {block_reason}")
            # Consider how to represent this to the user
            return f"[SYSTEM: Response blocked due to: {block_reason}]"
        else:
            # This case might occur if generation finishes abruptly or yields no candidates
            logger.warning(f"LLM returned no response parts. Prompt feedback: {response.prompt_feedback}")
            return "[SYSTEM: LLM returned no content.]"

    except Exception as e:
        # Catch potential API errors, network issues etc.
        logger.exception(f"Error calling Google AI API ({MODEL_NAME}): {e}", exc_info=True)
        return None # Indicate failure