# scripts/evaluate.py

import json
import os
import sys
import logging
import pprint # For pretty printing context
import time # To measure duration

# --- Add project root to path to allow importing orchestrator ---
# Assumes evaluate.py is in the scripts/ directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# --- End Path Addition ---

try:
    # Import the function that runs the RAG pipeline for evaluation
    # Also import the readiness check
    from orchestrator.main_app import run_rag_for_evaluation, is_retriever_ready
except ImportError as e:
    print(f"Error importing orchestrator: {e}. Ensure evaluate.py is run correctly relative to the project structure.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

# Setup logging specifically for evaluation
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Prevent duplicate handlers if run multiple times in same session
if not logger.handlers:
    # Log to console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    # Optional: Log to a file
    # file_handler = logging.FileHandler("evaluation.log", mode='w') # Overwrite log each time
    # file_handler.setFormatter(log_formatter)
    # logger.addHandler(file_handler)


# --- Configuration ---
EVALUATION_SET_FILE = "evaluation_set.json" # Path relative to project root
RESULTS_FILE = "evaluation_results.json" # Where to save detailed results
EVAL_TOP_K = 10 # How many documents to retrieve during evaluation (match main app or set differently)

def load_evaluation_set(filepath):
    """Loads the evaluation questions from a JSON file."""
    abs_path = os.path.join(project_root, filepath) # Construct absolute path
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        # Basic validation
        if not isinstance(eval_data, list):
             raise ValueError("Evaluation set should be a JSON list of objects.")
        for item in eval_data:
            if not isinstance(item, dict) or "query" not in item:
                 raise ValueError("Each item in evaluation set must be an object with at least a 'query' key.")
        return eval_data
    except FileNotFoundError:
        logger.error(f"Evaluation set file not found: {abs_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from evaluation set file: {abs_path}")
        return None
    except ValueError as ve:
        logger.error(f"Invalid format in evaluation set file {abs_path}: {ve}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred loading evaluation set: {e}", exc_info=True)
        return None

def save_results(results, filepath):
    """Saves the evaluation results to a JSON file."""
    abs_path = os.path.join(project_root, filepath) # Construct absolute path
    try:
        with open(abs_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2) # Use indent=2 for readability
        logger.info(f"Evaluation results saved to: {abs_path}")
    except Exception as e:
        logger.exception(f"Error saving evaluation results to {abs_path}: {e}", exc_info=True)

def main():
    logger.info("--- Starting Basic RAG Evaluation ---")
    start_time = time.time()

    # 1. Check if retriever is ready
    if not is_retriever_ready():
        logger.error("Retriever components are not ready. Cannot run evaluation.")
        print("\nFATAL ERROR: Retriever failed to initialize. Check orchestrator/main_app.py logs.")
        sys.exit(1)
    logger.info("Retriever components ready.")

    # 2. Load evaluation data
    logger.info(f"Loading evaluation set from: {EVALUATION_SET_FILE}")
    eval_set = load_evaluation_set(EVALUATION_SET_FILE)
    if not eval_set:
        sys.exit(1)
    logger.info(f"Loaded {len(eval_set)} evaluation questions.")

    # 3. Run RAG for each question
    evaluation_results = []
    for i, item in enumerate(eval_set):
        query_id = item.get("id", f"Q{i+1}")
        query = item.get("query")
        ideal_answer = item.get("ideal_answer", "N/A") # Get ideal answer for comparison

        if not query:
            logger.warning(f"Skipping item {query_id} due to missing query.")
            continue

        print(f"\n--- Evaluating {query_id}: '{query}' ---")
        logger.info(f"Running RAG pipeline for query ID: {query_id}")

        # Call the evaluation function from main_app
        # Ensure run_rag_for_evaluation exists and handles errors, returning a dict
        result_data = run_rag_for_evaluation(query, top_k=EVAL_TOP_K)

        # Store results along with ideal answer for comparison
        result_data["id"] = query_id
        result_data["ideal_answer"] = ideal_answer
        evaluation_results.append(result_data)

        # Print results for immediate manual review
        print(f"  Query: {result_data.get('query', 'N/A')}")
        print(f"  Ideal Answer: {result_data.get('ideal_answer', 'N/A')}")
        print(f"  Generated Answer: {result_data.get('generated_answer', 'ERROR')}")
        print(f"  Retrieved Context ({len(result_data.get('retrieved_context', []))} chunks):")
        # Use pprint for better readability of context list and metadata
        retrieved_context = result_data.get('retrieved_context', [])
        retrieved_metadata = result_data.get('retrieved_metadata', [])
        for idx, context_chunk in enumerate(retrieved_context):
             print(f"    --- Context Chunk {idx+1} ---")
             metadata = retrieved_metadata[idx] if idx < len(retrieved_metadata) else {}
             print(f"    Metadata:")
             pprint.pprint(metadata, indent=6, width=100)
             print(f"    Content: {context_chunk[:300]}...") # Show a bit more context
        print("-" * 20)

        # --- Placeholder for Automated Metrics ---
        # TODO: Add calls to RAGAS, LangChain Eval, or LLM-as-judge here
        # --- End Placeholder ---


    # 4. Save detailed results
    save_results(evaluation_results, RESULTS_FILE)

    # 5. Calculate and print summary statistics
    end_time = time.time()
    total_questions = len(evaluation_results)
    errors = sum(1 for r in evaluation_results if r.get("error"))
    duration = end_time - start_time

    print("\n--- Evaluation Summary ---")
    print(f"Total questions evaluated: {total_questions}")
    print(f"Number of errors encountered during processing: {errors}")
    print(f"Total evaluation duration: {duration:.2f} seconds")
    print(f"Detailed results saved to: {RESULTS_FILE}")
    print("\nPlease manually review the generated answers and context against the ideal answers in the results file.")
    print("Consider metrics like Context Relevance, Answer Faithfulness, and Answer Relevance.")
    print("--- Evaluation Finished ---")


if __name__ == "__main__":
    main()
