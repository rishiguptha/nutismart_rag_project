# scripts/evaluate.py

import json
import os
import sys
import logging
import pprint
import time

# --- Add Project Root to sys.path ---
# Assumes evaluate.py is in scripts/, and config.py is in the parent directory (root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Addition ---

# --- Configuration (Using restored version without config.py dependency for now) ---
# Define paths relative to this script's location or project root
EVALUATION_SET_FILE = os.path.join(project_root, "evaluation_set.json")
RESULTS_FILE = os.path.join(project_root, "evaluation_results.json")
# Define K value directly for evaluation run
EVAL_TOP_K = 10
# Define the exact fallback phrase used in your prompt template
CANNOT_ANSWER_PHRASE = "Based on the provided documents, I cannot answer this question."

# --- Import RAG functions AFTER setting sys.path ---
try:
    # Import the function that runs the RAG pipeline for evaluation
    # Also import the readiness check
    from orchestrator.main_app import run_rag_for_evaluation, is_retriever_ready
except ImportError as e:
    print(f"FATAL ERROR in evaluate.py: Could not import 'orchestrator.main_app': {e}. Check imports and sys.path.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during orchestrator import: {e}")
    sys.exit(1)

# Setup logging specifically for evaluation
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Prevent duplicate handlers if run multiple times
if not logger.handlers:
    logger.setLevel(logging.INFO)
    # Log to console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    # Optional: Log to a file
    # try:
    #     log_file_path = os.path.join(project_root, "evaluation.log")
    #     file_handler = logging.FileHandler(log_file_path, mode='w') # Overwrite log each time
    #     file_handler.setFormatter(log_formatter)
    #     logger.addHandler(file_handler)
    # except Exception as log_e:
    #     logger.error(f"Could not configure file logging: {log_e}")


def load_evaluation_set(filepath):
    """Loads the evaluation questions from a JSON file."""
    abs_path = filepath
    logger.info(f"Loading evaluation set from: {abs_path}")
    if not os.path.exists(abs_path):
        logger.error(f"Evaluation set file not found: {abs_path}")
        return None
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        # Basic validation
        if not isinstance(eval_data, list):
             raise ValueError("Evaluation set should be a JSON list of objects.")
        valid_items = []
        for i, item in enumerate(eval_data):
            if not isinstance(item, dict) or "query" not in item or not item["query"]:
                 logger.warning(f"Skipping invalid/empty item at index {i} in evaluation set: {item}")
                 continue
            # Ensure ID exists, create one if missing
            if "id" not in item or not item["id"]:
                item["id"] = f"AutoGen_Q{i+1}"
                logger.debug(f"Generated ID '{item['id']}' for item at index {i}")
            valid_items.append(item)

        if not valid_items:
             logger.error("No valid evaluation items found in the file.")
             return None

        logger.info(f"Loaded {len(valid_items)} valid evaluation questions.")
        return valid_items
    except json.JSONDecodeError as json_err:
        logger.error(f"Error decoding JSON from evaluation set file {abs_path}: {json_err}")
        return None
    except ValueError as ve:
        logger.error(f"Invalid format in evaluation set file {abs_path}: {ve}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred loading evaluation set {abs_path}: {e}", exc_info=True)
        return None

def save_results(results, filepath):
    """Saves the evaluation results to a JSON file."""
    abs_path = filepath
    logger.info(f"Attempting to save evaluation results to: {abs_path}")
    try:
        # Ensure the directory exists
        results_dir = os.path.dirname(abs_path)
        os.makedirs(results_dir, exist_ok=True)

        with open(abs_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation results saved successfully ({len(results)} items) to {abs_path}.")
    except TypeError as te:
         logger.error(f"Error serializing results to JSON (likely non-serializable data): {te}. Results were: {results}")
    except Exception as e:
        logger.exception(f"Error saving evaluation results to {abs_path}: {e}", exc_info=True)

def analyze_results(results_filepath):
    """Loads results and calculates basic metrics."""
    logger.info(f"Analyzing results from: {results_filepath}")
    if not os.path.exists(results_filepath):
        logger.error(f"Results file not found for analysis: {results_filepath}")
        return None

    try:
        with open(results_filepath, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load or parse results file {results_filepath}: {e}")
        return None

    if not isinstance(results_data, list):
        logger.error("Results file does not contain a valid JSON list.")
        return None

    total_questions = len(results_data)
    errors = 0
    cannot_answer_count = 0
    successful_answers = 0

    for result in results_data:
        if result.get("error"):
            errors += 1
        elif result.get("generated_answer", "").strip() == CANNOT_ANSWER_PHRASE:
            cannot_answer_count += 1
        else:
            # Assume any non-error, non-fallback response is a successful attempt
            successful_answers += 1

    summary = {
        "total_questions": total_questions,
        "successful_answers": successful_answers,
        "cannot_answer_fallback": cannot_answer_count,
        "processing_errors": errors,
    }
    return summary


def main():
    logger.info("--- Starting RAG Evaluation Script ---")
    start_time = time.time()

    # 1. Check if retriever is ready (essential)
    logger.info("Checking retriever readiness...")
    if not is_retriever_ready():
        logger.critical("Retriever components are not ready. Cannot run evaluation. Check orchestrator/main_app.py logs and ensure index exists.")
        print("\nFATAL ERROR: Retriever failed to initialize. Please run indexing and check logs.")
        sys.exit(1)
    logger.info("Retriever components ready.")

    # 2. Load evaluation data
    eval_set = load_evaluation_set(EVALUATION_SET_FILE)
    if not eval_set:
        logger.error("Failed to load evaluation set. Exiting.")
        sys.exit(1)

    # 3. Run RAG for each question
    evaluation_results = []
    logger.info(f"Running RAG pipeline for {len(eval_set)} questions (using final_k={EVAL_TOP_K})...")
    for i, item in enumerate(eval_set):
        query_id = item.get("id")
        query = item.get("query")
        ideal_answer = item.get("ideal_answer", "N/A")

        print(f"\n--- Evaluating [{i+1}/{len(eval_set)}] ID: {query_id} ---")
        logger.info(f"Running RAG pipeline for query ID: {query_id} | Query: '{query[:100]}...'")

        item_start_time = time.time()
        # Call the evaluation function from main_app, using EVAL_TOP_K
        result_data = run_rag_for_evaluation(query, top_k=EVAL_TOP_K)
        item_duration = time.time() - item_start_time
        logger.info(f"Finished RAG pipeline for {query_id} in {item_duration:.2f} seconds.")

        # Store results along with ideal answer for comparison
        result_data["id"] = query_id
        result_data["ideal_answer"] = ideal_answer
        evaluation_results.append(result_data)

        # --- Print results for immediate manual review ---
        print(f"  Query: {result_data.get('query', 'N/A')}")
        # print(f"  Ideal Answer: {result_data.get('ideal_answer', 'N/A')}") # Keep concise for now
        generated_answer_preview = result_data.get('generated_answer', 'ERROR')[:250] + ('...' if len(result_data.get('generated_answer', '')) > 250 else '')
        print(f"  Generated Answer Preview: {generated_answer_preview}")
        # retrieved_count = len(result_data.get('retrieved_context', []))
        # print(f"  Retrieved Context ({retrieved_count} chunks)") # Keep concise for now

        if result_data.get("error"):
            logger.error(f"Error encountered for query ID {query_id}: {result_data['error']}")
            print(f"  ERROR during processing: {result_data['error']}")
        # --- End Print Results ---

    # 4. Save detailed results
    save_results(evaluation_results, RESULTS_FILE)

    # 5. Analyze results and print summary
    logger.info("--- Analyzing Evaluation Results ---")
    summary_stats = analyze_results(RESULTS_FILE)
    duration = time.time() - start_time

    print("\n" + "="*10 + " Evaluation Summary " + "="*10)
    if summary_stats:
        print(f"Total questions evaluated: {summary_stats['total_questions']}")
        print(f"Successfully generated answers: {summary_stats['successful_answers']}")
        print(f"Used 'Cannot Answer' fallback: {summary_stats['cannot_answer_fallback']}")
        print(f"Processing errors: {summary_stats['processing_errors']}")
        # Optional: Calculate percentages
        total = summary_stats['total_questions']
        if total > 0:
            success_rate = (summary_stats['successful_answers'] / total) * 100
            fallback_rate = (summary_stats['cannot_answer_fallback'] / total) * 100
            error_rate = (summary_stats['processing_errors'] / total) * 100
            print(f"\nSuccess Rate (Generated Answer): {success_rate:.1f}%")
            print(f"Fallback Rate ('Cannot Answer'): {fallback_rate:.1f}%")
            print(f"Error Rate: {error_rate:.1f}%")
    else:
        print("Could not generate summary statistics.")

    print(f"\nTotal evaluation duration: {duration:.2f} seconds")
    print(f"Detailed results saved to: {RESULTS_FILE}")
    print("="*42)
    print("\nReview the detailed results file for qualitative analysis.")
    print("--- Evaluation Script Finished ---")


if __name__ == "__main__":
    main()
