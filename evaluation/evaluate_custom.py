import json
import os
import sys
import logging
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv
import re

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import RAG functions
from orchestrator.main_app import run_rag_for_evaluation, is_retriever_ready

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_gemini():
    """Setup Gemini API."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in .env file")
        return None
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')

def extract_json_from_code_block(text):
    # This will match ```json ... ```
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    # Fallback: try to match any code block
    match = re.search(r"```(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text  # If no code block, return as is

def evaluate_answer(model, question: str, answer: str, context: List[str], ground_truth: str) -> Dict:
    """Evaluate a single answer using Gemini."""
    evaluation_prompt = f"""
You are an expert evaluator of RAG (Retrieval-Augmented Generation) systems. Your task is to evaluate the response based on specific criteria and return ONLY a JSON object.

IMPORTANT: You must return ONLY the JSON object, with no additional text or explanation before or after it.

Evaluation Criteria:
1. Faithfulness (0-1): Is the answer supported by the provided context?
2. Answer Relevancy (0-1): Is the answer relevant to the question?
3. Context Relevancy (0-1): Is the retrieved context relevant to the question?
4. Completeness (0-1): Does the answer cover all important aspects of the question?

Question: {question}

Retrieved Context:
{chr(10).join([f'[{i+1}] {ctx}' for i, ctx in enumerate(context)])}

Generated Answer: {answer}

Ground Truth: {ground_truth}

Return ONLY this JSON object (no other text):
{{
    "faithfulness": {{
        "score": <number between 0 and 1>,
        "explanation": "<brief explanation>"
    }},
    "answer_relevancy": {{
        "score": <number between 0 and 1>,
        "explanation": "<brief explanation>"
    }},
    "context_relevancy": {{
        "score": <number between 0 and 1>,
        "explanation": "<brief explanation>"
    }},
    "completeness": {{
        "score": <number between 0 and 1>,
        "explanation": "<brief explanation>"
    }}
}}
"""
    try:
        response = model.generate_content(evaluation_prompt)
        response_text = response.text.strip()
        
        # Log the raw response for debugging
        logger.debug(f"Raw response from Gemini: {response_text}")
        
        # Remove Markdown code block if present
        cleaned_response = extract_json_from_code_block(response_text)
        
        # Try to parse the JSON response
        try:
            evaluation = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response was: {cleaned_response}")
            return {}
        
        # Validate the evaluation structure
        required_keys = ["faithfulness", "answer_relevancy", "context_relevancy", "completeness"]
        for key in required_keys:
            if key not in evaluation:
                logger.error(f"Missing required key in evaluation: {key}")
                return {}
            if "score" not in evaluation[key] or "explanation" not in evaluation[key]:
                logger.error(f"Missing score or explanation in {key}")
                return {}
            if not isinstance(evaluation[key]["score"], (int, float)):
                logger.error(f"Score for {key} is not a number")
                return {}
            if not isinstance(evaluation[key]["explanation"], str):
                logger.error(f"Explanation for {key} is not a string")
                return {}
        
        return evaluation
    except Exception as e:
        logger.error(f"Error evaluating answer: {e}")
        return {}

def main():
    logger.info("Starting Custom Evaluation with Gemini")
    
    # Setup Gemini
    model = setup_gemini()
    if not model:
        sys.exit(1)
    
    # Check if retriever is ready
    if not is_retriever_ready():
        logger.error("Retriever not ready. Please ensure the index is built.")
        sys.exit(1)
    
    # Load evaluation set
    eval_set_path = os.path.join(project_root, "evaluation/eval_set/evaluation_set.json")
    try:
        with open(eval_set_path, 'r') as f:
            eval_set = json.load(f)
    except Exception as e:
        logger.error(f"Error loading evaluation set: {e}")
        sys.exit(1)
    
    # Run RAG pipeline and evaluate each question
    evaluation_results = []
    logger.info(f"Running RAG pipeline for {len(eval_set)} questions...")
    
    for i, item in enumerate(eval_set):
        query = item.get("query")
        ideal_answer = item.get("ideal_answer", "")
        
        logger.info(f"Processing question {i+1}/{len(eval_set)}: {query[:100]}...")
        
        # Get RAG response
        result = run_rag_for_evaluation(query)
        result["ideal_answer"] = ideal_answer
        
        # Evaluate the response
        evaluation = evaluate_answer(
            model,
            query,
            result["generated_answer"],
            result["retrieved_context"],
            ideal_answer
        )
        
        result["evaluation"] = evaluation
        evaluation_results.append(result)
        
        # Print progress
        if evaluation:
            print(f"\nQuestion {i+1} Evaluation:")
            print("-" * 50)
            for metric, details in evaluation.items():
                print(f"{metric}: {details['score']:.2f} - {details['explanation']}")
            print("-" * 50)
    
    # Calculate average scores
    avg_scores = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_relevancy": 0.0,
        "completeness": 0.0
    }
    
    valid_evaluations = [r["evaluation"] for r in evaluation_results if r["evaluation"]]
    if valid_evaluations:
        for metric in avg_scores.keys():
            scores = [e[metric]["score"] for e in valid_evaluations]
            avg_scores[metric] = sum(scores) / len(scores)
    
    # Save detailed results
    results_path = os.path.join(project_root, "evaluation/eval_results/custom_evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "detailed_results": evaluation_results,
            "average_scores": avg_scores
        }, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_path}")
    
    # Print summary
    print("\nOverall Evaluation Results:")
    print("-" * 50)
    for metric, score in avg_scores.items():
        print(f"Average {metric}: {score:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    main() 