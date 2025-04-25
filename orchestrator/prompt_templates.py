# orchestrator/prompt_templates.py

import logging
import re
from typing import List, Dict
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# --- format_react_style_rag_prompt function remains the same ---
def format_react_style_rag_prompt(
    query: str,
    context_docs: List[Document],
    chat_history: List[Dict[str, str]] = None
    ) -> str:
    """
    Formats a simpler ReAct-style prompt for the LLM with retrieved context,
    chat history, and citation instructions. Asks for reasoning first, then the answer.
    """
    # --- Format Chat History ---
    history_str = ""
    if chat_history:
        history_str += "**Chat History (Recent Turns):**\n"
        for turn in chat_history:
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            history_str += f"{role}: {content}\n"
        history_str += "\n---\n\n"

    # --- Format Context ---
    context_str = ""
    if not context_docs:
        logger.warning("No context documents provided for prompt formatting.")
        context_str = "No relevant context found in the documents."
    else:
        context_pieces = []
        for i, doc in enumerate(context_docs):
            source = doc.metadata.get('source', 'Unknown Source')
            filename = source.split('/')[-1].split('\\')[-1]
            page = doc.metadata.get('page', 'N/A')
            header = f"Context Chunk {i+1} (Source: {filename}, Page: {page}):"
            context_pieces.append(f"{header}\n{doc.page_content}")
        context_str = "\n\n---\n\n".join(context_pieces)

    # --- Assemble Simplified ReAct Prompt ---
    prompt = f"""{history_str}**Instruction:** You are an AI assistant specialized in Nutrition and Fitness. Answer the user's latest query based *only* on the provided context chunks and considering the chat history.

1.  **Reasoning:** First, explain your reasoning step-by-step. Analyze the user's query (`{query}`) and the chat history. Read the context chunks. Identify relevant passages and explain how they connect to answer the query. State if the context is insufficient. *Do not add external knowledge.*

2.  **Answer:** After your reasoning, provide the final answer based *only* on the relevant information identified in your reasoning. For each piece of information from the context, cite the source document and page number like this: (Source: filename.pdf, Page X). If the context is insufficient, just state: "Based on the provided documents, I cannot answer this question."

**Example Answer Style Guidance (Illustrates citation format, do NOT use content from examples):**
* (Examples remain the same)

---
**BEGIN TASK**

**Provided Context:**
---
{context_str}
---

**Latest User Query:** {query}

---
**Output:**

**Reasoning:**
""" # <<< LLM STARTS GENERATING REASONING HERE

    log_query = query[:50].replace('\n', ' ')
    logger.info(f"Formatted Simplified ReAct prompt using {len(context_docs)} context docs and {len(chat_history or [])} history turns for query: '{log_query}...'")
    return prompt


# --- EVEN MORE ROBUST Extraction Function ---
def extract_final_answer(llm_output: str) -> str:
    """
    Extracts the Answer part from the LLM's simplified ReAct-style output.
    Prioritizes finding '**Answer:**' after '**Reasoning:**'.
    Includes fallbacks for missing markers or variations.
    """
    if not llm_output:
        logger.warning("Received empty LLM output for extraction.")
        return ""

    # Normalize potential markdown variations and whitespace around markers
    # Look for variations like **Answer:**, **Answer**: , **Answer** : etc.
    answer_marker_pattern = r"\*\*[Aa]nswer\*\*\s*:?\s*" # Case-insensitive, optional colon, optional space
    reasoning_marker_pattern = r"\*\*[Rr]easoning\*\*\s*:?\s*" # Case-insensitive, optional colon, optional space

    # Try to find the reasoning marker first
    reasoning_match = re.search(reasoning_marker_pattern, llm_output)
    search_start_index = 0
    if reasoning_match:
        search_start_index = reasoning_match.end()
        logger.debug("Found 'Reasoning:' marker.")
    else:
        logger.warning("Could not find 'Reasoning:' marker in LLM output.")
        # If reasoning is missing, we search for Answer from the beginning

    # Now search for the Answer marker *after* the reasoning section (or from start)
    answer_match = re.search(answer_marker_pattern, llm_output[search_start_index:])

    if answer_match:
        # Calculate the actual start index in the original string and get the rest
        answer_start_index = search_start_index + answer_match.end()
        final_answer = llm_output[answer_start_index:].strip()
        logger.info(f"Extracted Answer using marker (length: {len(final_answer)}).")
        # Simple check to remove trailing Reasoning/Thought if LLM repeats it
        if final_answer.endswith("**Reasoning:**") or final_answer.endswith("**Thought:**"):
             final_answer = final_answer[:-len("**Reasoning:**")].strip()
        return final_answer
    else:
        # Fallback 1: If 'Answer:' marker is missing, but 'Reasoning:' was found,
        # assume the answer is everything after 'Reasoning:'.
        if reasoning_match:
            logger.warning("Could not find 'Answer:' marker after 'Reasoning:'. Returning content after 'Reasoning:' as fallback.")
            potential_answer = llm_output[search_start_index:].strip()
            if potential_answer:
                return potential_answer
            else:
                 logger.warning("Content after 'Reasoning:' is empty.")
                 # If reasoning was found but nothing follows, maybe the answer was just "I cannot answer" within reasoning?
                 # Let's return the whole output in this ambiguous case for review.
                 return llm_output

        # Fallback 2: If neither marker was found clearly, return the whole output.
        logger.error("Could not reliably extract final answer using markers. Returning full LLM output.")
        return llm_output
