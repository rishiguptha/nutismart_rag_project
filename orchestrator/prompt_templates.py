# orchestrator/prompt_templates.py

import logging
import re
from typing import List, Dict
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# --- SIMPLIFIED ReAct-style Prompt ---
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

# --- MODIFIED Extraction Function ---
def extract_final_answer(llm_output: str) -> str:
    """
    Extracts the Answer part from the LLM's simplified ReAct-style output.
    Looks for the section starting with 'Answer:' after 'Reasoning:'.
    """
    # Find the start of the 'Answer:' section, making sure it comes *after* 'Reasoning:'
    reasoning_match = re.search(r"\*\*Reasoning\*\*:?", llm_output, re.IGNORECASE | re.DOTALL)
    if not reasoning_match:
        logger.warning("Could not find 'Reasoning:' section. Returning full output as fallback.")
        return llm_output # Can't find reasoning, return everything

    # Search for 'Answer:' *after* the reasoning section ends
    answer_match = re.search(r"\*\*Answer\*\*:?\s*(.*)", llm_output[reasoning_match.end():], re.IGNORECASE | re.DOTALL)

    if answer_match:
        final_answer = answer_match.group(1).strip()
        logger.debug(f"Extracted Answer: '{final_answer[:100]}...'")
        return final_answer
    else:
        # Fallback if 'Answer:' marker is missing after 'Reasoning:'
        logger.warning("Could not find 'Answer:' section after 'Reasoning:'. Returning content after 'Reasoning:' as fallback.")
        # Return everything after the reasoning section
        return llm_output[reasoning_match.end():].strip()

