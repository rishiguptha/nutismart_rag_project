# orchestrator/prompt_templates.py

import logging
import re
from typing import List, Dict # Import Dict
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# --- MODIFIED TO ACCEPT AND FORMAT CHAT HISTORY ---
def format_react_style_rag_prompt(
    query: str,
    context_docs: List[Document],
    chat_history: List[Dict[str, str]] = None # Add optional chat_history argument
    ) -> str:
    """
    Formats a ReAct-style prompt for the LLM with retrieved context,
    chat history, and citation instructions.
    """

    # --- Format Chat History ---
    history_str = ""
    if chat_history:
        history_str += "**Chat History (Recent Turns):**\n"
        for turn in chat_history:
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            history_str += f"{role}: {content}\n"
        history_str += "\n---\n\n" # Separator after history

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

    # --- Assemble Prompt ---
    # Include history before the main instruction/context
    prompt = f"""{history_str}**Instruction:** You are an AI assistant specialized in Nutrition and Fitness. Your task is to answer the user's latest query accurately based *only* on the provided context chunks and considering the preceding chat history for context. Follow these steps carefully:

1.  **Thought:** First, analyze the user's latest query (`{query}`) in the context of the chat history. Read through the provided context chunks, paying attention to the source file and page number. Identify and explicitly list the key sentences or passages from the context chunks that are directly relevant to answering the *latest query*, considering the conversation flow. Analyze how these relevant passages connect. Determine if the combined information is sufficient. *Do not add any external knowledge or assumptions.* If the context does not contain the necessary information, state that clearly in this thought process.

2.  **Final Answer:** Based *only* on the analysis in your Thought step, construct a comprehensive and explanatory answer to the *latest query*. **Crucially, for each piece of information you use from the context, cite the source document and page number in parentheses immediately after the information, like this: (Source: filename.pdf, Page X).** If your thought process concluded that the information is not available in the context, the final answer should *only* be: "Based on the provided documents, I cannot answer this question."

**Example Answer Style Guidance (Illustrates citation format, do NOT use content from examples):**
* (Examples remain the same as before)

---
**BEGIN TASK**

**Provided Context:**
---
{context_str}
---

**Latest User Query:** {query}

---
**Output:**

**Thought:**
""" # <<< LLM STARTS GENERATING THOUGHT PROCESS HERE

    log_query = query[:50].replace('\n', ' ')
    logger.info(f"Formatted ReAct-style prompt using {len(context_docs)} context documents and {len(chat_history or [])} history turns for query: '{log_query}...'")
    return prompt

# --- UNCHANGED ---
def extract_final_answer(llm_output: str) -> str:
    """Extracts the Final Answer part from the LLM's ReAct-style output."""
    match = re.search(r"\*\*Final Answer\*\*:?\s*(.*)", llm_output, re.IGNORECASE | re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
        # Remove potential leading "Thought:" section if extraction failed partially
        thought_prefix = "**Thought:**"
        if final_answer.lower().startswith(thought_prefix.lower()):
             final_answer = final_answer[len(thought_prefix):].strip()
        logger.debug(f"Extracted Final Answer: '{final_answer[:100]}...'")
        return final_answer
    else:
        logger.warning("Could not find 'Final Answer:' section in LLM output. Attempting fallback extraction.")
        # Fallback: Try to find the start of "Thought:" and return everything after it,
        # assuming the thought process might be shorter or missing the final marker.
        # This is less reliable.
        thought_match = re.search(r"\*\*Thought\*\*:?\s*(.*)", llm_output, re.IGNORECASE | re.DOTALL)
        if thought_match:
             # Find where the thought *ends* might be tricky. Let's assume the answer follows immediately.
             # This might include the thought process if the LLM didn't separate well.
             potential_answer = llm_output[thought_match.end():].strip()
             if potential_answer:
                 logger.warning("Returning content after 'Thought:' as fallback.")
                 return potential_answer
        logger.error("Fallback failed. Returning full LLM output.")
        return llm_output # Return everything if extraction fails completely
