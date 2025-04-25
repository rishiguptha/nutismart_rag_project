# orchestrator/prompt_templates.py

import logging
from typing import List, Dict
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# --- FINAL SIMPLIFIED PROMPT FUNCTION ---
def format_rag_prompt(
    query: str,
    context_docs: List[Document],
    chat_history: List[Dict[str, str]] = None
    ) -> str:
    """
    Formats a direct prompt for the LLM with retrieved context,
    chat history, and citation instructions. No explicit reasoning step requested.
    """

    # --- Format Chat History ---
    history_str = ""
    if chat_history:
        # Keep only the last few turns for brevity
        history_str += "**Chat History (for context):**\n"
        for turn in chat_history: # Assumes history is already trimmed if needed
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            # Basic truncation for long history items if needed
            content_preview = (content[:150] + '...') if len(content) > 150 else content
            history_str += f"{role}: {content_preview}\n"
        history_str += "\n---\n\n"

    # --- Format Context ---
    context_str = ""
    if not context_docs:
        logger.warning("No context documents provided for prompt formatting.")
        # Provide a clear signal to the LLM that context is missing
        context_str = "CONTEXT_IS_MISSING"
    else:
        # Format context clearly, including source and page number
        context_pieces = []
        for i, doc in enumerate(context_docs):
            source = doc.metadata.get('source', 'Unknown Source')
            filename = source.split('/')[-1].split('\\')[-1] # Handle both path separators
            page = doc.metadata.get('page', 'N/A')
            header = f"Context Chunk {i+1} (Source: {filename}, Page: {page}):"
            # Limit chunk length going into prompt if necessary
            content_preview = doc.page_content[:1500] # Limit context chunk length if needed
            context_pieces.append(f"{header}\n{content_preview}")
        context_str = "\n\n---\n\n".join(context_pieces)

    # --- Assemble Simplified Prompt ---
    # Focus on the core task: answer from context and cite.
    prompt = f"""{history_str}**Instruction:** You are an AI assistant specialized in Nutrition and Fitness. Answer the user's latest query based *only* on the provided context chunks. Consider the chat history for understanding the query if necessary.

-   Carefully read the provided context chunks below.
-   Construct a comprehensive and explanatory answer to the 'Latest User Query' using *only* information found in the context chunks.
-   For *each* piece of information used from the context, cite the source document and page number in parentheses immediately after the information, like this: (Source: filename.pdf, Page X).
-   **If the context does not contain the information needed to answer the query (or if the context is marked 'CONTEXT_IS_MISSING'), state *only*: "Based on the provided documents, I cannot answer this question."**
-   Do not add any information not present in the context. Do not add introductory or concluding remarks not directly answering the query. Do not explain your reasoning process unless the query specifically asks for it.

---
**Provided Context:**
---
{context_str}
---

**Latest User Query:** {query}

---
**Answer:**""" # <<< LLM STARTS GENERATING ANSWER HERE

    log_query = query[:50].replace('\n', ' ')
    logger.info(f"Formatted Final Simple prompt using {len(context_docs)} context docs and {len(chat_history or [])} history turns for query: '{log_query}...'")
    return prompt

# --- REMOVE the extract_final_answer function ---
# No longer needed as the prompt doesn't ask for separate sections.
