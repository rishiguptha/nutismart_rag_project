# orchestrator/prompt_templates.py

import logging
import re
from typing import List, Dict
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# --- Prompt Function - Encouraging Synthesis ---
def format_rag_prompt(
    query: str,
    context_docs: List[Document],
    chat_history: List[Dict[str, str]] = None
    ) -> str:
    """
    Formats a direct prompt for the LLM, encouraging synthesis from context
    and handling cases where direct examples might be definitions or pointers.
    """

    # --- Format Chat History ---
    history_str = ""
    if chat_history:
        history_str += "**Chat History (for context):**\n"
        for turn in chat_history:
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            content_preview = (content[:150] + '...') if len(content) > 150 else content
            history_str += f"{role}: {content_preview}\n"
        history_str += "\n---\n\n"

    # --- Format Context ---
    context_str = ""
    if not context_docs:
        logger.warning("No context documents provided for prompt formatting.")
        context_str = "CONTEXT_IS_MISSING"
    else:
        context_pieces = []
        for i, doc in enumerate(context_docs):
            source = doc.metadata.get('source', 'Unknown Source')
            filename = source.split('/')[-1].split('\\')[-1]
            page = doc.metadata.get('page', 'N/A')
            header = f"Context Chunk {i+1} (Source: {filename}, Page: {page}):"
            content_preview = doc.page_content[:1500]
            context_pieces.append(f"{header}\n{content_preview}")
        context_str = "\n\n---\n\n".join(context_pieces)

    # --- Assemble Prompt - Modified Instructions ---
    prompt = f"""{history_str}**Instruction:** You are an AI assistant specialized in Nutrition and Fitness. Answer the user's latest query based *only* on the provided context chunks, considering the chat history for understanding the query.

-   Carefully read the provided context chunks. Synthesize the information to answer the query comprehensively.
-   If the query asks for examples and the context provides definitions or classifications (like the talk test for intensity) instead of an explicit list, **use those definitions to describe the types of activities** that would fit.
-   Construct a comprehensive and explanatory answer to the 'Latest User Query' using *only* information found in the context chunks.
-   For *each* piece of information used from the context, cite the source document and page number in parentheses immediately after the information, like this: (Source: filename.pdf, Page X).
-   **If the context is truly insufficient to answer the query even using definitions or classifications (or if context is marked 'CONTEXT_IS_MISSING'), state *only*: "Based on the provided documents, I cannot answer this question."**
-   Do not add any information not present in the context. Do not add introductory or concluding remarks.

---
**BEGIN TASK**

**Provided Context:**
---
{context_str}
---

**Latest User Query:** {query}

---
**Answer:**""" # <<< LLM STARTS GENERATING ANSWER HERE

    log_query = query[:50].replace('\n', ' ')
    logger.info(f"Formatted Synthesis prompt using {len(context_docs)} context docs and {len(chat_history or [])} history turns for query: '{log_query}...'")
    return prompt

# --- REMOVE extract_final_answer function ---
# Not needed with this direct prompt structure.
