# orchestrator/prompt_templates.py

import logging
import re
from typing import List, Dict
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# --- format_rag_prompt function remains the same (Simplified version) ---
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

    # --- Assemble Simplified Prompt ---
    prompt = f"""{history_str}**Instruction:** You are an AI assistant specialized in Nutrition and Fitness. Answer the user's latest query based *only* on the provided context chunks. Consider the chat history for understanding the query if necessary.

-   Carefully read the provided context chunks below.
-   Construct a comprehensive and explanatory answer to the 'Latest User Query' using *only* information found in the context chunks.
-   For *each* piece of information used from the context, cite the source document and page number in parentheses immediately after the information, like this: (Source: filename.pdf, Page X).
-   **If the context does not contain the information needed to answer the query (or if the context is marked 'CONTEXT_IS_MISSING'), state *only*: "Based on the provided documents, I cannot answer this question."**
-   Do not add any information not present in the context. Do not add introductory or concluding remarks not directly answering the query. Do not explain your reasoning process unless the query specifically asks for it.

---
**BEGIN TASK**

**Provided Context:**
---
{context_str}
---

**Latest User Query:** {query}

---
**Answer:**"""

    log_query = query[:50].replace('\n', ' ')
    logger.info(f"Formatted RAG prompt using {len(context_docs)} context docs and {len(chat_history or [])} history turns for query: '{log_query}...'")
    return prompt


# --- ADD NEW FUNCTION FOR QUERY TRANSFORMATION ---
def format_query_transform_prompt(
    query: str,
    chat_history: List[Dict[str, str]]
    ) -> str:
    """Formats a prompt to ask the LLM to rewrite a query based on history."""

    history_str = ""
    if chat_history:
        for turn in chat_history:
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            # Include slightly more history content for transformation context
            content_preview = (content[:300] + '...') if len(content) > 300 else content
            history_str += f"{role}: {content}\n"
    else:
        # If no history, the original query is likely standalone
        return query # Return original query if no history

    prompt = f"""Given the following chat history and the latest user query, rewrite the latest user query to be a standalone question that incorporates the necessary context from the history. Only output the rewritten query, nothing else.

**Chat History:**
{history_str}
**Latest User Query:** {query}

**Standalone Query:**"""
    logger.debug("Formatted query transformation prompt.")
    return prompt

# --- REMOVE extract_final_answer function ---
# No longer needed with the simplified prompt
