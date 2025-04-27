# orchestrator/prompt_templates.py

import logging
import re
from typing import List, Dict
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# --- Prompt Function with Numerical Citations ---
def format_rag_prompt(
    query: str,
    context_docs: List[Document],
    chat_history: List[Dict[str, str]] = None
    ) -> str:
    """
    Formats a prompt for the LLM with retrieved context, chat history,
    and instructions for numerical citations.
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

    # --- Format Context with Numerical Indexing ---
    context_str = ""
    source_map = {} # To map number to source details
    if not context_docs:
        logger.warning("No context documents provided for prompt formatting.")
        context_str = "CONTEXT_IS_MISSING"
    else:
        context_pieces = []
        for i, doc in enumerate(context_docs):
            source_num = i + 1 # Start numbering from 1
            source = doc.metadata.get('source', 'Unknown Source')
            filename = source.split('/')[-1].split('\\')[-1] # Handle both path separators
            page = doc.metadata.get('page', 'Web') # Use 'Web' if no page number
            header = f"Context Source [{source_num}] (File: {filename}, Page: {page}):"
            content_preview = doc.page_content[:1500] # Limit context chunk length if needed
            context_pieces.append(f"{header}\n{content_preview}")
            # Store mapping for the LLM to use later in the Sources section
            source_map[source_num] = f"{filename}, Page {page}"

        context_str = "\n\n---\n\n".join(context_pieces)

    # --- Assemble Prompt with Numerical Citation Instructions ---
    prompt = f"""{history_str}**Instruction:** You are an AI assistant specialized in Nutrition and Fitness. Answer the user's latest query based *only* on the provided context chunks. Consider the chat history for understanding the query if necessary.

-   Carefully read the provided context chunks, each marked with a source number like `Context Source [1]`, `Context Source [2]`, etc.
-   Construct a comprehensive and explanatory answer to the 'Latest User Query' using *only* information found in the context chunks.
-   **IMPORTANT CITATION STYLE:** When you use information from a specific context chunk, add a numerical citation marker like `[1]`, `[2]`, etc., corresponding to the source number of that chunk, immediately after the relevant sentence or phrase. You may cite multiple sources for a single statement if applicable, like `[1, 3]`.
-   **After** providing the complete answer, add a section titled "**Sources:**" followed by a numbered list detailing the source file and page for each citation number used in your answer. Use the File and Page information provided in the context headers. For example:
    ```
    Sources:
    [1] filename1.pdf, Page X
    [2] filename2.pdf, Page Y
    [3] some_web_url, Page Web
    ```
-   **If the context does not contain the information needed to answer the query (or if the context is marked 'CONTEXT_IS_MISSING'), state *only*: "Based on the provided documents, I cannot answer this question." Do *not* include a Sources section in this case.**
-   Do not add any information not present in the context. Do not add introductory or concluding remarks not directly answering the query.

---
**BEGIN TASK**

**Provided Context:**
---
{context_str}
---

**Latest User Query:** {query}

---
**Answer:**""" # <<< LLM STARTS GENERATING ANSWER (including citations) and then the Sources section HERE

    log_query = query[:50].replace('\n', ' ')
    logger.info(f"Formatted Numerical Citation prompt using {len(context_docs)} context docs and {len(chat_history or [])} history turns for query: '{log_query}...'")
    return prompt

# --- REMOVE extract_final_answer and format_query_transform_prompt functions ---
