# orchestrator/prompt_templates.py

import logging
import re
from typing import List, Dict
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# --- Prompt Function - Focused on Fixing Sources List ---
def format_rag_prompt(
    query: str,
    context_docs: List[Document],
    chat_history: List[Dict[str, str]] = None
    ) -> str:
    """
    Formats a prompt for the LLM focused on generating high-quality,
    grounded answers with a correctly formatted numerical citation list.
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

    # --- Format Context with Numerical Indexing AND Store Source Details ---
    context_str = ""
    # Create a mapping from source number to its details (filename, page)
    # This map will be used to construct the final "Sources:" list instruction
    source_details_for_prompt = {}
    if not context_docs:
        logger.warning("No context documents provided for prompt formatting.")
        context_str = "CONTEXT_IS_MISSING"
    else:
        context_pieces = []
        for i, doc in enumerate(context_docs):
            source_num = i + 1 # Start numbering from 1
            source = doc.metadata.get('source', 'Unknown Source')
            filename = source.split('/')[-1].split('\\')[-1]
            page = doc.metadata.get('page', 'Web')
            header = f"Context Source [{source_num}]:" # Keep header simple for LLM
            content_preview = doc.page_content[:1500] # Limit context chunk length
            context_pieces.append(f"{header}\n{content_preview}")
            # Store details for the final source list instruction
            source_details_for_prompt[source_num] = f"{filename}, Page {page}"

        context_str = "\n\n---\n\n".join(context_pieces)

    # --- Construct the Sources List String for the Prompt ---
    # Create the exact string we want the LLM to replicate for the sources it uses
    sources_list_instruction = "\n".join([f"[{k}] {v}" for k, v in source_details_for_prompt.items()])


    # --- Assemble Prompt with Refined Instructions ---
    prompt = f"""{history_str}**Instruction:** You are an expert AI assistant specialized in Nutrition and Fitness, tasked with answering the user's query accurately and comprehensively based *strictly* on the provided context chunks.

**Your Process:**
1.  Analyze the 'Latest User Query' below, considering the 'Chat History' if present.
2.  Carefully review *all* 'Provided Context' chunks, each marked with a `Context Source [Number]`.
3.  Synthesize the relevant information *only* from these chunks to directly answer the query.
4.  Construct your answer clearly and explanatorily.
5.  **Citation Requirement:** For every piece of information or claim in your answer, you *must* include a numerical citation marker `[Number]` corresponding to the `Context Source` it came from. Place the marker immediately after the information it supports (e.g., "Adults need protein [1]."). If information comes from multiple sources, cite them all (e.g., "Hydration is important [2, 4].").
6.  **Sources List Requirement:** AFTER providing the complete answer text, add a section titled exactly "**Sources:**". Under this title, create a numbered list. For *each unique citation number* you used in your answer text, list its corresponding source details on a new line. Use the EXACT format shown in the 'Available Source Details' section below. ONLY include sources you actually cited in the answer.

**Available Source Details (Use these for the Sources list):**
{sources_list_instruction}

**Example of Correct Final Output Structure:**
```
[Your answer text with numerical citations like [1] and [2] embedded within.]

**Sources:**
[1] filename1.pdf, Page X
[2] some_web_url, Page Web
```
7.  **Handling Insufficient Context:** If the provided context chunks do *not* contain the specific information needed to answer the query, respond *only* with the exact phrase: "Based on the provided documents, I cannot answer this question." Do *not* include a 'Sources:' section in this case.
8.  **Constraints:** Do NOT include any information not explicitly found in the context. Do NOT add introductory phrases or concluding summaries.

---
**BEGIN TASK**

**Provided Context:**
---
{context_str}
---

**Latest User Query:** {query}

---
**Answer:**""" # <<< LLM starts generating answer + sources here

    log_query = query[:50].replace('\n', ' ')
    logger.info(f"Formatted Quality Focus prompt v2 using {len(context_docs)} context docs and {len(chat_history or [])} history turns for query: '{log_query}...'")
    return prompt

# --- REMOVE extract_final_answer function ---
# Still not needed as the full output (answer + sources) is desired.
