# orchestrator/prompt_templates.py

import logging
import re
import os # Import os for basename
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
    V3: Stronger emphasis on citation format.
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
    source_details_for_prompt = {}
    if not context_docs:
        logger.warning("No context documents provided for prompt formatting.")
        # Provide a clear indicator for the LLM if context is missing
        context_str = "NO CONTEXT PROVIDED."
    else:
        context_pieces = []
        unique_sources = {} # Track unique source/page combos to assign numbers
        source_number_counter = 1
        doc_to_source_num = {} # Map doc index to its assigned source number

        for i, doc in enumerate(context_docs):
            source = doc.metadata.get('source', 'Unknown Source')
            # Handle potential path separators for different OS
            filename = os.path.basename(source) # Get only the filename/URL end
            page = doc.metadata.get('page', 'Web')
            # Create a unique key for source file/URL and page number
            source_key = f"{filename}::{page}"

            if source_key not in unique_sources:
                 current_source_num = source_number_counter
                 unique_sources[source_key] = current_source_num
                 # Store details for the final source list instruction using the assigned number
                 source_details_for_prompt[current_source_num] = f"{filename}, Page {page}"
                 source_number_counter += 1
            else:
                 current_source_num = unique_sources[source_key]

            doc_to_source_num[i] = current_source_num # Store which source number this doc corresponds to

            # Format the context chunk header using the assigned source number
            header = f"Context Source [{current_source_num}]:"
            content_preview = doc.page_content[:1500] # Limit context chunk length
            context_pieces.append(f"{header}\n{content_preview}")

        context_str = "\n\n---\n\n".join(context_pieces)

    # --- Construct the Sources List String for the Prompt ---
    # Create the exact string we want the LLM to replicate for the sources it uses
    sources_list_instruction = "\n".join([f"[{k}] {v}" for k, v in sorted(source_details_for_prompt.items())])


    # --- Assemble Prompt with Refined Instructions ---
    prompt = f"""{history_str}**Instruction:** You are an expert AI assistant specialized in Nutrition and Fitness, tasked with answering the user's query accurately and comprehensively based *strictly* on the provided context chunks.

**Your Process:**
1.  Analyze the 'Latest User Query' below, considering the 'Chat History' if present.
2.  Carefully review *all* 'Provided Context' chunks, each marked with a `Context Source [Number]`. Note that multiple context chunks might share the same source number if they come from the same source page.
3.  Synthesize the relevant information *only* from these chunks to directly answer the query.
4.  Construct your answer clearly and explanatorily.
5.  **CRITICAL Citation Requirement:** For *every* piece of information or claim in your answer, you **MUST** include a numerical citation marker in the format `[Number]` corresponding to the `Context Source` it came from. Place the marker immediately after the information it supports (e.g., "Adults need protein [1]."). If information comes from multiple sources, cite them all (e.g., "Hydration is important [2, 4]."). **DO NOT use any other citation format like `(Source: ...)`**.
6.  **CRITICAL Sources List Requirement:** AFTER providing the complete answer text, add a section titled exactly `**Sources:**`. Under this title, create a numbered list. For *each unique source number* you cited in your answer text, list its corresponding source details on a new line. Use the EXACT format shown in the 'Available Source Details' section below. ONLY include sources you actually cited in the answer. Ensure the numbers in this list match the numbers used in your answer text.

**Available Source Details (Use these for the Sources list):**
{sources_list_instruction if sources_list_instruction else "No sources available."}

**Example of Correct Final Output Structure:**
```
[Your answer text with numerical citations like [1] and [2] embedded within.]

**Sources:**
[1] filename1.pdf, Page X
[2] some_web_url, Page Web
```
7.  **Handling Insufficient Context:** If the 'Provided Context' section below says "NO CONTEXT PROVIDED." or if the provided context chunks do *not* contain the specific information needed to answer the query, respond *only* with the exact phrase: `{CANNOT_ANSWER_PHRASE}` Do *not* include a 'Sources:' section in this case.
8.  **Constraints:** Do NOT include any information not explicitly found in the context. Do NOT add introductory phrases like "Based on the context..." or concluding summaries like "In summary...". Stick strictly to the requested format. Do NOT use the `(Source: ...)` citation style.

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
    # Use len(context_docs) for logging consistency
    logger.info(f"Formatted Prompt (Tuned Citations v3) using {len(context_docs)} context docs and {len(chat_history or [])} history turns for query: '{log_query}...'")
    return prompt

# Define the fallback phrase constant (used in prompt and evaluation script)
CANNOT_ANSWER_PHRASE = "Based on the provided documents, I cannot answer this question."

# --- REMOVED extract_final_answer function ---
# Not needed as the full output (answer + sources) is desired.

