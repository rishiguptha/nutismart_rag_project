import logging
from typing import List

logger = logging.getLogger(__name__)

def format_rag_prompt(query: str, context_chunks: List[str]) -> str:
    """Formats the prompt for the LLM with retrieved context."""

    if not context_chunks:
        logger.warning("No context chunks provided for prompt formatting.")
        # If no context, we might ask the LLM to answer from its general knowledge,
        # or state that info wasn't found. Let's explicitly state it wasn't found.
        context_str = "No relevant context found in the documents."
        # You could modify the instruction below if context_str == "No relevant context found."
    else:
        # Join chunks with a clear separator
        context_str = "\n\n---\n\n".join(context_chunks)

    # Prompt Engineering: Clearly instruct the LLM
    prompt = f"""**Instruction:** You are a helpful assistant specialized in Nutrition and Fitness. Answer the following user query based *only* on the provided context. If the context does not contain the information needed to answer the query, state clearly: "Based on the provided documents, I cannot answer this question." Do not add any information not present in the context. Be concise.

**Context:**
---
{context_str}
---

**Query:** {query}

**Answer:**"""

    logger.info(f"Formatted prompt using {len(context_chunks)} context chunks for query: '{query[:50]}...'")
    return prompt