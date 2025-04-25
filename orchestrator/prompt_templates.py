# orchestrator/prompt_templates.py

import logging
import re
from typing import List
from langchain_core.documents import Document # Import Document type

logger = logging.getLogger(__name__)

# --- MODIFIED TO ACCEPT DOCUMENTS AND FORMAT WITH METADATA ---
def format_react_style_rag_prompt(query: str, context_docs: List[Document]) -> str:
    """Formats a ReAct-style prompt for the LLM with retrieved context including metadata."""

    context_str = ""
    if not context_docs:
        logger.warning("No context documents provided for ReAct prompt formatting.")
        context_str = "No relevant context found in the documents."
    else:
        # Format context clearly, including source and page number
        context_pieces = []
        for i, doc in enumerate(context_docs):
            source = doc.metadata.get('source', 'Unknown Source')
            # Extract just the filename from the source path
            filename = source.split('/')[-1].split('\\')[-1] # Handle both path separators
            page = doc.metadata.get('page', 'N/A')
            header = f"Context Chunk {i+1} (Source: {filename}, Page: {page}):"
            context_pieces.append(f"{header}\n{doc.page_content}")
        context_str = "\n\n---\n\n".join(context_pieces)

    # ReAct-style Prompt Structure (Simulated Reasoning + Action = Final Answer)
    # --- MODIFIED INSTRUCTIONS FOR CITATION ---
    prompt = f"""**Instruction:** You are an AI assistant specialized in Nutrition and Fitness. Your task is to answer the user's query accurately based *only* on the provided context chunks. Follow these steps carefully:

1.  **Thought:** First, analyze the user's query (`{query}`). Read through the provided context chunks, paying attention to the source file and page number indicated for each chunk. Identify and explicitly list the key sentences or passages from the context chunks that are directly relevant to answering the query. Analyze how these relevant passages connect to form the answer. Determine if the combined information is sufficient. *Do not add any external knowledge or assumptions.* If the context does not contain the necessary information, state that clearly in this thought process.

2.  **Final Answer:** Based *only* on the analysis in your Thought step, construct a comprehensive and explanatory answer to the original query. **Crucially, for each piece of information you use from the context, cite the source document and page number in parentheses immediately after the information, like this: (Source: filename.pdf, Page X).** If your thought process concluded that the information is not available in the context, the final answer should *only* be: "Based on the provided documents, I cannot answer this question."

**Example Answer Style Guidance (Illustrates citation format, do NOT use content from examples):**
* *For "What are fat-soluble vitamins?":* Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K (Source: nutrition_book.pdf, Page 45). These are absorbed with fats (Source: nutrition_book.pdf, Page 45). Vitamin A is important for vision (Source: nutrition_book.pdf, Page 46), Vitamin D for bone health (Source: nutrition_book.pdf, Page 47), etc.
* *For "Causes of type 2 diabetes?":* Answer: Type 2 diabetes is often linked to overnutrition and obesity (Source: wellness_guide.pdf, Page 12). This can lead to insulin resistance (Source: wellness_guide.pdf, Page 12).
* *For "Importance of hydration?":* Answer: Water helps regulate body temperature and transport nutrients (Source: fitness_basics.pdf, Page 5). Dehydration can decrease performance (Source: fitness_basics.pdf, Page 6).

---
**BEGIN TASK**

**Provided Context:**
---
{context_str}
---

**User Query:** {query}

---
**Output:**

**Thought:**
""" # <<< LLM STARTS GENERATING THOUGHT PROCESS HERE

    logger.info(f"Formatted ReAct-style prompt using {len(context_docs)} context documents for query: '{query[:50]}...'")
    return prompt

# --- UNCHANGED ---
def extract_final_answer(llm_output: str) -> str:
    """Extracts the Final Answer part from the LLM's ReAct-style output."""
    match = re.search(r"\*\*Final Answer\*\*:?\s*(.*)", llm_output, re.IGNORECASE | re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
        logger.debug(f"Extracted Final Answer: '{final_answer[:100]}...'")
        return final_answer
    else:
        logger.warning("Could not find 'Final Answer:' section in LLM output. Returning full output as fallback.")
        thought_match = re.search(r"\*\*Thought\*\*:?\s*(.*)", llm_output, re.IGNORECASE | re.DOTALL)
        if thought_match:
             return llm_output[thought_match.end():].strip()
        return llm_output
