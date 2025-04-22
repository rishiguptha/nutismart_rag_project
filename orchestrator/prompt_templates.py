# In orchestrator/prompt_templates.py

import logging
import re # For extracting the final answer later
from typing import List

logger = logging.getLogger(__name__)

def format_react_style_rag_prompt(query: str, context_chunks: List[str]) -> str:
    """Formats a ReAct-style prompt for the LLM with retrieved context."""

    if not context_chunks:
        logger.warning("No context chunks provided for ReAct prompt formatting.")
        context_str = "No relevant context found in the documents."
    else:
        # Format context clearly, maybe numbering chunks
        context_str = "\n\n---\n\n".join(f"Context Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks))

    # ReAct-style Prompt Structure (Simulated Reasoning + Action = Final Answer)
    prompt = f"""**Instruction:** You are an AI assistant specialized in Nutrition and Fitness. Your task is to answer the user's query accurately based *only* on the provided context chunks. Follow these steps carefully:

1.  **Thought:** First, analyze the user's query (`{query}`). Read through the provided context chunks. Identify and explicitly list the key sentences or passages from the context chunks that are directly relevant to answering the query. Analyze how these relevant passages connect to form the answer. Determine if the combined information is sufficient. *Do not add any external knowledge or assumptions.* If the context does not contain the necessary information, state that clearly in this thought process.

2.  **Final Answer:** Based *only* on the analysis in your Thought step, construct a comprehensive and explanatory answer to the original query. If your thought process concluded that the information is not available in the context, the final answer should *only* be: "Based on the provided documents, I cannot answer this question."

**Example Answer Style Guidance (Do NOT use content from examples to answer the current query):**
* *For "What are fat-soluble vitamins?":* Answer should list Vitamin A, D, E, K and briefly explain their function and storage based *only* on context provided for *that* query.
* *For "Causes of type 2 diabetes?":* Answer should explain the link to overnutrition, insulin resistance etc., based *only* on context provided for *that* query.
* *For "Importance of hydration?":* Answer should explain roles in temperature regulation, nutrient transport, performance impact based *only* on context provided for *that* query.

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

    logger.info(f"Formatted ReAct-style prompt using {len(context_chunks)} context chunks for query: '{query[:50]}...'")
    return prompt

def extract_final_answer(llm_output: str) -> str:
    """Extracts the Final Answer part from the LLM's ReAct-style output."""
    # Use regex to find the 'Final Answer:' section, ignoring case and leading/trailing whitespace
    # It captures everything after 'Final Answer:' until the end of the string
    match = re.search(r"\*\*Final Answer\*\*:?\s*(.*)", llm_output, re.IGNORECASE | re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
        logger.debug(f"Extracted Final Answer: '{final_answer[:100]}...'")
        return final_answer
    else:
        # Fallback if the LLM didn't follow the format perfectly
        logger.warning("Could not find 'Final Answer:' section in LLM output. Returning full output as fallback.")
        # You might want to remove the "Thought:" part even in fallback
        thought_match = re.search(r"\*\*Thought\*\*:?\s*(.*)", llm_output, re.IGNORECASE | re.DOTALL)
        if thought_match:
             # Return everything after the thought section if final answer marker missing
             return llm_output[thought_match.end():].strip()
        return llm_output # Return everything if neither marker found clearly