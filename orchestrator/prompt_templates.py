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
    prompt = f"""**Instruction:** You are a helpful assistant specialized in Nutrition and Fitness. Based on the following context items, please answer the query.
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer.
    Make sure your answers are as explanatory as possible.
    Use the following examples as reference for the ideal answer style.
    \nExample 1:
    Query: What are the fat-soluble vitamins?
    Answer: The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
    \nExample 2:
    Query: What are the causes of type 2 diabetes?
    Answer: Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin. Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
    \nExample 3:
    Query: What is the importance of hydration for physical performance?
    Answer: Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
    \nNow use the following context items to answer the user query:
    {context_str}
    \nRelevant passages: <extract relevant passages from the context here>
    User query: {query}
    Answer:"""

    logger.info(f"Formatted prompt using {len(context_chunks)} context chunks for query: '{query[:50]}...'")
    return prompt