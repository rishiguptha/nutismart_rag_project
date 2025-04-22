# Hybrid RAG Project (Nutrition & Fitness)

This project implements a Retrieval-Augmented Generation (RAG) system to answer questions about nutrition and fitness.

It uses local documents, processes them into a local ChromaDB vector store, retrieves relevant context locally, and then uses an external LLM API (like Google's Gemma/Gemini) to generate the final answer based on the context.

## Setup

1.  **Clone:** `git clone ...`
2.  **Create Environment:** `python -m venv venv` (or use conda)
3.  **Activate:** `source venv/bin/activate` (macOS/Linux) or `.\venv\Scripts\activate` (Windows)
4.  **Install Deps:** `pip install -r requirements.txt`
5.  **Add Data:** Place your PDF/other documents into the `data/` directory.
6.  **Set API Key:** Create a `.env` file in the project root and add your external LLM API key (e.g., `GOOGLE_API_KEY="YOUR_API_KEY"`). Add `.env` to your `.gitignore`.
7.  **Index Data:** Run the indexing script: `python scripts/index_data.py`

## Running

1.  Activate your environment: `source venv/bin/activate`
2.  Run the main application: `python -m orchestrator.main_app`
3.  Enter queries when prompted. Type 'quit' to exit.

## Testing Retrieval

You can test the vector store retrieval independently:
`python scripts/test_retrieval.py`