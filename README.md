# Hybrid RAG Project (Nutrition & Fitness)

This project implements a Retrieval-Augmented Generation (RAG) system to answer questions about nutrition and fitness.

It uses local documents, processes them into a local ChromaDB vector store, retrieves relevant context locally, and then uses an external LLM API (like Google's Gemma/Gemini) to generate the final answer based on the context.

## Setup

1.  **Clone:** `git clone ...`
2.  **Create Environment:** `python -m venv venv` (or use conda)
3.  **Activate:** `source venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows)
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

## Project Structure

```
nutismart_rag_project/
├── app/                    # Core application code
│   └── core/              # Core RAG components
│       ├── embedder.py    # Text embedding functionality
│       ├── ranker.py      # Document ranking
│       ├── retriever.py   # Document retrieval
│       ├── generator.py   # Answer generation
│       ├── pipeline.py    # Main RAG pipeline
│       ├── utils.py       # Utility functions
│       └── exceptions.py  # Custom exceptions
├── config/                # Configuration files
│   ├── settings.py       # Application settings
│   └── logging_config.py # Logging configuration
├── data/                  # Data directory
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data files
├── logs/                  # Log files
├── scripts/              # Utility scripts
│   ├── example.py        # Pipeline usage example
│   ├── evaluate_custom.py # Custom evaluation script
│   └── index_data.py     # Data indexing script
├── tests/                # Test files
├── vector_store/         # Vector store files
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose configuration
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## Components

### Core RAG Pipeline

The main RAG pipeline consists of:
- **Embedder**: Handles text embedding using Sentence Transformers
- **Ranker**: Ranks documents using Cross-Encoder
- **Retriever**: Combines embedding and ranking for document retrieval
- **Generator**: Generates answers using Gemini

### Configuration

- `settings.py`: Contains model names, API keys, and other settings
- `logging_config.py`: Configures logging for the application

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.