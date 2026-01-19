# RAG Demo Project

This is an end-to-end Retrieval Augmented Generation (RAG) project for educational purposes, covering Unit 5 syllabus.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Add your Groq API key to `config.py`.
3. Place sample documents in `data/sample_docs/`.
4. Run the FastAPI server: `uvicorn api.main:app --reload`
5. Ingest data: POST to http://localhost:8000/ingest with JSON body {"directory": "data/sample_docs"}
6. Query: POST to http://localhost:8000/query with JSON body {"query": "Your question here", "evaluate": true}

## Syllabus Coverage
- See comments in source files for mappings to Unit 5 sections.

## Modular Structure
- `src/data_ingestion.py`: Handles loading, chunking (5.5), and metadata.
- `src/embeddings.py`: Generates vector embeddings (5.3).
- `src/vector_db.py`: Interacts with vector DB like Chroma (5.4).
- `src/retrieval.py`: Similarity search (5.6), hybrid search (5.7), re-ranking (5.8).
- `src/generation.py`: RAG pipeline with LLM, handling tokens (5.2).
- `src/evaluation.py`: Metrics like BLEU, precision/recall, etc. (5.9).
- `api/main.py`: FastAPI inference endpoints.