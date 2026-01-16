# Configuration file for the project
import os
from dotenv import load_dotenv

load_dotenv()

# LLM settings (5.2: Token limitations - Groq models have context limits, e.g., 8k tokens)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set this!
LLM_MODEL = "llama3-8b-8192"  # Groq model with ~8k token limit

# Embedding model (5.3: Vector embeddings)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # HuggingFace sentence-transformers

# Vector DB (5.4: Chroma for local demo; Pinecone for cloud)
VECTOR_DB_PATH = "./chroma_db"  # Local persistence for Chroma

# Chunking (5.5)
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50

# Retrieval (5.6, 5.7, 5.8)
TOP_K = 5  # For similarity search
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # For re-ranking