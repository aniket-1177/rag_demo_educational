from langchain_huggingface import HuggingFaceEmbeddings
import config

def get_embedding_model():
    """
    Get embedding model (5.3: Vector Embeddings - Principles and Generation).
    - Embeddings map text to dense vectors (e.g., 384 dims here).
    - Principle: Semantic similarity via cosine distance.
    """
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)