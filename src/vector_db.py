from langchain_chroma import Chroma
from src.embeddings import get_embedding_model
import config

def get_vector_db(persist_directory: str = config.VECTOR_DB_PATH):
    """
    Initialize vector database (5.4: Vector Databases - Chroma/Pinecone).
    - Chroma: Local vector store with embedding persistence.
    - Alternative: Pinecone for scalable cloud (requires API key).
    """
    embedding_model = get_embedding_model()
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

def ingest_to_vector_db(chunks):
    """Ingest chunks into vector DB."""
    db = get_vector_db()
    db.add_documents(chunks)
    return db