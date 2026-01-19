from langchain_community.retrievers import BM25Retriever  # For keyword search
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import config

def similarity_search(db, query_embedding, k: int = config.TOP_K):
    """
    Similarity search (5.6: Similarity Search and Retrieval Mechanisms).
    - Uses vector similarity (e.g., cosine) to retrieve top-k docs.
    """
    return db.similarity_search_by_vector(query_embedding, k=k)

def hybrid_search(db, chunks, query, query_embedding, k: int = config.TOP_K):
    """
    Hybrid search (5.7: Hybrid Search Methods).
    - Combines vector (semantic) and keyword (BM25) search.
    - Merges results by simple union and de-dup.
    """
    # Vector search
    vec_docs = similarity_search(db, query_embedding, k=k)
    
    # Keyword search (BM25 on chunks)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_docs = bm25_retriever.invoke(query)[:k]
    
    # Merge and de-dup
    merged = {doc.page_content: doc for doc in vec_docs + bm25_docs}.values()
    return list(merged)[:k]

def rerank_documents(docs, query):
    """
    Re-ranking techniques (5.8: Re-ranking Techniques).
    - Uses cross-encoder to score query-doc pairs for better relevance.
    """
    model = CrossEncoder(config.RERANK_MODEL)
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)
    sorted_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    return sorted_docs