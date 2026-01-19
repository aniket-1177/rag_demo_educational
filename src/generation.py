from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.embeddings import get_embedding_model
from src.vector_db import get_vector_db
from src.retrieval import hybrid_search, rerank_documents
import config
# from langchain.callbacks import get_openai_callback  # For token counting, but adapted for Groq

def get_llm():
    """Get LLM with token handling (5.2: Understanding Tokens and Token Limitations)."""
    return ChatGroq(groq_api_key=config.GROQ_API_KEY, model_name=config.LLM_MODEL)

def build_rag_chain(chunks):
    """
    Build RAG chain (5.1: Introduction to Retrieval Augmented Generation).
    - RAG: Retrieve relevant docs, augment prompt, generate response.
    - Handles token limits by chunking and top-k retrieval.
    """
    db = get_vector_db()
    embedding_model = get_embedding_model()
    
    def retrieve(query):
        query_embedding = embedding_model.embed_query(query)
        docs = hybrid_search(db, chunks, query, query_embedding)
        reranked = rerank_documents(docs, query)
        return "\n\n".join([doc.page_content for doc in reranked])
    
    template = """Answer the question based on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": RunnablePassthrough() | retrieve, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return chain

def generate_response(chain, query):
    """Generate response with token tracking."""
    # Note: Groq doesn't have built-in token callbacks like OpenAI; simulate or log.
    response = chain.invoke(query)
    # For demo: Print approximate tokens (5.2)
    print(f"Approximate tokens used: {len(query.split()) + len(response.split())} (actual varies)")
    return response