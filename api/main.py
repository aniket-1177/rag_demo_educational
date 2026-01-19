from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.data_ingestion import load_documents, chunk_documents
from src.vector_db import ingest_to_vector_db
from src.generation import build_rag_chain, generate_response
from src.evaluation import evaluate_bleu, evaluate_context_precision_recall, evaluate_faithfulness, evaluate_relevance
from src.retrieval import hybrid_search, rerank_documents
from src.embeddings import get_embedding_model
from src.vector_db import get_vector_db

app = FastAPI(title="RAG Demo API")

class IngestRequest(BaseModel):
    directory: str

class QueryRequest(BaseModel):
    query: str
    evaluate: bool = False
    ground_truth: str = ""  # For evaluation
    relevant_docs: list[str] = []  # Ground truth relevant texts

# Global variables for demo (in production, use dependency injection)
chunks = None
chain = None

@app.post("/ingest")
def ingest(request: IngestRequest):
    global chunks, chain
    try:
        docs = load_documents(request.directory)
        chunks = chunk_documents(docs)
        db = ingest_to_vector_db(chunks)
        chain = build_rag_chain(chunks)
        return {"status": "Ingestion complete", "num_chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query(request: QueryRequest):
    if not chain:
        raise HTTPException(status_code=400, detail="Ingest data first")
    
    try:
        response = generate_response(chain, request.query)
        
        if request.evaluate:
            # For demo: Retrieve docs for eval
            db = get_vector_db()
            embedding_model = get_embedding_model()
            query_embedding = embedding_model.embed_query(request.query)
            retrieved = hybrid_search(db, chunks, request.query, query_embedding)
            reranked = rerank_documents(retrieved, request.query)
            context = "\n\n".join([doc.page_content for doc in reranked])
            
            bleu = evaluate_bleu(request.ground_truth, response) if request.ground_truth else None
            precision, recall = evaluate_context_precision_recall(reranked, request.relevant_docs)
            faithfulness = evaluate_faithfulness(response, context)
            relevance = evaluate_relevance(response, request.query)
            
            eval_results = {
                "bleu": bleu,
                "context_precision": precision,
                "context_recall": recall,
                "faithfulness": faithfulness,
                "relevance": relevance
            }
        else:
            eval_results = None
        
        return {"response": response, "evaluation": eval_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))