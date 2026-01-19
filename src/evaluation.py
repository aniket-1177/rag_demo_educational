import nltk
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import precision_score, recall_score
from src.generation import get_llm

nltk.download('punkt', quiet=True)

def evaluate_bleu(reference: str, candidate: str):
    """BLEU score (5.9: Evaluation Metrics - BLEU)."""
    ref = [nltk.word_tokenize(reference)]
    cand = nltk.word_tokenize(candidate)
    return sentence_bleu(ref, cand)

def evaluate_context_precision_recall(retrieved_docs, relevant_docs):
    """
    Context precision and recall (5.9).
    - Precision: Relevant retrieved / total retrieved.
    - Recall: Relevant retrieved / total relevant.
    """
    retrieved_set = set([doc.page_content for doc in retrieved_docs])
    relevant_set = set(relevant_docs)  # Assume ground truth relevant texts
    true_pos = len(retrieved_set.intersection(relevant_set))
    precision = true_pos / len(retrieved_set) if retrieved_set else 0
    recall = true_pos / len(relevant_set) if relevant_set else 0
    return precision, recall

def evaluate_faithfulness(response: str, context: str):
    """Faithfulness (5.9): Use LLM to judge if response is grounded in context."""
    llm = get_llm()
    prompt = f"Is the following response faithful to the context? Response: {response}\nContext: {context}\nAnswer yes/no."
    judgment = llm.invoke(prompt).content.strip().lower()
    return 1 if judgment == "yes" else 0

def evaluate_relevance(response: str, query: str):
    """Relevance (5.9): Simple cosine similarity or LLM judgment."""
    llm = get_llm()
    prompt = f"Is the response relevant to the query? Response: {response}\nQuery: {query}\nAnswer yes/no."
    judgment = llm.invoke(prompt).content.strip().lower()
    return 1 if judgment == "yes" else 0