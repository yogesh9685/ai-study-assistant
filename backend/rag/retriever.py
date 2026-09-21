"""
retriever.py
------------
Document retrieval utilities for the RAG pipeline.
"""

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# MMR retriever (dense-only, kept for backward compatibility)
# ---------------------------------------------------------------------------

def create_retriever(vectorstore):
    """Create an MMR retriever from a vector store."""
    if vectorstore is None:
        raise ValueError("Vector store is required.")

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )


def retrieve_documents(retriever, question):
    """Retrieve relevant documents for a question."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    return retriever.invoke(question.strip())


def search_with_scores(vectorstore, question, k=4):
    """Search documents and return their similarity scores."""
    if vectorstore is None:
        raise ValueError("Vector store is required.")
    return vectorstore.similarity_search_with_score(question.strip(), k=k)


# ---------------------------------------------------------------------------
# Hybrid search  (BM25 + dense vector, fused by Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def hybrid_search(vectorstore, chunks: list[Document], question: str, k: int = 4) -> list[Document]:
    """
    Combine BM25 keyword search and FAISS dense search.

    Steps
    -----
    1. BM25   → rank chunks by keyword match
    2. FAISS  → rank chunks by semantic similarity
    3. RRF    → merge both ranked lists, return top-k

    Parameters
    ----------
    vectorstore : loaded FAISS vectorstore
    chunks      : list of Document chunks (same ones in the vectorstore)
    question    : user query
    k           : how many documents to return
    """
    from backend.rag.bm25_retriever import BM25Retriever

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    question = question.strip()
    fetch = max(k * 2, 10)  # fetch more candidates before fusing

    # 1. BM25 results
    bm25_results = BM25Retriever(chunks).get_top_k(question, k=fetch)

    # 2. Dense (MMR) results
    dense_results = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": fetch, "fetch_k": fetch * 2, "lambda_mult": 0.5},
    ).invoke(question)

    # 3. Reciprocal Rank Fusion — score = Σ 1/(60 + rank)
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in [bm25_results, dense_results]:
        for rank, doc in enumerate(ranked_list, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
            doc_map[key] = doc

    # Sort by fused score and return top-k
    top_keys = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
    return [doc_map[key] for key in top_keys]