"""
retriever.py
------------
Document retrieval utilities for the RAG pipeline.
"""

import logging
from langchain_core.documents import Document
from backend.rag.hybrid_search import hybrid_search, HYBRID_TOP_K
from backend.rag.reranker import rerank_documents, FINAL_TOP_K

logger = logging.getLogger(__name__)


def create_retriever(vectorstore):
    """Create an MMR retriever from a vector store."""
    if vectorstore is None:
        raise ValueError("Vector store is required.")

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )


def retrieve_hybrid_and_rerank(
    vectorstore,
    question: str,
    hybrid_top_k: int = HYBRID_TOP_K,
    final_top_k: int = FINAL_TOP_K,
) -> list[Document]:
    """
    Perform two-stage retrieval:
      1. Hybrid search (BM25 keyword + FAISS semantic search) -> candidate chunks.
      2. Cross-Encoder reranking -> top final_top_k chunks.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if vectorstore is None:
        raise ValueError("Vector store is required.")

    # 1. Hybrid search (BM25 + FAISS) -> candidate chunks
    candidates = hybrid_search(vectorstore, question.strip(), hybrid_top_k=hybrid_top_k)

    if not candidates:
        logger.warning("Hybrid search returned 0 candidates for query: %s", question)
        return []

    # 2. Cross-Encoder reranking -> top final_top_k
    reranked = rerank_documents(question.strip(), candidates, top_k=final_top_k)
    return reranked


def retrieve_documents(retriever, question: str) -> list[Document]:
    """
    Retrieve relevant documents for a question using hybrid search + reranking
    if an underlying vectorstore is accessible, with fallback to standard retriever.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Check if retriever wraps a vectorstore for hybrid search + reranking
    vectorstore = getattr(retriever, "vectorstore", None)
    if vectorstore is not None:
        try:
            return retrieve_hybrid_and_rerank(vectorstore, question.strip())
        except Exception as error:
            logger.warning("Hybrid reranked retrieval failed (%s). Falling back to standard retriever.", error)

    return retriever.invoke(question.strip())


def search_with_scores(vectorstore, question, k=4):
    """Search documents and return their similarity scores."""
    if vectorstore is None:
        raise ValueError("Vector store is required.")
    return vectorstore.similarity_search_with_score(question.strip(), k=k)