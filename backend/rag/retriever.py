"""
retriever.py
------------
Document retrieval utilities for the RAG pipeline.
"""

import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


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