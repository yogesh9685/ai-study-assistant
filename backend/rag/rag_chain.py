"""
rag_chain.py
------------
Conversational RAG pipeline.

Steps per turn:
  1. Load conversation history for the session
  2. Rewrite the question into a standalone question (resolves "it", "they", etc.)
  3. Retrieve relevant document chunks via FAISS MMR retriever
  4. Generate a grounded answer using only the retrieved context
  5. Save the turn to conversation history
  6. Return the answer and its sources
"""

import logging

logger = logging.getLogger(__name__)


def get_sources(documents):
    """Extract unique (source, page) pairs from retrieved documents."""
    sources = []
    seen = set()

    for document in documents:
        source = document.metadata.get("source")
        page = document.metadata.get("page")

        key = (source, page)

        if key not in seen:
            sources.append({
                "source": source,
                "page": page
            })
            seen.add(key)

    return sources


def run_conversational_rag(question: str, session_id: str) -> tuple[str, list]:
    """
    Full conversational RAG pipeline for a single turn.

    Parameters
    ----------
    question   : the raw user question (may contain pronoun references)
    session_id : unique identifier for this user's session

    Returns
    -------
    (answer: str, sources: list[dict])
    """
    from backend.rag.generator import create_llm, generate_answer
    from backend.rag.conversation import add_message, rewrite_question
    from backend.rag.vectorstore import load_vectorstore
    from backend.rag.retriever import create_retriever, retrieve_documents
    from backend.rag.session_store import get_history, save_history
    from pathlib import Path

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    # 1. Load conversation history for this session
    history = get_history(session_id)

    # Create LLM once — reused for both question rewriting and answer generation
    llm = create_llm()

    # 2. Rewrite follow-up question into a standalone question
    #    Returns the original question unchanged when history is empty
    try:
        standalone_question = rewrite_question(llm, question, history)
    except Exception:
        standalone_question = question

    from backend.rag.vectorstore import has_vectorstore

    # 3. Load vectorstore and retrieve relevant chunks via FAISS MMR
    if not has_vectorstore(session_id):
        raise RuntimeError(
            "No vector store found for this session. "
            "Please upload and index a document before asking questions."
        )

    vectorstore = load_vectorstore(session_id)
    retriever = create_retriever(vectorstore)
    documents = retrieve_documents(retriever, standalone_question)

    logger.info("MMR retriever returned %d document chunks.", len(documents))

    # 4. Generate grounded answer from retrieved context
    answer = generate_answer(llm, standalone_question, documents)

    # 5. Extract source metadata
    sources = get_sources(documents)

    # 6. Save this turn to conversation history
    history = add_message(history, "user", question)
    history = add_message(history, "assistant", answer)
    save_history(session_id, history)

    return answer, sources