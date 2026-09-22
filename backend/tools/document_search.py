"""
document_search.py
------------------
A LangChain tool that searches the session-isolated FAISS vector store
and returns relevant document chunks for a given question.

Each session maintains its own vector store, retriever cache, and
retrieved documents history.
"""

import logging
from contextvars import ContextVar
from typing import Any
from langchain_core.tools import tool

from backend.rag.vectorstore import load_vectorstore, has_vectorstore
from backend.rag.retriever import create_retriever

logger = logging.getLogger(__name__)

# Context variable tracking the active session_id for the current execution context
current_session_id: ContextVar[str | None] = ContextVar("current_session_id", default=None)

# Session-isolated retriever cache: {session_id: retriever}
_retrievers: dict[str, Any] = {}

# Session-isolated last retrieved documents: {session_id: list}
_last_documents: dict[str, list] = {}


def get_last_documents(session_id: str | None = None) -> list:
    """Return documents retrieved by the most recent document_search call for the session."""
    sid = session_id or current_session_id.get()
    if sid:
        return _last_documents.get(sid, [])
    return _last_documents.get("_default", [])


def reset_last_documents(session_id: str | None = None):
    """Reset the recorded retrieved documents for a session."""
    sid = session_id or current_session_id.get()
    if sid:
        _last_documents.pop(sid, None)
    else:
        _last_documents.clear()


def reset_retriever(session_id: str | None = None):
    """Reset cached retriever so subsequent calls reload FAISS from disk."""
    if session_id:
        _retrievers.pop(session_id, None)
        _last_documents.pop(session_id, None)
        logger.info("Reset retriever cache for session: %s", session_id)
    else:
        _retrievers.clear()
        _last_documents.clear()
        logger.info("Reset all retriever caches")


def _get_retriever(session_id: str | None = None):
    """
    Return the session-specific retriever, initializing it on first call.
    Returns None if no FAISS vector store exists on disk for this session.
    """
    sid = session_id or current_session_id.get()

    if sid and sid in _retrievers:
        return _retrievers[sid]

    if not sid and "_default" in _retrievers:
        return _retrievers["_default"]

    # Check that the saved FAISS index exists on disk for this session
    if not has_vectorstore(sid):
        return None

    try:
        vectorstore = load_vectorstore(sid)
        retriever = create_retriever(vectorstore)
        if sid:
            _retrievers[sid] = retriever
        else:
            _retrievers["_default"] = retriever
        return retriever

    except Exception as exc:
        logger.error("Failed to load vector store for session %s: %s", sid, exc)
        return None


def _format_results(documents):
    """
    Format a list of retrieved LangChain Document objects into a
    readable string the agent can use to answer the user's question.
    """
    if not documents:
        return "No relevant information was found in the uploaded documents."

    parts = []

    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page")
        content = doc.page_content.strip()

        header = f"--- Document {i} ---"
        source_line = f"Source: {source}"
        page_line   = f"Page: {page}" if page is not None else ""
        content_block = f"Content:\n{content}"

        section_lines = [header, source_line]
        if page_line:
            section_lines.append(page_line)
        section_lines.append(content_block)

        parts.append("\n".join(section_lines))

    return "\n\n".join(parts)


@tool
def document_search(question: str) -> str:
    """
    Search the uploaded study documents for information relevant
    to the user question. Use this tool when the user asks anything
    about the content of their uploaded documents.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    sid = current_session_id.get()

    # Get the session's retriever
    retriever = _get_retriever(sid)
    if retriever is None:
        if sid:
            _last_documents.pop(sid, None)
        else:
            _last_documents.pop("_default", None)
        return (
            "No study documents are currently uploaded or indexed for your session. "
            "Please inform the user that they must upload a study document first before "
            "asking questions about documents."
        )

    # Retrieve relevant document chunks using the MMR retriever
    documents = retriever.invoke(question.strip())
    if sid:
        _last_documents[sid] = documents
    else:
        _last_documents["_default"] = documents

    return _format_results(documents)
