"""
document_search.py
------------------
Branch: feature/document-search-tool

A LangChain tool that searches the existing FAISS vector store
and returns relevant document chunks for a given question.

Architecture
------------
  Agent
    |
    v
  document_search (this tool)
    |
    v
  existing MMR retriever  (backend.rag.retriever)
    |
    v
  FAISS vector store      (backend.rag.vectorstore)
    |
    v
  Formatted document results
    |
    v
  Agent -> LLM -> Final Answer

The tool does NOT call the LLM.
It only retrieves and formats relevant document chunks.

Usage
-----
  from backend.tools.document_search import document_search

  result = document_search.invoke("What is Python?")
  print(result)
"""

from pathlib import Path
from langchain_core.tools import tool

from backend.rag.vectorstore import load_vectorstore
from backend.rag.retriever import create_retriever


# ---------------------------------------------------------------------------
# Module-level retriever (initialized once, reused on every call)
#
# The retriever is created lazily the first time the tool is called.
# This avoids re-loading embeddings and FAISS on every query.
#
# _retriever = None means "not yet initialized".
# ---------------------------------------------------------------------------
_retriever = None
_last_documents = []


def get_last_documents():
    """Return documents retrieved by the most recent document_search call."""
    return _last_documents


def reset_last_documents():
    """Reset the recorded retrieved documents."""
    global _last_documents
    _last_documents = []


def reset_retriever():
    """Reset the cached retriever so subsequent calls reload FAISS from disk."""
    global _retriever, _last_documents
    _retriever = None
    _last_documents = []


def _get_retriever():
    """
    Return the shared retriever, initializing it on the first call.
    Returns None if no FAISS vector store exists on disk.
    """
    global _retriever

    if _retriever is not None:
        return _retriever

    # Check that the saved FAISS index actually exists on disk.
    index_path = Path("data/faiss_index")
    if not index_path.exists() or not any(index_path.iterdir()):
        return None

    try:
        vectorstore = load_vectorstore()
        _retriever = create_retriever(vectorstore)
        return _retriever

    except Exception:
        return None


def _format_results(documents):
    """
    Format a list of retrieved LangChain Document objects into a
    readable string the agent can use to answer the user's question.

    Each document shows its source, page number (if available),
    and the retrieved text content.
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

        # Build this document's section, skipping empty page line
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

    # Validate: reject empty questions
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    global _last_documents

    # Get the shared retriever (loads FAISS once, reuses after that)
    retriever = _get_retriever()
    if retriever is None:
        _last_documents = []
        return "No study documents are currently uploaded or indexed. Please inform the user that they must upload a study document first before asking questions about documents."

    # Retrieve relevant document chunks using the existing MMR retriever
    documents = retriever.invoke(question.strip())
    _last_documents = documents

    # Format and return the results - no LLM call happens here
    return _format_results(documents)

