"""
vectorstore.py
--------------
FAISS vector store creation, persistence, loading, and deletion.
Scoper to session_id for multi-user document isolation.
"""

import logging
import re
import shutil
from pathlib import Path
from langchain_community.vectorstores import FAISS
from backend.rag.embeddings import create_embeddings

logger = logging.getLogger(__name__)

VECTOR_STORE_BASE_DIR = Path("data/vectorstores")


def validate_session_id(session_id: str) -> str:
    """
    Validate that session_id is non-empty, contains only safe characters,
    and cannot be used for directory traversal attacks.
    """
    if not session_id or not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Session ID cannot be empty.")
    clean_id = session_id.strip()
    if not re.match(r"^[a-zA-Z0-9_-]+$", clean_id):
        raise ValueError(
            "Session ID contains invalid characters. Only alphanumeric, '-', and '_' are allowed."
        )
    return clean_id


def get_vectorstore_path(session_id: str | None = None) -> Path:
    """
    Resolve the storage directory for a session's FAISS index.
    """
    if session_id:
        clean_id = validate_session_id(session_id)
        return VECTOR_STORE_BASE_DIR / clean_id
    # Fallback path for backward compatibility
    return Path("data/faiss_index")


def has_vectorstore(session_id: str | None = None) -> bool:
    """
    Check if a saved FAISS index exists on disk for the given session.
    """
    path = get_vectorstore_path(session_id)
    return path.exists() and any(f.is_file() for f in path.iterdir())


def create_vectorstore(chunks):
    """
    Create an in-memory FAISS vector store from document chunks.
    """
    if not chunks:
        raise ValueError("No chunks provided.")

    embeddings = create_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_vectorstore(vectorstore, session_id: str | None = None):
    """
    Save the FAISS vector store into the session-specific directory.
    """
    target_path = get_vectorstore_path(session_id)
    target_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(target_path))
    logger.info("Vector store saved to %s", target_path)


def load_vectorstore(session_id: str | None = None):
    """
    Load the session-specific FAISS vector store from disk.
    """
    target_path = get_vectorstore_path(session_id)
    if not target_path.exists() or not any(target_path.iterdir()):
        raise FileNotFoundError(f"No vector store found at {target_path}")

    embeddings = create_embeddings()
    return FAISS.load_local(
        str(target_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def delete_vectorstore(session_id: str | None = None):
    """
    Delete the persisted FAISS vector store index from disk for the given session.
    """
    target_path = get_vectorstore_path(session_id)
    if target_path.exists():
        shutil.rmtree(target_path, ignore_errors=True)
        logger.info("Deleted vector store at %s", target_path)
