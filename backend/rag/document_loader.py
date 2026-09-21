"""
document_loader.py
------------------
Load study documents from disk into LangChain Document objects.

Supported formats: .pdf  .txt  .docx  .md  .csv
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = {".pdf", ".txt", ".docx", ".md", ".csv"}


def load_document(file_path):
    """
    Load a document from disk and return a list of LangChain Document objects.

    Parameters
    ----------
    file_path : str or Path

    Returns
    -------
    list[Document]

    Raises
    ------
    FileNotFoundError : file does not exist
    ValueError        : file is empty, unsupported type, or has no readable content
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError("File not found.")

    if path.stat().st_size == 0:
        raise ValueError("File is empty.")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file type: {extension}")

    logger.info("Loading document: %s (%s)", path.name, extension)

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))
        documents = loader.load()

    elif extension == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
        documents = loader.load()

    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))
        documents = loader.load()

    elif extension == ".md":
        try:
            loader = UnstructuredMarkdownLoader(str(path))
            documents = loader.load()
        except Exception:
            # Fall back to plain text loading if unstructured is unavailable
            loader = TextLoader(str(path), encoding="utf-8")
            documents = loader.load()

    else:  # .csv
        loader = CSVLoader(str(path))
        documents = loader.load()

    if not documents:
        raise ValueError("No readable content found in the document.")

    logger.info("Loaded %d page(s) from %s.", len(documents), path.name)
    return documents