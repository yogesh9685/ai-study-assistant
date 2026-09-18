from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)


SUPPORTED_TYPES = {
    ".pdf",
    ".txt",
    ".docx",
    ".md",
    ".csv",
}


def load_document(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError("File not found.")

    if path.stat().st_size == 0:
        raise ValueError("File is empty.")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file type: {extension}")

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))

    elif extension == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")

    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))

    elif extension == ".md":
        loader = UnstructuredMarkdownLoader(str(path))

    else:
        loader = CSVLoader(str(path))

    documents = loader.load()

    if not documents:
        raise ValueError("No readable content found.")

    return documents