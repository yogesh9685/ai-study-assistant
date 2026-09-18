import sys
from pathlib import Path

# Add project root to sys.path so `backend.*` imports resolve correctly
# when this script is run directly (e.g. python tests/test.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag.document_loader import load_document
from backend.rag.chunks import split_documents
from backend.rag.vectorstore import create_vectorstore
from backend.rag.retriever import create_retriever
from backend.rag.generator import (
    create_llm,
    generate_answer,
)

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "python.txt"

documents = load_document(str(DATA_FILE))

chunks = split_documents(documents)

vectorstore = create_vectorstore(chunks)

retriever = create_retriever(vectorstore)

question = "What is Python?"

results = retriever.invoke(question)

llm = create_llm()

answer = generate_answer(
    llm,
    question,
    results
)

print("\nAnswer:")
print(answer)