"""
test_fastapi_manual.py
----------------------
Comprehensive manual verification suite for FastAPI endpoints:
  - GET /health
  - GET /docs
  - POST /upload (validation, formats, vectorstore creation)
  - POST /chat (validation, calculator tool, document search tool, conversation)
"""

import sys
import io
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def safe_print(text: str):
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))

def run_tests():
    print("=" * 60)
    print("  FastAPI Manual Verification Suite")
    print("=" * 60)

    # 1. Health endpoint
    print("\n[1] Testing GET /health...")
    resp = client.get("/health")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert data.get("status") == "ok", f"Expected status 'ok', got {data}"
    print(" -> PASS: /health returned status: ok")

    # 2. Swagger docs
    print("\n[2] Testing GET /docs...")
    resp = client.get("/docs")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    print(" -> PASS: /docs is accessible")

    # 3. Upload Validation: Unsupported extension
    print("\n[3] Testing POST /upload with unsupported extension (.exe)...")
    resp = client.post(
        "/upload",
        files={"file": ("malicious.exe", b"MZDummyExecutable", "application/octet-stream")}
    )
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
    assert "Unsupported file type" in resp.json().get("detail", "")
    print(f" -> PASS: Rejected with 400: {resp.json()['detail']}")

    # 4. Upload Validation: Empty file
    print("\n[4] Testing POST /upload with empty file...")
    resp = client.post(
        "/upload",
        files={"file": ("empty.txt", b"", "text/plain")}
    )
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
    assert "Uploaded file is empty" in resp.json().get("detail", "")
    print(f" -> PASS: Rejected with 400: {resp.json()['detail']}")

    # 5. Upload Valid Text Document & Vectorstore Indexing
    print("\n[5] Testing POST /upload with valid .txt document...")
    doc_content = (
        "Python is a high-level, general-purpose programming language. "
        "Its design philosophy emphasizes code readability with the use of significant indentation. "
        "Python is dynamically-typed and garbage-collected. "
        "It supports multiple programming paradigms, including structured, object-oriented and functional programming."
    ).encode("utf-8")
    resp = client.post(
        "/upload",
        files={"file": ("python_study_notes.txt", doc_content, "text/plain")}
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    upload_data = resp.json()
    assert upload_data.get("chunks", 0) > 0, "Expected chunks > 0"
    print(f" -> PASS: Uploaded {upload_data['filename']} with {upload_data['chunks']} chunks")

    # 6. Upload Valid Markdown Document
    print("\n[6] Testing POST /upload with valid .md document...")
    md_content = b"# Study Guide\n\nFastAPI is a modern web framework for Python."
    resp = client.post(
        "/upload",
        files={"file": ("study_guide.md", md_content, "text/markdown")}
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    print(f" -> PASS: Uploaded {resp.json()['filename']} with {resp.json()['chunks']} chunks")

    # 7. Upload Valid CSV Document
    print("\n[7] Testing POST /upload with valid .csv document...")
    csv_content = b"topic,description\nRAG,Retrieval Augmented Generation\nAgent,Tool calling LLM\n"
    resp = client.post(
        "/upload",
        files={"file": ("study_topics.csv", csv_content, "text/csv")}
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    print(f" -> PASS: Uploaded {resp.json()['filename']} with {resp.json()['chunks']} chunks")

    # 8. Chat Validation: Empty question
    print("\n[8] Testing POST /chat with empty question...")
    resp = client.post("/chat", json={"question": ""})
    assert resp.status_code in (400, 422), f"Expected 400 or 422, got {resp.status_code}"
    print(f" -> PASS: Empty question rejected with status {resp.status_code}")

    # 9. Chat Validation: Whitespace question
    print("\n[9] Testing POST /chat with whitespace question...")
    resp = client.post("/chat", json={"question": "   "})
    assert resp.status_code in (400, 422), f"Expected 400 or 422, got {resp.status_code}"
    print(f" -> PASS: Whitespace question rejected with status {resp.status_code}")

    # 10. Chat: Calculator Tool
    print("\n[10] Testing POST /chat with calculator query (25 * 18)...")
    resp = client.post("/chat", json={"question": "What is 25 * 18?"})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    calc_data = resp.json()
    safe_print(f" Answer: {calc_data.get('answer')}")
    assert "450" in calc_data.get("answer", ""), "Expected 450 in answer"
    assert calc_data.get("sources") == [], f"Expected empty sources for math, got {calc_data.get('sources')}"
    print(" -> PASS: Calculator query returned 450 and empty sources []")

    # 11. Chat: Document Search Tool
    print("\n[11] Testing POST /chat with document query...")
    resp = client.post("/chat", json={"question": "What is Python according to the document?"})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    doc_data = resp.json()
    safe_print(f" Answer: {doc_data.get('answer')}")
    safe_print(f" Sources: {doc_data.get('sources')}")
    assert len(doc_data.get("sources", [])) > 0, "Expected sources from document search"
    print(f" -> PASS: Document query returned answer and {len(doc_data['sources'])} source(s)")

    # 12. Chat: Conversational query (no tools)
    print("\n[12] Testing POST /chat with conversational query ('Hello')...")
    resp = client.post("/chat", json={"question": "Hello"})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    conv_data = resp.json()
    safe_print(f" Answer: {conv_data.get('answer')}")
    assert conv_data.get("sources") == [], f"Expected empty sources for greeting, got {conv_data.get('sources')}"
    print(" -> PASS: Conversational query returned direct response and empty sources []")

    print("\n" + "=" * 60)
    print("  ALL API TESTS PASSED SUCCESSFULLY! (12/12)")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
