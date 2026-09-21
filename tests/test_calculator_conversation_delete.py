"""
test_calculator_conversation_delete.py
--------------------------------------
Verification suite for:
  1. Calculator tool in conversation (via /ask)
  2. Multi-turn math in conversation
  3. Document upload & search
  4. Document deletion (DELETE /documents)
  5. Post-deletion calculation (calculator still works)
  6. Post-deletion document query (graceful message, no crash)
"""

import sys
from pathlib import Path

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
    print("  Testing Calculator, Conversation & Document Deletion")
    print("=" * 60)

    session_id = "test_conv_calc_123"

    # 1. Test Calculator tool via /ask
    print("\n[1] Testing Calculator tool via POST /ask ('What is 25 * 18?')...")
    resp = client.post("/ask", json={"question": "What is 25 * 18?", "session_id": session_id})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    safe_print(f" Answer: {data.get('answer')}")
    assert "450" in data.get("answer", ""), f"Expected 450 in answer, got: {data.get('answer')}"
    assert data.get("sources") == [], f"Expected empty sources, got: {data.get('sources')}"
    print(" -> PASS: Calculator tool was executed via /ask and returned 450!")

    # 2. Test Multi-turn conversation calculation via /ask
    print("\n[2] Testing Multi-turn conversation via POST /ask ('Multiply that by 2')...")
    resp = client.post("/ask", json={"question": "Multiply that by 2", "session_id": session_id})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    safe_print(f" Answer: {data.get('answer')}")
    assert "900" in data.get("answer", ""), f"Expected 900 in answer, got: {data.get('answer')}"
    print(" -> PASS: Multi-turn conversation resolved 'that' and computed 900!")

    # 3. Test Document upload
    print("\n[3] Testing POST /upload with study document...")
    doc_content = b"Python was created by Guido van Rossum and released in 1991."
    resp = client.post("/upload", files={"file": ("test_guido.txt", doc_content, "text/plain")})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    print(" -> PASS: Document uploaded and indexed successfully.")

    # 4. Test Document search via /ask
    print("\n[4] Testing Document search via POST /ask ('Who created Python?')...")
    resp = client.post("/ask", json={"question": "Who created Python according to the document?", "session_id": session_id})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    safe_print(f" Answer: {data.get('answer')}")
    safe_print(f" Sources: {data.get('sources')}")
    assert "Guido" in data.get("answer", ""), f"Expected Guido in answer, got: {data.get('answer')}"
    assert len(data.get("sources", [])) > 0, "Expected sources for document search"
    print(" -> PASS: Document search retrieved document and returned sources!")

    # 5. Test Document deletion via DELETE /documents
    print("\n[5] Testing DELETE /documents...")
    resp = client.delete("/documents")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    faiss_dir = Path("data/faiss_index")
    assert not faiss_dir.exists() or not any(faiss_dir.iterdir()), "Expected faiss_index to be deleted"
    print(" -> PASS: Vector store and uploaded documents deleted!")

    # 6. Test Calculator tool works AFTER document deletion
    print("\n[6] Testing Calculator tool after document deletion ('What is 15 + 35?')...")
    session_post_del = "test_conv_post_del_456"
    resp = client.post("/ask", json={"question": "What is 15 + 35?", "session_id": session_post_del})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    safe_print(f" Answer: {data.get('answer')}")
    assert "50" in data.get("answer", ""), f"Expected 50 in answer, got: {data.get('answer')}"
    print(" -> PASS: Calculator works smoothly after document deletion!")

    # 7. Test Document search AFTER document deletion (graceful response, no crash)
    print("\n[7] Testing Document query after document deletion ('What is Python according to the document?')...")
    resp = client.post("/ask", json={"question": "What is Python according to the document?", "session_id": session_post_del})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    safe_print(f" Answer: {data.get('answer')}")
    print(" -> PASS: Returned graceful response without crashing!")

    print("\n" + "=" * 60)
    print("  ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
