"""
test_session_isolation.py
-------------------------
Multi-user document isolation and security verification suite:
  TEST 1: User/session A uploads doc_A. A sees doc_A, asks question about doc_A.
  TEST 2: User/session B uploads doc_B. B sees doc_B, asks question about doc_B.
  TEST 3: A cannot see doc_B via /status.
  TEST 4: B cannot see doc_A via /status.
  TEST 5: A's question cannot retrieve chunks or secrets from B's document.
  TEST 6: B's question cannot retrieve chunks or secrets from A's document.
  TEST 7: Conversation memory remains isolated between A and B.
  TEST 8: Security & validation: empty session_id and path-traversal attempts rejected.
  TEST 9: Document deletion isolation: clearing A does not impact B.
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


def run_isolation_tests():
    print("=" * 70)
    print("  RUNNING MULTI-USER DOCUMENT ISOLATION TEST SUITE")
    print("=" * 70)

    session_a = "user_session_alpha_101"
    session_b = "user_session_beta_202"

    # Clean up before testing
    client.delete("/documents", params={"session_id": session_a})
    client.delete("/documents", params={"session_id": session_b})

    # -------------------------------------------------------------------------
    # TEST 1: User A uploads A.txt, can see A.txt, asks question about A.txt
    # -------------------------------------------------------------------------
    print("\n[TEST 1] User A uploads doc_a.txt...")
    content_a = b"PROJECT ALPHA SECRET CODE: ALPHA-998877. The project manager is Alice."
    resp_upload_a = client.post(
        "/upload",
        files={"file": ("doc_a.txt", content_a, "text/plain")},
        data={"session_id": session_a},
    )
    assert resp_upload_a.status_code == 200, f"A upload failed: {resp_upload_a.text}"
    print(" -> A uploaded doc_a.txt successfully.")

    # A checks status
    status_a = client.get("/status", params={"session_id": session_a}).json()
    assert status_a["has_document"] is True
    assert status_a["filename"] == "doc_a.txt"
    print(f" -> A status confirms document: {status_a['filename']}")

    # A asks question about A's doc
    ask_a = client.post(
        "/ask",
        json={"question": "What is the secret code for Project Alpha according to the document?", "session_id": session_a},
    ).json()
    safe_print(f" -> A Answer: {ask_a['answer']}")
    assert "998877" in ask_a["answer"], f"Expected ALPHA-998877 in answer, got: {ask_a['answer']}"
    assert any("doc_a.txt" in s.get("source", "") for s in ask_a.get("sources", []))
    print(" -> PASS: User A can see and query A's document!")

    # -------------------------------------------------------------------------
    # TEST 2: User B uploads B.txt, can see B.txt, asks question about B.txt
    # -------------------------------------------------------------------------
    print("\n[TEST 2] User B uploads doc_b.txt...")
    content_b = b"PROJECT BETA SECRET CODE: BETA-443322. The lead architect is Bob."
    resp_upload_b = client.post(
        "/upload",
        files={"file": ("doc_b.txt", content_b, "text/plain")},
        data={"session_id": session_b},
    )
    assert resp_upload_b.status_code == 200, f"B upload failed: {resp_upload_b.text}"
    print(" -> B uploaded doc_b.txt successfully.")

    # B checks status
    status_b = client.get("/status", params={"session_id": session_b}).json()
    assert status_b["has_document"] is True
    assert status_b["filename"] == "doc_b.txt"
    print(f" -> B status confirms document: {status_b['filename']}")

    # B asks question about B's doc
    ask_b = client.post(
        "/ask",
        json={"question": "What is the secret code for Project Beta according to the document?", "session_id": session_b},
    ).json()
    safe_print(f" -> B Answer: {ask_b['answer']}")
    assert "443322" in ask_b["answer"], f"Expected BETA-443322 in answer, got: {ask_b['answer']}"
    assert any("doc_b.txt" in s.get("source", "") for s in ask_b.get("sources", []))
    print(" -> PASS: User B can see and query B's document!")

    # -------------------------------------------------------------------------
    # TEST 3: A cannot see doc_b.txt
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Verifying User A cannot see doc_b.txt via /status...")
    status_a_now = client.get("/status", params={"session_id": session_a}).json()
    assert status_a_now["filename"] == "doc_a.txt", f"A saw unexpected file: {status_a_now['filename']}"
    assert "doc_b" not in status_a_now["filename"]
    print(" -> PASS: User A only sees doc_a.txt, NOT doc_b.txt.")

    # -------------------------------------------------------------------------
    # TEST 4: B cannot see doc_a.txt
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Verifying User B cannot see doc_a.txt via /status...")
    status_b_now = client.get("/status", params={"session_id": session_b}).json()
    assert status_b_now["filename"] == "doc_b.txt", f"B saw unexpected file: {status_b_now['filename']}"
    assert "doc_a" not in status_b_now["filename"]
    print(" -> PASS: User B only sees doc_b.txt, NOT doc_a.txt.")

    # -------------------------------------------------------------------------
    # TEST 5: A's question cannot retrieve chunks from B's document
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Testing cross-session retrieval: A asks about Beta secret...")
    ask_a_cross = client.post(
        "/ask",
        json={"question": "What is the secret code for Project Beta according to the document?", "session_id": session_a},
    ).json()
    safe_print(f" -> A response for Beta question: {ask_a_cross['answer']}")
    assert "443322" not in ask_a_cross["answer"], "DATA LEAK: User A retrieved User B's secret code!"
    for s in ask_a_cross.get("sources", []):
        assert "doc_b.txt" not in s.get("source", ""), "DATA LEAK: User B's source returned to User A!"
    print(" -> PASS: User A cannot retrieve any content or sources from User B's document.")

    # -------------------------------------------------------------------------
    # TEST 6: B's question cannot retrieve chunks from A's document
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Testing cross-session retrieval: B asks about Alpha secret...")
    ask_b_cross = client.post(
        "/ask",
        json={"question": "What is the secret code for Project Alpha according to the document?", "session_id": session_b},
    ).json()
    safe_print(f" -> B response for Alpha question: {ask_b_cross['answer']}")
    assert "998877" not in ask_b_cross["answer"], "DATA LEAK: User B retrieved User A's secret code!"
    for s in ask_b_cross.get("sources", []):
        assert "doc_a.txt" not in s.get("source", ""), "DATA LEAK: User A's source returned to User B!"
    print(" -> PASS: User B cannot retrieve any content or sources from User A's document.")

    # -------------------------------------------------------------------------
    # TEST 7: Existing conversation memory still works separately for A and B
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Testing conversation memory isolation between A and B...")
    # Tell A something personal
    client.post("/ask", json={"question": "My favorite fruit is Mango.", "session_id": session_a})
    # Tell B something different
    client.post("/ask", json={"question": "My favorite fruit is Strawberry.", "session_id": session_b})

    # Ask A what my favorite fruit is
    mem_a = client.post("/ask", json={"question": "What is my favorite fruit?", "session_id": session_a}).json()
    safe_print(f" -> A Memory Answer: {mem_a['answer']}")
    assert "mango" in mem_a["answer"].lower(), f"Expected mango in A's memory, got: {mem_a['answer']}"
    assert "strawberry" not in mem_a["answer"].lower(), "Strawberry leaked into A's memory!"

    # Ask B what my favorite fruit is
    mem_b = client.post("/ask", json={"question": "What is my favorite fruit?", "session_id": session_b}).json()
    safe_print(f" -> B Memory Answer: {mem_b['answer']}")
    assert "strawberry" in mem_b["answer"].lower(), f"Expected strawberry in B's memory, got: {mem_b['answer']}"
    assert "mango" not in mem_b["answer"].lower(), "Mango leaked into B's memory!"
    print(" -> PASS: Conversation memory is strictly isolated between Session A and Session B.")

    # -------------------------------------------------------------------------
    # TEST 8: Validation & Path Traversal Security
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Testing validation and path-traversal prevention...")
    # Empty session_id on upload
    resp_empty = client.post("/upload", files={"file": ("test.txt", b"content", "text/plain")}, data={"session_id": ""})
    assert resp_empty.status_code == 400, f"Expected 400 for empty session_id, got: {resp_empty.status_code}"

    # Path traversal session_id
    resp_traversal = client.post(
        "/upload",
        files={"file": ("test.txt", b"content", "text/plain")},
        data={"session_id": "../../etc"},
    )
    assert resp_traversal.status_code == 400, f"Expected 400 for path traversal session_id, got: {resp_traversal.status_code}"

    resp_traversal_ask = client.post("/ask", json={"question": "hello", "session_id": "../evil"})
    assert resp_traversal_ask.status_code == 400, f"Expected 400 for path traversal ask, got: {resp_traversal_ask.status_code}"

    # Unindexed session graceful response
    unindexed_session = "unindexed_user_999"
    resp_unindexed = client.post(
        "/ask",
        json={"question": "What does my document say?", "session_id": unindexed_session},
    )
    assert resp_unindexed.status_code == 200
    safe_print(f" -> Unindexed session query result: {resp_unindexed.json()['answer']}")
    print(" -> PASS: Security validation and unindexed session edge case handled properly.")

    # -------------------------------------------------------------------------
    # TEST 9: Deletion Isolation
    # -------------------------------------------------------------------------
    print("\n[TEST 9] Testing deletion isolation: Deleting A's documents does NOT delete B's documents...")
    del_a = client.delete("/documents", params={"session_id": session_a})
    assert del_a.status_code == 200

    status_a_after = client.get("/status", params={"session_id": session_a}).json()
    assert status_a_after["has_document"] is False, "A still has document after deletion!"

    status_b_after = client.get("/status", params={"session_id": session_b}).json()
    assert status_b_after["has_document"] is True, "B lost document when A deleted!"
    assert status_b_after["filename"] == "doc_b.txt", "B filename corrupted after A deletion!"
    print(" -> PASS: Deleting User A's data leaves User B completely intact.")

    # Cleanup B
    client.delete("/documents", params={"session_id": session_b})

    print("\n" + "=" * 70)
    print("  ALL 9 MULTI-USER ISOLATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_isolation_tests()
