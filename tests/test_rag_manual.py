"""
test_rag_manual.py
------------------
Manual evaluation script for the RAG pipeline.

Purpose
-------
Verify that the RAG pipeline satisfies all quality requirements:

  TEST 1 — Direct question:          answer IS in the document
  TEST 2 — Cross-section question:   answer requires a different part of the document
  TEST 3 — Out-of-scope question:    answer is NOT in the document (safe refusal expected)
  TEST 4 — Follow-up question:       resolved using conversation context
  TEST 5 — Source question:          user asks for the source/origin of information

How to run
----------
From the project root:

    python tests/test_rag_manual.py

Read the printed output and manually verify:
  - Tests 1, 2, 5  →  relevant, grounded answer + correct source
  - Test 3         →  model says it cannot find the answer (no invented facts)
  - Test 4         →  follow-up resolved correctly using conversation history
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from backend.rag.document_loader import load_document
from backend.rag.chunks import split_documents
from backend.rag.vectorstore import create_vectorstore
from backend.rag.retriever import create_retriever
from backend.rag.generator import create_llm, generate_answer
from backend.rag.conversation import rewrite_question
from backend.rag.rag_chain import get_sources


# ---------------------------------------------------------------------------
# Sample document
# data/python.txt is the default test document.
# ---------------------------------------------------------------------------
SAMPLE_DOCUMENT = PROJECT_ROOT / "data" / "python.txt"


# ---------------------------------------------------------------------------
# Evaluation test cases (covers all 5 PART-3 requirements)
# ---------------------------------------------------------------------------
TEST_CASES = [
    {
        "id":       1,
        "type":     "Direct question",
        "expected": "Relevant answer + source cited",
        "question": "What is Python?",
        "history":  [],
    },
    {
        "id":       2,
        "type":     "Cross-section question",
        "expected": "Answer drawn from a different section of the document",
        "question": "Who created Python and when was it first released?",
        "history":  [],
    },
    {
        "id":       3,
        "type":     "Out-of-scope question",
        "expected": "Model says it cannot find the answer — NO invented facts",
        "question": "What is the capital of Mars?",
        "history":  [],
    },
    {
        "id":       4,
        "type":     "Follow-up question (conversation)",
        "expected": "System resolves 'it' from prior turn and answers about Python",
        "question": "What are its main use cases?",
        "history":  [
            {"role": "user",      "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level programming language."},
        ],
    },
    {
        "id":       5,
        "type":     "Source question",
        "expected": "Answer includes source file and page reference",
        "question": "What does the document say about Python's design philosophy?",
        "history":  [],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_separator(char="=", width=60):
    print("\n" + char * width)


def print_test_header(test):
    print_separator()
    print(f"TEST {test['id']} — {test['type']}")
    print(f"Expected : {test['expected']}")


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------

def setup_pipeline(document_path):
    """Load document, build FAISS index, return (retriever, llm)."""
    path = Path(document_path)

    if not path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Sample document not found: {path}\n"
            "\n"
            "  Place a document at the path above, or update\n"
            "  SAMPLE_DOCUMENT at the top of this script.\n"
            "\n"
            "  Do NOT create fake content just to pass the test."
        )

    print(f"[SETUP] Loading document  : {path.name}")
    documents = load_document(str(path))
    print(f"[SETUP] Pages loaded      : {len(documents)}")

    print("[SETUP] Splitting into chunks...")
    chunks = split_documents(documents)
    print(f"[SETUP] Chunks created    : {len(chunks)}")
    assert len(chunks) > 0, "No chunks were created — check the document."

    print("[SETUP] Building FAISS vector store...")
    vectorstore = create_vectorstore(chunks)
    print("[SETUP] Vector store ready.")

    retriever = create_retriever(vectorstore)
    print("[SETUP] MMR retriever ready.")

    print("[SETUP] Initialising LLM...")
    llm = create_llm()
    print("[SETUP] LLM ready.\n")

    return retriever, llm


# ---------------------------------------------------------------------------
# Single-test runner
# ---------------------------------------------------------------------------

def run_test(test, retriever, llm):
    """Run one test case and print the result."""
    print_test_header(test)

    question = test["question"]
    history  = test["history"]

    # For follow-up questions: rewrite using conversation history
    standalone = question
    if history:
        try:
            standalone = rewrite_question(llm, question, history)
            print(f"\nOriginal question : {question}")
            print(f"Rewritten         : {standalone}")
        except Exception as exc:
            print(f"[WARNING] Question rewrite failed: {exc} — using original.")

    try:
        # Retrieve
        docs = retriever.invoke(standalone)

        if not docs:
            print(f"\nQuestion          : {question}")
            print("Retrieved docs    : 0")
            print("[WARNING] No documents retrieved. Expected: safe refusal answer.")
        else:
            print(f"\nQuestion          : {question}")
            print(f"Retrieved docs    : {len(docs)}")

        # Generate grounded answer
        answer = generate_answer(llm, standalone, docs)
        print(f"\nAnswer:\n{answer}")

        # Show sources
        sources = get_sources(docs)
        print("\nSources:")
        if sources:
            for src in sources:
                name = Path(src.get("source") or "Unknown").name
                page = src.get("page")
                print(f"  • {name}" + (f"  (page {page})" if page is not None else ""))
        else:
            print("  (no source metadata)")

        # Grounding check hint
        if test["id"] == 3:
            print(
                "\n[CHECK] Above answer MUST say it could not find the information.\n"
                "        If it gives a real answer, the grounding prompt is failing."
            )

    except ValueError as err:
        print(f"[INPUT ERROR] {err}")
    except RuntimeError as err:
        print(f"[RUNTIME ERROR] {err}")
    except Exception as err:
        print(f"[UNEXPECTED ERROR] {type(err).__name__}: {err}")
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  AI Study Assistant — RAG Pipeline Evaluation")
    print("  Covers: direct, cross-section, out-of-scope,")
    print("          follow-up, and source questions")
    print("=" * 60)

    try:
        retriever, llm = setup_pipeline(SAMPLE_DOCUMENT)
    except FileNotFoundError as err:
        print(err)
        sys.exit(1)
    except Exception as err:
        print(f"\n[SETUP ERROR] {type(err).__name__}: {err}")
        raise

    for test in TEST_CASES:
        run_test(test, retriever, llm)

    print_separator()
    print("\n[DONE] Evaluation complete.")
    print(
        "\nManual review checklist"
        "\n-----------------------"
        "\n  TEST 1 (Direct)       → relevant answer + source"
        "\n  TEST 2 (Cross-section)→ relevant answer from a different section"
        "\n  TEST 3 (Out-of-scope) → safe refusal, no invented facts"
        "\n  TEST 4 (Follow-up)    → correctly resolved pronoun / reference"
        "\n  TEST 5 (Source)       → answer + source file shown"
    )


if __name__ == "__main__":
    main()
