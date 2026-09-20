"""
test_rag_manual.py
------------------
Branch: feature/rag-evaluation

Manual evaluation script for the existing RAG pipeline.

Purpose
-------
Verify that the RAG pipeline:
  1. Loads a document successfully.
  2. Splits the document into chunks.
  3. Creates a FAISS vector store from those chunks.
  4. Retrieves relevant chunks using the MMR retriever.
  5. Generates an answer from the retrieved context.
  6. Returns correct source information.
  7. Handles questions whose answers are NOT in the document safely
     (i.e. does not invent unsupported facts).

How to run
----------
From the project root:

    python tests/test_rag_manual.py

No automated scoring is used here.  Read the printed output and judge
manually whether the answers are correct and grounded in the document.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure `backend.*` imports resolve when the script is run directly
# from any working directory, e.g.:
#
#   python tests/test_rag_manual.py        (from project root)
#   python test_rag_manual.py              (from inside tests/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports from the existing RAG pipeline.
#
# Actual module locations in this project:
#   backend/rag/chunks.py      → split_documents
#   backend/rag/generator.py   → create_llm, generate_answer
#   backend/rag/rag_chain.py   → get_sources
# ---------------------------------------------------------------------------
from backend.rag.document_loader import load_document
from backend.rag.chunks import split_documents
from backend.rag.vectorstore import create_vectorstore
from backend.rag.retriever import create_retriever
from backend.rag.generator import create_llm, generate_answer
from backend.rag.rag_chain import get_sources


# ---------------------------------------------------------------------------
# Sample document
#
# data/python.txt is already present in this project and is used as the
# default test document.  Change SAMPLE_DOCUMENT if you want to test with
# a different file.
#
# Supported formats: .pdf  .txt  .docx  .md  .csv
# ---------------------------------------------------------------------------
SAMPLE_DOCUMENT = PROJECT_ROOT / "data" / "python.txt"


# ---------------------------------------------------------------------------
# Evaluation questions
#
# INSTRUCTIONS
# ------------
#   Question 1 & 2  →  Replace the placeholder strings with real questions
#                       whose answers actually exist inside SAMPLE_DOCUMENT.
#
#   Question 3      →  Keep as-is.  It is intentionally out-of-scope and
#                       checks that the model does NOT invent facts.
# ---------------------------------------------------------------------------
questions = [
    # [EDIT] Replace with a question whose answer IS in the document
    "what is python",

    # [EDIT] Replace with another question whose answer IS in the document
    "who create a python and its feature",

    # [KEEP] Out-of-scope safety test - do not change this one
    "What is the capital of Mars?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_separator(char="=", width=60):
    """Print a visual divider line."""
    print("\n" + char * width)


def validate_question(question):
    """
    Check whether a question is ready to be evaluated.

    Returns
    -------
    (is_valid : bool, error_message : str | None)
    """
    stripped = question.strip()

    if not stripped:
        return False, "Question is empty - skipping."

    if stripped.startswith("REPLACE_WITH"):
        return False, (
            f"Placeholder detected: '{stripped}'\n"
            "  -> Replace this string with a real question from your document."
        )

    return True, None


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------

def setup_pipeline(document_path):
    """
    Build the full RAG pipeline for evaluation.

    Steps
    -----
    1. Check the sample document exists.
    2. Load the document.
    3. Split into chunks.
    4. Build FAISS vector store.
    5. Create MMR retriever.
    6. Initialise the LLM.

    Returns
    -------
    (retriever, llm)
    """

    path = Path(document_path)

    # --- 1. Check the file exists -----------------------------------------
    if not path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Sample document not found: {path}\n"
            "\n"
            "  A real document is required to run this evaluation.\n"
            "  -> Place your file at the path shown above, or update\n"
            "     SAMPLE_DOCUMENT in this script to point to an existing file.\n"
            "\n"
            "  Do NOT create fake document content just to make the test pass."
        )

    # --- 2. Load ----------------------------------------------------------
    print(f"[SETUP] Loading document  : {path.name}")
    documents = load_document(str(path))
    print(f"[SETUP] Pages loaded      : {len(documents)}")

    # --- 3. Split ---------------------------------------------------------
    print("[SETUP] Splitting into chunks...")
    chunks = split_documents(documents)
    print(f"[SETUP] Chunks created    : {len(chunks)}")

    # --- 4. FAISS vector store --------------------------------------------
    print("[SETUP] Building FAISS vector store...")
    vectorstore = create_vectorstore(chunks)
    print("[SETUP] Vector store ready.")

    # --- 5. MMR retriever -------------------------------------------------
    retriever = create_retriever(vectorstore)
    print("[SETUP] MMR retriever ready.")

    # --- 6. LLM -----------------------------------------------------------
    print("[SETUP] Initialising LLM …")
    llm = create_llm()
    print("[SETUP] LLM ready.")

    return retriever, llm


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(retriever, llm, questions):
    """
    Iterate over the evaluation questions.

    For each question:
      • Validate it (skip placeholders and empty strings).
      • Retrieve relevant document chunks.
      • Generate an answer from those chunks.
      • Print the retrieved count, answer, and sources.
    """

    total = len(questions)

    for i, question in enumerate(questions, start=1):
        print_separator()
        print(f"Question {i} of {total}")

        # Validate --------------------------------------------------------
        is_valid, error_message = validate_question(question)
        if not is_valid:
            print(f"[SKIP] {error_message}")
            continue

        try:
            # Retrieve -----------------------------------------------------
            results = retriever.invoke(question)

            # Warn if nothing came back ------------------------------------
            if not results:
                print(f"\nQuestion:\n{question}")
                print("\nRetrieved documents: 0")
                print(
                    "\n[WARNING] The retriever returned no documents.\n"
                    "  -> The answer below is based on an empty context.\n"
                    "  -> Expected: the model says it cannot find the answer."
                )

            # Generate answer ----------------------------------------------
            answer = generate_answer(llm, question, results)

            # Collect sources ----------------------------------------------
            sources = get_sources(results)

            # Print results ------------------------------------------------
            print(f"\nQuestion:\n{question}")

            print(f"\nRetrieved documents: {len(results)}")

            print(f"\nAnswer:\n{answer}")

            print("\nSources:")
            if sources:
                for source in sources:
                    src_name = source.get("source", "Unknown")
                    page_num = source.get("page")
                    if page_num is not None:
                        print(f"  - {src_name}  (page {page_num})")
                    else:
                        print(f"  - {src_name}")
            else:
                print("  (no source metadata available)")

        except ValueError as error:
            # Bad input — e.g. empty question caught inside the pipeline
            print(f"[INPUT ERROR] {error}")

        except RuntimeError as error:
            # Retriever / vectorstore / LLM failures
            print(f"[RUNTIME ERROR] {error}")

        except Exception as error:
            # Unexpected errors — show the type so it is easy to debug
            print(f"[UNEXPECTED ERROR] {type(error).__name__}: {error}")
            raise  # re-raise so the full traceback is visible


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  RAG Pipeline - Manual Evaluation")
    print("  Branch : feature/rag-evaluation")
    print("=" * 60)

    # --- Build pipeline ---------------------------------------------------
    try:
        retriever, llm = setup_pipeline(SAMPLE_DOCUMENT)
    except FileNotFoundError as error:
        # Missing document: print a clear message and stop
        print(error)
        sys.exit(1)
    except Exception as error:
        print(f"\n[SETUP ERROR] {type(error).__name__}: {error}")
        raise

    # --- Run evaluation ---------------------------------------------------
    run_evaluation(retriever, llm, questions)

    # --- Summary ----------------------------------------------------------
    print_separator()
    print("\n[DONE] Manual evaluation complete.")
    print(
        "\nExpected behaviour\n"
        "------------------\n"
        "  Questions 1-2 (in-document)  ->  relevant answer + correct source.\n"
        "  Question  3   (out-of-scope) ->  model says it could not find the\n"
        "                                   answer; no invented facts."
    )


if __name__ == "__main__":
    main()
