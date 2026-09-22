"""
test_hybrid_search_reranking.py
-------------------------------
Verification suite for Hybrid Search (BM25 + FAISS) and Cross-Encoder Reranking.

Checks:
  1. FAISS semantic retrieval (semantic similarity).
  2. BM25 keyword retrieval (exact term matching).
  3. Hybrid search combines candidates from both result sets.
  4. Duplicate chunks are identified and removed.
  5. Cross-Encoder reranker prioritizes the most query-relevant chunk.
  6. Exactly FINAL_TOP_K documents are returned.
  7. Verification with 'Python was created by Guido van Rossum.' / 'Who created Python?'.
  8. Keyword vs. semantic divergence test (e.g. acronyms/identifiers vs concepts).
  9. Edge cases: empty query, empty docs, non-matching query.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from backend.rag.hybrid_search import (
    bm25_search,
    faiss_search,
    deduplicate_documents,
    hybrid_search,
    build_bm25_index,
    reset_bm25_cache,
    HYBRID_TOP_K,
)
from backend.rag.reranker import (
    rerank_documents,
    get_reranker,
    RERANKER_MODEL,
    FINAL_TOP_K,
)
from backend.rag.vectorstore import create_vectorstore
from backend.rag.retriever import retrieve_hybrid_and_rerank, retrieve_documents


def test_bm25_keyword_search():
    print("\n--- Test 1: BM25 Keyword Search ---")
    chunks = [
        Document(page_content="Python is a high-level programming language.", metadata={"source": "test.txt", "page": 1}),
        Document(page_content="Machine learning models require training data and optimization.", metadata={"source": "test.txt", "page": 2}),
        Document(page_content="Error code ERR_SOCKET_TIMEOUT occurred during connection.", metadata={"source": "test.txt", "page": 3}),
    ]

    # Exact rare keyword match
    results = bm25_search("ERR_SOCKET_TIMEOUT", chunks=chunks, k=2)
    assert len(results) >= 1, "BM25 should return at least one match"
    assert "ERR_SOCKET_TIMEOUT" in results[0].page_content, f"Expected error code chunk, got: {results[0].page_content}"
    print(f" -> PASS: BM25 matched exact keyword: '{results[0].page_content[:40]}...'")


def test_faiss_semantic_search():
    print("\n--- Test 2: FAISS Semantic Search ---")
    chunks = [
        Document(page_content="Cats and dogs are common domestic pets.", metadata={"source": "animals.txt", "page": 1}),
        Document(page_content="Vehicles like cars and trucks operate on roadways.", metadata={"source": "transport.txt", "page": 1}),
        Document(page_content="Fresh apples, oranges, and bananas provide essential vitamins.", metadata={"source": "nutrition.txt", "page": 1}),
    ]
    vs = create_vectorstore(chunks)

    # Paraphrased query with no overlapping words
    results = faiss_search(vs, "canines and felines in homes", k=1)
    assert len(results) > 0
    assert "domestic pets" in results[0].page_content
    print(f" -> PASS: FAISS semantically matched: '{results[0].page_content}'")
    return vs, chunks


def test_hybrid_combination_and_deduplication():
    print("\n--- Test 3 & 4: Hybrid Search Combination and Deduplication ---")
    doc_a = Document(page_content="Python was conceived in the late 1980s by Guido van Rossum.", metadata={"source": "py.txt", "page": 1})
    doc_b = Document(page_content="CPython is the reference implementation of Python.", metadata={"source": "py.txt", "page": 2})
    doc_c = Document(page_content="Java was developed by James Gosling at Sun Microsystems.", metadata={"source": "java.txt", "page": 1})

    chunks = [doc_a, doc_b, doc_c]
    reset_bm25_cache()
    vs = create_vectorstore(chunks)

    # Hybrid search
    candidates = hybrid_search(vs, "Who created Python?", hybrid_top_k=2)
    assert len(candidates) > 0

    # Ensure no duplicates in candidate list
    contents = [doc.page_content for doc in candidates]
    assert len(contents) == len(set(contents)), "Candidates must not contain duplicates"

    # Test deduplicate_documents explicitly
    duplicated = [doc_a, doc_b, doc_a, doc_b, doc_c]
    deduped = deduplicate_documents(duplicated)
    assert len(deduped) == 3, f"Expected 3 unique docs, got {len(deduped)}"
    print(" -> PASS: Hybrid search successfully combined and deduplicated candidates.")


def test_cross_encoder_reranker():
    print("\n--- Test 5: Cross-Encoder Reranker Reordering ---")
    query = "Who created Python?"

    doc_weak = Document(page_content="Python supports multiple programming paradigms including OOP.", metadata={"source": "py.txt", "page": 1})
    doc_strong = Document(page_content="Python was created by Guido van Rossum and released in 1991.", metadata={"source": "py.txt", "page": 2})
    doc_irrelevant = Document(page_content="The weather in Paris is sunny today.", metadata={"source": "weather.txt", "page": 1})

    # Pass candidates with the strong match NOT in the first position
    candidates = [doc_weak, doc_irrelevant, doc_strong]

    reranked = rerank_documents(query, candidates, top_k=FINAL_TOP_K)
    assert len(reranked) == 3
    # The strong document answering "Who created Python?" must be reranked to position 0
    assert "Guido van Rossum" in reranked[0].page_content, (
        f"Expected top document to be Guido van Rossum, got: {reranked[0].page_content}"
    )
    print(f" -> PASS: Cross-Encoder placed most relevant chunk at rank 1: '{reranked[0].page_content[:50]}...'")


def test_final_top_k_cutoff():
    print("\n--- Test 6: FINAL_TOP_K Output Constraint ---")
    query = "programming languages"
    docs = [
        Document(page_content=f"Language number {i} is used for software development.", metadata={"source": f"doc_{i}.txt"})
        for i in range(10)
    ]

    reranked = rerank_documents(query, docs, top_k=FINAL_TOP_K)
    assert len(reranked) == FINAL_TOP_K, f"Expected exactly {FINAL_TOP_K} docs, got {len(reranked)}"
    print(f" -> PASS: Candidate pool of {len(docs)} correctly truncated to FINAL_TOP_K = {FINAL_TOP_K}")


def test_guido_example():
    print("\n--- Test 7: User-Specified Guido van Rossum Example ---")
    doc_text = "Python was created by Guido van Rossum."
    chunks = [
        Document(page_content="JavaScript is primarily used for web browser scripting.", metadata={"source": "js.txt", "page": 1}),
        Document(page_content=doc_text, metadata={"source": "py.txt", "page": 1}),
        Document(page_content="Rust guarantees memory safety without a garbage collector.", metadata={"source": "rust.txt", "page": 1}),
    ]
    reset_bm25_cache()
    vs = create_vectorstore(chunks)

    final_docs = retrieve_hybrid_and_rerank(vs, "Who created Python?", final_top_k=2)
    assert len(final_docs) >= 1
    assert "Guido van Rossum" in final_docs[0].page_content
    print(f" -> PASS: Top retrieved & reranked document: '{final_docs[0].page_content}'")


def test_keyword_vs_semantic_contribution():
    print("\n--- Test 8: Keyword (BM25) vs Semantic (FAISS) Divergence ---")
    # doc1 has exact keyword match for an uncommon identifier "SPEC_XYZ_99"
    # doc2 has semantic similarity to "how to format source code" without exact terms
    doc1 = Document(
        page_content="Error code SPEC_XYZ_99 indicates memory allocation failure on port 8080.",
        metadata={"source": "errors.txt", "page": 1},
    )
    doc2 = Document(
        page_content="Guidelines for structuring clean and maintainable software routines.",
        metadata={"source": "style.txt", "page": 1},
    )
    chunks = [doc1, doc2]
    reset_bm25_cache()
    vs = create_vectorstore(chunks)

    # 1. Query with exact keyword that semantic models may not have seen
    bm25_res = bm25_search("SPEC_XYZ_99", chunks=chunks, k=1)
    assert len(bm25_res) == 1 and "SPEC_XYZ_99" in bm25_res[0].page_content
    print(" -> BM25 successfully captured exact keyword 'SPEC_XYZ_99'")

    # 2. Query with conceptual phrasing
    faiss_res = faiss_search(vs, "how to write clean code routines", k=1)
    assert len(faiss_res) == 1 and "clean and maintainable" in faiss_res[0].page_content
    print(" -> FAISS successfully captured semantic concept 'how to write clean code routines'")

    # 3. Hybrid search combines both into candidates
    hybrid_res = hybrid_search(vs, "SPEC_XYZ_99 guidelines", hybrid_top_k=2)
    assert len(hybrid_res) == 2
    print(" -> Hybrid search successfully brought in both keyword and semantic matches")


def test_edge_cases():
    print("\n--- Test 9: Edge Cases and Graceful Fallback ---")
    # Empty query
    assert rerank_documents("", []) == []
    assert hybrid_search(None, "") == []
    assert bm25_search("", chunks=[]) == []

    # Non-matching query
    doc = Document(page_content="Sunflowers grow in summer fields.", metadata={"source": "nature.txt"})
    bm25_empty = bm25_search("cryptocurrency blockchain", chunks=[doc])
    assert bm25_empty == [], "BM25 should return empty list when no terms match"

    print(" -> PASS: Handled empty queries, empty chunks, and non-matching terms gracefully.")


def main():
    print("=" * 65)
    print("  AI Study Assistant — Hybrid Search & Reranking Test Suite")
    print("=" * 65)

    test_bm25_keyword_search()
    test_faiss_semantic_search()
    test_hybrid_combination_and_deduplication()
    test_cross_encoder_reranker()
    test_final_top_k_cutoff()
    test_guido_example()
    test_keyword_vs_semantic_contribution()
    test_edge_cases()

    print("\n" + "=" * 65)
    print("  ALL HYBRID SEARCH & RERANKING TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
