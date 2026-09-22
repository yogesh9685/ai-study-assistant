"""
hybrid_search.py
----------------
Hybrid document retrieval combining BM25 keyword search and FAISS semantic search.

Flow:
  1. FAISS similarity search retrieves semantically similar document chunks.
  2. BM25 (rank-bm25) retrieves exact keyword / term-matching document chunks.
  3. Results are combined and deduplicated to produce a unified candidate pool.

The BM25 index is built from the exact same document chunks stored in FAISS and
is cached in memory so it is NOT rebuilt on every user query.
"""

import logging
import re
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Default configuration constants
HYBRID_TOP_K = 10

# Module-level cache for BM25 index and its underlying chunks
_cached_bm25 = None
_cached_chunks = None


def tokenize(text: str) -> list[str]:
    """
    Simple, fast tokenization for BM25: lowercase alphanumeric tokens.
    """
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


def get_chunks_from_vectorstore(vectorstore) -> list[Document]:
    """
    Extract all Document chunks stored in the FAISS vectorstore's in-memory docstore.
    """
    if vectorstore is None:
        return []

    docstore = getattr(vectorstore, "docstore", None)
    if docstore is None:
        return []

    doc_dict = getattr(docstore, "_dict", {})
    return list(doc_dict.values())


def reset_bm25_cache():
    """
    Clear the cached BM25 index and chunks.
    Called when documents are uploaded, replaced, or deleted.
    """
    global _cached_bm25, _cached_chunks
    _cached_bm25 = None
    _cached_chunks = None
    logger.info("BM25 cache cleared.")


def build_bm25_index(chunks: list[Document]) -> BM25Okapi:
    """
    Build a BM25Okapi index from a list of Document chunks.
    """
    if not chunks:
        raise ValueError("Cannot build BM25 index from empty chunks.")

    corpus = [tokenize(doc.page_content) for doc in chunks]
    return BM25Okapi(corpus)


def get_or_build_bm25(vectorstore=None, chunks=None) -> tuple[BM25Okapi | None, list[Document]]:
    """
    Retrieve the cached BM25 index, or build it once and cache it.
    Reuses chunks from the provided vectorstore or chunk list.
    """
    global _cached_bm25, _cached_chunks

    if _cached_bm25 is not None and _cached_chunks is not None:
        return _cached_bm25, _cached_chunks

    if chunks is None and vectorstore is not None:
        chunks = get_chunks_from_vectorstore(vectorstore)

    if not chunks:
        logger.warning("No chunks available to build BM25 index.")
        return None, []

    logger.info("Building BM25 index for %d chunks (first time or after reset)...", len(chunks))
    try:
        bm25 = build_bm25_index(chunks)
        _cached_bm25 = bm25
        _cached_chunks = chunks
        logger.info("BM25 index built and cached successfully.")
        return _cached_bm25, _cached_chunks
    except Exception as error:
        logger.error("Failed to build BM25 index: %s", error)
        return None, []


def bm25_search(
    query: str,
    vectorstore=None,
    chunks=None,
    k: int = HYBRID_TOP_K,
) -> list[Document]:
    """
    Retrieve top-k chunks using BM25 keyword matching.
    """
    if not query or not query.strip():
        return []

    bm25, doc_chunks = get_or_build_bm25(vectorstore=vectorstore, chunks=chunks)
    if bm25 is None or not doc_chunks:
        return []

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)

    # Match chunks that have positive BM25 score or share keyword tokens with the query
    # (In small corpora, Robertson BM25 IDF evaluates to 0 when terms appear in >= N/2 docs)
    query_terms = set(tokenized_query)
    matching_chunks = []
    for doc, score in zip(doc_chunks, scores):
        doc_terms = set(tokenize(doc.page_content))
        overlap = len(query_terms.intersection(doc_terms))
        if score > 0.0 or overlap > 0:
            effective_score = float(score) if score > 0.0 else (overlap * 0.1)
            matching_chunks.append((doc, effective_score))

    # Sort descending by score
    matching_chunks.sort(key=lambda item: item[1], reverse=True)

    results = [doc for doc, _ in matching_chunks[:k]]
    logger.info("BM25 keyword search returned %d matching chunks.", len(results))
    return results


def faiss_search(
    vectorstore,
    query: str,
    k: int = HYBRID_TOP_K,
) -> list[Document]:
    """
    Retrieve top-k chunks using FAISS dense similarity search.
    """
    if vectorstore is None or not query or not query.strip():
        return []

    try:
        results = vectorstore.similarity_search(query.strip(), k=k)
        logger.info("FAISS semantic search returned %d candidate chunks.", len(results))
        return results
    except Exception as error:
        logger.error("FAISS similarity search failed: %s", error)
        return []


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    """
    Remove duplicate chunks while preserving the original candidate order.
    Deduplication is based on stripped text content, source file, and page.
    """
    unique_docs = []
    seen = set()

    for doc in documents:
        key = (
            doc.page_content.strip(),
            doc.metadata.get("source"),
            doc.metadata.get("page"),
        )
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs


def hybrid_search(
    vectorstore,
    query: str,
    hybrid_top_k: int = HYBRID_TOP_K,
    chunks: list[Document] | None = None,
) -> list[Document]:
    """
    Perform hybrid retrieval:
      1. Retrieve top candidates from FAISS (semantic).
      2. Retrieve top candidates from BM25 (keyword).
      3. Combine both candidate lists (interleaved) and remove duplicates.

    Parameters
    ----------
    vectorstore : FAISS
        Loaded FAISS vectorstore instance.
    query : str
        The user's search query.
    hybrid_top_k : int, default HYBRID_TOP_K (10)
        Number of candidates to request from each retriever before merging.
    chunks : list[Document], optional
        Explicit chunk list if available (otherwise pulled from vectorstore).

    Returns
    -------
    list[Document]
        Combined, deduplicated candidate documents.
    """
    if not query or not query.strip():
        logger.warning("Empty query provided to hybrid_search.")
        return []

    clean_query = query.strip()

    # 1. Semantic search via FAISS
    faiss_docs = faiss_search(vectorstore, clean_query, k=hybrid_top_k)

    # 2. Keyword search via BM25
    bm25_docs = bm25_search(clean_query, vectorstore=vectorstore, chunks=chunks, k=hybrid_top_k)

    # If both return empty
    if not faiss_docs and not bm25_docs:
        logger.warning("Both FAISS and BM25 returned 0 results for query: '%s'", clean_query)
        return []

    # 3. Interleave results so both keyword and semantic matches are represented fairly
    combined_candidates = []
    max_len = max(len(faiss_docs), len(bm25_docs))
    for i in range(max_len):
        if i < len(faiss_docs):
            combined_candidates.append(faiss_docs[i])
        if i < len(bm25_docs):
            combined_candidates.append(bm25_docs[i])

    # 4. Remove duplicate chunks
    unique_candidates = deduplicate_documents(combined_candidates)

    logger.info(
        "Hybrid search produced %d unique candidate chunks (FAISS: %d, BM25: %d).",
        len(unique_candidates),
        len(faiss_docs),
        len(bm25_docs),
    )

    return unique_candidates
