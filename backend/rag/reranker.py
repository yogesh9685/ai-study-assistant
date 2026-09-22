"""
reranker.py
-----------
Cross-Encoder reranking module for the RAG pipeline.

Scores candidate query-document pairs jointly with a pretrained CrossEncoder
model and reorders documents so that the most semantically relevant chunks
are positioned at the top before being passed to the LLM.

The CrossEncoder model is loaded once and cached (singleton pattern) to prevent
expensive re-initialization on every query.
"""

import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Configurable constants in one place
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FINAL_TOP_K = 4

# Module-level singleton instance — loaded once and reused
_reranker_instance = None


def get_reranker(model_name: str = RERANKER_MODEL):
    """
    Return the shared CrossEncoder instance, initializing it on the first call.
    """
    global _reranker_instance
    if _reranker_instance is None:
        logger.info("Initializing CrossEncoder model: %s (first time only)", model_name)
        try:
            from sentence_transformers import CrossEncoder
            _reranker_instance = CrossEncoder(model_name)
            logger.info("CrossEncoder model loaded and cached successfully.")
        except Exception as error:
            logger.error("Failed to load CrossEncoder model %s: %s", model_name, error)
            raise RuntimeError(f"Could not initialize reranker model: {error}") from error
    return _reranker_instance


def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int = FINAL_TOP_K,
) -> list[Document]:
    """
    Score and reorder candidate documents for a given query using the Cross-Encoder.

    Parameters
    ----------
    query : str
        The user question.
    documents : list[Document]
        Candidate documents retrieved from hybrid search.
    top_k : int, default FINAL_TOP_K
        Number of top reranked documents to return.

    Returns
    -------
    list[Document]
        Top-k documents ordered by descending relevance score.
    """
    if not documents:
        return []

    if not query or not query.strip():
        logger.warning("Empty query provided for reranking. Returning candidate documents unchanged.")
        return documents[:top_k]

    if len(documents) <= 1:
        return documents[:top_k]

    try:
        reranker = get_reranker()

        # CrossEncoder expects pairs of [query, text]
        pairs = [[query.strip(), doc.page_content] for doc in documents]
        scores = reranker.predict(pairs)

        # Pair each document with its score and sort descending
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda item: item[1], reverse=True)

        reranked = [doc for doc, _ in scored_docs[:top_k]]
        logger.info(
            "Reranked %d candidate documents down to top %d.",
            len(documents),
            len(reranked),
        )
        return reranked

    except Exception as error:
        logger.error("Reranking failed (%s). Falling back to candidate documents.", error)
        # Fallback gracefully to candidate documents if reranking fails
        return documents[:top_k]
