"""
embeddings.py
-------------
HuggingFace embedding model initialization.

The embedding model is loaded ONCE and reused for every subsequent call
(singleton pattern). This avoids reloading the model from disk on every
upload and every question — which was the primary performance bottleneck.
"""

import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Module-level singleton — None until first call to create_embeddings()
_embeddings_instance = None


def create_embeddings():
    """
    Return the shared HuggingFace embedding model instance.

    The model is initialized on the first call and reused on every
    subsequent call. This eliminates the 3-5s model-loading overhead
    that was previously incurred on every upload and every question.
    """
    global _embeddings_instance

    if _embeddings_instance is not None:
        return _embeddings_instance

    logger.info("Loading embedding model: %s (first time only)", MODEL_NAME)

    try:
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

        test_vector = embeddings.embed_query("test")

        if not test_vector:
            raise ValueError("Embedding model returned an empty vector.")

        if len(test_vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION} dimensions, "
                f"got {len(test_vector)}."
            )

        _embeddings_instance = embeddings
        logger.info("Embedding model loaded and cached successfully.")
        return _embeddings_instance

    except Exception as error:
        raise RuntimeError(
            f"Failed to initialize embedding model: {error}"
        ) from error