from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


def create_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME
        )

        test_vector = embeddings.embed_query("test")

        if not test_vector:
            raise ValueError("Embedding model returned an empty vector.")

        if len(test_vector) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION} dimensions, "
                f"got {len(test_vector)}."
            )

        return embeddings

    except Exception as error:
        raise RuntimeError(
            f"Failed to initialize embedding model: {error}"
        ) from error