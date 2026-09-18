def create_retriever(vectorstore):
    """Create an MMR retriever from a vector store."""

    if vectorstore is None:
        raise ValueError("Vector store is required.")

    try:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5,
            },
        )

        return retriever

    except Exception as error:
        raise RuntimeError(
            f"Failed to create retriever: {error}"
        ) from error


def retrieve_documents(retriever, question):
    """Retrieve relevant documents for a question."""

    if retriever is None:
        raise ValueError("Retriever is required.")

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    try:
        documents = retriever.invoke(question.strip())

        if not documents:
            return []

        return documents

    except Exception as error:
        raise RuntimeError(
            f"Document retrieval failed: {error}"
        ) from error


def search_with_scores(vectorstore, question, k=4):
    """Search documents and return their similarity scores."""

    if vectorstore is None:
        raise ValueError("Vector store is required.")

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    try:
        results = vectorstore.similarity_search_with_score(
            question.strip(),
            k=k
        )

        return results

    except Exception as error:
        raise RuntimeError(
            f"Similarity search failed: {error}"
        ) from error