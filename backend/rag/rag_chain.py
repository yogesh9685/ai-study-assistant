def get_sources(documents):
    sources = []
    seen = set()

    for document in documents:
        source = document.metadata.get("source")
        page = document.metadata.get("page")

        key = (source, page)

        if key not in seen:
            sources.append({
                "source": source,
                "page": page
            })
            seen.add(key)

    return sources


def run_conversational_rag(question: str, session_id: str) -> tuple[str, list]:
    """
    Full conversational RAG pipeline for a single turn.

    Steps:
      1. Load conversation history for this session
      2. Rewrite the question into a standalone question (resolves references)
      3. Retrieve relevant documents using the standalone question
      4. Generate a grounded answer from the documents
      5. Save both turns (user + assistant) to history
      6. Return the answer and its sources

    Parameters
    ----------
    question   : the raw user question (may contain references like "it", "they")
    session_id : unique identifier for this user's session

    Returns
    -------
    (answer: str, sources: list[dict])
    """
    from backend.rag.generator import create_llm, generate_answer
    from backend.rag.conversation import create_chat_history, add_message, rewrite_question
    from backend.rag.vectorstore import load_vectorstore
    from backend.rag.retriever import create_retriever, retrieve_documents
    from backend.rag.session_store import get_history, save_history
    from pathlib import Path

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    # 1. Load history for this session
    history = get_history(session_id)

    # Create LLM once — reused for both question rewriting and answer generation
    llm = create_llm()

    # 2. Rewrite follow-up question into a standalone question
    #    If history is empty, rewrite_question returns the original question unchanged
    try:
        standalone_question = rewrite_question(llm, question, history)
    except Exception:
        # If rewriting fails, fall back to the original question
        standalone_question = question

    # 3. Load vectorstore and retrieve relevant documents
    index_path = Path("data/faiss_index")
    if not index_path.exists() or not any(index_path.iterdir()):
        raise RuntimeError(
            "No vector store found. "
            "Please upload and index a document before asking questions."
        )

    vectorstore = load_vectorstore()
    retriever = create_retriever(vectorstore)
    documents = retrieve_documents(retriever, standalone_question)

    # 4. Generate grounded answer from retrieved documents
    answer = generate_answer(llm, standalone_question, documents)

    # 5. Extract sources from retrieved documents
    sources = get_sources(documents)

    # 6. Save this turn to conversation history
    history = add_message(history, "user", question)
    history = add_message(history, "assistant", answer)
    save_history(session_id, history)

    return answer, sources