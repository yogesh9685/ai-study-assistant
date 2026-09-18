from pathlib import Path
from langchain_community.vectorstores import FAISS
from backend.rag.embeddings import create_embeddings


VECTOR_STORE_PATH = "data/faiss_index"

#this is use to create vector databse
def create_vectorstore(chunks):
    if not chunks:
        raise ValueError("No chunks provided.")

    embeddings = create_embeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore

# this function use to save database
def save_vectorstore(vectorstore):
    Path(VECTOR_STORE_PATH).mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore.save_local(VECTOR_STORE_PATH)

# this function use a load vectordata base
def load_vectorstore():
    embeddings = create_embeddings()

    return FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
