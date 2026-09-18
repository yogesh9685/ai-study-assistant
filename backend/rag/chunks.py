from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    if not documents:
        raise ValueError("No documents provided.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    chunks = [
        chunk for chunk in chunks
        if chunk.page_content.strip()
    ]

    if not chunks:
        raise ValueError("No readable chunks were created.")

    return chunks