from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()
def create_llm():
    return init_chat_model(  "groq:openai/gpt-oss-20b"
       
    )


def create_prompt(question, context):
    return f"""
Answer the question using only the provided context.

If the answer is not present in the context,
say that you could not find the answer in the documents.

Context:
{context}

Question:
{question}
"""


def generate_answer(llm, question, documents):
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not documents:
        return "I could not find relevant information in the documents."

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = create_prompt(question, context)

    response = llm.invoke(prompt)

    return response.content