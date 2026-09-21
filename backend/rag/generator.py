"""
generator.py
------------
LLM initialisation and answer generation for the RAG pipeline.
"""

import logging
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def create_llm():
    """Initialise and return the Groq LLM."""
    return init_chat_model("groq:openai/gpt-oss-20b")


def create_prompt(question, context):
    """Build a grounded prompt that instructs the LLM not to invent information."""
    return f"""You are a study assistant. Answer the question using ONLY the context provided below.

Rules:
1. Use ONLY information from the context. Do NOT use any external knowledge.
2. If the answer is not in the context, say exactly:
   "I could not find the answer to this question in the uploaded documents."
3. Do NOT guess, infer, or add information that is not explicitly in the context.
4. Keep your answer clear and concise.

Context:
{context}

Question:
{question}

Answer:"""


def generate_answer(llm, question, documents):
    """Generate a grounded answer from retrieved document chunks."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not documents:
        logger.warning("No documents retrieved — returning safe no-answer response.")
        return "I could not find relevant information in the documents."

    context = "\n\n".join(doc.page_content for doc in documents)
    prompt = create_prompt(question, context)

    logger.info("Generating answer from %d document chunks.", len(documents))
    response = llm.invoke(prompt)
    return response.content