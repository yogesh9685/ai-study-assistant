"""
generator.py
------------
LLM initialisation and answer generation for the RAG pipeline.

The LLM client is cached after the first call to avoid recreating
the Groq client on every request.
"""

import logging
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Module-level singleton — reused across all requests
_llm_instance = None


def create_llm():
    """Return the shared LLM instance, initializing it on the first call."""
    global _llm_instance
    if _llm_instance is None:
        logger.info("Initializing LLM client (first time only)...")
        _llm_instance = init_chat_model("groq:openai/gpt-oss-20b")
        logger.info("LLM client initialized and cached.")
    return _llm_instance


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