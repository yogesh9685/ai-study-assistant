"""
agent.py
--------
Branch: feature/agent

A simple LangChain tool-calling agent that decides whether to use the
calculator tool or the document search tool based on the user's question.

Architecture
------------
  User question
       |
       v
    Agent (LLM + tools)
       |
    decides
   /        \
Calculator  Document Search
   Tool         Tool
   |               |
result        retrieved
              chunks
   \               /
    \             /
      Agent (LLM)
          |
          v
     Final Answer

LangChain version: 1.4.1
API used: langchain.agents.create_agent(model, tools, system_prompt)

Usage
-----
  from backend.agent.agent import create_agent, run_agent

  agent = create_agent()
  answer = run_agent(agent, "What is 25 * 18?")
  print(answer)
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (.env file with GROQ_API_KEY etc.)
load_dotenv()

# Add project root to sys.path so backend.* imports work when this
# file is run directly (e.g. python backend/agent/agent.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

# Import the two existing tools
from backend.tools.document_search import document_search


# ---------------------------------------------------------------------------
# Calculator tool wrapper
#
# The existing calculate() function is a plain Python function.
# We wrap it with @tool here so the agent can discover and call it.
# The original calculator.py is NOT modified.
# ---------------------------------------------------------------------------
from backend.tools.calculator import calculate as _calculate

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Use this tool for any arithmetic calculation such as addition,
    subtraction, multiplication, division, or expressions with parentheses.
    Examples: '25 * 18', '100 / 4', '(10 + 5) * 2'.
    """
    try:
        result = _calculate(expression)
        # Return as a clean string so the agent can format the final answer
        return str(result)
    except ValueError as error:
        # Return the error message so the agent knows what went wrong
        return f"Calculation error: {error}"


# ---------------------------------------------------------------------------
# System prompt
#
# Tells the agent who it is and when to use each tool.
# Kept short and clear so it is easy to understand.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an AI Study Assistant.

Use the calculator tool for any mathematical calculation.

Use the document_search tool when the user asks about information
that may be in their uploaded study documents.

Do not invent information from documents. If the document_search tool
does not find relevant information, clearly tell the user that the
information was not found in the uploaded documents.

For simple conversational questions that need neither tool, answer normally.

Always give a clear and concise final answer."""


# ---------------------------------------------------------------------------
# LLM initialisation
# ---------------------------------------------------------------------------

def create_agent_llm():
    """
    Create the LLM used by the agent.
    Uses the same model already used by the RAG generation code.
    """
    return init_chat_model("groq:openai/gpt-oss-20b")


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------

def create_study_agent():
    """
    Build and return the tool-calling agent.

    The agent is created with:
      - The Groq LLM (same as RAG generation)
      - Two tools: calculator and document_search
      - A concise system prompt

    Returns
    -------
    A LangChain agent (compiled LangGraph runnable).
    """
    llm = create_agent_llm()

    tools = [calculator, document_search]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


# ---------------------------------------------------------------------------
# Run agent
# ---------------------------------------------------------------------------

def run_agent(agent, question: str, session_id: str | None = None, return_sources: bool = False):
    """
    Send a question to the agent and return its final answer.

    Parameters
    ----------
    agent : compiled LangGraph runnable
        The agent created by create_study_agent().
    question : str
        The user's question.
    session_id : str, optional
        Unique session ID to maintain conversation history.
    return_sources : bool, default False
        Whether to return the list of sources along with the answer.

    Returns
    -------
    str or tuple[str, list[dict]]
        The agent's final answer, or (answer, sources) if return_sources is True.

    Raises
    ------
    ValueError
        If the question is empty.
    RuntimeError
        If the agent fails unexpectedly.
    """

    # Validate: reject empty questions
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    try:
        from backend.tools.document_search import get_last_documents, reset_last_documents
        from backend.rag.rag_chain import get_sources

        reset_last_documents()

        messages = []
        if session_id:
            from backend.rag.session_store import get_history
            raw_history = get_history(session_id)
            for item in raw_history:
                role = item.get("role")
                content = item.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

            # Limit history to recent turns to avoid exceeding context window
            if len(messages) > 20:
                messages = messages[-20:]

        messages.append(HumanMessage(content=question.strip()))

        # The LangChain 1.x agent takes a dict with a "messages" list.
        result = agent.invoke({"messages": messages})

        # The result is a dict with a "messages" list.
        # The last message is the agent's final answer.
        final_message = result["messages"][-1]
        answer = final_message.content

        # Persist conversation turn if session_id is provided
        if session_id:
            from backend.rag.session_store import get_history, save_history
            from backend.rag.conversation import add_message
            history = get_history(session_id)
            history = add_message(history, "user", question.strip())
            history = add_message(history, "assistant", answer)
            save_history(session_id, history)

        if not return_sources:
            return answer

        used_document_search = False
        for msg in result.get("messages", []):
            if getattr(msg, "name", None) == "document_search":
                used_document_search = True
                break
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls and any(tc.get("name") == "document_search" for tc in tool_calls):
                used_document_search = True
                break

        if used_document_search:
            sources = get_sources(get_last_documents())
        else:
            sources = []

        return answer, sources


    except (ValueError, RuntimeError):
        raise

    except Exception as error:
        raise RuntimeError(
            f"Agent encountered an unexpected error: {error}"
        ) from error


# ---------------------------------------------------------------------------
# Manual test entry point
# (Only runs when this file is executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # safe_print encodes LLM output through the console's native encoding
    # (cp1252 on Windows) with 'replace' so Unicode chars don't crash the
    # test runner. The actual run_agent() return value is always a proper
    # Python str - this only affects how it is displayed in the terminal.
    def safe_print(text):
        encoding = sys.stdout.encoding or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding)
        print(safe)

    print("=" * 55)
    print("  AI Study Assistant - Agent Manual Tests")
    print("=" * 55)

    print("\nCreating agent...")
    agent = create_study_agent()
    print("Agent ready.\n")

    test_questions = [
        ("Calculator - basic multiply",  "What is 25 * 18?"),
        ("Calculator - division",        "What is 100 / 4?"),
        ("Document - in-document Q",     "What is Python according to the document?"),
        ("Document - out-of-scope Q",    "What is the capital of Mars?"),
        ("Conversational",               "Hello"),
        ("Calculator - invalid expr",    "What is 10 +?"),
    ]

    for label, question in test_questions:
        print("=" * 55)
        print(f"[{label}]")
        print(f"Question: {question}")
        try:
            answer = run_agent(agent, question)
            print("Answer:")
            safe_print(answer)
        except ValueError as e:
            safe_print(f"[INPUT ERROR] {e}")
        except RuntimeError as e:
            safe_print(f"[RUNTIME ERROR] {e}")
        print()

    # Test empty question
    print("=" * 55)
    print("[Empty question]")
    try:
        run_agent(agent, "")
        print("[FAIL] Expected ValueError")
    except ValueError as e:
        print(f"[PASS] ValueError: {e}")
