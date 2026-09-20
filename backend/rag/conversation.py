def create_chat_history():
    return []


def add_message(history, role, content):
    history.append({
        "role": role,
        "content": content
    })

    return history


def rewrite_question(llm, question, history):
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not history:
        return question

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history
    )

    prompt = f"""
Rewrite the user's latest question into a standalone question.

Use the conversation history only to resolve references.

Conversation:
{history_text}

Latest question:
{question}

Standalone question:
"""

    response = llm.invoke(prompt)

    return response.content.strip()