"""
session_store.py
----------------
Simple in-memory conversation session store.

Maps session_id (str) -> conversation history (list of role/content dicts).

Limitation: history is lost when the server restarts.
            Swap this module for a Redis/DB-backed store in production
            without changing any other file — the interface stays the same.

Session isolation: each session_id has its own list, so different users
                   never see each other's conversation history.
"""

# {session_id: [{"role": "user", "content": "..."}, ...]}
_sessions: dict[str, list] = {}


def get_history(session_id: str) -> list:
    """Return the conversation history for a session, or an empty list."""
    return _sessions.get(session_id, [])


def save_history(session_id: str, history: list) -> None:
    """Persist the updated conversation history for a session."""
    _sessions[session_id] = history
