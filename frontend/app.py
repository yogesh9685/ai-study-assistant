"""
app.py
------
AI Study Assistant — Streamlit Frontend

Layout:
  Sidebar  — document upload, status, clear session
  Main     — chat interface with session memory
"""

import os
import uuid
from pathlib import Path

import requests
import streamlit as st

# ─── Configuration ────────────────────────────────────────────────────────────


# Backend URL
if "API_BASE_URL" in st.secrets:
    API_BASE_URL = st.secrets["API_BASE_URL"]
else:
    API_BASE_URL = os.getenv(
        "API_BASE_URL",
        "http://localhost:8000"
    )

API_BASE_URL = API_BASE_URL.rstrip("/")
MAX_FILE_MB = 10

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* App background */
.stApp { background-color: #f8fafc; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    width: 100%;
    font-weight: 500;
    padding: 0.5rem 1rem;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #334155 !important;
    border-color: #475569 !important;
}

/* Doc status card */
.doc-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
}
.doc-card.ready { border-color: #10b981; }
.doc-card .doc-name {
    font-weight: 600;
    font-size: 0.9rem;
    color: #e2e8f0;
    word-break: break-all;
}
.doc-card .doc-meta {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 4px;
}

/* ── Source box ── */
.source-box {
    background: #f0f9ff;
    border-left: 3px solid #2563eb;
    border-radius: 6px;
    padding: 8px 14px;
    margin-top: 8px;
    font-size: 0.83rem;
}
.source-box .src-title {
    font-weight: 600;
    color: #1d4ed8;
    margin-bottom: 4px;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.source-box .src-item {
    color: #374151;
    padding: 1px 0;
}

/* ── Chat area ── */
.chat-header {
    padding: 1rem 0 0.5rem 0;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}
.chat-header h2 {
    margin: 0;
    font-size: 1.4rem;
    font-weight: 700;
    color: #0f172a;
}
.chat-header .doc-badge {
    font-size: 0.82rem;
    color: #10b981;
    font-weight: 500;
}
.chat-header .no-doc-badge {
    font-size: 0.82rem;
    color: #94a3b8;
}

/* Sidebar divider */
.sdivider {
    border-top: 1px solid #1e293b;
    margin: 14px 0;
}
</style>
""", unsafe_allow_html=True)


# ─── API Helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=15)
def check_health() -> bool:
    """Return True if the FastAPI backend is reachable."""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def get_backend_status() -> dict:
    """Ask the backend whether a document is currently indexed."""
    try:
        r = requests.get(f"{API_BASE_URL}/status", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"has_document": False, "filename": None}


def upload_document(file) -> dict:
    """Upload a document to the backend for indexing."""
    try:
        files = {
            "file": (file.name, file.getvalue(), file.type or "application/octet-stream")
        }
        r = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=120)
        if r.status_code == 200:
            data = r.json()
            return {
                "success": True,
                "filename": data.get("filename", file.name),
                "chunks": data.get("chunks"),
            }
        detail = "Upload failed."
        try:
            detail = r.json().get("detail", detail)
        except Exception:
            pass
        return {"success": False, "error": detail}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Upload timed out. The document may be too large."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to the backend. Is FastAPI running?"}
    except Exception:
        return {"success": False, "error": "Upload failed. Please try again."}


def ask_question(question: str, session_id: str) -> dict:
    """Send a question to the backend /ask endpoint."""
    try:
        r = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": question, "session_id": session_id},
            timeout=90,
        )
        if r.status_code == 200:
            data = r.json()
            return {
                "success": True,
                "answer": data.get("answer", ""),
                "sources": data.get("sources", []),
            }
        detail = "Unable to process your question."
        try:
            detail = r.json().get("detail", detail)
        except Exception:
            pass
        return {"success": False, "error": detail}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to the backend. Is FastAPI running?"}
    except Exception:
        return {"success": False, "error": "Something went wrong. Please try again."}


def clear_documents() -> bool:
    """Tell the backend to clear all documents and reset the FAISS index."""
    try:
        r = requests.delete(f"{API_BASE_URL}/documents", timeout=15)
        return r.status_code == 200
    except Exception:
        return False


# ─── Session State ────────────────────────────────────────────────────────────

def _get_stable_session_id() -> str:
    """
    Return a session ID that survives page refresh.

    Stored in URL query params (?sid=...) so it persists across page refreshes.
    Falls back to session_state for older Streamlit versions.
    """
    try:
        params = st.query_params
        if "sid" not in params:
            params["sid"] = str(uuid.uuid4())
        return params["sid"]
    except Exception:
        if "_sid" not in st.session_state:
            st.session_state._sid = str(uuid.uuid4())
        return st.session_state._sid


def init_session():
    """Initialize all session state variables. Called on every Streamlit run."""

    # Stable session ID (survives refresh via URL params)
    if "session_id" not in st.session_state:
        st.session_state.session_id = _get_stable_session_id()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Document state
    if "doc_name" not in st.session_state:
        st.session_state.doc_name = None
    if "doc_chunks" not in st.session_state:
        st.session_state.doc_chunks = None
    if "doc_ready" not in st.session_state:
        st.session_state.doc_ready = False
    if "last_file_id" not in st.session_state:
        st.session_state.last_file_id = None

    # Restore document state from backend on first load / after refresh
    if "doc_checked" not in st.session_state:
        status = get_backend_status()
        if status.get("has_document"):
            st.session_state.doc_name = status.get("filename") or "Document"
            st.session_state.doc_ready = True
        st.session_state.doc_checked = True


# ─── Source Display ───────────────────────────────────────────────────────────

def render_sources(sources: list):
    """Render document source citations below an assistant message."""
    if not sources:
        return
    items = "".join(
        f'<div class="src-item">📄 {Path(s.get("source", "Unknown")).name}'
        + (f" &mdash; Page {s['page']}" if s.get("page") is not None else "")
        + "</div>"
        for s in sources
    )
    st.markdown(
        f'<div class="source-box">'
        f'<div class="src-title">Sources</div>'
        f'{items}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render the document management sidebar."""
    with st.sidebar:
        # Title
        st.markdown("## 📚 AI Study\nAssistant")
        st.caption("Ask questions from your study documents")
        st.markdown('<div class="sdivider"></div>', unsafe_allow_html=True)

        # Backend status
        backend_ok = check_health()
        if backend_ok:
            st.markdown("🟢 **Backend connected**")
        else:
            st.markdown("🔴 **Backend offline**")
            st.warning("Please start the FastAPI server and refresh.")

        st.markdown('<div class="sdivider"></div>', unsafe_allow_html=True)

        # ── Document section ──────────────────────────────────────────────
        st.markdown("### 📄 Document")

        if st.session_state.doc_ready:
            _render_doc_ready()
        else:
            _render_upload_widget()

        st.markdown('<div class="sdivider"></div>', unsafe_allow_html=True)

        # ── Clear session ─────────────────────────────────────────────────
        if st.button("🗑️ Clear Session", use_container_width=True, key="btn_clear"):
            with st.spinner("Clearing..."):
                clear_documents()
            st.session_state.doc_name = None
            st.session_state.doc_chunks = None
            st.session_state.doc_ready = False
            st.session_state.last_file_id = None
            st.session_state.messages = []
            st.session_state.doc_checked = False
            check_health.clear()
            st.rerun()

        st.markdown('<div class="sdivider"></div>', unsafe_allow_html=True)

        # ── Supported formats ─────────────────────────────────────────────
        st.markdown(
            "**Supported formats**  \n"
            "`PDF` · `TXT` · `DOCX` · `MD` · `CSV`  \n"
            f"Max size: **{MAX_FILE_MB} MB**"
        )

        st.markdown('<div class="sdivider"></div>', unsafe_allow_html=True)
        st.caption(f"Session: `{st.session_state.session_id[:8]}…`")


def _render_doc_ready():
    """Show the current document card + option to replace it."""
    chunks_text = (
        f"{st.session_state.doc_chunks} chunks indexed"
        if st.session_state.doc_chunks
        else "Indexed"
    )
    st.markdown(
        f'<div class="doc-card ready">'
        f'<div class="doc-name">✅ {st.session_state.doc_name}</div>'
        f'<div class="doc-meta">{chunks_text} · Ready</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    with st.expander("📤 Upload a different document"):
        _render_upload_widget()


def _render_upload_widget():
    """File uploader with step-by-step status feedback."""
    uploaded = st.file_uploader(
        "Upload a study document",
        type=["pdf", "txt", "docx", "md", "csv"],
        label_visibility="collapsed",
        help=f"PDF, TXT, DOCX, MD, CSV · Max {MAX_FILE_MB} MB",
        key="file_uploader",
    )

    if uploaded is None:
        return

    file_id = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.last_file_id == file_id:
        return  # Already processed this exact file

    # Client-side validation before hitting the backend
    if uploaded.size == 0:
        st.error("❌ The file is empty. Please choose a valid document.")
        return

    if uploaded.size > MAX_FILE_MB * 1024 * 1024:
        st.error(f"❌ File exceeds the {MAX_FILE_MB} MB limit.")
        return

    status_area = st.empty()
    status_area.info("⬆️ Uploading and processing document…")

    result = upload_document(uploaded)

    if result["success"]:
        st.session_state.doc_name = result["filename"]
        st.session_state.doc_chunks = result.get("chunks")
        st.session_state.doc_ready = True
        st.session_state.last_file_id = file_id
        status_area.success(f"✅ {result['filename']} is ready!")
        st.rerun()
    else:
        status_area.error(f"❌ {result['error']}")
        st.session_state.last_file_id = None


# ─── Chat Interface ───────────────────────────────────────────────────────────

def render_chat():
    """Render the main chat area."""

    # Header row
    hcol1, hcol2 = st.columns([3, 1])
    with hcol1:
        st.markdown('<div class="chat-header"><h2>💬 Chat</h2></div>', unsafe_allow_html=True)
    with hcol2:
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        if st.session_state.doc_ready:
            st.markdown(
                f'<div class="chat-header doc-badge" style="text-align:right">'
                f'📄 {st.session_state.doc_name}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="chat-header no-doc-badge" style="text-align:right">'
                'No document loaded</div>',
                unsafe_allow_html=True,
            )

    # Welcome message when chat is empty
    if not st.session_state.messages:
        st.info(
            "👋 **Welcome to AI Study Assistant!**  \n"
            "Upload a study document in the sidebar, then ask questions about it.  \n"
            "You can also ask math questions like *'What is 144 ÷ 12?'*"
        )

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])

    # Chat input
    question = st.chat_input(
        placeholder="Ask a question about your document or a math problem…",
        key="chat_input",
    )
    if question:
        _handle_question(question.strip())


def _handle_question(question: str):
    """Process user question: append to history, call API, display result."""
    if not question:
        return

    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})

    # Call the backend with a spinner
    with st.spinner("Thinking…"):
        result = ask_question(question, st.session_state.session_id)

    # Append assistant response to history
    if result["success"]:
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
        })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ {result['error']}",
            "sources": [],
        })

    st.rerun()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    init_session()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
