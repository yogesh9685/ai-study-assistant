"""
AI Study Assistant - Streamlit Frontend
Branch: feature/streamlit-api-integration

Provides a clean, professional UI connected end-to-end to the FastAPI backend.
"""

import os
import sys
import uuid
from pathlib import Path
import requests
import streamlit as st

# Configurable API base URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


# --- Page Configuration ---

def configure_page():
    """Configure Streamlit page settings and styles."""
    st.set_page_config(
        page_title="AI Study Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }

        .stApp {
            background-color: #f5f7fa;
        }

        .hero-card {
            background: linear-gradient(135deg, #3b5bdb 0%, #5c7cfa 100%);
            border-radius: 16px;
            padding: 28px 36px;
            margin-bottom: 8px;
            color: white;
            box-shadow: 0 4px 24px rgba(59, 91, 219, 0.18);
        }
        .hero-card h1 {
            margin: 0 0 4px 0;
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        .hero-card p {
            margin: 0;
            opacity: 0.88;
            font-size: 1rem;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(255,255,255,0.18);
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 0.82rem;
            font-weight: 500;
            margin-top: 14px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }
        .dot-online {
            background: #69db7c;
            box-shadow: 0 0 6px #69db7c;
        }
        .dot-offline {
            background: #ff8787;
            box-shadow: 0 0 6px #ff8787;
        }

        .section-card {
            background: white;
            border-radius: 14px;
            padding: 24px 28px;
            margin-bottom: 16px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
            border: 1px solid #e9ecef;
        }
        .section-title {
            font-size: 1rem;
            font-weight: 600;
            color: #1a1a2e;
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .file-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: #eef2ff;
            border: 1px solid #c5d0fa;
            border-radius: 8px;
            padding: 8px 14px;
            font-size: 0.88rem;
            color: #3b5bdb;
            font-weight: 500;
            margin-top: 8px;
        }

        .source-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 0;
            font-size: 0.86rem;
            color: #495057;
            border-bottom: 1px solid #f1f3f5;
        }
        .source-item:last-child { border-bottom: none; }
        .source-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #5c7cfa;
            flex-shrink: 0;
        }

        .empty-state {
            text-align: center;
            padding: 36px 20px;
            color: #adb5bd;
        }
        .empty-state .icon {
            font-size: 2.4rem;
            margin-bottom: 10px;
        }
        .empty-state p {
            font-size: 0.92rem;
            margin: 0;
        }

        .stChatMessage {
            border-radius: 12px !important;
        }

        .stFileUploader > div {
            border: 2px dashed #c5d0fa !important;
            border-radius: 10px !important;
            background: #f8f9ff !important;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
    </style>
    """, unsafe_allow_html=True)


# --- API Helper Functions ---

@st.cache_data(ttl=10)
def check_backend_health() -> bool:
    """Check whether FastAPI backend is accessible."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200 and response.json().get("status") == "ok"
    except Exception:
        return False


def upload_document_to_api(file) -> dict:
    """Upload a study document to the FastAPI backend."""
    try:
        files = {
            "file": (
                file.name,
                file.getvalue(),
                file.type or "application/octet-stream",
            )
        }
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
            timeout=60,
        )
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "filename": data.get("filename", file.name),
                "chunks": data.get("chunks"),
                "message": data.get("message", "Document uploaded successfully."),
            }
        elif response.status_code == 400:
            error_detail = "Invalid document."
            try:
                error_detail = response.json().get("detail", error_detail)
            except Exception:
                pass
            return {
                "success": False,
                "error": f"❌ {error_detail}",
            }
        else:
            return {
                "success": False,
                "error": "❌ Document processing failed. Please try again.",
            }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "⏳ Document processing is taking too long. Please try again.",
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "❌ Cannot connect to the backend.\nPlease make sure FastAPI is running.",
        }
    except Exception:
        return {
            "success": False,
            "error": "❌ Document processing failed. Please try again.",
        }


def delete_documents_api() -> dict:
    """Delete all uploaded documents and clear FAISS index on backend."""
    try:
        response = requests.delete(f"{API_BASE_URL}/documents", timeout=15)
        if response.status_code == 200:
            return {"success": True, "message": "Documents and vector store cleared."}
        else:
            return {"success": False, "error": "Failed to delete documents on backend."}
    except Exception as e:
        return {"success": False, "error": f"Connection error: {e}"}


def ask_question_api(question: str) -> dict:

    """Send a user question to the FastAPI /chat endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"question": question},
            timeout=60,
        )
        if response.status_code == 200:
            data = response.json()
            if not isinstance(data, dict) or "answer" not in data:
                return {
                    "success": False,
                    "error": "❌ Received unexpected response format from the server.",
                }
            return {
                "success": True,
                "answer": data["answer"],
                "sources": data.get("sources", []),
            }
        elif response.status_code == 400:
            error_detail = "Invalid request."
            try:
                error_detail = response.json().get("detail", error_detail)
            except Exception:
                pass
            return {
                "success": False,
                "error": f"⚠️ {error_detail}",
            }
        elif response.status_code == 422:
            return {
                "success": False,
                "error": "⚠️ Question cannot be empty.",
            }
        else:
            return {
                "success": False,
                "error": "❌ An error occurred while processing your question. Please try again.",
            }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "⏳ Request timed out. The agent is taking too long to respond.",
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "❌ Cannot connect to the backend.\nPlease make sure FastAPI is running.",
        }
    except Exception:
        return {
            "success": False,
            "error": "❌ Failed to reach the assistant service. Please try again.",
        }


def ask_question_conversational_api(question: str, session_id: str) -> dict:
    """
    Send a question to the FastAPI /ask endpoint with conversation memory.

    The session_id ties this question to the user's conversation history
    stored in the backend session store.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": question, "session_id": session_id},
            timeout=60,
        )
        if response.status_code == 200:
            data = response.json()
            if not isinstance(data, dict) or "answer" not in data:
                return {
                    "success": False,
                    "error": "❌ Received unexpected response format from the server.",
                }
            return {
                "success": True,
                "answer": data["answer"],
                "sources": data.get("sources", []),
            }
        elif response.status_code == 400:
            error_detail = "Invalid request."
            try:
                error_detail = response.json().get("detail", error_detail)
            except Exception:
                pass
            return {
                "success": False,
                "error": f"⚠️ {error_detail}",
            }
        elif response.status_code == 422:
            return {
                "success": False,
                "error": "⚠️ Question cannot be empty.",
            }
        else:
            return {
                "success": False,
                "error": "❌ An error occurred while processing your question. Please try again.",
            }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "⏳ Request timed out. Please try again.",
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "❌ Cannot connect to the backend.\nPlease make sure FastAPI is running.",
        }
    except Exception:
        return {
            "success": False,
            "error": "❌ Failed to reach the assistant service. Please try again.",
        }


# --- Session State Initialisation ---

def init_session_state():
    """Initialise required session state keys."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! Upload a study document and ask me questions about it, or ask me any math calculation.",
                "sources": [],
            }
        ]
    if "current_doc_name" not in st.session_state:
        st.session_state.current_doc_name = None
    if "current_doc_chunks" not in st.session_state:
        st.session_state.current_doc_chunks = None
    if "last_processed_file_id" not in st.session_state:
        st.session_state.last_processed_file_id = None
    # Generate a unique session ID once per browser session.
    # This ties conversation history on the backend to this specific user.
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())


# --- Header ---

def render_header(backend_online: bool):
    """Render the top hero header with status badge."""
    if backend_online:
        badge_html = """
   
            🟢 connected
  
        """
    else:
        badge_html = """
        
            🔴 offline

        """

    st.markdown(f"""
    <div class="hero-card">
        <h1>📚 AI Study Assistant</h1>
        <p>Ask questions from your study documents</p>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


# --- Document Upload Section ---

def render_upload_section():
    """Render the document upload card and handle backend upload."""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📄 Your Documents</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="Upload a study document",
        type=["pdf", "txt", "docx", "md", "csv"],
        label_visibility="collapsed",
        help="Supported formats: PDF, TXT, DOCX, MD, CSV",
    )

    if uploaded is not None:
        file_id = f"{uploaded.name}_{uploaded.size}"
        
        # Upload to backend if not already uploaded in this session
        if st.session_state.last_processed_file_id != file_id:
            with st.spinner("Processing document..."):
                result = upload_document_to_api(uploaded)

            if result["success"]:
                st.session_state.last_processed_file_id = file_id
                st.session_state.current_doc_name = result["filename"]
                st.session_state.current_doc_chunks = result.get("chunks")
                st.success("✅ Document uploaded successfully")
            else:
                st.session_state.last_processed_file_id = None
                st.session_state.current_doc_name = None
                st.session_state.current_doc_chunks = None
                st.error(result["error"])

        if st.session_state.current_doc_name:
            size_kb = round(uploaded.size / 1024, 1)
            st.markdown(
                f'<div class="file-pill">📎 {st.session_state.current_doc_name} &nbsp;&middot;&nbsp; {size_kb} KB</div>',
                unsafe_allow_html=True,
            )
            st.success("✅ Document indexed successfully")
            if st.session_state.current_doc_chunks is not None:
                st.caption(f"📊 {st.session_state.current_doc_chunks} chunks indexed")

            if st.button("🗑️ Delete Document", key="btn_delete_doc", use_container_width=True):
                with st.spinner("Deleting document and clearing index..."):
                    delete_documents_api()
                st.session_state.last_processed_file_id = None
                st.session_state.current_doc_name = None
                st.session_state.current_doc_chunks = None
                st.success("🗑️ Document deleted and vector store cleared.")
                st.rerun()
    else:
        # Clear state and remove index when file is removed
        if st.session_state.last_processed_file_id is not None:
            delete_documents_api()
            st.session_state.last_processed_file_id = None
            st.session_state.current_doc_name = None
            st.session_state.current_doc_chunks = None
            st.info("🗑️ Document removed and index cleared.")
            st.rerun()

        st.markdown("""
        <div class="empty-state">
            <div class="icon">📂</div>
            <p>No document uploaded yet.<br>Drop a file above to get started.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)



# --- Sources Display ---

def format_source_label(src) -> str:
    """Format a source entry into a user-friendly label."""
    if isinstance(src, dict):
        source_path = src.get("source") or "Document"
        source_name = Path(source_path).name
        page = src.get("page")
        if page is not None:
            return f"{source_name} — Page {page}"
        return source_name
    return str(src)


def render_sources(sources: list):
    """Render document source references for an assistant message."""
    if not sources:
        return

    items_html = "".join(
        f'<div class="source-item"><span class="source-dot"></span>{format_source_label(src)}</div>'
        for src in sources
    )
    st.markdown(f"""
    <div style="margin-top:8px; padding:12px 16px; background:#f8f9ff;
                border-radius:10px; border:1px solid #dde3ff;">
        <div style="font-size:0.82rem; font-weight:600; color:#3b5bdb; margin-bottom:8px;">
            📚 Sources
        </div>
        {items_html}
    </div>
    """, unsafe_allow_html=True)


# --- Chat Interface ---

def render_chat():
    """Render the full chat interface."""
    st.markdown('<div class="section-card" style="min-height:420px;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💬 Ask your document</div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">💭</div>
            <p>No conversation yet. Ask a question below!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    render_sources(msg["sources"])

    st.markdown('</div>', unsafe_allow_html=True)

    user_input = st.chat_input(
        placeholder="Ask a question about your document or math calculation...",
        key="chat_input",
    )

    if user_input:
        _handle_user_input(user_input.strip())


def _handle_user_input(question: str):
    """Process a user question using the conversational RAG endpoint."""
    if not question:
        st.warning("Please type a question before submitting.", icon="⚠️")
        return

    # Append user question to the visible chat history
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "sources": [],
    })

    # Call the conversational RAG endpoint (with session_id for memory)
    with st.spinner("Thinking..."):
        result = ask_question_conversational_api(
            question=question,
            session_id=st.session_state.session_id,
        )

    if result["success"]:
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
        })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["error"],
            "sources": [],
        })

    st.rerun()


# --- Main Entry Point ---

def main():
    configure_page()
    init_session_state()

    backend_online = check_backend_health()
    render_header(backend_online)

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        render_upload_section()

    with right_col:
        render_chat()


if __name__ == "__main__":
    main()
