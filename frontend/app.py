"""
AI Study Assistant - Streamlit Frontend
Branch: feature/streamlit-ui

Provides a clean, professional UI for the AI Study Assistant.
Backend integration will be connected in the next branch.
"""

import streamlit as st


# --- Page Configuration ---

def configure_page():
    """Configure Streamlit page settings."""
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
            background: #69db7c;
            display: inline-block;
            box-shadow: 0 0 6px #69db7c;
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


# --- Session State Initialisation ---

def init_session_state():
    """Initialise required session state keys."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! Upload a study document and ask me a question about it.",
                "sources": [],
            }
        ]
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None


# --- Header ---

def render_header():
    """Render the top hero header."""
    st.markdown("""
    <div class="hero-card">
        <h1>📚 AI Study Assistant</h1>
        <p>Ask questions from your study documents</p>
        <div class="status-badge">
            <span class="status-dot"></span>
            Ready
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- Document Upload Section ---

def render_upload_section():
    """Render the document upload card."""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📄 Your Documents</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="Upload a study document",
        type=["pdf", "txt", "docx", "md", "csv"],
        label_visibility="collapsed",
        help="Supported formats: PDF, TXT, DOCX, MD, CSV",
    )

    if uploaded is not None:
        st.session_state.uploaded_file = {
            "name": uploaded.name,
            "size": uploaded.size,
            "type": uploaded.type,
        }
        size_kb = round(uploaded.size / 1024, 1)
        st.markdown(
            f'<div class="file-pill">📎 {uploaded.name} &nbsp;&middot;&nbsp; {size_kb} KB</div>',
            unsafe_allow_html=True,
        )
        st.success("Document ready. You can now ask questions below.", icon="✅")
    else:
        if st.session_state.uploaded_file is not None:
            st.session_state.uploaded_file = None
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📂</div>
            <p>No document uploaded yet.<br>Drop a file above to get started.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# --- Sources Display ---

def render_sources(sources: list):
    """Render placeholder source references for an assistant message."""
    if not sources:
        return

    items_html = "".join(
        f'<div class="source-item"><span class="source-dot"></span>{src}</div>'
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

PLACEHOLDER_SOURCES = [
    "Semester1.pdf - Page 4",
    "Semester1.pdf - Page 7",
]


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
        placeholder="Ask a question about your document...",
        key="chat_input",
    )

    if user_input:
        _handle_user_input(user_input.strip())


def _handle_user_input(question: str):
    """Process a user question and generate a placeholder assistant response."""

    if not question:
        st.warning("Please type a question before submitting.", icon="⚠️")
        return

    if st.session_state.uploaded_file is None:
        st.warning("Please upload a document before asking questions.", icon="📄")
        return

    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "sources": [],
    })

    file_name = st.session_state.uploaded_file["name"]
    placeholder_response = (
        f"Your question has been received.\n\n"
        f"**Document:** {file_name}\n\n"
        f"Backend integration will be connected in the next step."
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": placeholder_response,
        "sources": PLACEHOLDER_SOURCES,
    })

    st.rerun()


# --- Main Entry Point ---

def main():
    configure_page()
    init_session_state()

    render_header()

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        render_upload_section()

    with right_col:
        render_chat()


if __name__ == "__main__":
    main()
