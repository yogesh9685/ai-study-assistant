"""
routes.py
---------
API route definitions for health checks, document uploads, and agent chat.
All document uploads, status, vector store operations, and conversational
retrieval are isolated per session_id.
"""

import logging
import time
from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

from backend.api.schemas import (
    AskRequest,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    StatusResponse,
    UploadResponse,
)
from backend.rag.document_loader import load_document
from backend.rag.chunks import split_documents
from backend.rag.vectorstore import (
    create_vectorstore,
    save_vectorstore,
    delete_vectorstore,
    has_vectorstore,
    validate_session_id,
)
from backend.tools.document_search import reset_retriever
from backend.agent.agent import create_study_agent, run_agent


logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration constants
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md", ".csv"}
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
UPLOAD_BASE_DIR = Path("data/uploads")

# Shared agent instance (initialized once, reused across requests)
_agent = None


def get_session_upload_dir(session_id: str) -> Path:
    """Return and validate the session-specific upload directory."""
    clean_id = validate_session_id(session_id)
    return UPLOAD_BASE_DIR / clean_id


def get_agent():
    """Get or initialize the shared agent instance."""
    global _agent
    if _agent is None:
        logger.info("Initializing study agent...")
        _agent = create_study_agent()
        logger.info("Study agent initialized successfully.")
    return _agent


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
    description="Check whether the API is running without calling LLM or RAG services.",
)
async def health_check():
    """Health check endpoint to verify backend service availability."""
    return HealthResponse(status="ok")


# ---------------------------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Check document/index status",
    description="Returns whether a document is currently indexed for the given session.",
)
async def get_status(
    session_id: str | None = Query(default=None, description="Unique session identifier"),
):
    """
    Check if a FAISS index and uploaded document exist on disk for the specified session.
    """
    if not session_id or not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID is required and cannot be empty.",
        )

    try:
        clean_id = validate_session_id(session_id)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )

    has_document = has_vectorstore(clean_id)
    session_upload_dir = get_session_upload_dir(clean_id)

    filename = None
    if has_document and session_upload_dir.exists():
        candidates = [
            f for f in session_upload_dir.iterdir()
            if f.is_file() and f.name != ".gitkeep"
        ]
        if candidates:
            filename = max(candidates, key=lambda f: f.stat().st_mtime).name

    logger.info("Status check for session %s: has_document=%s filename=%s", clean_id, has_document, filename)
    return StatusResponse(has_document=has_document, filename=filename)


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload and index a study document",
    description="Upload a study document (.pdf, .txt, .docx, .md, .csv) and index it into the session-specific vector store.",
)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None, description="Unique session identifier"),
):
    """
    Handle document upload, chunking, embedding generation, and FAISS indexing
    isolated per session_id.
    """
    if not session_id or not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID is required and cannot be empty.",
        )

    try:
        clean_id = validate_session_id(session_id)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )

    if not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing.",
        )

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type.",
        )

    # Read uploaded file contents
    try:
        contents = await file.read()
    except Exception as error:
        logger.error("Failed to read uploaded file: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read uploaded file.",
        )

    # Validate empty file
    if not contents or len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Validate file size
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum allowed size ({MAX_FILE_SIZE_MB} MB).",
        )

    # Ensure session uploads directory exists: data/uploads/<session_id>/
    session_upload_dir = get_session_upload_dir(clean_id)
    session_upload_dir.mkdir(parents=True, exist_ok=True)

    # Save file safely without overwriting unrelated files (safe filename prevents traversal)
    safe_filename = Path(file.filename).name
    dest_path = session_upload_dir / safe_filename
    if dest_path.exists():
        stem = dest_path.stem
        counter = 1
        while (session_upload_dir / f"{stem}_{counter}{file_ext}").exists():
            counter += 1
        dest_path = session_upload_dir / f"{stem}_{counter}{file_ext}"

    logger.info("Document upload started for session %s: %s", clean_id, dest_path.name)

    try:
        dest_path.write_bytes(contents)
    except Exception as error:
        logger.error("Failed to save uploaded file: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",
        )

    # Process document through the existing RAG pipeline
    try:
        t0 = time.perf_counter()

        documents = load_document(str(dest_path))
        t1 = time.perf_counter()
        logger.info("Document loading: %.2fs", t1 - t0)

        chunks = split_documents(documents)
        t2 = time.perf_counter()
        logger.info("Chunking: %.2fs (%d chunks)", t2 - t1, len(chunks))

        vectorstore = create_vectorstore(chunks)
        t3 = time.perf_counter()
        logger.info("Embedding + FAISS creation: %.2fs", t3 - t2)

        save_vectorstore(vectorstore, session_id=clean_id)
        t4 = time.perf_counter()
        logger.info("FAISS save: %.2fs", t4 - t3)

        # Invalidate the cached retriever for this session
        reset_retriever(clean_id)

        logger.info(
            "Total upload processing for session %s: %.2fs (%s, %d chunks)",
            clean_id,
            t4 - t0,
            dest_path.name,
            len(chunks),
        )

        return UploadResponse(
            message="Document uploaded successfully.",
            filename=dest_path.name,
            chunks=len(chunks),
        )

    except ValueError as val_err:
        logger.error("Validation error processing document: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as error:
        logger.error("Document processing failed: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process and index document.",
        )


# ---------------------------------------------------------------------------
# Delete documents endpoint
# ---------------------------------------------------------------------------

@router.delete(
    "/documents",
    summary="Delete uploaded documents and reset vector store for session",
    description="Deletes all uploaded files, removes the FAISS index, and resets retriever for the given session.",
)
async def delete_documents(
    session_id: str | None = Query(default=None, description="Session ID whose documents to delete"),
):
    """Clear uploaded files, vector store index, and retriever for the specified session."""
    if not session_id or not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID is required and cannot be empty.",
        )

    try:
        clean_id = validate_session_id(session_id)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )

    try:
        session_upload_dir = get_session_upload_dir(clean_id)
        if session_upload_dir.exists():
            for file_path in session_upload_dir.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                    except Exception as err:
                        logger.warning("Could not delete file %s: %s", file_path, err)
            try:
                session_upload_dir.rmdir()
            except Exception:
                pass

        delete_vectorstore(clean_id)
        reset_retriever(clean_id)

        logger.info("Documents and vector store cleared successfully for session: %s", clean_id)
        return {"message": f"Documents and vector store index cleared successfully for session {clean_id}."}

    except Exception as error:
        logger.error("Failed to delete documents for session %s: %s", clean_id, error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete documents: {error}",
        )


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question using the AI Study Assistant",
    description="Send a question to the AI agent, which routes to calculator, document search, or conversation.",
)
async def chat(request: ChatRequest):
    """
    Handle user question using the tool-calling agent.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    if request.session_id:
        try:
            validate_session_id(request.session_id)
        except ValueError as val_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(val_err),
            )

    logger.info("Chat request received (session: %s)", request.session_id)

    try:
        agent = get_agent()
        answer, sources = run_agent(
            agent,
            request.question,
            session_id=request.session_id,
            return_sources=True,
        )

        logger.info("Agent execution completed")
        return ChatResponse(answer=answer, sources=sources)

    except ValueError as val_err:
        logger.error("Chat request validation error: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except RuntimeError as run_err:
        logger.error("Agent execution error: %s", run_err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent encountered an error while processing the request.",
        )
    except Exception as error:
        logger.error("Agent execution failed: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the request.",
        )


# ---------------------------------------------------------------------------
# Conversational endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/ask",
    response_model=ChatResponse,
    summary="Ask a question with conversation memory",
    description=(
        "Send a question with a session ID. The tool-calling agent resolves conversation "
        "context and routes to calculator, document search, or conversational answer."
    ),
)
async def ask(request: AskRequest):
    """
    Conversational agent endpoint with session memory and isolated document retrieval.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    if not request.session_id or not request.session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID cannot be empty.",
        )

    try:
        clean_session_id = validate_session_id(request.session_id)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )

    logger.info("Conversational request received (session: %s)", clean_session_id)

    try:
        agent = get_agent()
        answer, sources = run_agent(
            agent,
            request.question,
            session_id=clean_session_id,
            return_sources=True,
        )

        logger.info("Conversational execution completed (session: %s)", clean_session_id)
        return ChatResponse(answer=answer, sources=sources)

    except ValueError as val_err:
        logger.error("Ask request validation error: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except RuntimeError as run_err:
        logger.error("Agent execution error: %s", run_err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent encountered an error while processing the request.",
        )
    except Exception as error:
        logger.error("Ask endpoint failed: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the request.",
        )
