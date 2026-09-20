"""
main.py
-------
FastAPI application entry point for AI Study Assessment API.
"""

import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (.env file)
load_dotenv()

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai_study_assistant")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown events."""
    logger.info("Starting AI Study Assessment API...")
    # Pre-create upload directory
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    yield
    logger.info("Shutting down AI Study Assessment API...")


# Create FastAPI application
app = FastAPI(
    title="AI Study Assessment API",
    description=(
        "Production-oriented API for the AI Study Assistant, providing "
        "document upload and indexing, agent chat, and health checking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
ALLOWED_ORIGINS = [
    "http://localhost:8501",   # Streamlit default
    "http://127.0.0.1:8501",
    "http://localhost:3000",   # Frontend dev server
    "http://127.0.0.1:3000",
    "http://localhost:8000",   # Local backend / swagger
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
