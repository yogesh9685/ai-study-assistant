"""
schemas.py
----------
Pydantic models for API request and response validation.
"""

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str = "ok"


class Source(BaseModel):
    """Document source reference with optional page number."""
    source: str | None = None
    page: int | None = None


class ChatRequest(BaseModel):
    """Request model for /chat endpoint."""
    question: str = Field(..., description="The user's query or instruction.")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Question cannot be empty.")
        return value.strip()


class ChatResponse(BaseModel):
    """Response model for /chat endpoint."""
    answer: str
    sources: list[Source] = Field(default_factory=list)


class UploadResponse(BaseModel):
    """Response model for /upload endpoint."""
    message: str
    filename: str
    chunks: int | None = None
