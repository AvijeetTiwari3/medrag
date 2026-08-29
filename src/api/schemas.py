"""
MedRAG API Schemas
Pydantic v2 request/response models for the FastAPI application.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for the /query endpoint."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Medical question to answer",
        examples=["What are the symptoms of Type 2 Diabetes?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve",
    )
    use_rag: bool = Field(
        default=True,
        description="Use RAG pipeline (True) or return only retrieved context (False)",
    )

    model_config = {"json_schema_extra": {"example": {
        "question": "What are the symptoms of Type 2 Diabetes?",
        "top_k": 5,
        "use_rag": True,
    }}}


class SourceDoc(BaseModel):
    """A retrieved source document."""

    content: str = Field(..., description="Document chunk content")
    source: str = Field(..., description="Source document name")
    score: float = Field(..., description="Relevance score (0-1)")
    chunk_id: str = Field(..., description="Unique chunk identifier")


class QueryResponse(BaseModel):
    """Response model for the /query endpoint."""

    question: str
    answer: str
    sources: List[SourceDoc]
    latency_ms: float
    model: str
    method: str

    model_config = {"json_schema_extra": {"example": {
        "question": "What are the symptoms of Type 2 Diabetes?",
        "answer": "Common symptoms of Type 2 Diabetes include increased thirst...",
        "sources": [],
        "latency_ms": 1250.5,
        "model": "phi3:mini",
        "method": "rag",
    }}}


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""

    status: str
    version: str
    chroma_docs: int
    ollama_available: bool
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Response model for the /metrics endpoint."""

    total_queries: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str
