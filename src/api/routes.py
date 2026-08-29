"""
MedRAG API Routes
FastAPI route handlers for /query, /health, /metrics endpoints.
"""

import logging
import statistics
import time
from typing import List

import mlflow
from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import (
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    SourceDoc,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["MedRAG"])

# ─── In-Memory Metrics ────────────────────────────────────────────────────────
_query_latencies: List[float] = []
_total_queries: int = 0
_error_count: int = 0
_start_time: float = time.time()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Answer a medical question using RAG",
)
async def query_endpoint(request: Request, body: QueryRequest) -> QueryResponse:
    """
    Submit a medical question and receive an AI-generated answer with sources.
    """
    global _total_queries, _error_count

    _total_queries += 1
    start = time.time()

    try:
        rag_pipeline = request.app.state.rag_pipeline

        result = rag_pipeline.query(question=body.question, top_k=body.top_k)

        latency_ms = round((time.time() - start) * 1000, 2)
        _query_latencies.append(latency_ms)

        # Log to MLflow (async-safe: best-effort)
        try:
            with mlflow.start_run(
                run_name="api_query",
                experiment_id=mlflow.get_experiment_by_name("medrag_experiments").experiment_id,
                nested=True,
            ):
                mlflow.log_metric("latency_ms", latency_ms)
                mlflow.log_metric("retrieved_docs", len(result.get("sources", [])))
        except Exception:
            pass  # Non-critical

        sources = [
            SourceDoc(
                content=s.get("content", ""),
                source=s.get("source", ""),
                score=s.get("score", 0.0),
                chunk_id=s.get("chunk_id", ""),
            )
            for s in result.get("sources", [])
        ]

        return QueryResponse(
            question=body.question,
            answer=result.get("answer", ""),
            sources=sources,
            latency_ms=latency_ms,
            model=result.get("model", "unknown"),
            method=result.get("method", "rag"),
        )

    except Exception as e:
        _error_count += 1
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=str(e))


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health_endpoint(request: Request) -> HealthResponse:
    """Check API health, ChromaDB status, and Ollama availability."""
    rag_pipeline = request.app.state.rag_pipeline

    # Check ChromaDB
    try:
        stats = rag_pipeline.embedder.get_collection_stats()
        chroma_docs = stats.get("total_documents", 0)
    except Exception:
        chroma_docs = -1

    # Check Ollama
    ollama_ok = rag_pipeline._is_ollama_available()

    return HealthResponse(
        status="healthy" if chroma_docs > 0 else "degraded",
        version="1.0.0",
        chroma_docs=chroma_docs,
        ollama_available=ollama_ok,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Query performance metrics",
)
async def metrics_endpoint() -> MetricsResponse:
    """Return aggregated query performance metrics."""
    latencies = _query_latencies or [0.0]

    return MetricsResponse(
        total_queries=_total_queries,
        avg_latency_ms=round(statistics.mean(latencies), 2),
        p95_latency_ms=round(
            sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0], 2
        ),
        p99_latency_ms=round(
            sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[0], 2
        ),
        error_rate=round(_error_count / max(_total_queries, 1), 4),
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@router.get(
    "/docs-count",
    summary="Number of documents in ChromaDB",
)
async def docs_count_endpoint(request: Request) -> dict:
    """Return the total number of indexed document chunks."""
    rag_pipeline = request.app.state.rag_pipeline
    try:
        stats = rag_pipeline.embedder.get_collection_stats()
        return {"total_documents": stats.get("total_documents", 0)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
