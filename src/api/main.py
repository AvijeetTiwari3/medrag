"""
MedRAG FastAPI Application
Main entry point for the REST API.
"""

import logging
import time
from contextlib import asynccontextmanager

import mlflow
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.models.rag import RAGPipeline

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle events."""
    # STARTUP
    logger.info("=" * 60)
    logger.info("  MedRAG API starting up ...")
    logger.info("=" * 60)

    try:
        config = load_config()
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        logger.info("MLflow configured.")
    except Exception as e:
        logger.warning(f"MLflow setup failed (non-critical): {e}")

    logger.info("Initialising RAG pipeline ...")
    app.state.rag_pipeline = RAGPipeline()
    app.state.start_time = time.time()

    logger.info("RAG pipeline ready.")
    logger.info("API is live at http://localhost:8000")
    logger.info("Interactive docs at http://localhost:8000/docs")

    yield

    # SHUTDOWN
    logger.info("MedRAG API shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedRAG API",
    description=(
        "Medical Question Answering System using Retrieval-Augmented Generation.\n\n"
        "Powered by ChromaDB + sentence-transformers + Phi-3 Mini (via Ollama).\n\n"
        "GitHub: https://github.com/your-username/medrag"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router)


# ─── Root ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    """Welcome endpoint."""
    return JSONResponse(content={
        "message": "Welcome to MedRAG API",
        "description": "Medical QA system powered by RAG + fine-tuned LLM",
        "version": "1.0.0",
        "endpoints": {
            "query":    "POST /api/v1/query",
            "health":   "GET  /api/v1/health",
            "metrics":  "GET  /api/v1/metrics",
            "docs":     "GET  /docs",
        },
    })


# ─── Entry Point ──────────────────────────────────────────────────────────────

def start():
    """Start the Uvicorn server."""
    config = load_config()
    api_cfg = config.get("api", {})
    uvicorn.run(
        "src.api.main:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=api_cfg.get("reload", False),
        workers=api_cfg.get("workers", 1),
    )


if __name__ == "__main__":
    start()
