<div align="center">

<h1>MedRAG</h1>
<h3>Production-Grade Medical Question Answering with Retrieval-Augmented Generation</h3>

<p>
  <img src="https://img.shields.io/badge/tests-passing-brightgreen?logo=github" alt="Tests">
  <a href="https://medrag-avijeet.streamlit.app/"><img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit" alt="Live Demo"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Dataset-MedQuAD%2047K-orange" alt="Dataset">
</p>

<p>
  <b><a href="https://medrag-avijeet.streamlit.app/">Try the Live Demo</a></b> &nbsp;|&nbsp;
  <b><a href="#-quick-start">Quick Start</a></b> &nbsp;|&nbsp;
  <b><a href="#-results">Results</a></b> &nbsp;|&nbsp;
  <b><a href="#-api-reference">API Docs</a></b>
</p>

<img src="https://img.shields.io/badge/Hit@3-68%25-brightgreen" alt="Hit@3">
<img src="https://img.shields.io/badge/MRR-0.567-brightgreen" alt="MRR">
<img src="https://img.shields.io/badge/55%25%20over%20BM25-improvement-blue" alt="improvement">

</div>

---

## Overview

MedRAG is an end-to-end **Retrieval-Augmented Generation** system for medical question answering, built entirely with free and open-source tools. It retrieves relevant passages from a knowledge base of **14,981 curated medical documents** (MedQuAD dataset) and generates grounded answers using a locally-running LLM — no paid APIs required.

**Key highlights:**
- **+55% retrieval accuracy** over BM25 baseline (Hit@1: 0.28 → 0.44)
- **Full MLOps stack** — experiment tracking, drift monitoring, CI/CD, containerization
- **Production-ready API** — FastAPI with Pydantic v2, async endpoints, rate limiting
- **100% free & open-source** — runs entirely on local hardware

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         User Query (REST / UI)        │
                    └──────────────────┬──────────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │   Sentence-Transformers Encoder       │
                    │   all-MiniLM-L6-v2  (384-dim)        │
                    └──────────────────┬──────────────────┘
                                       │ query embedding
                    ┌──────────────────▼──────────────────┐
                    │         ChromaDB Vector Store         │
                    │      14,981 medical doc chunks        │
                    │       cosine similarity search        │
                    └──────────────────┬──────────────────┘
                                       │ top-5 passages
                    ┌──────────────────▼──────────────────┐
                    │       Phi-3 Mini via Ollama           │
                    │    grounded answer generation         │
                    └──────────────────┬──────────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │     Answer + Source Citations         │
                    └─────────────────────────────────────┘
```

---

## Results

> Evaluated on **200 held-out test samples** from MedQuAD

| Metric | TF-IDF | BM25 | RAG (MiniLM + ChromaDB) |
|:-------|:------:|:----:|:-----------------------:|
| Hit@1  | 0.170  | 0.283 | **0.440** (+55%) |
| Hit@3  | 0.410  | 0.520 | **0.680** (+31%) |
| Hit@5  | 0.527  | 0.627 | **0.745** (+19%) |
| MRR    | 0.299  | 0.407 | **0.567** (+39%) |

> All metrics tracked in **MLflow** (`mlflow ui --port 5000`)

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Dataset** | MedQuAD (HuggingFace) | 16,407 medical QA pairs |
| **Preprocessing** | NLTK, scikit-learn | Chunking, dedup (14,981 chunks) |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Dense retrieval (384-dim) |
| **Vector Store** | ChromaDB | Persistent local vector DB |
| **LLM** | Phi-3 Mini via Ollama | Local inference (no API key) |
| **Baseline** | TF-IDF + BM25 (rank-bm25) | Retrieval comparison |
| **Fine-tuning** | QLoRA (PEFT + bitsandbytes) | Phi-3-mini domain adaptation |
| **API** | FastAPI + Uvicorn | REST endpoints |
| **Demo** | Streamlit | Interactive UI |
| **Tracking** | MLflow | Experiment logging |
| **Monitoring** | Evidently AI | Query drift detection |
| **CI/CD** | GitHub Actions | Lint + test on push |
| **Container** | Docker + docker-compose | Reproducible deployment |

---

## Project Structure

```
medrag/
├── src/
│   ├── data/
│   │   ├── loader.py          # HuggingFace dataset download + splits
│   │   ├── preprocessor.py    # Clean → chunk → deduplicate pipeline
│   │   └── embedder.py        # Sentence-transformer embeddings + ChromaDB
│   ├── models/
│   │   ├── baseline.py        # TF-IDF and BM25 retrieval baselines
│   │   ├── rag.py             # Full RAG pipeline (retrieve + generate)
│   │   └── finetuning.py      # QLoRA fine-tuning on Kaggle T4 GPU
│   ├── api/
│   │   ├── main.py            # FastAPI app with lifespan context
│   │   ├── routes.py          # /query, /health, /metrics, /docs-count
│   │   └── schemas.py         # Pydantic v2 request/response models
│   ├── evaluation/
│   │   └── metrics.py         # Hit@K, MRR, Faithfulness, Answer Relevancy
│   └── monitoring/
│       └── drift.py           # Evidently AI query drift detection
├── tests/
│   ├── test_pipeline.py       # DataLoader, Preprocessor, Embedder tests
│   ├── test_api.py            # FastAPI endpoint tests (TestClient)
│   └── test_models.py         # BaselineRetriever, Evaluator tests
├── docker/
│   ├── Dockerfile             # python:3.11-slim multi-stage build
│   └── docker-compose.yml     # medrag-api + mlflow services
├── .github/workflows/
│   ├── ci.yml                 # pytest + ruff on push/PR
│   └── deploy.yml             # Auto-deploy to Streamlit on merge
├── configs/
│   ├── config.yaml            # All hyperparameters and paths
│   └── model_config.yaml      # QLoRA fine-tuning configuration
├── data/
│   ├── demo_chunks.parquet    # 3,000 chunks for live demo
│   └── demo_embeddings.npy    # Pre-computed embeddings (3000x384)
├── streamlit_app.py           # Streamlit demo (deployed)
├── app.py                     # Gradio demo (local)
├── Makefile                   # make ingest / run / test / demo
└── README.md
```

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt

# Ollama (local LLM runtime) — https://ollama.com/download
ollama pull phi3:mini   # ~2.2 GB one-time download
```

### Build Knowledge Index

```bash
python -m src.data.loader        # Download MedQuAD (16K QA pairs)
python -m src.data.preprocessor  # Chunk + deduplicate → 14,981 chunks
python -m src.data.embedder      # Embed + index into ChromaDB
# ~15-30 min on first run (CPU)
```

### Run the API

```bash
uvicorn src.api.main:app --reload
# API:       http://localhost:8000
# Swagger:   http://localhost:8000/docs
```

### Run the Demo

```bash
streamlit run streamlit_app.py   # http://localhost:8501
# or
python app.py                    # Gradio at http://localhost:7860
```

---

## API Reference

### `POST /api/v1/query`

```json
// Request
{
  "question": "What are the symptoms of Type 2 Diabetes?",
  "top_k": 5
}

// Response
{
  "question": "What are the symptoms of Type 2 Diabetes?",
  "answer": "Symptoms include increased thirst, frequent urination, fatigue, unexplained weight loss, blurred vision...",
  "sources": [
    { "content": "...", "source": "NIH", "score": 0.799, "chunk_id": "doc_0_chunk_0" }
  ],
  "latency_ms": 59070.26,
  "model": "phi3:mini",
  "method": "rag"
}
```

### `GET /api/v1/health`

```json
{ "status": "healthy", "version": "1.0.0", "chroma_docs": 14981, "ollama_available": true }
```

### `GET /api/v1/metrics`

Returns P50 / P95 / P99 latency, error rate, and total query count.

---

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=html   # Full suite with coverage
pytest tests/ -v -m "not slow"                 # Fast tests only (no network)
```

---

## MLflow Experiment Tracking

```bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --port 5000
```

Tracked experiments:
- `baseline_evaluation` — TF-IDF and BM25 retrieval metrics
- `rag_retrieval_evaluation` — ChromaDB semantic retrieval metrics
- Fine-tuning loss curves and eval metrics

---

## Fine-tuning on Kaggle (Free T4 GPU)

The `src/models/finetuning.py` script fine-tunes **Phi-3-mini-4k-instruct** on MedQuAD using **QLoRA** (4-bit quantization + LoRA adapters).

```bash
# On Kaggle Notebook (free GPU T4 x2):
# 1. Upload finetuning.py + configs/model_config.yaml
# 2. Enable GPU: Settings -> Accelerator -> GPU T4 x2
python finetuning.py
# ~1-2 hours | Model saved to HuggingFace Hub
```

**QLoRA config:** r=16, alpha=32, target_modules=q/k/v/o/gate/up/down proj, dropout=0.05

---

## Docker

```bash
docker build -f docker/Dockerfile -t medrag:latest .
docker-compose -f docker/docker-compose.yml up -d
# API:    http://localhost:8000
# MLflow: http://localhost:5000
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | MedQuAD (NIH, CDC, Mayo Clinic, MedlinePlus, WebMD) |
| Raw samples | 16,407 QA pairs |
| After chunking | 15,811 chunks |
| After deduplication | **14,981 chunks** (830 removed) |
| Chunk size | 512 tokens, 64 token overlap |
| Dedup threshold | TF-IDF cosine similarity > 0.95 |

---

## License

This project is licensed under the **MIT License**.

---

<div align="center">
<sub>Built as a production-grade AI/ML portfolio project demonstrating full-stack ML engineering.</sub>
</div>