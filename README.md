# MedRAG - Medical Question Answering System

> **End-to-end RAG pipeline** for answering medical questions using ChromaDB, sentence-transformers, and Phi-3 Mini - fully production-grade with MLflow tracking, FastAPI serving, Gradio demo, and CI/CD.

[![CI](https://github.com/your-username/medrag/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/medrag/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B)](https://medrag-avijeet.streamlit.app/)

---

## Live Demo

> **Try it live:** [medrag-avijeet.streamlit.app](https://medrag-avijeet.streamlit.app/)

---

## Architecture

```
User Query
    |
    v
[FastAPI / Gradio UI]
    |
    v
[Sentence-Transformers Embedder]  <-- all-MiniLM-L6-v2 (384-dim)
    |
    v
[ChromaDB Vector Store]  <-- 14,981 medical document chunks
    |
    v
[Top-5 Relevant Chunks Retrieved]
    |
    v
[Phi-3 Mini LLM via Ollama]  <-- grounded answer generation
    |
    v
[Answer + Source Citations]
```

---

## Results (Evaluated on 200 test samples from MedQuAD)

| Metric | TF-IDF | BM25 | RAG (MiniLM + ChromaDB) | RAG + Fine-tuned* |
|--------|:------:|:----:|:-----------------------:|:-----------------:|
| Hit@1  | 0.170  | 0.283 | **0.440**              | *in progress*     |
| Hit@3  | 0.410  | 0.520 | **0.680**              | *in progress*     |
| Hit@5  | 0.527  | 0.627 | **0.745**              | *in progress*     |
| MRR    | 0.299  | 0.407 | **0.567**              | *in progress*     |
| CPU Latency | 45ms | 52ms | ~60s (phi3:mini, CPU) | - |

> *RAG vs BM25: +55% improvement in Hit@1, +39% improvement in MRR*
>
> *Fine-tuning in progress on Kaggle T4 GPU using QLoRA (PEFT)*

---

## Dataset

- **MedQuAD** (Medical Question Answering Dataset)
- 16,407 raw medical QA pairs from NIH, CDC, Mayo Clinic, MedlinePlus
- After preprocessing: **14,981 clean chunks** (512 tokens, 64 overlap)
- 830 near-duplicate chunks removed (TF-IDF cosine similarity threshold = 0.95)
- Sources: NIH, CDC, Mayo Clinic, WebMD, MedlinePlus

---

## Tech Stack (100% Free and Open Source)

| Component | Tool |
|-----------|------|
| Dataset | MedQuAD (HuggingFace, free) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local) |
| Vector DB | ChromaDB (local, persistent) |
| LLM | Phi-3 Mini via Ollama (local, free) |
| Fine-tuning | QLoRA (PEFT + bitsandbytes) on Kaggle T4 GPU |
| Experiment Tracking | MLflow (local) |
| API | FastAPI + Uvicorn |
| Demo UI | Gradio (HuggingFace Spaces, free) |
| Monitoring | Evidently AI (drift detection) |
| CI/CD | GitHub Actions (free for public repos) |
| Containerization | Docker + docker-compose |

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/your-username/medrag.git
cd medrag
pip install -r requirements.txt
```

### 2. Install Ollama and Pull Model

```bash
# Install from https://ollama.com/download
ollama serve
ollama pull phi3:mini   # ~2.2 GB download
```

### 3. Build Knowledge Index

```bash
python -m src.data.loader        # Download MedQuAD dataset
python -m src.data.preprocessor  # Clean, chunk, deduplicate
python -m src.data.embedder      # Build ChromaDB vector index
# Takes ~15-30 minutes on first run
```

### 4. Run the API

```bash
uvicorn src.api.main:app --reload
# API:  http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 5. Run the Gradio Demo

```bash
python app.py
# Demo: http://localhost:7860
```

---

## Project Structure

```
medrag/
+-- src/
|   +-- data/
|   |   +-- loader.py          # HuggingFace dataset download
|   |   +-- preprocessor.py    # Clean, chunk, deduplicate (14,981 chunks)
|   |   +-- embedder.py        # Embeddings + ChromaDB indexing
|   +-- models/
|   |   +-- baseline.py        # TF-IDF + BM25 retrieval baselines
|   |   +-- rag.py             # Full RAG pipeline (retrieve + generate)
|   |   +-- finetuning.py      # QLoRA fine-tuning (run on Kaggle)
|   +-- api/
|   |   +-- main.py            # FastAPI application
|   |   +-- routes.py          # /query, /health, /metrics endpoints
|   |   +-- schemas.py         # Pydantic v2 request/response models
|   +-- evaluation/
|   |   +-- metrics.py         # Hit@K, MRR, Faithfulness, Relevancy
|   +-- monitoring/
|       +-- drift.py           # Evidently AI drift detection
+-- tests/                     # pytest test suite
+-- docker/                    # Dockerfile + docker-compose
+-- .github/workflows/         # CI (lint+test) + CD (HF Spaces deploy)
+-- configs/
|   +-- config.yaml            # All hyperparameters
|   +-- model_config.yaml      # QLoRA fine-tuning config
+-- app.py                     # Gradio demo
+-- Makefile                   # make run, make test, make ingest, etc.
```

---

## API Reference

### POST /api/v1/query

```json
Request:
{
  "question": "What are the symptoms of Type 2 Diabetes?",
  "top_k": 5
}

Response:
{
  "answer": "The symptoms of Type 2 Diabetes include:\n- Increased thirst\n- Increased hunger\n- Fatigue\n- Increased urination...",
  "sources": [
    {"content": "...", "source": "NIH", "score": 0.799, "chunk_id": "doc_0_chunk_0"}
  ],
  "latency_ms": 59070.26,
  "model": "phi3:mini",
  "method": "rag"
}
```

### GET /api/v1/health

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "chroma_docs": 14981,
  "ollama_available": true,
  "uptime_seconds": 69.02
}
```

### GET /api/v1/metrics

Returns P50/P95/P99 latency, error rate, and total query count.

---

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=html   # full suite
pytest tests/ -v -m "not slow"                 # fast tests only
```

---

## MLflow Experiment Tracking

```bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --port 5000
# Open: http://localhost:5000
```

All experiments are tracked:
- Baseline evaluation (TF-IDF, BM25)
- RAG retrieval evaluation
- QLoRA fine-tuning metrics
- API query latencies

---

## Fine-tuning on Kaggle (Free T4 GPU)

1. Go to [kaggle.com/code](https://kaggle.com/code) and create New Notebook
2. Enable GPU: Settings -> Accelerator -> GPU T4 x2
3. Upload `src/models/finetuning.py` and `configs/`
4. Run: `python finetuning.py`
5. Trained model is automatically saved to HuggingFace Hub

---

## Docker Deployment

```bash
docker build -f docker/Dockerfile -t medrag:latest .
docker-compose -f docker/docker-compose.yml up -d
# API:    http://localhost:8000
# MLflow: http://localhost:5000
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

*Built as a production-grade AI/ML portfolio project demonstrating end-to-end ML engineering:*
*data pipeline, vector search, LLM integration, REST API, monitoring, CI/CD, and containerization.*
