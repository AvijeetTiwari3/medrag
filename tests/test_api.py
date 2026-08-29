"""
MedRAG API Tests
Tests for FastAPI endpoints using httpx TestClient.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ─── Mock RAG Pipeline ────────────────────────────────────────────────────────

def make_mock_rag():
    mock = MagicMock()
    mock.query.return_value = {
        "question": "What is diabetes?",
        "answer": "Diabetes is a condition characterized by high blood sugar levels.",
        "sources": [
            {
                "content": "Diabetes involves elevated blood glucose.",
                "source": "NIH",
                "score": 0.87,
                "chunk_id": "chunk_0",
            }
        ],
        "latency_ms": 350.0,
        "model": "phi3:mini",
        "method": "rag",
    }
    mock._is_ollama_available.return_value = True
    mock.embedder.get_collection_stats.return_value = {
        "total_documents": 1000,
        "collection_name": "medrag_docs",
    }
    return mock


# ─── Client Fixture ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from src.api.main import app

    mock_rag = make_mock_rag()

    with patch("src.api.main.RAGPipeline", return_value=mock_rag):
        with TestClient(app) as c:
            c.app.state.rag_pipeline = mock_rag
            yield c


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestRootEndpoint:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_endpoints(self, client):
        resp = client.get("/")
        data = resp.json()
        assert "endpoints" in data
        assert "query" in data["endpoints"]


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_has_required_fields(self, client):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "status" in data
        assert "version" in data
        assert "chroma_docs" in data
        assert "ollama_available" in data

    def test_health_status_is_string(self, client):
        resp = client.get("/api/v1/health")
        assert isinstance(resp.json()["status"], str)


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200

    def test_metrics_has_required_fields(self, client):
        resp = client.get("/api/v1/metrics")
        data = resp.json()
        assert "total_queries" in data
        assert "avg_latency_ms" in data
        assert "error_rate" in data


class TestQueryEndpoint:
    def test_query_missing_body_returns_422(self, client):
        resp = client.post("/api/v1/query", json={})
        assert resp.status_code == 422

    def test_query_too_short_returns_422(self, client):
        resp = client.post("/api/v1/query", json={"question": "Hi"})
        assert resp.status_code == 422

    def test_query_too_long_returns_422(self, client):
        resp = client.post("/api/v1/query", json={"question": "x" * 501})
        assert resp.status_code == 422

    def test_valid_query_returns_200(self, client):
        resp = client.post(
            "/api/v1/query",
            json={"question": "What are the symptoms of diabetes?", "top_k": 3},
        )
        assert resp.status_code == 200

    def test_valid_query_response_structure(self, client):
        resp = client.post(
            "/api/v1/query",
            json={"question": "What is hypertension?"},
        )
        data = resp.json()
        assert "question" in data
        assert "answer" in data
        assert "sources" in data
        assert "latency_ms" in data
        assert isinstance(data["sources"], list)

    def test_docs_count_endpoint(self, client):
        resp = client.get("/api/v1/docs-count")
        assert resp.status_code == 200
        assert "total_documents" in resp.json()
