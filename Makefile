.PHONY: help install install-dev ingest run demo test test-fast lint mlflow docker-build docker-run docker-stop monitor clean

help:
	@echo ""
	@echo "  MedRAG - Available Commands"
	@echo "  ========================================"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install all dependencies (dev + prod)"
	@echo "  make ingest         Download dataset and build ChromaDB index"
	@echo "  make run            Start FastAPI server on http://localhost:8000"
	@echo "  make demo           Start Gradio demo on http://localhost:7860"
	@echo "  make test           Run all tests with coverage"
	@echo "  make test-fast      Run only fast tests (skip slow/network tests)"
	@echo "  make lint           Run ruff linter and formatter"
	@echo "  make mlflow         Open MLflow UI on http://localhost:5000"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-run     Run full stack with docker-compose"
	@echo "  make docker-stop    Stop docker-compose stack"
	@echo "  make monitor        Run drift detection report"
	@echo "  make clean          Remove cache and temp files"
	@echo ""

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

ingest:
	@echo ">>> Downloading MedQuAD dataset ..."
	python -m src.data.loader
	@echo ">>> Preprocessing documents ..."
	python -m src.data.preprocessor
	@echo ">>> Building ChromaDB index ..."
	python -m src.data.embedder
	@echo ">>> Ingestion complete!"

run:
	@echo ">>> Starting MedRAG API on http://localhost:8000"
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

demo:
	@echo ">>> Starting Gradio demo on http://localhost:7860"
	python app.py

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow"

lint:
	ruff check src/ tests/ --fix
	ruff format src/ tests/

mlflow:
	mlflow ui --port 5000

docker-build:
	docker build -f docker/Dockerfile -t medrag:latest .
	@echo ">>> Image medrag:latest built."

docker-run:
	docker-compose -f docker/docker-compose.yml up -d
	@echo ">>> Services running:"
	@echo "    API:    http://localhost:8000"
	@echo "    Docs:   http://localhost:8000/docs"
	@echo "    MLflow: http://localhost:5000"

docker-stop:
	docker-compose -f docker/docker-compose.yml down

monitor:
	python -m src.monitoring.drift

clean:
	find . -type f -name "*.pyc" -delete 2>/dev/null; \
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; \
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null; \
	find . -name ".coverage" -delete 2>/dev/null; \
	echo "Cleaned up!"
