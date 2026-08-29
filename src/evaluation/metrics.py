"""
MedRAG Evaluation Metrics
Faithfulness, answer relevancy, retrieval hit rates, and MRR.
"""

import logging
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class Evaluator:
    """Evaluation metrics for retrieval and generation quality."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.eval_cfg = self.config["evaluation"]
        self._embedding_model = None

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            model_name = self.config["embedding"]["model_name"]
            self._embedding_model = SentenceTransformer(model_name)
        return self._embedding_model

    # ─── Retrieval Metrics ────────────────────────────────────────────────────

    def hit_at_k(self, retrieved_ids: List[str], relevant_id: str, k: int) -> int:
        """Return 1 if relevant_id is in top-k retrieved_ids, else 0."""
        return 1 if relevant_id in retrieved_ids[:k] else 0

    def mean_reciprocal_rank(self, retrieved_ids: List[str], relevant_id: str) -> float:
        """Compute MRR: 1/rank if found, else 0."""
        if relevant_id in retrieved_ids:
            rank = retrieved_ids.index(relevant_id) + 1
            return 1.0 / rank
        return 0.0

    def precision_at_k(self, retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Precision@K: fraction of top-k retrieved that are relevant."""
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        return len(retrieved_k & relevant_set) / k if k > 0 else 0.0

    # ─── Generation Metrics ───────────────────────────────────────────────────

    def compute_faithfulness(self, question: str, answer: str, context: str) -> float:
        """
        Keyword-overlap faithfulness: fraction of answer content words found in context.
        (Proxy metric; real faithfulness uses LLM-as-judge.)
        """
        import re
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "of", "in", "to", "and", "or", "for", "with", "on", "at",
            "this", "that", "it", "its", "as", "by", "from", "not",
        }
        def content_words(text: str) -> set:
            words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            return set(w for w in words if w not in stop_words)

        answer_words = content_words(answer)
        context_words = content_words(context)

        if not answer_words:
            return 0.0
        overlap = answer_words & context_words
        return round(len(overlap) / len(answer_words), 4)

    def compute_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Answer relevancy: cosine similarity between question and answer embeddings.
        Higher = more relevant answer.
        """
        if not question.strip() or not answer.strip():
            return 0.0

        model = self._get_embedding_model()
        embeddings = model.encode([question, answer], normalize_embeddings=True)
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        return round(max(0.0, similarity), 4)

    # ─── Full Evaluation Runs ─────────────────────────────────────────────────

    def evaluate_retrieval(
        self,
        retriever,
        test_df: pd.DataFrame,
        method: str = "bm25",
        sample_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run retrieval evaluation (Hit@k, MRR) on test set."""
        if sample_size:
            test_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)

        hit1, hit3, hit5, mrr_vals = [], [], [], []

        retrieve_fn = getattr(retriever, f"retrieve_{method}", None)
        if retrieve_fn is None:
            raise ValueError(f"Retriever has no method 'retrieve_{method}'")

        for _, row in test_df.iterrows():
            question = str(row.get("question", ""))
            relevant_id = str(row.get("chunk_id", ""))
            results = retrieve_fn(question, top_k=5)
            retrieved_ids = [r["chunk_id"] for r in results]

            hit1.append(self.hit_at_k(retrieved_ids, relevant_id, 1))
            hit3.append(self.hit_at_k(retrieved_ids, relevant_id, 3))
            hit5.append(self.hit_at_k(retrieved_ids, relevant_id, 5))
            mrr_vals.append(self.mean_reciprocal_rank(retrieved_ids, relevant_id))

        metrics = {
            "hit_at_1": float(np.mean(hit1)),
            "hit_at_3": float(np.mean(hit3)),
            "hit_at_5": float(np.mean(hit5)),
            "mrr": float(np.mean(mrr_vals)),
            "method": method,
            "n_eval": len(test_df),
        }
        logger.info(f"[{method.upper()}] {metrics}")
        return metrics

    def evaluate_rag(
        self,
        rag_pipeline,
        test_df: pd.DataFrame,
        sample_size: int = 100,
    ) -> Dict[str, float]:
        """Run full RAG evaluation: retrieval + faithfulness + relevancy."""
        test_sample = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)

        faithfulness_vals, relevancy_vals, latencies = [], [], []
        hit1, hit3, hit5, mrr_vals = [], [], [], []

        for _, row in test_sample.iterrows():
            question = str(row.get("question", ""))
            relevant_id = str(row.get("chunk_id", ""))

            result = rag_pipeline.query(question)
            latencies.append(result["latency_ms"])

            retrieved_ids = [s.get("chunk_id", "") for s in result["sources"]]
            hit1.append(self.hit_at_k(retrieved_ids, relevant_id, 1))
            hit3.append(self.hit_at_k(retrieved_ids, relevant_id, 3))
            hit5.append(self.hit_at_k(retrieved_ids, relevant_id, 5))
            mrr_vals.append(self.mean_reciprocal_rank(retrieved_ids, relevant_id))

            context = " ".join([s["content"] for s in result["sources"]])
            faithfulness_vals.append(
                self.compute_faithfulness(question, result["answer"], context)
            )
            relevancy_vals.append(
                self.compute_answer_relevancy(question, result["answer"])
            )

        return {
            "hit_at_1": float(np.mean(hit1)),
            "hit_at_3": float(np.mean(hit3)),
            "hit_at_5": float(np.mean(hit5)),
            "mrr": float(np.mean(mrr_vals)),
            "faithfulness": float(np.mean(faithfulness_vals)),
            "answer_relevancy": float(np.mean(relevancy_vals)),
            "avg_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "n_eval": len(test_sample),
        }

    def compare_methods(self, *method_results: Dict[str, Any]) -> None:
        """Print a formatted comparison table of evaluation results."""
        if not method_results:
            return

        print("\n" + "=" * 70)
        print(f"{'Metric':<20}", end="")
        for r in method_results:
            print(f"  {r.get('method', 'unknown').upper():<14}", end="")
        print()
        print("-" * 70)

        metric_keys = ["hit_at_1", "hit_at_3", "hit_at_5", "mrr", "faithfulness", "answer_relevancy", "avg_latency_ms"]
        for key in metric_keys:
            vals = [r.get(key) for r in method_results]
            if any(v is not None for v in vals):
                print(f"{key:<20}", end="")
                for v in vals:
                    print(f"  {v:.4f}      " if v is not None else f"  {'N/A':<14}", end="")
                print()
        print("=" * 70 + "\n")

    def log_to_mlflow(self, metrics: Dict[str, Any], run_name: str = "evaluation") -> None:
        """Log evaluation metrics to MLflow."""
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("medrag_experiments")
        with mlflow.start_run(run_name=run_name):
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
        logger.info(f"Metrics logged to MLflow run: {run_name}")
