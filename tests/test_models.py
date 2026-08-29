"""
MedRAG Model Tests
Unit tests for BaselineRetriever and Evaluator.
"""

import numpy as np
import pandas as pd
import pytest


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_corpus():
    return pd.DataFrame({
        "content": [
            "Diabetes symptoms include frequent urination and thirst.",
            "Hypertension is high blood pressure affecting arteries.",
            "Asthma causes wheezing, coughing, and shortness of breath.",
            "Heart failure occurs when the heart cannot pump enough blood.",
            "Kidney disease involves reduced kidney function over time.",
        ],
        "question": [
            "What are diabetes symptoms?",
            "What is hypertension?",
            "What causes asthma?",
            "What is heart failure?",
            "What is kidney disease?",
        ],
        "source": ["NIH", "CDC", "Mayo", "WebMD", "MedlinePlus"],
        "chunk_id": [f"chunk_{i}" for i in range(5)],
    })


@pytest.fixture(scope="module")
def fitted_retriever(small_corpus):
    from src.models.baseline import BaselineRetriever
    retriever = BaselineRetriever()
    retriever.fit(small_corpus)
    return retriever


@pytest.fixture(scope="module")
def evaluator():
    from src.evaluation.metrics import Evaluator
    return Evaluator()


# ─── BaselineRetriever Tests ──────────────────────────────────────────────────

class TestBaselineRetriever:

    def test_fit(self, small_corpus):
        from src.models.baseline import BaselineRetriever
        retriever = BaselineRetriever()
        retriever.fit(small_corpus)
        assert retriever.tfidf_vectorizer is not None
        assert retriever.bm25 is not None
        assert retriever.corpus_df is not None

    def test_retrieve_tfidf_returns_top_k(self, fitted_retriever):
        results = fitted_retriever.retrieve_tfidf("diabetes symptoms", top_k=3)
        assert len(results) == 3
        assert all("content" in r for r in results)
        assert all("score" in r for r in results)
        assert all(isinstance(r["score"], float) for r in results)

    def test_retrieve_bm25_returns_top_k(self, fitted_retriever):
        results = fitted_retriever.retrieve_bm25("blood pressure heart", top_k=3)
        assert len(results) == 3
        assert all("content" in r for r in results)
        assert all("method" in r and r["method"] == "bm25" for r in results)

    def test_tfidf_scores_descending(self, fitted_retriever):
        results = fitted_retriever.retrieve_tfidf("kidney disease function", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_evaluate_returns_metrics(self, fitted_retriever, small_corpus):
        metrics = fitted_retriever.evaluate(small_corpus, method="bm25", sample_size=5)
        assert "hit@1" in metrics
        assert "hit@3" in metrics
        assert "hit@5" in metrics
        assert "mrr" in metrics
        assert all(0.0 <= v <= 1.0 for k, v in metrics.items() if isinstance(v, float))


# ─── Evaluator Tests ──────────────────────────────────────────────────────────

class TestEvaluator:

    def test_hit_at_k_found(self, evaluator):
        retrieved = ["chunk_1", "chunk_2", "chunk_3"]
        assert evaluator.hit_at_k(retrieved, "chunk_1", k=1) == 1
        assert evaluator.hit_at_k(retrieved, "chunk_2", k=1) == 0
        assert evaluator.hit_at_k(retrieved, "chunk_2", k=3) == 1

    def test_hit_at_k_not_found(self, evaluator):
        retrieved = ["chunk_1", "chunk_2"]
        assert evaluator.hit_at_k(retrieved, "chunk_99", k=5) == 0

    def test_mrr_first_position(self, evaluator):
        retrieved = ["chunk_1", "chunk_2", "chunk_3"]
        assert evaluator.mean_reciprocal_rank(retrieved, "chunk_1") == 1.0

    def test_mrr_third_position(self, evaluator):
        retrieved = ["chunk_1", "chunk_2", "chunk_3"]
        assert abs(evaluator.mean_reciprocal_rank(retrieved, "chunk_3") - 1/3) < 1e-9

    def test_mrr_not_found(self, evaluator):
        retrieved = ["chunk_1", "chunk_2"]
        assert evaluator.mean_reciprocal_rank(retrieved, "chunk_99") == 0.0

    def test_faithfulness_score_range(self, evaluator):
        question = "What is diabetes?"
        answer = "Diabetes involves high blood sugar levels."
        context = "Diabetes is a condition where blood sugar levels are elevated."
        score = evaluator.compute_faithfulness(question, answer, context)
        assert 0.0 <= score <= 1.0

    def test_faithfulness_empty_answer(self, evaluator):
        score = evaluator.compute_faithfulness("question", "", "some context")
        assert score == 0.0

    def test_answer_relevancy_range(self, evaluator):
        question = "What are symptoms of hypertension?"
        answer = "Hypertension symptoms include headaches and dizziness."
        score = evaluator.compute_answer_relevancy(question, answer)
        assert 0.0 <= score <= 1.0

    def test_answer_relevancy_similar_texts(self, evaluator):
        question = "What is diabetes?"
        answer_relevant = "Diabetes is a metabolic disease with high blood sugar."
        answer_irrelevant = "Cats are popular domestic animals that meow."
        score_relevant = evaluator.compute_answer_relevancy(question, answer_relevant)
        score_irrelevant = evaluator.compute_answer_relevancy(question, answer_irrelevant)
        assert score_relevant > score_irrelevant
