"""
MedRAG Data Pipeline Tests
Unit tests for DataLoader, Preprocessor, and Embedder.
"""

import pytest
import pandas as pd
import numpy as np


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_df():
    """Small sample DataFrame mimicking MedQuAD structure."""
    return pd.DataFrame({
        "question": [
            "What are the symptoms of diabetes?",
            "How is hypertension treated?",
            "What causes heart disease?",
            "What is asthma?",
            "How do kidneys work?",
        ],
        "answer": [
            "Symptoms of diabetes include frequent urination, increased thirst, unexplained weight loss, and fatigue.",
            "Hypertension is treated with lifestyle changes such as diet and exercise, and medications including ACE inhibitors.",
            "Heart disease is caused by plaque buildup in arteries, high blood pressure, smoking, and diabetes.",
            "Asthma is a chronic respiratory condition characterized by airway inflammation and bronchospasm.",
            "Kidneys filter blood, removing waste products and excess fluid which are excreted as urine.",
        ],
        "source": ["NIH", "CDC", "Mayo", "WebMD", "MedlinePlus"],
        "chunk_id": [f"doc_{i}_chunk_0" for i in range(5)],
    })


@pytest.fixture(scope="module")
def preprocessor():
    from src.data.preprocessor import Preprocessor
    return Preprocessor()


@pytest.fixture(scope="module")
def embedder():
    from src.data.embedder import Embedder
    return Embedder()


# ─── DataLoader Tests ─────────────────────────────────────────────────────────

class TestDataLoader:

    def test_loader_init(self):
        from src.data.loader import DataLoader
        loader = DataLoader()
        assert loader is not None
        assert loader.config is not None

    def test_train_test_split(self, sample_df):
        from src.data.loader import DataLoader
        loader = DataLoader()
        train, val, test = loader.get_train_test_split(sample_df)
        total = len(train) + len(val) + len(test)
        assert total == len(sample_df)
        assert len(train) > 0
        assert len(test) > 0

    @pytest.mark.slow
    def test_load_small_sample(self):
        """Downloads 10 samples from HuggingFace (requires internet)."""
        from src.data.loader import DataLoader
        loader = DataLoader()
        df = loader.load_from_hub(max_samples=10)
        assert len(df) <= 10
        assert "question" in df.columns
        assert "answer" in df.columns


# ─── Preprocessor Tests ───────────────────────────────────────────────────────

class TestPreprocessor:

    def test_clean_text_removes_html(self, preprocessor):
        dirty = "<p>Hello <b>world</b>&amp;nbsp;</p>"
        clean = preprocessor.clean_text(dirty)
        assert "<" not in clean
        assert ">" not in clean
        assert "Hello" in clean

    def test_clean_text_whitespace(self, preprocessor):
        text = "  too   many   spaces  "
        result = preprocessor.clean_text(text)
        assert "  " not in result
        assert result == result.strip()

    def test_chunk_documents(self, preprocessor, sample_df):
        chunked = preprocessor.chunk_documents(sample_df, text_col="answer")
        assert len(chunked) >= len(sample_df)
        assert "content" in chunked.columns
        assert "chunk_id" in chunked.columns

    def test_no_empty_chunks(self, preprocessor, sample_df):
        chunked = preprocessor.chunk_documents(sample_df, text_col="answer")
        assert chunked["content"].str.strip().str.len().min() >= preprocessor.pp_cfg["min_chunk_length"]

    def test_chunk_has_metadata(self, preprocessor, sample_df):
        chunked = preprocessor.chunk_documents(sample_df, text_col="answer")
        assert "source" in chunked.columns
        assert "question" in chunked.columns


# ─── Embedder Tests ───────────────────────────────────────────────────────────

class TestEmbedder:

    def test_embedder_init(self, embedder):
        assert embedder is not None
        assert embedder.config is not None

    def test_load_model(self, embedder):
        model = embedder.load_model()
        assert model is not None

    def test_embed_single_text(self, embedder):
        embeddings = embedder.embed_texts(["What is diabetes?"])
        assert embeddings.shape == (1, 384)

    def test_embed_batch(self, embedder):
        texts = ["text one", "text two", "text three"]
        embeddings = embedder.embed_texts(texts)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == 384

    def test_embeddings_normalized(self, embedder):
        embeddings = embedder.embed_texts(["Normalized embedding test"])
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
