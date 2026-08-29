"""
MedRAG Preprocessor
Cleans, chunks, and deduplicates medical documents.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

import nltk
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class Preprocessor:
    """Cleans, chunks, and deduplicates medical documents for RAG indexing."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.pp_cfg = self.config["preprocessing"]
        self.paths = self.config["paths"]
        self.processed_dir = Path(self.paths["data_processed"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ─── Text Cleaning ────────────────────────────────────────────────────────

    def clean_text(self, text: str) -> str:
        """Remove HTML tags, fix encoding, normalise whitespace."""
        if not isinstance(text, str):
            return ""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Fix common HTML entities
        html_entities = {
            "&amp;": "&", "&lt;": "<", "&gt;": ">",
            "&nbsp;": " ", "&quot;": '"', "&#39;": "'",
        }
        for entity, char in html_entities.items():
            text = text.replace(entity, char)
        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)
        # Remove special chars (keep punctuation)
        text = re.sub(r"[^\w\s.,;:!?()-]", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ─── Chunking ─────────────────────────────────────────────────────────────

    def chunk_text(self, text: str) -> List[str]:
        """Split a long text into overlapping chunks by word count."""
        chunk_size = self.pp_cfg["chunk_size"]
        overlap = self.pp_cfg["chunk_overlap"]
        min_len = self.pp_cfg["min_chunk_length"]

        words = text.split()
        if len(words) <= chunk_size:
            return [text] if len(words) >= min_len else []

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if len(chunk.split()) >= min_len:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def chunk_documents(self, df: pd.DataFrame, text_col: str = "answer") -> pd.DataFrame:
        """Chunk all documents in the dataframe."""
        records = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking documents"):
            text = self.clean_text(str(row.get(text_col, "")))
            chunks = self.chunk_text(text)
            for ci, chunk in enumerate(chunks):
                record = {
                    "chunk_id": f"doc_{idx}_chunk_{ci}",
                    "content": chunk,
                    "question": self.clean_text(str(row.get("question", ""))),
                    "source": str(row.get("source", "MedQuAD")),
                    "original_idx": idx,
                    "chunk_index": ci,
                }
                records.append(record)
        result = pd.DataFrame(records)
        logger.info(f"Created {len(result):,} chunks from {len(df):,} documents.")
        return result

    # ─── Deduplication ────────────────────────────────────────────────────────

    def deduplicate(
        self,
        df: pd.DataFrame,
        text_col: str = "content",
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Remove near-duplicate chunks using TF-IDF cosine similarity."""
        threshold = threshold or self.pp_cfg["similarity_threshold"]
        logger.info(f"Deduplicating {len(df):,} chunks (threshold={threshold}) ...")

        # Exact duplicates first
        before = len(df)
        df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
        logger.info(f"Exact duplicates removed: {before - len(df):,}")

        # Near-duplicate removal via TF-IDF (batch to avoid memory issues)
        if len(df) < 2:
            return df

        batch_size = 5000
        to_keep = set(range(len(df)))

        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_texts = df[text_col].iloc[batch_start:batch_end].tolist()

            vectorizer = TfidfVectorizer(max_features=10000)
            tfidf_matrix = vectorizer.fit_transform(batch_texts)
            sim_matrix = cosine_similarity(tfidf_matrix)

            for i in range(len(batch_texts)):
                global_i = batch_start + i
                if global_i not in to_keep:
                    continue
                for j in range(i + 1, len(batch_texts)):
                    global_j = batch_start + j
                    if sim_matrix[i, j] >= threshold:
                        to_keep.discard(global_j)

        result = df.iloc[sorted(to_keep)].reset_index(drop=True)
        logger.info(f"Near-duplicates removed: {len(df) - len(result):,}")
        return result

    # ─── Full Pipeline ─────────────────────────────────────────────────────────

    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full pipeline: clean -> chunk -> deduplicate."""
        logger.info("Starting preprocessing pipeline ...")

        # Clean text columns
        logger.info("Cleaning text ...")
        for col in ["answer", "question"]:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)

        # Chunk
        chunked_df = self.chunk_documents(df)

        # Deduplicate
        if self.pp_cfg.get("remove_duplicates", True):
            chunked_df = self.deduplicate(chunked_df)

        logger.info(f"Preprocessing complete. Final chunks: {len(chunked_df):,}")
        return chunked_df

    def save_processed(self, df: pd.DataFrame, filename: str = "chunks.parquet") -> Path:
        """Save processed chunks to disk."""
        fpath = self.processed_dir / filename
        df.to_parquet(fpath, index=False)
        logger.info(f"Saved {len(df):,} chunks to {fpath}")
        return fpath


if __name__ == "__main__":
    from src.data.loader import DataLoader

    loader = DataLoader()
    try:
        raw_df = loader.load_from_local()
    except FileNotFoundError:
        raw_df = loader.load_from_hub()
        loader.save_raw(raw_df)

    preprocessor = Preprocessor()
    processed_df = preprocessor.process_dataset(raw_df)
    preprocessor.save_processed(processed_df)

    print(f"\nStats:")
    print(f"  Raw documents : {len(raw_df):,}")
    print(f"  Final chunks  : {len(processed_df):,}")
    print(processed_df.head(3))
