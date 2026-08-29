"""
MedRAG Embedder
Generates embeddings and manages ChromaDB vector index.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
import pandas as pd
import yaml
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class Embedder:
    """Embeds documents using sentence-transformers and indexes them in ChromaDB."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.emb_cfg = self.config["embedding"]
        self.chroma_cfg = self.config["chroma"]
        self.paths = self.config["paths"]

        self.model: Optional[SentenceTransformer] = None
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[chromadb.Collection] = None

    # ─── Model ────────────────────────────────────────────────────────────────

    def load_model(self) -> SentenceTransformer:
        """Load the sentence-transformers embedding model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.emb_cfg['model_name']}")
            self.model = SentenceTransformer(
                self.emb_cfg["model_name"],
                device=self.emb_cfg.get("device", "cpu"),
            )
        return self.model

    # ─── ChromaDB ─────────────────────────────────────────────────────────────

    def _get_client(self) -> chromadb.PersistentClient:
        if self.client is None:
            db_path = str(Path(self.paths["chroma_db"]))
            Path(db_path).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=db_path)
        return self.client

    def _get_collection(self) -> chromadb.Collection:
        if self.collection is None:
            client = self._get_client()
            self.collection = client.get_or_create_collection(
                name=self.chroma_cfg["collection_name"],
                metadata={"hnsw:space": self.chroma_cfg["distance_metric"]},
            )
        return self.collection

    # ─── Embedding ────────────────────────────────────────────────────────────

    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Generate embeddings for a list of texts in batches."""
        model = self.load_model()
        bs = batch_size or self.emb_cfg.get("batch_size", 64)
        logger.info(f"Embedding {len(texts):,} texts (batch_size={bs}) ...")

        all_embeddings = []
        for i in tqdm(range(0, len(texts), bs), desc="Embedding batches"):
            batch = texts[i : i + bs]
            embs = model.encode(
                batch,
                normalize_embeddings=self.emb_cfg.get("normalize", True),
                show_progress_bar=False,
            )
            all_embeddings.append(embs)

        return np.vstack(all_embeddings)

    # ─── Index Building ───────────────────────────────────────────────────────

    def build_index(self, df: pd.DataFrame, batch_size: int = 500) -> None:
        """Ingest all document chunks into ChromaDB."""
        collection = self._get_collection()

        texts = df["content"].tolist()
        logger.info(f"Building index for {len(texts):,} chunks ...")

        embeddings = self.embed_texts(texts)

        ids = df["chunk_id"].tolist() if "chunk_id" in df.columns else [str(uuid.uuid4()) for _ in texts]
        metadatas = []
        for _, row in df.iterrows():
            meta = {
                "source": str(row.get("source", "unknown")),
                "question": str(row.get("question", ""))[:500],
                "chunk_index": int(row.get("chunk_index", 0)),
            }
            metadatas.append(meta)

        # Insert in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Inserting into ChromaDB"):
            batch_ids = ids[i : i + batch_size]
            batch_embs = embeddings[i : i + batch_size].tolist()
            batch_docs = texts[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embs,
                documents=batch_docs,
                metadatas=batch_meta,
            )

        logger.info(f"Index built. Total documents in collection: {collection.count():,}")

    # ─── Querying ─────────────────────────────────────────────────────────────

    def query(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """Query ChromaDB and return top_k relevant chunks with scores."""
        model = self.load_model()
        collection = self._get_collection()

        query_emb = model.encode(
            [text], normalize_embeddings=self.emb_cfg.get("normalize", True)
        ).tolist()

        results = collection.query(
            query_embeddings=query_emb,
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results.get("ids", [[]])[0]
        # Convert L2/cosine distance to similarity score
        scores = [1 - d for d in distances]

        # Inject chunk_id into each metadata for downstream evaluation
        for i, meta in enumerate(metas):
            if i < len(ids):
                meta["chunk_id"] = ids[i]

        return {
            "documents": docs,
            "metadatas": metas,
            "scores": scores,
            "ids": ids,
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return stats about the ChromaDB collection."""
        collection = self._get_collection()
        return {
            "collection_name": self.chroma_cfg["collection_name"],
            "total_documents": collection.count(),
            "embedding_model": self.emb_cfg["model_name"],
            "embedding_dim": self.emb_cfg.get("embedding_dim", 384),
        }


if __name__ == "__main__":
    from src.data.loader import DataLoader
    from src.data.preprocessor import Preprocessor

    loader = DataLoader()
    try:
        raw_df = loader.load_from_local()
    except FileNotFoundError:
        raw_df = loader.load_from_hub()
        loader.save_raw(raw_df)

    preprocessor = Preprocessor()
    try:
        chunks_df = pd.read_parquet("data/processed/chunks.parquet")
        logger.info(f"Loaded {len(chunks_df):,} chunks from disk.")
    except FileNotFoundError:
        chunks_df = preprocessor.process_dataset(raw_df)
        preprocessor.save_processed(chunks_df)

    embedder = Embedder()
    embedder.build_index(chunks_df)

    stats = embedder.get_collection_stats()
    print(f"\nChromaDB Stats: {stats}")

    # Test query
    result = embedder.query("What are the symptoms of diabetes?", top_k=3)
    print(f"\nTop 3 results for test query:")
    for i, (doc, score) in enumerate(zip(result["documents"], result["scores"])):
        print(f"\n[{i+1}] Score: {score:.3f}")
        print(f"     {doc[:200]}...")
