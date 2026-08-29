"""
MedRAG Data Loader
Downloads and loads the MedQuAD dataset from HuggingFace.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DataLoader:
    """Loads and manages the MedQuAD medical QA dataset."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.ds_config = self.config["dataset"]
        self.paths = self.config["paths"]
        self.raw_dir = Path(self.paths["data_raw"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def load_from_hub(self, max_samples: Optional[int] = None) -> pd.DataFrame:
        """Download MedQuAD dataset from HuggingFace Hub."""
        dataset_name = self.ds_config["name"]
        split = self.ds_config["split"]
        n = max_samples or self.ds_config["max_samples"]

        logger.info(f"Downloading dataset '{dataset_name}' (split='{split}') ...")
        try:
            ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
            df = ds.to_pandas()
        except Exception:
            logger.warning("lavita/MedQuAD failed, falling back to 'keivalya/MedQuAD_MedInfo'")
            ds = load_dataset("keivalya/MedQuAD_MedInfo", split="train", trust_remote_code=True)
            df = ds.to_pandas()
            # Standardise column names
            col_map = {}
            for c in df.columns:
                if "question" in c.lower():
                    col_map[c] = "question"
                elif "answer" in c.lower():
                    col_map[c] = "answer"
            df = df.rename(columns=col_map)
            if "source" not in df.columns:
                df["source"] = "MedQuAD"

        # Ensure required columns exist
        required = [
            self.ds_config["question_column"],
            self.ds_config["text_column"],
        ]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found. Available: {df.columns.tolist()}")

        df = df.dropna(subset=required)
        df = df[df[self.ds_config["text_column"]].str.strip().astype(bool)]
        df = df[df[self.ds_config["question_column"]].str.strip().astype(bool)]

        if n and len(df) > n:
            df = df.sample(n=n, random_state=self.ds_config["seed"]).reset_index(drop=True)

        logger.info(f"Loaded {len(df):,} samples.")
        return df

    def load_from_local(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load dataset from local parquet or CSV file."""
        fpath = Path(path) if path else self.raw_dir / "medquad.parquet"
        if not fpath.exists():
            raise FileNotFoundError(f"Local data not found at {fpath}. Run load_from_hub() first.")
        logger.info(f"Loading from local file: {fpath}")
        if fpath.suffix == ".parquet":
            return pd.read_parquet(fpath)
        return pd.read_csv(fpath)

    def get_train_test_split(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataframe into train / val / test sets."""
        if df is None:
            df = self.load_from_local()

        seed = self.ds_config["seed"]
        test_size = self.ds_config["test_size"]
        val_size = self.ds_config["val_size"]

        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
        train_df, val_df = train_test_split(
            train_df, test_size=val_size / (1 - test_size), random_state=seed
        )

        logger.info(
            f"Split sizes -> train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,}"
        )
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def get_sample(self, n: int = 10, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Return n random samples from the dataset."""
        if df is None:
            df = self.load_from_local()
        return df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)

    def save_raw(self, df: pd.DataFrame, filename: str = "medquad.parquet") -> Path:
        """Save raw dataframe to disk."""
        fpath = self.raw_dir / filename
        df.to_parquet(fpath, index=False)
        logger.info(f"Saved {len(df):,} rows to {fpath}")
        return fpath


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_from_hub()
    loader.save_raw(df)
    train, val, test = loader.get_train_test_split(df)
    logger.info("Data loading complete.")
    print(df.head(3))
