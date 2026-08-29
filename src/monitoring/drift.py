"""
MedRAG Drift Monitor
Detects query distribution drift using Evidently AI.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class DriftMonitor:
    """Monitors query drift using Evidently AI and sentence-transformer embeddings."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.mon_cfg = self.config["monitoring"]
        self.report_dir = Path(self.mon_cfg["report_output_dir"])
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_model = None

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            model_name = self.config["embedding"]["model_name"]
            self._embedding_model = SentenceTransformer(model_name)
        return self._embedding_model

    def _featurize(self, texts: list) -> pd.DataFrame:
        """Convert query texts into feature DataFrame for drift detection."""
        model = self._get_embedding_model()
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        features = {
            "text_length": [len(t) for t in texts],
            "word_count": [len(t.split()) for t in texts],
            "question_mark": [1 if "?" in t else 0 for t in texts],
            "emb_mean": embeddings.mean(axis=1).tolist(),
            "emb_std": embeddings.std(axis=1).tolist(),
            "emb_norm": np.linalg.norm(embeddings, axis=1).tolist(),
        }

        # Add top 10 PCA-like embedding dims as features
        for dim in range(min(10, embeddings.shape[1])):
            features[f"emb_dim_{dim}"] = embeddings[:, dim].tolist()

        return pd.DataFrame(features)

    def save_reference_data(self, queries: list, path: Optional[str] = None) -> Path:
        """Save reference query distribution to CSV for future drift comparison."""
        fpath = Path(path or self.mon_cfg["reference_data_path"])
        fpath.parent.mkdir(parents=True, exist_ok=True)

        df = self._featurize(queries)
        df.to_csv(fpath, index=False)
        logger.info(f"Reference data saved: {len(df)} queries -> {fpath}")
        return fpath

    def load_current_data(self, log_path: Optional[str] = None) -> pd.DataFrame:
        """Load recent query log and featurize."""
        default_log = "logs/query_log.csv"
        fpath = Path(log_path or default_log)

        if not fpath.exists():
            logger.warning(f"Query log not found at {fpath}. Returning empty DataFrame.")
            return pd.DataFrame()

        df = pd.read_csv(fpath)
        if "question" not in df.columns:
            raise ValueError("Query log must have a 'question' column.")

        texts = df["question"].dropna().tolist()
        return self._featurize(texts)

    def compute_text_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> dict:
        """Compute feature drift between reference and current query distributions."""
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_df, current_data=current_df)

            report_dict = report.as_dict()
            drift_results = report_dict.get("metrics", [{}])[0]
            drift_score = drift_results.get("result", {}).get("dataset_drift", False)
            share_drifted = drift_results.get("result", {}).get("share_of_drifted_columns", 0.0)

            return {
                "drift_detected": bool(drift_score),
                "drift_share": float(share_drifted),
                "n_reference": len(reference_df),
                "n_current": len(current_df),
            }

        except ImportError:
            logger.warning("Evidently not installed. Falling back to manual drift check.")
            return self._manual_drift_check(reference_df, current_df)
        except Exception as e:
            logger.error(f"Drift computation failed: {e}")
            return self._manual_drift_check(reference_df, current_df)

    def _manual_drift_check(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> dict:
        """Fallback: simple mean-shift drift detection."""
        numeric_cols = reference_df.select_dtypes(include=[np.number]).columns
        drifted = 0
        for col in numeric_cols:
            ref_mean = reference_df[col].mean()
            cur_mean = current_df[col].mean()
            ref_std = reference_df[col].std() + 1e-9
            if abs(cur_mean - ref_mean) / ref_std > 2.0:
                drifted += 1

        share = drifted / max(len(numeric_cols), 1)
        return {
            "drift_detected": share > self.mon_cfg["alert_threshold"],
            "drift_share": round(share, 4),
            "n_reference": len(reference_df),
            "n_current": len(current_df),
        }

    def generate_report(self, output_path: Optional[str] = None) -> Path:
        """Generate a full Evidently HTML drift report."""
        ref_path = Path(self.mon_cfg["reference_data_path"])
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Reference data not found at {ref_path}. "
                "Run save_reference_data() first."
            )

        reference_df = pd.read_csv(ref_path)
        current_df = self.load_current_data()

        if current_df.empty:
            raise ValueError("No current query data found for drift report.")

        output_path = output_path or str(self.report_dir / "drift_report.html")

        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_df, current_data=current_df)
            report.save_html(output_path)
            logger.info(f"Drift report saved to: {output_path}")
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            output_path = str(self.report_dir / "drift_summary.txt")
            result = self._manual_drift_check(reference_df, current_df)
            Path(output_path).write_text(str(result))

        return Path(output_path)

    def get_drift_score(self) -> float:
        """Quick drift score (0.0 = no drift, 1.0 = maximum drift)."""
        try:
            ref_df = pd.read_csv(self.mon_cfg["reference_data_path"])
            cur_df = self.load_current_data()
            if cur_df.empty:
                return 0.0
            result = self.compute_text_drift(ref_df, cur_df)
            return result.get("drift_share", 0.0)
        except Exception as e:
            logger.warning(f"Could not compute drift score: {e}")
            return 0.0

    def check_and_alert(self, threshold: Optional[float] = None) -> bool:
        """Check drift and print alert if score exceeds threshold."""
        threshold = threshold or self.mon_cfg["alert_threshold"]
        score = self.get_drift_score()

        if score > threshold:
            logger.warning(
                f"DRIFT ALERT: drift_share={score:.3f} exceeds threshold={threshold}.\n"
                "Consider retraining or updating reference data."
            )
            return True

        logger.info(f"Drift check passed: drift_share={score:.3f} (threshold={threshold})")
        return False


if __name__ == "__main__":
    monitor = DriftMonitor()

    # Create sample reference data if query log exists
    sample_queries = [
        "What are symptoms of diabetes?",
        "How is hypertension treated?",
        "What causes heart disease?",
        "What is the treatment for asthma?",
        "How do I manage high blood pressure?",
    ]
    monitor.save_reference_data(sample_queries)
    alert = monitor.check_and_alert()
    print(f"Drift alert triggered: {alert}")
