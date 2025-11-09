"""Training utilities for lifestyle risk models using processed BRFSS data."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier
except ModuleNotFoundError:  # pragma: no cover
    XGBClassifier = None  # type: ignore


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for training baseline models."""

    dataset_path: Path
    artifacts_dir: Path
    target_col: str = "HIGH_RISK"
    categorical_cols: Sequence[str] = (
        "_STATE",
        "FMONTH",
        "SEXVAR",
        "_AGEG5YR",
        "GENHLTH",
        "POORHLTH",
        "EXERANY2",
        "SMOKE100",
        "SMOKDAY2",
        "_SMOKER3",
        "_RFSMOK3",
        "ECIGNOW1",
        "DRNKANY5",
        "_BMI5CAT",
        "INCOME3",
    )


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {path}")
    logger.info("Loading processed dataset from %s", path)
    return pd.read_parquet(path)


def _build_preprocessor(categorical_cols: Sequence[str], numeric_cols: Sequence[str]) -> ColumnTransformer:
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, list(categorical_cols)),
            ("numeric", numeric_transformer, list(numeric_cols)),
        ]
    )


def _train_logistic(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def _train_xgb(preprocessor: ColumnTransformer) -> Pipeline | None:
    if XGBClassifier is None:
        logger.warning("xgboost is not installed; skipping XGBClassifier training")
        return None
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="auc",
                    random_state=42,
                ),
            ),
        ]
    )


def _evaluate(model: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series) -> Dict[str, float]:
    proba = model.predict_proba(X_valid)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_valid, proba)),
        "accuracy": float(accuracy_score(y_valid, pred)),
    }


def _classification_report(model: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series) -> str:
    pred = model.predict(X_valid)
    return classification_report(y_valid, pred, digits=3)


def train_models(config: TrainConfig) -> Dict[str, Dict[str, float]]:
    """Train baseline models and persist artifacts/metrics."""

    df = _load_dataset(config.dataset_path)
    feature_cols = [col for col in df.columns if col != config.target_col]
    numeric_cols = [col for col in feature_cols if col not in config.categorical_cols]

    X = df[feature_cols]
    y = df[config.target_col].astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = _build_preprocessor(config.categorical_cols, numeric_cols)

    models: Dict[str, Pipeline] = {}
    artifacts_dir = config.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Dict[str, float]] = {}
    feature_names: List[str] | None = None

    log_reg_model = _train_logistic(preprocessor)
    log_reg_model.fit(X_train, y_train)
    metrics["logistic_regression"] = _evaluate(log_reg_model, X_valid, y_valid)
    logger.info("Logistic Regression metrics: %s", metrics["logistic_regression"])
    logger.debug("Classification report:\n%s", _classification_report(log_reg_model, X_valid, y_valid))
    models["logistic_regression"] = log_reg_model
    feature_names = list(log_reg_model.named_steps["preprocess"].get_feature_names_out())

    xgb_model = _train_xgb(preprocessor)
    if xgb_model is not None:
        xgb_model.fit(X_train, y_train)
        metrics["xgboost"] = _evaluate(xgb_model, X_valid, y_valid)
        logger.info("XGBoost metrics: %s", metrics["xgboost"])
        logger.debug("XGBoost classification report:\n%s", _classification_report(xgb_model, X_valid, y_valid))
        models["xgboost"] = xgb_model

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = artifacts_dir / timestamp
    run_dir.mkdir(exist_ok=True, parents=True)

    for name, model in models.items():
        model_file = run_dir / f"{name}.joblib"
        joblib.dump(model, model_file)
        logger.info("Saved %s model to %s", name, model_file)

    metrics_file = run_dir / "metrics.json"
    metrics_payload = {
        "timestamp": timestamp,
        "models": metrics,
        "feature_names": feature_names,
    }
    metrics_file.write_text(json.dumps(metrics_payload, indent=2))
    logger.info("Persisted metrics to %s", metrics_file)

    return metrics


def parse_args(argv: Sequence[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Train baseline lifestyle risk models.")
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to processed dataset parquet.")
    parser.add_argument("--artifacts-dir", type=Path, required=True, help="Directory to store model artifacts.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    config = TrainConfig(dataset_path=args.dataset_path, artifacts_dir=args.artifacts_dir)
    train_models(config)


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["TrainConfig", "train_models", "main"]

