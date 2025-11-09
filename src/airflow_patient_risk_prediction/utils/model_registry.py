"""Helpers to locate the most recent trained model and metadata."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import joblib

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifacts:
    run_dir: Path
    models: Dict[str, Path]
    metrics: Dict[str, Dict[str, float]]
    feature_names: Optional[list[str]]


def find_latest_run(artifacts_root: Path) -> Optional[Path]:
    """Return the most recent timestamped run directory."""
    if not artifacts_root.exists():
        logger.warning("Artifacts root %s does not exist", artifacts_root)
        return None
    run_dirs = sorted(
        (p for p in artifacts_root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def load_artifacts(artifacts_root: Path) -> Optional[ModelArtifacts]:
    """Load the latest model artifacts and metrics."""

    latest_run = find_latest_run(artifacts_root)
    if latest_run is None:
        logger.warning("No model runs found under %s", artifacts_root)
        return None

    metrics_path = latest_run / "metrics.json"
    if not metrics_path.exists():
        logger.warning("Metrics file missing in %s", latest_run)
        metrics = {}
        feature_names = None
    else:
        payload = json.loads(metrics_path.read_text())
        metrics = payload.get("models", {})
        feature_names = payload.get("feature_names")

    model_files: Dict[str, Path] = {}
    for model_path in latest_run.glob("*.joblib"):
        model_files[model_path.stem] = model_path

    if not model_files:
        logger.warning("No model files found in %s", latest_run)

    return ModelArtifacts(
        run_dir=latest_run,
        models=model_files,
        metrics=metrics,
        feature_names=feature_names,
    )


def load_model(model_path: Path):
    """Load a persisted model artifact with joblib."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


__all__ = ["ModelArtifacts", "find_latest_run", "load_artifacts", "load_model"]

