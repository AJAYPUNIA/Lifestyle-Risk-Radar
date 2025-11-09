"""Utilities to build modeling-ready datasets from staged BRFSS tables."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for transforming staged BRFSS data into modeling features."""

    staged_dir: Path
    output_path: Path
    chronic_groups: Dict[str, Sequence[str]] = field(
        default_factory=lambda: {
            "cardio": ("CVDINFR4", "CVDCRHD4", "CVDSTRK3"),
            "metabolic": ("DIABETE4",),
            "respiratory": ("ASTHMA3",),
            "renal": ("CHCKDNY2",),
        }
    )
    essential_cols: Sequence[str] = (
        "GENHLTH",
        "_SMOKER3",
        "DRNKANY5",
        "EXERANY2",
        "_AGEG5YR",
    )
    numerical_impute_cols: Sequence[str] = (
        "_BMI5",
        "AVEDRNK3",
        "DRNK3GE5",
        "MAXDRNKS",
        "WEIGHT2",
        "HEIGHT3",
        "PHYSHLTH",
        "MENTHLTH",
        "_DRNKWK1",
    )
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
    feature_cols: Sequence[str] = (
        "_STATE",
        "FMONTH",
        "SEXVAR",
        "_AGEG5YR",
        "GENHLTH",
        "PHYSHLTH",
        "MENTHLTH",
        "POORHLTH",
        "EXERANY2",
        "SMOKE100",
        "SMOKDAY2",
        "_SMOKER3",
        "_RFSMOK3",
        "ECIGNOW1",
        "DRNKANY5",
        "AVEDRNK3",
        "DRNK3GE5",
        "MAXDRNKS",
        "_DRNKWK1",
        "_BMI5",
        "_BMI5CAT",
        "WEIGHT2",
        "HEIGHT3",
        "INCOME3",
    )


def _load_staged_parts(staged_dir: Path) -> pd.DataFrame:
    parts = sorted(staged_dir.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No staged parquet parts found in {staged_dir}")
    logger.info("Loading %d parquet part(s) from %s", len(parts), staged_dir)
    frames: Iterable[pd.DataFrame] = (pd.read_parquet(part) for part in parts)
    return pd.concat(frames, ignore_index=True)


def _apply_chronic_flags(df: pd.DataFrame, chronic_groups: Dict[str, Sequence[str]]) -> pd.DataFrame:
    df = df.copy()
    for group_name, columns in chronic_groups.items():
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise KeyError(f"Missing expected chronic columns {missing} for group {group_name}")
        flag_col = f"{group_name.upper()}_FLAG"
        df[flag_col] = df[list(columns)].eq(1.0).any(axis=1)
        logger.debug("Computed %s from %s", flag_col, columns)
    return df


def build_modeling_dataset(config: PreprocessConfig) -> pd.DataFrame:
    """Create the modeling dataset and persist it to the configured output path."""

    staged_df = _load_staged_parts(config.staged_dir)
    staged_df = _apply_chronic_flags(staged_df, config.chronic_groups)

    flag_cols: List[str] = [col for col in staged_df.columns if col.endswith("_FLAG")]
    staged_df["HIGH_RISK"] = staged_df[flag_cols].any(axis=1)

    missing_features = [col for col in config.feature_cols if col not in staged_df.columns]
    if missing_features:
        raise KeyError(f"Missing expected feature columns: {missing_features}")

    dataset = staged_df[list(config.feature_cols) + ["HIGH_RISK"]].copy()
    dataset = dataset.dropna(subset=list(config.essential_cols))

    numeric_cols = [col for col in config.numerical_impute_cols if col in dataset.columns]
    if len(numeric_cols) != len(config.numerical_impute_cols):
        missing_numeric = set(config.numerical_impute_cols) - set(numeric_cols)
        logger.warning("Skipping missing numeric columns during imputation: %s", sorted(missing_numeric))
    for col in numeric_cols:
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")
    if numeric_cols:
        dataset[numeric_cols] = dataset[numeric_cols].fillna(dataset[numeric_cols].median())

    categorical_cols = [col for col in config.categorical_cols if col in dataset.columns]
    if len(categorical_cols) != len(config.categorical_cols):
        missing_categorical = set(config.categorical_cols) - set(categorical_cols)
        logger.warning("Skipping missing categorical columns during fill: %s", sorted(missing_categorical))
    if categorical_cols:
        dataset[categorical_cols] = dataset[categorical_cols].fillna("Unknown").astype(str)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(config.output_path, index=False)
    logger.info(
        "Persisted modeling dataset with %d rows and %d columns to %s",
        len(dataset),
        len(dataset.columns),
        config.output_path,
    )
    return dataset


def parse_args(argv: Sequence[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Build modeling dataset from staged BRFSS data.")
    parser.add_argument("--staged-dir", type=Path, required=True, help="Path to staged parquet directory.")
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Destination path for processed dataset parquet.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    config = PreprocessConfig(staged_dir=args.staged_dir, output_path=args.output_path)
    build_modeling_dataset(config)


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["PreprocessConfig", "build_modeling_dataset", "main"]

