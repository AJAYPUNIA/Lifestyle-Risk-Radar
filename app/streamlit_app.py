"""Streamlit app for lifestyle-based disease risk monitoring."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from airflow_patient_risk_prediction.utils.model_registry import (
    ModelArtifacts,
    load_artifacts,
    load_model,
)

DEFAULT_ARTIFACT_ROOT = Path(os.environ.get("MODEL_ARTIFACT_ROOT", "data/brfss/artifacts/models"))
DEFAULT_DATASET_PATH = Path(os.environ.get("PROCESSED_DATA_PATH", "data/processed/brfss_lifestyle_risk.parquet"))

LIFESTYLE_OPTIONS: Dict[str, Dict[str, str]] = {
    "GENHLTH": {
        "1": "Excellent",
        "2": "Very good",
        "3": "Good",
        "4": "Fair",
        "5": "Poor",
    },
    "_SMOKER3": {
        "1": "Every day smoker",
        "2": "Some days smoker",
        "3": "Former smoker",
        "4": "Never smoked",
    },
    "DRNKANY5": {
        "1": "Yes",
        "2": "No",
    },
    "EXERANY2": {
        "1": "Yes",
        "2": "No",
    },
    "_AGEG5YR": {
        "1": "18-24",
        "2": "25-29",
        "3": "30-34",
        "4": "35-39",
        "5": "40-44",
        "6": "45-49",
        "7": "50-54",
        "8": "55-59",
        "9": "60-64",
        "10": "65-69",
        "11": "70-74",
        "12": "75-79",
        "13": "80+",
    },
    "_BMI5CAT": {
        "1": "Underweight",
        "2": "Normal",
        "3": "Overweight",
        "4": "Obese",
    },
    "SEXVAR": {
        "1": "Male",
        "2": "Female",
    },
    "INCOME3": {
        "1": "Less than $10k",
        "2": "$10k-$15k",
        "3": "$15k-$20k",
        "4": "$20k-$25k",
        "5": "$25k-$35k",
        "6": "$35k-$50k",
        "7": "$50k-$75k",
        "8": "≥$75k",
    },
}

NUMERIC_FEATURES = [
    "_BMI5",
    "AVEDRNK3",
    "DRNK3GE5",
    "MAXDRNKS",
    "_DRNKWK1",
    "WEIGHT2",
    "HEIGHT3",
    "PHYSHLTH",
    "MENTHLTH",
]


@st.cache_data(show_spinner=False)
def load_dataset_sample(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    df = df.drop(columns=["HIGH_RISK"], errors="ignore")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.columns.difference(numeric_cols)

    numeric_defaults = df[numeric_cols].median().astype(float)

    if len(categorical_cols) > 0:
        mode_df = df[categorical_cols].mode(dropna=True)
        if not mode_df.empty:
            categorical_defaults = mode_df.iloc[0].astype(str)
        else:
            categorical_defaults = pd.Series("Unknown", index=categorical_cols, dtype=str)
    else:
        categorical_defaults = pd.Series(dtype=str)

    defaults = pd.concat([numeric_defaults, categorical_defaults])
    defaults = defaults.reindex(df.columns)
    defaults = defaults.fillna("Unknown")
    return defaults


@st.cache_resource(show_spinner=False)
def load_latest_artifacts(artifact_root: Path) -> Optional[ModelArtifacts]:
    return load_artifacts(artifact_root)


def render_metrics_section(artifacts: Optional[ModelArtifacts]) -> None:
    st.header("Latest Model Metrics")
    if artifacts is None or not artifacts.metrics:
        st.info("No trained models available yet. Run the Airflow pipeline to generate artifacts.")
        return
    metrics_df = (
        pd.DataFrame(artifacts.metrics)
        .T.reset_index()
        .rename(columns={"index": "model"})
        .sort_values(by="roc_auc", ascending=False)
    )
    st.dataframe(metrics_df, use_container_width=True)


def collect_patient_inputs(reference_row: pd.Series) -> pd.Series:
    st.header("Patient Risk Prediction")
    inputs = reference_row.copy()
    cols = st.columns(2)
    with cols[0]:
        for feature, options in LIFESTYLE_OPTIONS.items():
            if feature in inputs.index:
                inverse = {label: code for code, label in options.items()}
                default_label = options.get(str(inputs.get(feature, "2")), next(iter(options.values())))
                selection = st.selectbox(feature, options.values(), index=list(options.values()).index(default_label))
                inputs[feature] = inverse[selection]
        for feature in ["SMOKE100", "SMOKDAY2", "_RFSMOK3", "ECIGNOW1"]:
            if feature in inputs.index:
                inputs[feature] = st.selectbox(
                    feature,
                    options=["1", "2"],
                    index=0 if str(inputs.get(feature, "2")) == "1" else 1,
                )
    with cols[1]:
        for feature in NUMERIC_FEATURES:
            if feature in inputs.index:
                value = float(inputs.get(feature, 0.0))
                inputs[feature] = st.number_input(feature, value=value, step=1.0)
    for feature in ["GENHLTH", "POORHLTH", "EXERANY2", "DRNKANY5"]:
        if feature in inputs.index:
            inputs[feature] = str(inputs[feature])
    return inputs


def predict_risk(model, sample: pd.Series) -> float:
    df = pd.DataFrame([sample])
    probability = model.predict_proba(df)[0][1]
    return float(probability)


def render_prediction_section(artifacts: Optional[ModelArtifacts], dataset_path: Path) -> None:
    if artifacts is None or not artifacts.models:
        return
    sample_row = load_dataset_sample(dataset_path)
    user_inputs = collect_patient_inputs(sample_row)

    model_choices = list(artifacts.models.keys())
    model_name = st.selectbox("Select model", model_choices, index=0)
    model = load_model(artifacts.models[model_name])

    if st.button("Predict Risk"):
        probability = predict_risk(model, user_inputs)
        st.metric("Predicted High-Risk Probability", f"{probability:.2%}")


def render_recent_runs(artifacts_root: Path) -> None:
    st.header("Artifact Runs")
    if not artifacts_root.exists():
        st.write("No artifacts directory found.")
        return
    run_dirs = sorted((p for p in artifacts_root.glob("*") if p.is_dir()), reverse=True)
    if not run_dirs:
        st.write("No model runs yet.")
        return
    for run in run_dirs[:5]:
        st.write(f"- {run.name}")


def main() -> None:
    st.set_page_config(page_title="Lifestyle Risk Monitoring", layout="wide")
    st.title("Lifestyle-Based Disease Risk Dashboard")

    artifacts_root = DEFAULT_ARTIFACT_ROOT
    dataset_path = DEFAULT_DATASET_PATH

    artifacts = load_latest_artifacts(artifacts_root)
    render_metrics_section(artifacts)
    render_prediction_section(artifacts, dataset_path)
    render_recent_runs(artifacts_root)


if __name__ == "__main__":
    main()

