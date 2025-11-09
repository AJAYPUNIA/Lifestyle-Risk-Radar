# Lifestyle Risk Pipeline Runbook

## Overview
This project automates ingestion, preprocessing, model training, and dashboard updates for lifestyle-based disease risk prediction using BRFSS survey data. Key components:

- **Airflow DAG** (`dags/brfss_ingestion_dag.py`): orchestrates data download, staging, preprocessing, and model training.
- **Pipelines** (`src/airflow_patient_risk_prediction/pipelines/`): reusable preprocessing and training modules with CLI entrypoints.
- **Streamlit App** (`app/streamlit_app.py`): displays latest model metrics and supports individual risk predictions.
- **Artifacts**: stored under `data/brfss/<year>/` structure with processed datasets and model versions.

## Prerequisites
- Python environment with required dependencies (see `requirements.txt` or export from notebooks).
- Airflow configured to read DAGs from the project’s `dags/` directory.
- Network access to download BRFSS data (CDC).

## Step-by-Step Execution

### 1. Bootstrap Data
- Ensure Airflow scheduler is running and the `brfss_ingestion` DAG is active.
- Trigger the DAG manually or wait for scheduled run (default monthly). Tasks executed:
  1. `prepare_directories`
  2. `fetch_assets` (downloads BRFSS XPT + codebook)
  3. `extract_xpt`
  4. `convert_to_parquet_task` (stages BRFSS data)
  5. `preprocess_dataset` (calls preprocessing pipeline)
  6. `train_models_task` (training pipeline)
  7. `record_metadata`

Artifacts produced in `data/brfss/<year>/`:
- `raw/`: source zip, codebook
- `staged/`: parquet parts
- `processed/brfss_lifestyle_risk.parquet`
- `artifacts/models/<timestamp>/`: joblib models and metrics

### 2. Manual CLI Execution (optional)
Run pipelines without Airflow:
```bash
# Preprocess staged data
python -m airflow_patient_risk_prediction.pipelines.preprocess \
  --staged-dir data/staged/brfss/2021/llcp2021_dataset \
  --output-path data/processed/brfss_lifestyle_risk.parquet

# Train models
python -m airflow_patient_risk_prediction.pipelines.train \
  --dataset-path data/processed/brfss_lifestyle_risk.parquet \
  --artifacts-dir data/brfss/2021/artifacts/models
```

### 3. Launch Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```
Environment variables (optional):
- `MODEL_ARTIFACT_ROOT`: override artifacts directory
- `PROCESSED_DATA_PATH`: override processed dataset path

The app displays:
- Latest model metrics (ROC AUC, accuracy)
- Patient-level risk prediction form
- Recent artifact run directories

### 4. Monitoring & Maintenance
- Check Airflow UI for task logs and retries.
- Review `data/brfss/<year>/artifacts/models/<timestamp>/metrics.json` for evaluation metrics.
- Refresh Streamlit app after new model runs (app caches data; rerun or restart to reload).

## Troubleshooting
- Missing processed dataset? Run preprocessing pipeline (Airflow or CLI).
- No models on dashboard? Ensure `artifacts/models` contains timestamped runs with `.joblib` and `metrics.json`.
- Dataset download failing? Verify network access or manually mirror the BRFSS files.

## Next Enhancements
- Add automated data quality checks before training.
- Integrate alerting on metric regressions.
- Expand Streamlit with cohort analytics and trend charts.

