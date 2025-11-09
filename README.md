# Lifestyle Risk Radar

Hospitals often track lifestyle habits such as smoking, drinking, exercise, and BMI, but it’s hard to turn that information into timely alerts about who might be at risk. Our system automates this process:
It takes lifestyle inputs (plus basic health info) and predicts the chance a patient already has—or may soon develop—chronic heart, metabolic, breathing, or kidney issues.
Airflow and our reusable pipelines keep the data fresh: they download new BRFSS records, clean them, retrain the models, save metrics, and refresh the dashboard on a schedule—no manual crunching needed.
The Streamlit app shows current model performance and gives clinicians a simple form to test “what-if” scenarios for any patient, so they can instantly see how lifestyle changes affect risk.
By doing this, teams get continuous risk monitoring without a full-time data scientist, clear insight into how lifestyle drives risk, and actionable scores to reach out sooner to patients whose risk jumps—even before they show symptoms.
Automated data ingestion, model retraining, and dashboarding pipeline for lifestyle-driven disease risk using Apache Airflow and Streamlit.

## Project Structure
# 🩺 Lifestyle Risk Radar

Hospitals often track lifestyle habits such as smoking, drinking, exercise, and BMI, but it’s hard to turn that information into timely alerts about who might be at risk.  
**Lifestyle Risk Radar** automates this process: it takes lifestyle inputs (plus basic health info) and predicts the chance a patient already has—or may soon develop—chronic **heart, metabolic, breathing, or kidney** issues.

Apache Airflow and reusable pipelines keep the data fresh — they download new BRFSS records, clean them, retrain models, save metrics, and refresh the dashboard on a schedule — **no manual crunching needed**.  
The Streamlit app shows current model performance and provides a clinician-friendly form to test “what-if” lifestyle scenarios for any patient, instantly visualizing how changes affect risk.  

By doing this, teams gain continuous risk monitoring **without needing a full-time data scientist**, clear insight into how lifestyle drives health risk, and actionable alerts to reach out sooner to patients whose risk increases — even before symptoms appear.

---

## 🧠 Summary

Automated data ingestion, preprocessing, model retraining, and dashboarding pipeline for lifestyle-driven disease risk using **Apache Airflow** and **Streamlit**.

---

## 🏗️ Project Structure

```
app/                         # Streamlit app
src/airflow_patient_risk_prediction/
  data/                      # BRFSS staging utilities
  pipelines/                 # Preprocessing & training pipelines (CLI-ready)
  utils/                     # Model registry helpers
notebooks/                   # EDA & modeling experiments
 d ags/                      # Airflow DAG definitions
 data/                       # Raw, staged, processed, and artifact storage
 docs/                       # Runbook & design notes
```

## Quickstart

### 🧩 1. Local (Python)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)/src

# Build processed dataset and train models
python -m airflow_patient_risk_prediction.pipelines.preprocess \
  --staged-dir data/staged/brfss/2021/llcp2021_dataset \
  --output-path data/processed/brfss_lifestyle_risk.parquet

python -m airflow_patient_risk_prediction.pipelines.train \
  --dataset-path data/processed/brfss_lifestyle_risk.parquet \
  --artifacts-dir data/brfss/2021/artifacts/models

# Launch dashboard
streamlit run app/streamlit_app.pyamlit run app/streamlit_app.py
```

### 2. Docker Compose
```bash
docker compose up --build streamlit
```
- Streamlit: http://localhost:8501
- Data/metrics persist to local `data/` folder.
- To start Airflow UI using the lightweight standalone executor:
  ```bash
  docker compose --profile airflow up airflow
  ```
  Airflow UI: http://localhost:8080 (username/password: `airflow`/`airflow`).

## Airflow DAG
- `brfss_ingestion` downloads BRFSS assets, stages parquet, runs preprocessing & model retraining, and logs metadata.
- Trigger via Airflow UI or CLI (`airflow dags trigger brfss_ingestion`).

## Streamlit App Features
- Latest model metrics summary.
- Patient-level risk prediction with editable lifestyle inputs.
- Recent model run history for transparency.

## Testing Checklist
- [ ] Airflow DAG completes end-to-end and artifacts populate `data/brfss/<year>/`.
- [ ] CLI pipelines run without errors.
- [ ] Streamlit app displays metrics and predictions using latest artifacts.
- [ ] Docker images build successfully and services start.

For detailed operational guidance, see `docs/runbook.md`.
Languages & Libraries: Python, Pandas, Scikit-learn, XGBoost
Workflow Orchestration: Apache Airflow
Dashboarding: Streamlit
Containerization: Docker / Docker Compose
Data Source: BRFSS (Behavioral Risk Factor Surveillance System)

