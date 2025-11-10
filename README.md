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






How to Use the Dashboard.


## 🩺 How to Use the Lifestyle Risk Radar Dashboard (Simple Guide)

This dashboard helps hospitals and health workers quickly understand which patients may be at higher risk for chronic conditions (like heart, kidney, or breathing diseases) — based on their lifestyle habits and basic health data.

You don’t need any coding skills to use it — everything runs through a simple **Streamlit web app**.

---

### 1️⃣ **Latest Model Metrics**

📊 **What you see:**  
At the top of the dashboard, there’s a table showing all the machine learning models that have been trained so far.  
For each model, it lists:
- **ROC AUC** → how well the model separates high-risk vs low-risk patients  
- **Accuracy** → how often the model’s prediction was correct  

💡 **How it updates:**  
- Whenever you retrain the models (either using the command line or Apache Airflow), the metrics refresh automatically.  
- Each new model version is saved with a timestamp like `data/brfss/2021/artifacts/models/20251109T221325Z/`.  
- The dashboard automatically shows the newest one.  
- You can even add new metrics (like precision or recall) by editing `pipelines/train.py` — the app will show them too.

---

### 2️⃣ **Patient Risk Prediction Form**

This is the interactive part where you can **simulate a patient’s lifestyle** and see their predicted health risk.

Every field on the form represents one lifestyle or health feature used by the model.

#### 🧩 **Examples of Fields:**

**Lifestyle habits:**
- Smoking (`_SMOKER3`, `SMOKE100`, `SMOKDAY2`): whether the person smokes every day, sometimes, or never  
- Drinking (`DRNKANY5`, `AVEDRNK3`, `MAXDRNKS`): alcohol use in the last 30 days and drinking frequency  
- Exercise (`EXERANY2`): whether they did any exercise outside of work  

**Personal info:**
- Age group (`_AGEG5YR`)  
- Sex (`SEXVAR`)  
- Income (`INCOME3`)  
- BMI (`_BMI5` and `_BMI5CAT`)  

**Health indicators:**
- `PHYSHLTH`: days in the past month with poor physical health  
- `MENTHLTH`: days with poor mental health  
- `WEIGHT2` / `HEIGHT3`: self-reported weight and height  

💡 **Tip:**  
Some options use numbers (because that’s how BRFSS encodes them).  
You can make the app more friendly by showing plain-text options like “Every day smoker” or “Never smoked” and converting them back to the numeric codes behind the scenes.

---

### 3️⃣ **Model Selector**

There’s a dropdown to pick which model you want to use (for example, `logistic_regression` or `xgboost`).  
Each model was trained separately and might give slightly different risk scores.  
You can switch between them to compare results.

---

### 4️⃣ **Predict Risk Button**

Once you’ve filled in the lifestyle and health details, click **“Predict Risk.”**

The app:
1. Collects your answers into a single data row  
2. Runs the same data cleaning and scaling steps used during training  
3. Uses the chosen model to predict the **probability of high health risk**

📈 **Output example:**  
- `0%` → very low risk  
- `50%` → moderate risk  
- `100%` → very high risk  

This lets you test **“what-if” scenarios** — for example:
- If a person stops smoking (`_SMOKER3` = “Never”), how does the risk change?  
- What if BMI decreases from 31 (obese) to 25 (normal)?  
- What if exercise frequency increases?

---

### 5️⃣ **Artifact Runs**

At the bottom, you’ll see the list of recent model runs (folders with timestamps like `20251110T003431Z`).  
This shows that your Airflow pipeline is working and creating updated models automatically.

💡 It’s for information only — each timestamp means a new round of data processing and training was completed.

---

### 🧭 **In Simple Terms**

- The **top table** tells you how well the models are performing.  
- The **middle form** lets you enter a patient’s info to see their predicted risk.  
- The **dropdown** lets you choose which model to use.  
- The **bottom list** confirms your automation pipeline is running regularly.  

Together, these give hospitals a live, easy-to-understand system to **monitor patient risk** and test how lifestyle improvements can lower that risk.

---

### 🪄 **Future Improvements (for Simplicity)**

To make it even easier to use:
- Replace numeric codes with readable options like “Yes/No” or “Smoker/Non-smoker.”  
- Show a friendly output like:  
  > “This patient has a 62% chance of developing a chronic condition. Consider regular checkups.”  
- Add visuals like feature importance or SHAP charts to explain *why* the model gave that score.





