# Testing Checklist & Stakeholder Summary

## Pre-Deployment Checks
- [ ] Airflow `brfss_ingestion` DAG completes successfully (download, preprocess, train, metadata tasks).
- [ ] `data/processed/brfss_lifestyle_risk.parquet` regenerated without errors.
- [ ] Model artifacts present in `data/brfss/<year>/artifacts/models/<timestamp>/` with `metrics.json`.
- [ ] Streamlit app loads latest metrics and serves predictions end-to-end.
- [ ] Docker image builds and launches (`docker compose up --build streamlit`).

## Acceptance Criteria
- ROC AUC ≥ 0.70 and accuracy ≥ baseline thresholds stated in `metrics.json`.
- Runtime for Airflow DAG < 30 minutes on scheduled cadence.
- Streamlit response time < 2 seconds for single prediction.

## Next Milestones
1. Integrate Airflow alerts (Slack/email) for failed tasks or metric regressions.
2. Configure secure hosting (HTTPS, authentication) for Streamlit.
3. Extend dashboard with cohort trend analytics and download/export options.
4. Explore model monitoring (data drift, performance drift) with scheduled reports.

## Contacts
- Data Engineering Lead: _TBD_
- ML Engineer: _TBD_
- Analytics Stakeholders: _TBD_
