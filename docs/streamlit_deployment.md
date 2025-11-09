# Streamlit Deployment Guide

## Local Machine
1. Ensure artifacts exist (`data/processed/brfss_lifestyle_risk.parquet` and `data/brfss/<year>/artifacts/models/...`).
2. Install dependencies and set `PYTHONPATH` (see README quickstart).
3. Run `streamlit run app/streamlit_app.py`.
4. Access dashboard at `http://localhost:8501`.

## Docker Container
```bash
docker compose up --build streamlit
```
- Environment variables (override as needed):
  - `MODEL_ARTIFACT_ROOT=/app/data/brfss/2021/artifacts/models`
  - `PROCESSED_DATA_PATH=/app/data/processed/brfss_lifestyle_risk.parquet`
- Mount `./data` to `/app/data` to share artifacts between host & container.

## Streamlit Cloud (Managed)
1. Push repository to GitHub.
2. In Streamlit Cloud, create new app pointing to `app/streamlit_app.py`.
3. Set environment variables in the deployment settings:
   - `MODEL_ARTIFACT_ROOT` (e.g., `data/brfss/2021/artifacts/models`)
   - `PROCESSED_DATA_PATH` (e.g., `data/processed/brfss_lifestyle_risk.parquet`)
4. Upload or sync artifacts via Git LFS / external store (S3, GCS). Update the app to read from cloud storage if hosting data externally.

## Tips
- Configure caching TTL in `streamlit_app.py` if data/model updates occur frequently.
- Protect the app behind authentication (Streamlit Cloud sharing settings or reverse proxy) when handling sensitive data.
- Consider adding health checks (`streamlit healthz`) for production monitoring.
