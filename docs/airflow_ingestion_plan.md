# BRFSS Data Ingestion Plan

## Objectives
- Automate retrieval of annual BRFSS survey assets (ASCII dataset, codebook, layout files).
- Normalize raw files into structured parquet tables suitable for model training.
- Maintain reproducible metadata (source URLs, download timestamps, Airflow run IDs).

## Data Sources
- `https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}ASC.ZIP`
- `https://www.cdc.gov/brfss/annual_data/{year}/pdf/codebook{yy}_llcp.pdf`
- Optional: layout file for fixed-width parsing (`LLCP{year}Layout.csv` when available).

## Storage Layout
- Raw: `/data/raw/brfss/{year}/` holding original ZIP, ASCII extract, documentation.
- Staged: `/data/staged/brfss/{year}/` containing columnar parquet partitions.
- Metadata: `/data/metadata/brfss_ingestions.parquet` capturing fetch history.

## Airflow DAG Outline
1. **`check_source_availability`**
   - Perform HEAD requests on dataset and codebook URLs.
   - Soft-fail with alert if files missing.
2. **`download_assets`**
   - Stream ZIP/PDF into raw directory with date-stamped filenames.
   - Record SHA256 checksums.
3. **`extract_ascii`**
   - Unzip ASCII file; optionally download layout CSV.
4. **`parse_fixed_width`**
   - Use documented column positions to convert ASCII to dataframe.
   - Persist to staged parquet (partition by state/year if feasible).
5. **`update_metadata`**
   - Append run context (execution date, Airflow run ID, row counts, checksum, source URLs).
6. **`cleanup_old_runs`** (optional)
   - Rotate outdated raw files based on retention policy.

## Operational Considerations
- Include Socrata-based sampling task for lightweight EDA (`data/raw/brfss/samples/`).
- Store Airflow connections for CDC downloads (no auth, but configure proxy/SSL options).
- Retry downloads with exponential backoff; enable checksum verification.
- Use `XCom` to share local paths between tasks.
- Document environment vars (`BRFSS_DATA_DIR`, `REQUESTS_CA_BUNDLE` for custom certs).

## Next Steps
- Build Python utilities for fixed-width parsing (leverage codebook layout).
- Prototype ingestion DAG in `dags/brfss_ingestion_dag.py` once utilities ready.
- Define data quality checks (row counts vs. published totals, null thresholds).
