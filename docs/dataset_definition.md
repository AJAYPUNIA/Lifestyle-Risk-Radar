# Lifestyle Risk Modeling Dataset

## Source
- Derived from `data/staged/brfss/2021/llcp2021_dataset` (BRFSS 2021 curated subset).
- Generated via `notebooks/brfss_eda.ipynb` preprocessing notebook.

## Label Definition
- `HIGH_RISK` = 1 if any of the following condition groups are reported (code value `1`):
  - Cardiovascular: `CVDINFR4`, `CVDCRHD4`, `CVDSTRK3`
  - Metabolic: `DIABETE4`
  - Respiratory: `ASTHMA3`
  - Renal: `CHCKDNY2`
- Group-level flags (`CARDIO_FLAG`, `METABOLIC_FLAG`, etc.) retained for auditing.

## Feature Set
- Demographics: `_STATE`, `FMONTH`, `SEXVAR`, `_AGEG5YR`, `INCOME3`
- General health & mental health: `GENHLTH`, `PHYSHLTH`, `MENTHLTH`, `POORHLTH`
- Lifestyle behaviors: `EXERANY2`, `SMOKE100`, `SMOKDAY2`, `_SMOKER3`, `_RFSMOK3`, `ECIGNOW1`, `DRNKANY5`, `AVEDRNK3`, `DRNK3GE5`, `MAXDRNKS`, `_DRNKWK1`
- Anthropometrics: `_BMI5`, `_BMI5CAT`, `WEIGHT2`, `HEIGHT3`

## Cleaning Rules
- Drop records missing any of `GENHLTH`, `_SMOKER3`, `DRNKANY5`, `EXERANY2`, `_AGEG5YR`.
- Median-impute numeric quantities: `_BMI5`, `AVEDRNK3`, `DRNK3GE5`, `MAXDRNKS`, `WEIGHT2`, `HEIGHT3`.
- Fill categorical holes (`POORHLTH`, `SMOKE100`, `SMOKDAY2`, `ECIGNOW1`, `INCOME3`) with `"Unknown"` placeholder.
- Resulting dataset size (after 2021 run): recorded in notebook output.

## Output Artifact
- Stored at `data/processed/brfss_lifestyle_risk.parquet` (updated per preprocessing run).
- Downstream tasks (feature engineering, modeling DAG) should treat this path as the canonical training input.
- Consider versioning via timestamped subdirectories when integrating with Airflow.

