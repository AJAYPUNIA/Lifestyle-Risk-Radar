"""Airflow DAG to ingest BRFSS survey data and publish staged parquet tables."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import pendulum
import requests
from airflow import DAG
from airflow.decorators import task
from airflow.models.param import Param

from airflow_patient_risk_prediction.data.brfss_parser import convert_xpt_to_parquet
from airflow_patient_risk_prediction.pipelines.preprocess import (
    PreprocessConfig,
    build_modeling_dataset,
)
from airflow_patient_risk_prediction.pipelines.train import TrainConfig, train_models


DEFAULT_YEAR = 2021
BRFSS_FILE_TEMPLATE = "https://www.cdc.gov/brfss/annual_data/{year}/files/LLCP{year}XPT.zip"
CODEBOOK_TEMPLATE = "https://www.cdc.gov/brfss/annual_data/{year}/pdf/codebook{yy}_llcp.pdf"


def _build_data_root() -> Path:
    root = os.environ.get("BRFSS_DATA_ROOT", "/opt/airflow/data/brfss")
    return Path(root)


def _download_file(url: str, destination: Path) -> Dict[str, str]:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    checksum = hashlib.sha256(response.content).hexdigest()
    return {"path": str(destination), "checksum": checksum, "source_url": url}


with DAG(
    dag_id="brfss_ingestion",
    schedule="0 6 1 * *",  # monthly run
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    default_args={"owner": "ml-platform", "retries": 1},
    params={
        "year": Param(DEFAULT_YEAR, type="integer", minimum=2015, maximum=datetime.now().year),
    },
    tags=["brfss", "ingestion"],
) as dag:

    @task
    def prepare_directories(year: int) -> Dict[str, str]:
        root = _build_data_root() / str(year)
        raw_dir = root / "raw"
        staged_dir = root / "staged"
        docs_dir = raw_dir / "docs"
        raw_dir.mkdir(parents=True, exist_ok=True)
        staged_dir.mkdir(parents=True, exist_ok=True)
        docs_dir.mkdir(parents=True, exist_ok=True)
        return {
            "root": str(root),
            "raw_dir": str(raw_dir),
            "staged_dir": str(staged_dir),
            "docs_dir": str(docs_dir),
        }

    @task
    def fetch_assets(year: int, dirs: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        yy = str(year)[-2:]
        xpt_url = BRFSS_FILE_TEMPLATE.format(year=year)
        codebook_url = CODEBOOK_TEMPLATE.format(year=year, yy=yy)
        xpt_meta = _download_file(
            xpt_url,
            Path(dirs["raw_dir"]) / f"LLCP{year}XPT.zip",
        )
        codebook_meta = _download_file(
            codebook_url,
            Path(dirs["docs_dir"]) / f"codebook_{year}.pdf",
        )
        return {"xpt": xpt_meta, "codebook": codebook_meta}

    @task
    def extract_xpt(meta: Dict[str, Dict[str, str]], dirs: Dict[str, str]) -> str:
        zip_path = Path(meta["xpt"]["path"])
        extract_dir = Path(dirs["raw_dir"])
        extract_dir.mkdir(parents=True, exist_ok=True)
        import zipfile

        with zipfile.ZipFile(zip_path, "r") as archive:
            for name in archive.namelist():
                if name.upper().endswith(".XPT"):
                    target = extract_dir / Path(name).name.strip()
                    archive.extract(name, path=extract_dir)
                    if name != target.name:
                        (extract_dir / name).rename(target)
                    return str(target)
        raise FileNotFoundError("No XPT file found inside archive")

    @task
    def convert_to_parquet_task(xpt_path: str, dirs: Dict[str, str]) -> str:
        columns = [
            "SEQNO",
            "_STATE",
            "FMONTH",
            "GENHLTH",
            "PHYSHLTH",
            "MENTHLTH",
            "POORHLTH",
            "CVDINFR4",
            "CVDCRHD4",
            "CVDSTRK3",
            "ASTHMA3",
            "CHCKDNY2",
            "DIABETE4",
            "SMOKE100",
            "SMOKDAY2",
            "_SMOKER3",
            "_RFSMOK3",
            "ECIGNOW1",
            "AVEDRNK3",
            "DRNK3GE5",
            "MAXDRNKS",
            "DRNKANY5",
            "_DRNKWK1",
            "EXERANY2",
            "WEIGHT2",
            "HEIGHT3",
            "_BMI5",
            "_BMI5CAT",
            "_RFBMI5",
            "INCOME3",
            "_AGEG5YR",
            "SEXVAR",
        ]
        dataset_path = convert_xpt_to_parquet(
            xpt_path=xpt_path,
            output_dir=Path(dirs["staged_dir"]),
            columns=columns,
            chunk_size=100000,
        )
        return str(dataset_path)

    @task
    def record_metadata(
        year: int,
        dirs: Dict[str, str],
        assets: Dict[str, Dict[str, str]],
        parquet_path: str,
    ) -> str:
        metadata_path = Path(dirs["root"]) / "ingestion_metadata.jsonl"
        record = {
            "year": year,
            "parquet_path": parquet_path,
            "assets": assets,
            "ingested_at": datetime.utcnow().isoformat(),
        }
        with metadata_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record) + "\n")
        return str(metadata_path)

    @task
    def preprocess_dataset(dirs: Dict[str, str]) -> str:
        staged_dir = Path(dirs["staged_dir"]) / "llcp2021_dataset"
        output_path = Path(dirs["root"]) / "processed" / "brfss_lifestyle_risk.parquet"
        config = PreprocessConfig(staged_dir=staged_dir, output_path=output_path)
        dataset = build_modeling_dataset(config)
        return str(output_path)

    @task
    def train_models_task(processed_path: str, dirs: Dict[str, str]) -> str:
        artifacts_dir = Path(dirs["root"]) / "artifacts" / "models"
        config = TrainConfig(
            dataset_path=Path(processed_path),
            artifacts_dir=artifacts_dir,
        )
        metrics = train_models(config)
        metrics_path = artifacts_dir / "latest_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        return str(metrics_path)

    directories = prepare_directories(dag.params["year"])
    assets = fetch_assets(dag.params["year"], directories)
    xpt_file = extract_xpt(assets, directories)
    parquet_dir = convert_to_parquet_task(xpt_file, directories)
    preprocess_path = preprocess_dataset(directories)
    training_metrics = train_models_task(preprocess_path, directories)
    record_metadata(dag.params["year"], directories, assets, parquet_dir)
