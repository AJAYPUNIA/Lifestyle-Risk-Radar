"""Utilities to parse BRFSS survey extracts into analytics-friendly tables."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional
import shutil

import pandas as pd

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def read_brfss_xpt(
    xpt_path: Path | str,
    columns: Optional[List[str]] = None,
    chunk_size: Optional[int] = None,
) -> Iterable[pd.DataFrame]:
    """Load BRFSS data from a SAS transport (XPT) file.

    Parameters
    ----------
    xpt_path: Path | str
        Location of the `.XPT` file.
    columns: Optional[List[str]]
        Subset of columns to read. If None, loads all fields.
    chunk_size: Optional[int]
        Number of rows per chunk. If None, yields a single dataframe with the
        entire dataset. Chunking requires `pyreadstat` to be installed.

    Yields
    ------
    pandas.DataFrame
        Dataframe(s) containing the requested BRFSS records.
    """

    xpt_path = Path(xpt_path)
    if not xpt_path.exists():
        raise FileNotFoundError(f"XPT file not found: {xpt_path}")

    if chunk_size is None:
        logger.info("Loading BRFSS XPT file %s", xpt_path)
        df = pd.read_sas(xpt_path, format="xport", encoding="utf-8", usecols=columns)
        logger.info("Loaded %d rows with %d columns", len(df), len(df.columns))
        yield df
        return

    try:
        import pyreadstat
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "Chunked loading requires `pyreadstat`. Install it or read the full file."
        ) from exc

    logger.info(
        "Loading BRFSS XPT file %s in chunks of %d rows (pyreadstat)", xpt_path, chunk_size
    )
    reader = pyreadstat.read_file_in_chunks(
        pyreadstat.read_xport,
        file_path=str(xpt_path),
        chunksize=chunk_size,
        usecols=columns,
    )
    for chunk_df, _ in reader:
        logger.debug("Loaded chunk with %d rows", len(chunk_df))
        yield chunk_df


def convert_xpt_to_parquet(
    xpt_path: Path | str,
    output_dir: Path | str,
    columns: Optional[List[str]] = None,
    chunk_size: Optional[int] = None,
    compression: str = "snappy",
) -> Path:
    """Convert a BRFSS XPT file into partitioned Parquet.

    Parameters
    ----------
    xpt_path: Path | str
        Input SAS transport file.
    output_dir: Path | str
        Directory where the parquet dataset should live.
    columns: Optional[List[str]]
        Subset of columns to persist.
    chunk_size: Optional[int]
        Number of rows per chunk. See :func:`read_brfss_xpt`.
    compression: str
        Parquet compression codec.

    Returns
    -------
    Path
        Path to the root of the parquet dataset.
    """

    xpt_path = Path(xpt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / f"{xpt_path.stem.lower()}_dataset"

    if dataset_path.exists():
        logger.warning("Overwriting existing dataset at %s", dataset_path)
        shutil.rmtree(dataset_path)

    dataset_path.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover - required dependency
        raise RuntimeError("pyarrow is required to write parquet files") from exc

    metadata = {
        b"source": str(xpt_path).encode(),
        b"columns": ",".join(columns).encode() if columns else b"all",
    }

    for idx, chunk in enumerate(read_brfss_xpt(xpt_path, columns=columns, chunk_size=chunk_size)):
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        table = table.replace_schema_metadata(metadata)
        output_file = dataset_path / f"part-{idx:05d}.parquet"
        pq.write_table(table, output_file, compression=compression)
        total_rows += table.num_rows
        logger.info("Written chunk %d with %d rows", idx + 1, table.num_rows)

    if total_rows == 0:
        logger.warning("No data written from %s", xpt_path)

    logger.info("Persisted %d rows to %s", total_rows, dataset_path)
    return dataset_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BRFSS XPT to Parquet")
    parser.add_argument("xpt_path", type=Path, help="Path to the BRFSS .XPT file")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Destination directory for parquet output",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Optional subset of columns to retain",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Rows per chunk (requires pyreadstat)",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        help="Parquet compression codec",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    _configure_logging(verbose=args.verbose)
    convert_xpt_to_parquet(
        xpt_path=args.xpt_path,
        output_dir=args.output_dir,
        columns=args.columns,
        chunk_size=args.chunk_size,
        compression=args.compression,
    )


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
