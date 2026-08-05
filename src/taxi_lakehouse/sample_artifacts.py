"""Filesystem utilities for deterministic sample artifacts."""

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from pyspark.sql import DataFrame

from taxi_lakehouse.data_acquisition import calculate_sha256


@dataclass(frozen=True, slots=True)
class GeneratedSampleFile:
    """Metadata describing one generated sample file."""

    file_path: Path
    content_length_bytes: int
    sha256: str


def write_single_parquet_file(
    frame: DataFrame,
    destination_path: Path,
    temporary_root: Path,
) -> GeneratedSampleFile:
    """Write a DataFrame as one atomically replaced Parquet file."""
    if destination_path.suffix != ".parquet":
        raise ValueError("Parquet destination path must end with .parquet.")

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output_directory = temporary_root / (
        f"{destination_path.stem}-{uuid4().hex}"
    )
    replacement_path = destination_path.with_name(f".{destination_path.name}.tmp")

    replacement_path.unlink(missing_ok=True)

    try:
        (
            frame.coalesce(1)
            .write.mode("errorifexists")
            .parquet(temporary_output_directory.as_posix())
        )

        parquet_parts = tuple(sorted(temporary_output_directory.glob("part-*.parquet")))

        if len(parquet_parts) != 1:
            raise RuntimeError(
                "Spark output must contain exactly one "
                f"Parquet part file, found {len(parquet_parts)}."
            )

        shutil.copyfile(
            parquet_parts[0],
            replacement_path,
        )
        replacement_path.replace(destination_path)
    except Exception:
        replacement_path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(
            temporary_output_directory,
            ignore_errors=True,
        )

    return GeneratedSampleFile(
        file_path=destination_path,
        content_length_bytes=(destination_path.stat().st_size),
        sha256=calculate_sha256(destination_path),
    )


def copy_sample_file_atomically(
    source_path: Path,
    destination_path: Path,
) -> GeneratedSampleFile:
    """Copy one sample artifact through an atomic replacement file."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Sample source file does not exist: {source_path}.")

    if source_path.resolve() == destination_path.resolve():
        raise ValueError("Sample source and destination paths must differ.")

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    replacement_path = destination_path.with_name(f".{destination_path.name}.tmp")
    replacement_path.unlink(missing_ok=True)

    try:
        shutil.copyfile(
            source_path,
            replacement_path,
        )
        replacement_path.replace(destination_path)
    except Exception:
        replacement_path.unlink(missing_ok=True)
        raise

    return GeneratedSampleFile(
        file_path=destination_path,
        content_length_bytes=destination_path.stat().st_size,
        sha256=calculate_sha256(destination_path),
    )


def write_json_artifact_atomically(
    payload: Mapping[str, Any],
    destination_path: Path,
) -> GeneratedSampleFile:
    """Write deterministic JSON through an atomic replacement file."""
    if destination_path.suffix != ".json":
        raise ValueError("JSON destination path must end with .json.")

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    replacement_path = destination_path.with_name(f".{destination_path.name}.tmp")
    replacement_path.unlink(missing_ok=True)

    serialized_payload = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    try:
        replacement_path.write_text(
            serialized_payload,
            encoding="utf-8",
        )
        replacement_path.replace(destination_path)
    except Exception:
        replacement_path.unlink(missing_ok=True)
        raise

    return GeneratedSampleFile(
        file_path=destination_path,
        content_length_bytes=destination_path.stat().st_size,
        sha256=calculate_sha256(destination_path),
    )
