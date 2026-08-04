"""Tests for deterministic sample artifact filesystem utilities."""

import hashlib
import json
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from taxi_lakehouse.sample_artifacts import (
    copy_sample_file_atomically,
    write_json_artifact_atomically,
    write_single_parquet_file,
)


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Provide one local Spark session for artifact tests."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("sample-artifact-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


def test_write_single_parquet_file_preserves_data_and_schema(
    spark: SparkSession,
    tmp_path: Path,
) -> None:
    """A Spark directory output should become one Parquet file."""
    frame = spark.createDataFrame(
        [
            (2, "second"),
            (1, "first"),
        ],
        (
            "trip_id",
            "label",
        ),
    )

    destination_path = tmp_path / "sample" / "yellow_tripdata_2024-01.parquet"
    temporary_root = tmp_path / "temporary"

    result = write_single_parquet_file(
        frame,
        destination_path,
        temporary_root,
    )

    assert result.file_path == destination_path
    assert destination_path.is_file()
    assert result.content_length_bytes > 0
    assert len(result.sha256) == 64

    reloaded_frame = spark.read.parquet(destination_path.as_posix())

    assert reloaded_frame.schema == frame.schema
    assert sorted(tuple(row) for row in reloaded_frame.collect()) == [
        (1, "first"),
        (2, "second"),
    ]

    assert list(temporary_root.iterdir()) == []
    assert not destination_path.with_name(f".{destination_path.name}.tmp").exists()


def test_write_single_parquet_file_atomically_replaces_existing_file(
    spark: SparkSession,
    tmp_path: Path,
) -> None:
    """A later generation should replace the previous artifact."""
    destination_path = tmp_path / "sample" / "yellow_tripdata_2024-02.parquet"
    temporary_root = tmp_path / "temporary"

    first_result = write_single_parquet_file(
        spark.createDataFrame(
            [(1,)],
            ("value",),
        ),
        destination_path,
        temporary_root,
    )

    second_result = write_single_parquet_file(
        spark.createDataFrame(
            [
                (2,),
                (3,),
            ],
            ("value",),
        ),
        destination_path,
        temporary_root,
    )

    assert first_result.sha256 != second_result.sha256

    values = sorted(
        row["value"]
        for row in (spark.read.parquet(destination_path.as_posix()).collect())
    )

    assert values == [2, 3]
    assert list(temporary_root.iterdir()) == []


def test_write_single_parquet_file_rejects_invalid_suffix(
    spark: SparkSession,
    tmp_path: Path,
) -> None:
    """The configured artifact should use a Parquet suffix."""
    with pytest.raises(
        ValueError,
        match="must end with .parquet",
    ):
        write_single_parquet_file(
            spark.range(1),
            tmp_path / "sample-output",
            tmp_path / "temporary",
        )


def test_copy_sample_file_atomically_replaces_destination(
    tmp_path: Path,
) -> None:
    """A copied artifact should atomically replace old content."""
    source_path = tmp_path / "landing" / "taxi_zone_lookup.csv"
    destination_path = tmp_path / "sample" / "taxi_zone_lookup.csv"

    source_path.parent.mkdir(parents=True)
    destination_path.parent.mkdir(parents=True)

    source_content = b"LocationID,Borough\n1,EWR\n"
    source_path.write_bytes(source_content)
    destination_path.write_text(
        "obsolete",
        encoding="utf-8",
    )

    result = copy_sample_file_atomically(
        source_path,
        destination_path,
    )

    assert destination_path.read_bytes() == source_content
    assert result.file_path == destination_path
    assert result.content_length_bytes == len(source_content)
    assert result.sha256 == hashlib.sha256(source_content).hexdigest()

    assert not destination_path.with_name(f".{destination_path.name}.tmp").exists()


def test_copy_sample_file_atomically_rejects_missing_source(
    tmp_path: Path,
) -> None:
    """A missing source artifact should fail explicitly."""
    with pytest.raises(
        FileNotFoundError,
        match="Sample source file does not exist",
    ):
        copy_sample_file_atomically(
            tmp_path / "missing.csv",
            tmp_path / "sample" / "zones.csv",
        )


def test_write_json_artifact_atomically_is_deterministic(
    tmp_path: Path,
) -> None:
    """JSON artifacts should use stable formatting and key order."""
    destination_path = tmp_path / "sample" / "sample_manifest.json"

    destination_path.parent.mkdir(parents=True)
    destination_path.write_text(
        "obsolete",
        encoding="utf-8",
    )

    payload = {
        "zeta": 2,
        "alpha": {
            "second": 2,
            "first": 1,
        },
    }

    result = write_json_artifact_atomically(
        payload,
        destination_path,
    )

    expected_content = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    assert destination_path.read_text(encoding="utf-8") == expected_content

    assert result.file_path == destination_path
    assert result.content_length_bytes == len(expected_content.encode())
    assert result.sha256 == hashlib.sha256(expected_content.encode()).hexdigest()

    assert not destination_path.with_name(f".{destination_path.name}.tmp").exists()


def test_write_json_artifact_atomically_rejects_invalid_suffix(
    tmp_path: Path,
) -> None:
    """A JSON artifact destination should use a JSON suffix."""
    with pytest.raises(
        ValueError,
        match="must end with .json",
    ):
        write_json_artifact_atomically(
            {"schema_version": "1.0"},
            tmp_path / "sample_manifest.txt",
        )
