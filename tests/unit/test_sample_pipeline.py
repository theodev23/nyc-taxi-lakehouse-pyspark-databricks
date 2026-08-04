"""Tests for deterministic sample pipeline orchestration."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from taxi_lakehouse.data_acquisition import (
    SourceFile,
    load_source_files,
)
from taxi_lakehouse.sample_pipeline import (
    load_taxi_zone_ids,
    resolve_sample_sources,
)
from taxi_lakehouse.sample_specification import (
    load_sample_specification,
)

PROJECT_SOURCE_MANIFEST_PATH = Path("data/source_manifest.json")
PROJECT_SAMPLE_SPECIFICATION_PATH = Path("data/sample/sample_spec.json")


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Provide one local Spark session for pipeline tests."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("sample-pipeline-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


def create_local_source_files(
    landing_directory: Path,
) -> tuple[SourceFile, ...]:
    """Create valid local substitutes for project source records."""
    landing_directory.mkdir(parents=True)

    source_files = load_source_files(PROJECT_SOURCE_MANIFEST_PATH)
    local_source_files = []

    for source_file in source_files:
        content = (f"local-test:{source_file.filename}\n").encode()

        file_path = landing_directory / source_file.filename
        file_path.write_bytes(content)

        local_source_files.append(
            replace(
                source_file,
                content_length_bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                downloaded_at_utc=("2026-08-04T09:00:00Z"),
            )
        )

    return tuple(local_source_files)


def test_resolve_sample_sources_preserves_month_order(
    tmp_path: Path,
) -> None:
    """Resolved monthly sources should follow specification order."""
    specification = load_sample_specification(PROJECT_SAMPLE_SPECIFICATION_PATH)
    landing_directory = tmp_path / "landing"

    source_files = create_local_source_files(landing_directory)

    resolved = resolve_sample_sources(
        specification,
        source_files,
        landing_directory,
    )

    assert (
        tuple(source.source_month for source in resolved.monthly_trip_sources)
        == specification.source_months
    )

    assert len(resolved.monthly_trip_sources) == 6
    assert resolved.taxi_zone_source.kind == ("taxi_zone_lookup")
    assert resolved.taxi_zone_path == (landing_directory / "taxi_zone_lookup.csv")


def test_resolve_sample_sources_rejects_missing_month(
    tmp_path: Path,
) -> None:
    """Every month declared by the specification is required."""
    specification = load_sample_specification(PROJECT_SAMPLE_SPECIFICATION_PATH)
    landing_directory = tmp_path / "landing"

    source_files = create_local_source_files(landing_directory)

    incomplete_source_files = tuple(
        source_file
        for source_file in source_files
        if source_file.source_month != "2024-06"
    )

    with pytest.raises(
        ValueError,
        match="Missing Yellow Taxi source months",
    ):
        resolve_sample_sources(
            specification,
            incomplete_source_files,
            landing_directory,
        )


def test_load_taxi_zone_ids_returns_sorted_unique_domain(
    spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Zone identifiers should be returned uniquely and sorted."""
    taxi_zone_path = tmp_path / "taxi_zone_lookup.csv"

    taxi_zone_path.write_text(
        "LocationID,Borough,Zone,service_zone\n"
        "3,Queens,Three,Boro Zone\n"
        "1,EWR,One,EWR\n"
        "2,Brooklyn,Two,Boro Zone\n",
        encoding="utf-8",
    )

    assert load_taxi_zone_ids(
        spark,
        taxi_zone_path,
        expected_row_count=3,
    ) == (
        1,
        2,
        3,
    )


def test_load_taxi_zone_ids_rejects_duplicates(
    spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Duplicate LocationID values should fail explicitly."""
    taxi_zone_path = tmp_path / "taxi_zone_lookup.csv"

    taxi_zone_path.write_text(
        "LocationID,Borough,Zone,service_zone\n"
        "1,EWR,One,EWR\n"
        "1,Queens,Duplicate,Boro Zone\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="LocationID values must be unique",
    ):
        load_taxi_zone_ids(
            spark,
            taxi_zone_path,
            expected_row_count=2,
        )
