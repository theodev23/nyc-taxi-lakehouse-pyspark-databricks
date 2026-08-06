"""Tests for Bronze DataFrame loading and lineage metadata."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from taxi_lakehouse.bronze_loading import (
    BRONZE_METADATA_COLUMNS,
    format_utc_timestamp_ntz,
    load_bronze_taxi_zone_source,
    load_bronze_trip_source,
)
from taxi_lakehouse.bronze_schemas import (
    TAXI_ZONE_SOURCE_SCHEMA,
    YELLOW_TAXI_SOURCE_SCHEMA,
)
from taxi_lakehouse.bronze_sources import ResolvedBronzeTripSource
from taxi_lakehouse.data_acquisition import (
    SourceFile,
    load_source_files,
)

PROJECT_MANIFEST_PATH = Path("data/source_manifest.json")
SAMPLE_TRIP_PATH = Path("data/sample/yellow_tripdata_2024-01.parquet")
SAMPLE_TAXI_ZONE_PATH = Path("data/sample/taxi_zone_lookup.csv")


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Create one local Spark session for Bronze loading tests."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("bronze-loading-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


def january_source_file() -> SourceFile:
    """Return the January source metadata from the project manifest."""
    return next(
        source_file
        for source_file in load_source_files(PROJECT_MANIFEST_PATH)
        if source_file.source_month == "2024-01"
    )


def taxi_zone_source_file() -> SourceFile:
    """Return the taxi-zone metadata from the project manifest."""
    return next(
        source_file
        for source_file in load_source_files(PROJECT_MANIFEST_PATH)
        if source_file.kind == "taxi_zone_lookup"
    )


def resolved_january_sample(
    source_file: SourceFile | None = None,
) -> ResolvedBronzeTripSource:
    """Build a resolved source backed by the versioned January sample."""
    actual_source_file = source_file or january_source_file()

    return ResolvedBronzeTripSource(
        source_month="2024-01",
        source_file=actual_source_file,
        file_path=SAMPLE_TRIP_PATH,
    )


def test_format_utc_timestamp_ntz_preserves_utc_clock_time() -> None:
    """UTC instants should become timezone-free timestamp text."""
    assert (
        format_utc_timestamp_ntz(
            datetime(
                2026,
                8,
                6,
                7,
                45,
                12,
                345678,
                tzinfo=UTC,
            )
        )
        == "2026-08-06 07:45:12.345678"
    )


@pytest.mark.parametrize(
    "invalid_timestamp",
    [
        datetime(2026, 8, 6, 7, 45),
        datetime(
            2026,
            8,
            6,
            9,
            45,
            tzinfo=timezone(timedelta(hours=2)),
        ),
    ],
)
def test_format_utc_timestamp_ntz_rejects_non_utc_values(
    invalid_timestamp: datetime,
) -> None:
    """The ingestion timestamp must be timezone-aware and UTC."""
    with pytest.raises(
        ValueError,
        match="timezone-aware and expressed in UTC",
    ):
        format_utc_timestamp_ntz(invalid_timestamp)


def test_load_bronze_trip_source_adds_lineage_metadata(
    spark: SparkSession,
) -> None:
    """Bronze loading should preserve source columns and add lineage."""
    ingestion_instant = datetime(
        2026,
        8,
        6,
        7,
        45,
        tzinfo=UTC,
    )

    frame = load_bronze_trip_source(
        spark,
        resolved_january_sample(),
        ingestion_instant,
    )

    assert frame.columns == (
        YELLOW_TAXI_SOURCE_SCHEMA.fieldNames() + list(BRONZE_METADATA_COLUMNS)
    )
    assert frame.count() == 250
    assert frame.schema["_ingested_at_utc"].dataType.simpleString() == ("timestamp_ntz")

    metadata = frame.select(*BRONZE_METADATA_COLUMNS).first()

    assert metadata["_source_file"] == ("yellow_tripdata_2024-01.parquet")
    assert metadata["_source_kind"] == "yellow_taxi_trip_data"
    assert metadata["_source_month"] == "2024-01"
    assert metadata["_source_sha256"] == january_source_file().sha256
    assert metadata["_ingested_at_utc"] == datetime(
        2026,
        8,
        6,
        7,
        45,
    )


def test_load_bronze_trip_source_requires_sha256(
    spark: SparkSession,
) -> None:
    """Bronze lineage requires an immutable source checksum."""
    source_file = replace(
        january_source_file(),
        sha256=None,
    )

    with pytest.raises(
        ValueError,
        match="requires a source SHA-256 checksum",
    ):
        load_bronze_trip_source(
            spark,
            resolved_january_sample(source_file),
            datetime(2026, 8, 6, 7, 45, tzinfo=UTC),
        )


def test_load_bronze_taxi_zone_source_adds_lineage_metadata(
    spark: SparkSession,
) -> None:
    """Taxi-zone loading should preserve source columns and add lineage."""
    ingestion_instant = datetime(
        2026,
        8,
        6,
        7,
        50,
        tzinfo=UTC,
    )

    frame = load_bronze_taxi_zone_source(
        spark,
        taxi_zone_source_file(),
        SAMPLE_TAXI_ZONE_PATH,
        ingestion_instant,
    )

    assert frame.columns == (
        TAXI_ZONE_SOURCE_SCHEMA.fieldNames() + list(BRONZE_METADATA_COLUMNS)
    )
    assert frame.count() == 265
    assert frame.schema["_source_month"].dataType.simpleString() == "string"
    assert frame.schema["_ingested_at_utc"].dataType.simpleString() == ("timestamp_ntz")

    metadata = frame.select(*BRONZE_METADATA_COLUMNS).first()

    assert metadata["_source_file"] == "taxi_zone_lookup.csv"
    assert metadata["_source_kind"] == "taxi_zone_lookup"
    assert metadata["_source_month"] is None
    assert metadata["_source_sha256"] == taxi_zone_source_file().sha256
    assert metadata["_ingested_at_utc"] == datetime(
        2026,
        8,
        6,
        7,
        50,
    )


def test_load_bronze_taxi_zone_source_rejects_source_month(
    spark: SparkSession,
) -> None:
    """The non-monthly taxi-zone source must not declare a month."""
    source_file = replace(
        taxi_zone_source_file(),
        source_month="2024-01",
    )

    with pytest.raises(
        ValueError,
        match="source_month must be null",
    ):
        load_bronze_taxi_zone_source(
            spark,
            source_file,
            SAMPLE_TAXI_ZONE_PATH,
            datetime(2026, 8, 6, 7, 50, tzinfo=UTC),
        )
