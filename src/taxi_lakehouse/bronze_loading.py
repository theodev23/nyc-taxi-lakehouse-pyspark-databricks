"""Loading utilities for Bronze DataFrames."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from taxi_lakehouse.bronze_schemas import (
    TAXI_ZONE_SOURCE_SCHEMA,
    YELLOW_TAXI_SOURCE_SCHEMA,
)
from taxi_lakehouse.bronze_sources import ResolvedBronzeTripSource
from taxi_lakehouse.data_acquisition import SourceFile

BRONZE_METADATA_COLUMNS = (
    "_source_file",
    "_source_kind",
    "_source_month",
    "_source_sha256",
    "_ingested_at_utc",
)


def format_utc_timestamp_ntz(value: datetime) -> str:
    """Validate and format one UTC instant for Spark timestamp_ntz."""
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("ingested_at_utc must be timezone-aware and expressed in UTC.")

    return (
        value.astimezone(UTC)
        .replace(tzinfo=None)
        .isoformat(
            sep=" ",
            timespec="microseconds",
        )
    )


def _add_bronze_lineage(
    frame: DataFrame,
    *,
    source_file: SourceFile,
    source_month: str | None,
    ingested_at_utc: datetime,
) -> DataFrame:
    """Add standardized Bronze lineage columns to one source DataFrame."""
    source_sha256 = source_file.sha256

    if source_sha256 is None:
        raise ValueError("Bronze ingestion requires a source SHA-256 checksum.")

    ingestion_timestamp_text = format_utc_timestamp_ntz(ingested_at_utc)

    return (
        frame.withColumn(
            "_source_file",
            F.lit(source_file.filename),
        )
        .withColumn(
            "_source_kind",
            F.lit(source_file.kind),
        )
        .withColumn(
            "_source_month",
            F.lit(source_month).cast("string"),
        )
        .withColumn(
            "_source_sha256",
            F.lit(source_sha256),
        )
        .withColumn(
            "_ingested_at_utc",
            F.lit(ingestion_timestamp_text).cast("timestamp_ntz"),
        )
    )


def load_bronze_trip_source(
    spark: SparkSession,
    source: ResolvedBronzeTripSource,
    ingested_at_utc: datetime,
) -> DataFrame:
    """Load one monthly Yellow Taxi source with Bronze lineage metadata."""
    frame = spark.read.schema(YELLOW_TAXI_SOURCE_SCHEMA).parquet(
        source.file_path.as_posix()
    )

    return _add_bronze_lineage(
        frame,
        source_file=source.source_file,
        source_month=source.source_month,
        ingested_at_utc=ingested_at_utc,
    )


def load_bronze_taxi_zone_source(
    spark: SparkSession,
    source_file: SourceFile,
    file_path: Path,
    ingested_at_utc: datetime,
) -> DataFrame:
    """Load the taxi-zone lookup with Bronze lineage metadata."""
    if source_file.source_month is not None:
        raise ValueError("Taxi zone lookup source_month must be null.")

    frame = (
        spark.read.schema(TAXI_ZONE_SOURCE_SCHEMA)
        .option("header", True)
        .option("mode", "FAILFAST")
        .csv(file_path.as_posix())
    )

    return _add_bronze_lineage(
        frame,
        source_file=source_file,
        source_month=None,
        ingested_at_utc=ingested_at_utc,
    )
