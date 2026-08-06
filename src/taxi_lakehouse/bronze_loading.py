"""Loading utilities for Bronze DataFrames."""

from datetime import UTC, datetime, timedelta

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from taxi_lakehouse.bronze_schemas import YELLOW_TAXI_SOURCE_SCHEMA
from taxi_lakehouse.bronze_sources import ResolvedBronzeTripSource

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


def load_bronze_trip_source(
    spark: SparkSession,
    source: ResolvedBronzeTripSource,
    ingested_at_utc: datetime,
) -> DataFrame:
    """Load one monthly Yellow Taxi source with Bronze lineage metadata."""
    source_sha256 = source.source_file.sha256

    if source_sha256 is None:
        raise ValueError("Bronze ingestion requires a source SHA-256 checksum.")

    ingestion_timestamp_text = format_utc_timestamp_ntz(ingested_at_utc)

    return (
        spark.read.schema(YELLOW_TAXI_SOURCE_SCHEMA)
        .parquet(source.file_path.as_posix())
        .withColumn(
            "_source_file",
            F.lit(source.source_file.filename),
        )
        .withColumn(
            "_source_kind",
            F.lit(source.source_file.kind),
        )
        .withColumn(
            "_source_month",
            F.lit(source.source_month),
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
