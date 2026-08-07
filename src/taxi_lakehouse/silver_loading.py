"""Loading utilities for Silver processing from Bronze Delta tables."""

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from taxi_lakehouse.quality_rules import source_month_bounds

BRONZE_TRIP_PARTITION_COLUMN = "_source_month"
TAXI_ZONE_ID_COLUMN = "LocationID"


def load_bronze_trip_month(
    spark: SparkSession,
    source_path: Path,
    source_month: str,
) -> DataFrame:
    """Load one monthly partition from the Bronze trips Delta table."""
    source_month_bounds(source_month)

    frame = spark.read.format("delta").load(source_path.as_posix())

    if BRONZE_TRIP_PARTITION_COLUMN not in frame.columns:
        raise ValueError(
            "Bronze trip DataFrame must contain "
            f"column {BRONZE_TRIP_PARTITION_COLUMN!r}."
        )

    return frame.filter(F.col(BRONZE_TRIP_PARTITION_COLUMN) == source_month)


def load_bronze_zone_ids(
    spark: SparkSession,
    source_path: Path,
) -> tuple[int, ...]:
    """Load the distinct non-null taxi-zone identifiers from Bronze."""
    frame = spark.read.format("delta").load(source_path.as_posix())

    if TAXI_ZONE_ID_COLUMN not in frame.columns:
        raise ValueError(
            f"Bronze taxi-zone DataFrame must contain column {TAXI_ZONE_ID_COLUMN!r}."
        )

    zone_ids = tuple(
        row[TAXI_ZONE_ID_COLUMN]
        for row in (
            frame.select(TAXI_ZONE_ID_COLUMN)
            .where(F.col(TAXI_ZONE_ID_COLUMN).isNotNull())
            .distinct()
            .orderBy(TAXI_ZONE_ID_COLUMN)
            .collect()
        )
    )

    if not zone_ids:
        raise ValueError("Bronze taxi-zone table must contain at least one LocationID.")

    return zone_ids
