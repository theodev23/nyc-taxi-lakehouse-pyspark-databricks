"""Loading utilities for Gold processing from Silver and Bronze Delta tables."""

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from taxi_lakehouse.quality_rules import source_month_bounds

SILVER_TRIP_PARTITION_COLUMN = "_source_month"

TAXI_ZONE_REQUIRED_COLUMNS = {
    "LocationID",
    "Borough",
    "Zone",
    "service_zone",
}


def require_frame_columns(
    frame: DataFrame,
    required_columns: set[str],
    frame_name: str,
) -> None:
    """Require every expected column to be present in a DataFrame."""
    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"{frame_name} is missing required columns: {sorted(missing_columns)!r}."
        )


def load_silver_accepted_month(
    spark: SparkSession,
    source_path: Path,
    source_month: str,
) -> DataFrame:
    """Load one monthly partition from Silver accepted trips."""
    source_month_bounds(source_month)

    frame = spark.read.format("delta").load(source_path.as_posix())

    require_frame_columns(
        frame,
        {SILVER_TRIP_PARTITION_COLUMN},
        "Silver accepted trip DataFrame",
    )

    return frame.filter(F.col(SILVER_TRIP_PARTITION_COLUMN) == source_month)


def load_silver_source_months(
    spark: SparkSession,
    source_path: Path,
) -> tuple[str, ...]:
    """Load the ordered source-month domain from Silver accepted trips."""
    frame = spark.read.format("delta").load(source_path.as_posix())

    require_frame_columns(
        frame,
        {SILVER_TRIP_PARTITION_COLUMN},
        "Silver accepted trip DataFrame",
    )

    source_months = tuple(
        row[SILVER_TRIP_PARTITION_COLUMN]
        for row in (
            frame.select(SILVER_TRIP_PARTITION_COLUMN)
            .where(F.col(SILVER_TRIP_PARTITION_COLUMN).isNotNull())
            .distinct()
            .orderBy(SILVER_TRIP_PARTITION_COLUMN)
            .collect()
        )
    )

    if not source_months:
        raise ValueError(
            "Silver accepted trip table must contain at least one source month."
        )

    for source_month in source_months:
        source_month_bounds(source_month)

    return source_months


def load_taxi_zones(
    spark: SparkSession,
    source_path: Path,
) -> DataFrame:
    """Load the Bronze taxi-zone reference required by Gold."""
    frame = spark.read.format("delta").load(source_path.as_posix())

    require_frame_columns(
        frame,
        TAXI_ZONE_REQUIRED_COLUMNS,
        "Bronze taxi-zone DataFrame",
    )

    return frame
