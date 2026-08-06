"""Delta Lake writing utilities for Bronze DataFrames."""

from pathlib import Path

from pyspark.sql import DataFrame

from taxi_lakehouse.bronze_sources import require_source_month

DELTA_FORMAT = "delta"
BRONZE_TRIP_PARTITION_COLUMN = "_source_month"


def require_frame_column(
    frame: DataFrame,
    column_name: str,
) -> None:
    """Require one column to be present in a DataFrame."""
    if column_name not in frame.columns:
        raise ValueError(f"DataFrame must contain column {column_name!r}.")


def write_bronze_trip_month(
    frame: DataFrame,
    destination_path: Path,
    source_month: str,
) -> None:
    """Replace one monthly partition in the Bronze trips Delta table."""
    validated_source_month = require_source_month(source_month)

    require_frame_column(
        frame,
        BRONZE_TRIP_PARTITION_COLUMN,
    )

    replacement_predicate = (
        f"{BRONZE_TRIP_PARTITION_COLUMN} = '{validated_source_month}'"
    )

    (
        frame.write.format(DELTA_FORMAT)
        .mode("overwrite")
        .option(
            "replaceWhere",
            replacement_predicate,
        )
        .partitionBy(BRONZE_TRIP_PARTITION_COLUMN)
        .save(destination_path.as_posix())
    )


def write_bronze_taxi_zones(
    frame: DataFrame,
    destination_path: Path,
) -> None:
    """Replace the complete Bronze taxi-zone Delta snapshot."""
    (
        frame.write.format(DELTA_FORMAT)
        .mode("overwrite")
        .save(destination_path.as_posix())
    )
