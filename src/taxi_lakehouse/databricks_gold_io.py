"""Unity Catalog I/O for Databricks Gold processing."""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from taxi_lakehouse.databricks_table_io import (
    read_managed_table,
    replace_managed_table_rows,
)
from taxi_lakehouse.gold_loading import (
    SILVER_TRIP_PARTITION_COLUMN,
    TAXI_ZONE_REQUIRED_COLUMNS,
    require_frame_columns,
)
from taxi_lakehouse.gold_writing import GOLD_PARTITION_COLUMN
from taxi_lakehouse.quality_rules import source_month_bounds


def load_managed_silver_accepted_month(
    spark: SparkSession,
    table_name: str,
    source_month: str,
) -> DataFrame:
    """Load one accepted Silver month from a managed table."""
    source_month_bounds(source_month)

    frame = read_managed_table(
        spark,
        table_name,
    )

    require_frame_columns(
        frame,
        {SILVER_TRIP_PARTITION_COLUMN},
        "Silver accepted trip DataFrame",
    )

    return frame.filter(F.col(SILVER_TRIP_PARTITION_COLUMN) == source_month)


def load_managed_silver_source_months(
    spark: SparkSession,
    table_name: str,
) -> tuple[str, ...]:
    """Return ordered source months from managed Silver accepted trips."""
    frame = read_managed_table(
        spark,
        table_name,
    )

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
            "Managed Silver accepted trip table must contain at least one source month."
        )

    for source_month in source_months:
        source_month_bounds(source_month)

    return source_months


def load_managed_taxi_zones(
    spark: SparkSession,
    table_name: str,
) -> DataFrame:
    """Load the managed Bronze taxi-zone reference."""
    frame = read_managed_table(
        spark,
        table_name,
    )

    require_frame_columns(
        frame,
        TAXI_ZONE_REQUIRED_COLUMNS,
        "Bronze taxi-zone DataFrame",
    )

    return frame


def write_managed_gold_month(
    frame: DataFrame,
    table_name: str,
    source_month: str,
) -> None:
    """Replace one Gold month in a Unity Catalog managed table."""
    source_month_bounds(source_month)

    require_frame_columns(
        frame,
        {GOLD_PARTITION_COLUMN},
        "Gold analytical DataFrame",
    )

    predicate = f"{GOLD_PARTITION_COLUMN} = '{source_month}'"

    replace_managed_table_rows(
        frame,
        table_name,
        predicate,
    )
