"""Unity Catalog I/O for Databricks Silver processing."""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from taxi_lakehouse.databricks_table_io import (
    read_managed_table,
    replace_managed_table_rows,
)
from taxi_lakehouse.quality_rules import source_month_bounds

BRONZE_TRIP_PARTITION_COLUMN = "_source_month"
SILVER_TRIP_PARTITION_COLUMN = "_source_month"
TAXI_ZONE_ID_COLUMN = "LocationID"


def require_frame_column(
    frame: DataFrame,
    column_name: str,
) -> None:
    """Require one expected DataFrame column."""
    if column_name not in frame.columns:
        raise ValueError(f"DataFrame must contain column {column_name!r}.")


def load_managed_bronze_trip_month(
    spark: SparkSession,
    table_name: str,
    source_month: str,
) -> DataFrame:
    """Load one Bronze trip month from a Unity Catalog table."""
    source_month_bounds(source_month)

    frame = read_managed_table(
        spark,
        table_name,
    )

    require_frame_column(
        frame,
        BRONZE_TRIP_PARTITION_COLUMN,
    )

    return frame.filter(F.col(BRONZE_TRIP_PARTITION_COLUMN) == source_month)


def load_managed_bronze_source_months(
    spark: SparkSession,
    table_name: str,
) -> tuple[str, ...]:
    """Return the ordered source-month domain from managed Bronze trips."""
    frame = read_managed_table(
        spark,
        table_name,
    )

    require_frame_column(
        frame,
        BRONZE_TRIP_PARTITION_COLUMN,
    )

    source_months = tuple(
        row[BRONZE_TRIP_PARTITION_COLUMN]
        for row in (
            frame.select(BRONZE_TRIP_PARTITION_COLUMN)
            .where(F.col(BRONZE_TRIP_PARTITION_COLUMN).isNotNull())
            .distinct()
            .orderBy(BRONZE_TRIP_PARTITION_COLUMN)
            .collect()
        )
    )

    if not source_months:
        raise ValueError(
            "Managed Bronze trip table must contain at least one source month."
        )

    for source_month in source_months:
        source_month_bounds(source_month)

    return source_months


def load_managed_bronze_zone_ids(
    spark: SparkSession,
    table_name: str,
) -> tuple[int, ...]:
    """Return ordered taxi-zone identifiers from managed Bronze."""
    frame = read_managed_table(
        spark,
        table_name,
    )

    require_frame_column(
        frame,
        TAXI_ZONE_ID_COLUMN,
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
        raise ValueError(
            "Managed Bronze taxi-zone table must contain at least one LocationID."
        )

    return zone_ids


def write_managed_silver_trip_month(
    frame: DataFrame,
    table_name: str,
    source_month: str,
) -> None:
    """Replace one Silver month in a Unity Catalog managed table."""
    source_month_bounds(source_month)

    require_frame_column(
        frame,
        SILVER_TRIP_PARTITION_COLUMN,
    )

    predicate = f"{SILVER_TRIP_PARTITION_COLUMN} = '{source_month}'"

    replace_managed_table_rows(
        frame,
        table_name,
        predicate,
    )
