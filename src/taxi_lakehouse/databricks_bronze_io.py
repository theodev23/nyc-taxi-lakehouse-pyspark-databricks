"""Unity Catalog I/O for Databricks Bronze processing."""

from pyspark.sql import DataFrame

from taxi_lakehouse.bronze_sources import require_source_month
from taxi_lakehouse.bronze_writing import (
    BRONZE_TRIP_PARTITION_COLUMN,
    require_frame_column,
)
from taxi_lakehouse.databricks_table_io import (
    overwrite_managed_table,
    replace_managed_table_rows,
)


def write_managed_bronze_trip_month(
    frame: DataFrame,
    table_name: str,
    source_month: str,
) -> None:
    """Replace one Bronze trip month in a managed table."""
    validated_source_month = require_source_month(source_month)

    require_frame_column(
        frame,
        BRONZE_TRIP_PARTITION_COLUMN,
    )

    predicate = f"{BRONZE_TRIP_PARTITION_COLUMN} = '{validated_source_month}'"

    replace_managed_table_rows(
        frame,
        table_name,
        predicate,
    )


def write_managed_bronze_taxi_zones(
    frame: DataFrame,
    table_name: str,
) -> None:
    """Replace the complete managed Bronze taxi-zone snapshot."""
    overwrite_managed_table(
        frame,
        table_name,
    )
