"""Delta Lake writing utilities for Gold analytical DataFrames."""

from pathlib import Path

from pyspark.sql import DataFrame

from taxi_lakehouse.quality_rules import source_month_bounds

DELTA_FORMAT = "delta"

GOLD_PARTITION_COLUMN = "_source_month"


def require_frame_column(
    frame: DataFrame,
    column_name: str,
) -> None:
    """Require one column to be present in a DataFrame."""
    if column_name not in frame.columns:
        raise ValueError(f"DataFrame must contain column {column_name!r}.")


def write_gold_month(
    frame: DataFrame,
    destination_path: Path,
    source_month: str,
) -> None:
    """Replace one monthly partition in a Gold Delta table."""
    source_month_bounds(source_month)

    require_frame_column(
        frame,
        GOLD_PARTITION_COLUMN,
    )

    replacement_predicate = f"{GOLD_PARTITION_COLUMN} = '{source_month}'"

    (
        frame.write.format(DELTA_FORMAT)
        .mode("overwrite")
        .option(
            "replaceWhere",
            replacement_predicate,
        )
        .partitionBy(GOLD_PARTITION_COLUMN)
        .save(destination_path.as_posix())
    )
