"""Deterministic profiling for the generated NYC Taxi sample."""

from datetime import datetime
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

SAMPLE_PROFILE_SCHEMA_VERSION = "1.0"

PROFILE_DISTINCT_COLUMNS = (
    "VendorID",
    "RatecodeID",
    "store_and_fwd_flag",
    "payment_type",
    "PULocationID",
    "DOLocationID",
)

PROFILE_TEMPORAL_COLUMNS = (
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
)

PROFILE_NUMERIC_COLUMNS = (
    "passenger_count",
    "trip_distance",
    "total_amount",
)


def serialize_profile_value(
    value: Any,
) -> Any:
    """Convert Spark values to deterministic JSON-compatible values."""
    if isinstance(value, datetime):
        return value.isoformat(timespec="microseconds")

    return value


def build_sample_profile_payload(
    frame: DataFrame,
    source_columns: tuple[str, ...],
    source_month_column: str = "_source_month",
    quality_bucket_column: str = "_quality_bucket",
) -> dict[str, Any]:
    """Build deterministic quality metrics for an enriched sample frame."""
    if not source_columns:
        raise ValueError("source_columns must contain at least one column.")

    if len(source_columns) != len(set(source_columns)):
        raise ValueError("source_columns must not contain duplicates.")

    required_profile_columns = set(
        PROFILE_DISTINCT_COLUMNS + PROFILE_TEMPORAL_COLUMNS + PROFILE_NUMERIC_COLUMNS
    )

    missing_profile_columns = tuple(
        sorted(required_profile_columns - set(source_columns))
    )

    if missing_profile_columns:
        raise ValueError(
            "Required profile columns are absent from "
            f"source_columns: {missing_profile_columns!r}."
        )

    required_frame_columns = (
        *source_columns,
        source_month_column,
        quality_bucket_column,
    )

    missing_frame_columns = tuple(
        column_name
        for column_name in required_frame_columns
        if column_name not in frame.columns
    )

    if missing_frame_columns:
        raise ValueError(
            "Required profile columns are missing from "
            f"the DataFrame: {missing_frame_columns!r}."
        )

    total_row_count = frame.count()

    if total_row_count == 0:
        raise ValueError("Sample profile requires at least one row.")

    rows_by_source_month = {
        row[source_month_column]: row["row_count"]
        for row in (
            frame.groupBy(source_month_column)
            .agg(F.count(F.lit(1)).alias("row_count"))
            .orderBy(source_month_column)
            .collect()
        )
    }

    if None in rows_by_source_month:
        raise ValueError("Source-month values must not be null.")

    quality_bucket_counts = {
        row[quality_bucket_column]: row["row_count"]
        for row in (
            frame.groupBy(quality_bucket_column)
            .agg(F.count(F.lit(1)).alias("row_count"))
            .orderBy(quality_bucket_column)
            .collect()
        )
    }

    if None in quality_bucket_counts:
        raise ValueError("Quality-bucket values must not be null.")

    null_count_row = frame.agg(
        *[
            F.sum(
                F.when(
                    F.col(column_name).isNull(),
                    F.lit(1),
                ).otherwise(F.lit(0))
            ).alias(column_name)
            for column_name in source_columns
        ]
    ).first()

    null_statistics = {}

    for column_name in source_columns:
        null_count = int(null_count_row[column_name])

        null_statistics[column_name] = {
            "null_count": null_count,
            "non_null_count": (total_row_count - null_count),
            "null_rate": round(
                null_count / total_row_count,
                6,
            ),
        }

    distinct_count_row = frame.agg(
        *[
            (
                F.countDistinct(F.col(column_name))
                + F.max(F.col(column_name).isNull().cast("long"))
            ).alias(column_name)
            for column_name in PROFILE_DISTINCT_COLUMNS
        ]
    ).first()

    distinct_counts = {
        column_name: int(distinct_count_row[column_name])
        for column_name in PROFILE_DISTINCT_COLUMNS
    }

    temporal_expressions = []

    for column_name in PROFILE_TEMPORAL_COLUMNS:
        temporal_expressions.extend(
            (
                F.min(column_name).alias(f"{column_name}__min"),
                F.max(column_name).alias(f"{column_name}__max"),
            )
        )

    temporal_row = frame.agg(*temporal_expressions).first()

    temporal_ranges = {
        column_name: {
            "minimum": serialize_profile_value(temporal_row[f"{column_name}__min"]),
            "maximum": serialize_profile_value(temporal_row[f"{column_name}__max"]),
        }
        for column_name in PROFILE_TEMPORAL_COLUMNS
    }

    numeric_expressions = []

    for column_name in PROFILE_NUMERIC_COLUMNS:
        numeric_expressions.extend(
            (
                F.min(column_name).alias(f"{column_name}__min"),
                F.max(column_name).alias(f"{column_name}__max"),
            )
        )

    numeric_row = frame.agg(*numeric_expressions).first()

    numeric_ranges = {
        column_name: {
            "minimum": serialize_profile_value(numeric_row[f"{column_name}__min"]),
            "maximum": serialize_profile_value(numeric_row[f"{column_name}__max"]),
        }
        for column_name in PROFILE_NUMERIC_COLUMNS
    }

    field_by_name = {field.name: field for field in frame.schema.fields}

    schema_profile = [
        {
            "name": column_name,
            "data_type": (field_by_name[column_name].dataType.simpleString()),
            "nullable": field_by_name[column_name].nullable,
        }
        for column_name in source_columns
    ]

    return {
        "schema_version": (SAMPLE_PROFILE_SCHEMA_VERSION),
        "summary": {
            "row_count": total_row_count,
            "column_count": len(source_columns),
            "source_month_count": len(rows_by_source_month),
            "quality_bucket_count": len(quality_bucket_counts),
        },
        "schema": schema_profile,
        "rows_by_source_month": (rows_by_source_month),
        "quality_bucket_counts": (quality_bucket_counts),
        "null_statistics": null_statistics,
        "distinct_counts_including_null": (distinct_counts),
        "temporal_ranges": temporal_ranges,
        "numeric_ranges": numeric_ranges,
    }
