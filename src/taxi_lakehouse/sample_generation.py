"""PySpark transformations for deterministic sample generation."""

import hashlib

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from taxi_lakehouse.quality_rules import (
    QUALITY_RULE_NAMES,
    build_quality_rule_conditions,
    source_month_bounds,
)

QUALITY_BUCKET_NAMES = QUALITY_RULE_NAMES | frozenset({"normal"})


def build_quality_bucket_expression(
    source_month: str,
    zone_ids: tuple[int, ...],
    quality_bucket_priority: tuple[str, ...],
) -> Column:
    """Build the prioritized quality-bucket Spark expression."""
    if not zone_ids:
        raise ValueError("zone_ids must contain at least one identifier.")

    if len(zone_ids) != len(set(zone_ids)):
        raise ValueError("zone_ids must not contain duplicates.")

    unknown_buckets = set(quality_bucket_priority) - QUALITY_BUCKET_NAMES

    if unknown_buckets:
        raise ValueError(f"Unsupported quality buckets: {sorted(unknown_buckets)!r}.")

    if "normal" not in quality_bucket_priority:
        raise ValueError("quality_bucket_priority must contain normal.")

    if len(quality_bucket_priority) != len(set(quality_bucket_priority)):
        raise ValueError("quality_bucket_priority must not contain duplicates.")

    non_normal_buckets = tuple(
        bucket_name
        for bucket_name in quality_bucket_priority
        if bucket_name != "normal"
    )

    if not non_normal_buckets:
        return F.lit("normal")

    month_start, month_end = source_month_bounds(source_month)

    month_start_literal = F.lit(month_start.strftime("%Y-%m-%d %H:%M:%S")).cast(
        "timestamp_ntz"
    )

    month_end_literal = F.lit(month_end.strftime("%Y-%m-%d %H:%M:%S")).cast(
        "timestamp_ntz"
    )

    conditions = build_quality_rule_conditions(
        zone_ids,
        month_start_literal,
        month_end_literal,
    )

    bucket_expression: Column | None = None

    for bucket_name in quality_bucket_priority:
        if bucket_name == "normal":
            continue

        condition = conditions[bucket_name]

        if bucket_expression is None:
            bucket_expression = F.when(
                condition,
                F.lit(bucket_name),
            )
        else:
            bucket_expression = bucket_expression.when(
                condition,
                F.lit(bucket_name),
            )

    if bucket_expression is None:
        return F.lit("normal")

    return bucket_expression.otherwise(F.lit("normal"))


def build_canonical_row_json_expression(
    source_columns: tuple[str, ...],
) -> Column:
    """Build the canonical JSON representation of one source row."""
    if not source_columns:
        raise ValueError("source_columns must contain at least one column.")

    if any(
        not isinstance(column_name, str) or not column_name
        for column_name in source_columns
    ):
        raise ValueError("source_columns must contain only non-empty strings.")

    if len(source_columns) != len(set(source_columns)):
        raise ValueError("source_columns must not contain duplicates.")

    return F.to_json(
        F.struct(
            *[F.col(column_name).alias(column_name) for column_name in source_columns]
        ),
        {
            "ignoreNullFields": "false",
            "timestampFormat": ("yyyy-MM-dd'T'HH:mm:ss.SSSSSS"),
            "timestampNTZFormat": ("yyyy-MM-dd'T'HH:mm:ss.SSSSSS"),
        },
    )


def build_row_hash_expression(
    source_month: str,
    canonical_row_json: Column,
) -> Column:
    """Build the deterministic SHA-256 hash of one source row."""
    source_month_bounds(source_month)

    return F.sha2(
        F.concat_ws(
            "\u001f",
            F.lit(source_month),
            canonical_row_json,
        ),
        256,
    )


LOGICAL_SAMPLE_SHA256_ALGORITHM = "sha256_ordered_row_hashes_v1"


def calculate_logical_sample_sha256(
    frame: DataFrame,
    source_month: str,
    source_columns: tuple[str, ...],
) -> str:
    """Calculate a deterministic checksum of logical sample rows."""
    canonical_json = build_canonical_row_json_expression(source_columns)

    row_hash = build_row_hash_expression(
        source_month,
        canonical_json,
    )

    ordered_row_hashes = tuple(
        row["_row_hash"]
        for row in (
            frame.select(
                row_hash.alias("_row_hash"),
                canonical_json.alias("_canonical_json"),
            )
            .orderBy(
                "_row_hash",
                "_canonical_json",
            )
            .collect()
        )
    )

    serialized_hashes = "\n".join(ordered_row_hashes).encode("utf-8")

    return hashlib.sha256(serialized_hashes).hexdigest()


TECHNICAL_SAMPLE_COLUMNS = frozenset(
    {
        "_quality_bucket",
        "_canonical_json",
        "_row_hash",
        "_configured_quota",
        "_sample_rank",
    }
)


def select_deterministic_sample_rows(
    frame: DataFrame,
    source_month: str,
    source_columns: tuple[str, ...],
    zone_ids: tuple[int, ...],
    quality_bucket_priority: tuple[str, ...],
    quota_by_bucket: dict[str, int],
) -> DataFrame:
    """Select a deterministic quota of source rows per quality bucket."""
    source_month_bounds(source_month)

    if not source_columns:
        raise ValueError("source_columns must contain at least one column.")

    missing_columns = tuple(
        column_name
        for column_name in source_columns
        if column_name not in frame.columns
    )

    if missing_columns:
        raise ValueError(
            f"Source columns are missing from the DataFrame: {missing_columns!r}."
        )

    technical_collisions = tuple(sorted(set(frame.columns) & TECHNICAL_SAMPLE_COLUMNS))

    if technical_collisions:
        raise ValueError(
            f"DataFrame contains reserved sample columns: {technical_collisions!r}."
        )

    if not quota_by_bucket:
        raise ValueError("quota_by_bucket must contain at least one quota.")

    invalid_quota_names = tuple(
        bucket_name
        for bucket_name in quota_by_bucket
        if not isinstance(bucket_name, str) or not bucket_name
    )

    if invalid_quota_names:
        raise ValueError("Quota bucket names must be non-empty strings.")

    invalid_quota_values = tuple(
        bucket_name
        for bucket_name, quota in quota_by_bucket.items()
        if not isinstance(quota, int) or isinstance(quota, bool) or quota <= 0
    )

    if invalid_quota_values:
        raise ValueError(
            f"Sample quotas must be positive integers: {invalid_quota_values!r}."
        )

    unsupported_quota_buckets = tuple(
        sorted(set(quota_by_bucket) - set(quality_bucket_priority))
    )

    if unsupported_quota_buckets:
        raise ValueError(
            "Quota buckets are absent from quality priority: "
            f"{unsupported_quota_buckets!r}."
        )

    quality_bucket = build_quality_bucket_expression(
        source_month,
        zone_ids,
        quality_bucket_priority,
    )

    canonical_json = build_canonical_row_json_expression(source_columns)

    row_hash = build_row_hash_expression(
        source_month,
        canonical_json,
    )

    quota_entries = []

    for bucket_name in sorted(quota_by_bucket):
        quota_entries.extend(
            [
                F.lit(bucket_name),
                F.lit(quota_by_bucket[bucket_name]),
            ]
        )

    quota_map = F.create_map(*quota_entries)

    ranking_window = Window.partitionBy("_quality_bucket").orderBy(
        F.col("_row_hash"),
        F.col("_canonical_json"),
    )

    return (
        frame.withColumn(
            "_quality_bucket",
            quality_bucket,
        )
        .withColumn(
            "_canonical_json",
            canonical_json,
        )
        .withColumn(
            "_row_hash",
            row_hash,
        )
        .withColumn(
            "_configured_quota",
            F.element_at(
                quota_map,
                F.col("_quality_bucket"),
            ),
        )
        .withColumn(
            "_sample_rank",
            F.row_number().over(ranking_window),
        )
        .where(
            F.col("_configured_quota").isNotNull()
            & (F.col("_sample_rank") <= F.col("_configured_quota"))
        )
        .select(*[F.col(column_name) for column_name in source_columns])
    )
