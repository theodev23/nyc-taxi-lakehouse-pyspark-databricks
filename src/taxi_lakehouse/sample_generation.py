"""PySpark transformations for deterministic sample generation."""

from datetime import datetime

from pyspark.sql import Column
from pyspark.sql import functions as F

QUALITY_BUCKET_NAMES = frozenset(
    {
        "pickup_zone_unknown",
        "dropoff_zone_unknown",
        "distance_negative",
        "duration_over_24h",
        "passenger_over_6",
        "total_amount_zero",
        "duration_nonpositive",
        "pickup_outside_month",
        "total_amount_negative",
        "passenger_nonpositive",
        "distance_zero",
        "passenger_missing",
        "normal",
    }
)


def source_month_bounds(
    source_month: str,
) -> tuple[datetime, datetime]:
    """Return inclusive start and exclusive end timestamps for a month."""
    try:
        month_start = datetime.strptime(
            source_month,
            "%Y-%m",
        )
    except ValueError as error:
        raise ValueError(f"Invalid source month: {source_month!r}.") from error

    if month_start.strftime("%Y-%m") != source_month:
        raise ValueError(f"Invalid source month: {source_month!r}.")

    if month_start.month == 12:
        month_end = datetime(
            month_start.year + 1,
            1,
            1,
        )
    else:
        month_end = datetime(
            month_start.year,
            month_start.month + 1,
            1,
        )

    return month_start, month_end


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

    pickup = F.col("tpep_pickup_datetime")
    dropoff = F.col("tpep_dropoff_datetime")
    passenger_count = F.col("passenger_count")
    trip_distance = F.col("trip_distance")
    pickup_zone = F.col("PULocationID")
    dropoff_zone = F.col("DOLocationID")
    total_amount = F.col("total_amount")

    duration_seconds = F.unix_timestamp(dropoff) - F.unix_timestamp(pickup)

    conditions = {
        "pickup_zone_unknown": (pickup_zone.isNull() | ~pickup_zone.isin(*zone_ids)),
        "dropoff_zone_unknown": (dropoff_zone.isNull() | ~dropoff_zone.isin(*zone_ids)),
        "distance_negative": (trip_distance < 0),
        "duration_over_24h": (duration_seconds > 86_400),
        "passenger_over_6": (passenger_count.isNotNull() & (passenger_count > 6)),
        "total_amount_zero": (total_amount == 0),
        "duration_nonpositive": (duration_seconds.isNull() | (duration_seconds <= 0)),
        "pickup_outside_month": (
            pickup.isNull()
            | (pickup < month_start_literal)
            | (pickup >= month_end_literal)
        ),
        "total_amount_negative": (total_amount < 0),
        "passenger_nonpositive": (passenger_count.isNotNull() & (passenger_count <= 0)),
        "distance_zero": (trip_distance == 0),
        "passenger_missing": (passenger_count.isNull()),
    }

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
