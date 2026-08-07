"""Shared PySpark expressions for NYC Taxi data-quality rules."""

from pyspark.sql import Column
from pyspark.sql import functions as F

QUALITY_RULE_NAMES = frozenset(
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
    }
)


def build_quality_rule_conditions(
    zone_ids: tuple[int, ...],
    month_start: Column,
    month_end: Column,
) -> dict[str, Column]:
    """Build every shared NYC Taxi quality-rule condition."""
    if not zone_ids:
        raise ValueError("zone_ids must contain at least one identifier.")

    if len(zone_ids) != len(set(zone_ids)):
        raise ValueError("zone_ids must not contain duplicates.")

    pickup = F.col("tpep_pickup_datetime")
    dropoff = F.col("tpep_dropoff_datetime")
    passenger_count = F.col("passenger_count")
    trip_distance = F.col("trip_distance")
    pickup_zone = F.col("PULocationID")
    dropoff_zone = F.col("DOLocationID")
    total_amount = F.col("total_amount")

    duration_seconds = F.unix_timestamp(dropoff) - F.unix_timestamp(pickup)

    return {
        "pickup_zone_unknown": (pickup_zone.isNull() | ~pickup_zone.isin(*zone_ids)),
        "dropoff_zone_unknown": (dropoff_zone.isNull() | ~dropoff_zone.isin(*zone_ids)),
        "distance_negative": (trip_distance < 0),
        "duration_over_24h": (duration_seconds > 86_400),
        "passenger_over_6": (passenger_count.isNotNull() & (passenger_count > 6)),
        "total_amount_zero": (total_amount == 0),
        "duration_nonpositive": (duration_seconds.isNull() | (duration_seconds <= 0)),
        "pickup_outside_month": (
            pickup.isNull() | (pickup < month_start) | (pickup >= month_end)
        ),
        "total_amount_negative": (total_amount < 0),
        "passenger_nonpositive": (passenger_count.isNotNull() & (passenger_count <= 0)),
        "distance_zero": (trip_distance == 0),
        "passenger_missing": (passenger_count.isNull()),
    }
