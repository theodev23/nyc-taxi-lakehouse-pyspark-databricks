"""PySpark transformations for Gold analytical metrics."""

from dataclasses import dataclass

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from taxi_lakehouse.gold_analytics_specification import (
    GoldAnalyticsSpecification,
)

TRIP_METRICS_GRAIN = (
    "_source_month",
    "pickup_date",
    "pickup_location_id",
    "payment_type",
)

TRIP_METRICS_DIMENSIONS = (
    "pickup_borough",
    "pickup_zone",
    "pickup_service_zone",
)

DAILY_METRICS_GRAIN = (
    "_source_month",
    "pickup_date",
)

SUPPORTED_METRICS = (
    "trip_count",
    "quality_flagged_trip_count",
    "trip_distance_sum",
    "trip_distance_avg",
    "trip_duration_minutes_avg",
    "fare_amount_sum",
    "tip_amount_sum",
    "total_amount_sum",
    "total_amount_avg",
)

_REQUIRED_TRIP_COLUMNS = {
    "_source_month",
    "_quality_flags",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "payment_type",
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
}

_REQUIRED_ZONE_COLUMNS = {
    "LocationID",
    "Borough",
    "Zone",
    "service_zone",
}


@dataclass(frozen=True, slots=True)
class GoldAnalyticsFrames:
    """Gold analytical DataFrames before Delta persistence."""

    trip_metrics: DataFrame
    daily_metrics: DataFrame


def _require_columns(
    frame: DataFrame,
    required_columns: set[str],
    frame_name: str,
) -> None:
    """Require every input column used by the Gold transformation."""
    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"{frame_name} is missing required columns: {sorted(missing_columns)!r}."
        )


def _validate_specification(
    specification: GoldAnalyticsSpecification,
) -> None:
    """Require the configured Gold contract implemented by this module."""
    metric_names = tuple(metric.name for metric in specification.metrics)

    if metric_names != SUPPORTED_METRICS:
        raise ValueError(f"Unsupported Gold metric contract: {metric_names!r}.")

    if specification.outputs.trip_metrics.grain != TRIP_METRICS_GRAIN:
        raise ValueError("Unsupported Gold trip-metrics grain.")

    if specification.outputs.trip_metrics.dimensions != TRIP_METRICS_DIMENSIONS:
        raise ValueError("Unsupported Gold trip-metrics dimensions.")

    if specification.outputs.daily_metrics.grain != DAILY_METRICS_GRAIN:
        raise ValueError("Unsupported Gold daily-metrics grain.")

    if specification.outputs.daily_metrics.dimensions:
        raise ValueError("Gold daily metrics must not define dimensions.")


def _metric_expressions() -> tuple[Column, ...]:
    """Return the business metric aggregations shared by Gold outputs."""
    trip_duration_minutes = (
        F.unix_timestamp("tpep_dropoff_datetime")
        - F.unix_timestamp("tpep_pickup_datetime")
    ) / F.lit(60.0)

    return (
        F.count(F.lit(1)).alias("trip_count"),
        F.sum((F.size(F.col("_quality_flags")) > 0).cast("long")).alias(
            "quality_flagged_trip_count"
        ),
        F.sum("trip_distance").alias("trip_distance_sum"),
        F.avg("trip_distance").alias("trip_distance_avg"),
        F.avg(trip_duration_minutes).alias("trip_duration_minutes_avg"),
        F.sum("fare_amount").alias("fare_amount_sum"),
        F.sum("tip_amount").alias("tip_amount_sum"),
        F.sum("total_amount").alias("total_amount_sum"),
        F.avg("total_amount").alias("total_amount_avg"),
    )


def _prepare_trip_dimensions(
    trips: DataFrame,
) -> DataFrame:
    """Derive Gold date and pickup-location dimensions."""
    return trips.withColumn(
        "pickup_date",
        F.to_date("tpep_pickup_datetime"),
    ).withColumnRenamed(
        "PULocationID",
        "pickup_location_id",
    )


def _prepare_pickup_zones(
    zones: DataFrame,
) -> DataFrame:
    """Project the taxi-zone lookup as pickup dimensions."""
    return zones.select(
        F.col("LocationID").alias("pickup_location_id"),
        F.col("Borough").alias("pickup_borough"),
        F.col("Zone").alias("pickup_zone"),
        F.col("service_zone").alias("pickup_service_zone"),
    )


def build_gold_analytics_frames(
    trips: DataFrame,
    zones: DataFrame,
    specification: GoldAnalyticsSpecification,
) -> GoldAnalyticsFrames:
    """Build Gold trip-level aggregates and daily business metrics."""
    _validate_specification(specification)
    _require_columns(
        trips,
        _REQUIRED_TRIP_COLUMNS,
        "Silver accepted trips",
    )
    _require_columns(
        zones,
        _REQUIRED_ZONE_COLUMNS,
        "Taxi-zone lookup",
    )

    prepared_trips = _prepare_trip_dimensions(trips)

    trip_metrics = (
        prepared_trips.join(
            _prepare_pickup_zones(zones),
            on="pickup_location_id",
            how="left",
        )
        .groupBy(
            *TRIP_METRICS_GRAIN,
            *TRIP_METRICS_DIMENSIONS,
        )
        .agg(*_metric_expressions())
    )

    daily_metrics = prepared_trips.groupBy(*DAILY_METRICS_GRAIN).agg(
        *_metric_expressions()
    )

    return GoldAnalyticsFrames(
        trip_metrics=trip_metrics,
        daily_metrics=daily_metrics,
    )
