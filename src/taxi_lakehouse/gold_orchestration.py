"""End-to-end orchestration for Gold analytical processing."""

from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from taxi_lakehouse.gold_analytics_specification import (
    GoldAnalyticsSpecification,
    load_gold_analytics_specification,
)
from taxi_lakehouse.gold_loading import (
    load_silver_accepted_month,
    load_silver_source_months,
    load_taxi_zones,
)
from taxi_lakehouse.gold_transformation import (
    GoldAnalyticsFrames,
    build_gold_analytics_frames,
)
from taxi_lakehouse.gold_writing import write_gold_month


@dataclass(frozen=True, slots=True)
class GoldMonthWriteResult:
    """Metadata describing one monthly Gold processing result."""

    source_month: str
    trip_metrics_destination_path: Path
    daily_metrics_destination_path: Path
    trip_metrics_row_count: int
    daily_metrics_row_count: int


@dataclass(frozen=True, slots=True)
class GoldBuildResult:
    """Metadata describing one complete Gold processing run."""

    specification: GoldAnalyticsSpecification
    source_months: tuple[str, ...]
    taxi_zone_row_count: int
    monthly_writes: tuple[GoldMonthWriteResult, ...]


def _validate_taxi_zone_reference(
    zones: DataFrame,
) -> int:
    """Validate the taxi-zone join domain and return its row count."""
    statistics = zones.agg(
        F.count(F.lit(1)).alias("row_count"),
        F.count("LocationID").alias("nonnull_location_id_count"),
        F.countDistinct("LocationID").alias("distinct_location_id_count"),
    ).first()

    row_count = int(statistics["row_count"])
    nonnull_location_id_count = int(statistics["nonnull_location_id_count"])
    distinct_location_id_count = int(statistics["distinct_location_id_count"])

    if row_count == 0:
        raise ValueError("Gold taxi-zone reference must contain at least one row.")

    if nonnull_location_id_count != row_count:
        raise ValueError(
            "Gold taxi-zone reference must not contain null LocationID values."
        )

    if distinct_location_id_count != row_count:
        raise ValueError(
            "Gold taxi-zone reference must contain unique LocationID values."
        )

    return row_count


def _materialize_and_write_month(
    analytics_frames: GoldAnalyticsFrames,
    specification: GoldAnalyticsSpecification,
    source_month: str,
    *,
    use_cache: bool = True,
) -> tuple[int, int]:
    """Count and write one month of Gold aggregates with optional caching."""
    trip_metrics = (
        analytics_frames.trip_metrics.cache()
        if use_cache
        else analytics_frames.trip_metrics
    )
    daily_metrics = (
        analytics_frames.daily_metrics.cache()
        if use_cache
        else analytics_frames.daily_metrics
    )

    try:
        trip_metrics_row_count = trip_metrics.count()
        daily_metrics_row_count = daily_metrics.count()

        write_gold_month(
            trip_metrics,
            specification.outputs.trip_metrics.table,
            source_month,
        )
        write_gold_month(
            daily_metrics,
            specification.outputs.daily_metrics.table,
            source_month,
        )
    finally:
        if use_cache:
            daily_metrics.unpersist(blocking=True)
            trip_metrics.unpersist(blocking=True)

    return (
        trip_metrics_row_count,
        daily_metrics_row_count,
    )


def build_gold_dataset(
    spark: SparkSession,
    specification_path: Path,
    *,
    use_cache: bool = True,
) -> GoldBuildResult:
    """Build Gold analytical Delta tables from Silver accepted trips."""
    specification = load_gold_analytics_specification(specification_path)

    source_months = load_silver_source_months(
        spark,
        specification.source.accepted_trip_table,
    )

    zones = load_taxi_zones(
        spark,
        specification.source.taxi_zone_table,
    )
    materialized_zones = zones.cache() if use_cache else zones

    try:
        taxi_zone_row_count = _validate_taxi_zone_reference(materialized_zones)

        monthly_writes: list[GoldMonthWriteResult] = []

        for source_month in source_months:
            silver_frame = load_silver_accepted_month(
                spark,
                specification.source.accepted_trip_table,
                source_month,
            )

            analytics_frames = build_gold_analytics_frames(
                silver_frame,
                materialized_zones,
                specification,
            )

            (
                trip_metrics_row_count,
                daily_metrics_row_count,
            ) = _materialize_and_write_month(
                analytics_frames,
                specification,
                source_month,
                use_cache=use_cache,
            )

            monthly_writes.append(
                GoldMonthWriteResult(
                    source_month=source_month,
                    trip_metrics_destination_path=(
                        specification.outputs.trip_metrics.table
                    ),
                    daily_metrics_destination_path=(
                        specification.outputs.daily_metrics.table
                    ),
                    trip_metrics_row_count=trip_metrics_row_count,
                    daily_metrics_row_count=daily_metrics_row_count,
                )
            )
    finally:
        if use_cache:
            materialized_zones.unpersist(blocking=True)

    return GoldBuildResult(
        specification=specification,
        source_months=source_months,
        taxi_zone_row_count=taxi_zone_row_count,
        monthly_writes=tuple(monthly_writes),
    )
