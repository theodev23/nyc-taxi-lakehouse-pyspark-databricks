"""Gold orchestration for Databricks Unity Catalog execution."""

from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import SparkSession

from taxi_lakehouse.databricks_configuration import (
    DatabricksPipelineConfiguration,
)
from taxi_lakehouse.databricks_gold_io import (
    load_managed_silver_accepted_month,
    load_managed_silver_source_months,
    load_managed_taxi_zones,
    write_managed_gold_month,
)
from taxi_lakehouse.gold_analytics_specification import (
    GoldAnalyticsSpecification,
    load_gold_analytics_specification,
)
from taxi_lakehouse.gold_orchestration import (
    _validate_taxi_zone_reference,
)
from taxi_lakehouse.gold_transformation import (
    GoldAnalyticsFrames,
    build_gold_analytics_frames,
)


@dataclass(frozen=True, slots=True)
class DatabricksGoldMonthWriteResult:
    """Metadata for one Databricks Gold month."""

    source_month: str
    trip_metrics_destination_table: str
    daily_metrics_destination_table: str
    trip_metrics_row_count: int
    daily_metrics_row_count: int


@dataclass(frozen=True, slots=True)
class DatabricksGoldBuildResult:
    """Metadata for one complete Databricks Gold run."""

    specification: GoldAnalyticsSpecification
    configuration: DatabricksPipelineConfiguration
    source_months: tuple[str, ...]
    taxi_zone_row_count: int
    monthly_writes: tuple[DatabricksGoldMonthWriteResult, ...]


def _materialize_and_write_month(
    analytics_frames: GoldAnalyticsFrames,
    source_month: str,
    trip_metrics_table: str,
    daily_metrics_table: str,
) -> tuple[int, int]:
    """Count and write one Gold month without Spark caching."""
    trip_metrics_row_count = analytics_frames.trip_metrics.count()
    daily_metrics_row_count = analytics_frames.daily_metrics.count()

    write_managed_gold_month(
        analytics_frames.trip_metrics,
        trip_metrics_table,
        source_month,
    )
    write_managed_gold_month(
        analytics_frames.daily_metrics,
        daily_metrics_table,
        source_month,
    )

    return (
        trip_metrics_row_count,
        daily_metrics_row_count,
    )


def build_databricks_gold_dataset(
    spark: SparkSession,
    specification_path: Path,
    configuration: DatabricksPipelineConfiguration,
) -> DatabricksGoldBuildResult:
    """Build managed Gold tables from managed Silver accepted trips."""
    specification = load_gold_analytics_specification(specification_path)

    source_months = load_managed_silver_source_months(
        spark,
        configuration.silver_accepted_table,
    )

    zones = load_managed_taxi_zones(
        spark,
        configuration.bronze_taxi_zone_table,
    )
    taxi_zone_row_count = _validate_taxi_zone_reference(zones)

    monthly_writes: list[DatabricksGoldMonthWriteResult] = []

    for source_month in source_months:
        silver_frame = load_managed_silver_accepted_month(
            spark,
            configuration.silver_accepted_table,
            source_month,
        )

        analytics_frames = build_gold_analytics_frames(
            silver_frame,
            zones,
            specification,
        )

        (
            trip_metrics_row_count,
            daily_metrics_row_count,
        ) = _materialize_and_write_month(
            analytics_frames,
            source_month,
            configuration.gold_trip_metrics_table,
            configuration.gold_daily_metrics_table,
        )

        monthly_writes.append(
            DatabricksGoldMonthWriteResult(
                source_month=source_month,
                trip_metrics_destination_table=(configuration.gold_trip_metrics_table),
                daily_metrics_destination_table=(
                    configuration.gold_daily_metrics_table
                ),
                trip_metrics_row_count=trip_metrics_row_count,
                daily_metrics_row_count=daily_metrics_row_count,
            )
        )

    return DatabricksGoldBuildResult(
        specification=specification,
        configuration=configuration,
        source_months=source_months,
        taxi_zone_row_count=taxi_zone_row_count,
        monthly_writes=tuple(monthly_writes),
    )
