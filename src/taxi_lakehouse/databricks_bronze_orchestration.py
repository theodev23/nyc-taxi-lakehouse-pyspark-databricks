"""Bronze orchestration for Databricks Unity Catalog execution."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from taxi_lakehouse.bronze_loading import (
    format_utc_timestamp_ntz,
    load_bronze_taxi_zone_source,
    load_bronze_trip_source,
)
from taxi_lakehouse.bronze_sources import (
    ResolvedBronzeSources,
    resolve_bronze_sources,
)
from taxi_lakehouse.data_acquisition import load_source_files
from taxi_lakehouse.databricks_bronze_io import (
    write_managed_bronze_taxi_zones,
    write_managed_bronze_trip_month,
)
from taxi_lakehouse.databricks_configuration import (
    DatabricksPipelineConfiguration,
)


@dataclass(frozen=True, slots=True)
class DatabricksBronzeTripWriteResult:
    """Metadata for one Databricks Bronze trip month."""

    source_month: str
    source_path: Path
    destination_table: str
    row_count: int


@dataclass(frozen=True, slots=True)
class DatabricksBronzeIngestionResult:
    """Metadata for one complete Databricks Bronze ingestion."""

    ingested_at_utc: datetime
    resolved_sources: ResolvedBronzeSources
    configuration: DatabricksPipelineConfiguration
    monthly_trip_writes: tuple[DatabricksBronzeTripWriteResult, ...]
    taxi_zone_destination_table: str
    taxi_zone_row_count: int


def _materialize_and_write(
    frame: DataFrame,
    write_operation: Callable[[DataFrame], None],
) -> int:
    """Count and write one Bronze frame without Spark caching."""
    row_count = frame.count()
    write_operation(frame)
    return row_count


def ingest_databricks_bronze_dataset(
    spark: SparkSession,
    manifest_path: Path,
    configuration: DatabricksPipelineConfiguration,
    ingested_at_utc: datetime,
) -> DatabricksBronzeIngestionResult:
    """Load validated Volume sources into managed Bronze tables."""
    format_utc_timestamp_ntz(ingested_at_utc)

    source_files = load_source_files(manifest_path)
    resolved_sources = resolve_bronze_sources(
        source_files,
        configuration.landing_directory,
    )

    monthly_trip_writes: list[DatabricksBronzeTripWriteResult] = []

    for monthly_source in resolved_sources.monthly_trip_sources:
        trip_frame = load_bronze_trip_source(
            spark,
            monthly_source,
            ingested_at_utc,
        )

        def write_trip_frame(
            frame: DataFrame,
            source_month: str = monthly_source.source_month,
        ) -> None:
            write_managed_bronze_trip_month(
                frame,
                configuration.bronze_trip_table,
                source_month,
            )

        row_count = _materialize_and_write(
            trip_frame,
            write_trip_frame,
        )

        monthly_trip_writes.append(
            DatabricksBronzeTripWriteResult(
                source_month=monthly_source.source_month,
                source_path=monthly_source.file_path,
                destination_table=configuration.bronze_trip_table,
                row_count=row_count,
            )
        )

    taxi_zone_frame = load_bronze_taxi_zone_source(
        spark,
        resolved_sources.taxi_zone_source,
        resolved_sources.taxi_zone_path,
        ingested_at_utc,
    )

    def write_taxi_zone_frame(
        frame: DataFrame,
    ) -> None:
        write_managed_bronze_taxi_zones(
            frame,
            configuration.bronze_taxi_zone_table,
        )

    taxi_zone_row_count = _materialize_and_write(
        taxi_zone_frame,
        write_taxi_zone_frame,
    )

    return DatabricksBronzeIngestionResult(
        ingested_at_utc=ingested_at_utc,
        resolved_sources=resolved_sources,
        configuration=configuration,
        monthly_trip_writes=tuple(monthly_trip_writes),
        taxi_zone_destination_table=configuration.bronze_taxi_zone_table,
        taxi_zone_row_count=taxi_zone_row_count,
    )
