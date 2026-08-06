"""End-to-end orchestration for Bronze Delta ingestion."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
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
from taxi_lakehouse.bronze_writing import (
    write_bronze_taxi_zones,
    write_bronze_trip_month,
)
from taxi_lakehouse.data_acquisition import load_source_files

BRONZE_TRIP_DIRECTORY_NAME = "yellow_taxi_trips"
BRONZE_TAXI_ZONE_DIRECTORY_NAME = "taxi_zones"


@dataclass(frozen=True, slots=True)
class BronzeTripWriteResult:
    """Metadata describing one monthly Bronze trip write."""

    source_month: str
    source_path: Path
    destination_path: Path
    row_count: int


@dataclass(frozen=True, slots=True)
class BronzeIngestionResult:
    """Metadata describing one complete Bronze ingestion run."""

    ingested_at_utc: datetime
    resolved_sources: ResolvedBronzeSources
    monthly_trip_writes: tuple[BronzeTripWriteResult, ...]
    taxi_zone_destination_path: Path
    taxi_zone_row_count: int


def _materialize_and_write(
    frame: DataFrame,
    write_operation: Callable[[DataFrame], None],
) -> int:
    """Cache, count, write, and release one source DataFrame."""
    cached_frame = frame.cache()

    try:
        row_count = cached_frame.count()
        write_operation(cached_frame)
    finally:
        cached_frame.unpersist(blocking=True)

    return row_count


def ingest_bronze_dataset(
    spark: SparkSession,
    manifest_path: Path,
    landing_directory: Path,
    bronze_root: Path,
    ingested_at_utc: datetime,
) -> BronzeIngestionResult:
    """Load all validated sources into the Bronze Delta layer."""
    format_utc_timestamp_ntz(ingested_at_utc)

    source_files = load_source_files(manifest_path)
    resolved_sources = resolve_bronze_sources(
        source_files,
        landing_directory,
    )

    trip_destination_path = bronze_root / BRONZE_TRIP_DIRECTORY_NAME
    taxi_zone_destination_path = bronze_root / BRONZE_TAXI_ZONE_DIRECTORY_NAME

    monthly_trip_writes: list[BronzeTripWriteResult] = []

    for monthly_source in resolved_sources.monthly_trip_sources:
        trip_frame = load_bronze_trip_source(
            spark,
            monthly_source,
            ingested_at_utc,
        )

        write_trip_frame = partial(
            write_bronze_trip_month,
            destination_path=trip_destination_path,
            source_month=monthly_source.source_month,
        )

        row_count = _materialize_and_write(
            trip_frame,
            write_trip_frame,
        )

        monthly_trip_writes.append(
            BronzeTripWriteResult(
                source_month=monthly_source.source_month,
                source_path=monthly_source.file_path,
                destination_path=trip_destination_path,
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
        cached_frame: DataFrame,
    ) -> None:
        write_bronze_taxi_zones(
            cached_frame,
            taxi_zone_destination_path,
        )

    taxi_zone_row_count = _materialize_and_write(
        taxi_zone_frame,
        write_taxi_zone_frame,
    )

    return BronzeIngestionResult(
        ingested_at_utc=ingested_at_utc,
        resolved_sources=resolved_sources,
        monthly_trip_writes=tuple(monthly_trip_writes),
        taxi_zone_destination_path=taxi_zone_destination_path,
        taxi_zone_row_count=taxi_zone_row_count,
    )
