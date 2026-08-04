"""Orchestration utilities for deterministic sample generation."""

from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import SparkSession

from taxi_lakehouse.data_acquisition import (
    SourceFile,
    validate_local_source_file,
)
from taxi_lakehouse.sample_specification import (
    SampleSpecification,
)


@dataclass(frozen=True, slots=True)
class ResolvedMonthlyTripSource:
    """One validated monthly trip source."""

    source_month: str
    source_file: SourceFile
    file_path: Path


@dataclass(frozen=True, slots=True)
class ResolvedSampleSources:
    """Validated local sources required for sample generation."""

    monthly_trip_sources: tuple[ResolvedMonthlyTripSource, ...]
    taxi_zone_source: SourceFile
    taxi_zone_path: Path


def resolve_sample_sources(
    specification: SampleSpecification,
    source_files: tuple[SourceFile, ...],
    landing_directory: Path,
) -> ResolvedSampleSources:
    """Resolve and validate every local source required by the sample."""
    trip_sources_by_month: dict[str, SourceFile] = {}
    taxi_zone_sources: list[SourceFile] = []

    for source_file in source_files:
        if source_file.kind == "yellow_taxi_trip_data":
            if source_file.source_month is None:
                raise ValueError("Yellow Taxi source must declare a source month.")

            if source_file.source_month in trip_sources_by_month:
                raise ValueError(
                    f"Duplicate Yellow Taxi source month: {source_file.source_month!r}."
                )

            trip_sources_by_month[source_file.source_month] = source_file

        elif source_file.kind == "taxi_zone_lookup":
            taxi_zone_sources.append(source_file)

    expected_months = set(specification.source_months)
    actual_months = set(trip_sources_by_month)

    missing_months = tuple(sorted(expected_months - actual_months))
    unexpected_months = tuple(sorted(actual_months - expected_months))

    if missing_months:
        raise ValueError(f"Missing Yellow Taxi source months: {missing_months!r}.")

    if unexpected_months:
        raise ValueError(
            f"Unexpected Yellow Taxi source months: {unexpected_months!r}."
        )

    if len(taxi_zone_sources) != 1:
        raise ValueError(
            "Source manifest must contain exactly one taxi_zone_lookup file."
        )

    taxi_zone_source = taxi_zone_sources[0]

    if taxi_zone_source.source_month is not None:
        raise ValueError("Taxi zone lookup source_month must be null.")

    monthly_trip_sources = []

    for source_month in specification.source_months:
        source_file = trip_sources_by_month[source_month]
        file_path = landing_directory / source_file.filename

        if not validate_local_source_file(
            source_file,
            file_path,
        ):
            raise ValueError(f"Local source file does not match manifest: {file_path}.")

        monthly_trip_sources.append(
            ResolvedMonthlyTripSource(
                source_month=source_month,
                source_file=source_file,
                file_path=file_path,
            )
        )

    taxi_zone_path = landing_directory / taxi_zone_source.filename

    if not validate_local_source_file(
        taxi_zone_source,
        taxi_zone_path,
    ):
        raise ValueError(
            f"Local source file does not match manifest: {taxi_zone_path}."
        )

    return ResolvedSampleSources(
        monthly_trip_sources=tuple(monthly_trip_sources),
        taxi_zone_source=taxi_zone_source,
        taxi_zone_path=taxi_zone_path,
    )


def load_taxi_zone_ids(
    spark: SparkSession,
    taxi_zone_path: Path,
    expected_row_count: int,
) -> tuple[int, ...]:
    """Load and validate the taxi-zone LocationID domain."""
    if (
        not isinstance(expected_row_count, int)
        or isinstance(expected_row_count, bool)
        or expected_row_count <= 0
    ):
        raise ValueError("expected_row_count must be a positive integer.")

    if not taxi_zone_path.is_file():
        raise FileNotFoundError(
            f"Taxi zone lookup file does not exist: {taxi_zone_path}."
        )

    zones = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(taxi_zone_path.as_posix())
    )

    if "LocationID" not in zones.columns:
        raise ValueError("Taxi zone lookup must contain LocationID.")

    raw_location_ids = tuple(
        row["LocationID"] for row in zones.select("LocationID").collect()
    )

    if len(raw_location_ids) != expected_row_count:
        raise ValueError(
            "Taxi zone row count does not match specification: "
            f"expected={expected_row_count}, "
            f"actual={len(raw_location_ids)}."
        )

    if any(
        not isinstance(location_id, int)
        or isinstance(location_id, bool)
        or location_id <= 0
        for location_id in raw_location_ids
    ):
        raise ValueError("Taxi zone LocationID values must be positive integers.")

    if len(raw_location_ids) != len(set(raw_location_ids)):
        raise ValueError("Taxi zone LocationID values must be unique.")

    return tuple(sorted(raw_location_ids))
