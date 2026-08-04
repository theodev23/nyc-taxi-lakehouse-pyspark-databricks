"""Orchestration utilities for deterministic sample generation."""

from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import SparkSession

from taxi_lakehouse.data_acquisition import (
    SourceFile,
    validate_local_source_file,
)
from taxi_lakehouse.sample_artifacts import (
    GeneratedSampleFile,
    write_single_parquet_file,
)
from taxi_lakehouse.sample_generation import (
    calculate_logical_sample_sha256,
    select_deterministic_sample_rows,
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


@dataclass(frozen=True, slots=True)
class GeneratedMonthlySample:
    """Metadata for one generated monthly trip sample."""

    source_month: str
    source_file: SourceFile
    source_path: Path
    source_column_count: int
    selected_row_count: int
    logical_sha256: str
    output_file: GeneratedSampleFile


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


def generate_monthly_sample(
    spark: SparkSession,
    monthly_source: ResolvedMonthlyTripSource,
    specification: SampleSpecification,
    zone_ids: tuple[int, ...],
    temporary_root: Path,
) -> GeneratedMonthlySample:
    """Generate and validate one deterministic monthly trip sample."""
    if monthly_source.source_month not in specification.source_months:
        raise ValueError(
            "Monthly source is absent from the sample specification: "
            f"{monthly_source.source_month!r}."
        )

    source_frame = spark.read.parquet(monthly_source.file_path.as_posix())
    source_columns = tuple(source_frame.columns)
    source_column_count = len(source_columns)

    expected_column_count = specification.source_profile.trip_column_count

    if source_column_count != expected_column_count:
        raise ValueError(
            "Source column count does not match specification: "
            f"expected={expected_column_count}, "
            f"actual={source_column_count}, "
            f"month={monthly_source.source_month!r}."
        )

    quota_by_bucket = {
        quota.quality_bucket: quota.rows_per_month
        for quota in (specification.real_sample_quotas_per_month)
    }

    selected_frame = select_deterministic_sample_rows(
        source_frame,
        monthly_source.source_month,
        source_columns,
        zone_ids,
        specification.quality_bucket_priority,
        quota_by_bucket,
    ).persist()

    try:
        if selected_frame.schema != source_frame.schema:
            raise RuntimeError("Selected sample schema does not match source schema.")

        selected_row_count = selected_frame.count()

        if selected_row_count != specification.expected_rows_per_month:
            raise ValueError(
                "Selected row count does not match monthly target: "
                f"expected={specification.expected_rows_per_month}, "
                f"actual={selected_row_count}, "
                f"month={monthly_source.source_month!r}."
            )

        logical_sha256 = calculate_logical_sample_sha256(
            selected_frame,
            monthly_source.source_month,
            source_columns,
        )

        output_path = Path(
            specification.output.trip_file_pattern.format(
                source_month=monthly_source.source_month
            )
        )

        output_file = write_single_parquet_file(
            selected_frame,
            output_path,
            temporary_root,
        )
    finally:
        selected_frame.unpersist()

    return GeneratedMonthlySample(
        source_month=monthly_source.source_month,
        source_file=monthly_source.source_file,
        source_path=monthly_source.file_path,
        source_column_count=source_column_count,
        selected_row_count=selected_row_count,
        logical_sha256=logical_sha256,
        output_file=output_file,
    )
