"""Resolution and validation of local Bronze source files."""

import re
from dataclasses import dataclass
from pathlib import Path

from taxi_lakehouse.data_acquisition import (
    SourceFile,
    validate_local_source_file,
)

YELLOW_TAXI_SOURCE_KIND = "yellow_taxi_trip_data"
TAXI_ZONE_SOURCE_KIND = "taxi_zone_lookup"

_SOURCE_MONTH_PATTERN = re.compile(r"\d{4}-(0[1-9]|1[0-2])")


@dataclass(frozen=True, slots=True)
class ResolvedBronzeTripSource:
    """One validated monthly Yellow Taxi source."""

    source_month: str
    source_file: SourceFile
    file_path: Path


@dataclass(frozen=True, slots=True)
class ResolvedBronzeSources:
    """All validated local sources required by Bronze ingestion."""

    monthly_trip_sources: tuple[ResolvedBronzeTripSource, ...]
    taxi_zone_source: SourceFile
    taxi_zone_path: Path


def require_source_month(value: object) -> str:
    """Validate and return one source month in YYYY-MM format."""
    if not isinstance(value, str) or _SOURCE_MONTH_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"Yellow Taxi source month must use YYYY-MM format: {value!r}."
        )

    return value


def validate_source_path(
    source_file: SourceFile,
    landing_directory: Path,
) -> Path:
    """Resolve one local source path and validate its manifest metadata."""
    file_path = landing_directory / source_file.filename

    if not validate_local_source_file(source_file, file_path):
        raise ValueError(f"Local source file does not match manifest: {file_path}.")

    return file_path


def resolve_bronze_sources(
    source_files: tuple[SourceFile, ...],
    landing_directory: Path,
) -> ResolvedBronzeSources:
    """Resolve and validate every source required by Bronze ingestion."""
    trip_sources_by_month: dict[str, SourceFile] = {}
    taxi_zone_sources: list[SourceFile] = []

    for source_file in source_files:
        if source_file.kind == YELLOW_TAXI_SOURCE_KIND:
            source_month = require_source_month(source_file.source_month)

            if source_month in trip_sources_by_month:
                raise ValueError(
                    f"Duplicate Yellow Taxi source month: {source_month!r}."
                )

            trip_sources_by_month[source_month] = source_file

        elif source_file.kind == TAXI_ZONE_SOURCE_KIND:
            if source_file.source_month is not None:
                raise ValueError("Taxi zone lookup source_month must be null.")

            taxi_zone_sources.append(source_file)

        else:
            raise ValueError(f"Unsupported source kind: {source_file.kind!r}.")

    if not trip_sources_by_month:
        raise ValueError(
            "Source manifest must contain at least one Yellow Taxi trip file."
        )

    if len(taxi_zone_sources) != 1:
        raise ValueError(
            "Source manifest must contain exactly one taxi_zone_lookup file."
        )

    monthly_trip_sources = tuple(
        ResolvedBronzeTripSource(
            source_month=source_month,
            source_file=source_file,
            file_path=validate_source_path(
                source_file,
                landing_directory,
            ),
        )
        for source_month, source_file in sorted(trip_sources_by_month.items())
    )

    taxi_zone_source = taxi_zone_sources[0]
    taxi_zone_path = validate_source_path(
        taxi_zone_source,
        landing_directory,
    )

    return ResolvedBronzeSources(
        monthly_trip_sources=monthly_trip_sources,
        taxi_zone_source=taxi_zone_source,
        taxi_zone_path=taxi_zone_path,
    )
