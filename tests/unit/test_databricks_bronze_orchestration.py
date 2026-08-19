"""Tests for Databricks Bronze orchestration."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import taxi_lakehouse.databricks_bronze_orchestration as orchestration
from taxi_lakehouse.bronze_sources import (
    ResolvedBronzeSources,
    ResolvedBronzeTripSource,
)
from taxi_lakehouse.data_acquisition import load_source_files
from taxi_lakehouse.databricks_configuration import (
    DatabricksPipelineConfiguration,
)

PROJECT_MANIFEST_PATH = Path("data/source_manifest.json")


class FakeFrame:
    """Synthetic frame exposing only row counting."""

    def __init__(
        self,
        label: str,
        row_count: int,
    ) -> None:
        self.label = label
        self.row_count = row_count
        self.count_count = 0

    def count(self) -> int:
        """Return the configured row count."""
        self.count_count += 1
        return self.row_count


def build_resolved_sources(
    landing_directory: Path,
) -> tuple[tuple[Any, ...], ResolvedBronzeSources]:
    """Build two trip sources and one taxi-zone source."""
    source_files = load_source_files(PROJECT_MANIFEST_PATH)

    trip_sources_by_month = {
        source_file.source_month: source_file
        for source_file in source_files
        if (
            source_file.kind == "yellow_taxi_trip_data"
            and source_file.source_month is not None
        )
    }

    monthly_trip_sources = tuple(
        ResolvedBronzeTripSource(
            source_month=source_month,
            source_file=trip_sources_by_month[source_month],
            file_path=(
                landing_directory / trip_sources_by_month[source_month].filename
            ),
        )
        for source_month in (
            "2024-01",
            "2024-02",
        )
    )

    taxi_zone_source = next(
        source_file
        for source_file in source_files
        if source_file.kind == "taxi_zone_lookup"
    )

    return (
        source_files,
        ResolvedBronzeSources(
            monthly_trip_sources=monthly_trip_sources,
            taxi_zone_source=taxi_zone_source,
            taxi_zone_path=(landing_directory / taxi_zone_source.filename),
        ),
    )


def test_materialize_and_write_is_cache_free() -> None:
    """Databricks Bronze materialization should not require caching."""
    frame = FakeFrame(
        "trip",
        123,
    )
    written_frames: list[FakeFrame] = []

    result = orchestration._materialize_and_write(
        frame,  # type: ignore[arg-type]
        written_frames.append,  # type: ignore[arg-type]
    )

    assert result == 123
    assert frame.count_count == 1
    assert written_frames == [frame]


def test_ingest_databricks_bronze_dataset_orchestrates_all_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Databricks Bronze should ingest every resolved Volume source."""
    configuration = DatabricksPipelineConfiguration(
        catalog="workspace",
        schema="nyc_taxi",
        volume="source_files",
    )
    ingested_at_utc = datetime(
        2026,
        8,
        19,
        12,
        30,
        tzinfo=UTC,
    )
    source_files, resolved_sources = build_resolved_sources(
        configuration.landing_directory
    )

    fake_spark = object()
    events: list[str] = []

    trip_frames = {
        "2024-01": FakeFrame("2024-01", 101),
        "2024-02": FakeFrame("2024-02", 202),
    }
    taxi_zone_frame = FakeFrame("taxi_zones", 265)

    monkeypatch.setattr(
        orchestration,
        "format_utc_timestamp_ntz",
        lambda value: (
            events.append("validate_timestamp") or "2026-08-19 12:30:00.000000"
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "load_source_files",
        lambda path: events.append("load_source_files") or source_files,
    )

    def fake_resolve_sources(
        received_source_files: tuple[Any, ...],
        landing_directory: Path,
    ) -> ResolvedBronzeSources:
        events.append("resolve_sources")
        assert received_source_files is source_files
        assert landing_directory == configuration.landing_directory
        return resolved_sources

    monkeypatch.setattr(
        orchestration,
        "resolve_bronze_sources",
        fake_resolve_sources,
    )

    def fake_load_trip(
        spark: object,
        source: ResolvedBronzeTripSource,
        timestamp: datetime,
    ) -> FakeFrame:
        events.append(f"load_trip:{source.source_month}")
        assert spark is fake_spark
        assert timestamp == ingested_at_utc
        return trip_frames[source.source_month]

    monkeypatch.setattr(
        orchestration,
        "load_bronze_trip_source",
        fake_load_trip,
    )

    def fake_write_trip(
        frame: FakeFrame,
        table_name: str,
        source_month: str,
    ) -> None:
        events.append(f"write_trip:{source_month}")
        assert frame is trip_frames[source_month]
        assert table_name == configuration.bronze_trip_table

    monkeypatch.setattr(
        orchestration,
        "write_managed_bronze_trip_month",
        fake_write_trip,
    )

    def fake_load_zones(
        spark: object,
        source_file: Any,
        file_path: Path,
        timestamp: datetime,
    ) -> FakeFrame:
        events.append("load_zones")
        assert spark is fake_spark
        assert source_file is resolved_sources.taxi_zone_source
        assert file_path == resolved_sources.taxi_zone_path
        assert timestamp == ingested_at_utc
        return taxi_zone_frame

    monkeypatch.setattr(
        orchestration,
        "load_bronze_taxi_zone_source",
        fake_load_zones,
    )

    def fake_write_zones(
        frame: FakeFrame,
        table_name: str,
    ) -> None:
        events.append("write_zones")
        assert frame is taxi_zone_frame
        assert table_name == configuration.bronze_taxi_zone_table

    monkeypatch.setattr(
        orchestration,
        "write_managed_bronze_taxi_zones",
        fake_write_zones,
    )

    result = orchestration.ingest_databricks_bronze_dataset(
        fake_spark,  # type: ignore[arg-type]
        PROJECT_MANIFEST_PATH,
        configuration,
        ingested_at_utc,
    )

    assert result.ingested_at_utc == ingested_at_utc
    assert result.resolved_sources is resolved_sources
    assert result.configuration is configuration
    assert result.taxi_zone_destination_table == (configuration.bronze_taxi_zone_table)
    assert result.taxi_zone_row_count == 265

    assert [
        (
            write.source_month,
            write.source_path,
            write.destination_table,
            write.row_count,
        )
        for write in result.monthly_trip_writes
    ] == [
        (
            "2024-01",
            resolved_sources.monthly_trip_sources[0].file_path,
            configuration.bronze_trip_table,
            101,
        ),
        (
            "2024-02",
            resolved_sources.monthly_trip_sources[1].file_path,
            configuration.bronze_trip_table,
            202,
        ),
    ]

    assert events == [
        "validate_timestamp",
        "load_source_files",
        "resolve_sources",
        "load_trip:2024-01",
        "write_trip:2024-01",
        "load_trip:2024-02",
        "write_trip:2024-02",
        "load_zones",
        "write_zones",
    ]

    assert trip_frames["2024-01"].count_count == 1
    assert trip_frames["2024-02"].count_count == 1
    assert taxi_zone_frame.count_count == 1
