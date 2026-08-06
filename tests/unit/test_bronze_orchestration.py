"""Tests for end-to-end Bronze ingestion orchestration."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import taxi_lakehouse.bronze_orchestration as orchestration
from taxi_lakehouse.bronze_sources import (
    ResolvedBronzeSources,
    ResolvedBronzeTripSource,
)
from taxi_lakehouse.data_acquisition import load_source_files

PROJECT_MANIFEST_PATH = Path("data/source_manifest.json")


class FakeFrame:
    """Record cache, count, and unpersist operations."""

    def __init__(
        self,
        label: str,
        row_count: int,
        events: list[str] | None = None,
    ) -> None:
        self.label = label
        self.row_count = row_count
        self.events = events
        self.cache_count = 0
        self.count_count = 0
        self.unpersist_count = 0

    def cache(self) -> "FakeFrame":
        """Record one cache request."""
        self.cache_count += 1

        if self.events is not None:
            self.events.append(f"cache:{self.label}")

        return self

    def count(self) -> int:
        """Return the configured row count."""
        self.count_count += 1

        if self.events is not None:
            self.events.append(f"count:{self.label}")

        return self.row_count

    def unpersist(self) -> "FakeFrame":
        """Record release of the cached frame."""
        self.unpersist_count += 1

        if self.events is not None:
            self.events.append(f"unpersist:{self.label}")

        return self


def build_resolved_sources(
    landing_directory: Path,
) -> tuple[tuple[Any, ...], ResolvedBronzeSources]:
    """Build two monthly sources and one taxi-zone source."""
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


def test_materialize_and_write_releases_cached_frame() -> None:
    """Successful writes should materialize and release the frame."""
    frame = FakeFrame(
        label="january",
        row_count=123,
    )
    written_frames: list[FakeFrame] = []

    result = orchestration._materialize_and_write(
        frame,
        written_frames.append,
    )

    assert result == 123
    assert written_frames == [frame]
    assert frame.cache_count == 1
    assert frame.count_count == 1
    assert frame.unpersist_count == 1


def test_materialize_and_write_releases_frame_after_failure() -> None:
    """A failed write should still release the cached frame."""
    frame = FakeFrame(
        label="january",
        row_count=123,
    )

    def fail_write(cached_frame: FakeFrame) -> None:
        assert cached_frame is frame
        raise RuntimeError("synthetic Delta write failure")

    with pytest.raises(
        RuntimeError,
        match="synthetic Delta write failure",
    ):
        orchestration._materialize_and_write(
            frame,
            fail_write,
        )

    assert frame.cache_count == 1
    assert frame.count_count == 1
    assert frame.unpersist_count == 1


def test_ingest_bronze_dataset_orchestrates_all_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The orchestrator should load, write, count, and report all sources."""
    manifest_path = tmp_path / "source_manifest.json"
    landing_directory = tmp_path / "landing"
    bronze_root = tmp_path / "lakehouse" / "bronze"
    ingested_at_utc = datetime(
        2026,
        8,
        6,
        8,
        30,
        tzinfo=UTC,
    )

    source_files, resolved_sources = build_resolved_sources(
        landing_directory,
    )

    events: list[str] = []

    trip_frames = {
        "2024-01": FakeFrame(
            "2024-01",
            101,
            events,
        ),
        "2024-02": FakeFrame(
            "2024-02",
            202,
            events,
        ),
    }
    taxi_zone_frame = FakeFrame(
        "taxi_zones",
        265,
        events,
    )

    def fake_format_utc_timestamp_ntz(
        value: datetime,
    ) -> str:
        events.append("validate_timestamp")
        assert value == ingested_at_utc
        return "2026-08-06 08:30:00.000000"

    def fake_load_source_files(
        path: Path,
    ) -> tuple[Any, ...]:
        events.append("load_source_files")
        assert path == manifest_path
        return source_files

    def fake_resolve_bronze_sources(
        received_source_files: tuple[Any, ...],
        received_landing_directory: Path,
    ) -> ResolvedBronzeSources:
        events.append("resolve_sources")
        assert received_source_files is source_files
        assert received_landing_directory == landing_directory
        return resolved_sources

    def fake_load_bronze_trip_source(
        spark: object,
        source: ResolvedBronzeTripSource,
        received_timestamp: datetime,
    ) -> FakeFrame:
        events.append(f"load_trip:{source.source_month}")
        assert spark is fake_spark
        assert received_timestamp == ingested_at_utc
        return trip_frames[source.source_month]

    def fake_write_bronze_trip_month(
        frame: FakeFrame,
        destination_path: Path,
        source_month: str,
    ) -> None:
        events.append(f"write_trip:{source_month}")
        assert frame is trip_frames[source_month]
        assert destination_path == (bronze_root / "yellow_taxi_trips")

    def fake_load_bronze_taxi_zone_source(
        spark: object,
        source_file: Any,
        file_path: Path,
        received_timestamp: datetime,
    ) -> FakeFrame:
        events.append("load_taxi_zones")
        assert spark is fake_spark
        assert source_file is resolved_sources.taxi_zone_source
        assert file_path == resolved_sources.taxi_zone_path
        assert received_timestamp == ingested_at_utc
        return taxi_zone_frame

    def fake_write_bronze_taxi_zones(
        frame: FakeFrame,
        destination_path: Path,
    ) -> None:
        events.append("write_taxi_zones")
        assert frame is taxi_zone_frame
        assert destination_path == (bronze_root / "taxi_zones")

    fake_spark = object()

    monkeypatch.setattr(
        orchestration,
        "format_utc_timestamp_ntz",
        fake_format_utc_timestamp_ntz,
    )
    monkeypatch.setattr(
        orchestration,
        "load_source_files",
        fake_load_source_files,
    )
    monkeypatch.setattr(
        orchestration,
        "resolve_bronze_sources",
        fake_resolve_bronze_sources,
    )
    monkeypatch.setattr(
        orchestration,
        "load_bronze_trip_source",
        fake_load_bronze_trip_source,
    )
    monkeypatch.setattr(
        orchestration,
        "write_bronze_trip_month",
        fake_write_bronze_trip_month,
    )
    monkeypatch.setattr(
        orchestration,
        "load_bronze_taxi_zone_source",
        fake_load_bronze_taxi_zone_source,
    )
    monkeypatch.setattr(
        orchestration,
        "write_bronze_taxi_zones",
        fake_write_bronze_taxi_zones,
    )

    result = orchestration.ingest_bronze_dataset(
        fake_spark,
        manifest_path,
        landing_directory,
        bronze_root,
        ingested_at_utc,
    )

    assert result.ingested_at_utc == ingested_at_utc
    assert result.resolved_sources is resolved_sources
    assert tuple(
        (
            write.source_month,
            write.row_count,
            write.destination_path,
        )
        for write in result.monthly_trip_writes
    ) == (
        (
            "2024-01",
            101,
            bronze_root / "yellow_taxi_trips",
        ),
        (
            "2024-02",
            202,
            bronze_root / "yellow_taxi_trips",
        ),
    )
    assert result.taxi_zone_destination_path == (bronze_root / "taxi_zones")
    assert result.taxi_zone_row_count == 265

    assert events == [
        "validate_timestamp",
        "load_source_files",
        "resolve_sources",
        "load_trip:2024-01",
        "cache:2024-01",
        "count:2024-01",
        "write_trip:2024-01",
        "unpersist:2024-01",
        "load_trip:2024-02",
        "cache:2024-02",
        "count:2024-02",
        "write_trip:2024-02",
        "unpersist:2024-02",
        "load_taxi_zones",
        "cache:taxi_zones",
        "count:taxi_zones",
        "write_taxi_zones",
        "unpersist:taxi_zones",
    ]

    for frame in (
        *trip_frames.values(),
        taxi_zone_frame,
    ):
        assert frame.cache_count == 1
        assert frame.count_count == 1
        assert frame.unpersist_count == 1
