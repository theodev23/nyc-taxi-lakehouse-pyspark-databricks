"""Tests for Bronze source resolution and validation."""

from dataclasses import replace
from pathlib import Path

import pytest

from taxi_lakehouse import bronze_sources
from taxi_lakehouse.data_acquisition import (
    SourceFile,
    load_source_files,
)

PROJECT_MANIFEST_PATH = Path("data/source_manifest.json")
EXPECTED_MONTHS = (
    "2024-01",
    "2024-02",
    "2024-03",
    "2024-04",
    "2024-05",
    "2024-06",
)


def load_project_sources() -> tuple[SourceFile, ...]:
    """Load the project source manifest records."""
    return load_source_files(PROJECT_MANIFEST_PATH)


def allow_local_source_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make local manifest validation succeed for resolver tests."""
    monkeypatch.setattr(
        bronze_sources,
        "validate_local_source_file",
        lambda source_file, file_path: True,
    )


def test_resolve_bronze_sources_orders_months_and_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resolved monthly sources should be chronological."""
    allow_local_source_files(monkeypatch)
    source_files = tuple(reversed(load_project_sources()))

    result = bronze_sources.resolve_bronze_sources(
        source_files,
        tmp_path,
    )

    assert (
        tuple(source.source_month for source in result.monthly_trip_sources)
        == EXPECTED_MONTHS
    )

    assert tuple(
        source.file_path.name for source in result.monthly_trip_sources
    ) == tuple(
        f"yellow_tripdata_{source_month}.parquet" for source_month in EXPECTED_MONTHS
    )

    assert result.taxi_zone_path == tmp_path / "taxi_zone_lookup.csv"


def test_resolve_bronze_sources_rejects_duplicate_month(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Two trip sources cannot represent the same month."""
    allow_local_source_files(monkeypatch)
    source_files = list(load_project_sources())
    source_files[1] = replace(
        source_files[1],
        source_month="2024-01",
    )

    with pytest.raises(
        ValueError,
        match="Duplicate Yellow Taxi source month",
    ):
        bronze_sources.resolve_bronze_sources(
            tuple(source_files),
            tmp_path,
        )


def test_resolve_bronze_sources_rejects_invalid_month(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Trip source months should use the YYYY-MM contract."""
    allow_local_source_files(monkeypatch)
    source_files = list(load_project_sources())
    source_files[0] = replace(
        source_files[0],
        source_month="2024-13",
    )

    with pytest.raises(
        ValueError,
        match="must use YYYY-MM format",
    ):
        bronze_sources.resolve_bronze_sources(
            tuple(source_files),
            tmp_path,
        )


def test_resolve_bronze_sources_rejects_unknown_kind(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unknown source kinds should fail explicitly."""
    allow_local_source_files(monkeypatch)
    source_files = list(load_project_sources())
    source_files[0] = replace(
        source_files[0],
        kind="unexpected_source",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported source kind",
    ):
        bronze_sources.resolve_bronze_sources(
            tuple(source_files),
            tmp_path,
        )


def test_resolve_bronze_sources_requires_trip_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """At least one Yellow Taxi trip file is required."""
    allow_local_source_files(monkeypatch)
    taxi_zone_source = load_project_sources()[-1]

    with pytest.raises(
        ValueError,
        match="at least one Yellow Taxi trip file",
    ):
        bronze_sources.resolve_bronze_sources(
            (taxi_zone_source,),
            tmp_path,
        )


def test_resolve_bronze_sources_requires_one_zone_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exactly one taxi-zone lookup file is required."""
    allow_local_source_files(monkeypatch)
    source_files = load_project_sources()[:-1]

    with pytest.raises(
        ValueError,
        match="exactly one taxi_zone_lookup",
    ):
        bronze_sources.resolve_bronze_sources(
            source_files,
            tmp_path,
        )


def test_resolve_bronze_sources_rejects_zone_month(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The taxi-zone lookup must not declare a source month."""
    allow_local_source_files(monkeypatch)
    source_files = list(load_project_sources())
    source_files[-1] = replace(
        source_files[-1],
        source_month="2024-01",
    )

    with pytest.raises(
        ValueError,
        match="source_month must be null",
    ):
        bronze_sources.resolve_bronze_sources(
            tuple(source_files),
            tmp_path,
        )


def test_resolve_bronze_sources_rejects_invalid_local_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A file that differs from its manifest should be rejected."""
    monkeypatch.setattr(
        bronze_sources,
        "validate_local_source_file",
        lambda source_file, file_path: False,
    )

    with pytest.raises(
        ValueError,
        match="does not match manifest",
    ):
        bronze_sources.resolve_bronze_sources(
            load_project_sources(),
            tmp_path,
        )
