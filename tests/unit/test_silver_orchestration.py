"""Tests for end-to-end Silver data-quality orchestration."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import taxi_lakehouse.silver_orchestration as orchestration
from taxi_lakehouse.silver_quality_specification import (
    load_silver_quality_specification,
)

PROJECT_SPECIFICATION_PATH = Path("data/silver_quality_spec.json")


class FakeFrame:
    """Record cache, count, and unpersist operations."""

    def __init__(
        self,
        label: str,
        row_count: int = 0,
        events: list[str] | None = None,
    ) -> None:
        self.label = label
        self.row_count = row_count
        self.events = events
        self.cache_count = 0
        self.count_count = 0
        self.unpersist_count = 0
        self.unpersist_blocking_values: list[bool] = []

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

    def unpersist(
        self,
        blocking: bool = False,
    ) -> "FakeFrame":
        """Record release of the cached frame."""
        self.unpersist_count += 1
        self.unpersist_blocking_values.append(blocking)

        if self.events is not None:
            self.events.append(f"unpersist:{self.label}")

        return self


def test_materialize_split_and_write_releases_cached_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful Silver writes should release the annotated cache."""
    specification = load_silver_quality_specification(PROJECT_SPECIFICATION_PATH)
    events: list[str] = []

    annotated_frame = FakeFrame(
        "annotated",
        events=events,
    )
    accepted_frame = FakeFrame(
        "accepted",
        row_count=101,
        events=events,
    )
    rejected_frame = FakeFrame(
        "rejected",
        row_count=2,
        events=events,
    )

    def fake_split(
        frame: FakeFrame,
        received_specification: Any,
    ) -> SimpleNamespace:
        events.append("split")
        assert frame is annotated_frame
        assert received_specification is specification

        return SimpleNamespace(
            accepted=accepted_frame,
            rejected=rejected_frame,
        )

    def fake_write(
        frame: FakeFrame,
        destination_path: Path,
        source_month: str,
    ) -> None:
        events.append(f"write:{frame.label}:{source_month}")

        if frame is accepted_frame:
            assert destination_path == (specification.output.accepted_table)
        else:
            assert frame is rejected_frame
            assert destination_path == (specification.output.rejected_table)

    monkeypatch.setattr(
        orchestration,
        "split_silver_quality_rows",
        fake_split,
    )
    monkeypatch.setattr(
        orchestration,
        "write_silver_trip_month",
        fake_write,
    )

    result = orchestration._materialize_split_and_write(
        annotated_frame,
        specification,
        "2024-01",
    )

    assert result == (101, 2)
    assert annotated_frame.cache_count == 1
    assert annotated_frame.unpersist_count == 1
    assert annotated_frame.unpersist_blocking_values == [True]
    assert accepted_frame.count_count == 1
    assert rejected_frame.count_count == 1
    assert events == [
        "cache:annotated",
        "split",
        "count:accepted",
        "count:rejected",
        "write:accepted:2024-01",
        "write:rejected:2024-01",
        "unpersist:annotated",
    ]


def test_materialize_split_and_write_releases_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed Silver write should still release the annotated cache."""
    specification = load_silver_quality_specification(PROJECT_SPECIFICATION_PATH)

    annotated_frame = FakeFrame("annotated")
    accepted_frame = FakeFrame(
        "accepted",
        row_count=101,
    )
    rejected_frame = FakeFrame(
        "rejected",
        row_count=2,
    )

    monkeypatch.setattr(
        orchestration,
        "split_silver_quality_rows",
        lambda frame, received_specification: SimpleNamespace(
            accepted=accepted_frame,
            rejected=rejected_frame,
        ),
    )

    write_count = 0

    def fail_second_write(
        frame: FakeFrame,
        destination_path: Path,
        source_month: str,
    ) -> None:
        nonlocal write_count
        write_count += 1

        if write_count == 2:
            raise RuntimeError("synthetic Silver Delta write failure")

    monkeypatch.setattr(
        orchestration,
        "write_silver_trip_month",
        fail_second_write,
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic Silver Delta write failure",
    ):
        orchestration._materialize_split_and_write(
            annotated_frame,
            specification,
            "2024-01",
        )

    assert write_count == 2
    assert annotated_frame.cache_count == 1
    assert annotated_frame.unpersist_count == 1
    assert annotated_frame.unpersist_blocking_values == [True]


def test_build_silver_dataset_orchestrates_all_months(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator should process every discovered Bronze month."""
    specification = load_silver_quality_specification(PROJECT_SPECIFICATION_PATH)
    fake_spark = object()
    events: list[str] = []

    source_months = (
        "2024-01",
        "2024-02",
    )
    zone_ids = (
        1,
        2,
        3,
    )

    bronze_frames = {
        source_month: FakeFrame(f"bronze:{source_month}")
        for source_month in source_months
    }
    annotated_frames = {
        source_month: FakeFrame(f"annotated:{source_month}")
        for source_month in source_months
    }

    monkeypatch.setattr(
        orchestration,
        "load_silver_quality_specification",
        lambda path: specification,
    )

    def fake_load_months(
        spark: object,
        source_path: Path,
    ) -> tuple[str, ...]:
        events.append("load_months")
        assert spark is fake_spark
        assert source_path == (specification.source_trip_table)
        return source_months

    def fake_load_zones(
        spark: object,
        source_path: Path,
    ) -> tuple[int, ...]:
        events.append("load_zones")
        assert spark is fake_spark
        assert source_path == (specification.source_taxi_zone_table)
        return zone_ids

    def fake_load_trip_month(
        spark: object,
        source_path: Path,
        source_month: str,
    ) -> FakeFrame:
        events.append(f"load_trip:{source_month}")
        assert spark is fake_spark
        assert source_path == (specification.source_trip_table)
        return bronze_frames[source_month]

    def fake_annotate(
        frame: FakeFrame,
        source_month: str,
        received_zone_ids: tuple[int, ...],
        received_specification: Any,
    ) -> FakeFrame:
        events.append(f"annotate:{source_month}")
        assert frame is bronze_frames[source_month]
        assert received_zone_ids == zone_ids
        assert received_specification is specification
        return annotated_frames[source_month]

    monthly_counts = {
        "2024-01": (100, 1),
        "2024-02": (200, 2),
    }

    def fake_materialize(
        frame: FakeFrame,
        received_specification: Any,
        source_month: str,
    ) -> tuple[int, int]:
        events.append(f"materialize:{source_month}")
        assert frame is annotated_frames[source_month]
        assert received_specification is specification
        return monthly_counts[source_month]

    monkeypatch.setattr(
        orchestration,
        "load_bronze_source_months",
        fake_load_months,
    )
    monkeypatch.setattr(
        orchestration,
        "load_bronze_zone_ids",
        fake_load_zones,
    )
    monkeypatch.setattr(
        orchestration,
        "load_bronze_trip_month",
        fake_load_trip_month,
    )
    monkeypatch.setattr(
        orchestration,
        "annotate_silver_quality",
        fake_annotate,
    )
    monkeypatch.setattr(
        orchestration,
        "_materialize_split_and_write",
        fake_materialize,
    )

    result = orchestration.build_silver_dataset(
        fake_spark,
        PROJECT_SPECIFICATION_PATH,
    )

    assert result.specification is specification
    assert result.source_months == source_months
    assert result.zone_id_count == 3

    assert [
        (
            write.source_month,
            write.accepted_row_count,
            write.rejected_row_count,
            write.total_row_count,
        )
        for write in result.monthly_writes
    ] == [
        ("2024-01", 100, 1, 101),
        ("2024-02", 200, 2, 202),
    ]

    assert events == [
        "load_months",
        "load_zones",
        "load_trip:2024-01",
        "annotate:2024-01",
        "materialize:2024-01",
        "load_trip:2024-02",
        "annotate:2024-02",
        "materialize:2024-02",
    ]
