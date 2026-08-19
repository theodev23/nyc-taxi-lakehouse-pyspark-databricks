"""Tests for end-to-end Gold analytical orchestration."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pyspark.sql import SparkSession

import taxi_lakehouse.gold_orchestration as orchestration
from taxi_lakehouse.gold_analytics_specification import (
    load_gold_analytics_specification,
)

PROJECT_SPECIFICATION_PATH = Path("data/gold_analytics_spec.json")


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
        """Record release of one cached frame."""
        self.unpersist_count += 1
        self.unpersist_blocking_values.append(blocking)

        if self.events is not None:
            self.events.append(f"unpersist:{self.label}")

        return self


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Provide one local Spark session for taxi-zone validation."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("gold-orchestration-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


def test_validate_taxi_zone_reference_returns_row_count(
    spark: SparkSession,
) -> None:
    """A valid unique taxi-zone domain should return its row count."""
    zones = spark.createDataFrame(
        [
            (1,),
            (2,),
            (3,),
        ],
        "LocationID int",
    )

    assert orchestration._validate_taxi_zone_reference(zones) == 3


@pytest.mark.parametrize(
    ("rows", "error_message"),
    [
        (
            [],
            "must contain at least one row",
        ),
        (
            [(1,), (None,)],
            "must not contain null LocationID values",
        ),
        (
            [(1,), (1,)],
            "must contain unique LocationID values",
        ),
    ],
)
def test_validate_taxi_zone_reference_rejects_invalid_domain(
    spark: SparkSession,
    rows: list[tuple[int | None]],
    error_message: str,
) -> None:
    """Invalid taxi-zone domains should fail before Gold aggregation."""
    zones = spark.createDataFrame(
        rows,
        "LocationID int",
    )

    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        orchestration._validate_taxi_zone_reference(zones)


def test_materialize_and_write_month_releases_cached_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful Gold writes should release both aggregate caches."""
    specification = load_gold_analytics_specification(PROJECT_SPECIFICATION_PATH)
    events: list[str] = []

    trip_metrics = FakeFrame(
        "trip_metrics",
        row_count=101,
        events=events,
    )
    daily_metrics = FakeFrame(
        "daily_metrics",
        row_count=31,
        events=events,
    )

    analytics_frames = SimpleNamespace(
        trip_metrics=trip_metrics,
        daily_metrics=daily_metrics,
    )

    def fake_write(
        frame: FakeFrame,
        destination_path: Path,
        source_month: str,
    ) -> None:
        events.append(f"write:{frame.label}:{source_month}")

        if frame is trip_metrics:
            assert destination_path == specification.outputs.trip_metrics.table
        else:
            assert frame is daily_metrics
            assert destination_path == specification.outputs.daily_metrics.table

    monkeypatch.setattr(
        orchestration,
        "write_gold_month",
        fake_write,
    )

    result = orchestration._materialize_and_write_month(
        analytics_frames,
        specification,
        "2024-01",
    )

    assert result == (101, 31)
    assert trip_metrics.cache_count == 1
    assert daily_metrics.cache_count == 1
    assert trip_metrics.count_count == 1
    assert daily_metrics.count_count == 1
    assert trip_metrics.unpersist_count == 1
    assert daily_metrics.unpersist_count == 1
    assert trip_metrics.unpersist_blocking_values == [True]
    assert daily_metrics.unpersist_blocking_values == [True]

    assert events == [
        "cache:trip_metrics",
        "cache:daily_metrics",
        "count:trip_metrics",
        "count:daily_metrics",
        "write:trip_metrics:2024-01",
        "write:daily_metrics:2024-01",
        "unpersist:daily_metrics",
        "unpersist:trip_metrics",
    ]


def test_materialize_and_write_month_without_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Databricks execution should write Gold without Spark caching."""
    specification = load_gold_analytics_specification(PROJECT_SPECIFICATION_PATH)

    trip_metrics = FakeFrame(
        "trip_metrics",
        row_count=101,
    )
    daily_metrics = FakeFrame(
        "daily_metrics",
        row_count=31,
    )

    analytics_frames = SimpleNamespace(
        trip_metrics=trip_metrics,
        daily_metrics=daily_metrics,
    )

    monkeypatch.setattr(
        orchestration,
        "write_gold_month",
        lambda frame, destination_path, source_month: None,
    )

    result = orchestration._materialize_and_write_month(
        analytics_frames,
        specification,
        "2024-01",
        use_cache=False,
    )

    assert result == (101, 31)
    assert trip_metrics.cache_count == 0
    assert daily_metrics.cache_count == 0
    assert trip_metrics.count_count == 1
    assert daily_metrics.count_count == 1
    assert trip_metrics.unpersist_count == 0
    assert daily_metrics.unpersist_count == 0


def test_materialize_and_write_month_releases_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed Gold write should still release both aggregate caches."""
    specification = load_gold_analytics_specification(PROJECT_SPECIFICATION_PATH)

    trip_metrics = FakeFrame(
        "trip_metrics",
        row_count=101,
    )
    daily_metrics = FakeFrame(
        "daily_metrics",
        row_count=31,
    )

    analytics_frames = SimpleNamespace(
        trip_metrics=trip_metrics,
        daily_metrics=daily_metrics,
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
            raise RuntimeError("synthetic Gold Delta write failure")

    monkeypatch.setattr(
        orchestration,
        "write_gold_month",
        fail_second_write,
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic Gold Delta write failure",
    ):
        orchestration._materialize_and_write_month(
            analytics_frames,
            specification,
            "2024-01",
        )

    assert write_count == 2
    assert trip_metrics.unpersist_count == 1
    assert daily_metrics.unpersist_count == 1
    assert trip_metrics.unpersist_blocking_values == [True]
    assert daily_metrics.unpersist_blocking_values == [True]


def test_build_gold_dataset_propagates_disabled_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Databricks Gold execution should avoid and propagate caching."""
    specification = load_gold_analytics_specification(PROJECT_SPECIFICATION_PATH)
    zones = FakeFrame("zones")
    received_cache_values: list[bool] = []

    monkeypatch.setattr(
        orchestration,
        "load_gold_analytics_specification",
        lambda path: specification,
    )
    monkeypatch.setattr(
        orchestration,
        "load_silver_source_months",
        lambda spark, source_path: ("2024-01",),
    )
    monkeypatch.setattr(
        orchestration,
        "load_taxi_zones",
        lambda spark, source_path: zones,
    )
    monkeypatch.setattr(
        orchestration,
        "_validate_taxi_zone_reference",
        lambda frame: 265,
    )
    monkeypatch.setattr(
        orchestration,
        "load_silver_accepted_month",
        lambda spark, source_path, source_month: FakeFrame("silver"),
    )
    monkeypatch.setattr(
        orchestration,
        "build_gold_analytics_frames",
        lambda frame, received_zones, spec: SimpleNamespace(
            trip_metrics=FakeFrame("trip_metrics"),
            daily_metrics=FakeFrame("daily_metrics"),
        ),
    )

    def fake_materialize(
        frames: SimpleNamespace,
        received_specification: Any,
        source_month: str,
        *,
        use_cache: bool = True,
    ) -> tuple[int, int]:
        received_cache_values.append(use_cache)
        return (100, 31)

    monkeypatch.setattr(
        orchestration,
        "_materialize_and_write_month",
        fake_materialize,
    )

    orchestration.build_gold_dataset(
        object(),
        PROJECT_SPECIFICATION_PATH,
        use_cache=False,
    )

    assert zones.cache_count == 0
    assert zones.unpersist_count == 0
    assert received_cache_values == [False]


def test_build_gold_dataset_orchestrates_all_months(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator should process every discovered Silver month."""
    specification = load_gold_analytics_specification(PROJECT_SPECIFICATION_PATH)
    fake_spark = object()
    events: list[str] = []

    source_months = (
        "2024-01",
        "2024-02",
    )

    zones = FakeFrame(
        "zones",
        events=events,
    )

    silver_frames = {
        source_month: FakeFrame(f"silver:{source_month}")
        for source_month in source_months
    }

    analytics_frames = {
        source_month: SimpleNamespace(
            trip_metrics=FakeFrame(f"trip_metrics:{source_month}"),
            daily_metrics=FakeFrame(f"daily_metrics:{source_month}"),
        )
        for source_month in source_months
    }

    monkeypatch.setattr(
        orchestration,
        "load_gold_analytics_specification",
        lambda path: specification,
    )

    def fake_load_months(
        spark: object,
        source_path: Path,
    ) -> tuple[str, ...]:
        events.append("load_months")
        assert spark is fake_spark
        assert source_path == specification.source.accepted_trip_table
        return source_months

    def fake_load_zones(
        spark: object,
        source_path: Path,
    ) -> FakeFrame:
        events.append("load_zones")
        assert spark is fake_spark
        assert source_path == specification.source.taxi_zone_table
        return zones

    def fake_validate_zones(
        frame: FakeFrame,
    ) -> int:
        events.append("validate_zones")
        assert frame is zones
        return 265

    def fake_load_month(
        spark: object,
        source_path: Path,
        source_month: str,
    ) -> FakeFrame:
        events.append(f"load_silver:{source_month}")
        assert spark is fake_spark
        assert source_path == specification.source.accepted_trip_table
        return silver_frames[source_month]

    def fake_transform(
        frame: FakeFrame,
        received_zones: FakeFrame,
        received_specification: Any,
    ) -> SimpleNamespace:
        source_month = frame.label.removeprefix("silver:")
        events.append(f"transform:{source_month}")
        assert frame is silver_frames[source_month]
        assert received_zones is zones
        assert received_specification is specification
        return analytics_frames[source_month]

    monthly_counts = {
        "2024-01": (100, 31),
        "2024-02": (200, 29),
    }

    def fake_materialize(
        frames: SimpleNamespace,
        received_specification: Any,
        source_month: str,
        *,
        use_cache: bool = True,
    ) -> tuple[int, int]:
        events.append(f"materialize:{source_month}")
        assert frames is analytics_frames[source_month]
        assert received_specification is specification
        assert use_cache is True
        return monthly_counts[source_month]

    monkeypatch.setattr(
        orchestration,
        "load_silver_source_months",
        fake_load_months,
    )
    monkeypatch.setattr(
        orchestration,
        "load_taxi_zones",
        fake_load_zones,
    )
    monkeypatch.setattr(
        orchestration,
        "_validate_taxi_zone_reference",
        fake_validate_zones,
    )
    monkeypatch.setattr(
        orchestration,
        "load_silver_accepted_month",
        fake_load_month,
    )
    monkeypatch.setattr(
        orchestration,
        "build_gold_analytics_frames",
        fake_transform,
    )
    monkeypatch.setattr(
        orchestration,
        "_materialize_and_write_month",
        fake_materialize,
    )

    result = orchestration.build_gold_dataset(
        fake_spark,
        PROJECT_SPECIFICATION_PATH,
    )

    assert result.specification is specification
    assert result.source_months == source_months
    assert result.taxi_zone_row_count == 265

    assert [
        (
            write.source_month,
            write.trip_metrics_row_count,
            write.daily_metrics_row_count,
            write.trip_metrics_destination_path,
            write.daily_metrics_destination_path,
        )
        for write in result.monthly_writes
    ] == [
        (
            "2024-01",
            100,
            31,
            specification.outputs.trip_metrics.table,
            specification.outputs.daily_metrics.table,
        ),
        (
            "2024-02",
            200,
            29,
            specification.outputs.trip_metrics.table,
            specification.outputs.daily_metrics.table,
        ),
    ]

    assert zones.cache_count == 1
    assert zones.unpersist_count == 1
    assert zones.unpersist_blocking_values == [True]

    assert events == [
        "load_months",
        "load_zones",
        "cache:zones",
        "validate_zones",
        "load_silver:2024-01",
        "transform:2024-01",
        "materialize:2024-01",
        "load_silver:2024-02",
        "transform:2024-02",
        "materialize:2024-02",
        "unpersist:zones",
    ]
