"""Tests for Databricks Gold orchestration."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import taxi_lakehouse.databricks_gold_orchestration as orchestration
from taxi_lakehouse.databricks_configuration import (
    DatabricksPipelineConfiguration,
)
from taxi_lakehouse.gold_analytics_specification import (
    load_gold_analytics_specification,
)

PROJECT_SPECIFICATION_PATH = Path("data/gold_analytics_spec.json")


class FakeFrame:
    """Synthetic frame exposing only row counting."""

    def __init__(
        self,
        label: str,
        row_count: int = 0,
    ) -> None:
        self.label = label
        self.row_count = row_count
        self.count_count = 0

    def count(self) -> int:
        """Return the configured row count."""
        self.count_count += 1
        return self.row_count


def test_materialize_and_write_month_uses_managed_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gold materialization should write both managed outputs."""
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
    events: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        orchestration,
        "write_managed_gold_month",
        lambda frame, table_name, source_month: events.append(
            (
                frame,
                table_name,
                source_month,
            )
        ),
    )

    result = orchestration._materialize_and_write_month(
        analytics_frames,
        "2024-01",
        "workspace.nyc_taxi.gold_trip_metrics",
        "workspace.nyc_taxi.gold_daily_metrics",
    )

    assert result == (101, 31)
    assert trip_metrics.count_count == 1
    assert daily_metrics.count_count == 1
    assert events == [
        (
            trip_metrics,
            "workspace.nyc_taxi.gold_trip_metrics",
            "2024-01",
        ),
        (
            daily_metrics,
            "workspace.nyc_taxi.gold_daily_metrics",
            "2024-01",
        ),
    ]


def test_build_databricks_gold_dataset_orchestrates_all_months(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Databricks Gold should process every managed Silver month."""
    specification = load_gold_analytics_specification(PROJECT_SPECIFICATION_PATH)
    configuration = DatabricksPipelineConfiguration(
        catalog="workspace",
        schema="nyc_taxi",
        volume="source_files",
    )
    fake_spark = object()
    zones = FakeFrame("zones")
    events: list[str] = []

    source_months = (
        "2024-01",
        "2024-02",
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
        table_name: str,
    ) -> tuple[str, ...]:
        events.append("load_months")
        assert spark is fake_spark
        assert table_name == configuration.silver_accepted_table
        return source_months

    def fake_load_zones(
        spark: object,
        table_name: str,
    ) -> FakeFrame:
        events.append("load_zones")
        assert spark is fake_spark
        assert table_name == configuration.bronze_taxi_zone_table
        return zones

    def fake_validate_zones(
        frame: FakeFrame,
    ) -> int:
        events.append("validate_zones")
        assert frame is zones
        return 265

    def fake_load_month(
        spark: object,
        table_name: str,
        source_month: str,
    ) -> FakeFrame:
        events.append(f"load_silver:{source_month}")
        assert spark is fake_spark
        assert table_name == configuration.silver_accepted_table
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
        source_month: str,
        trip_metrics_table: str,
        daily_metrics_table: str,
    ) -> tuple[int, int]:
        events.append(f"materialize:{source_month}")
        assert frames is analytics_frames[source_month]
        assert trip_metrics_table == configuration.gold_trip_metrics_table
        assert daily_metrics_table == configuration.gold_daily_metrics_table
        return monthly_counts[source_month]

    monkeypatch.setattr(
        orchestration,
        "load_managed_silver_source_months",
        fake_load_months,
    )
    monkeypatch.setattr(
        orchestration,
        "load_managed_taxi_zones",
        fake_load_zones,
    )
    monkeypatch.setattr(
        orchestration,
        "_validate_taxi_zone_reference",
        fake_validate_zones,
    )
    monkeypatch.setattr(
        orchestration,
        "load_managed_silver_accepted_month",
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

    result = orchestration.build_databricks_gold_dataset(
        fake_spark,  # type: ignore[arg-type]
        PROJECT_SPECIFICATION_PATH,
        configuration,
    )

    assert result.specification is specification
    assert result.configuration is configuration
    assert result.source_months == source_months
    assert result.taxi_zone_row_count == 265

    assert [
        (
            write.source_month,
            write.trip_metrics_destination_table,
            write.daily_metrics_destination_table,
            write.trip_metrics_row_count,
            write.daily_metrics_row_count,
        )
        for write in result.monthly_writes
    ] == [
        (
            "2024-01",
            configuration.gold_trip_metrics_table,
            configuration.gold_daily_metrics_table,
            100,
            31,
        ),
        (
            "2024-02",
            configuration.gold_trip_metrics_table,
            configuration.gold_daily_metrics_table,
            200,
            29,
        ),
    ]

    assert events == [
        "load_months",
        "load_zones",
        "validate_zones",
        "load_silver:2024-01",
        "transform:2024-01",
        "materialize:2024-01",
        "load_silver:2024-02",
        "transform:2024-02",
        "materialize:2024-02",
    ]
