"""Tests for Databricks Silver orchestration."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import taxi_lakehouse.databricks_silver_orchestration as orchestration
from taxi_lakehouse.databricks_configuration import (
    DatabricksPipelineConfiguration,
)
from taxi_lakehouse.silver_quality_specification import (
    load_silver_quality_specification,
)

PROJECT_SPECIFICATION_PATH = Path("data/silver_quality_spec.json")


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


def test_materialize_split_and_write_uses_managed_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silver materialization should write both managed outputs."""
    specification = load_silver_quality_specification(PROJECT_SPECIFICATION_PATH)
    annotated_frame = FakeFrame("annotated")
    accepted_frame = FakeFrame(
        "accepted",
        row_count=100,
    )
    rejected_frame = FakeFrame(
        "rejected",
        row_count=2,
    )
    events: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        orchestration,
        "split_silver_quality_rows",
        lambda frame, received_specification: SimpleNamespace(
            accepted=accepted_frame,
            rejected=rejected_frame,
        ),
    )

    monkeypatch.setattr(
        orchestration,
        "write_managed_silver_trip_month",
        lambda frame, table_name, source_month: events.append(
            (
                frame,
                table_name,
                source_month,
            )
        ),
    )

    result = orchestration._materialize_split_and_write(
        annotated_frame,
        specification,
        "2024-01",
        "workspace.nyc_taxi.silver_accepted",
        "workspace.nyc_taxi.silver_rejected",
    )

    assert result == (100, 2)
    assert accepted_frame.count_count == 1
    assert rejected_frame.count_count == 1
    assert events == [
        (
            accepted_frame,
            "workspace.nyc_taxi.silver_accepted",
            "2024-01",
        ),
        (
            rejected_frame,
            "workspace.nyc_taxi.silver_rejected",
            "2024-01",
        ),
    ]


def test_build_databricks_silver_dataset_orchestrates_all_months(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Databricks Silver should process every managed Bronze month."""
    specification = load_silver_quality_specification(PROJECT_SPECIFICATION_PATH)
    configuration = DatabricksPipelineConfiguration(
        catalog="workspace",
        schema="nyc_taxi",
        volume="source_files",
    )
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
        table_name: str,
    ) -> tuple[str, ...]:
        events.append("load_months")
        assert spark is fake_spark
        assert table_name == configuration.bronze_trip_table
        return source_months

    def fake_load_zones(
        spark: object,
        table_name: str,
    ) -> tuple[int, ...]:
        events.append("load_zones")
        assert spark is fake_spark
        assert table_name == configuration.bronze_taxi_zone_table
        return zone_ids

    def fake_load_trip_month(
        spark: object,
        table_name: str,
        source_month: str,
    ) -> FakeFrame:
        events.append(f"load_trip:{source_month}")
        assert spark is fake_spark
        assert table_name == configuration.bronze_trip_table
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
        accepted_table: str,
        rejected_table: str,
    ) -> tuple[int, int]:
        events.append(f"materialize:{source_month}")
        assert frame is annotated_frames[source_month]
        assert received_specification is specification
        assert accepted_table == configuration.silver_accepted_table
        assert rejected_table == configuration.silver_rejected_table
        return monthly_counts[source_month]

    monkeypatch.setattr(
        orchestration,
        "load_managed_bronze_source_months",
        fake_load_months,
    )
    monkeypatch.setattr(
        orchestration,
        "load_managed_bronze_zone_ids",
        fake_load_zones,
    )
    monkeypatch.setattr(
        orchestration,
        "load_managed_bronze_trip_month",
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

    result = orchestration.build_databricks_silver_dataset(
        fake_spark,  # type: ignore[arg-type]
        PROJECT_SPECIFICATION_PATH,
        configuration,
    )

    assert result.specification is specification
    assert result.configuration is configuration
    assert result.source_months == source_months
    assert result.zone_id_count == 3

    assert [
        (
            write.source_month,
            write.accepted_destination_table,
            write.rejected_destination_table,
            write.accepted_row_count,
            write.rejected_row_count,
            write.total_row_count,
        )
        for write in result.monthly_writes
    ] == [
        (
            "2024-01",
            configuration.silver_accepted_table,
            configuration.silver_rejected_table,
            100,
            1,
            101,
        ),
        (
            "2024-02",
            configuration.silver_accepted_table,
            configuration.silver_rejected_table,
            200,
            2,
            202,
        ),
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
