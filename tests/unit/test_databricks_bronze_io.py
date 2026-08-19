"""Tests for Databricks Bronze Unity Catalog I/O."""

import pytest

import taxi_lakehouse.databricks_bronze_io as bronze_io


class FakeFrame:
    """Synthetic frame exposing a configurable column list."""

    def __init__(
        self,
        columns: list[str],
    ) -> None:
        self.columns = columns


def test_write_managed_bronze_trip_month_uses_replace_where(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed Bronze trip writes should replace only one month."""
    frame = FakeFrame(
        [
            "_source_month",
            "VendorID",
        ]
    )
    events: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        bronze_io,
        "replace_managed_table_rows",
        lambda received_frame, table_name, predicate: events.append(
            (
                received_frame,
                table_name,
                predicate,
            )
        ),
    )

    table_name = "workspace.nyc_taxi.bronze_yellow_taxi_trips"

    bronze_io.write_managed_bronze_trip_month(
        frame,  # type: ignore[arg-type]
        table_name,
        "2024-01",
    )

    assert events == [
        (
            frame,
            table_name,
            "_source_month = '2024-01'",
        )
    ]


def test_write_managed_bronze_taxi_zones_uses_full_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed Bronze taxi zones should replace the complete snapshot."""
    frame = FakeFrame(
        [
            "LocationID",
            "Borough",
        ]
    )
    events: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        bronze_io,
        "overwrite_managed_table",
        lambda received_frame, table_name: events.append(
            (
                received_frame,
                table_name,
            )
        ),
    )

    table_name = "workspace.nyc_taxi.bronze_taxi_zones"

    bronze_io.write_managed_bronze_taxi_zones(
        frame,  # type: ignore[arg-type]
        table_name,
    )

    assert events == [
        (
            frame,
            table_name,
        )
    ]


def test_write_managed_bronze_trip_month_rejects_invalid_month(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid source months should fail before managed-table writes."""
    frame = FakeFrame(
        [
            "_source_month",
        ]
    )
    events: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        bronze_io,
        "replace_managed_table_rows",
        lambda received_frame, table_name, predicate: events.append(
            (
                received_frame,
                table_name,
                predicate,
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="YYYY-MM",
    ):
        bronze_io.write_managed_bronze_trip_month(
            frame,  # type: ignore[arg-type]
            "workspace.nyc_taxi.bronze_yellow_taxi_trips",
            "2024-13",
        )

    assert events == []


def test_write_managed_bronze_trip_month_requires_source_month_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bronze managed writes must preserve the source-month column."""
    frame = FakeFrame(
        [
            "VendorID",
        ]
    )
    events: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        bronze_io,
        "replace_managed_table_rows",
        lambda received_frame, table_name, predicate: events.append(
            (
                received_frame,
                table_name,
                predicate,
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="_source_month",
    ):
        bronze_io.write_managed_bronze_trip_month(
            frame,  # type: ignore[arg-type]
            "workspace.nyc_taxi.bronze_yellow_taxi_trips",
            "2024-01",
        )

    assert events == []
