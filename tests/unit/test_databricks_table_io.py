"""Tests for Databricks Unity Catalog managed-table I/O."""

import pytest

from taxi_lakehouse.databricks_table_io import (
    overwrite_managed_table,
    read_managed_table,
    replace_managed_table_rows,
    require_fully_qualified_table_name,
)


class FakeReader:
    """Record managed-table reads."""

    def __init__(self) -> None:
        self.table_names: list[str] = []

    def table(self, table_name: str) -> object:
        """Record and return one synthetic DataFrame."""
        self.table_names.append(table_name)
        return object()


class FakeWriter:
    """Record chained DataFrameWriter operations."""

    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []

    def format(self, value: str) -> "FakeWriter":
        self.events.append(("format", value))
        return self

    def mode(self, value: str) -> "FakeWriter":
        self.events.append(("mode", value))
        return self

    def option(
        self,
        key: str,
        value: str,
    ) -> "FakeWriter":
        self.events.append(("option", key, value))
        return self

    def saveAsTable(self, table_name: str) -> None:
        self.events.append(("saveAsTable", table_name))


class FakeFrame:
    """Expose one synthetic DataFrameWriter."""

    def __init__(self) -> None:
        self.write = FakeWriter()


class FakeSpark:
    """Expose one synthetic DataFrameReader."""

    def __init__(self) -> None:
        self.read = FakeReader()


@pytest.mark.parametrize(
    "table_name",
    [
        "workspace.nyc_taxi.bronze_yellow_taxi_trips",
        "main.analytics.daily_metrics",
        "_catalog._schema._table",
    ],
)
def test_require_fully_qualified_table_name_accepts_valid_names(
    table_name: str,
) -> None:
    """Valid three-part table names should be returned unchanged."""
    assert require_fully_qualified_table_name(table_name) == table_name


@pytest.mark.parametrize(
    "table_name",
    [
        "",
        "table",
        "schema.table",
        "catalog.schema.table.extra",
        "catalog.schema.invalid-table",
        "catalog.9schema.table",
        None,
    ],
)
def test_require_fully_qualified_table_name_rejects_invalid_names(
    table_name: object,
) -> None:
    """Invalid managed-table names should fail before Spark access."""
    with pytest.raises(ValueError):
        require_fully_qualified_table_name(table_name)


def test_read_managed_table_uses_fully_qualified_name() -> None:
    """Managed tables should be loaded through Spark's table API."""
    spark = FakeSpark()
    table_name = "workspace.nyc_taxi.bronze_taxi_zones"

    result = read_managed_table(
        spark,  # type: ignore[arg-type]
        table_name,
    )

    assert spark.read.table_names == [table_name]
    assert result is not None


def test_overwrite_managed_table_uses_delta_table_write() -> None:
    """Complete snapshots should overwrite one managed Delta table."""
    frame = FakeFrame()
    table_name = "workspace.nyc_taxi.bronze_taxi_zones"

    overwrite_managed_table(
        frame,  # type: ignore[arg-type]
        table_name,
    )

    assert frame.write.events == [
        ("format", "delta"),
        ("mode", "overwrite"),
        ("saveAsTable", table_name),
    ]


def test_replace_managed_table_rows_uses_replace_where() -> None:
    """Selective managed-table writes should use replaceWhere."""
    frame = FakeFrame()
    table_name = "workspace.nyc_taxi.silver_yellow_taxi_trips_accepted"

    replace_managed_table_rows(
        frame,  # type: ignore[arg-type]
        table_name,
        "_source_month = '2024-01'",
    )

    assert frame.write.events == [
        ("format", "delta"),
        ("mode", "overwrite"),
        (
            "option",
            "replaceWhere",
            "_source_month = '2024-01'",
        ),
        ("saveAsTable", table_name),
    ]


@pytest.mark.parametrize(
    "predicate",
    [
        "",
        "   ",
    ],
)
def test_replace_managed_table_rows_rejects_empty_predicate(
    predicate: str,
) -> None:
    """Selective overwrites must never run without a predicate."""
    frame = FakeFrame()

    with pytest.raises(
        ValueError,
        match="predicate",
    ):
        replace_managed_table_rows(
            frame,  # type: ignore[arg-type]
            "workspace.nyc_taxi.gold_daily_trip_metrics",
            predicate,
        )

    assert frame.write.events == []
