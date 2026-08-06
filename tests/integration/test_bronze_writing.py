"""Integration tests for Bronze Delta Lake writing."""

from pathlib import Path

import pytest
from delta.tables import DeltaTable
from pyspark.sql import SparkSession

from taxi_lakehouse.bronze_writing import (
    write_bronze_taxi_zones,
    write_bronze_trip_month,
)
from taxi_lakehouse.spark_session import build_local_spark_session


@pytest.fixture(scope="module")
def delta_spark() -> SparkSession:
    """Create one Delta-enabled Spark session for integration tests."""
    session = build_local_spark_session(
        "bronze-writing-integration-tests",
        enable_delta=True,
    )
    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


def test_write_bronze_trip_month_replaces_only_target_partition(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """A monthly rewrite should preserve every other partition."""
    destination_path = tmp_path / "yellow_taxi_trips"

    january_initial = delta_spark.createDataFrame(
        [
            (1, "2024-01"),
            (2, "2024-01"),
        ],
        "trip_id long, _source_month string",
    )
    february = delta_spark.createDataFrame(
        [(3, "2024-02")],
        "trip_id long, _source_month string",
    )
    january_replacement = delta_spark.createDataFrame(
        [(4, "2024-01")],
        "trip_id long, _source_month string",
    )

    write_bronze_trip_month(
        january_initial,
        destination_path,
        "2024-01",
    )
    write_bronze_trip_month(
        february,
        destination_path,
        "2024-02",
    )
    write_bronze_trip_month(
        january_replacement,
        destination_path,
        "2024-01",
    )

    rows = sorted(
        (row["trip_id"], row["_source_month"])
        for row in (
            delta_spark.read.format("delta")
            .load(destination_path.as_posix())
            .select(
                "trip_id",
                "_source_month",
            )
            .collect()
        )
    )

    delta_table = DeltaTable.forPath(
        delta_spark,
        destination_path.as_posix(),
    )
    table_details = delta_table.detail().first()

    assert rows == [
        (3, "2024-02"),
        (4, "2024-01"),
    ]
    assert table_details["partitionColumns"] == ["_source_month"]
    assert delta_table.history().count() == 3


def test_write_bronze_taxi_zones_replaces_complete_snapshot(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """A taxi-zone rewrite should replace the previous snapshot."""
    destination_path = tmp_path / "taxi_zones"

    initial_snapshot = delta_spark.createDataFrame(
        [
            (1, "First zone"),
            (2, "Second zone"),
        ],
        "LocationID int, Zone string",
    )
    replacement_snapshot = delta_spark.createDataFrame(
        [(3, "Replacement zone")],
        "LocationID int, Zone string",
    )

    write_bronze_taxi_zones(
        initial_snapshot,
        destination_path,
    )
    write_bronze_taxi_zones(
        replacement_snapshot,
        destination_path,
    )

    rows = [
        (row["LocationID"], row["Zone"])
        for row in (
            delta_spark.read.format("delta")
            .load(destination_path.as_posix())
            .select(
                "LocationID",
                "Zone",
            )
            .collect()
        )
    ]

    delta_table = DeltaTable.forPath(
        delta_spark,
        destination_path.as_posix(),
    )

    assert rows == [(3, "Replacement zone")]
    assert delta_table.history().count() == 2


def test_write_bronze_trip_month_rejects_invalid_month(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """The partition month should respect the YYYY-MM contract."""
    frame = delta_spark.createDataFrame(
        [(1, "2024-13")],
        "trip_id long, _source_month string",
    )
    destination_path = tmp_path / "invalid_month"

    with pytest.raises(
        ValueError,
        match="must use YYYY-MM format",
    ):
        write_bronze_trip_month(
            frame,
            destination_path,
            "2024-13",
        )

    assert not destination_path.exists()


def test_write_bronze_trip_month_requires_partition_column(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Monthly writes require the Bronze partition column."""
    frame = delta_spark.createDataFrame(
        [(1,)],
        "trip_id long",
    )
    destination_path = tmp_path / "missing_partition_column"

    with pytest.raises(
        ValueError,
        match="must contain column '_source_month'",
    ):
        write_bronze_trip_month(
            frame,
            destination_path,
            "2024-01",
        )

    assert not destination_path.exists()
