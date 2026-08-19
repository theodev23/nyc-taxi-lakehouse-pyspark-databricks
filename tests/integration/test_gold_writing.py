"""Integration tests for Gold Delta Lake writing."""

from pathlib import Path

import pytest
from delta.tables import DeltaTable
from pyspark.sql import SparkSession

from taxi_lakehouse.gold_writing import write_gold_month
from taxi_lakehouse.spark_session import build_local_spark_session


@pytest.fixture(scope="module")
def delta_spark() -> SparkSession:
    """Create one Delta-enabled Spark session for integration tests."""
    session = build_local_spark_session(
        "gold-writing-integration-tests",
        enable_delta=True,
        master="local[1]",
    )

    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


def test_write_gold_month_replaces_only_target_partition(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """A monthly rewrite should preserve every other Gold partition."""
    destination_path = tmp_path / "gold_metrics"

    january_initial = delta_spark.createDataFrame(
        [
            ("2024-01", "2024-01-01", 10),
            ("2024-01", "2024-01-02", 20),
        ],
        "_source_month string, pickup_date string, trip_count long",
    )

    february = delta_spark.createDataFrame(
        [
            ("2024-02", "2024-02-01", 30),
        ],
        "_source_month string, pickup_date string, trip_count long",
    )

    january_replacement = delta_spark.createDataFrame(
        [
            ("2024-01", "2024-01-03", 40),
        ],
        "_source_month string, pickup_date string, trip_count long",
    )

    write_gold_month(
        january_initial,
        destination_path,
        "2024-01",
    )

    write_gold_month(
        february,
        destination_path,
        "2024-02",
    )

    write_gold_month(
        january_replacement,
        destination_path,
        "2024-01",
    )

    rows = sorted(
        (
            row["_source_month"],
            row["pickup_date"],
            row["trip_count"],
        )
        for row in (
            delta_spark.read.format("delta")
            .load(destination_path.as_posix())
            .select(
                "_source_month",
                "pickup_date",
                "trip_count",
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
        ("2024-01", "2024-01-03", 40),
        ("2024-02", "2024-02-01", 30),
    ]
    assert table_details["partitionColumns"] == ["_source_month"]
    assert delta_table.history().count() == 3


def test_write_gold_month_rejects_invalid_month(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """The Gold partition month should respect YYYY-MM."""
    frame = delta_spark.createDataFrame(
        [
            ("2024-13", 1),
        ],
        "_source_month string, trip_count long",
    )

    destination_path = tmp_path / "invalid_month"

    with pytest.raises(
        ValueError,
        match="Invalid source month",
    ):
        write_gold_month(
            frame,
            destination_path,
            "2024-13",
        )

    assert not destination_path.exists()


def test_write_gold_month_requires_partition_column(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Monthly Gold writes require the partition column."""
    frame = delta_spark.createDataFrame(
        [
            (1,),
        ],
        "trip_count long",
    )

    destination_path = tmp_path / "missing_partition_column"

    with pytest.raises(
        ValueError,
        match="must contain column '_source_month'",
    ):
        write_gold_month(
            frame,
            destination_path,
            "2024-01",
        )

    assert not destination_path.exists()
