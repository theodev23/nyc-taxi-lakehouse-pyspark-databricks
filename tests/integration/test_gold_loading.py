"""Integration tests for Gold loading from Silver and Bronze Delta tables."""

from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from taxi_lakehouse.gold_loading import (
    load_silver_accepted_month,
    load_silver_source_months,
    load_taxi_zones,
)
from taxi_lakehouse.spark_session import build_local_spark_session


@pytest.fixture(scope="module")
def delta_spark() -> SparkSession:
    """Create one Delta-enabled Spark session for Gold loading tests."""
    session = build_local_spark_session(
        "gold-loading-integration-tests",
        enable_delta=True,
        master="local[1]",
    )

    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


def test_load_silver_accepted_month_filters_requested_partition(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Only rows from the requested Silver month should be loaded."""
    source_path = tmp_path / "silver_accepted"

    frame = delta_spark.createDataFrame(
        [
            (1, "2024-01"),
            (2, "2024-01"),
            (3, "2024-02"),
        ],
        "trip_id long, _source_month string",
    )

    (
        frame.write.format("delta")
        .partitionBy("_source_month")
        .save(source_path.as_posix())
    )

    january = load_silver_accepted_month(
        delta_spark,
        source_path,
        "2024-01",
    )

    rows = [
        (row["trip_id"], row["_source_month"])
        for row in january.orderBy("trip_id").collect()
    ]

    assert rows == [
        (1, "2024-01"),
        (2, "2024-01"),
    ]


def test_load_silver_source_months_returns_sorted_domain(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Source-month discovery should return sorted unique Silver partitions."""
    source_path = tmp_path / "silver_source_months"

    frame = delta_spark.createDataFrame(
        [
            (1, "2024-03"),
            (2, "2024-01"),
            (3, "2024-02"),
            (4, "2024-01"),
        ],
        "trip_id long, _source_month string",
    )

    (
        frame.write.format("delta")
        .partitionBy("_source_month")
        .save(source_path.as_posix())
    )

    assert load_silver_source_months(
        delta_spark,
        source_path,
    ) == (
        "2024-01",
        "2024-02",
        "2024-03",
    )


def test_load_silver_source_months_requires_nonempty_domain(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Silver source-month discovery should reject an all-null domain."""
    source_path = tmp_path / "empty_silver_source_months"

    frame = delta_spark.createDataFrame(
        [
            (1, None),
        ],
        "trip_id long, _source_month string",
    )

    frame.write.format("delta").save(source_path.as_posix())

    with pytest.raises(
        ValueError,
        match="must contain at least one source month",
    ):
        load_silver_source_months(
            delta_spark,
            source_path,
        )


def test_load_silver_accepted_month_rejects_invalid_month(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Malformed source months should fail before Delta loading."""
    source_path = tmp_path / "missing_table"

    with pytest.raises(
        ValueError,
        match="Invalid source month",
    ):
        load_silver_accepted_month(
            delta_spark,
            source_path,
            "2024-13",
        )


def test_load_silver_accepted_month_requires_partition_column(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Silver accepted trips must expose the source-month column."""
    source_path = tmp_path / "silver_without_month"

    frame = delta_spark.createDataFrame(
        [
            (1,),
            (2,),
        ],
        "trip_id long",
    )

    frame.write.format("delta").save(source_path.as_posix())

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        load_silver_accepted_month(
            delta_spark,
            source_path,
            "2024-01",
        )


def test_load_taxi_zones_returns_reference_frame(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Gold should load the Bronze taxi-zone reference unchanged."""
    source_path = tmp_path / "taxi_zones"

    frame = delta_spark.createDataFrame(
        [
            (1, "Manhattan", "Zone A", "Yellow Zone"),
            (2, "Queens", "Zone B", "Boro Zone"),
        ],
        "LocationID int, Borough string, Zone string, service_zone string",
    )

    frame.write.format("delta").save(source_path.as_posix())

    loaded = load_taxi_zones(
        delta_spark,
        source_path,
    )

    rows = [
        (
            row["LocationID"],
            row["Borough"],
            row["Zone"],
            row["service_zone"],
        )
        for row in loaded.orderBy("LocationID").collect()
    ]

    assert rows == [
        (1, "Manhattan", "Zone A", "Yellow Zone"),
        (2, "Queens", "Zone B", "Boro Zone"),
    ]


def test_load_taxi_zones_requires_reference_columns(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Gold taxi-zone loading should reject incomplete references."""
    source_path = tmp_path / "incomplete_taxi_zones"

    frame = delta_spark.createDataFrame(
        [
            (1, "Zone A"),
        ],
        "LocationID int, Zone string",
    )

    frame.write.format("delta").save(source_path.as_posix())

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        load_taxi_zones(
            delta_spark,
            source_path,
        )
