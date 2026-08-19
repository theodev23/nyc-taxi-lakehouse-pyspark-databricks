"""Tests for Databricks Silver Unity Catalog I/O."""

import pytest
from pyspark.sql import SparkSession

import taxi_lakehouse.databricks_silver_io as silver_io


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Provide one local Spark session for Silver I/O tests."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("databricks-silver-io-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


def test_load_managed_bronze_trip_month(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed Bronze loading should select only one month."""
    source = spark.createDataFrame(
        [
            ("2024-01", 1),
            ("2024-02", 2),
        ],
        "_source_month string, value int",
    )

    monkeypatch.setattr(
        silver_io,
        "read_managed_table",
        lambda received_spark, table_name: source,
    )

    result = silver_io.load_managed_bronze_trip_month(
        spark,
        "workspace.nyc_taxi.bronze_yellow_taxi_trips",
        "2024-01",
    )

    assert [row["value"] for row in result.collect()] == [1]


def test_load_managed_bronze_source_months(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed Bronze months should be unique and ordered."""
    source = spark.createDataFrame(
        [
            ("2024-02",),
            ("2024-01",),
            ("2024-02",),
            (None,),
        ],
        "_source_month string",
    )

    monkeypatch.setattr(
        silver_io,
        "read_managed_table",
        lambda received_spark, table_name: source,
    )

    assert silver_io.load_managed_bronze_source_months(
        spark,
        "workspace.nyc_taxi.bronze_yellow_taxi_trips",
    ) == (
        "2024-01",
        "2024-02",
    )


def test_load_managed_bronze_zone_ids(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed taxi-zone identifiers should be unique and ordered."""
    source = spark.createDataFrame(
        [
            (3,),
            (1,),
            (3,),
            (None,),
            (2,),
        ],
        "LocationID int",
    )

    monkeypatch.setattr(
        silver_io,
        "read_managed_table",
        lambda received_spark, table_name: source,
    )

    assert silver_io.load_managed_bronze_zone_ids(
        spark,
        "workspace.nyc_taxi.bronze_taxi_zones",
    ) == (
        1,
        2,
        3,
    )


def test_write_managed_silver_trip_month(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed Silver writes should replace only the requested month."""
    frame = spark.createDataFrame(
        [
            ("2024-01", 1),
        ],
        "_source_month string, value int",
    )
    events: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        silver_io,
        "replace_managed_table_rows",
        lambda received_frame, table_name, predicate: events.append(
            (
                received_frame,
                table_name,
                predicate,
            )
        ),
    )

    table_name = "workspace.nyc_taxi.silver_yellow_taxi_trips_accepted"

    silver_io.write_managed_silver_trip_month(
        frame,
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
