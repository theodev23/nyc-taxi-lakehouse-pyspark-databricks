"""Tests for Databricks Gold Unity Catalog I/O."""

import pytest
from pyspark.sql import SparkSession

import taxi_lakehouse.databricks_gold_io as gold_io


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Provide one local Spark session for Gold I/O tests."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("databricks-gold-io-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


def test_load_managed_silver_accepted_month(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed Silver loading should select only one month."""
    source = spark.createDataFrame(
        [
            ("2024-01", 1),
            ("2024-02", 2),
        ],
        "_source_month string, value int",
    )

    monkeypatch.setattr(
        gold_io,
        "read_managed_table",
        lambda received_spark, table_name: source,
    )

    result = gold_io.load_managed_silver_accepted_month(
        spark,
        "workspace.nyc_taxi.silver_yellow_taxi_trips_accepted",
        "2024-01",
    )

    assert [row["value"] for row in result.collect()] == [1]


def test_load_managed_silver_source_months(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed Silver months should be unique and ordered."""
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
        gold_io,
        "read_managed_table",
        lambda received_spark, table_name: source,
    )

    assert gold_io.load_managed_silver_source_months(
        spark,
        "workspace.nyc_taxi.silver_yellow_taxi_trips_accepted",
    ) == (
        "2024-01",
        "2024-02",
    )


def test_load_managed_taxi_zones(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed Bronze zones should expose every Gold-required column."""
    source = spark.createDataFrame(
        [
            (
                1,
                "Manhattan",
                "Central Park",
                "Yellow Zone",
            ),
        ],
        ("LocationID int, Borough string, Zone string, service_zone string"),
    )

    monkeypatch.setattr(
        gold_io,
        "read_managed_table",
        lambda received_spark, table_name: source,
    )

    result = gold_io.load_managed_taxi_zones(
        spark,
        "workspace.nyc_taxi.bronze_taxi_zones",
    )

    assert result is source


def test_write_managed_gold_month(
    spark: SparkSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed Gold writes should replace only the requested month."""
    frame = spark.createDataFrame(
        [
            ("2024-01", 1),
        ],
        "_source_month string, value int",
    )
    events: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        gold_io,
        "replace_managed_table_rows",
        lambda received_frame, table_name, predicate: events.append(
            (
                received_frame,
                table_name,
                predicate,
            )
        ),
    )

    table_name = "workspace.nyc_taxi.gold_trip_metrics_by_date_pickup_zone_payment"

    gold_io.write_managed_gold_month(
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
