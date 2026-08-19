"""Tests for Gold analytical transformations."""

from datetime import datetime
from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
)

from taxi_lakehouse.gold_analytics_specification import (
    load_gold_analytics_specification,
)
from taxi_lakehouse.gold_transformation import (
    build_gold_analytics_frames,
)

PROJECT_SPECIFICATION_PATH = Path("data/gold_analytics_spec.json")


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Provide one local Spark session for Gold transformations."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("gold-transformation-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


@pytest.fixture(scope="module")
def gold_inputs(spark: SparkSession):
    """Build representative Silver trips and taxi-zone rows."""
    trip_schema = StructType(
        [
            StructField("_source_month", StringType(), False),
            StructField(
                "_quality_flags",
                ArrayType(StringType(), containsNull=False),
                False,
            ),
            StructField(
                "tpep_pickup_datetime",
                TimestampNTZType(),
                False,
            ),
            StructField(
                "tpep_dropoff_datetime",
                TimestampNTZType(),
                False,
            ),
            StructField("PULocationID", IntegerType(), False),
            StructField("payment_type", LongType(), False),
            StructField("trip_distance", DoubleType(), True),
            StructField("fare_amount", DoubleType(), True),
            StructField("tip_amount", DoubleType(), True),
            StructField("total_amount", DoubleType(), True),
        ]
    )

    trip_rows = [
        (
            "2024-01",
            [],
            datetime(2024, 1, 2, 10, 0),
            datetime(2024, 1, 2, 10, 30),
            1,
            1,
            2.0,
            10.0,
            2.0,
            13.0,
        ),
        (
            "2024-01",
            ["total_amount_negative"],
            datetime(2024, 1, 2, 11, 0),
            datetime(2024, 1, 2, 12, 0),
            1,
            1,
            4.0,
            20.0,
            4.0,
            25.0,
        ),
        (
            "2024-01",
            [],
            datetime(2024, 1, 2, 9, 0),
            datetime(2024, 1, 2, 9, 15),
            2,
            2,
            1.0,
            5.0,
            1.0,
            7.0,
        ),
        (
            "2024-01",
            [],
            datetime(2024, 1, 3, 8, 0),
            datetime(2024, 1, 3, 8, 20),
            1,
            1,
            3.0,
            12.0,
            3.0,
            16.0,
        ),
    ]

    zone_schema = StructType(
        [
            StructField("LocationID", IntegerType(), False),
            StructField("Borough", StringType(), False),
            StructField("Zone", StringType(), False),
            StructField("service_zone", StringType(), False),
        ]
    )

    zone_rows = [
        (1, "Manhattan", "Zone A", "Yellow Zone"),
        (2, "Queens", "Zone B", "Boro Zone"),
    ]

    return (
        spark.createDataFrame(
            trip_rows,
            schema=trip_schema,
        ),
        spark.createDataFrame(
            zone_rows,
            schema=zone_schema,
        ),
    )


def test_build_gold_trip_metrics(
    gold_inputs,
) -> None:
    """Trip metrics should aggregate by date, pickup zone, and payment."""
    trips, zones = gold_inputs
    specification = load_gold_analytics_specification(PROJECT_SPECIFICATION_PATH)

    result = build_gold_analytics_frames(
        trips,
        zones,
        specification,
    )

    rows = {
        (
            row["_source_month"],
            str(row["pickup_date"]),
            row["pickup_location_id"],
            row["payment_type"],
        ): row
        for row in result.trip_metrics.collect()
    }

    assert len(rows) == 3

    row = rows[("2024-01", "2024-01-02", 1, 1)]

    assert row["pickup_borough"] == "Manhattan"
    assert row["pickup_zone"] == "Zone A"
    assert row["pickup_service_zone"] == "Yellow Zone"

    assert row["trip_count"] == 2
    assert row["quality_flagged_trip_count"] == 1
    assert row["trip_distance_sum"] == pytest.approx(6.0)
    assert row["trip_distance_avg"] == pytest.approx(3.0)
    assert row["trip_duration_minutes_avg"] == pytest.approx(45.0)
    assert row["fare_amount_sum"] == pytest.approx(30.0)
    assert row["tip_amount_sum"] == pytest.approx(6.0)
    assert row["total_amount_sum"] == pytest.approx(38.0)
    assert row["total_amount_avg"] == pytest.approx(19.0)


def test_build_gold_daily_metrics(
    gold_inputs,
) -> None:
    """Daily metrics should aggregate all accepted trips for each date."""
    trips, zones = gold_inputs
    specification = load_gold_analytics_specification(PROJECT_SPECIFICATION_PATH)

    result = build_gold_analytics_frames(
        trips,
        zones,
        specification,
    )

    rows = {
        (
            row["_source_month"],
            str(row["pickup_date"]),
        ): row
        for row in result.daily_metrics.collect()
    }

    assert len(rows) == 2

    row = rows[("2024-01", "2024-01-02")]

    assert row["trip_count"] == 3
    assert row["quality_flagged_trip_count"] == 1
    assert row["trip_distance_sum"] == pytest.approx(7.0)
    assert row["trip_distance_avg"] == pytest.approx(7.0 / 3.0)
    assert row["trip_duration_minutes_avg"] == pytest.approx(35.0)
    assert row["fare_amount_sum"] == pytest.approx(35.0)
    assert row["tip_amount_sum"] == pytest.approx(7.0)
    assert row["total_amount_sum"] == pytest.approx(45.0)
    assert row["total_amount_avg"] == pytest.approx(15.0)


def test_build_gold_analytics_requires_input_columns(
    spark: SparkSession,
    gold_inputs,
) -> None:
    """Gold transformation should fail explicitly on incomplete inputs."""
    _, zones = gold_inputs
    specification = load_gold_analytics_specification(PROJECT_SPECIFICATION_PATH)

    incomplete_trips = spark.createDataFrame(
        [
            ("2024-01",),
        ],
        ["_source_month"],
    )

    with pytest.raises(
        ValueError,
        match="Silver accepted trips is missing required columns",
    ):
        build_gold_analytics_frames(
            incomplete_trips,
            zones,
            specification,
        )
