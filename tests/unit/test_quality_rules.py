"""Tests for shared NYC Taxi data-quality rules."""

from datetime import datetime

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
)

from taxi_lakehouse.quality_rules import (
    QUALITY_RULE_NAMES,
    build_quality_rule_conditions,
)


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Provide one local Spark session for quality-rule tests."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("quality-rule-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


def test_quality_rule_conditions_match_expected_cases(
    spark: SparkSession,
) -> None:
    """Every shared quality rule should detect its representative case."""
    schema = StructType(
        [
            StructField("case", StringType(), False),
            StructField("tpep_pickup_datetime", TimestampNTZType(), True),
            StructField("tpep_dropoff_datetime", TimestampNTZType(), True),
            StructField("passenger_count", LongType(), True),
            StructField("trip_distance", DoubleType(), True),
            StructField("PULocationID", LongType(), True),
            StructField("DOLocationID", LongType(), True),
            StructField("total_amount", DoubleType(), True),
        ]
    )

    pickup = datetime(2024, 1, 2, 10, 0)
    dropoff = datetime(2024, 1, 2, 10, 30)

    rows = [
        ("valid", pickup, dropoff, 1, 2.5, 1, 2, 20.0),
        ("pickup_zone_unknown", pickup, dropoff, 1, 2.5, None, 2, 20.0),
        ("dropoff_zone_unknown", pickup, dropoff, 1, 2.5, 1, 999, 20.0),
        ("distance_negative", pickup, dropoff, 1, -1.0, 1, 2, 20.0),
        (
            "duration_over_24h",
            pickup,
            datetime(2024, 1, 3, 11, 0),
            1,
            2.5,
            1,
            2,
            20.0,
        ),
        ("passenger_over_6", pickup, dropoff, 7, 2.5, 1, 2, 20.0),
        ("total_amount_zero", pickup, dropoff, 1, 2.5, 1, 2, 0.0),
        ("duration_nonpositive", pickup, pickup, 1, 2.5, 1, 2, 20.0),
        (
            "pickup_outside_month",
            datetime(2023, 12, 31, 23, 0),
            datetime(2024, 1, 1, 0, 30),
            1,
            2.5,
            1,
            2,
            20.0,
        ),
        ("total_amount_negative", pickup, dropoff, 1, 2.5, 1, 2, -5.0),
        ("passenger_nonpositive", pickup, dropoff, 0, 2.5, 1, 2, 20.0),
        ("distance_zero", pickup, dropoff, 1, 0.0, 1, 2, 20.0),
        ("passenger_missing", pickup, dropoff, None, 2.5, 1, 2, 20.0),
    ]

    frame = spark.createDataFrame(rows, schema=schema)

    month_start = F.lit("2024-01-01 00:00:00").cast("timestamp_ntz")
    month_end = F.lit("2024-02-01 00:00:00").cast("timestamp_ntz")

    conditions = build_quality_rule_conditions(
        (1, 2),
        month_start,
        month_end,
    )

    assert set(conditions) == QUALITY_RULE_NAMES

    results = {
        row["case"]: row
        for row in frame.select(
            "case",
            *[
                condition.cast("int").alias(rule_name)
                for rule_name, condition in conditions.items()
            ],
        ).collect()
    }

    assert all(results["valid"][rule_name] == 0 for rule_name in QUALITY_RULE_NAMES)

    for rule_name in QUALITY_RULE_NAMES:
        assert results[rule_name][rule_name] == 1


def test_quality_rule_conditions_reject_empty_zones(
    spark: SparkSession,
) -> None:
    """Quality rules require at least one taxi-zone identifier."""
    with pytest.raises(
        ValueError,
        match="zone_ids must contain at least one identifier",
    ):
        build_quality_rule_conditions(
            (),
            F.lit(None),
            F.lit(None),
        )


def test_quality_rule_conditions_reject_duplicate_zones(
    spark: SparkSession,
) -> None:
    """Taxi-zone identifiers should not contain duplicates."""
    with pytest.raises(
        ValueError,
        match="zone_ids must not contain duplicates",
    ):
        build_quality_rule_conditions(
            (1, 1),
            F.lit(None),
            F.lit(None),
        )
