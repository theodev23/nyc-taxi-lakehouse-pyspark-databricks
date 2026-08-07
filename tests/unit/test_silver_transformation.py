"""Tests for Silver trip data-quality transformations."""

from datetime import datetime
from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
)

from taxi_lakehouse.silver_quality_specification import (
    load_silver_quality_specification,
)
from taxi_lakehouse.silver_transformation import (
    annotate_silver_quality,
    split_silver_quality_rows,
)

PROJECT_SPECIFICATION_PATH = Path("data/silver_quality_spec.json")


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Provide one local Spark session for Silver transformations."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("silver-transformation-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


@pytest.fixture(scope="module")
def annotated_frame(spark: SparkSession):
    """Build representative annotated Silver rows."""
    schema = StructType(
        [
            StructField("trip_id", LongType(), False),
            StructField(
                "tpep_pickup_datetime",
                TimestampNTZType(),
                True,
            ),
            StructField(
                "tpep_dropoff_datetime",
                TimestampNTZType(),
                True,
            ),
            StructField(
                "passenger_count",
                LongType(),
                True,
            ),
            StructField(
                "trip_distance",
                DoubleType(),
                True,
            ),
            StructField(
                "PULocationID",
                IntegerType(),
                True,
            ),
            StructField(
                "DOLocationID",
                IntegerType(),
                True,
            ),
            StructField(
                "total_amount",
                DoubleType(),
                True,
            ),
            StructField(
                "_source_month",
                StringType(),
                False,
            ),
        ]
    )

    rows = [
        (
            1,
            datetime(2024, 1, 2, 10, 0),
            datetime(2024, 1, 2, 10, 30),
            1,
            2.5,
            1,
            2,
            20.0,
            "2024-01",
        ),
        (
            2,
            datetime(2023, 12, 31, 23, 0),
            datetime(2023, 12, 31, 23, 0),
            7,
            0.0,
            1,
            2,
            -5.0,
            "2024-01",
        ),
        (
            3,
            datetime(2024, 1, 2, 11, 0),
            datetime(2024, 1, 2, 11, 30),
            None,
            0.0,
            1,
            2,
            0.0,
            "2024-01",
        ),
        (
            4,
            datetime(2024, 1, 2, 12, 0),
            datetime(2024, 1, 3, 13, 0),
            1,
            -1.0,
            None,
            999,
            20.0,
            "2024-01",
        ),
    ]

    specification = load_silver_quality_specification(PROJECT_SPECIFICATION_PATH)

    return annotate_silver_quality(
        spark.createDataFrame(
            rows,
            schema=schema,
        ),
        "2024-01",
        (1, 2),
        specification,
    )


def test_annotate_silver_quality_records_all_matching_rules(
    annotated_frame,
) -> None:
    """Silver annotations should preserve every matching rule."""
    rows = {row["trip_id"]: row for row in annotated_frame.orderBy("trip_id").collect()}

    assert rows[1]["_rejection_reasons"] == []
    assert rows[1]["_quality_flags"] == []

    assert rows[2]["_rejection_reasons"] == [
        "duration_nonpositive",
        "pickup_outside_month",
        "passenger_over_6",
    ]
    assert rows[2]["_quality_flags"] == [
        "total_amount_negative",
        "distance_zero",
    ]

    assert rows[3]["_rejection_reasons"] == []
    assert rows[3]["_quality_flags"] == [
        "total_amount_zero",
        "distance_zero",
        "passenger_missing",
    ]

    assert rows[4]["_rejection_reasons"] == [
        "pickup_zone_unknown",
        "dropoff_zone_unknown",
        "distance_negative",
        "duration_over_24h",
    ]
    assert rows[4]["_quality_flags"] == []

    assert "_source_month" in annotated_frame.columns
    assert "_rejection_reasons" in annotated_frame.columns
    assert "_quality_flags" in annotated_frame.columns


def test_split_silver_quality_rows(
    annotated_frame,
) -> None:
    """Rows with rejection reasons should be isolated from accepted rows."""
    specification = load_silver_quality_specification(PROJECT_SPECIFICATION_PATH)

    split = split_silver_quality_rows(
        annotated_frame,
        specification,
    )

    accepted_ids = [
        row["trip_id"]
        for row in split.accepted.orderBy("trip_id").select("trip_id").collect()
    ]
    rejected_ids = [
        row["trip_id"]
        for row in split.rejected.orderBy("trip_id").select("trip_id").collect()
    ]

    assert accepted_ids == [1, 3]
    assert rejected_ids == [2, 4]

    assert split.accepted.count() + split.rejected.count() == 4
