"""Tests for deterministic sample profiling."""

from datetime import datetime

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

from taxi_lakehouse.sample_profiling import (
    build_sample_profile_payload,
)

PROFILE_TEST_SCHEMA = StructType(
    [
        StructField(
            "VendorID",
            IntegerType(),
            True,
        ),
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
            "RatecodeID",
            LongType(),
            True,
        ),
        StructField(
            "store_and_fwd_flag",
            StringType(),
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
            "payment_type",
            LongType(),
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
        StructField(
            "_quality_bucket",
            StringType(),
            False,
        ),
    ]
)

SOURCE_COLUMNS = (
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "total_amount",
)


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Provide one local Spark session for profiling tests."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("sample-profiling-tests")
        .config("spark.ui.enabled", "false")
        .config(
            "spark.sql.session.timeZone",
            "UTC",
        )
        .getOrCreate()
    )

    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


def test_build_sample_profile_payload_records_metrics(
    spark: SparkSession,
) -> None:
    """The payload should contain deterministic profile metrics."""
    frame = spark.createDataFrame(
        [
            (
                1,
                datetime(2024, 1, 1, 8, 0),
                datetime(2024, 1, 1, 8, 15),
                1,
                1.5,
                1,
                "N",
                1,
                2,
                1,
                12.5,
                "2024-01",
                "normal",
            ),
            (
                2,
                datetime(2024, 1, 2, 9, 0),
                datetime(2024, 1, 2, 9, 5),
                None,
                0.0,
                None,
                None,
                3,
                4,
                2,
                0.0,
                "2024-01",
                "distance_zero",
            ),
            (
                1,
                datetime(2024, 2, 1, 10, 0),
                datetime(2024, 2, 1, 10, 30),
                0,
                2.0,
                2,
                "Y",
                3,
                5,
                1,
                -5.0,
                "2024-02",
                "total_amount_negative",
            ),
            (
                2,
                datetime(2024, 2, 2, 11, 0),
                datetime(2024, 2, 2, 11, 45),
                7,
                3.0,
                6,
                "N",
                4,
                6,
                4,
                30.0,
                "2024-02",
                "normal",
            ),
        ],
        schema=PROFILE_TEST_SCHEMA,
    )

    payload = build_sample_profile_payload(
        frame,
        SOURCE_COLUMNS,
    )

    assert payload["schema_version"] == "1.0"
    assert payload["summary"] == {
        "row_count": 4,
        "column_count": 11,
        "source_month_count": 2,
        "quality_bucket_count": 3,
    }

    assert payload["rows_by_source_month"] == {
        "2024-01": 2,
        "2024-02": 2,
    }

    assert payload["quality_bucket_counts"] == {
        "distance_zero": 1,
        "normal": 2,
        "total_amount_negative": 1,
    }

    assert payload["null_statistics"]["passenger_count"] == {
        "null_count": 1,
        "non_null_count": 3,
        "null_rate": 0.25,
    }

    assert payload["distinct_counts_including_null"] == {
        "VendorID": 2,
        "RatecodeID": 4,
        "store_and_fwd_flag": 3,
        "payment_type": 3,
        "PULocationID": 3,
        "DOLocationID": 4,
    }

    assert payload["temporal_ranges"]["tpep_pickup_datetime"] == {
        "minimum": "2024-01-01T08:00:00.000000",
        "maximum": "2024-02-02T11:00:00.000000",
    }

    assert payload["numeric_ranges"] == {
        "passenger_count": {
            "minimum": 0,
            "maximum": 7,
        },
        "trip_distance": {
            "minimum": 0.0,
            "maximum": 3.0,
        },
        "total_amount": {
            "minimum": -5.0,
            "maximum": 30.0,
        },
    }

    assert payload["schema"][1] == {
        "name": "tpep_pickup_datetime",
        "data_type": "timestamp_ntz",
        "nullable": True,
    }


def test_build_sample_profile_payload_rejects_missing_columns(
    spark: SparkSession,
) -> None:
    """All configured profile columns should be available."""
    frame = spark.createDataFrame(
        [
            (
                1,
                "2024-01",
                "normal",
            )
        ],
        (
            "VendorID",
            "_source_month",
            "_quality_bucket",
        ),
    )

    with pytest.raises(
        ValueError,
        match="Required profile columns are absent",
    ):
        build_sample_profile_payload(
            frame,
            ("VendorID",),
        )
