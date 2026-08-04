"""Tests for deterministic sample-generation transformations."""

from datetime import datetime
from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StructField,
    StructType,
    TimestampNTZType,
)

from taxi_lakehouse.sample_generation import (
    build_quality_bucket_expression,
    source_month_bounds,
)
from taxi_lakehouse.sample_specification import (
    load_sample_specification,
)

PROJECT_SPECIFICATION_PATH = Path("data/sample/sample_spec.json")


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Provide one local Spark session for transformation tests."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("sample-generation-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


def test_source_month_bounds() -> None:
    """Month bounds should use an exclusive next-month boundary."""
    assert source_month_bounds("2024-01") == (
        datetime(2024, 1, 1),
        datetime(2024, 2, 1),
    )

    assert source_month_bounds("2024-12") == (
        datetime(2024, 12, 1),
        datetime(2025, 1, 1),
    )


def test_source_month_bounds_rejects_invalid_month() -> None:
    """Malformed source months should fail explicitly."""
    with pytest.raises(
        ValueError,
        match="Invalid source month",
    ):
        source_month_bounds("2024-13")


def test_quality_bucket_expression_respects_priority(
    spark: SparkSession,
) -> None:
    """The first matching quality rule should own the row."""
    specification = load_sample_specification(PROJECT_SPECIFICATION_PATH)

    schema = StructType(
        [
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
        ]
    )

    rows = [
        (
            datetime(2024, 1, 2, 10, 0),
            datetime(2024, 1, 2, 10, 30),
            1,
            2.5,
            1,
            2,
            20.0,
        ),
        (
            datetime(2024, 1, 2, 10, 0),
            datetime(2024, 1, 2, 10, 30),
            None,
            0.0,
            1,
            2,
            20.0,
        ),
        (
            datetime(2024, 1, 2, 10, 0),
            datetime(2024, 1, 2, 10, 30),
            1,
            -1.0,
            1,
            2,
            20.0,
        ),
        (
            datetime(2024, 1, 2, 10, 0),
            datetime(2024, 1, 2, 10, 30),
            1,
            2.5,
            999,
            2,
            20.0,
        ),
        (
            datetime(2023, 12, 31, 23, 0),
            datetime(2023, 12, 31, 23, 30),
            0,
            0.0,
            1,
            2,
            -5.0,
        ),
    ]

    frame = spark.createDataFrame(
        rows,
        schema=schema,
    )

    bucket_expression = build_quality_bucket_expression(
        "2024-01",
        (1, 2),
        specification.quality_bucket_priority,
    )

    buckets = [
        row["quality_bucket"]
        for row in (frame.select(bucket_expression.alias("quality_bucket")).collect())
    ]

    assert buckets == [
        "normal",
        "distance_zero",
        "distance_negative",
        "pickup_zone_unknown",
        "pickup_outside_month",
    ]


def test_quality_bucket_expression_rejects_unknown_bucket() -> None:
    """An unsupported bucket should fail before Spark execution."""
    with pytest.raises(
        ValueError,
        match="Unsupported quality buckets",
    ):
        build_quality_bucket_expression(
            "2024-01",
            (1, 2),
            ("unknown_bucket", "normal"),
        )


def test_quality_bucket_expression_rejects_empty_zones() -> None:
    """The quality transformation requires a zone domain."""
    with pytest.raises(
        ValueError,
        match="zone_ids must contain",
    ):
        build_quality_bucket_expression(
            "2024-01",
            (),
            ("normal",),
        )


def test_quality_bucket_expression_handles_normal_only(
    spark: SparkSession,
) -> None:
    """A normal-only priority should classify every row as normal."""
    result = (
        spark.range(1)
        .select(
            build_quality_bucket_expression(
                "2024-01",
                (1,),
                ("normal",),
            ).alias("quality_bucket")
        )
        .select(F.col("quality_bucket"))
        .first()
    )

    assert result["quality_bucket"] == "normal"
