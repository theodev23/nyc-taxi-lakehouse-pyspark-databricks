"""Tests for deterministic sample-generation transformations."""

import hashlib
from datetime import datetime
from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
)

from taxi_lakehouse.sample_generation import (
    build_canonical_row_json_expression,
    build_quality_bucket_expression,
    build_row_hash_expression,
    calculate_logical_sample_sha256,
    select_deterministic_sample_rows,
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


def test_canonical_row_json_preserves_order_and_nulls(
    spark: SparkSession,
) -> None:
    """Canonical JSON should preserve source order and null fields."""
    schema = StructType(
        [
            StructField(
                "trip_id",
                IntegerType(),
                False,
            ),
            StructField(
                "pickup_at",
                TimestampNTZType(),
                False,
            ),
            StructField(
                "note",
                StringType(),
                True,
            ),
        ]
    )

    frame = spark.createDataFrame(
        [
            (
                7,
                datetime(
                    2024,
                    1,
                    2,
                    3,
                    4,
                    5,
                    123456,
                ),
                None,
            )
        ],
        schema=schema,
    )

    canonical_json = frame.select(
        build_canonical_row_json_expression(
            (
                "trip_id",
                "pickup_at",
                "note",
            )
        ).alias("canonical_json")
    ).first()["canonical_json"]

    assert canonical_json == (
        '{"trip_id":7,"pickup_at":"2024-01-02T03:04:05.123456","note":null}'
    )


def test_row_hash_matches_sha256_contract(
    spark: SparkSession,
) -> None:
    """The Spark hash should match the documented SHA-256 input."""
    frame = spark.createDataFrame(
        [
            (
                11,
                "airport",
            )
        ],
        schema=StructType(
            [
                StructField(
                    "trip_id",
                    IntegerType(),
                    False,
                ),
                StructField(
                    "label",
                    StringType(),
                    False,
                ),
            ]
        ),
    )

    canonical_expression = build_canonical_row_json_expression(
        (
            "trip_id",
            "label",
        )
    )

    result = frame.select(
        canonical_expression.alias("canonical_json"),
        build_row_hash_expression(
            "2024-01",
            canonical_expression,
        ).alias("row_hash"),
    ).first()

    expected_hash = hashlib.sha256(
        (f"2024-01\u001f{result['canonical_json']}").encode()
    ).hexdigest()

    assert len(result["row_hash"]) == 64
    assert result["row_hash"] == expected_hash


def test_logical_sample_sha256_is_partition_independent(
    spark: SparkSession,
) -> None:
    """Logical checksums should ignore Spark partitioning."""
    frame = spark.createDataFrame(
        [
            (2, "second"),
            (1, None),
            (3, "third"),
        ],
        (
            "trip_id",
            "label",
        ),
    )

    source_columns = (
        "trip_id",
        "label",
    )

    first_digest = calculate_logical_sample_sha256(
        frame.repartition(3),
        "2024-01",
        source_columns,
    )

    second_digest = calculate_logical_sample_sha256(
        frame.orderBy(
            "trip_id",
            ascending=False,
        ).repartition(2),
        "2024-01",
        source_columns,
    )

    canonical_rows = tuple(
        row["canonical_json"]
        for row in (
            frame.select(
                build_canonical_row_json_expression(source_columns).alias(
                    "canonical_json"
                )
            ).collect()
        )
    )

    ordered_row_hashes = sorted(
        hashlib.sha256((f"2024-01\u001f{canonical_row}").encode()).hexdigest()
        for canonical_row in canonical_rows
    )

    expected_digest = hashlib.sha256(
        "\n".join(ordered_row_hashes).encode("utf-8")
    ).hexdigest()

    assert first_digest == expected_digest
    assert second_digest == expected_digest
    assert len(first_digest) == 64


def test_canonical_row_json_rejects_duplicate_columns(
    spark: SparkSession,
) -> None:
    """Duplicate source columns should fail before execution."""
    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        build_canonical_row_json_expression(
            (
                "trip_id",
                "trip_id",
            )
        )

    assert spark.version


def test_deterministic_selection_respects_quotas_and_schema(
    spark: SparkSession,
) -> None:
    """Selection should be stable and preserve the source schema."""
    specification = load_sample_specification(PROJECT_SPECIFICATION_PATH)

    schema = StructType(
        [
            StructField("trip_id", IntegerType(), False),
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
            1,
            datetime(2024, 1, 2, 10, 0),
            datetime(2024, 1, 2, 10, 30),
            1,
            2.5,
            1,
            2,
            20.0,
        ),
        (
            2,
            datetime(2024, 1, 3, 10, 0),
            datetime(2024, 1, 3, 10, 30),
            1,
            3.5,
            1,
            2,
            25.0,
        ),
        (
            3,
            datetime(2024, 1, 4, 10, 0),
            datetime(2024, 1, 4, 10, 30),
            2,
            4.5,
            1,
            2,
            30.0,
        ),
        (
            10,
            datetime(2024, 1, 5, 10, 0),
            datetime(2024, 1, 5, 10, 30),
            1,
            0.0,
            1,
            2,
            15.0,
        ),
        (
            11,
            datetime(2024, 1, 6, 10, 0),
            datetime(2024, 1, 6, 10, 30),
            1,
            0.0,
            1,
            2,
            16.0,
        ),
        (
            20,
            datetime(2024, 1, 7, 10, 0),
            datetime(2024, 1, 7, 10, 30),
            None,
            2.0,
            1,
            2,
            18.0,
        ),
        (
            21,
            datetime(2024, 1, 8, 10, 0),
            datetime(2024, 1, 8, 10, 30),
            None,
            2.1,
            1,
            2,
            19.0,
        ),
        (
            30,
            datetime(2024, 1, 9, 10, 0),
            datetime(2024, 1, 9, 10, 30),
            1,
            2.0,
            1,
            2,
            -4.0,
        ),
        (
            31,
            datetime(2024, 1, 10, 10, 0),
            datetime(2024, 1, 10, 10, 30),
            1,
            2.0,
            1,
            2,
            -5.0,
        ),
    ]

    frame = spark.createDataFrame(
        rows,
        schema=schema,
    )

    source_columns = tuple(frame.columns)
    quota_by_bucket = {
        "normal": 2,
        "distance_zero": 1,
        "passenger_missing": 1,
        "total_amount_negative": 1,
    }

    first_selection = select_deterministic_sample_rows(
        frame.repartition(1),
        "2024-01",
        source_columns,
        (1, 2),
        specification.quality_bucket_priority,
        quota_by_bucket,
    )

    second_selection = select_deterministic_sample_rows(
        frame.repartition(3),
        "2024-01",
        source_columns,
        (1, 2),
        specification.quality_bucket_priority,
        quota_by_bucket,
    )

    first_trip_ids = sorted(row["trip_id"] for row in first_selection.collect())

    second_trip_ids = sorted(row["trip_id"] for row in second_selection.collect())

    assert first_trip_ids == second_trip_ids
    assert len(first_trip_ids) == 5
    assert first_selection.columns == list(source_columns)
    assert first_selection.schema == frame.schema

    selected_bucket_counts = {
        row["quality_bucket"]: row["count"]
        for row in (
            first_selection.withColumn(
                "quality_bucket",
                build_quality_bucket_expression(
                    "2024-01",
                    (1, 2),
                    specification.quality_bucket_priority,
                ),
            )
            .groupBy("quality_bucket")
            .count()
            .collect()
        )
    }

    assert selected_bucket_counts == {
        "normal": 2,
        "distance_zero": 1,
        "passenger_missing": 1,
        "total_amount_negative": 1,
    }


def test_deterministic_selection_rejects_missing_source_column(
    spark: SparkSession,
) -> None:
    """Missing canonical source columns should fail explicitly."""
    with pytest.raises(
        ValueError,
        match="Source columns are missing",
    ):
        select_deterministic_sample_rows(
            spark.range(1),
            "2024-01",
            ("missing_column",),
            (1,),
            ("normal",),
            {"normal": 1},
        )


def test_deterministic_selection_rejects_nonpositive_quota(
    spark: SparkSession,
) -> None:
    """Sample bucket quotas should be strictly positive."""
    with pytest.raises(
        ValueError,
        match="Sample quotas must be positive integers",
    ):
        select_deterministic_sample_rows(
            spark.range(1),
            "2024-01",
            ("id",),
            (1,),
            ("normal",),
            {"normal": 0},
        )
