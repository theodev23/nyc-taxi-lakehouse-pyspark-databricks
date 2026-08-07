"""Integration tests for Silver loading from Bronze Delta tables."""

from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    IntegerType,
    StructField,
    StructType,
)

from taxi_lakehouse.silver_loading import (
    load_bronze_trip_month,
    load_bronze_zone_ids,
)
from taxi_lakehouse.spark_session import (
    build_local_spark_session,
)


@pytest.fixture(scope="module")
def delta_spark() -> SparkSession:
    """Create one Delta-enabled Spark session for loading tests."""
    session = build_local_spark_session(
        "silver-loading-integration-tests",
        enable_delta=True,
    )
    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


def test_load_bronze_trip_month_filters_requested_partition(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Only rows from the requested Bronze month should be loaded."""
    source_path = tmp_path / "yellow_taxi_trips"

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

    january = load_bronze_trip_month(
        delta_spark,
        source_path,
        "2024-01",
    )

    rows = [
        (row["trip_id"], row["_source_month"])
        for row in (january.orderBy("trip_id").collect())
    ]

    assert rows == [
        (1, "2024-01"),
        (2, "2024-01"),
    ]


def test_load_bronze_trip_month_rejects_invalid_month(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Malformed source months should fail before Delta loading."""
    source_path = tmp_path / "missing_table"

    with pytest.raises(
        ValueError,
        match="Invalid source month",
    ):
        load_bronze_trip_month(
            delta_spark,
            source_path,
            "2024-13",
        )


def test_load_bronze_trip_month_requires_partition_column(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Bronze trips must expose the source-month lineage column."""
    source_path = tmp_path / "trips_without_month"

    frame = delta_spark.createDataFrame(
        [(1,), (2,)],
        "trip_id long",
    )
    frame.write.format("delta").save(source_path.as_posix())

    with pytest.raises(
        ValueError,
        match="must contain column '_source_month'",
    ):
        load_bronze_trip_month(
            delta_spark,
            source_path,
            "2024-01",
        )


def test_load_bronze_zone_ids_returns_sorted_unique_domain(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Zone loading should remove nulls and duplicates deterministically."""
    source_path = tmp_path / "taxi_zones"

    schema = StructType(
        [
            StructField(
                "LocationID",
                IntegerType(),
                True,
            ),
        ]
    )

    frame = delta_spark.createDataFrame(
        [
            (3,),
            (1,),
            (2,),
            (3,),
            (None,),
        ],
        schema=schema,
    )
    frame.write.format("delta").save(source_path.as_posix())

    assert load_bronze_zone_ids(
        delta_spark,
        source_path,
    ) == (
        1,
        2,
        3,
    )


def test_load_bronze_zone_ids_requires_location_id(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Bronze taxi zones must expose their LocationID column."""
    source_path = tmp_path / "zones_without_id"

    frame = delta_spark.createDataFrame(
        [("Zone A",)],
        "Zone string",
    )
    frame.write.format("delta").save(source_path.as_posix())

    with pytest.raises(
        ValueError,
        match="must contain column 'LocationID'",
    ):
        load_bronze_zone_ids(
            delta_spark,
            source_path,
        )


def test_load_bronze_zone_ids_rejects_empty_domain(
    delta_spark: SparkSession,
    tmp_path: Path,
) -> None:
    """A Bronze taxi-zone table must provide at least one valid ID."""
    source_path = tmp_path / "zones_without_valid_id"

    schema = StructType(
        [
            StructField(
                "LocationID",
                IntegerType(),
                True,
            ),
        ]
    )

    frame = delta_spark.createDataFrame(
        [(None,)],
        schema=schema,
    )
    frame.write.format("delta").save(source_path.as_posix())

    with pytest.raises(
        ValueError,
        match="must contain at least one LocationID",
    ):
        load_bronze_zone_ids(
            delta_spark,
            source_path,
        )
