"""Tests for explicit Bronze source schemas."""

from pyspark.sql.types import StructType

from taxi_lakehouse.bronze_schemas import (
    TAXI_ZONE_SOURCE_SCHEMA,
    YELLOW_TAXI_SOURCE_SCHEMA,
)


def schema_signature(
    schema: StructType,
) -> tuple[tuple[str, str, bool], ...]:
    """Return the ordered name, type, and nullability contract."""
    return tuple(
        (
            field.name,
            field.dataType.simpleString(),
            field.nullable,
        )
        for field in schema.fields
    )


def test_yellow_taxi_source_schema_matches_source_contract() -> None:
    """The trip schema should preserve all 19 source columns."""
    assert schema_signature(YELLOW_TAXI_SOURCE_SCHEMA) == (
        ("VendorID", "int", True),
        ("tpep_pickup_datetime", "timestamp_ntz", True),
        ("tpep_dropoff_datetime", "timestamp_ntz", True),
        ("passenger_count", "bigint", True),
        ("trip_distance", "double", True),
        ("RatecodeID", "bigint", True),
        ("store_and_fwd_flag", "string", True),
        ("PULocationID", "int", True),
        ("DOLocationID", "int", True),
        ("payment_type", "bigint", True),
        ("fare_amount", "double", True),
        ("extra", "double", True),
        ("mta_tax", "double", True),
        ("tip_amount", "double", True),
        ("tolls_amount", "double", True),
        ("improvement_surcharge", "double", True),
        ("total_amount", "double", True),
        ("congestion_surcharge", "double", True),
        ("Airport_fee", "double", True),
    )


def test_taxi_zone_source_schema_matches_csv_contract() -> None:
    """The zone schema should preserve all four source columns."""
    assert schema_signature(TAXI_ZONE_SOURCE_SCHEMA) == (
        ("LocationID", "int", True),
        ("Borough", "string", True),
        ("Zone", "string", True),
        ("service_zone", "string", True),
    )
