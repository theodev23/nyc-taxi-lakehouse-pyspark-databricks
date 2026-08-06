"""Explicit source schemas used by the Bronze ingestion layer."""

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
)

YELLOW_TAXI_SOURCE_SCHEMA = StructType(
    [
        StructField("VendorID", IntegerType(), nullable=True),
        StructField(
            "tpep_pickup_datetime",
            TimestampNTZType(),
            nullable=True,
        ),
        StructField(
            "tpep_dropoff_datetime",
            TimestampNTZType(),
            nullable=True,
        ),
        StructField("passenger_count", LongType(), nullable=True),
        StructField("trip_distance", DoubleType(), nullable=True),
        StructField("RatecodeID", LongType(), nullable=True),
        StructField("store_and_fwd_flag", StringType(), nullable=True),
        StructField("PULocationID", IntegerType(), nullable=True),
        StructField("DOLocationID", IntegerType(), nullable=True),
        StructField("payment_type", LongType(), nullable=True),
        StructField("fare_amount", DoubleType(), nullable=True),
        StructField("extra", DoubleType(), nullable=True),
        StructField("mta_tax", DoubleType(), nullable=True),
        StructField("tip_amount", DoubleType(), nullable=True),
        StructField("tolls_amount", DoubleType(), nullable=True),
        StructField(
            "improvement_surcharge",
            DoubleType(),
            nullable=True,
        ),
        StructField("total_amount", DoubleType(), nullable=True),
        StructField(
            "congestion_surcharge",
            DoubleType(),
            nullable=True,
        ),
        StructField("Airport_fee", DoubleType(), nullable=True),
    ]
)

TAXI_ZONE_SOURCE_SCHEMA = StructType(
    [
        StructField("LocationID", IntegerType(), nullable=True),
        StructField("Borough", StringType(), nullable=True),
        StructField("Zone", StringType(), nullable=True),
        StructField("service_zone", StringType(), nullable=True),
    ]
)
