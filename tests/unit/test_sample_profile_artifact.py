"""Tests for deterministic sample profile artifact generation."""

import json
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

from taxi_lakehouse.sample_artifacts import (
    write_single_parquet_file,
)
from taxi_lakehouse.sample_profile_artifact import (
    generate_sample_profile_artifact,
)
from taxi_lakehouse.sample_specification import (
    SampleSpecification,
    load_sample_specification,
)

PROJECT_SPECIFICATION_PATH = Path("data/sample/sample_spec.json")

PROFILE_ARTIFACT_SCHEMA = StructType(
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
    ]
)


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Provide one local Spark session for artifact tests."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("sample-profile-artifact-tests")
        .config("spark.ui.enabled", "false")
        .config(
            "spark.sql.session.timeZone",
            "UTC",
        )
        .config(
            "spark.sql.shuffle.partitions",
            "1",
        )
        .getOrCreate()
    )

    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


def build_test_specification(
    tmp_path: Path,
) -> tuple[SampleSpecification, Path]:
    """Create a valid small specification for profile tests."""
    sample_directory = tmp_path / "sample"
    specification_path = tmp_path / "sample_spec.json"

    specification_mapping = json.loads(
        PROJECT_SPECIFICATION_PATH.read_text(encoding="utf-8")
    )

    specification_mapping["source_months"] = [
        "2024-01",
        "2024-02",
    ]

    specification_mapping["quality_bucket_priority"] = [
        "distance_negative",
        "passenger_missing",
        "normal",
    ]

    specification_mapping["real_sample_quotas_per_month"] = {
        "passenger_missing": 1,
        "normal": 1,
    }

    specification_mapping["synthetic_only_buckets"] = [
        "distance_negative",
    ]

    specification_mapping["expected_rows_per_month"] = 2

    specification_mapping["expected_total_trip_rows"] = 4

    specification_mapping["source_profile"]["trip_column_count"] = 11

    specification_mapping["source_profile"]["taxi_zone_row_count"] = 2

    specification_mapping["output"]["trip_file_pattern"] = (
        sample_directory / "yellow_tripdata_{source_month}.parquet"
    ).as_posix()

    specification_mapping["output"]["taxi_zone_lookup_path"] = (
        sample_directory / "taxi_zone_lookup.csv"
    ).as_posix()

    specification_mapping["output"]["sample_manifest_path"] = (
        sample_directory / "sample_manifest.json"
    ).as_posix()

    specification_path.write_text(
        json.dumps(
            specification_mapping,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    specification = load_sample_specification(specification_path)

    return specification, specification_path


def write_test_artifacts(
    spark: SparkSession,
    specification: SampleSpecification,
    *,
    include_february: bool = True,
    february_row_count: int = 2,
) -> None:
    """Write small Parquet and taxi-zone fixtures."""
    zone_path = specification.output.taxi_zone_lookup_path

    zone_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    zone_path.write_text(
        (
            "LocationID,Borough,Zone,service_zone\n"
            "1,Manhattan,First,Yellow Zone\n"
            "2,Manhattan,Second,Yellow Zone\n"
        ),
        encoding="utf-8",
    )

    rows_by_month = {
        "2024-01": [
            (
                1,
                datetime(2024, 1, 2, 8, 0),
                datetime(2024, 1, 2, 8, 15),
                1,
                1.5,
                1,
                "N",
                1,
                2,
                1,
                12.5,
            ),
            (
                2,
                datetime(2024, 1, 3, 9, 0),
                datetime(2024, 1, 3, 9, 10),
                None,
                2.0,
                None,
                None,
                1,
                2,
                2,
                10.0,
            ),
        ],
        "2024-02": [
            (
                1,
                datetime(2024, 2, 2, 10, 0),
                datetime(2024, 2, 2, 10, 20),
                1,
                2.0,
                1,
                "N",
                1,
                2,
                1,
                20.0,
            ),
            (
                2,
                datetime(2024, 2, 3, 11, 0),
                datetime(2024, 2, 3, 11, 30),
                None,
                3.0,
                None,
                None,
                2,
                1,
                2,
                15.0,
            ),
        ],
    }

    months = specification.source_months if include_february else ("2024-01",)

    for source_month in months:
        output_path = Path(
            specification.output.trip_file_pattern.format(source_month=source_month)
        )

        rows = rows_by_month[source_month]

        if source_month == "2024-02":
            rows = rows[:february_row_count]

        frame = spark.createDataFrame(
            rows,
            schema=PROFILE_ARTIFACT_SCHEMA,
        )

        write_single_parquet_file(
            frame=frame,
            destination_path=output_path,
            temporary_root=(output_path.parent / ".generation_tmp"),
        )


def test_generate_sample_profile_artifact_writes_metrics(
    spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Generated sample files should produce one JSON profile."""
    (
        specification,
        specification_path,
    ) = build_test_specification(tmp_path)

    write_test_artifacts(
        spark,
        specification,
    )

    output_path = tmp_path / "sample" / "sample_profile.json"

    result = generate_sample_profile_artifact(
        spark,
        specification_path,
        output_path,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert result.file_path == output_path
    assert result.content_length_bytes > 0
    assert len(result.sha256) == 64

    assert payload["schema_version"] == "1.0"
    assert payload["summary"] == {
        "row_count": 4,
        "column_count": 11,
        "source_month_count": 2,
        "quality_bucket_count": 2,
    }

    assert payload["rows_by_source_month"] == {
        "2024-01": 2,
        "2024-02": 2,
    }

    assert payload["quality_bucket_counts"] == {
        "normal": 2,
        "passenger_missing": 2,
    }

    assert payload["sample_specification_path"] == specification_path.as_posix()

    assert payload["sample_manifest_path"] == (
        specification.output.sample_manifest_path.as_posix()
    )


def test_generate_sample_profile_artifact_rejects_missing_month(
    spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Every configured monthly sample should exist."""
    (
        specification,
        specification_path,
    ) = build_test_specification(tmp_path)

    write_test_artifacts(
        spark,
        specification,
        include_february=False,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Monthly sample file does not exist",
    ):
        generate_sample_profile_artifact(
            spark,
            specification_path,
            tmp_path / "sample_profile.json",
        )


def test_generate_sample_profile_artifact_rejects_wrong_total(
    spark: SparkSession,
    tmp_path: Path,
) -> None:
    """The profile row count should match the specification."""
    (
        specification,
        specification_path,
    ) = build_test_specification(tmp_path)

    write_test_artifacts(
        spark,
        specification,
        february_row_count=1,
    )

    with pytest.raises(
        ValueError,
        match="Profiled sample row count",
    ):
        generate_sample_profile_artifact(
            spark,
            specification_path,
            tmp_path / "sample_profile.json",
        )
