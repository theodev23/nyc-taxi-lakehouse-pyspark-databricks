"""Tests for deterministic sample pipeline orchestration."""

import hashlib
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StructField,
    StructType,
    TimestampNTZType,
)

from taxi_lakehouse.data_acquisition import (
    SourceFile,
    load_source_files,
)
from taxi_lakehouse.sample_pipeline import (
    ResolvedMonthlyTripSource,
    generate_monthly_sample,
    load_taxi_zone_ids,
    resolve_sample_sources,
)
from taxi_lakehouse.sample_specification import (
    SampleQuota,
    SampleSpecification,
    load_sample_specification,
)

PROJECT_SOURCE_MANIFEST_PATH = Path("data/source_manifest.json")
PROJECT_SAMPLE_SPECIFICATION_PATH = Path("data/sample/sample_spec.json")


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """Provide one local Spark session for pipeline tests."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("sample-pipeline-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


def create_local_source_files(
    landing_directory: Path,
) -> tuple[SourceFile, ...]:
    """Create valid local substitutes for project source records."""
    landing_directory.mkdir(parents=True)

    source_files = load_source_files(PROJECT_SOURCE_MANIFEST_PATH)
    local_source_files = []

    for source_file in source_files:
        content = (f"local-test:{source_file.filename}\n").encode()

        file_path = landing_directory / source_file.filename
        file_path.write_bytes(content)

        local_source_files.append(
            replace(
                source_file,
                content_length_bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                downloaded_at_utc=("2026-08-04T09:00:00Z"),
            )
        )

    return tuple(local_source_files)


def test_resolve_sample_sources_preserves_month_order(
    tmp_path: Path,
) -> None:
    """Resolved monthly sources should follow specification order."""
    specification = load_sample_specification(PROJECT_SAMPLE_SPECIFICATION_PATH)
    landing_directory = tmp_path / "landing"

    source_files = create_local_source_files(landing_directory)

    resolved = resolve_sample_sources(
        specification,
        source_files,
        landing_directory,
    )

    assert (
        tuple(source.source_month for source in resolved.monthly_trip_sources)
        == specification.source_months
    )

    assert len(resolved.monthly_trip_sources) == 6
    assert resolved.taxi_zone_source.kind == ("taxi_zone_lookup")
    assert resolved.taxi_zone_path == (landing_directory / "taxi_zone_lookup.csv")


def test_resolve_sample_sources_rejects_missing_month(
    tmp_path: Path,
) -> None:
    """Every month declared by the specification is required."""
    specification = load_sample_specification(PROJECT_SAMPLE_SPECIFICATION_PATH)
    landing_directory = tmp_path / "landing"

    source_files = create_local_source_files(landing_directory)

    incomplete_source_files = tuple(
        source_file
        for source_file in source_files
        if source_file.source_month != "2024-06"
    )

    with pytest.raises(
        ValueError,
        match="Missing Yellow Taxi source months",
    ):
        resolve_sample_sources(
            specification,
            incomplete_source_files,
            landing_directory,
        )


def test_load_taxi_zone_ids_returns_sorted_unique_domain(
    spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Zone identifiers should be returned uniquely and sorted."""
    taxi_zone_path = tmp_path / "taxi_zone_lookup.csv"

    taxi_zone_path.write_text(
        "LocationID,Borough,Zone,service_zone\n"
        "3,Queens,Three,Boro Zone\n"
        "1,EWR,One,EWR\n"
        "2,Brooklyn,Two,Boro Zone\n",
        encoding="utf-8",
    )

    assert load_taxi_zone_ids(
        spark,
        taxi_zone_path,
        expected_row_count=3,
    ) == (
        1,
        2,
        3,
    )


def test_load_taxi_zone_ids_rejects_duplicates(
    spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Duplicate LocationID values should fail explicitly."""
    taxi_zone_path = tmp_path / "taxi_zone_lookup.csv"

    taxi_zone_path.write_text(
        "LocationID,Borough,Zone,service_zone\n"
        "1,EWR,One,EWR\n"
        "1,Queens,Duplicate,Boro Zone\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="LocationID values must be unique",
    ):
        load_taxi_zone_ids(
            spark,
            taxi_zone_path,
            expected_row_count=2,
        )


MONTHLY_SOURCE_SCHEMA = StructType(
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


def build_monthly_test_specification(
    tmp_path: Path,
) -> SampleSpecification:
    """Build a compact specification for monthly generation tests."""
    specification = load_sample_specification(PROJECT_SAMPLE_SPECIFICATION_PATH)

    return replace(
        specification,
        source_profile=replace(
            specification.source_profile,
            trip_column_count=(len(MONTHLY_SOURCE_SCHEMA.fields)),
        ),
        source_months=("2024-01",),
        quality_bucket_priority=(
            "distance_zero",
            "normal",
        ),
        real_sample_quotas_per_month=(
            SampleQuota(
                quality_bucket="distance_zero",
                rows_per_month=1,
            ),
            SampleQuota(
                quality_bucket="normal",
                rows_per_month=2,
            ),
        ),
        synthetic_only_buckets=(),
        expected_rows_per_month=3,
        expected_total_trip_rows=3,
        output=replace(
            specification.output,
            trip_file_pattern=(
                tmp_path / "sample" / "yellow_tripdata_{source_month}.parquet"
            ).as_posix(),
        ),
    )


def write_monthly_source_frame(
    spark: SparkSession,
    source_path: Path,
    rows: list[tuple[object, ...]],
) -> None:
    """Write one synthetic monthly source for pipeline tests."""
    (
        spark.createDataFrame(
            rows,
            schema=MONTHLY_SOURCE_SCHEMA,
        )
        .write.mode("overwrite")
        .parquet(source_path.as_posix())
    )


def build_resolved_monthly_source(
    source_path: Path,
) -> ResolvedMonthlyTripSource:
    """Build a resolved January source around a test Parquet."""
    source_file = next(
        source_file
        for source_file in load_source_files(PROJECT_SOURCE_MANIFEST_PATH)
        if source_file.source_month == "2024-01"
    )

    return ResolvedMonthlyTripSource(
        source_month="2024-01",
        source_file=source_file,
        file_path=source_path,
    )


def test_generate_monthly_sample_writes_expected_artifact(
    spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Monthly generation should preserve schema and meet quotas."""
    source_path = tmp_path / "landing" / "january.parquet"

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
            1,
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
    ]

    write_monthly_source_frame(
        spark,
        source_path,
        rows,
    )

    source_parquet_schema = spark.read.parquet(source_path.as_posix()).schema

    specification = build_monthly_test_specification(tmp_path)

    result = generate_monthly_sample(
        spark,
        build_resolved_monthly_source(source_path),
        specification,
        (1, 2),
        tmp_path / "temporary",
    )

    expected_output_path = tmp_path / "sample" / "yellow_tripdata_2024-01.parquet"

    assert result.source_month == "2024-01"
    assert result.source_column_count == 8
    assert result.selected_row_count == 3
    assert len(result.logical_sha256) == 64
    assert set(result.logical_sha256) <= set("0123456789abcdef")
    assert result.output_file.file_path == (expected_output_path)
    assert expected_output_path.is_file()
    assert len(result.output_file.sha256) == 64

    output_frame = spark.read.parquet(expected_output_path.as_posix())

    assert output_frame.schema == source_parquet_schema
    assert output_frame.count() == 3


def test_generate_monthly_sample_rejects_column_count_mismatch(
    spark: SparkSession,
    tmp_path: Path,
) -> None:
    """The source schema width should match the specification."""
    source_path = tmp_path / "landing" / "january.parquet"

    write_monthly_source_frame(
        spark,
        source_path,
        [
            (
                1,
                datetime(2024, 1, 2, 10, 0),
                datetime(2024, 1, 2, 10, 30),
                1,
                2.5,
                1,
                2,
                20.0,
            )
        ],
    )

    specification = build_monthly_test_specification(tmp_path)
    specification = replace(
        specification,
        source_profile=replace(
            specification.source_profile,
            trip_column_count=9,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Source column count does not match",
    ):
        generate_monthly_sample(
            spark,
            build_resolved_monthly_source(source_path),
            specification,
            (1, 2),
            tmp_path / "temporary",
        )


def test_generate_monthly_sample_rejects_incomplete_quotas(
    spark: SparkSession,
    tmp_path: Path,
) -> None:
    """Generation should fail when source buckets cannot fill quotas."""
    source_path = tmp_path / "landing" / "january.parquet"

    write_monthly_source_frame(
        spark,
        source_path,
        [
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
                10,
                datetime(2024, 1, 3, 10, 0),
                datetime(2024, 1, 3, 10, 30),
                1,
                0.0,
                1,
                2,
                15.0,
            ),
        ],
    )

    specification = build_monthly_test_specification(tmp_path)

    with pytest.raises(
        ValueError,
        match="Selected row count does not match",
    ):
        generate_monthly_sample(
            spark,
            build_resolved_monthly_source(source_path),
            specification,
            (1, 2),
            tmp_path / "temporary",
        )

    assert not (tmp_path / "sample" / "yellow_tripdata_2024-01.parquet").exists()
