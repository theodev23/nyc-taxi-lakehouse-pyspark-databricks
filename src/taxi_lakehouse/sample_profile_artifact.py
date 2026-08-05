"""Generation of deterministic sample profile artifacts."""

from functools import reduce
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from taxi_lakehouse.sample_artifacts import (
    GeneratedSampleFile,
    write_json_artifact_atomically,
)
from taxi_lakehouse.sample_generation import (
    build_quality_bucket_expression,
)
from taxi_lakehouse.sample_pipeline import (
    load_taxi_zone_ids,
)
from taxi_lakehouse.sample_profiling import (
    build_sample_profile_payload,
)
from taxi_lakehouse.sample_specification import (
    load_sample_specification,
)


def generate_sample_profile_artifact(
    spark: SparkSession,
    specification_path: Path,
    output_path: Path,
) -> GeneratedSampleFile:
    """Profile generated sample files and write one JSON artifact."""
    specification = load_sample_specification(specification_path)

    zone_ids = load_taxi_zone_ids(
        spark,
        specification.output.taxi_zone_lookup_path,
        expected_row_count=(specification.source_profile.taxi_zone_row_count),
    )

    enriched_frames: list[DataFrame] = []
    source_columns: tuple[str, ...] | None = None
    source_schema = None

    for source_month in specification.source_months:
        sample_path = Path(
            specification.output.trip_file_pattern.format(source_month=source_month)
        )

        if not sample_path.is_file():
            raise FileNotFoundError(
                f"Monthly sample file does not exist: {sample_path}."
            )

        monthly_frame = spark.read.parquet(sample_path.as_posix())

        monthly_columns = tuple(monthly_frame.columns)

        if len(monthly_columns) != specification.source_profile.trip_column_count:
            raise ValueError(
                "Monthly sample column count does not "
                "match the specification: "
                f"month={source_month!r}, "
                "expected="
                f"{specification.source_profile.trip_column_count}, "
                f"actual={len(monthly_columns)}."
            )

        if source_columns is None:
            source_columns = monthly_columns
            source_schema = monthly_frame.schema

        elif monthly_columns != source_columns or monthly_frame.schema != source_schema:
            raise ValueError(
                f"Monthly sample schemas are inconsistent: {source_month!r}."
            )

        enriched_frames.append(
            monthly_frame.withColumn(
                "_source_month",
                F.lit(source_month),
            ).withColumn(
                "_quality_bucket",
                build_quality_bucket_expression(
                    source_month,
                    zone_ids,
                    specification.quality_bucket_priority,
                ),
            )
        )

    if not enriched_frames or source_columns is None:
        raise ValueError(
            "The sample specification must contain at least one source month."
        )

    sample_frame = reduce(
        DataFrame.unionByName,
        enriched_frames,
    ).persist()

    try:
        profile_payload = build_sample_profile_payload(
            sample_frame,
            source_columns,
        )

        actual_row_count = profile_payload["summary"]["row_count"]

        expected_row_count = specification.expected_total_trip_rows

        if actual_row_count != expected_row_count:
            raise ValueError(
                "Profiled sample row count does not "
                "match the specification: "
                f"expected={expected_row_count}, "
                f"actual={actual_row_count}."
            )

        profile_payload["sample_specification_path"] = specification_path.as_posix()

        profile_payload["sample_manifest_path"] = (
            specification.output.sample_manifest_path.as_posix()
        )

        return write_json_artifact_atomically(
            profile_payload,
            output_path,
        )
    finally:
        sample_frame.unpersist()
