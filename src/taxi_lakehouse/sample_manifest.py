"""Deterministic manifest construction for generated sample artifacts."""

from collections.abc import Sequence
from typing import Any

from taxi_lakehouse.sample_artifacts import GeneratedSampleFile
from taxi_lakehouse.sample_pipeline import (
    GeneratedMonthlySample,
    ResolvedSampleSources,
)
from taxi_lakehouse.sample_specification import (
    SampleSpecification,
)


def generated_file_mapping(
    generated_file: GeneratedSampleFile,
) -> dict[str, Any]:
    """Convert generated-file metadata to a JSON-compatible mapping."""
    return {
        "path": generated_file.file_path.as_posix(),
        "content_length_bytes": (generated_file.content_length_bytes),
        "sha256": generated_file.sha256,
    }


def build_sample_manifest_payload(
    specification: SampleSpecification,
    resolved_sources: ResolvedSampleSources,
    monthly_samples: Sequence[GeneratedMonthlySample],
    taxi_zone_output: GeneratedSampleFile,
) -> dict[str, Any]:
    """Build the deterministic sample manifest payload."""
    if not monthly_samples:
        raise ValueError("monthly_samples must contain at least one generated sample.")

    monthly_samples_by_month = {
        sample.source_month: sample for sample in monthly_samples
    }

    if len(monthly_samples_by_month) != len(monthly_samples):
        raise ValueError("monthly_samples must not contain duplicate source months.")

    expected_months = specification.source_months
    actual_months = tuple(monthly_samples_by_month)

    if set(actual_months) != set(expected_months):
        raise ValueError("Generated sample months do not match the specification.")

    if resolved_sources.taxi_zone_source.sha256 is None:
        raise ValueError("Taxi zone source must contain a SHA-256 checksum.")

    trip_files = []

    for source_month in expected_months:
        sample = monthly_samples_by_month[source_month]

        if sample.source_file.sha256 is None:
            raise ValueError(
                f"Monthly source must contain a SHA-256 checksum: {source_month!r}."
            )

        trip_files.append(
            {
                "source_month": source_month,
                "source": {
                    "path": sample.source_path.as_posix(),
                    "filename": sample.source_file.filename,
                    "sha256": sample.source_file.sha256,
                },
                "output": generated_file_mapping(sample.output_file),
                "row_count": sample.selected_row_count,
                "column_count": sample.source_column_count,
            }
        )

    total_trip_rows = sum(sample.selected_row_count for sample in monthly_samples)

    if total_trip_rows != specification.expected_total_trip_rows:
        raise ValueError(
            "Generated trip row total does not match specification: "
            f"expected={specification.expected_total_trip_rows}, "
            f"actual={total_trip_rows}."
        )

    quotas = {
        quota.quality_bucket: quota.rows_per_month
        for quota in (specification.real_sample_quotas_per_month)
    }

    return {
        "schema_version": "1.0",
        "description": specification.description,
        "source_manifest_path": (specification.source_manifest_path.as_posix()),
        "selection": {
            "algorithm": specification.selection.algorithm,
            "timezone": specification.selection.timezone,
            "partition_keys": list(specification.selection.partition_keys),
            "quality_bucket_priority": list(specification.quality_bucket_priority),
            "real_sample_quotas_per_month": quotas,
            "synthetic_only_buckets": list(specification.synthetic_only_buckets),
        },
        "expected": {
            "rows_per_month": (specification.expected_rows_per_month),
            "total_trip_rows": (specification.expected_total_trip_rows),
            "trip_column_count": (specification.source_profile.trip_column_count),
            "taxi_zone_row_count": (specification.source_profile.taxi_zone_row_count),
        },
        "artifacts": {
            "trip_files": trip_files,
            "taxi_zone_lookup": {
                "source": {
                    "path": (resolved_sources.taxi_zone_path.as_posix()),
                    "filename": (resolved_sources.taxi_zone_source.filename),
                    "sha256": (resolved_sources.taxi_zone_source.sha256),
                },
                "output": generated_file_mapping(taxi_zone_output),
                "row_count": (specification.source_profile.taxi_zone_row_count),
            },
        },
        "totals": {
            "trip_file_count": len(trip_files),
            "trip_row_count": total_trip_rows,
            "artifact_count": len(trip_files) + 1,
        },
    }
