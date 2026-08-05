"""End-to-end orchestration for deterministic sample generation."""

from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import SparkSession

from taxi_lakehouse.data_acquisition import load_source_files
from taxi_lakehouse.sample_artifacts import (
    GeneratedSampleFile,
    copy_sample_file_atomically,
    write_json_artifact_atomically,
)
from taxi_lakehouse.sample_manifest import (
    build_sample_manifest_payload,
)
from taxi_lakehouse.sample_pipeline import (
    GeneratedMonthlySample,
    ResolvedSampleSources,
    generate_monthly_sample,
    load_taxi_zone_ids,
    resolve_sample_sources,
)
from taxi_lakehouse.sample_specification import (
    SampleSpecification,
    load_sample_specification,
)


@dataclass(frozen=True, slots=True)
class GeneratedSampleDataset:
    """Metadata for one complete deterministic sample dataset."""

    specification: SampleSpecification
    resolved_sources: ResolvedSampleSources
    monthly_samples: tuple[GeneratedMonthlySample, ...]
    taxi_zone_output: GeneratedSampleFile
    manifest_output: GeneratedSampleFile


def generate_sample_dataset(
    spark: SparkSession,
    specification_path: Path,
    landing_directory: Path,
    temporary_root: Path,
) -> GeneratedSampleDataset:
    """Generate every deterministic sample artifact."""
    specification = load_sample_specification(specification_path)

    source_files = load_source_files(specification.source_manifest_path)

    resolved_sources = resolve_sample_sources(
        specification,
        source_files,
        landing_directory,
    )

    zone_ids = load_taxi_zone_ids(
        spark,
        resolved_sources.taxi_zone_path,
        expected_row_count=(specification.source_profile.taxi_zone_row_count),
    )

    monthly_samples = tuple(
        generate_monthly_sample(
            spark,
            monthly_source,
            specification,
            zone_ids,
            temporary_root,
        )
        for monthly_source in (resolved_sources.monthly_trip_sources)
    )

    taxi_zone_output = copy_sample_file_atomically(
        resolved_sources.taxi_zone_path,
        specification.output.taxi_zone_lookup_path,
    )

    manifest_payload = build_sample_manifest_payload(
        specification,
        resolved_sources,
        monthly_samples,
        taxi_zone_output,
    )

    manifest_output = write_json_artifact_atomically(
        manifest_payload,
        specification.output.sample_manifest_path,
    )

    return GeneratedSampleDataset(
        specification=specification,
        resolved_sources=resolved_sources,
        monthly_samples=monthly_samples,
        taxi_zone_output=taxi_zone_output,
        manifest_output=manifest_output,
    )
