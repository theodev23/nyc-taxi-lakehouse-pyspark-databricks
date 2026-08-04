"""Tests for deterministic sample manifest construction."""

from dataclasses import replace
from pathlib import Path

import pytest

from taxi_lakehouse.data_acquisition import (
    load_source_files,
)
from taxi_lakehouse.sample_artifacts import (
    GeneratedSampleFile,
)
from taxi_lakehouse.sample_manifest import (
    build_sample_manifest_payload,
)
from taxi_lakehouse.sample_pipeline import (
    GeneratedMonthlySample,
    ResolvedMonthlyTripSource,
    ResolvedSampleSources,
)
from taxi_lakehouse.sample_specification import (
    SampleQuota,
    load_sample_specification,
)

SOURCE_MANIFEST_PATH = Path("data/source_manifest.json")
SAMPLE_SPECIFICATION_PATH = Path("data/sample/sample_spec.json")


def build_manifest_fixture() -> tuple[
    object,
    ResolvedSampleSources,
    tuple[GeneratedMonthlySample, ...],
    GeneratedSampleFile,
]:
    """Build a compact deterministic manifest fixture."""
    specification = load_sample_specification(SAMPLE_SPECIFICATION_PATH)

    specification = replace(
        specification,
        source_months=(
            "2024-01",
            "2024-02",
        ),
        real_sample_quotas_per_month=(
            SampleQuota(
                quality_bucket="normal",
                rows_per_month=2,
            ),
        ),
        quality_bucket_priority=("normal",),
        synthetic_only_buckets=(),
        expected_rows_per_month=2,
        expected_total_trip_rows=4,
    )

    source_files = load_source_files(SOURCE_MANIFEST_PATH)

    january_source = next(
        source_file
        for source_file in source_files
        if source_file.source_month == "2024-01"
    )
    february_source = next(
        source_file
        for source_file in source_files
        if source_file.source_month == "2024-02"
    )
    zone_source = next(
        source_file
        for source_file in source_files
        if source_file.kind == "taxi_zone_lookup"
    )

    monthly_sources = (
        ResolvedMonthlyTripSource(
            source_month="2024-01",
            source_file=january_source,
            file_path=Path("data/landing/" + january_source.filename),
        ),
        ResolvedMonthlyTripSource(
            source_month="2024-02",
            source_file=february_source,
            file_path=Path("data/landing/" + february_source.filename),
        ),
    )

    resolved_sources = ResolvedSampleSources(
        monthly_trip_sources=monthly_sources,
        taxi_zone_source=zone_source,
        taxi_zone_path=Path("data/landing/taxi_zone_lookup.csv"),
    )

    monthly_samples = tuple(
        GeneratedMonthlySample(
            source_month=monthly_source.source_month,
            source_file=monthly_source.source_file,
            source_path=monthly_source.file_path,
            source_column_count=19,
            selected_row_count=2,
            output_file=GeneratedSampleFile(
                file_path=Path(
                    f"data/sample/yellow_tripdata_{monthly_source.source_month}.parquet"
                ),
                content_length_bytes=1_000,
                sha256=(
                    "a" * 64 if monthly_source.source_month == "2024-01" else "b" * 64
                ),
            ),
        )
        for monthly_source in monthly_sources
    )

    taxi_zone_output = GeneratedSampleFile(
        file_path=Path("data/sample/taxi_zone_lookup.csv"),
        content_length_bytes=12_331,
        sha256="c" * 64,
    )

    return (
        specification,
        resolved_sources,
        monthly_samples,
        taxi_zone_output,
    )


def test_build_sample_manifest_payload_records_artifacts() -> None:
    """The payload should record sources, outputs and totals."""
    (
        specification,
        resolved_sources,
        monthly_samples,
        taxi_zone_output,
    ) = build_manifest_fixture()

    payload = build_sample_manifest_payload(
        specification,
        resolved_sources,
        monthly_samples,
        taxi_zone_output,
    )

    assert payload["schema_version"] == "1.0"
    assert payload["selection"]["algorithm"] == ("sha256_row_rank_v1")
    assert payload["selection"]["real_sample_quotas_per_month"] == {
        "normal": 2,
    }

    trip_files = payload["artifacts"]["trip_files"]

    assert [trip_file["source_month"] for trip_file in trip_files] == [
        "2024-01",
        "2024-02",
    ]

    assert trip_files[0]["row_count"] == 2
    assert trip_files[0]["column_count"] == 19
    assert trip_files[0]["output"]["sha256"] == ("a" * 64)

    assert payload["artifacts"]["taxi_zone_lookup"]["output"]["sha256"] == ("c" * 64)

    assert payload["totals"] == {
        "trip_file_count": 2,
        "trip_row_count": 4,
        "artifact_count": 3,
    }


def test_build_sample_manifest_payload_preserves_month_order() -> None:
    """Manifest trip files should follow specification order."""
    (
        specification,
        resolved_sources,
        monthly_samples,
        taxi_zone_output,
    ) = build_manifest_fixture()

    payload = build_sample_manifest_payload(
        specification,
        resolved_sources,
        tuple(reversed(monthly_samples)),
        taxi_zone_output,
    )

    assert [
        trip_file["source_month"] for trip_file in payload["artifacts"]["trip_files"]
    ] == [
        "2024-01",
        "2024-02",
    ]


def test_build_sample_manifest_payload_rejects_missing_month() -> None:
    """Every configured source month should be represented."""
    (
        specification,
        resolved_sources,
        monthly_samples,
        taxi_zone_output,
    ) = build_manifest_fixture()

    with pytest.raises(
        ValueError,
        match="do not match the specification",
    ):
        build_sample_manifest_payload(
            specification,
            resolved_sources,
            monthly_samples[:1],
            taxi_zone_output,
        )


def test_build_sample_manifest_payload_rejects_wrong_total() -> None:
    """Generated row totals should match the declared target."""
    (
        specification,
        resolved_sources,
        monthly_samples,
        taxi_zone_output,
    ) = build_manifest_fixture()

    invalid_samples = (
        replace(
            monthly_samples[0],
            selected_row_count=1,
        ),
        monthly_samples[1],
    )

    with pytest.raises(
        ValueError,
        match="row total does not match",
    ):
        build_sample_manifest_payload(
            specification,
            resolved_sources,
            invalid_samples,
            taxi_zone_output,
        )
