"""Tests for deterministic sample dataset orchestration."""

from pathlib import Path

import pytest

import taxi_lakehouse.sample_orchestration as orchestration
from taxi_lakehouse.data_acquisition import (
    SourceFile,
    load_source_files,
)
from taxi_lakehouse.sample_artifacts import (
    GeneratedSampleFile,
)
from taxi_lakehouse.sample_pipeline import (
    GeneratedMonthlySample,
    ResolvedMonthlyTripSource,
    ResolvedSampleSources,
)
from taxi_lakehouse.sample_specification import (
    SampleSpecification,
    load_sample_specification,
)

SOURCE_MANIFEST_PATH = Path("data/source_manifest.json")
SAMPLE_SPECIFICATION_PATH = Path("data/sample/sample_spec.json")


def build_orchestration_fixture() -> tuple[
    SampleSpecification,
    tuple[SourceFile, ...],
    ResolvedSampleSources,
]:
    """Build source metadata for orchestration tests."""
    specification = load_sample_specification(SAMPLE_SPECIFICATION_PATH)
    source_files = load_source_files(SOURCE_MANIFEST_PATH)

    trip_sources_by_month = {
        source_file.source_month: source_file
        for source_file in source_files
        if (
            source_file.kind == "yellow_taxi_trip_data"
            and source_file.source_month is not None
        )
    }

    monthly_trip_sources = tuple(
        ResolvedMonthlyTripSource(
            source_month=source_month,
            source_file=trip_sources_by_month[source_month],
            file_path=(
                Path("data/landing") / trip_sources_by_month[source_month].filename
            ),
        )
        for source_month in specification.source_months
    )

    taxi_zone_source = next(
        source_file
        for source_file in source_files
        if source_file.kind == "taxi_zone_lookup"
    )

    resolved_sources = ResolvedSampleSources(
        monthly_trip_sources=monthly_trip_sources,
        taxi_zone_source=taxi_zone_source,
        taxi_zone_path=Path("data/landing/taxi_zone_lookup.csv"),
    )

    return (
        specification,
        source_files,
        resolved_sources,
    )


def test_generate_sample_dataset_orchestrates_all_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The orchestrator should generate and record every artifact."""
    (
        specification,
        source_files,
        resolved_sources,
    ) = build_orchestration_fixture()

    spark = object()
    specification_path = tmp_path / "sample_spec.json"
    landing_directory = tmp_path / "landing"
    temporary_root = tmp_path / "temporary"
    call_sequence: list[str] = []

    def fake_load_sample_specification(
        path: Path,
    ) -> SampleSpecification:
        call_sequence.append("load_specification")
        assert path == specification_path
        return specification

    def fake_load_source_files(
        path: Path,
    ) -> tuple[SourceFile, ...]:
        call_sequence.append("load_source_files")
        assert path == specification.source_manifest_path
        return source_files

    def fake_resolve_sample_sources(
        specification_argument: SampleSpecification,
        source_files_argument: tuple[SourceFile, ...],
        landing_directory_argument: Path,
    ) -> ResolvedSampleSources:
        call_sequence.append("resolve_sources")
        assert specification_argument is specification
        assert source_files_argument is source_files
        assert landing_directory_argument == (landing_directory)
        return resolved_sources

    def fake_load_taxi_zone_ids(
        spark_argument: object,
        taxi_zone_path: Path,
        expected_row_count: int,
    ) -> tuple[int, ...]:
        call_sequence.append("load_zone_ids")
        assert spark_argument is spark
        assert taxi_zone_path == (resolved_sources.taxi_zone_path)
        assert expected_row_count == 265
        return (1, 2, 3)

    def fake_generate_monthly_sample(
        spark_argument: object,
        monthly_source: ResolvedMonthlyTripSource,
        specification_argument: SampleSpecification,
        zone_ids: tuple[int, ...],
        temporary_root_argument: Path,
    ) -> GeneratedMonthlySample:
        call_sequence.append(f"generate:{monthly_source.source_month}")
        assert spark_argument is spark
        assert specification_argument is specification
        assert zone_ids == (1, 2, 3)
        assert temporary_root_argument == temporary_root

        return GeneratedMonthlySample(
            source_month=monthly_source.source_month,
            source_file=monthly_source.source_file,
            source_path=monthly_source.file_path,
            source_column_count=19,
            selected_row_count=250,
            output_file=GeneratedSampleFile(
                file_path=Path(
                    specification.output.trip_file_pattern.format(
                        source_month=(monthly_source.source_month)
                    )
                ),
                content_length_bytes=1_000,
                sha256=monthly_source.source_month.replace(
                    "-",
                    "",
                ).ljust(64, "0"),
            ),
        )

    def fake_copy_sample_file_atomically(
        source_path: Path,
        destination_path: Path,
    ) -> GeneratedSampleFile:
        call_sequence.append("copy_zone")
        assert source_path == (resolved_sources.taxi_zone_path)
        assert destination_path == (specification.output.taxi_zone_lookup_path)

        return GeneratedSampleFile(
            file_path=destination_path,
            content_length_bytes=12_331,
            sha256="c" * 64,
        )

    manifest_payload = {
        "schema_version": "1.0",
    }

    def fake_build_sample_manifest_payload(
        specification_argument: SampleSpecification,
        resolved_sources_argument: ResolvedSampleSources,
        monthly_samples: tuple[
            GeneratedMonthlySample,
            ...,
        ],
        taxi_zone_output: GeneratedSampleFile,
    ) -> dict[str, str]:
        call_sequence.append("build_manifest")
        assert specification_argument is specification
        assert resolved_sources_argument is (resolved_sources)
        assert len(monthly_samples) == 6
        assert taxi_zone_output.sha256 == "c" * 64
        return manifest_payload

    def fake_write_json_artifact_atomically(
        payload: dict[str, str],
        destination_path: Path,
    ) -> GeneratedSampleFile:
        call_sequence.append("write_manifest")
        assert payload is manifest_payload
        assert destination_path == (specification.output.sample_manifest_path)

        return GeneratedSampleFile(
            file_path=destination_path,
            content_length_bytes=10_000,
            sha256="d" * 64,
        )

    monkeypatch.setattr(
        orchestration,
        "load_sample_specification",
        fake_load_sample_specification,
    )
    monkeypatch.setattr(
        orchestration,
        "load_source_files",
        fake_load_source_files,
    )
    monkeypatch.setattr(
        orchestration,
        "resolve_sample_sources",
        fake_resolve_sample_sources,
    )
    monkeypatch.setattr(
        orchestration,
        "load_taxi_zone_ids",
        fake_load_taxi_zone_ids,
    )
    monkeypatch.setattr(
        orchestration,
        "generate_monthly_sample",
        fake_generate_monthly_sample,
    )
    monkeypatch.setattr(
        orchestration,
        "copy_sample_file_atomically",
        fake_copy_sample_file_atomically,
    )
    monkeypatch.setattr(
        orchestration,
        "build_sample_manifest_payload",
        fake_build_sample_manifest_payload,
    )
    monkeypatch.setattr(
        orchestration,
        "write_json_artifact_atomically",
        fake_write_json_artifact_atomically,
    )

    result = orchestration.generate_sample_dataset(
        spark,
        specification_path,
        landing_directory,
        temporary_root,
    )

    assert result.specification is specification
    assert result.resolved_sources is resolved_sources
    assert len(result.monthly_samples) == 6
    assert result.taxi_zone_output.sha256 == "c" * 64
    assert result.manifest_output.sha256 == "d" * 64

    assert call_sequence == [
        "load_specification",
        "load_source_files",
        "resolve_sources",
        "load_zone_ids",
        "generate:2024-01",
        "generate:2024-02",
        "generate:2024-03",
        "generate:2024-04",
        "generate:2024-05",
        "generate:2024-06",
        "copy_zone",
        "build_manifest",
        "write_manifest",
    ]


def test_generate_sample_dataset_stops_before_metadata_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A monthly failure should prevent zone and manifest writes."""
    (
        specification,
        source_files,
        resolved_sources,
    ) = build_orchestration_fixture()

    generated_months: list[str] = []

    monkeypatch.setattr(
        orchestration,
        "load_sample_specification",
        lambda path: specification,
    )
    monkeypatch.setattr(
        orchestration,
        "load_source_files",
        lambda path: source_files,
    )
    monkeypatch.setattr(
        orchestration,
        "resolve_sample_sources",
        lambda specification_argument, source_files_argument, landing_directory: (
            resolved_sources
        ),
    )
    monkeypatch.setattr(
        orchestration,
        "load_taxi_zone_ids",
        lambda spark, taxi_zone_path, expected_row_count: (1, 2, 3),
    )

    def fail_during_march(
        spark: object,
        monthly_source: ResolvedMonthlyTripSource,
        specification_argument: SampleSpecification,
        zone_ids: tuple[int, ...],
        temporary_root: Path,
    ) -> GeneratedMonthlySample:
        generated_months.append(monthly_source.source_month)

        if monthly_source.source_month == "2024-03":
            raise RuntimeError("synthetic monthly generation failure")

        return GeneratedMonthlySample(
            source_month=monthly_source.source_month,
            source_file=monthly_source.source_file,
            source_path=monthly_source.file_path,
            source_column_count=19,
            selected_row_count=250,
            output_file=GeneratedSampleFile(
                file_path=tmp_path / (monthly_source.source_month + ".parquet"),
                content_length_bytes=1,
                sha256="a" * 64,
            ),
        )

    def unexpected_metadata_write(
        *args: object,
        **kwargs: object,
    ) -> None:
        raise AssertionError("Metadata artifacts must not be written.")

    monkeypatch.setattr(
        orchestration,
        "generate_monthly_sample",
        fail_during_march,
    )
    monkeypatch.setattr(
        orchestration,
        "copy_sample_file_atomically",
        unexpected_metadata_write,
    )
    monkeypatch.setattr(
        orchestration,
        "build_sample_manifest_payload",
        unexpected_metadata_write,
    )
    monkeypatch.setattr(
        orchestration,
        "write_json_artifact_atomically",
        unexpected_metadata_write,
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic monthly generation failure",
    ):
        orchestration.generate_sample_dataset(
            object(),
            tmp_path / "sample_spec.json",
            tmp_path / "landing",
            tmp_path / "temporary",
        )

    assert generated_months == [
        "2024-01",
        "2024-02",
        "2024-03",
    ]
