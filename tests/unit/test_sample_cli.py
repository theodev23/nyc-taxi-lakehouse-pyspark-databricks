"""Tests for the deterministic sample generation CLI."""

from pathlib import Path

import pytest

import taxi_lakehouse.sample_cli as sample_cli
from taxi_lakehouse.data_acquisition import (
    load_source_files,
)
from taxi_lakehouse.sample_artifacts import (
    GeneratedSampleFile,
)
from taxi_lakehouse.sample_orchestration import (
    GeneratedSampleDataset,
)
from taxi_lakehouse.sample_pipeline import (
    GeneratedMonthlySample,
    ResolvedMonthlyTripSource,
    ResolvedSampleSources,
)
from taxi_lakehouse.sample_specification import (
    load_sample_specification,
)

SOURCE_MANIFEST_PATH = Path("data/source_manifest.json")
SAMPLE_SPECIFICATION_PATH = Path("data/sample/sample_spec.json")


class FakeSparkSession:
    """Minimal Spark substitute recording stop calls."""

    def __init__(self) -> None:
        self.stop_count = 0

    def stop(self) -> None:
        """Record one Spark shutdown."""
        self.stop_count += 1


def build_generated_dataset() -> GeneratedSampleDataset:
    """Build deterministic generated metadata for CLI tests."""
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

    monthly_sources = tuple(
        ResolvedMonthlyTripSource(
            source_month=source_month,
            source_file=trip_sources_by_month[source_month],
            file_path=(
                Path("data/landing") / trip_sources_by_month[source_month].filename
            ),
        )
        for source_month in specification.source_months
    )

    zone_source = next(
        source_file
        for source_file in source_files
        if source_file.kind == "taxi_zone_lookup"
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
            selected_row_count=250,
            output_file=GeneratedSampleFile(
                file_path=Path(
                    specification.output.trip_file_pattern.format(
                        source_month=(monthly_source.source_month)
                    )
                ),
                content_length_bytes=1_000,
                sha256=(monthly_source.source_month.replace("-", "").ljust(64, "0")),
            ),
        )
        for monthly_source in monthly_sources
    )

    return GeneratedSampleDataset(
        specification=specification,
        resolved_sources=resolved_sources,
        monthly_samples=monthly_samples,
        taxi_zone_output=GeneratedSampleFile(
            file_path=(specification.output.taxi_zone_lookup_path),
            content_length_bytes=12_331,
            sha256="c" * 64,
        ),
        manifest_output=GeneratedSampleFile(
            file_path=(specification.output.sample_manifest_path),
            content_length_bytes=10_000,
            sha256="d" * 64,
        ),
    )


def test_main_uses_defaults_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should use defaults and summarize all artifacts."""
    spark = FakeSparkSession()
    result = build_generated_dataset()
    captured_arguments: dict[str, object] = {}

    def fake_generate_sample_dataset(
        spark_argument: object,
        specification_path: Path,
        landing_directory: Path,
        temporary_root: Path,
    ) -> GeneratedSampleDataset:
        captured_arguments.update(
            {
                "spark": spark_argument,
                "specification_path": (specification_path),
                "landing_directory": (landing_directory),
                "temporary_root": temporary_root,
            }
        )
        return result

    monkeypatch.setattr(
        sample_cli,
        "build_spark_session",
        lambda: spark,
    )
    monkeypatch.setattr(
        sample_cli,
        "generate_sample_dataset",
        fake_generate_sample_dataset,
    )

    exit_code = sample_cli.main([])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert spark.stop_count == 1
    assert captured_arguments == {
        "spark": spark,
        "specification_path": (sample_cli.DEFAULT_SPECIFICATION_PATH),
        "landing_directory": (sample_cli.DEFAULT_LANDING_DIRECTORY),
        "temporary_root": (sample_cli.DEFAULT_TEMPORARY_ROOT),
    }

    assert output.count("TRIP_SAMPLE\tmonth=") == 6
    assert "ZONE_SAMPLE\trows=265" in output
    assert "SAMPLE_MANIFEST\tpath=data/sample/sample_manifest.json" in output
    assert (
        "SAMPLE_GENERATION_COMPLETE "
        "trip_files=6 "
        "trip_rows=1500 "
        "data_artifacts=7" in output
    )


def test_main_forwards_path_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Explicit path arguments should reach orchestration."""
    spark = FakeSparkSession()
    result = build_generated_dataset()
    captured_paths: tuple[Path, Path, Path] | None = None

    specification_path = tmp_path / "custom-specification.json"
    landing_directory = tmp_path / "custom-landing"
    temporary_root = tmp_path / "custom-temporary"

    def fake_generate_sample_dataset(
        spark_argument: object,
        received_specification_path: Path,
        received_landing_directory: Path,
        received_temporary_root: Path,
    ) -> GeneratedSampleDataset:
        nonlocal captured_paths

        assert spark_argument is spark

        captured_paths = (
            received_specification_path,
            received_landing_directory,
            received_temporary_root,
        )
        return result

    monkeypatch.setattr(
        sample_cli,
        "build_spark_session",
        lambda: spark,
    )
    monkeypatch.setattr(
        sample_cli,
        "generate_sample_dataset",
        fake_generate_sample_dataset,
    )

    exit_code = sample_cli.main(
        [
            "--specification",
            str(specification_path),
            "--landing-dir",
            str(landing_directory),
            "--temporary-root",
            str(temporary_root),
        ]
    )

    assert exit_code == 0
    assert spark.stop_count == 1
    assert captured_paths == (
        specification_path,
        landing_directory,
        temporary_root,
    )


def test_main_stops_spark_when_generation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spark should stop while generation errors remain visible."""
    spark = FakeSparkSession()

    def fail_generation(
        spark_argument: object,
        specification_path: Path,
        landing_directory: Path,
        temporary_root: Path,
    ) -> GeneratedSampleDataset:
        raise RuntimeError("synthetic generation failure")

    monkeypatch.setattr(
        sample_cli,
        "build_spark_session",
        lambda: spark,
    )
    monkeypatch.setattr(
        sample_cli,
        "generate_sample_dataset",
        fail_generation,
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic generation failure",
    ):
        sample_cli.main([])

    assert spark.stop_count == 1
