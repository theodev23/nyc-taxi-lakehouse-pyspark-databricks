"""Tests for the Bronze Delta ingestion command-line interface."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

import taxi_lakehouse.bronze_cli as bronze_cli
from taxi_lakehouse.bronze_orchestration import (
    BronzeIngestionResult,
    BronzeTripWriteResult,
)
from taxi_lakehouse.bronze_sources import (
    ResolvedBronzeSources,
    ResolvedBronzeTripSource,
)
from taxi_lakehouse.data_acquisition import load_source_files

PROJECT_MANIFEST_PATH = Path("data/source_manifest.json")
INGESTION_TIMESTAMP = datetime(
    2026,
    8,
    6,
    8,
    30,
    tzinfo=UTC,
)


class FakeSparkContext:
    """Record Spark log-level configuration."""

    def __init__(self) -> None:
        self.log_levels: list[str] = []

    def setLogLevel(self, log_level: str) -> None:
        """Record one requested Spark log level."""
        self.log_levels.append(log_level)


class FakeSparkSession:
    """Provide the Spark operations required by the CLI."""

    def __init__(self) -> None:
        self.sparkContext = FakeSparkContext()
        self.stop_count = 0

    def stop(self) -> None:
        """Record one Spark shutdown."""
        self.stop_count += 1


def build_ingestion_result(
    bronze_root: Path = Path("data/lakehouse/bronze"),
) -> BronzeIngestionResult:
    """Build representative Bronze ingestion metadata."""
    source_files = load_source_files(PROJECT_MANIFEST_PATH)

    trip_sources_by_month = {
        source_file.source_month: source_file
        for source_file in source_files
        if (
            source_file.kind == "yellow_taxi_trip_data"
            and source_file.source_month is not None
        )
    }

    monthly_sources = tuple(
        ResolvedBronzeTripSource(
            source_month=source_month,
            source_file=trip_sources_by_month[source_month],
            file_path=(
                Path("data/landing") / trip_sources_by_month[source_month].filename
            ),
        )
        for source_month in (
            "2024-01",
            "2024-02",
        )
    )

    taxi_zone_source = next(
        source_file
        for source_file in source_files
        if source_file.kind == "taxi_zone_lookup"
    )

    resolved_sources = ResolvedBronzeSources(
        monthly_trip_sources=monthly_sources,
        taxi_zone_source=taxi_zone_source,
        taxi_zone_path=Path("data/landing/taxi_zone_lookup.csv"),
    )

    monthly_trip_writes = tuple(
        BronzeTripWriteResult(
            source_month=monthly_source.source_month,
            source_path=monthly_source.file_path,
            destination_path=bronze_root / "yellow_taxi_trips",
            row_count=row_count,
        )
        for monthly_source, row_count in zip(
            monthly_sources,
            (101, 202),
            strict=True,
        )
    )

    return BronzeIngestionResult(
        ingested_at_utc=INGESTION_TIMESTAMP,
        resolved_sources=resolved_sources,
        monthly_trip_writes=monthly_trip_writes,
        taxi_zone_destination_path=bronze_root / "taxi_zones",
        taxi_zone_row_count=265,
    )


def test_build_spark_session_enables_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Bronze CLI should request a Delta-enabled Spark session."""
    spark = FakeSparkSession()
    captured_arguments: dict[str, object] = {}

    def fake_build_local_spark_session(
        app_name: str,
        *,
        enable_delta: bool = False,
    ) -> FakeSparkSession:
        captured_arguments.update(
            {
                "app_name": app_name,
                "enable_delta": enable_delta,
            }
        )
        return spark

    monkeypatch.setattr(
        bronze_cli,
        "build_local_spark_session",
        fake_build_local_spark_session,
    )

    result = bronze_cli.build_spark_session()

    assert result is spark
    assert captured_arguments == {
        "app_name": "nyc-taxi-bronze-ingestion",
        "enable_delta": True,
    }


def test_main_uses_defaults_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default command should ingest and summarize every output."""
    spark = FakeSparkSession()
    result = build_ingestion_result()
    captured_arguments: dict[str, object] = {}

    def fake_ingest_bronze_dataset(
        spark_argument: object,
        manifest_path: Path,
        landing_directory: Path,
        bronze_root: Path,
        ingested_at_utc: datetime,
    ) -> BronzeIngestionResult:
        captured_arguments.update(
            {
                "spark": spark_argument,
                "manifest_path": manifest_path,
                "landing_directory": landing_directory,
                "bronze_root": bronze_root,
                "ingested_at_utc": ingested_at_utc,
            }
        )
        return result

    monkeypatch.setattr(
        bronze_cli,
        "build_spark_session",
        lambda: spark,
    )
    monkeypatch.setattr(
        bronze_cli,
        "current_utc_datetime",
        lambda: INGESTION_TIMESTAMP,
    )
    monkeypatch.setattr(
        bronze_cli,
        "ingest_bronze_dataset",
        fake_ingest_bronze_dataset,
    )

    exit_code = bronze_cli.main([])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert spark.sparkContext.log_levels == ["ERROR"]
    assert spark.stop_count == 1
    assert captured_arguments == {
        "spark": spark,
        "manifest_path": bronze_cli.DEFAULT_MANIFEST_PATH,
        "landing_directory": bronze_cli.DEFAULT_LANDING_DIRECTORY,
        "bronze_root": bronze_cli.DEFAULT_BRONZE_ROOT,
        "ingested_at_utc": INGESTION_TIMESTAMP,
    }

    assert output.count("BRONZE_TRIP\tmonth=") == 2
    assert "BRONZE_TRIP\tmonth=2024-01\trows=101" in output
    assert "BRONZE_TRIP\tmonth=2024-02\trows=202" in output
    assert (
        "BRONZE_TAXI_ZONES\trows=265\tdestination=data/lakehouse/bronze/taxi_zones"
    ) in output
    assert (
        "BRONZE_INGESTION_COMPLETE "
        "trip_files=2 "
        "trip_rows=303 "
        "taxi_zone_rows=265 "
        "ingested_at_utc=2026-08-06T08:30:00Z"
    ) in output


def test_main_forwards_path_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Explicit CLI paths should reach Bronze orchestration."""
    spark = FakeSparkSession()
    manifest_path = tmp_path / "custom-manifest.json"
    landing_directory = tmp_path / "custom-landing"
    bronze_root = tmp_path / "custom-bronze"
    received_paths: tuple[Path, Path, Path] | None = None

    def fake_ingest_bronze_dataset(
        spark_argument: object,
        received_manifest_path: Path,
        received_landing_directory: Path,
        received_bronze_root: Path,
        ingested_at_utc: datetime,
    ) -> BronzeIngestionResult:
        nonlocal received_paths

        assert spark_argument is spark
        assert ingested_at_utc == INGESTION_TIMESTAMP

        received_paths = (
            received_manifest_path,
            received_landing_directory,
            received_bronze_root,
        )

        return build_ingestion_result(received_bronze_root)

    monkeypatch.setattr(
        bronze_cli,
        "build_spark_session",
        lambda: spark,
    )
    monkeypatch.setattr(
        bronze_cli,
        "current_utc_datetime",
        lambda: INGESTION_TIMESTAMP,
    )
    monkeypatch.setattr(
        bronze_cli,
        "ingest_bronze_dataset",
        fake_ingest_bronze_dataset,
    )

    exit_code = bronze_cli.main(
        [
            "--manifest",
            str(manifest_path),
            "--landing-dir",
            str(landing_directory),
            "--bronze-root",
            str(bronze_root),
        ]
    )

    assert exit_code == 0
    assert received_paths == (
        manifest_path,
        landing_directory,
        bronze_root,
    )
    assert spark.stop_count == 1


def test_main_stops_spark_when_ingestion_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spark should stop while ingestion errors remain visible."""
    spark = FakeSparkSession()

    def fail_ingestion(
        spark_argument: object,
        manifest_path: Path,
        landing_directory: Path,
        bronze_root: Path,
        ingested_at_utc: datetime,
    ) -> BronzeIngestionResult:
        assert spark_argument is spark
        raise RuntimeError("synthetic Bronze ingestion failure")

    monkeypatch.setattr(
        bronze_cli,
        "build_spark_session",
        lambda: spark,
    )
    monkeypatch.setattr(
        bronze_cli,
        "current_utc_datetime",
        lambda: INGESTION_TIMESTAMP,
    )
    monkeypatch.setattr(
        bronze_cli,
        "ingest_bronze_dataset",
        fail_ingestion,
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic Bronze ingestion failure",
    ):
        bronze_cli.main([])

    assert spark.sparkContext.log_levels == ["ERROR"]
    assert spark.stop_count == 1
