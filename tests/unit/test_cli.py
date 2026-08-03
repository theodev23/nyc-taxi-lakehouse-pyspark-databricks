"""Tests for the source acquisition command-line interface."""

from dataclasses import replace
from pathlib import Path

import pytest

from taxi_lakehouse.cli import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DESTINATION_DIR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_TIMEOUT_SECONDS,
    main,
)
from taxi_lakehouse.data_acquisition import (
    AcquisitionResult,
    load_source_files,
)

PROJECT_MANIFEST_PATH = Path("data/source_manifest.json")


def test_main_uses_defaults_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The CLI should use project defaults and summarize actions."""
    base_source_file = load_source_files(PROJECT_MANIFEST_PATH)[0]

    downloaded_source = replace(
        base_source_file,
        filename="downloaded.parquet",
        sha256="a" * 64,
        downloaded_at_utc="2026-08-03T08:55:00Z",
    )
    metadata_source = replace(
        base_source_file,
        filename="metadata.parquet",
        sha256="b" * 64,
        downloaded_at_utc="2026-08-03T08:55:00Z",
    )
    reused_source = replace(
        base_source_file,
        filename="reused.parquet",
        sha256="c" * 64,
        downloaded_at_utc="2026-08-03T08:00:00Z",
    )

    results = (
        AcquisitionResult(
            source_file=downloaded_source,
            file_path=tmp_path / downloaded_source.filename,
            action="downloaded",
        ),
        AcquisitionResult(
            source_file=metadata_source,
            file_path=tmp_path / metadata_source.filename,
            action="metadata_recorded",
        ),
        AcquisitionResult(
            source_file=reused_source,
            file_path=tmp_path / reused_source.filename,
            action="reused",
        ),
    )

    captured_arguments: dict[str, object] = {}

    def fake_acquire_source_files(
        manifest_path: Path,
        destination_dir: Path,
        downloaded_at_utc: str,
        chunk_size: int,
        timeout_seconds: float,
    ) -> tuple[AcquisitionResult, ...]:
        captured_arguments.update(
            {
                "manifest_path": manifest_path,
                "destination_dir": destination_dir,
                "downloaded_at_utc": downloaded_at_utc,
                "chunk_size": chunk_size,
                "timeout_seconds": timeout_seconds,
            }
        )
        return results

    monkeypatch.setattr(
        "taxi_lakehouse.cli.acquire_source_files",
        fake_acquire_source_files,
    )
    monkeypatch.setattr(
        "taxi_lakehouse.cli.current_utc_timestamp",
        lambda: "2026-08-03T08:55:00Z",
    )

    exit_code = main([])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert captured_arguments == {
        "manifest_path": DEFAULT_MANIFEST_PATH,
        "destination_dir": DEFAULT_DESTINATION_DIR,
        "downloaded_at_utc": "2026-08-03T08:55:00Z",
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
    }
    assert (f"downloaded\tdownloaded.parquet\tsha256={'a' * 64}") in output
    assert (f"metadata_recorded\tmetadata.parquet\tsha256={'b' * 64}") in output
    assert (f"reused\treused.parquet\tsha256={'c' * 64}") in output
    assert (
        "ACQUISITION_COMPLETE total=3 downloaded=1 metadata_recorded=1 reused=1"
    ) in output


def test_main_forwards_cli_overrides(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Explicit CLI arguments should reach batch acquisition."""
    manifest_path = tmp_path / "custom-manifest.json"
    destination_dir = tmp_path / "custom-landing"

    captured_arguments: dict[str, object] = {}

    def fake_acquire_source_files(
        received_manifest_path: Path,
        received_destination_dir: Path,
        downloaded_at_utc: str,
        chunk_size: int,
        timeout_seconds: float,
    ) -> tuple[AcquisitionResult, ...]:
        captured_arguments.update(
            {
                "manifest_path": received_manifest_path,
                "destination_dir": received_destination_dir,
                "downloaded_at_utc": downloaded_at_utc,
                "chunk_size": chunk_size,
                "timeout_seconds": timeout_seconds,
            }
        )
        return ()

    monkeypatch.setattr(
        "taxi_lakehouse.cli.acquire_source_files",
        fake_acquire_source_files,
    )
    monkeypatch.setattr(
        "taxi_lakehouse.cli.current_utc_timestamp",
        lambda: "2026-08-03T09:00:00Z",
    )

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--destination-dir",
            str(destination_dir),
            "--chunk-size",
            "4096",
            "--timeout-seconds",
            "12.5",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert captured_arguments == {
        "manifest_path": manifest_path,
        "destination_dir": destination_dir,
        "downloaded_at_utc": "2026-08-03T09:00:00Z",
        "chunk_size": 4096,
        "timeout_seconds": 12.5,
    }
    assert (
        "ACQUISITION_COMPLETE total=0 downloaded=0 metadata_recorded=0 reused=0"
    ) in output
