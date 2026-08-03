"""Tests for NYC TLC source acquisition utilities."""

import json
from pathlib import Path

import pytest

from taxi_lakehouse.data_acquisition import (
    SourceFile,
    calculate_sha256,
    load_source_files,
)

PROJECT_MANIFEST_PATH = Path("data/source_manifest.json")


def test_load_source_files_from_project_manifest() -> None:
    """The project manifest should load into immutable source records."""
    source_files = load_source_files(PROJECT_MANIFEST_PATH)

    assert len(source_files) == 7
    assert isinstance(source_files, tuple)
    assert all(isinstance(source_file, SourceFile) for source_file in source_files)

    assert source_files[0].filename == "yellow_tripdata_2024-01.parquet"
    assert source_files[0].source_month == "2024-01"
    assert source_files[-1].filename == "taxi_zone_lookup.csv"
    assert source_files[-1].source_month is None


def test_source_file_rejects_directory_in_filename() -> None:
    """A source filename should not control its destination directory."""
    value = json.loads(PROJECT_MANIFEST_PATH.read_text(encoding="utf-8"))["files"][0]
    value["filename"] = "../unexpected.parquet"

    with pytest.raises(
        ValueError,
        match="must not contain directories",
    ):
        SourceFile.from_mapping(value)


def test_load_source_files_rejects_duplicate_filenames(
    tmp_path: Path,
) -> None:
    """Duplicate filenames should fail before downloading any data."""
    manifest = json.loads(PROJECT_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["files"].append(manifest["files"][0].copy())

    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="duplicate filenames",
    ):
        load_source_files(manifest_path)


def test_load_source_files_rejects_unknown_schema(
    tmp_path: Path,
) -> None:
    """An unsupported manifest schema should fail explicitly."""
    manifest = json.loads(PROJECT_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["schema_version"] = "2.0"

    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported source manifest schema version",
    ):
        load_source_files(manifest_path)


def test_calculate_sha256_reads_file_in_chunks(
    tmp_path: Path,
) -> None:
    """The checksum should match a known SHA-256 digest."""
    file_path = tmp_path / "source.bin"
    file_path.write_bytes(b"nyc-tlc\n")

    assert calculate_sha256(file_path, chunk_size=3) == (
        "534c7d3822b4fae29cfdf088cfbb8ad792a34a05861fac0ccd88700a1fd5eb59"
    )


def test_calculate_sha256_rejects_invalid_chunk_size(
    tmp_path: Path,
) -> None:
    """A non-positive chunk size should fail before reading the file."""
    file_path = tmp_path / "source.bin"
    file_path.write_bytes(b"data")

    with pytest.raises(
        ValueError,
        match="chunk_size must be positive",
    ):
        calculate_sha256(file_path, chunk_size=0)
