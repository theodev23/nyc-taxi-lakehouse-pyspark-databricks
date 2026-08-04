"""Tests for NYC TLC source acquisition utilities."""

import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest

from taxi_lakehouse.data_acquisition import (
    AcquisitionResult,
    SourceFile,
    acquire_source_file,
    acquire_source_files,
    calculate_sha256,
    download_source_file,
    load_source_files,
    record_download_metadata,
    validate_local_source_file,
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


def test_validate_local_source_file_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """A missing source file should not be considered valid."""
    source_file = load_source_files(PROJECT_MANIFEST_PATH)[0]

    assert not validate_local_source_file(
        source_file,
        tmp_path / source_file.filename,
    )


def test_validate_local_source_file_rejects_size_mismatch(
    tmp_path: Path,
) -> None:
    """A local file with an unexpected size should be rejected."""
    file_path = tmp_path / "source.parquet"
    file_path.write_bytes(b"short")

    source_file = replace(
        load_source_files(PROJECT_MANIFEST_PATH)[0],
        content_length_bytes=len(b"expected"),
    )

    assert not validate_local_source_file(source_file, file_path)


def test_validate_local_source_file_accepts_matching_size(
    tmp_path: Path,
) -> None:
    """Size validation should succeed before checksums are available."""
    payload = b"size-only-validation"
    file_path = tmp_path / "source.parquet"
    file_path.write_bytes(payload)

    source_file = replace(
        load_source_files(PROJECT_MANIFEST_PATH)[0],
        content_length_bytes=len(payload),
        sha256=None,
    )

    assert validate_local_source_file(source_file, file_path)


def test_validate_local_source_file_checks_sha256(
    tmp_path: Path,
) -> None:
    """A recorded checksum should detect corrupted local content."""
    payload = b"checksum-validation"
    file_path = tmp_path / "source.parquet"
    file_path.write_bytes(payload)

    source_file = replace(
        load_source_files(PROJECT_MANIFEST_PATH)[0],
        content_length_bytes=len(payload),
        sha256=calculate_sha256(file_path),
    )

    assert validate_local_source_file(source_file, file_path)

    file_path.write_bytes(b"corrupted-content!!")

    assert file_path.stat().st_size == source_file.content_length_bytes
    assert not validate_local_source_file(source_file, file_path)


def test_record_download_metadata_updates_manifest(
    tmp_path: Path,
) -> None:
    """Download metadata should be persisted atomically."""
    payload = b"manifest-metadata"
    manifest = json.loads(PROJECT_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["files"][0]["content_length_bytes"] = len(payload)
    manifest["files"][0]["sha256"] = None
    manifest["files"][0]["downloaded_at_utc"] = None
    unrelated_entry_before = manifest["files"][1].copy()

    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    source_file = load_source_files(manifest_path)[0]
    file_path = tmp_path / source_file.filename
    file_path.write_bytes(payload)

    downloaded_at_utc = "2026-08-03T08:15:00Z"

    updated_source_file = record_download_metadata(
        manifest_path,
        source_file,
        file_path,
        downloaded_at_utc,
    )

    updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    updated_entry = updated_manifest["files"][0]

    assert updated_source_file.sha256 == calculate_sha256(file_path)
    assert updated_source_file.downloaded_at_utc == downloaded_at_utc
    assert updated_entry["sha256"] == updated_source_file.sha256
    assert updated_entry["downloaded_at_utc"] == downloaded_at_utc
    assert updated_manifest["files"][1] == unrelated_entry_before
    assert validate_local_source_file(
        updated_source_file,
        file_path,
    )
    assert not Path(f"{manifest_path}.tmp").exists()


def test_record_download_metadata_rejects_size_mismatch(
    tmp_path: Path,
) -> None:
    """Invalid local data should not modify the manifest."""
    manifest = json.loads(PROJECT_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["files"][0]["content_length_bytes"] = len(b"expected")

    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    original_manifest = manifest_path.read_text(encoding="utf-8")

    source_file = load_source_files(manifest_path)[0]
    file_path = tmp_path / source_file.filename
    file_path.write_bytes(b"short")

    with pytest.raises(
        ValueError,
        match="Local source byte count does not match manifest",
    ):
        record_download_metadata(
            manifest_path,
            source_file,
            file_path,
            "2026-08-03T08:15:00Z",
        )

    assert manifest_path.read_text(encoding="utf-8") == original_manifest
    assert not Path(f"{manifest_path}.tmp").exists()


def test_download_source_file_reuses_valid_local_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A valid local source should be reused without network access."""
    payload = b"already-downloaded"
    destination_dir = tmp_path / "landing"
    destination_dir.mkdir()

    destination_path = destination_dir / "source.parquet"
    destination_path.write_bytes(payload)

    source_file = replace(
        load_source_files(PROJECT_MANIFEST_PATH)[0],
        filename=destination_path.name,
        content_length_bytes=len(payload),
        sha256=calculate_sha256(destination_path),
    )

    def unexpected_urlopen(
        url: str,
        timeout: float,
    ) -> BytesIO:
        raise AssertionError(
            f"Network access was not expected: url={url}, timeout={timeout}"
        )

    monkeypatch.setattr(
        "taxi_lakehouse.data_acquisition.urlopen",
        unexpected_urlopen,
    )

    result = download_source_file(
        source_file,
        destination_dir,
        chunk_size=4,
        timeout_seconds=5.0,
    )

    assert result == destination_path
    assert result.read_bytes() == payload
    assert not Path(f"{destination_path}.part").exists()


def test_download_source_file_writes_validated_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A complete download should replace its temporary file."""
    payload = b"nyc-tlc-download"
    source_file = replace(
        load_source_files(PROJECT_MANIFEST_PATH)[0],
        content_length_bytes=len(payload),
    )

    def fake_urlopen(
        url: str,
        timeout: float,
    ) -> BytesIO:
        assert url == source_file.url
        assert timeout == 5.0
        return BytesIO(payload)

    monkeypatch.setattr(
        "taxi_lakehouse.data_acquisition.urlopen",
        fake_urlopen,
    )

    destination_path = download_source_file(
        source_file,
        tmp_path,
        chunk_size=4,
        timeout_seconds=5.0,
    )

    assert destination_path == tmp_path / source_file.filename
    assert destination_path.read_bytes() == payload
    assert not Path(f"{destination_path}.part").exists()


def test_download_source_file_removes_invalid_partial_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A size mismatch should leave neither final nor partial data."""
    payload = b"incomplete"
    source_file = replace(
        load_source_files(PROJECT_MANIFEST_PATH)[0],
        content_length_bytes=len(payload) + 1,
    )

    def fake_urlopen(
        url: str,
        timeout: float,
    ) -> BytesIO:
        assert url == source_file.url
        assert timeout == 5.0
        return BytesIO(payload)

    monkeypatch.setattr(
        "taxi_lakehouse.data_acquisition.urlopen",
        fake_urlopen,
    )

    destination_path = tmp_path / source_file.filename
    temporary_path = Path(f"{destination_path}.part")

    with pytest.raises(
        ValueError,
        match="Downloaded byte count does not match manifest",
    ):
        download_source_file(
            source_file,
            tmp_path,
            chunk_size=4,
            timeout_seconds=5.0,
        )

    assert not destination_path.exists()
    assert not temporary_path.exists()


def test_acquire_source_file_downloads_and_records_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A missing source should be downloaded and recorded."""
    payload = b"download-and-record"
    manifest = json.loads(PROJECT_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["files"][0]["content_length_bytes"] = len(payload)

    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    source_file = load_source_files(manifest_path)[0]
    destination_dir = tmp_path / "landing"

    def fake_urlopen(
        url: str,
        timeout: float,
    ) -> BytesIO:
        assert url == source_file.url
        assert timeout == 5.0
        return BytesIO(payload)

    monkeypatch.setattr(
        "taxi_lakehouse.data_acquisition.urlopen",
        fake_urlopen,
    )

    result = acquire_source_file(
        manifest_path,
        source_file,
        destination_dir,
        "2026-08-03T08:30:00Z",
        chunk_size=4,
        timeout_seconds=5.0,
    )

    persisted_source_file = load_source_files(manifest_path)[0]

    assert isinstance(result, AcquisitionResult)
    assert result.action == "downloaded"
    assert result.file_path.read_bytes() == payload
    assert result.source_file == persisted_source_file
    assert result.source_file.sha256 == calculate_sha256(result.file_path)
    assert result.source_file.downloaded_at_utc == "2026-08-03T08:30:00Z"
    assert not Path(f"{result.file_path}.part").exists()


def test_acquire_source_file_records_missing_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Existing size-valid data should receive metadata."""
    payload = b"existing-without-metadata"
    manifest = json.loads(PROJECT_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["files"][0]["content_length_bytes"] = len(payload)
    manifest["files"][0]["sha256"] = None
    manifest["files"][0]["downloaded_at_utc"] = None

    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    source_file = load_source_files(manifest_path)[0]
    destination_dir = tmp_path / "landing"
    destination_dir.mkdir()

    destination_path = destination_dir / source_file.filename
    destination_path.write_bytes(payload)

    def unexpected_urlopen(
        url: str,
        timeout: float,
    ) -> BytesIO:
        raise AssertionError(
            f"Network access was not expected: url={url}, timeout={timeout}"
        )

    monkeypatch.setattr(
        "taxi_lakehouse.data_acquisition.urlopen",
        unexpected_urlopen,
    )

    result = acquire_source_file(
        manifest_path,
        source_file,
        destination_dir,
        "2026-08-03T08:35:00Z",
        chunk_size=4,
        timeout_seconds=5.0,
    )

    persisted_source_file = load_source_files(manifest_path)[0]

    assert result.action == "metadata_recorded"
    assert result.file_path == destination_path
    assert result.source_file == persisted_source_file
    assert result.source_file.sha256 == calculate_sha256(destination_path)
    assert result.source_file.downloaded_at_utc == "2026-08-03T08:35:00Z"


def test_acquire_source_file_reuses_complete_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Complete local data should remain unchanged."""
    payload = b"fully-recorded-source"
    manifest = json.loads(PROJECT_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["files"][0]["content_length_bytes"] = len(payload)

    destination_dir = tmp_path / "landing"
    destination_dir.mkdir()

    destination_path = destination_dir / manifest["files"][0]["filename"]
    destination_path.write_bytes(payload)

    manifest["files"][0]["sha256"] = calculate_sha256(destination_path)
    manifest["files"][0]["downloaded_at_utc"] = "2026-08-03T08:00:00Z"

    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    original_manifest = manifest_path.read_text(encoding="utf-8")

    source_file = load_source_files(manifest_path)[0]

    def unexpected_urlopen(
        url: str,
        timeout: float,
    ) -> BytesIO:
        raise AssertionError(
            f"Network access was not expected: url={url}, timeout={timeout}"
        )

    monkeypatch.setattr(
        "taxi_lakehouse.data_acquisition.urlopen",
        unexpected_urlopen,
    )

    result = acquire_source_file(
        manifest_path,
        source_file,
        destination_dir,
        "2026-08-03T08:40:00Z",
        chunk_size=4,
        timeout_seconds=5.0,
    )

    assert result.action == "reused"
    assert result.source_file == source_file
    assert result.file_path == destination_path
    assert manifest_path.read_text(encoding="utf-8") == original_manifest


def test_acquire_source_files_processes_manifest_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Batch acquisition should follow manifest order."""
    manifest = json.loads(PROJECT_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["files"] = manifest["files"][:2]

    payload_by_url: dict[str, bytes] = {}

    for index, raw_source_file in enumerate(
        manifest["files"],
        start=1,
    ):
        payload = f"batch-source-{index}".encode()
        raw_source_file["content_length_bytes"] = len(payload)
        raw_source_file["sha256"] = None
        raw_source_file["downloaded_at_utc"] = None
        payload_by_url[raw_source_file["url"]] = payload

    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    destination_dir = tmp_path / "landing"

    def fake_urlopen(
        url: str,
        timeout: float,
    ) -> BytesIO:
        assert timeout == 5.0
        return BytesIO(payload_by_url[url])

    monkeypatch.setattr(
        "taxi_lakehouse.data_acquisition.urlopen",
        fake_urlopen,
    )

    results = acquire_source_files(
        manifest_path,
        destination_dir,
        "2026-08-03T08:45:00Z",
        chunk_size=4,
        timeout_seconds=5.0,
    )

    persisted_source_files = load_source_files(manifest_path)

    assert tuple(result.action for result in results) == (
        "downloaded",
        "downloaded",
    )
    assert tuple(result.source_file.filename for result in results) == tuple(
        raw_source_file["filename"] for raw_source_file in manifest["files"]
    )
    assert tuple(result.source_file for result in results) == persisted_source_files

    for result in results:
        expected_payload = payload_by_url[result.source_file.url]

        assert result.file_path.read_bytes() == expected_payload
        assert result.source_file.sha256 == calculate_sha256(result.file_path)
        assert result.source_file.downloaded_at_utc == "2026-08-03T08:45:00Z"
        assert not Path(f"{result.file_path}.part").exists()


def test_acquire_source_files_reuses_complete_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A complete batch should be fully reusable."""
    manifest = json.loads(PROJECT_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["files"] = manifest["files"][:2]

    destination_dir = tmp_path / "landing"
    destination_dir.mkdir()

    for index, raw_source_file in enumerate(
        manifest["files"],
        start=1,
    ):
        payload = f"complete-source-{index}".encode()
        destination_path = destination_dir / raw_source_file["filename"]
        destination_path.write_bytes(payload)

        raw_source_file["content_length_bytes"] = len(payload)
        raw_source_file["sha256"] = calculate_sha256(destination_path)
        raw_source_file["downloaded_at_utc"] = "2026-08-03T08:00:00Z"

    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    original_manifest = manifest_path.read_text(encoding="utf-8")
    original_source_files = load_source_files(manifest_path)

    def unexpected_urlopen(
        url: str,
        timeout: float,
    ) -> BytesIO:
        raise AssertionError(
            f"Network access was not expected: url={url}, timeout={timeout}"
        )

    monkeypatch.setattr(
        "taxi_lakehouse.data_acquisition.urlopen",
        unexpected_urlopen,
    )

    results = acquire_source_files(
        manifest_path,
        destination_dir,
        "2026-08-03T08:50:00Z",
        chunk_size=4,
        timeout_seconds=5.0,
    )

    assert tuple(result.action for result in results) == (
        "reused",
        "reused",
    )
    assert tuple(result.source_file for result in results) == original_source_files
    assert manifest_path.read_text(encoding="utf-8") == original_manifest
