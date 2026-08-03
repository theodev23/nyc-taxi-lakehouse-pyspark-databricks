"""Utilities for acquiring versioned NYC TLC source files."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen


@dataclass(frozen=True, slots=True)
class SourceFile:
    """Metadata describing one remote source file."""

    filename: str
    source_month: str | None
    kind: str
    url: str
    content_type: str
    content_length_bytes: int
    last_modified: str
    etag: str
    accept_ranges: str
    sha256: str | None
    downloaded_at_utc: str | None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "SourceFile":
        """Build and validate a source file from manifest data."""
        filename = value["filename"]
        url = value["url"]
        content_length_bytes = value["content_length_bytes"]

        if not isinstance(filename, str) or not filename:
            raise ValueError("Source filename must be a non-empty string.")

        if Path(filename).name != filename:
            raise ValueError(
                f"Source filename must not contain directories: {filename!r}."
            )

        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError(f"Source URL must use HTTPS: {url!r}.")

        if (
            not isinstance(content_length_bytes, int)
            or isinstance(content_length_bytes, bool)
            or content_length_bytes <= 0
        ):
            raise ValueError("Source content_length_bytes must be a positive integer.")

        return cls(
            filename=filename,
            source_month=value["source_month"],
            kind=value["kind"],
            url=url,
            content_type=value["content_type"],
            content_length_bytes=content_length_bytes,
            last_modified=value["last_modified"],
            etag=value["etag"],
            accept_ranges=value["accept_ranges"],
            sha256=value["sha256"],
            downloaded_at_utc=value["downloaded_at_utc"],
        )


def load_source_files(manifest_path: Path) -> tuple[SourceFile, ...]:
    """Load all source file records from a JSON manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("schema_version") != "1.0":
        raise ValueError("Unsupported source manifest schema version.")

    raw_files = manifest.get("files")

    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("Source manifest must contain a non-empty files list.")

    source_files = tuple(SourceFile.from_mapping(raw_file) for raw_file in raw_files)

    filenames = [source_file.filename for source_file in source_files]

    if len(filenames) != len(set(filenames)):
        raise ValueError("Source manifest contains duplicate filenames.")

    return source_files


def calculate_sha256(
    file_path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculate the SHA-256 checksum of a local file."""
    if chunk_size <= 0:
        raise ValueError("SHA-256 chunk_size must be positive.")

    digest = hashlib.sha256()

    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_source_file(
    source_file: SourceFile,
    destination_dir: Path,
    chunk_size: int = 1024 * 1024,
    timeout_seconds: float = 60.0,
) -> Path:
    """Download one source file through an atomic temporary file."""
    if chunk_size <= 0:
        raise ValueError("Download chunk_size must be positive.")

    if timeout_seconds <= 0:
        raise ValueError("Download timeout_seconds must be positive.")

    destination_dir.mkdir(parents=True, exist_ok=True)

    destination_path = destination_dir / source_file.filename
    temporary_path = destination_path.with_name(f"{destination_path.name}.part")
    bytes_written = 0

    try:
        with urlopen(
            source_file.url,
            timeout=timeout_seconds,
        ) as response:
            with temporary_path.open("wb") as destination:
                while chunk := response.read(chunk_size):
                    destination.write(chunk)
                    bytes_written += len(chunk)

        if bytes_written != source_file.content_length_bytes:
            raise ValueError(
                "Downloaded byte count does not match manifest: "
                f"expected={source_file.content_length_bytes}, "
                f"actual={bytes_written}, "
                f"filename={source_file.filename!r}."
            )

        temporary_path.replace(destination_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return destination_path
