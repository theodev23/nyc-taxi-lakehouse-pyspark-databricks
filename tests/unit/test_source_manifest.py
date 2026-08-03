"""Tests for the versioned NYC TLC source manifest."""

import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

MANIFEST_PATH = Path("data/source_manifest.json")

EXPECTED_MONTHS = [
    "2024-01",
    "2024-02",
    "2024-03",
    "2024-04",
    "2024-05",
    "2024-06",
]

EXPECTED_TRIP_FILENAMES = [
    f"yellow_tripdata_{source_month}.parquet" for source_month in EXPECTED_MONTHS
]


def load_manifest() -> dict:
    """Load the versioned source manifest."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_dataset_scope() -> None:
    """The manifest should describe the intended official dataset period."""
    manifest = load_manifest()

    assert manifest["schema_version"] == "1.0"
    assert manifest["dataset"]["name"] == "NYC TLC Yellow Taxi Trip Records"
    assert manifest["dataset"]["period_start"] == "2024-01"
    assert manifest["dataset"]["period_end"] == "2024-06"
    assert manifest["audit"]["etag_is_checksum"] is False


def test_manifest_contains_expected_sources() -> None:
    """The manifest should contain six monthly files and one zone lookup."""
    files = load_manifest()["files"]

    trip_files = [item for item in files if item["kind"] == "yellow_taxi_trip_data"]
    lookup_files = [item for item in files if item["kind"] == "taxi_zone_lookup"]

    assert len(files) == 7
    assert len(trip_files) == 6
    assert len(lookup_files) == 1

    assert [item["source_month"] for item in trip_files] == EXPECTED_MONTHS
    assert [item["filename"] for item in trip_files] == EXPECTED_TRIP_FILENAMES
    assert lookup_files[0]["filename"] == "taxi_zone_lookup.csv"


def test_manifest_source_metadata_is_valid() -> None:
    """Each source should have valid official and acquisition metadata."""
    files = load_manifest()["files"]

    for item in files:
        parsed_url = urlparse(item["url"])
        sha256 = item["sha256"]
        downloaded_at_utc = item["downloaded_at_utc"]

        assert parsed_url.scheme == "https"
        assert parsed_url.netloc == "d37ci6vzurychx.cloudfront.net"
        assert item["content_length_bytes"] > 0
        assert item["accept_ranges"] == "bytes"

        assert (sha256 is None) == (downloaded_at_utc is None)

        if sha256 is not None:
            assert isinstance(sha256, str)
            assert len(sha256) == 64
            assert all(character in "0123456789abcdef" for character in sha256)

            assert isinstance(downloaded_at_utc, str)
            assert downloaded_at_utc.endswith("Z")

            parsed_downloaded_at = datetime.fromisoformat(
                downloaded_at_utc.replace("Z", "+00:00")
            )

            assert parsed_downloaded_at.utcoffset() is not None
            assert parsed_downloaded_at.utcoffset().total_seconds() == 0
