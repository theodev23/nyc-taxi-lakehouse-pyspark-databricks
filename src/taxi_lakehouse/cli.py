"""Command-line interface for NYC TLC source acquisition."""

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from taxi_lakehouse.data_acquisition import (
    AcquisitionResult,
    acquire_source_files,
)

DEFAULT_MANIFEST_PATH = Path("data/source_manifest.json")
DEFAULT_DESTINATION_DIR = Path("data/landing")
DEFAULT_CHUNK_SIZE = 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 60.0


def build_parser() -> argparse.ArgumentParser:
    """Build the source acquisition argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download and validate NYC TLC source files "
            "declared in the project manifest."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=("Path to the source manifest (default: data/source_manifest.json)."),
    )
    parser.add_argument(
        "--destination-dir",
        type=Path,
        default=DEFAULT_DESTINATION_DIR,
        help=("Directory for downloaded source files (default: data/landing)."),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Download chunk size in bytes.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Network timeout in seconds.",
    )
    return parser


def current_utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def format_result(result: AcquisitionResult) -> str:
    """Format one acquisition result for terminal output."""
    checksum = result.source_file.sha256 or "missing"

    return f"{result.action}\t{result.source_file.filename}\tsha256={checksum}"


def main(argv: Sequence[str] | None = None) -> int:
    """Run source acquisition from command-line arguments."""
    arguments = build_parser().parse_args(argv)
    downloaded_at_utc = current_utc_timestamp()

    results = acquire_source_files(
        arguments.manifest,
        arguments.destination_dir,
        downloaded_at_utc,
        chunk_size=arguments.chunk_size,
        timeout_seconds=arguments.timeout_seconds,
    )

    for result in results:
        print(format_result(result))

    downloaded_count = sum(result.action == "downloaded" for result in results)
    metadata_recorded_count = sum(
        result.action == "metadata_recorded" for result in results
    )
    reused_count = sum(result.action == "reused" for result in results)

    print(
        "ACQUISITION_COMPLETE "
        f"total={len(results)} "
        f"downloaded={downloaded_count} "
        f"metadata_recorded={metadata_recorded_count} "
        f"reused={reused_count}"
    )

    return 0
