"""Command-line interface for Bronze Delta ingestion."""

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pyspark.sql import SparkSession

from taxi_lakehouse.bronze_orchestration import (
    BronzeIngestionResult,
    BronzeTripWriteResult,
    ingest_bronze_dataset,
)
from taxi_lakehouse.spark_session import build_local_spark_session

DEFAULT_MANIFEST_PATH = Path("data/source_manifest.json")
DEFAULT_LANDING_DIRECTORY = Path("data/landing")
DEFAULT_BRONZE_ROOT = Path("data/lakehouse/bronze")


def build_parser() -> argparse.ArgumentParser:
    """Build the Bronze ingestion argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate NYC Taxi source files and ingest them "
            "into local Bronze Delta tables."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=("Path to the source manifest (default: data/source_manifest.json)."),
    )
    parser.add_argument(
        "--landing-dir",
        type=Path,
        default=DEFAULT_LANDING_DIRECTORY,
        help=("Directory containing validated source files (default: data/landing)."),
    )
    parser.add_argument(
        "--bronze-root",
        type=Path,
        default=DEFAULT_BRONZE_ROOT,
        help=(
            "Destination directory for Bronze Delta tables "
            "(default: data/lakehouse/bronze)."
        ),
    )
    return parser


def current_utc_datetime() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(UTC)


def build_spark_session() -> SparkSession:
    """Create the Delta-enabled Spark session used for Bronze ingestion."""
    return build_local_spark_session(
        "nyc-taxi-bronze-ingestion",
        enable_delta=True,
    )


def format_trip_write(
    trip_write: BronzeTripWriteResult,
) -> str:
    """Format one monthly Bronze trip write."""
    return (
        "BRONZE_TRIP"
        f"\tmonth={trip_write.source_month}"
        f"\trows={trip_write.row_count}"
        f"\tsource={trip_write.source_path.as_posix()}"
        f"\tdestination={trip_write.destination_path.as_posix()}"
    )


def print_ingestion_summary(
    result: BronzeIngestionResult,
) -> None:
    """Print every Bronze output and aggregate ingestion counts."""
    for trip_write in result.monthly_trip_writes:
        print(format_trip_write(trip_write))

    print(
        "BRONZE_TAXI_ZONES"
        f"\trows={result.taxi_zone_row_count}"
        f"\tdestination="
        f"{result.taxi_zone_destination_path.as_posix()}"
    )

    trip_row_count = sum(
        trip_write.row_count for trip_write in result.monthly_trip_writes
    )

    ingestion_timestamp = result.ingested_at_utc.isoformat(
        timespec="seconds",
    ).replace(
        "+00:00",
        "Z",
    )

    print(
        "BRONZE_INGESTION_COMPLETE "
        f"trip_files={len(result.monthly_trip_writes)} "
        f"trip_rows={trip_row_count} "
        f"taxi_zone_rows={result.taxi_zone_row_count} "
        f"ingested_at_utc={ingestion_timestamp}"
    )


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    """Run Bronze ingestion from command-line arguments."""
    parsed_arguments = build_parser().parse_args(arguments)
    ingested_at_utc = current_utc_datetime()

    spark = build_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        result = ingest_bronze_dataset(
            spark,
            parsed_arguments.manifest,
            parsed_arguments.landing_dir,
            parsed_arguments.bronze_root,
            ingested_at_utc,
        )
    finally:
        spark.stop()

    print_ingestion_summary(result)

    return 0
