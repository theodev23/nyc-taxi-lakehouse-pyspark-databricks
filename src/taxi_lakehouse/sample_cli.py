"""Command-line interface for deterministic sample generation."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from pyspark.sql import SparkSession

from taxi_lakehouse.sample_orchestration import (
    GeneratedSampleDataset,
    generate_sample_dataset,
)
from taxi_lakehouse.sample_pipeline import (
    GeneratedMonthlySample,
)

DEFAULT_SPECIFICATION_PATH = Path("data/sample/sample_spec.json")
DEFAULT_LANDING_DIRECTORY = Path("data/landing")
DEFAULT_TEMPORARY_ROOT = Path("data/sample/.generation_tmp")


def build_parser() -> argparse.ArgumentParser:
    """Build the deterministic sample generation parser."""
    parser = argparse.ArgumentParser(
        description=("Generate the deterministic NYC Yellow Taxi development sample.")
    )
    parser.add_argument(
        "--specification",
        type=Path,
        default=DEFAULT_SPECIFICATION_PATH,
        help=(
            "Path to the sample specification (default: data/sample/sample_spec.json)."
        ),
    )
    parser.add_argument(
        "--landing-dir",
        type=Path,
        default=DEFAULT_LANDING_DIRECTORY,
        help=("Directory containing source files (default: data/landing)."),
    )
    parser.add_argument(
        "--temporary-root",
        type=Path,
        default=DEFAULT_TEMPORARY_ROOT,
        help=(
            "Directory for temporary Spark output "
            "(default: data/sample/.generation_tmp)."
        ),
    )
    return parser


def build_spark_session() -> SparkSession:
    """Create the local Spark session used for sample generation."""
    return (
        SparkSession.builder.master("local[*]")
        .appName("nyc-taxi-deterministic-sample")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def format_monthly_sample(
    sample: GeneratedMonthlySample,
) -> str:
    """Format one generated monthly sample."""
    return (
        "TRIP_SAMPLE"
        f"\tmonth={sample.source_month}"
        f"\trows={sample.selected_row_count}"
        f"\tcolumns={sample.source_column_count}"
        f"\tpath={sample.output_file.file_path.as_posix()}"
        f"\tsha256={sample.output_file.sha256}"
    )


def print_generation_summary(
    result: GeneratedSampleDataset,
) -> None:
    """Print generated artifacts and aggregate counts."""
    for monthly_sample in result.monthly_samples:
        print(format_monthly_sample(monthly_sample))

    print(
        "ZONE_SAMPLE"
        "\trows="
        + str(result.specification.source_profile.taxi_zone_row_count)
        + "\tpath="
        + result.taxi_zone_output.file_path.as_posix()
        + "\tsha256="
        + result.taxi_zone_output.sha256
    )

    print(
        "SAMPLE_MANIFEST"
        "\tpath="
        + result.manifest_output.file_path.as_posix()
        + "\tsha256="
        + result.manifest_output.sha256
    )

    trip_row_count = sum(sample.selected_row_count for sample in result.monthly_samples)

    print(
        "SAMPLE_GENERATION_COMPLETE "
        f"trip_files={len(result.monthly_samples)} "
        f"trip_rows={trip_row_count} "
        f"data_artifacts={len(result.monthly_samples) + 1}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the sample dataset from command-line arguments."""
    arguments = build_parser().parse_args(argv)
    spark = build_spark_session()

    try:
        result = generate_sample_dataset(
            spark,
            arguments.specification,
            arguments.landing_dir,
            arguments.temporary_root,
        )

        print_generation_summary(result)
        return 0
    finally:
        spark.stop()
