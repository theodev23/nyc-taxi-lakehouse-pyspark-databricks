"""Command-line interface for deterministic sample profiling."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from pyspark.sql import SparkSession

from taxi_lakehouse.sample_artifacts import (
    GeneratedSampleFile,
)
from taxi_lakehouse.sample_profile_artifact import (
    generate_sample_profile_artifact,
)

DEFAULT_SAMPLE_SPECIFICATION_PATH = Path("data/sample/sample_spec.json")
DEFAULT_SAMPLE_PROFILE_PATH = Path("data/sample/sample_profile.json")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the sample profiling command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic JSON profile from the versioned NYC Taxi sample."
        )
    )

    parser.add_argument(
        "--specification",
        type=Path,
        default=DEFAULT_SAMPLE_SPECIFICATION_PATH,
        help=(
            "Path to the sample specification JSON. "
            f"Default: {DEFAULT_SAMPLE_SPECIFICATION_PATH}."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SAMPLE_PROFILE_PATH,
        help=(
            "Path of the generated profile JSON. "
            f"Default: {DEFAULT_SAMPLE_PROFILE_PATH}."
        ),
    )

    return parser


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse sample profiling command-line arguments."""
    return build_argument_parser().parse_args(arguments)


def create_spark_session() -> SparkSession:
    """Create the local Spark session used for profiling."""
    return (
        SparkSession.builder.master("local[*]")
        .appName("nyc-taxi-sample-profile")
        .config("spark.ui.enabled", "false")
        .config(
            "spark.sql.session.timeZone",
            "UTC",
        )
        .getOrCreate()
    )


def print_profile_summary(
    result: GeneratedSampleFile,
) -> None:
    """Print deterministic profile artifact metadata."""
    print(
        "SAMPLE_PROFILE\t"
        f"path={result.file_path.as_posix()}\t"
        f"bytes={result.content_length_bytes}\t"
        f"sha256={result.sha256}"
    )

    print("SAMPLE_PROFILE_COMPLETE")


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    """Generate the deterministic sample profile artifact."""
    parsed_arguments = parse_arguments(arguments)

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        result = generate_sample_profile_artifact(
            spark,
            parsed_arguments.specification,
            parsed_arguments.output,
        )
    finally:
        spark.stop()

    print_profile_summary(result)

    return 0
