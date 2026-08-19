"""Command-line interface for Gold analytical processing."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from pyspark.sql import SparkSession

from taxi_lakehouse.gold_orchestration import (
    GoldBuildResult,
    GoldMonthWriteResult,
    build_gold_dataset,
)
from taxi_lakehouse.spark_session import build_local_spark_session

DEFAULT_SPECIFICATION_PATH = Path("data/gold_analytics_spec.json")

GOLD_LOCAL_MASTER = "local[2]"


def build_parser() -> argparse.ArgumentParser:
    """Build the Gold processing argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build Gold analytical tables from accepted Silver NYC Taxi trips."
        )
    )
    parser.add_argument(
        "--specification",
        type=Path,
        default=DEFAULT_SPECIFICATION_PATH,
        help=(
            "Path to the Gold analytical specification "
            "(default: data/gold_analytics_spec.json)."
        ),
    )
    return parser


def build_spark_session() -> SparkSession:
    """Create the Delta-enabled Spark session used for Gold."""
    return build_local_spark_session(
        "nyc-taxi-gold-build",
        enable_delta=True,
        master=GOLD_LOCAL_MASTER,
    )


def format_month_write(
    month_write: GoldMonthWriteResult,
) -> str:
    """Format one monthly Gold processing result."""
    return (
        "GOLD_MONTH"
        f"\tmonth={month_write.source_month}"
        f"\ttrip_metrics_rows={month_write.trip_metrics_row_count}"
        f"\tdaily_metrics_rows={month_write.daily_metrics_row_count}"
        f"\ttrip_metrics_destination="
        f"{month_write.trip_metrics_destination_path.as_posix()}"
        f"\tdaily_metrics_destination="
        f"{month_write.daily_metrics_destination_path.as_posix()}"
    )


def print_build_summary(
    result: GoldBuildResult,
) -> None:
    """Print monthly Gold outputs and aggregate row counts."""
    for month_write in result.monthly_writes:
        print(format_month_write(month_write))

    trip_metrics_row_count = sum(
        month_write.trip_metrics_row_count for month_write in result.monthly_writes
    )
    daily_metrics_row_count = sum(
        month_write.daily_metrics_row_count for month_write in result.monthly_writes
    )

    print(
        "GOLD_BUILD_COMPLETE "
        f"months={len(result.monthly_writes)} "
        f"trip_metrics_rows={trip_metrics_row_count} "
        f"daily_metrics_rows={daily_metrics_row_count} "
        f"taxi_zone_rows={result.taxi_zone_row_count}"
    )


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    """Run Gold processing from command-line arguments."""
    parsed_arguments = build_parser().parse_args(arguments)

    spark = build_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        result = build_gold_dataset(
            spark,
            parsed_arguments.specification,
        )
    finally:
        spark.stop()

    print_build_summary(result)

    return 0
