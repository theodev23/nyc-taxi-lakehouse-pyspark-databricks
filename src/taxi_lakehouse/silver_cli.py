"""Command-line interface for Silver data-quality processing."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from pyspark.sql import SparkSession

from taxi_lakehouse.silver_orchestration import (
    SilverBuildResult,
    SilverMonthWriteResult,
    build_silver_dataset,
)
from taxi_lakehouse.spark_session import build_local_spark_session

DEFAULT_SPECIFICATION_PATH = Path("data/silver_quality_spec.json")
SILVER_LOCAL_MASTER = "local[2]"


def build_parser() -> argparse.ArgumentParser:
    """Build the Silver processing argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply the Silver data-quality contract to Bronze NYC Taxi Delta tables."
        )
    )
    parser.add_argument(
        "--specification",
        type=Path,
        default=DEFAULT_SPECIFICATION_PATH,
        help=(
            "Path to the Silver quality specification "
            "(default: data/silver_quality_spec.json)."
        ),
    )
    return parser


def build_spark_session() -> SparkSession:
    """Create the Delta-enabled Spark session used for Silver."""
    return build_local_spark_session(
        "nyc-taxi-silver-build",
        enable_delta=True,
        master=SILVER_LOCAL_MASTER,
    )


def format_month_write(
    month_write: SilverMonthWriteResult,
) -> str:
    """Format one monthly Silver processing result."""
    return (
        "SILVER_MONTH"
        f"\tmonth={month_write.source_month}"
        f"\taccepted_rows={month_write.accepted_row_count}"
        f"\trejected_rows={month_write.rejected_row_count}"
        f"\ttotal_rows={month_write.total_row_count}"
        f"\taccepted_destination="
        f"{month_write.accepted_destination_path.as_posix()}"
        f"\trejected_destination="
        f"{month_write.rejected_destination_path.as_posix()}"
    )


def print_build_summary(
    result: SilverBuildResult,
) -> None:
    """Print monthly Silver outputs and aggregate row counts."""
    for month_write in result.monthly_writes:
        print(format_month_write(month_write))

    accepted_row_count = sum(
        month_write.accepted_row_count for month_write in result.monthly_writes
    )
    rejected_row_count = sum(
        month_write.rejected_row_count for month_write in result.monthly_writes
    )

    print(
        "SILVER_BUILD_COMPLETE "
        f"months={len(result.monthly_writes)} "
        f"rows={accepted_row_count + rejected_row_count} "
        f"accepted_rows={accepted_row_count} "
        f"rejected_rows={rejected_row_count} "
        f"zone_ids={result.zone_id_count}"
    )


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    """Run Silver processing from command-line arguments."""
    parsed_arguments = build_parser().parse_args(arguments)

    spark = build_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        result = build_silver_dataset(
            spark,
            parsed_arguments.specification,
        )
    finally:
        spark.stop()

    print_build_summary(result)

    return 0
