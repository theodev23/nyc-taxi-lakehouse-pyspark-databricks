"""End-to-end orchestration for Silver data-quality processing."""

from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from taxi_lakehouse.silver_loading import (
    load_bronze_source_months,
    load_bronze_trip_month,
    load_bronze_zone_ids,
)
from taxi_lakehouse.silver_quality_specification import (
    SilverQualitySpecification,
    load_silver_quality_specification,
)
from taxi_lakehouse.silver_transformation import (
    annotate_silver_quality,
    split_silver_quality_rows,
)
from taxi_lakehouse.silver_writing import (
    write_silver_trip_month,
)


@dataclass(frozen=True, slots=True)
class SilverMonthWriteResult:
    """Metadata describing one monthly Silver processing result."""

    source_month: str
    accepted_destination_path: Path
    rejected_destination_path: Path
    accepted_row_count: int
    rejected_row_count: int

    @property
    def total_row_count(self) -> int:
        """Return the total number of processed rows."""
        return self.accepted_row_count + self.rejected_row_count


@dataclass(frozen=True, slots=True)
class SilverBuildResult:
    """Metadata describing one complete Silver processing run."""

    specification: SilverQualitySpecification
    source_months: tuple[str, ...]
    zone_id_count: int
    monthly_writes: tuple[SilverMonthWriteResult, ...]


def _materialize_split_and_write(
    annotated_frame: DataFrame,
    specification: SilverQualitySpecification,
    source_month: str,
) -> tuple[int, int]:
    """Cache, count, write, and release one annotated Silver month."""
    cached_frame = annotated_frame.cache()

    try:
        split = split_silver_quality_rows(
            cached_frame,
            specification,
        )

        accepted_row_count = split.accepted.count()
        rejected_row_count = split.rejected.count()

        write_silver_trip_month(
            split.accepted,
            specification.output.accepted_table,
            source_month,
        )
        write_silver_trip_month(
            split.rejected,
            specification.output.rejected_table,
            source_month,
        )
    finally:
        cached_frame.unpersist(blocking=True)

    return (
        accepted_row_count,
        rejected_row_count,
    )


def build_silver_dataset(
    spark: SparkSession,
    specification_path: Path,
) -> SilverBuildResult:
    """Build accepted and rejected Silver Delta tables from Bronze."""
    specification = load_silver_quality_specification(specification_path)

    source_months = load_bronze_source_months(
        spark,
        specification.source_trip_table,
    )
    zone_ids = load_bronze_zone_ids(
        spark,
        specification.source_taxi_zone_table,
    )

    monthly_writes: list[SilverMonthWriteResult] = []

    for source_month in source_months:
        bronze_frame = load_bronze_trip_month(
            spark,
            specification.source_trip_table,
            source_month,
        )

        annotated_frame = annotate_silver_quality(
            bronze_frame,
            source_month,
            zone_ids,
            specification,
        )

        (
            accepted_row_count,
            rejected_row_count,
        ) = _materialize_split_and_write(
            annotated_frame,
            specification,
            source_month,
        )

        monthly_writes.append(
            SilverMonthWriteResult(
                source_month=source_month,
                accepted_destination_path=(specification.output.accepted_table),
                rejected_destination_path=(specification.output.rejected_table),
                accepted_row_count=accepted_row_count,
                rejected_row_count=rejected_row_count,
            )
        )

    return SilverBuildResult(
        specification=specification,
        source_months=source_months,
        zone_id_count=len(zone_ids),
        monthly_writes=tuple(monthly_writes),
    )
