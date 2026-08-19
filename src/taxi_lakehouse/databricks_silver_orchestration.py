"""Silver orchestration for Databricks Unity Catalog execution."""

from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from taxi_lakehouse.databricks_configuration import (
    DatabricksPipelineConfiguration,
)
from taxi_lakehouse.databricks_silver_io import (
    load_managed_bronze_source_months,
    load_managed_bronze_trip_month,
    load_managed_bronze_zone_ids,
    write_managed_silver_trip_month,
)
from taxi_lakehouse.silver_quality_specification import (
    SilverQualitySpecification,
    load_silver_quality_specification,
)
from taxi_lakehouse.silver_transformation import (
    annotate_silver_quality,
    split_silver_quality_rows,
)


@dataclass(frozen=True, slots=True)
class DatabricksSilverMonthWriteResult:
    """Metadata for one Databricks Silver month."""

    source_month: str
    accepted_destination_table: str
    rejected_destination_table: str
    accepted_row_count: int
    rejected_row_count: int

    @property
    def total_row_count(self) -> int:
        """Return the total processed row count."""
        return self.accepted_row_count + self.rejected_row_count


@dataclass(frozen=True, slots=True)
class DatabricksSilverBuildResult:
    """Metadata for one complete Databricks Silver run."""

    specification: SilverQualitySpecification
    configuration: DatabricksPipelineConfiguration
    source_months: tuple[str, ...]
    zone_id_count: int
    monthly_writes: tuple[DatabricksSilverMonthWriteResult, ...]


def _materialize_split_and_write(
    annotated_frame: DataFrame,
    specification: SilverQualitySpecification,
    source_month: str,
    accepted_table: str,
    rejected_table: str,
) -> tuple[int, int]:
    """Count and write one Silver month without Spark caching."""
    split = split_silver_quality_rows(
        annotated_frame,
        specification,
    )

    accepted_row_count = split.accepted.count()
    rejected_row_count = split.rejected.count()

    write_managed_silver_trip_month(
        split.accepted,
        accepted_table,
        source_month,
    )
    write_managed_silver_trip_month(
        split.rejected,
        rejected_table,
        source_month,
    )

    return (
        accepted_row_count,
        rejected_row_count,
    )


def build_databricks_silver_dataset(
    spark: SparkSession,
    specification_path: Path,
    configuration: DatabricksPipelineConfiguration,
) -> DatabricksSilverBuildResult:
    """Build managed Silver tables from managed Bronze tables."""
    specification = load_silver_quality_specification(specification_path)

    source_months = load_managed_bronze_source_months(
        spark,
        configuration.bronze_trip_table,
    )
    zone_ids = load_managed_bronze_zone_ids(
        spark,
        configuration.bronze_taxi_zone_table,
    )

    monthly_writes: list[DatabricksSilverMonthWriteResult] = []

    for source_month in source_months:
        bronze_frame = load_managed_bronze_trip_month(
            spark,
            configuration.bronze_trip_table,
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
            configuration.silver_accepted_table,
            configuration.silver_rejected_table,
        )

        monthly_writes.append(
            DatabricksSilverMonthWriteResult(
                source_month=source_month,
                accepted_destination_table=(configuration.silver_accepted_table),
                rejected_destination_table=(configuration.silver_rejected_table),
                accepted_row_count=accepted_row_count,
                rejected_row_count=rejected_row_count,
            )
        )

    return DatabricksSilverBuildResult(
        specification=specification,
        configuration=configuration,
        source_months=source_months,
        zone_id_count=len(zone_ids),
        monthly_writes=tuple(monthly_writes),
    )
