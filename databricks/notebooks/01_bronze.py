# Databricks notebook source
from datetime import UTC, datetime

from _bootstrap import prepare_repository_imports

REPOSITORY_ROOT = prepare_repository_imports()

# COMMAND ----------


def require_widget(
    dbutils_runtime: object,
    name: str,
) -> str:
    """Return one required non-empty notebook parameter."""
    value = dbutils_runtime.widgets.get(name).strip()

    if not value:
        raise ValueError(f"Databricks notebook parameter {name!r} must be non-empty.")

    return value


def run(
    spark_session: object,
    dbutils_runtime: object,
) -> None:
    """Run managed Bronze ingestion using the Databricks Spark session."""
    from taxi_lakehouse.databricks_bronze_orchestration import (
        ingest_databricks_bronze_dataset,
    )
    from taxi_lakehouse.databricks_configuration import (
        DatabricksPipelineConfiguration,
    )

    for widget_name in (
        "catalog",
        "schema",
        "volume",
    ):
        dbutils_runtime.widgets.text(
            widget_name,
            "",
        )

    configuration = DatabricksPipelineConfiguration(
        catalog=require_widget(
            dbutils_runtime,
            "catalog",
        ),
        schema=require_widget(
            dbutils_runtime,
            "schema",
        ),
        volume=require_widget(
            dbutils_runtime,
            "volume",
        ),
    )

    result = ingest_databricks_bronze_dataset(
        spark_session,
        REPOSITORY_ROOT / "data" / "source_manifest.json",
        configuration,
        datetime.now(UTC),
    )

    trip_row_count = sum(write.row_count for write in result.monthly_trip_writes)

    print(
        "DATABRICKS_BRONZE_COMPLETE "
        f"months={len(result.monthly_trip_writes)} "
        f"trip_rows={trip_row_count} "
        f"taxi_zone_rows={result.taxi_zone_row_count}"
    )


# COMMAND ----------

run(spark, dbutils)  # noqa: F821
