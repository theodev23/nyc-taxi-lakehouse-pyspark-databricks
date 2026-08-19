# Databricks notebook source

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
    """Run managed Gold processing using the Databricks Spark session."""
    from taxi_lakehouse.databricks_configuration import (
        DatabricksPipelineConfiguration,
    )
    from taxi_lakehouse.databricks_gold_orchestration import (
        build_databricks_gold_dataset,
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

    result = build_databricks_gold_dataset(
        spark_session,
        REPOSITORY_ROOT / "data" / "gold_analytics_spec.json",
        configuration,
    )

    trip_metrics_row_count = sum(
        write.trip_metrics_row_count for write in result.monthly_writes
    )
    daily_metrics_row_count = sum(
        write.daily_metrics_row_count for write in result.monthly_writes
    )

    print(
        "DATABRICKS_GOLD_COMPLETE "
        f"months={len(result.monthly_writes)} "
        f"trip_metrics_rows={trip_metrics_row_count} "
        f"daily_metrics_rows={daily_metrics_row_count} "
        f"taxi_zone_rows={result.taxi_zone_row_count}"
    )


# COMMAND ----------

run(spark, dbutils)  # noqa: F821
