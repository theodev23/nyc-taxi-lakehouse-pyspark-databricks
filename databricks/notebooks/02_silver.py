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
    """Run managed Silver processing using the Databricks Spark session."""
    from taxi_lakehouse.databricks_configuration import (
        DatabricksPipelineConfiguration,
    )
    from taxi_lakehouse.databricks_silver_orchestration import (
        build_databricks_silver_dataset,
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

    result = build_databricks_silver_dataset(
        spark_session,
        REPOSITORY_ROOT / "data" / "silver_quality_spec.json",
        configuration,
    )

    accepted_row_count = sum(
        write.accepted_row_count for write in result.monthly_writes
    )
    rejected_row_count = sum(
        write.rejected_row_count for write in result.monthly_writes
    )

    print(
        "DATABRICKS_SILVER_COMPLETE "
        f"months={len(result.monthly_writes)} "
        f"rows={accepted_row_count + rejected_row_count} "
        f"accepted_rows={accepted_row_count} "
        f"rejected_rows={rejected_row_count} "
        f"zone_ids={result.zone_id_count}"
    )


# COMMAND ----------

run(spark, dbutils)  # noqa: F821
