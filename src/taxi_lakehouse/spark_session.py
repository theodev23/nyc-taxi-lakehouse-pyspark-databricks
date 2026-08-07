"""Shared Spark session construction for local pipeline commands."""

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

DEFAULT_LOCAL_MASTER = "local[*]"
DELTA_EXTENSION = "io.delta.sql.DeltaSparkSessionExtension"
DELTA_CATALOG = "org.apache.spark.sql.delta.catalog.DeltaCatalog"


def build_local_spark_session(
    app_name: str,
    *,
    enable_delta: bool = False,
    master: str = DEFAULT_LOCAL_MASTER,
) -> SparkSession:
    """Create a consistently configured local Spark session."""
    builder = (
        SparkSession.builder.master(master)
        .appName(app_name)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
    )

    if enable_delta:
        builder = builder.config(
            "spark.sql.extensions",
            DELTA_EXTENSION,
        ).config(
            "spark.sql.catalog.spark_catalog",
            DELTA_CATALOG,
        )
        builder = configure_spark_with_delta_pip(builder)

    return builder.getOrCreate()
