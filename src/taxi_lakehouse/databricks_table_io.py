"""Unity Catalog managed-table I/O for Databricks execution."""

from pyspark.sql import DataFrame, SparkSession

from taxi_lakehouse.databricks_configuration import (
    require_unquoted_identifier,
)


def require_fully_qualified_table_name(
    value: object,
) -> str:
    """Require one catalog.schema.table Unity Catalog name."""
    if not isinstance(value, str):
        raise ValueError("table_name must use catalog.schema.table format.")

    identifiers = value.split(".")

    if len(identifiers) != 3:
        raise ValueError("table_name must use catalog.schema.table format.")

    catalog, schema, table = identifiers

    require_unquoted_identifier(catalog, "catalog")
    require_unquoted_identifier(schema, "schema")
    require_unquoted_identifier(table, "table")

    return value


def read_managed_table(
    spark: SparkSession,
    table_name: str,
) -> DataFrame:
    """Read one Unity Catalog managed table."""
    validated_table_name = require_fully_qualified_table_name(table_name)

    return spark.read.table(validated_table_name)


def overwrite_managed_table(
    frame: DataFrame,
    table_name: str,
) -> None:
    """Replace one complete Unity Catalog managed table."""
    validated_table_name = require_fully_qualified_table_name(table_name)

    (frame.write.format("delta").mode("overwrite").saveAsTable(validated_table_name))


def replace_managed_table_rows(
    frame: DataFrame,
    table_name: str,
    predicate: str,
) -> None:
    """Atomically replace managed-table rows matching one predicate."""
    validated_table_name = require_fully_qualified_table_name(table_name)

    if not isinstance(predicate, str) or not predicate.strip():
        raise ValueError("replaceWhere predicate must be a non-empty string.")

    (
        frame.write.format("delta")
        .mode("overwrite")
        .option(
            "replaceWhere",
            predicate,
        )
        .saveAsTable(validated_table_name)
    )
