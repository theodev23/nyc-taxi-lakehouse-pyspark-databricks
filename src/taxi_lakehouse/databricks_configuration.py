"""Unity Catalog configuration for Databricks pipeline execution."""

import re
from dataclasses import dataclass
from pathlib import Path

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def require_unquoted_identifier(
    value: object,
    field_name: str,
) -> str:
    """Require one simple unquoted Unity Catalog identifier."""
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a valid unquoted "
            f"Unity Catalog identifier: {value!r}."
        )

    return value


@dataclass(frozen=True, slots=True)
class DatabricksPipelineConfiguration:
    """Unity Catalog objects used by the Databricks pipeline."""

    catalog: str
    schema: str
    volume: str

    def __post_init__(self) -> None:
        """Validate every configurable Unity Catalog identifier."""
        require_unquoted_identifier(
            self.catalog,
            "catalog",
        )
        require_unquoted_identifier(
            self.schema,
            "schema",
        )
        require_unquoted_identifier(
            self.volume,
            "volume",
        )

    @property
    def volume_root(self) -> Path:
        """Return the Unity Catalog volume filesystem path."""
        return Path("/Volumes") / self.catalog / self.schema / self.volume

    @property
    def landing_directory(self) -> Path:
        """Return the landing directory inside the project volume."""
        return self.volume_root / "landing"

    def table_name(
        self,
        table: str,
    ) -> str:
        """Return one fully qualified Unity Catalog table name."""
        validated_table = require_unquoted_identifier(
            table,
            "table",
        )

        return f"{self.catalog}.{self.schema}.{validated_table}"

    @property
    def bronze_trip_table(self) -> str:
        """Return the managed Bronze trip-table name."""
        return self.table_name("bronze_yellow_taxi_trips")

    @property
    def bronze_taxi_zone_table(self) -> str:
        """Return the managed Bronze taxi-zone table name."""
        return self.table_name("bronze_taxi_zones")

    @property
    def silver_accepted_table(self) -> str:
        """Return the managed Silver accepted-trip table name."""
        return self.table_name("silver_yellow_taxi_trips_accepted")

    @property
    def silver_rejected_table(self) -> str:
        """Return the managed Silver rejected-trip table name."""
        return self.table_name("silver_yellow_taxi_trips_rejected")

    @property
    def gold_trip_metrics_table(self) -> str:
        """Return the managed Gold trip-metrics table name."""
        return self.table_name("gold_trip_metrics_by_date_pickup_zone_payment")

    @property
    def gold_daily_metrics_table(self) -> str:
        """Return the managed Gold daily-metrics table name."""
        return self.table_name("gold_daily_trip_metrics")
