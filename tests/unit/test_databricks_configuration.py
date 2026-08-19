"""Tests for Databricks Unity Catalog pipeline configuration."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from taxi_lakehouse.databricks_configuration import (
    DatabricksPipelineConfiguration,
    require_unquoted_identifier,
)


def build_configuration() -> DatabricksPipelineConfiguration:
    """Build one representative Databricks pipeline configuration."""
    return DatabricksPipelineConfiguration(
        catalog="workspace",
        schema="nyc_taxi",
        volume="pipeline_data",
    )


def test_configuration_builds_volume_paths() -> None:
    """The configuration should expose deterministic Volume paths."""
    configuration = build_configuration()

    assert configuration.volume_root == Path(
        "/Volumes/workspace/nyc_taxi/pipeline_data"
    )
    assert configuration.landing_directory == Path(
        "/Volumes/workspace/nyc_taxi/pipeline_data/landing"
    )


def test_configuration_builds_managed_table_names() -> None:
    """Every pipeline layer should use fully qualified table names."""
    configuration = build_configuration()

    assert (
        configuration.bronze_trip_table == "workspace.nyc_taxi.bronze_yellow_taxi_trips"
    )
    assert (
        configuration.bronze_taxi_zone_table == "workspace.nyc_taxi.bronze_taxi_zones"
    )
    assert (
        configuration.silver_accepted_table
        == "workspace.nyc_taxi.silver_yellow_taxi_trips_accepted"
    )
    assert (
        configuration.silver_rejected_table
        == "workspace.nyc_taxi.silver_yellow_taxi_trips_rejected"
    )
    assert configuration.gold_trip_metrics_table == (
        "workspace.nyc_taxi.gold_trip_metrics_by_date_pickup_zone_payment"
    )
    assert (
        configuration.gold_daily_metrics_table
        == "workspace.nyc_taxi.gold_daily_trip_metrics"
    )


def test_configuration_is_immutable() -> None:
    """Databricks pipeline configuration should remain immutable."""
    configuration = build_configuration()

    with pytest.raises(FrozenInstanceError):
        configuration.catalog = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("value", "field_name"),
    [
        ("", "catalog"),
        ("contains-hyphen", "catalog"),
        ("9starts_with_digit", "schema"),
        ("contains space", "volume"),
        (None, "catalog"),
        (True, "schema"),
    ],
)
def test_configuration_rejects_invalid_identifiers(
    value: object,
    field_name: str,
) -> None:
    """Unity Catalog identifiers should use the supported simple form."""
    arguments = {
        "catalog": "workspace",
        "schema": "nyc_taxi",
        "volume": "pipeline_data",
    }
    arguments[field_name] = value

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        DatabricksPipelineConfiguration(**arguments)  # type: ignore[arg-type]


def test_table_name_validates_requested_identifier() -> None:
    """Dynamic table names should receive the same identifier validation."""
    configuration = build_configuration()

    assert configuration.table_name("analytics") == "workspace.nyc_taxi.analytics"

    with pytest.raises(
        ValueError,
        match="table",
    ):
        configuration.table_name("invalid-table")


@pytest.mark.parametrize(
    "value",
    [
        "_private",
        "table_2024",
        "TableName",
    ],
)
def test_require_unquoted_identifier_accepts_supported_values(
    value: str,
) -> None:
    """Simple SQL identifiers should be returned unchanged."""
    assert (
        require_unquoted_identifier(
            value,
            "identifier",
        )
        == value
    )
