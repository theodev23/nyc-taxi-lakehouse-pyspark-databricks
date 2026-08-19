"""Tests for the Gold analytical specification."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from taxi_lakehouse.gold_analytics_specification import (
    GoldAnalyticsSpecification,
    load_gold_analytics_specification,
)

PROJECT_SPECIFICATION_PATH = Path("data/gold_analytics_spec.json")


def load_project_mapping() -> dict:
    """Load a mutable copy of the project Gold specification."""
    return json.loads(PROJECT_SPECIFICATION_PATH.read_text(encoding="utf-8"))


def write_specification(
    tmp_path: Path,
    specification: object,
) -> Path:
    """Write one temporary Gold specification."""
    specification_path = tmp_path / "gold_analytics_spec.json"
    specification_path.write_text(
        json.dumps(
            specification,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return specification_path


def test_project_gold_analytics_specification_loads() -> None:
    """The project Gold specification should load into typed records."""
    specification = load_gold_analytics_specification(PROJECT_SPECIFICATION_PATH)

    assert isinstance(
        specification,
        GoldAnalyticsSpecification,
    )
    assert specification.source.accepted_trip_table == Path(
        "data/lakehouse/silver/yellow_taxi_trips_accepted"
    )
    assert specification.source.taxi_zone_table == Path(
        "data/lakehouse/bronze/taxi_zones"
    )
    assert specification.outputs.trip_metrics.table == Path(
        "data/lakehouse/gold/trip_metrics_by_date_pickup_zone_payment"
    )
    assert specification.outputs.trip_metrics.grain == (
        "_source_month",
        "pickup_date",
        "pickup_location_id",
        "payment_type",
    )
    assert specification.outputs.trip_metrics.dimensions == (
        "pickup_borough",
        "pickup_zone",
        "pickup_service_zone",
    )
    assert specification.outputs.daily_metrics.grain == (
        "_source_month",
        "pickup_date",
    )
    assert tuple(metric.name for metric in specification.metrics) == (
        "trip_count",
        "quality_flagged_trip_count",
        "trip_distance_sum",
        "trip_distance_avg",
        "trip_duration_minutes_avg",
        "fare_amount_sum",
        "tip_amount_sum",
        "total_amount_sum",
        "total_amount_avg",
    )


def test_gold_analytics_specification_is_immutable() -> None:
    """Loaded Gold specification records should be immutable."""
    specification = load_gold_analytics_specification(PROJECT_SPECIFICATION_PATH)

    with pytest.raises(FrozenInstanceError):
        specification.description = "changed"


def test_gold_analytics_specification_rejects_unknown_schema(
    tmp_path: Path,
) -> None:
    """Unsupported Gold specification versions should fail."""
    specification = load_project_mapping()
    specification["schema_version"] = "2.0"

    with pytest.raises(
        ValueError,
        match="Unsupported Gold analytical specification schema version",
    ):
        load_gold_analytics_specification(
            write_specification(
                tmp_path,
                specification,
            )
        )


def test_gold_analytics_specification_requires_distinct_sources(
    tmp_path: Path,
) -> None:
    """Gold trip and taxi-zone sources must be different."""
    specification = load_project_mapping()
    specification["source"]["taxi_zone_table"] = specification["source"][
        "accepted_trip_table"
    ]

    with pytest.raises(
        ValueError,
        match="Gold source tables must be different",
    ):
        load_gold_analytics_specification(
            write_specification(
                tmp_path,
                specification,
            )
        )


def test_gold_analytics_specification_requires_distinct_outputs(
    tmp_path: Path,
) -> None:
    """Gold analytical outputs must use different tables."""
    specification = load_project_mapping()
    specification["outputs"]["daily_metrics"]["table"] = specification["outputs"][
        "trip_metrics"
    ]["table"]

    with pytest.raises(
        ValueError,
        match="Gold output tables must be different",
    ):
        load_gold_analytics_specification(
            write_specification(
                tmp_path,
                specification,
            )
        )


def test_gold_analytics_specification_requires_month_partition(
    tmp_path: Path,
) -> None:
    """Gold outputs should remain partitioned by source month."""
    specification = load_project_mapping()
    specification["outputs"]["trip_metrics"]["partition_column"] = "pickup_date"

    with pytest.raises(
        ValueError,
        match="partition_column must be _source_month",
    ):
        load_gold_analytics_specification(
            write_specification(
                tmp_path,
                specification,
            )
        )


def test_gold_analytics_specification_requires_partition_in_grain(
    tmp_path: Path,
) -> None:
    """Every Gold output grain should include its partition column."""
    specification = load_project_mapping()
    specification["outputs"]["daily_metrics"]["grain"] = [
        "pickup_date",
    ]

    with pytest.raises(
        ValueError,
        match="grain must contain the partition column",
    ):
        load_gold_analytics_specification(
            write_specification(
                tmp_path,
                specification,
            )
        )


def test_gold_analytics_specification_rejects_grain_dimension_overlap(
    tmp_path: Path,
) -> None:
    """Gold grain and descriptive dimensions must remain distinct."""
    specification = load_project_mapping()
    specification["outputs"]["trip_metrics"]["dimensions"].append("pickup_date")

    with pytest.raises(
        ValueError,
        match="grain and dimensions must not overlap",
    ):
        load_gold_analytics_specification(
            write_specification(
                tmp_path,
                specification,
            )
        )


def test_gold_analytics_specification_rejects_duplicate_metrics(
    tmp_path: Path,
) -> None:
    """Gold metric names should be unique."""
    specification = load_project_mapping()
    specification["metrics"].append(dict(specification["metrics"][0]))

    with pytest.raises(
        ValueError,
        match="metrics must not contain duplicate metric names",
    ):
        load_gold_analytics_specification(
            write_specification(
                tmp_path,
                specification,
            )
        )


def test_gold_analytics_specification_rejects_non_object_root(
    tmp_path: Path,
) -> None:
    """The Gold JSON root must be an object."""
    specification_path = write_specification(
        tmp_path,
        [],
    )

    with pytest.raises(
        ValueError,
        match="Gold analytical specification root must be an object",
    ):
        load_gold_analytics_specification(specification_path)
