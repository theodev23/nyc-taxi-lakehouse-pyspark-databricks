"""Tests for the Silver data-quality specification."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from taxi_lakehouse.silver_quality_specification import (
    SilverQualitySpecification,
    load_silver_quality_specification,
)

PROJECT_SPECIFICATION_PATH = Path("data/silver_quality_spec.json")


def load_project_mapping() -> dict:
    """Load a mutable copy of the project Silver specification."""
    return json.loads(PROJECT_SPECIFICATION_PATH.read_text(encoding="utf-8"))


def write_specification(
    tmp_path: Path,
    specification: object,
) -> Path:
    """Write one temporary Silver specification."""
    specification_path = tmp_path / "silver_quality_spec.json"
    specification_path.write_text(
        json.dumps(
            specification,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return specification_path


def test_project_silver_quality_specification_loads() -> None:
    """The project specification should load into typed records."""
    specification = load_silver_quality_specification(PROJECT_SPECIFICATION_PATH)

    assert isinstance(
        specification,
        SilverQualitySpecification,
    )
    assert specification.source_layer == "bronze"
    assert specification.source_trip_table == Path(
        "data/lakehouse/bronze/yellow_taxi_trips"
    )
    assert specification.output.accepted_table == Path(
        "data/lakehouse/silver/yellow_taxi_trips_accepted"
    )
    assert specification.output.rejected_table == Path(
        "data/lakehouse/silver/yellow_taxi_trips_rejected"
    )
    assert specification.output.partition_column == ("_source_month")

    rejection_names = tuple(
        rule.name for rule in (specification.quality_contract.rejection_rules)
    )
    quality_flag_names = tuple(
        rule.name for rule in (specification.quality_contract.quality_flag_rules)
    )

    assert rejection_names == (
        "pickup_zone_unknown",
        "dropoff_zone_unknown",
        "distance_negative",
        "duration_nonpositive",
        "duration_over_24h",
        "pickup_outside_month",
        "passenger_over_6",
    )
    assert quality_flag_names == (
        "total_amount_zero",
        "total_amount_negative",
        "passenger_nonpositive",
        "distance_zero",
        "passenger_missing",
    )


def test_silver_quality_specification_is_immutable() -> None:
    """Loaded specification records should be immutable."""
    specification = load_silver_quality_specification(PROJECT_SPECIFICATION_PATH)

    with pytest.raises(FrozenInstanceError):
        specification.source_layer = "silver"


def test_silver_quality_specification_rejects_unknown_schema(
    tmp_path: Path,
) -> None:
    """Unsupported specification versions should fail explicitly."""
    specification = load_project_mapping()
    specification["schema_version"] = "2.0"

    with pytest.raises(
        ValueError,
        match=("Unsupported Silver quality specification schema version"),
    ):
        load_silver_quality_specification(
            write_specification(
                tmp_path,
                specification,
            )
        )


def test_silver_quality_specification_requires_bronze_source(
    tmp_path: Path,
) -> None:
    """Silver processing should explicitly consume Bronze data."""
    specification = load_project_mapping()
    specification["source_layer"] = "raw"

    with pytest.raises(
        ValueError,
        match="source layer must be bronze",
    ):
        load_silver_quality_specification(
            write_specification(
                tmp_path,
                specification,
            )
        )


def test_silver_quality_specification_requires_distinct_outputs(
    tmp_path: Path,
) -> None:
    """Accepted and rejected rows must use different tables."""
    specification = load_project_mapping()
    specification["output"]["rejected_table"] = specification["output"][
        "accepted_table"
    ]

    with pytest.raises(
        ValueError,
        match=("Accepted and rejected Silver tables must be different"),
    ):
        load_silver_quality_specification(
            write_specification(
                tmp_path,
                specification,
            )
        )


def test_silver_quality_specification_requires_month_partition(
    tmp_path: Path,
) -> None:
    """Silver outputs should remain partitioned by source month."""
    specification = load_project_mapping()
    specification["output"]["partition_column"] = "tpep_pickup_datetime"

    with pytest.raises(
        ValueError,
        match=("Silver output partition column must be _source_month"),
    ):
        load_silver_quality_specification(
            write_specification(
                tmp_path,
                specification,
            )
        )


def test_silver_quality_specification_requires_multiple_reasons(
    tmp_path: Path,
) -> None:
    """The contract must preserve simultaneous rejection reasons."""
    specification = load_project_mapping()
    specification["quality_contract"]["allow_multiple_rejection_reasons"] = False

    with pytest.raises(
        ValueError,
        match=("allow_multiple_rejection_reasons must be true"),
    ):
        load_silver_quality_specification(
            write_specification(
                tmp_path,
                specification,
            )
        )


def test_silver_quality_specification_rejects_duplicate_rules(
    tmp_path: Path,
) -> None:
    """Rule names should be unique within one rule family."""
    specification = load_project_mapping()

    duplicate_rule = dict(specification["quality_contract"]["rejection_rules"][0])
    specification["quality_contract"]["rejection_rules"].append(duplicate_rule)

    with pytest.raises(
        ValueError,
        match="must not contain duplicate rule names",
    ):
        load_silver_quality_specification(
            write_specification(
                tmp_path,
                specification,
            )
        )


def test_silver_quality_specification_rejects_rule_overlap(
    tmp_path: Path,
) -> None:
    """A rule cannot be both a rejection and a quality flag."""
    specification = load_project_mapping()

    specification["quality_contract"]["quality_flag_rules"][0]["name"] = (
        "duration_nonpositive"
    )

    with pytest.raises(
        ValueError,
        match=("Rejection rules and quality-flag rules must not overlap"),
    ):
        load_silver_quality_specification(
            write_specification(
                tmp_path,
                specification,
            )
        )


def test_silver_quality_specification_rejects_non_object_root(
    tmp_path: Path,
) -> None:
    """The JSON root must be an object."""
    specification_path = write_specification(
        tmp_path,
        [],
    )

    with pytest.raises(
        ValueError,
        match=("Silver quality specification root must be an object"),
    ):
        load_silver_quality_specification(specification_path)
