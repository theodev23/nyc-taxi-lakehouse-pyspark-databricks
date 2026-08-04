"""Tests for the deterministic sample specification."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from taxi_lakehouse.sample_specification import (
    SampleSpecification,
    load_sample_specification,
)

PROJECT_SPECIFICATION_PATH = Path("data/sample/sample_spec.json")


def load_project_mapping() -> dict:
    """Load a mutable copy of the project sample specification."""
    return json.loads(PROJECT_SPECIFICATION_PATH.read_text(encoding="utf-8"))


def write_specification(
    tmp_path: Path,
    specification: dict,
) -> Path:
    """Write one temporary sample specification."""
    specification_path = tmp_path / "sample_spec.json"
    specification_path.write_text(
        json.dumps(specification, indent=2) + "\n",
        encoding="utf-8",
    )
    return specification_path


def test_project_sample_specification_loads() -> None:
    """The project specification should load into typed records."""
    specification = load_sample_specification(PROJECT_SPECIFICATION_PATH)

    assert isinstance(specification, SampleSpecification)
    assert specification.source_months == (
        "2024-01",
        "2024-02",
        "2024-03",
        "2024-04",
        "2024-05",
        "2024-06",
    )
    assert specification.source_profile.trip_row_count == 20_332_093
    assert specification.expected_rows_per_month == 250
    assert specification.expected_total_trip_rows == 1_500
    assert specification.quota_for("normal") == 100
    assert specification.quota_for("duration_over_24h") == 5
    assert specification.output.taxi_zone_lookup_path == Path(
        "data/sample/taxi_zone_lookup.csv"
    )


def test_sample_specification_is_immutable() -> None:
    """Loaded specification records should be immutable."""
    specification = load_sample_specification(PROJECT_SPECIFICATION_PATH)

    with pytest.raises(FrozenInstanceError):
        specification.expected_rows_per_month = 1


def test_sample_specification_rejects_unknown_schema(
    tmp_path: Path,
) -> None:
    """An unsupported specification schema should fail explicitly."""
    specification = load_project_mapping()
    specification["schema_version"] = "2.0"

    specification_path = write_specification(
        tmp_path,
        specification,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported sample specification schema version",
    ):
        load_sample_specification(specification_path)


def test_sample_specification_rejects_invalid_month(
    tmp_path: Path,
) -> None:
    """Malformed source months should fail before generation."""
    specification = load_project_mapping()
    specification["source_months"][0] = "2024-13"

    specification_path = write_specification(
        tmp_path,
        specification,
    )

    with pytest.raises(
        ValueError,
        match="Invalid source month",
    ):
        load_sample_specification(specification_path)


def test_sample_specification_rejects_nonpositive_quota(
    tmp_path: Path,
) -> None:
    """Every real sample bucket should have a positive quota."""
    specification = load_project_mapping()
    specification["real_sample_quotas_per_month"]["normal"] = 0

    specification_path = write_specification(
        tmp_path,
        specification,
    )

    with pytest.raises(
        ValueError,
        match="must be a positive integer",
    ):
        load_sample_specification(specification_path)


def test_sample_specification_rejects_inconsistent_total(
    tmp_path: Path,
) -> None:
    """Declared totals should match configured quotas and months."""
    specification = load_project_mapping()
    specification["expected_total_trip_rows"] = 1_499

    specification_path = write_specification(
        tmp_path,
        specification,
    )

    with pytest.raises(
        ValueError,
        match="do not match expected_total_trip_rows",
    ):
        load_sample_specification(specification_path)


def test_sample_specification_rejects_incomplete_bucket_priority(
    tmp_path: Path,
) -> None:
    """The priority should cover every real and synthetic bucket."""
    specification = load_project_mapping()
    specification["quality_bucket_priority"].remove("normal")

    specification_path = write_specification(
        tmp_path,
        specification,
    )

    with pytest.raises(
        ValueError,
        match="must exactly cover",
    ):
        load_sample_specification(specification_path)
