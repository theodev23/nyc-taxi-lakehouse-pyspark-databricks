"""Utilities for loading the Silver data-quality specification."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _require_non_empty_string(
    value: Any,
    field_name: str,
) -> str:
    """Return a validated non-empty string."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")

    return value


def _require_mapping(
    value: Any,
    field_name: str,
) -> dict[str, Any]:
    """Return a validated mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object.")

    return value


def _require_true(
    value: Any,
    field_name: str,
) -> bool:
    """Require one configuration flag to be true."""
    if value is not True:
        raise ValueError(f"{field_name} must be true.")

    return value


@dataclass(frozen=True, slots=True)
class SilverQualityRule:
    """One named Silver data-quality rule."""

    name: str
    description: str

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
        field_name: str,
    ) -> "SilverQualityRule":
        """Build and validate one quality rule."""
        return cls(
            name=_require_non_empty_string(
                value["name"],
                f"{field_name}.name",
            ),
            description=_require_non_empty_string(
                value["description"],
                f"{field_name}.description",
            ),
        )


def _load_rules(
    value: Any,
    field_name: str,
) -> tuple[SilverQualityRule, ...]:
    """Build a validated non-empty tuple of unique rules."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list.")

    rules = tuple(
        SilverQualityRule.from_mapping(
            _require_mapping(
                item,
                f"{field_name} item",
            ),
            f"{field_name}[{index}]",
        )
        for index, item in enumerate(value)
    )

    names = tuple(rule.name for rule in rules)

    if len(names) != len(set(names)):
        raise ValueError(f"{field_name} must not contain duplicate rule names.")

    return rules


@dataclass(frozen=True, slots=True)
class SilverQualityOutput:
    """Destination configuration for Silver Delta tables."""

    accepted_table: Path
    rejected_table: Path
    partition_column: str

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
    ) -> "SilverQualityOutput":
        """Build and validate Silver output configuration."""
        accepted_table = Path(
            _require_non_empty_string(
                value["accepted_table"],
                "output.accepted_table",
            )
        )
        rejected_table = Path(
            _require_non_empty_string(
                value["rejected_table"],
                "output.rejected_table",
            )
        )
        partition_column = _require_non_empty_string(
            value["partition_column"],
            "output.partition_column",
        )

        if accepted_table == rejected_table:
            raise ValueError("Accepted and rejected Silver tables must be different.")

        if partition_column != "_source_month":
            raise ValueError("Silver output partition column must be _source_month.")

        return cls(
            accepted_table=accepted_table,
            rejected_table=rejected_table,
            partition_column=partition_column,
        )


@dataclass(frozen=True, slots=True)
class SilverQualityContract:
    """Validated Silver rejection and quality-flag contract."""

    rejection_reason_column: str
    quality_flag_column: str
    rejection_rules: tuple[SilverQualityRule, ...]
    quality_flag_rules: tuple[SilverQualityRule, ...]

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
    ) -> "SilverQualityContract":
        """Build and validate the Silver quality contract."""
        _require_true(
            value["allow_multiple_rejection_reasons"],
            "quality_contract.allow_multiple_rejection_reasons",
        )
        _require_true(
            value["allow_multiple_quality_flags"],
            "quality_contract.allow_multiple_quality_flags",
        )

        rejection_reason_column = _require_non_empty_string(
            value["rejection_reason_column"],
            "quality_contract.rejection_reason_column",
        )
        quality_flag_column = _require_non_empty_string(
            value["quality_flag_column"],
            "quality_contract.quality_flag_column",
        )

        if rejection_reason_column == quality_flag_column:
            raise ValueError(
                "Rejection-reason and quality-flag columns must be different."
            )

        rejection_rules = _load_rules(
            value["rejection_rules"],
            "quality_contract.rejection_rules",
        )
        quality_flag_rules = _load_rules(
            value["quality_flag_rules"],
            "quality_contract.quality_flag_rules",
        )

        rejection_names = {rule.name for rule in rejection_rules}
        quality_flag_names = {rule.name for rule in quality_flag_rules}

        overlap = rejection_names & quality_flag_names

        if overlap:
            raise ValueError(
                "Rejection rules and quality-flag rules must not overlap: "
                f"{sorted(overlap)!r}."
            )

        return cls(
            rejection_reason_column=rejection_reason_column,
            quality_flag_column=quality_flag_column,
            rejection_rules=rejection_rules,
            quality_flag_rules=quality_flag_rules,
        )


@dataclass(frozen=True, slots=True)
class SilverQualitySpecification:
    """Validated Silver data-quality specification."""

    description: str
    source_layer: str
    source_trip_table: Path
    source_taxi_zone_table: Path
    output: SilverQualityOutput
    quality_contract: SilverQualityContract

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
    ) -> "SilverQualitySpecification":
        """Build and validate a Silver quality specification mapping."""
        if value.get("schema_version") != "1.0":
            raise ValueError("Unsupported Silver quality specification schema version.")

        source_layer = _require_non_empty_string(
            value["source_layer"],
            "source_layer",
        )

        if source_layer != "bronze":
            raise ValueError(
                "Silver quality specification source layer must be bronze."
            )

        return cls(
            description=_require_non_empty_string(
                value["description"],
                "description",
            ),
            source_layer=source_layer,
            source_trip_table=Path(
                _require_non_empty_string(
                    value["source_trip_table"],
                    "source_trip_table",
                )
            ),
            source_taxi_zone_table=Path(
                _require_non_empty_string(
                    value["source_taxi_zone_table"],
                    "source_taxi_zone_table",
                )
            ),
            output=SilverQualityOutput.from_mapping(
                _require_mapping(
                    value["output"],
                    "output",
                )
            ),
            quality_contract=SilverQualityContract.from_mapping(
                _require_mapping(
                    value["quality_contract"],
                    "quality_contract",
                )
            ),
        )


def load_silver_quality_specification(
    specification_path: Path,
) -> SilverQualitySpecification:
    """Load a Silver data-quality specification from JSON."""
    raw_specification = json.loads(specification_path.read_text(encoding="utf-8"))

    if not isinstance(raw_specification, dict):
        raise ValueError("Silver quality specification root must be an object.")

    return SilverQualitySpecification.from_mapping(raw_specification)
