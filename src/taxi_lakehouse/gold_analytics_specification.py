"""Utilities for loading the Gold analytical specification."""

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


def _require_unique_strings(
    value: Any,
    field_name: str,
) -> tuple[str, ...]:
    """Return a validated non-empty tuple of unique strings."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list.")

    strings = tuple(
        _require_non_empty_string(
            item,
            f"{field_name}[{index}]",
        )
        for index, item in enumerate(value)
    )

    if len(strings) != len(set(strings)):
        raise ValueError(f"{field_name} must not contain duplicate values.")

    return strings


@dataclass(frozen=True, slots=True)
class GoldAnalyticsSource:
    """Input Delta tables required by the Gold layer."""

    accepted_trip_table: Path
    taxi_zone_table: Path

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
    ) -> "GoldAnalyticsSource":
        """Build and validate Gold source configuration."""
        accepted_trip_table = Path(
            _require_non_empty_string(
                value["accepted_trip_table"],
                "source.accepted_trip_table",
            )
        )
        taxi_zone_table = Path(
            _require_non_empty_string(
                value["taxi_zone_table"],
                "source.taxi_zone_table",
            )
        )

        if accepted_trip_table == taxi_zone_table:
            raise ValueError("Gold source tables must be different.")

        return cls(
            accepted_trip_table=accepted_trip_table,
            taxi_zone_table=taxi_zone_table,
        )


@dataclass(frozen=True, slots=True)
class GoldAnalyticsOutput:
    """One Gold analytical Delta output."""

    table: Path
    partition_column: str
    grain: tuple[str, ...]
    dimensions: tuple[str, ...]

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
        field_name: str,
    ) -> "GoldAnalyticsOutput":
        """Build and validate one Gold output configuration."""
        table = Path(
            _require_non_empty_string(
                value["table"],
                f"{field_name}.table",
            )
        )
        partition_column = _require_non_empty_string(
            value["partition_column"],
            f"{field_name}.partition_column",
        )
        grain = _require_unique_strings(
            value["grain"],
            f"{field_name}.grain",
        )

        raw_dimensions = value.get("dimensions", [])

        if not isinstance(raw_dimensions, list):
            raise ValueError(f"{field_name}.dimensions must be a list.")

        dimensions = tuple(
            _require_non_empty_string(
                item,
                f"{field_name}.dimensions[{index}]",
            )
            for index, item in enumerate(raw_dimensions)
        )

        if len(dimensions) != len(set(dimensions)):
            raise ValueError(
                f"{field_name}.dimensions must not contain duplicate values."
            )

        if partition_column != "_source_month":
            raise ValueError(f"{field_name}.partition_column must be _source_month.")

        if partition_column not in grain:
            raise ValueError(f"{field_name}.grain must contain the partition column.")

        overlap = set(grain) & set(dimensions)

        if overlap:
            raise ValueError(
                f"{field_name}.grain and dimensions must not overlap: "
                f"{sorted(overlap)!r}."
            )

        return cls(
            table=table,
            partition_column=partition_column,
            grain=grain,
            dimensions=dimensions,
        )


@dataclass(frozen=True, slots=True)
class GoldAnalyticsOutputs:
    """Validated Gold analytical output tables."""

    trip_metrics: GoldAnalyticsOutput
    daily_metrics: GoldAnalyticsOutput

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
    ) -> "GoldAnalyticsOutputs":
        """Build and validate Gold output tables."""
        trip_metrics = GoldAnalyticsOutput.from_mapping(
            _require_mapping(
                value["trip_metrics"],
                "outputs.trip_metrics",
            ),
            "outputs.trip_metrics",
        )
        daily_metrics = GoldAnalyticsOutput.from_mapping(
            _require_mapping(
                value["daily_metrics"],
                "outputs.daily_metrics",
            ),
            "outputs.daily_metrics",
        )

        if trip_metrics.table == daily_metrics.table:
            raise ValueError("Gold output tables must be different.")

        return cls(
            trip_metrics=trip_metrics,
            daily_metrics=daily_metrics,
        )


@dataclass(frozen=True, slots=True)
class GoldMetric:
    """One named Gold business metric."""

    name: str
    description: str

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
        field_name: str,
    ) -> "GoldMetric":
        """Build and validate one Gold metric."""
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


def _load_metrics(
    value: Any,
) -> tuple[GoldMetric, ...]:
    """Build a validated non-empty tuple of unique Gold metrics."""
    if not isinstance(value, list) or not value:
        raise ValueError("metrics must be a non-empty list.")

    metrics = tuple(
        GoldMetric.from_mapping(
            _require_mapping(
                item,
                f"metrics[{index}]",
            ),
            f"metrics[{index}]",
        )
        for index, item in enumerate(value)
    )

    names = tuple(metric.name for metric in metrics)

    if len(names) != len(set(names)):
        raise ValueError("metrics must not contain duplicate metric names.")

    return metrics


@dataclass(frozen=True, slots=True)
class GoldAnalyticsSpecification:
    """Validated Gold analytical specification."""

    description: str
    source: GoldAnalyticsSource
    outputs: GoldAnalyticsOutputs
    metrics: tuple[GoldMetric, ...]

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
    ) -> "GoldAnalyticsSpecification":
        """Build and validate a Gold analytical specification mapping."""
        if value.get("schema_version") != "1.0":
            raise ValueError(
                "Unsupported Gold analytical specification schema version."
            )

        return cls(
            description=_require_non_empty_string(
                value["description"],
                "description",
            ),
            source=GoldAnalyticsSource.from_mapping(
                _require_mapping(
                    value["source"],
                    "source",
                )
            ),
            outputs=GoldAnalyticsOutputs.from_mapping(
                _require_mapping(
                    value["outputs"],
                    "outputs",
                )
            ),
            metrics=_load_metrics(value["metrics"]),
        )


def load_gold_analytics_specification(
    specification_path: Path,
) -> GoldAnalyticsSpecification:
    """Load a Gold analytical specification from JSON."""
    raw_specification = json.loads(specification_path.read_text(encoding="utf-8"))

    if not isinstance(raw_specification, dict):
        raise ValueError("Gold analytical specification root must be an object.")

    return GoldAnalyticsSpecification.from_mapping(raw_specification)
