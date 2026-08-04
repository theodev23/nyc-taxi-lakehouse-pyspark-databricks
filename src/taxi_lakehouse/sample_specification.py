"""Utilities for loading the deterministic sample specification."""

import json
from dataclasses import dataclass
from datetime import datetime
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


def _require_positive_integer(
    value: Any,
    field_name: str,
) -> int:
    """Return a validated positive integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")

    return value


def _require_unique_string_tuple(
    value: Any,
    field_name: str,
) -> tuple[str, ...]:
    """Return a validated tuple of unique non-empty strings."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list.")

    items = tuple(
        _require_non_empty_string(
            item,
            f"{field_name} item",
        )
        for item in value
    )

    if len(items) != len(set(items)):
        raise ValueError(f"{field_name} must not contain duplicates.")

    return items


@dataclass(frozen=True, slots=True)
class SampleSourceProfile:
    """Profile of the source dataset used to design the sample."""

    trip_row_count: int
    trip_column_count: int
    schema_signature: str
    taxi_zone_row_count: int

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
    ) -> "SampleSourceProfile":
        """Build and validate source profiling metadata."""
        schema_signature = _require_non_empty_string(
            value["schema_signature"],
            "source_profile.schema_signature",
        )

        if len(schema_signature) != 16 or any(
            character not in "0123456789abcdef" for character in schema_signature
        ):
            raise ValueError(
                "source_profile.schema_signature must contain "
                "16 lowercase hexadecimal characters."
            )

        return cls(
            trip_row_count=_require_positive_integer(
                value["trip_row_count"],
                "source_profile.trip_row_count",
            ),
            trip_column_count=_require_positive_integer(
                value["trip_column_count"],
                "source_profile.trip_column_count",
            ),
            schema_signature=schema_signature,
            taxi_zone_row_count=_require_positive_integer(
                value["taxi_zone_row_count"],
                "source_profile.taxi_zone_row_count",
            ),
        )


@dataclass(frozen=True, slots=True)
class SampleQuota:
    """Number of real source rows selected for one quality bucket."""

    quality_bucket: str
    rows_per_month: int


@dataclass(frozen=True, slots=True)
class SampleSelection:
    """Deterministic row-selection configuration."""

    algorithm: str
    timezone: str
    partition_keys: tuple[str, ...]
    preserve_source_schema: bool
    include_sampling_columns_in_output: bool

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
    ) -> "SampleSelection":
        """Build and validate the deterministic selection configuration."""
        algorithm = _require_non_empty_string(
            value["algorithm"],
            "selection.algorithm",
        )
        timezone = _require_non_empty_string(
            value["timezone"],
            "selection.timezone",
        )
        partition_keys = _require_unique_string_tuple(
            value["partition_keys"],
            "selection.partition_keys",
        )
        preserve_source_schema = value["preserve_source_schema"]
        include_sampling_columns = value["include_sampling_columns_in_output"]

        if algorithm != "sha256_row_rank_v1":
            raise ValueError("Unsupported sample selection algorithm.")

        if timezone != "UTC":
            raise ValueError("Sample selection timezone must be UTC.")

        if partition_keys != (
            "source_month",
            "quality_bucket",
        ):
            raise ValueError(
                "Sample selection partition keys must be "
                "source_month and quality_bucket."
            )

        if preserve_source_schema is not True:
            raise ValueError("Sample selection must preserve the source schema.")

        if include_sampling_columns is not False:
            raise ValueError("Sampling columns must not be included in sample output.")

        return cls(
            algorithm=algorithm,
            timezone=timezone,
            partition_keys=partition_keys,
            preserve_source_schema=preserve_source_schema,
            include_sampling_columns_in_output=(include_sampling_columns),
        )


@dataclass(frozen=True, slots=True)
class SampleOutput:
    """Paths of the generated deterministic sample artifacts."""

    trip_file_pattern: str
    taxi_zone_lookup_path: Path
    sample_manifest_path: Path

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
    ) -> "SampleOutput":
        """Build and validate the sample output configuration."""
        trip_file_pattern = _require_non_empty_string(
            value["trip_file_pattern"],
            "output.trip_file_pattern",
        )

        if "{source_month}" not in trip_file_pattern:
            raise ValueError(
                "output.trip_file_pattern must contain the {source_month} placeholder."
            )

        return cls(
            trip_file_pattern=trip_file_pattern,
            taxi_zone_lookup_path=Path(
                _require_non_empty_string(
                    value["taxi_zone_lookup_path"],
                    "output.taxi_zone_lookup_path",
                )
            ),
            sample_manifest_path=Path(
                _require_non_empty_string(
                    value["sample_manifest_path"],
                    "output.sample_manifest_path",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class SampleSpecification:
    """Validated deterministic sample specification."""

    description: str
    source_manifest_path: Path
    source_profile: SampleSourceProfile
    source_months: tuple[str, ...]
    quality_bucket_priority: tuple[str, ...]
    real_sample_quotas_per_month: tuple[SampleQuota, ...]
    synthetic_only_buckets: tuple[str, ...]
    selection: SampleSelection
    expected_rows_per_month: int
    expected_total_trip_rows: int
    output: SampleOutput

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
    ) -> "SampleSpecification":
        """Build and validate a sample specification mapping."""
        if value.get("schema_version") != "1.0":
            raise ValueError("Unsupported sample specification schema version.")

        source_months = _require_unique_string_tuple(
            value["source_months"],
            "source_months",
        )

        for source_month in source_months:
            try:
                parsed_month = datetime.strptime(
                    source_month,
                    "%Y-%m",
                )
            except ValueError as error:
                raise ValueError(f"Invalid source month: {source_month!r}.") from error

            if parsed_month.strftime("%Y-%m") != source_month:
                raise ValueError(f"Invalid source month: {source_month!r}.")

        quality_bucket_priority = _require_unique_string_tuple(
            value["quality_bucket_priority"],
            "quality_bucket_priority",
        )

        raw_quotas = value["real_sample_quotas_per_month"]

        if not isinstance(raw_quotas, dict) or not raw_quotas:
            raise ValueError("real_sample_quotas_per_month must be a non-empty object.")

        quotas = tuple(
            SampleQuota(
                quality_bucket=_require_non_empty_string(
                    quality_bucket,
                    "sample quota quality bucket",
                ),
                rows_per_month=_require_positive_integer(
                    rows_per_month,
                    (f"real_sample_quotas_per_month[{quality_bucket!r}]"),
                ),
            )
            for quality_bucket, rows_per_month in raw_quotas.items()
        )

        synthetic_only_buckets = _require_unique_string_tuple(
            value["synthetic_only_buckets"],
            "synthetic_only_buckets",
        )

        real_buckets = {quota.quality_bucket for quota in quotas}
        synthetic_buckets = set(synthetic_only_buckets)
        priority_buckets = set(quality_bucket_priority)

        if real_buckets & synthetic_buckets:
            raise ValueError(
                "Real and synthetic-only quality buckets must not overlap."
            )

        if real_buckets | synthetic_buckets != priority_buckets:
            raise ValueError(
                "Quality bucket priority must exactly cover "
                "real and synthetic-only buckets."
            )

        expected_rows_per_month = _require_positive_integer(
            value["expected_rows_per_month"],
            "expected_rows_per_month",
        )
        expected_total_trip_rows = _require_positive_integer(
            value["expected_total_trip_rows"],
            "expected_total_trip_rows",
        )

        actual_rows_per_month = sum(quota.rows_per_month for quota in quotas)
        actual_total_trip_rows = actual_rows_per_month * len(source_months)

        if actual_rows_per_month != expected_rows_per_month:
            raise ValueError("Sample quotas do not match expected_rows_per_month.")

        if actual_total_trip_rows != expected_total_trip_rows:
            raise ValueError(
                "Sample quotas and months do not match expected_total_trip_rows."
            )

        raw_source_profile = value["source_profile"]
        raw_selection = value["selection"]
        raw_output = value["output"]

        if not isinstance(raw_source_profile, dict):
            raise ValueError("source_profile must be an object.")

        if not isinstance(raw_selection, dict):
            raise ValueError("selection must be an object.")

        if not isinstance(raw_output, dict):
            raise ValueError("output must be an object.")

        return cls(
            description=_require_non_empty_string(
                value["description"],
                "description",
            ),
            source_manifest_path=Path(
                _require_non_empty_string(
                    value["source_manifest_path"],
                    "source_manifest_path",
                )
            ),
            source_profile=SampleSourceProfile.from_mapping(raw_source_profile),
            source_months=source_months,
            quality_bucket_priority=quality_bucket_priority,
            real_sample_quotas_per_month=quotas,
            synthetic_only_buckets=synthetic_only_buckets,
            selection=SampleSelection.from_mapping(raw_selection),
            expected_rows_per_month=expected_rows_per_month,
            expected_total_trip_rows=expected_total_trip_rows,
            output=SampleOutput.from_mapping(raw_output),
        )

    def quota_for(self, quality_bucket: str) -> int:
        """Return the configured monthly quota for a real bucket."""
        for quota in self.real_sample_quotas_per_month:
            if quota.quality_bucket == quality_bucket:
                return quota.rows_per_month

        raise KeyError(quality_bucket)


def load_sample_specification(
    specification_path: Path,
) -> SampleSpecification:
    """Load a deterministic sample specification from JSON."""
    raw_specification = json.loads(specification_path.read_text(encoding="utf-8"))

    if not isinstance(raw_specification, dict):
        raise ValueError("Sample specification root must be an object.")

    return SampleSpecification.from_mapping(raw_specification)
