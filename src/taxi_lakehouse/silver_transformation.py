"""PySpark transformations for Silver trip data quality."""

from dataclasses import dataclass

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from taxi_lakehouse.quality_rules import (
    QUALITY_RULE_NAMES,
    build_quality_rule_conditions,
    source_month_bounds,
)
from taxi_lakehouse.silver_quality_specification import (
    SilverQualitySpecification,
)


@dataclass(frozen=True, slots=True)
class SilverQualitySplit:
    """Accepted and rejected Silver trip rows."""

    accepted: DataFrame
    rejected: DataFrame


def _rule_names(
    specification: SilverQualitySpecification,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return validated rejection and quality-flag rule names."""
    rejection_names = tuple(
        rule.name for rule in specification.quality_contract.rejection_rules
    )
    quality_flag_names = tuple(
        rule.name for rule in specification.quality_contract.quality_flag_rules
    )

    configured_names = set(rejection_names) | set(quality_flag_names)
    unsupported_names = configured_names - QUALITY_RULE_NAMES

    if unsupported_names:
        raise ValueError(
            f"Unsupported Silver quality rules: {sorted(unsupported_names)!r}."
        )

    return rejection_names, quality_flag_names


def _rule_array_expression(
    conditions: dict[str, Column],
    rule_names: tuple[str, ...],
) -> Column:
    """Build an ordered array containing every matching rule name."""
    rule_values = F.array(
        *[
            F.when(
                conditions[rule_name],
                F.lit(rule_name),
            )
            for rule_name in rule_names
        ]
    )

    return F.filter(
        rule_values,
        lambda rule_name: rule_name.isNotNull(),
    )


def annotate_silver_quality(
    frame: DataFrame,
    source_month: str,
    zone_ids: tuple[int, ...],
    specification: SilverQualitySpecification,
) -> DataFrame:
    """Add Silver rejection reasons and quality flags to one trip month."""
    rejection_names, quality_flag_names = _rule_names(specification)

    month_start, month_end = source_month_bounds(source_month)

    month_start_literal = F.lit(month_start.strftime("%Y-%m-%d %H:%M:%S")).cast(
        "timestamp_ntz"
    )
    month_end_literal = F.lit(month_end.strftime("%Y-%m-%d %H:%M:%S")).cast(
        "timestamp_ntz"
    )

    conditions = build_quality_rule_conditions(
        zone_ids,
        month_start_literal,
        month_end_literal,
    )

    rejection_reason_column = specification.quality_contract.rejection_reason_column
    quality_flag_column = specification.quality_contract.quality_flag_column

    return frame.withColumn(
        rejection_reason_column,
        _rule_array_expression(
            conditions,
            rejection_names,
        ),
    ).withColumn(
        quality_flag_column,
        _rule_array_expression(
            conditions,
            quality_flag_names,
        ),
    )


def split_silver_quality_rows(
    frame: DataFrame,
    specification: SilverQualitySpecification,
) -> SilverQualitySplit:
    """Split annotated Silver rows into accepted and rejected DataFrames."""
    rejection_reason_column = specification.quality_contract.rejection_reason_column

    if rejection_reason_column not in frame.columns:
        raise ValueError(
            "Annotated Silver DataFrame must contain "
            f"column {rejection_reason_column!r}."
        )

    rejection_count = F.size(F.col(rejection_reason_column))

    return SilverQualitySplit(
        accepted=frame.filter(rejection_count == 0),
        rejected=frame.filter(rejection_count > 0),
    )
