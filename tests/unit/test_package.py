"""Tests for the package metadata."""

from importlib.metadata import version

import taxi_lakehouse


def test_package_is_importable() -> None:
    """The project package should be importable."""
    assert taxi_lakehouse.__name__ == "taxi_lakehouse"
    assert taxi_lakehouse.__doc__ == "NYC Taxi lakehouse pipeline package."


def test_distribution_version() -> None:
    """The installed distribution should expose the expected version."""
    assert version("nyc-taxi-lakehouse-pyspark-databricks") == "0.1.0"
