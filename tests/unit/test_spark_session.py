"""Tests for shared local Spark session construction."""

import pytest

from taxi_lakehouse import spark_session


class FakeBuilder:
    """Record Spark builder configuration calls."""

    def __init__(self) -> None:
        self.master_value: str | None = None
        self.app_name: str | None = None
        self.configuration: dict[str, str] = {}
        self.created_session = object()

    def master(self, value: str) -> "FakeBuilder":
        self.master_value = value
        return self

    def appName(self, value: str) -> "FakeBuilder":
        self.app_name = value
        return self

    def config(self, key: str, value: str) -> "FakeBuilder":
        self.configuration[key] = value
        return self

    def getOrCreate(self) -> object:
        return self.created_session


def install_fake_builder(
    monkeypatch: pytest.MonkeyPatch,
    builder: FakeBuilder,
) -> None:
    """Replace the PySpark session builder used by the module."""

    class FakeSparkSession:
        pass

    FakeSparkSession.builder = builder

    monkeypatch.setattr(
        spark_session,
        "SparkSession",
        FakeSparkSession,
    )


def test_build_local_spark_session_uses_shared_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A standard session should use the shared local configuration."""
    builder = FakeBuilder()
    install_fake_builder(monkeypatch, builder)

    result = spark_session.build_local_spark_session("sample-app")

    assert result is builder.created_session
    assert builder.master_value == "local[*]"
    assert builder.app_name == "sample-app"
    assert builder.configuration == {
        "spark.ui.enabled": "false",
        "spark.sql.session.timeZone": "UTC",
    }


def test_build_local_spark_session_configures_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delta sessions should configure extensions, catalog, and packages."""
    builder = FakeBuilder()
    install_fake_builder(monkeypatch, builder)

    configured_builders: list[FakeBuilder] = []

    def fake_configure_delta(actual_builder: FakeBuilder) -> FakeBuilder:
        configured_builders.append(actual_builder)
        return actual_builder

    monkeypatch.setattr(
        spark_session,
        "configure_spark_with_delta_pip",
        fake_configure_delta,
    )

    result = spark_session.build_local_spark_session(
        "bronze-ingestion",
        enable_delta=True,
    )

    assert result is builder.created_session
    assert configured_builders == [builder]
    assert builder.configuration == {
        "spark.ui.enabled": "false",
        "spark.sql.session.timeZone": "UTC",
        "spark.sql.extensions": spark_session.DELTA_EXTENSION,
        "spark.sql.catalog.spark_catalog": spark_session.DELTA_CATALOG,
    }
