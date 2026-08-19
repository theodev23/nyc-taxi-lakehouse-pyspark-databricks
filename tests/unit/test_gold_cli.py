"""Tests for the Gold analytical command-line interface."""

from pathlib import Path

import pytest

import taxi_lakehouse.gold_cli as gold_cli
from taxi_lakehouse.gold_analytics_specification import (
    load_gold_analytics_specification,
)
from taxi_lakehouse.gold_orchestration import (
    GoldBuildResult,
    GoldMonthWriteResult,
)

PROJECT_SPECIFICATION_PATH = Path("data/gold_analytics_spec.json")


class FakeSparkContext:
    """Record Spark log-level configuration."""

    def __init__(self) -> None:
        self.log_levels: list[str] = []

    def setLogLevel(
        self,
        log_level: str,
    ) -> None:
        """Record one requested Spark log level."""
        self.log_levels.append(log_level)


class FakeSparkSession:
    """Provide the Spark operations required by the CLI."""

    def __init__(self) -> None:
        self.sparkContext = FakeSparkContext()
        self.stop_count = 0

    def stop(self) -> None:
        """Record one Spark shutdown."""
        self.stop_count += 1


def build_gold_result() -> GoldBuildResult:
    """Build representative Gold processing metadata."""
    specification = load_gold_analytics_specification(PROJECT_SPECIFICATION_PATH)

    monthly_writes = (
        GoldMonthWriteResult(
            source_month="2024-01",
            trip_metrics_destination_path=(specification.outputs.trip_metrics.table),
            daily_metrics_destination_path=(specification.outputs.daily_metrics.table),
            trip_metrics_row_count=100,
            daily_metrics_row_count=31,
        ),
        GoldMonthWriteResult(
            source_month="2024-02",
            trip_metrics_destination_path=(specification.outputs.trip_metrics.table),
            daily_metrics_destination_path=(specification.outputs.daily_metrics.table),
            trip_metrics_row_count=200,
            daily_metrics_row_count=29,
        ),
    )

    return GoldBuildResult(
        specification=specification,
        source_months=(
            "2024-01",
            "2024-02",
        ),
        taxi_zone_row_count=265,
        monthly_writes=monthly_writes,
    )


def test_build_spark_session_enables_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Gold CLI should request a bounded Delta-enabled Spark session."""
    spark = FakeSparkSession()
    captured_arguments: dict[str, object] = {}

    def fake_build_local_spark_session(
        app_name: str,
        *,
        enable_delta: bool = False,
        master: str = "local[*]",
    ) -> FakeSparkSession:
        captured_arguments.update(
            {
                "app_name": app_name,
                "enable_delta": enable_delta,
                "master": master,
            }
        )
        return spark

    monkeypatch.setattr(
        gold_cli,
        "build_local_spark_session",
        fake_build_local_spark_session,
    )

    result = gold_cli.build_spark_session()

    assert result is spark
    assert captured_arguments == {
        "app_name": "nyc-taxi-gold-build",
        "enable_delta": True,
        "master": "local[2]",
    }


def test_main_uses_default_specification_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default command should build and summarize every Gold month."""
    spark = FakeSparkSession()
    result = build_gold_result()
    captured_arguments: dict[str, object] = {}

    def fake_build_gold_dataset(
        spark_argument: object,
        specification_path: Path,
    ) -> GoldBuildResult:
        captured_arguments.update(
            {
                "spark": spark_argument,
                "specification_path": specification_path,
            }
        )
        return result

    monkeypatch.setattr(
        gold_cli,
        "build_spark_session",
        lambda: spark,
    )
    monkeypatch.setattr(
        gold_cli,
        "build_gold_dataset",
        fake_build_gold_dataset,
    )

    exit_code = gold_cli.main([])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert spark.sparkContext.log_levels == ["ERROR"]
    assert spark.stop_count == 1
    assert captured_arguments == {
        "spark": spark,
        "specification_path": gold_cli.DEFAULT_SPECIFICATION_PATH,
    }

    assert output.count("GOLD_MONTH\tmonth=") == 2
    assert (
        "GOLD_MONTH\tmonth=2024-01\ttrip_metrics_rows=100\tdaily_metrics_rows=31"
    ) in output
    assert (
        "GOLD_MONTH\tmonth=2024-02\ttrip_metrics_rows=200\tdaily_metrics_rows=29"
    ) in output
    assert (
        "GOLD_BUILD_COMPLETE "
        "months=2 "
        "trip_metrics_rows=300 "
        "daily_metrics_rows=60 "
        "taxi_zone_rows=265"
    ) in output


def test_main_forwards_specification_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An explicit Gold specification path should reach orchestration."""
    spark = FakeSparkSession()
    specification_path = tmp_path / "custom-gold-spec.json"
    received_path: Path | None = None

    def fake_build_gold_dataset(
        spark_argument: object,
        path: Path,
    ) -> GoldBuildResult:
        nonlocal received_path

        assert spark_argument is spark
        received_path = path
        return build_gold_result()

    monkeypatch.setattr(
        gold_cli,
        "build_spark_session",
        lambda: spark,
    )
    monkeypatch.setattr(
        gold_cli,
        "build_gold_dataset",
        fake_build_gold_dataset,
    )

    exit_code = gold_cli.main(
        [
            "--specification",
            str(specification_path),
        ]
    )

    assert exit_code == 0
    assert received_path == specification_path
    assert spark.stop_count == 1


def test_main_stops_spark_when_build_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spark should stop while Gold processing errors remain visible."""
    spark = FakeSparkSession()

    def fail_build(
        spark_argument: object,
        specification_path: Path,
    ) -> GoldBuildResult:
        assert spark_argument is spark
        raise RuntimeError("synthetic Gold processing failure")

    monkeypatch.setattr(
        gold_cli,
        "build_spark_session",
        lambda: spark,
    )
    monkeypatch.setattr(
        gold_cli,
        "build_gold_dataset",
        fail_build,
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic Gold processing failure",
    ):
        gold_cli.main([])

    assert spark.sparkContext.log_levels == ["ERROR"]
    assert spark.stop_count == 1
