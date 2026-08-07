"""Tests for the Silver data-quality command-line interface."""

from pathlib import Path

import pytest

import taxi_lakehouse.silver_cli as silver_cli
from taxi_lakehouse.silver_orchestration import (
    SilverBuildResult,
    SilverMonthWriteResult,
)
from taxi_lakehouse.silver_quality_specification import (
    load_silver_quality_specification,
)

PROJECT_SPECIFICATION_PATH = Path("data/silver_quality_spec.json")


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


def build_silver_result() -> SilverBuildResult:
    """Build representative Silver processing metadata."""
    specification = load_silver_quality_specification(PROJECT_SPECIFICATION_PATH)

    monthly_writes = (
        SilverMonthWriteResult(
            source_month="2024-01",
            accepted_destination_path=(specification.output.accepted_table),
            rejected_destination_path=(specification.output.rejected_table),
            accepted_row_count=100,
            rejected_row_count=1,
        ),
        SilverMonthWriteResult(
            source_month="2024-02",
            accepted_destination_path=(specification.output.accepted_table),
            rejected_destination_path=(specification.output.rejected_table),
            accepted_row_count=200,
            rejected_row_count=2,
        ),
    )

    return SilverBuildResult(
        specification=specification,
        source_months=(
            "2024-01",
            "2024-02",
        ),
        zone_id_count=265,
        monthly_writes=monthly_writes,
    )


def test_build_spark_session_enables_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Silver CLI should request a Delta-enabled Spark session."""
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
        silver_cli,
        "build_local_spark_session",
        fake_build_local_spark_session,
    )

    result = silver_cli.build_spark_session()

    assert result is spark
    assert captured_arguments == {
        "app_name": "nyc-taxi-silver-build",
        "enable_delta": True,
        "master": "local[2]",
    }


def test_main_uses_default_specification_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default command should build and summarize every Silver month."""
    spark = FakeSparkSession()
    result = build_silver_result()
    captured_arguments: dict[str, object] = {}

    def fake_build_silver_dataset(
        spark_argument: object,
        specification_path: Path,
    ) -> SilverBuildResult:
        captured_arguments.update(
            {
                "spark": spark_argument,
                "specification_path": specification_path,
            }
        )
        return result

    monkeypatch.setattr(
        silver_cli,
        "build_spark_session",
        lambda: spark,
    )
    monkeypatch.setattr(
        silver_cli,
        "build_silver_dataset",
        fake_build_silver_dataset,
    )

    exit_code = silver_cli.main([])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert spark.sparkContext.log_levels == ["ERROR"]
    assert spark.stop_count == 1
    assert captured_arguments == {
        "spark": spark,
        "specification_path": (silver_cli.DEFAULT_SPECIFICATION_PATH),
    }

    assert output.count("SILVER_MONTH\tmonth=") == 2
    assert (
        "SILVER_MONTH\tmonth=2024-01"
        "\taccepted_rows=100"
        "\trejected_rows=1"
        "\ttotal_rows=101"
    ) in output
    assert (
        "SILVER_MONTH\tmonth=2024-02"
        "\taccepted_rows=200"
        "\trejected_rows=2"
        "\ttotal_rows=202"
    ) in output
    assert (
        "SILVER_BUILD_COMPLETE "
        "months=2 "
        "rows=303 "
        "accepted_rows=300 "
        "rejected_rows=3 "
        "zone_ids=265"
    ) in output


def test_main_forwards_specification_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An explicit specification path should reach orchestration."""
    spark = FakeSparkSession()
    specification_path = tmp_path / "custom-silver-spec.json"
    received_path: Path | None = None

    def fake_build_silver_dataset(
        spark_argument: object,
        path: Path,
    ) -> SilverBuildResult:
        nonlocal received_path

        assert spark_argument is spark
        received_path = path
        return build_silver_result()

    monkeypatch.setattr(
        silver_cli,
        "build_spark_session",
        lambda: spark,
    )
    monkeypatch.setattr(
        silver_cli,
        "build_silver_dataset",
        fake_build_silver_dataset,
    )

    exit_code = silver_cli.main(
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
    """Spark should stop while Silver processing errors remain visible."""
    spark = FakeSparkSession()

    def fail_build(
        spark_argument: object,
        specification_path: Path,
    ) -> SilverBuildResult:
        assert spark_argument is spark
        raise RuntimeError("synthetic Silver processing failure")

    monkeypatch.setattr(
        silver_cli,
        "build_spark_session",
        lambda: spark,
    )
    monkeypatch.setattr(
        silver_cli,
        "build_silver_dataset",
        fail_build,
    )

    with pytest.raises(
        RuntimeError,
        match="synthetic Silver processing failure",
    ):
        silver_cli.main([])

    assert spark.sparkContext.log_levels == ["ERROR"]
    assert spark.stop_count == 1
