"""Tests for the sample profiling command-line interface."""

from pathlib import Path

from taxi_lakehouse import sample_profile_cli
from taxi_lakehouse.sample_artifacts import (
    GeneratedSampleFile,
)


class FakeSparkContext:
    """Record the configured Spark log level."""

    def __init__(self) -> None:
        self.log_level: str | None = None

    def setLogLevel(
        self,
        log_level: str,
    ) -> None:
        """Record the requested log level."""
        self.log_level = log_level


class FakeSparkSession:
    """Provide the Spark methods required by the CLI."""

    def __init__(self) -> None:
        self.sparkContext = FakeSparkContext()
        self.stopped = False

    def stop(self) -> None:
        """Record that the Spark session was stopped."""
        self.stopped = True


def test_parse_arguments_uses_project_defaults() -> None:
    """The CLI should use versioned project paths by default."""
    arguments = sample_profile_cli.parse_arguments([])

    assert arguments.specification == Path("data/sample/sample_spec.json")

    assert arguments.output == Path("data/sample/sample_profile.json")


def test_main_generates_profile_and_prints_summary(
    monkeypatch,
    capsys,
) -> None:
    """The default command should generate and report the profile."""
    spark = FakeSparkSession()
    calls = {}

    monkeypatch.setattr(
        sample_profile_cli,
        "create_spark_session",
        lambda: spark,
    )

    def fake_generate_sample_profile_artifact(
        actual_spark,
        specification_path: Path,
        output_path: Path,
    ) -> GeneratedSampleFile:
        calls["spark"] = actual_spark
        calls["specification_path"] = specification_path
        calls["output_path"] = output_path

        return GeneratedSampleFile(
            file_path=output_path,
            content_length_bytes=321,
            sha256="a" * 64,
        )

    monkeypatch.setattr(
        sample_profile_cli,
        "generate_sample_profile_artifact",
        fake_generate_sample_profile_artifact,
    )

    exit_code = sample_profile_cli.main([])

    captured_output = capsys.readouterr().out

    assert exit_code == 0
    assert calls["spark"] is spark
    assert calls["specification_path"] == Path("data/sample/sample_spec.json")
    assert calls["output_path"] == Path("data/sample/sample_profile.json")

    assert spark.sparkContext.log_level == "ERROR"
    assert spark.stopped is True

    assert (
        "SAMPLE_PROFILE\t"
        "path=data/sample/sample_profile.json\t"
        "bytes=321\t"
        f"sha256={'a' * 64}"
    ) in captured_output

    assert "SAMPLE_PROFILE_COMPLETE" in (captured_output)


def test_main_accepts_custom_paths(
    monkeypatch,
) -> None:
    """Explicit specification and output paths should be forwarded."""
    spark = FakeSparkSession()
    calls = {}

    monkeypatch.setattr(
        sample_profile_cli,
        "create_spark_session",
        lambda: spark,
    )

    def fake_generate_sample_profile_artifact(
        actual_spark,
        specification_path: Path,
        output_path: Path,
    ) -> GeneratedSampleFile:
        calls["spark"] = actual_spark
        calls["specification_path"] = specification_path
        calls["output_path"] = output_path

        return GeneratedSampleFile(
            file_path=output_path,
            content_length_bytes=123,
            sha256="b" * 64,
        )

    monkeypatch.setattr(
        sample_profile_cli,
        "generate_sample_profile_artifact",
        fake_generate_sample_profile_artifact,
    )

    exit_code = sample_profile_cli.main(
        [
            "--specification",
            "custom/specification.json",
            "--output",
            "custom/profile.json",
        ]
    )

    assert exit_code == 0
    assert calls["spark"] is spark
    assert calls["specification_path"] == Path("custom/specification.json")
    assert calls["output_path"] == Path("custom/profile.json")
    assert spark.stopped is True
