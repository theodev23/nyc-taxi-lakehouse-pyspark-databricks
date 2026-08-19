"""Shared bootstrap utilities for Databricks source notebooks."""

import sys
from pathlib import Path


def prepare_repository_imports(
    notebook_directory: Path | None = None,
) -> Path:
    """Expose the repository src layout to Databricks notebooks."""
    resolved_notebook_directory = (
        Path.cwd() if notebook_directory is None else notebook_directory.resolve()
    )
    repository_root = resolved_notebook_directory.parents[1]
    source_root = repository_root / "src"
    package_root = source_root / "taxi_lakehouse"

    if not package_root.is_dir():
        raise RuntimeError(f"taxi_lakehouse package not found under {source_root}.")

    source_root_text = source_root.as_posix()

    if source_root_text not in sys.path:
        sys.path.insert(
            0,
            source_root_text,
        )

    return repository_root
