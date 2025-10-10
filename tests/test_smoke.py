import importlib
from pathlib import Path
from typing import List


def test_repository_scaffold_exists():
    """Smoke test that checks the basic repository scaffold exists.

    Ensures that essential files and directories for the starter repo are present.
    This is a minimal guard that the teaching materials and starter structure exist.

    Returns
    -------
    None
        The function asserts presence of files/directories; no return value.
    """
    essential_paths: List[Path] = [
        Path("README.md"),
        Path("requirements.txt"),
        Path("Makefile"),
        Path("src"),
        Path("src/mlops"),
        Path("src/mlops/app.py"),
        Path("src/mlops/flow.py"),
        Path("src/mlops/train.py"),
        Path("tests"),
    ]
    for p in essential_paths:
        assert p.exists(), f"Missing expected path: {p}"


def test_import_mlops_modules():
    """Smoke test that core mlops modules are importable.

    This verifies that the Python package is well-formed and imports cleanly,
    which is a prerequisite for any further exercises (tracking, orchestration, serving).

    Returns
    -------
    None
        The function imports modules and asserts they are not None; no return value.
    """
    modules = ["mlops", "mlops.app", "mlops.flow", "mlops.train"]
    for name in modules:
        mod = importlib.import_module(name)
        assert mod is not None, f"Failed to import module: {name}"
