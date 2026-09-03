"""Architecture guard: ``perimeter.core`` may import nothing but the standard library.

This is the dependency rule from CLAUDE.md, enforced as a test so that a
violation fails CI rather than being caught in review. The access-control
logic must be unit-testable without a network, a database, or NumPy.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

CORE_DIR = Path(__file__).resolve().parents[1] / "perimeter" / "core"


def _imported_top_level_modules(source: str, filename: str) -> set[str]:
    tree = ast.parse(source, filename=filename)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # Relative import inside the core package: allowed.
                continue
            if node.module is None:
                continue
            names.add(node.module.split(".")[0])
    return names


def _core_files() -> list[Path]:
    return sorted(CORE_DIR.rglob("*.py"))


def test_core_package_exists() -> None:
    assert CORE_DIR.is_dir(), f"expected core package at {CORE_DIR}"
    assert (CORE_DIR / "__init__.py").is_file()


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: p.name)
def test_core_imports_only_stdlib(path: Path) -> None:
    imported = _imported_top_level_modules(path.read_text(), str(path))
    allowed = set(sys.stdlib_module_names) | {"perimeter"}
    offenders = sorted(imported - allowed)
    assert not offenders, (
        f"{path.relative_to(CORE_DIR.parents[1])} imports non-stdlib modules: "
        f"{offenders}. perimeter.core must depend on the standard library only."
    )


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: p.name)
def test_core_does_not_import_outside_core(path: Path) -> None:
    """``perimeter.core`` may import itself, never adapters/index/pipeline/server."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("perimeter.")
        ):
            assert node.module.startswith("perimeter.core"), (
                f"{path.name} imports {node.module}; core must not depend on outer layers"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("perimeter."):
                    assert alias.name.startswith("perimeter.core"), (
                        f"{path.name} imports {alias.name}; core must not depend on outer layers"
                    )


def test_guard_detects_third_party_import() -> None:
    """The guard itself must be able to catch a violation, or it proves nothing."""
    offending = "import numpy as np\nfrom fastapi import FastAPI\nfrom .acl import Grant\n"
    found = _imported_top_level_modules(offending, "synthetic.py")
    assert found == {"numpy", "fastapi"}
