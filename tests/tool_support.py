"""Source-file loading for tools exercised by several test suites."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"


def load(name: str, directory: str = "tools") -> ModuleType:
    """The repository's `directory/name.py` as the module `name_under_test`,
    registered before it runs so its dataclasses resolve their module."""
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", REPO_ROOT / directory / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
