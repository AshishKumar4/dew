"""Source-file loading for tools exercised by several test suites, and git in a test's scratch repository."""

import importlib.util
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
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


@contextmanager
def run_from(name: str, directory: str) -> Iterator[ModuleType]:
    """`directory/name.py` loaded as `load` loads it, with its directory first on sys.path while held: a
    script run from its directory, which imports its siblings."""
    path = str(REPO_ROOT / directory)
    sys.path.insert(0, path)
    try:
        yield load(name, directory)
    finally:
        sys.path.remove(path)


def git(root: Path, *arguments: str) -> str:
    """Git in the scratch repository `root`, its author and committer named, its output returned; a
    failure raises."""
    return subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True, text=True,
                          env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}).stdout
