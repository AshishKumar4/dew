"""The code in the documentation runs.

Every ```python block in README.md and docs/**/*.md is executed, in file
order, in one namespace per file, on CPU with tiny shapes where the block
defines them. A block that names a symbol the library no longer has, or
calls it with arguments it no longer takes, fails here rather than in a
reader's terminal. Blocks that need the network, a GPU, a dataset on disk or a
trained checkpoint carry a `# doctest: skip` first line and are compiled but
not run, so a syntax error or a misspelled import still fails.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def blocks(path: Path) -> list[tuple[int, str]]:
    text = path.read_text()
    found = []
    for match in BLOCK.finditer(text):
        line = text.count("\n", 0, match.start()) + 2
        found.append((line, match.group(1)))
    return found


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_the_documented_code_runs(path, tmp_path, monkeypatch):
    found = blocks(path)
    if not found:
        pytest.skip("no python blocks")
    monkeypatch.chdir(tmp_path)
    namespace: dict = {"__name__": "__docs__"}
    for line, source in found:
        code = compile(source, f"{path.relative_to(ROOT)}:{line}", "exec")
        if source.lstrip().startswith("# doctest: skip"):
            continue
        try:
            exec(code, namespace)
        except Exception as error:  # noqa: BLE001 - the report names the block
            raise AssertionError(
                f"{path.relative_to(ROOT)} block at line {line} failed: "
                f"{type(error).__name__}: {error}") from error
