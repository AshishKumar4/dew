"""No dates in Dew's prose: tools/public_text.py, in a git repository of its own, fails a change that adds a
date to a doc or a commit's message, and passes one that adds none, or one whose date is a recording's or a
fixture's data."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from tool_support import git

REPO = Path(__file__).resolve().parents[1]
GUIDE = "# Guide\n\nMeasured on 2026-09-02, before this rule.\n"


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """A repository whose `base` branch holds a doc that already names a date, which a change may keep."""
    (tmp_path / "tools").mkdir()
    shutil.copyfile(REPO / "tools/public_text.py", tmp_path / "tools/public_text.py")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/guide.md").write_text(GUIDE)
    git(tmp_path, "init", "-q", "-b", "base")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "base")
    git(tmp_path, "checkout", "-q", "-b", "head")
    return tmp_path


def change(root: Path, files: dict[str, str], message: str) -> int:
    """`files` written and committed as `message` on the head; the gate's exit status for it."""
    for relative, text in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)
    git(root, "add", "-A")
    git(root, "commit", "-qm", message)
    return subprocess.run([sys.executable, "tools/public_text.py", "base"], cwd=root, capture_output=True,
                          timeout=60).returncode


@pytest.mark.parametrize(("files", "message", "status"), [
    ({"docs/guide.md": f"{GUIDE}\nMore, measured at a commit.\n"}, "docs: a measurement at its commit", 0),
    ({"docs/guide.md": f"{GUIDE}\nRerun on 2026-10-22.\n"}, "docs: a rerun", 1),
    ({"README.md": "Released October 22nd.\n"}, "docs: a release", 1),
    ({"README.md": "Released on the 22nd Oct. 2026.\n"}, "docs: a release", 1),
    ({"docs/guide.md": f"{GUIDE}\n++ rerun 2026-10-22\n"}, "docs: a line that begins as a header", 1),
    ({"docs/new.md": "A new page.\n"}, "docs: a page, landed 2026-10-22", 1),
    ({"tests/fixtures/run.md": "recorded 2026-10-22\n", "site/public/a/capture.json": '"2026-10-22"'},
     "tests: a recording and a fixture", 0),
], ids=["no date", "a doc's ISO date", "a month and its day", "an ordinal day and a dotted month",
        "a line that begins as a header", "a commit message's date", "data's dates"])
def test_a_change_that_adds_a_date_to_prose_fails(repository, files, message, status):
    assert change(repository, files, message) == status
