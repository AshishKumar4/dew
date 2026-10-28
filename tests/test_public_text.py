"""No forecasts in Dew's prose: tools/public_text.py, in a git repository of its own, fails a change that adds
to a doc or a commit's message a date after the one it was written on, and passes one that adds none, one
whose date is on or before its own (a measurement's timestamp), or one whose date is a recording's or a
fixture's data."""

import subprocess
import sys
from pathlib import Path

import pytest
from tool_support import git

REPO = Path(__file__).resolve().parents[1]
GUIDE = "# Guide\n\nMeasured on 2026-09-02.\n"
WRITTEN = "2026-10-22T12:00:00"
"""When each change in these tests is written: its commit's author date."""


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """A repository whose `base` branch holds a doc that already names a date, which a change may keep."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/guide.md").write_text(GUIDE)
    git(tmp_path, "init", "-q", "-b", "base")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "base")
    git(tmp_path, "checkout", "-q", "-b", "head")
    return tmp_path


def change(root: Path, files: dict[str, str], message: str) -> int:
    """`files` written and committed as `message` on the head, at WRITTEN; the gate's exit status for it, run
    in the repository as check.sh runs it in Dew's."""
    for relative, text in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)
    git(root, "add", "-A")
    git(root, "commit", "-qm", message, f"--date={WRITTEN}")
    return subprocess.run([sys.executable, str(REPO / "tools/public_text.py"), "base"], cwd=root,
                          capture_output=True, timeout=60).returncode


@pytest.mark.parametrize(("files", "message", "status"), [
    ({"docs/guide.md": f"{GUIDE}\nMore, measured at a commit.\n"}, "docs: a measurement at its commit", 0),
    ({"docs/guide.md": f"{GUIDE}\nRerun on 2026-10-21.\n"}, "docs: a rerun", 0),
    ({"docs/guide.md": f"{GUIDE}\nRerun on October 22nd.\n"}, "docs: a rerun, measured 2026-10-22", 0),
    ({"docs/guide.md": f"{GUIDE}\nRerun on 2026-10-23.\n"}, "docs: a rerun", 1),
    ({"README.md": "Released November 3rd.\n"}, "docs: a release", 1),
    ({"README.md": "Released on the 3rd Nov. 2026.\n"}, "docs: a release", 1),
    ({"README.md": "Released 3 November.\n"}, "docs: a release", 1),
    ({"README.md": "A release.\n"}, "docs: released on the 3rd of Nov.", 1),
    ({"docs/guide.md": f"{GUIDE}\n++ rerun 2026-11-03\n"}, "docs: a line that begins as a header", 1),
    ({"docs/new.md": "A new page.\n"}, "docs: a page, landing 2026-11-03", 1),
    ({"tests/fixtures/run.md": "recorded 2026-11-03\n", "site/public/a/capture.json": '"2026-11-03"'},
     "tests: a recording and a fixture", 0),
], ids=["no date", "a measurement's ISO date before its commit", "a measurement's day, its commit's own",
        "a doc's ISO date after its commit", "a month and its later day", "an ordinal day and a dotted month",
        "a day and its month, no year", "a commit message's ordinal day of a later month",
        "a line that begins as a header", "a commit message's later date", "data's dates"])
def test_a_change_that_adds_a_forecast_to_prose_fails(repository, files, message, status):
    assert change(repository, files, message) == status


def test_a_date_of_the_day_in_utc_as_the_commit_is_written_is_no_forecast(repository):
    """A commit written in an evening west of UTC may name the day it is in UTC, as Dew's records date their
    work: no forecast; a day after that still is (2026-10-28: a commit written at 23:00 -05:00 on the 27th,
    naming "Main, 2026-10-28", failed the gate)."""
    (repository / "docs/guide.md").write_text(f"{GUIDE}\nMeasured as the commit was written.\n")
    git(repository, "add", "-A")
    git(repository, "commit", "-qm", "docs: a measurement (Main, 2026-10-23)",
        "--date=2026-10-22T23:30:00-05:00")
    gate = [sys.executable, str(REPO / "tools/public_text.py"), "base"]
    assert subprocess.run(gate, cwd=repository, capture_output=True, timeout=60).returncode == 0
    git(repository, "commit", "-q", "--allow-empty", "-m", "docs: landing 2026-10-24",
        "--date=2026-10-22T23:30:00-05:00")
    assert subprocess.run(gate, cwd=repository, capture_output=True, timeout=60).returncode == 1
