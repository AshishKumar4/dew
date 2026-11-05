"""CHANGELOG.md merges as its .gitattributes says (tools/changelog_merge.py): where two branches each add
entries at the same place, both are kept, as lanes add theirs side by side; where one side deleted or rewrote
an entry the other's change meets, the merge stops on a conflict and keeps neither side silently. A union
merge kept the other side's copy there: a corrected entry came back beside its correction (row 3284), and a
head's deletion vanished in its train's merge (row 3297). A clone that never named the driver conflicts there
too."""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DRIVER = f'python3 "{REPO / "tools/changelog_merge.py"}" %O %A %B %L'
OLD = "- Old entry, since corrected."
BASE = (f"# Changelog\n\n## Unreleased\n\n- Kept entry.\n{OLD}\n- Another kept entry.\n"
        "\n## 0.2.0\n\n- Released entry.\n")
# Git reads no configuration of the host's: only what a test passes it.
QUIET = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def git(root: Path, *args: str, driver: bool = True) -> subprocess.CompletedProcess:
    named = ["-c", f"merge.changelog.driver={DRIVER}"] if driver else []
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *named, *args], cwd=root,
                          env=QUIET, capture_output=True, text=True)


def merged(tmp_path: Path, ours: str, theirs: str,
           driver: bool = True) -> tuple[subprocess.CompletedProcess, Path]:
    """Branch `theirs`'s changelog merged into main's (`ours`), both cut from BASE under Dew's
    .gitattributes."""
    root = tmp_path / "repository"
    root.mkdir()
    (root / ".gitattributes").write_text((REPO / ".gitattributes").read_text())
    (root / "CHANGELOG.md").write_text(BASE)
    for command in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-qm", "base"],
                    ["checkout", "-q", "-b", "theirs"]):
        git(root, *command).check_returncode()
    for branch, text in (("theirs", theirs), ("main", ours)):
        git(root, "checkout", "-q", branch).check_returncode()
        (root / "CHANGELOG.md").write_text(text)
        git(root, "commit", "-qam", branch).check_returncode()
    return git(root, "merge", "--no-edit", "theirs", driver=driver), root


@pytest.mark.parametrize("driver", [True, False], ids=["the driver", "a clone without it"])
def test_a_deletion_met_by_an_entry_added_beside_it_stops_on_a_conflict(tmp_path, driver):
    """Train 434's merge: main added an entry right above two lines a head deleted."""
    ours = BASE.replace(OLD, f"- An entry main added.\n{OLD}")
    theirs = BASE.replace(f"{OLD}\n", "")
    merge, root = merged(tmp_path, ours, theirs, driver)
    assert merge.returncode != 0
    assert git(root, "diff", "--name-only", "--diff-filter=U").stdout.split() == ["CHANGELOG.md"]


@pytest.mark.parametrize("driver", [True, False], ids=["the driver", "a clone without it"])
def test_a_rewrite_met_by_an_entry_added_beside_it_stops_on_a_conflict(tmp_path, driver):
    """Row 3284's merge: one side corrected an entry, the other added one where the old entry stood."""
    ours = BASE.replace(OLD, f"{OLD}\n- An entry main added.")
    theirs = BASE.replace(OLD, "- The corrected entry.")
    merge, root = merged(tmp_path, ours, theirs, driver)
    assert merge.returncode != 0
    assert git(root, "diff", "--name-only", "--diff-filter=U").stdout.split() == ["CHANGELOG.md"]


def test_entries_both_sides_add_at_the_head_of_unreleased_are_both_kept(tmp_path):
    """As lanes add theirs side by side, and main's automatic merge into base/kernels then stops on none."""
    ours = BASE.replace("## Unreleased\n\n", "## Unreleased\n\n- Main's new entry.\n")
    theirs = BASE.replace("## Unreleased\n\n", "## Unreleased\n\n- A head's new entry.\n")
    merge, root = merged(tmp_path, ours, theirs)
    assert merge.returncode == 0, merge.stdout + merge.stderr
    assert (root / "CHANGELOG.md").read_text() == BASE.replace(
        "## Unreleased\n\n", "## Unreleased\n\n- Main's new entry.\n- A head's new entry.\n")


def test_an_entry_both_sides_carry_merges_once(tmp_path):
    """Train 451's merge: main took a train's entries, and a head carries the same entries as a commit of its
    own beside its new one; the two sides' insertions meet at the head of `## Unreleased`, and the entries
    both hold merge once, the head's new entry beside them (row 3431)."""
    carried = "- A train's entry.\n- Its second entry.\n"
    ours = BASE.replace("## Unreleased\n\n", f"## Unreleased\n\n{carried}")
    theirs = BASE.replace("## Unreleased\n\n", f"## Unreleased\n\n- A head's new entry.\n{carried}")
    merge, root = merged(tmp_path, ours, theirs)
    assert merge.returncode == 0, merge.stdout + merge.stderr
    assert (root / "CHANGELOG.md").read_text() == BASE.replace(
        "## Unreleased\n\n", f"## Unreleased\n\n{carried}- A head's new entry.\n")


def test_an_entry_sharing_its_last_line_with_the_other_sides_stays_whole(tmp_path):
    """Two new entries, one a side's, that end on the same continuation line are two entries: each merges
    whole, its shared last line kept in both, where a line's dedupe cut the second one short (row 3431's
    review)."""
    ours = BASE.replace("## Unreleased\n\n", "## Unreleased\n\n- Our entry,\n  as the docs say.\n")
    theirs = BASE.replace("## Unreleased\n\n", "## Unreleased\n\n- Their entry,\n  as the docs say.\n")
    merge, root = merged(tmp_path, ours, theirs)
    assert merge.returncode == 0, merge.stdout + merge.stderr
    assert (root / "CHANGELOG.md").read_text() == BASE.replace(
        "## Unreleased\n\n",
        "## Unreleased\n\n- Our entry,\n  as the docs say.\n- Their entry,\n  as the docs say.\n")


def test_entries_both_sides_carry_in_another_order_merge_once(tmp_path):
    """Entries both sides carry merge once wherever each side holds them, a side's new entry between or
    after them kept where its side has it."""
    ours = BASE.replace("## Unreleased\n\n", "## Unreleased\n\n- First.\n- Second.\n")
    theirs = BASE.replace("## Unreleased\n\n", "## Unreleased\n\n- Second.\n- New.\n- First.\n")
    merge, root = merged(tmp_path, ours, theirs)
    assert merge.returncode == 0, merge.stdout + merge.stderr
    assert (root / "CHANGELOG.md").read_text() == BASE.replace(
        "## Unreleased\n\n", "## Unreleased\n\n- First.\n- Second.\n- New.\n")


def test_a_deletion_apart_from_the_other_sides_new_entry_merges_with_it(tmp_path):
    ours = BASE.replace("## Unreleased\n\n", "## Unreleased\n\n- Main's new entry.\n")
    theirs = BASE.replace(f"{OLD}\n", "")
    merge, root = merged(tmp_path, ours, theirs)
    assert merge.returncode == 0, merge.stdout + merge.stderr
    assert (root / "CHANGELOG.md").read_text() == ours.replace(f"{OLD}\n", "")
