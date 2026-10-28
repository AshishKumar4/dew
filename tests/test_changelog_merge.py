"""Lanes add their own entries to CHANGELOG.md and edit no other's, so two branches that each add one merge
with both entries kept: the repository's .gitattributes merges the changelog by union, which keeps the
automatic merge of main into base/kernels from stopping on it."""

import shutil
from pathlib import Path

from tool_support import git

REPO = Path(__file__).resolve().parents[1]
CHANGELOG = "# Changes\n\n## Unreleased\n\n{}- an entry both had\n"


def test_two_branches_each_adding_a_changelog_entry_merge_with_both(tmp_path):
    shutil.copyfile(REPO / ".gitattributes", tmp_path / ".gitattributes")
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG.format(""))
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "base")
    git(tmp_path, "checkout", "-q", "-b", "kernels")
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG.format("- a kernel's entry\n"))
    git(tmp_path, "commit", "-qam", "a kernel's entry")
    git(tmp_path, "checkout", "-q", "main")
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG.format("- main's entry\n"))
    git(tmp_path, "commit", "-qam", "main's entry")
    git(tmp_path, "checkout", "-q", "kernels")
    git(tmp_path, "merge", "-q", "--no-edit", "main")
    merged = (tmp_path / "CHANGELOG.md").read_text()
    assert "- a kernel's entry\n" in merged and "- main's entry\n" in merged and "<<<<<<<" not in merged
