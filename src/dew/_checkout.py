"""The Dew checkout a process runs from, as its records name it: its git revision, and whether it is
clean. An installed Dew is no checkout."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
"""Where Dew's checkout would hold its `.git`: a file in a worktree, a directory in a clone."""


def checked_out() -> bool:
    """Whether Dew runs from a git checkout rather than an installed wheel."""
    return (ROOT / ".git").exists()


def revision() -> str | None:
    """The checkout's commit, or None for an installed Dew. Raises where git cannot read it."""
    return _git("rev-parse", "HEAD") if checked_out() else None


def clean() -> bool:
    """Whether the checkout changes no tracked file and adds no file to Dew's own source (`src/dew`);
    an installed Dew is clean. Raises where git cannot read it."""
    if not checked_out():
        return True
    return not _git("status", "--porcelain", "--untracked-files=no") and not _git(
        "ls-files", "--others", "--exclude-standard", "src/dew")


def _git(*arguments: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *arguments], check=True, capture_output=True,
                          text=True).stdout.strip()
