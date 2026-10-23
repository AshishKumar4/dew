"""No dates in Dew's prose (the bible: no dates, ever): the lines a change adds to its docs and prose
(Markdown, anywhere but recorded and fixture data) and its commits' messages name no calendar date. A
landing, a measurement or a decision is named by its commit, its ref or its record, never by when.

    python tools/public_text.py BASE

reads what HEAD adds since its merge-base with BASE, prints each date it finds, and fails on any.
Recordings' and fixtures' own `date` metadata are data, not prose, and are not read. tools/check_gates
runs it as the public-text gate.
"""

import argparse
import re
import subprocess
import sys

MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December|"
          "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
DATE = re.compile(rf"\b(?:19|20)\d\d-[01]\d-[0-3]\d\b|\b(?:{MONTHS})\.? [0-3]?\d(?:st|nd|rd|th)?\b"
                  rf"|\b[0-3]?\d(?:st|nd|rd|th)? (?:{MONTHS})\.? (?:19|20)\d\d\b")
"""A calendar date: ISO (2026-10-22), a month and its day (October 22, Oct. 22nd), a day, month and
year (22nd Oct. 2026)."""
PROSE = ("*.md", "*.mdx")
DATA = (":(exclude)site/public", ":(exclude)tests/fixtures")


def git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], check=True, capture_output=True, text=True).stdout


HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def added(base: str) -> list[tuple[str, str]]:
    """Each prose line HEAD adds since its merge-base with `base`: (its `path:line`, its text). A hunk's
    lines are read by its header's counts, so an added line that begins `++ ` is not a file's header."""
    found, path, line, old, new = [], "", 0, 0, 0
    for text in git("diff", "-U0", "--no-color", f"{base}...HEAD", "--", *PROSE, *DATA).splitlines():
        if old or new:
            if text.startswith("+"):
                found.append((f"{path}:{line}", text[1:]))
                line, new = line + 1, new - 1
            elif text.startswith("-"):
                old -= 1
        elif text.startswith("+++ "):
            path = text.removeprefix("+++ b/")
        elif hunk := HUNK.match(text):
            removed, start, counted = hunk.groups()
            old, new, line = int(removed or 1), int(counted or 1), int(start)
    return found


def messages(base: str) -> list[tuple[str, str]]:
    """Each line of HEAD's commit messages since its merge-base with `base`: (its commit, its text)."""
    found = []
    for commit in git("rev-list", f"{base}..HEAD").split():
        found += [(commit[:9], text) for text in git("log", "-1", "--format=%B", commit).splitlines()]
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("base", help="the ref HEAD's change is read from (its merge-base with it)")
    options = parser.parse_args()
    read = added(options.base) + messages(options.base)
    dated = [(where, text) for where, text in read if DATE.search(text)]
    for where, text in dated:
        print(f"public-text: a date: {where}: {text}")
    return 1 if dated else 0


if __name__ == "__main__":
    sys.exit(main())
