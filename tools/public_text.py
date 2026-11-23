"""No forecasts in Dew's prose (the bible): the lines a change adds to its docs and prose (Markdown,
anywhere but recorded and fixture data) and its commits' messages name no calendar date after the day they
were written. A date work is expected by is a forecast; a measurement's timestamp, on or before its
commit, is not. Nor does any line a change adds, in any file, or its commits' messages name one of the
maintainers' reviewers (`REVIEWER`), an internal handle no reader of Dew's history can resolve (owner,
2026-11-22).

    python tools/public_text.py BASE

reads what HEAD adds since its merge-base with BASE, against HEAD's author date, and each commit's message
against its own, prints each forecast it finds, and fails on any. Recordings' and fixtures' own `date`
metadata are data, not prose, and are not read. tools/check_gates runs it as the public-text gate.
"""

import argparse
import calendar
import datetime
import re
import subprocess
import sys

MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December|"
          "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
YEAR = r"(?:,? (?P<{0}>(?:19|20)\d\d)\b)?"
DATE = re.compile(rf"\b(?P<iso>(?:19|20)\d\d-[01]\d-[0-3]\d)\b"
                  rf"|\b(?P<month>{MONTHS})\.? (?P<day>[0-3]?\d)(?:st|nd|rd|th)?\b{YEAR.format('year')}"
                  rf"|\b(?P<first>[0-3]?\d)(?:st|nd|rd|th)? (?:of )?(?P<named>{MONTHS})\b\.?"
                  rf"{YEAR.format('of')}")
"""A calendar date: ISO (2026-10-22), a month and its day (October 22, Oct. 22nd), a day and its month,
with or without a year (22 October, the 22nd of Oct. 2026)."""
NUMBERS = {name[:3].lower(): number for number, name in enumerate(calendar.month_name) if name}
"""Each month's number by its name's first three letters, which every spelling of it (Sept., September)
shares."""
REVIEWER = re.compile(r"\bReview[A-Z]{2}\d+\b")
"""A reviewer's internal handle: the word Review, two capitals and a number, written as one word."""
PROSE = ("*.md", "*.mdx")
DATA = (":(exclude)site/public", ":(exclude)tests/fixtures")


def git(*arguments: str) -> str:
    """Git's output, a byte its encoding cannot read replaced, as a file's text may be in any encoding."""
    return subprocess.run(["git", *arguments], check=True, capture_output=True, text=True,
                          errors="replace").stdout


HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def added(changed: str, paths: tuple[str, ...] = PROSE) -> list[tuple[str, str]]:
    """Each line the revisions `changed` (`BASE...HEAD`, a commit's own `C~1..C`) add to `paths` (prose by
    default), recordings' and fixtures' data not read: (its `path:line`, its text). A hunk's lines are read
    by its header's counts, so an added line that begins `++ ` is not a file's header."""
    found, path, line, old, new = [], "", 0, 0, 0
    for text in git("diff", "-U0", "--no-color", changed, "--", *paths, *DATA).splitlines():
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


def written(commit: str) -> datetime.date:
    """The day `commit` was written: its author date, the later of its author's own day and UTC's, as Dew's
    records date their work in UTC and an evening west of it is already the next day there."""
    local = datetime.date.fromisoformat(git("log", "-1", "--format=%as", commit).strip())
    utc = datetime.datetime.fromtimestamp(int(git("log", "-1", "--format=%at", commit)), datetime.UTC).date()
    return max(local, utc)


def messages(base: str) -> list[tuple[str, str, datetime.date]]:
    """Each line of HEAD's commit messages since its merge-base with `base`: (its commit, its text, the day
    the commit was written)."""
    found = []
    for commit in git("rev-list", f"{base}..HEAD").split():
        day = written(commit)
        found += [(commit[:9], text, day) for text in git("log", "-1", "--format=%B", commit).splitlines()]
    return found


def named(date: re.Match, year: int) -> datetime.date | None:
    """The day a date names, in `year` where it names none; None for a day its month does not have."""
    if date["iso"]:
        year, month, day = map(int, date["iso"].split("-"))
    else:
        month = NUMBERS[(date["month"] or date["named"])[:3].lower()]
        day = int(date["day"] or date["first"])
        year = int(date["year"] or date["of"] or year)
    real = 1 <= month <= 12 and 1 <= day <= calendar.monthrange(year, month)[1]
    return datetime.date(year, month, day) if real else None


def forecasts(text: str, day: datetime.date) -> list[str]:
    """Each date `text`, written on `day`, names after it."""
    return [date[0] for date in DATE.finditer(text)
            if (when := named(date, day.year)) is not None and when > day]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="the ref HEAD's change is read from (its merge-base with it)")
    options = parser.parse_args()
    head = written("HEAD")
    said = messages(options.base)
    read = [(where, text, head) for where, text in added(f"{options.base}...HEAD")] + said
    found = [(where, text) for where, text, day in read if forecasts(text, day)]
    for where, text in found:
        print(f"public-text: a forecast: {where}: {text}")
    # Each commit's own lines, as each is published: a handle one commit adds and a later one removes stays
    # in the history.
    every = [*((f"{commit[:9]} {where}", text) for commit in git("rev-list", f"{options.base}..HEAD").split()
               for where, text in added(f"{commit}~1..{commit}", (".",))),
             *((where, text) for where, text, _ in said)]
    handles = [(where, text) for where, text in every if REVIEWER.search(text)]
    for where, text in handles:
        print(f"public-text: a reviewer's handle: {where}: {text}")
    return 1 if found or handles else 0


if __name__ == "__main__":
    sys.exit(main())
