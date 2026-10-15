"""The duplicate-code gate: pylint's similarity check over the given roots, failing on any block of
more than six lines two files share once imports, signatures, docstrings and comments are set aside.
tools/check.sh runs it over src/dew, tools and tests with no suppression and no baseline, so a copy
lands as one owner its callers share, or not at all. tests/test_duplicates.py proves it refuses a
copied block.

    python tools/duplicates.py src/dew tools tests

It runs the test environment's pylint, which pyproject.toml's test extra pins.
"""

import sys

from pylint.lint import Run

ARGUMENTS = ("--disable=all", "--enable=duplicate-code", "--min-similarity-lines=6", "--ignore-imports=y",
             "--ignore-signatures=y", "--ignore-docstrings=y", "--ignore-comments=y", "--score=n")
"""The one statement of the check, which the gate and its test both run."""


if __name__ == "__main__":
    Run([*ARGUMENTS, *sys.argv[1:]])
