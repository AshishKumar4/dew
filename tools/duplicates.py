"""The duplicate-code gate: pylint's similarity check over the given roots, failing on any block of
eight or more lines two files share once imports, signatures, docstrings and comments are set aside.
tools/check.sh runs it over src/dew, tools and tests with no suppression and no baseline, so a copy
lands as one owner its callers share, or not at all. tests/test_duplicates.py proves it refuses a
copied block.

    uv run --no-project --with pylint==4.1.3 python tools/duplicates.py src/dew tools tests
"""

import sys

from pylint.lint import Run

ARGUMENTS = ("--disable=all", "--enable=duplicate-code", "--min-similarity-lines=8", "--ignore-imports=y",
             "--ignore-signatures=y", "--ignore-docstrings=y", "--ignore-comments=y", "--score=n")
"""The one statement of the check, which the gate and its test both run."""


if __name__ == "__main__":
    Run([*ARGUMENTS, *sys.argv[1:]])
