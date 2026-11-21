"""The one reader of the release a requirement list pins a distribution at (pyproject.toml's `==` pins), which
the reference fixtures' cohorts (reference_releases.py) and the installation page's builds sentence
(site/scripts/gen_install.py) both read."""

import re


def pinned(requirements: list[str], name: str) -> str | None:
    """The release `requirements` pin `name` at: its first `name==release` requirement, an extra in brackets
    or an environment marker beside it; None where none pins it."""
    for requirement in requirements:
        if found := re.match(rf"{re.escape(name)}(?:\[[^]]*\])?==([^;\s]+)", requirement):
            return found[1]
    return None
