"""The suite's scenarios (tests/conftest.py SCENARIOS) and the test files whose tests mark one
(`@pytest.mark.scenario("<name>")` on a test's own decorator), read from the source, as the plan reads its
marks: the scenario suite's coverage runs those tests alone (ci.py `coverage items --scenarios`), and CI's
check refuses a scenario no test marks (ci.py `scenarios --check`). Standard library alone, as ci.py's
siblings are."""

import ast
import re
import sys
from pathlib import Path

MARK = re.compile(r'@pytest\.mark\.scenario\(\s*"([^"]+)"\s*\)')
"""A test's scenario mark on its own decorator, its name a literal."""


def declared(conftest: Path = Path("tests/conftest.py")) -> tuple[str, ...]:
    """tests/conftest.py's `SCENARIOS`, read from its source (the plan runs where no test imports)."""
    tree = ast.parse(conftest.read_text())
    return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and [target.id for target in node.targets if isinstance(target, ast.Name)] == ["SCENARIOS"])


def marked(files: list[str]) -> dict[str, set[str]]:
    """Each declared scenario and the `files` whose tests mark it; a name no scenario has, by the files
    that mark it."""
    found: dict[str, set[str]] = {name: set() for name in declared()}
    for name in files:
        for scenario in MARK.findall(Path(name).read_text()):
            found.setdefault(scenario, set()).add(name)
    return found


def scenarios(files: list[str], *, check: bool) -> int:
    """Each scenario and the `files` that mark it, a line each; with `check`, refuse a scenario no test
    marks and a mark naming no scenario (docs/design/testing.md: the scenario suite's coverage is
    published)."""
    found, names = marked(files), set(declared())
    if not check:
        for scenario, holding in found.items():
            print(scenario, *sorted(holding), sep="\t")
        return 0
    problems = [f"no test marks the scenario {scenario!r} (@pytest.mark.scenario)"
                for scenario, holding in found.items() if scenario in names and not holding]
    problems += [f"{', '.join(sorted(holding))}: mark the scenario {scenario!r}, which tests/conftest.py's "
                 f"SCENARIOS does not name" for scenario, holding in found.items() if scenario not in names]
    print("\n".join(problems) if problems else f"{len(names)} scenarios, each marked", file=sys.stderr)
    return 1 if problems else 0
