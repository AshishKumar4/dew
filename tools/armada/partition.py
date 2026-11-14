"""Whether a whole run ran each test its files collect once (`ci.py partition`): a file's collected tests,
as pytest lists them under CI's selection, against the tests its rows of one Python ran, a split file's
groups, its sites' judges and a resumed task's recalled tests among them. A task of a split file refuses a
cut that does not hold each collected test once (split.py), every task cutting alike; this reads the same of
a run from outside its tasks, on a whole verdict (a train's: Main, 2026-11-12, after a
landing run's tails, read without their resumed attempts, seemed to leave a test unrun). The standard
library alone."""

import collections
from collections.abc import Container, Iterable, Mapping


def collected(lines: Iterable[str]) -> dict[str, set[str]]:
    """Each test file's node ids among `lines`, as `pytest --collect-only -q` prints them, a line a test."""
    found: dict[str, set[str]] = {}
    for line in lines:
        node = line.strip()
        if node.startswith("tests/") and "::" in node:
            found.setdefault(node.split("::")[0], set()).add(node)
    return found


def compared(nodes: Mapping[str, set[str]], rows: Iterable[Mapping], python: str,
             aside: Container[str] = ()) -> list[str]:
    """What is wrong between the tests `nodes` collects (`collected`) and those a verdict's `rows` of `python`
    ran (each row's `tests`), a line a file and kind: a collected test in no row, a test in two rows, a row's
    test the file does not collect. Files `aside` run apart from CI's tasks (ci.py's `SKIPPED`). None where
    each collected test ran in one row."""
    ran: dict[str, collections.Counter[str]] = {}
    for row in rows:
        version, _, rest = row["name"].partition(":")
        if version == python and rest:
            # A row's tests by their node ids, each with its seconds: counted once a row.
            ran.setdefault(rest.partition("#")[0], collections.Counter()).update(list(row.get("tests") or {}))
    wrong = []
    for name in sorted(set(nodes) | set(ran)):
        if name in aside:
            continue
        held, done = nodes.get(name, set()), ran.get(name, collections.Counter())
        for kind, tests in (("collected and in no row", held - set(done)),
                            ("in two rows", {node for node, count in done.items() if count > 1}),
                            ("in a row and not collected", set(done) - held)):
            if tests:
                wrong.append(f"{name}: {len(tests)} of its {len(held)} tests {kind}: "
                             + ", ".join(node.partition("::")[2] for node in sorted(tests)[:5])
                             + (" ..." if len(tests) > 5 else ""))
    return wrong
