"""The coverage pass (tools/armada/ci.py `coverage`): its tasks, each running a plan task's files under
coverage.py (`items`), and its report (`report`): the coverage data those tasks left among their artifacts,
combined, and a table a module a line, the least covered first. The report runs where the artifacts were
downloaded, under the interpreter that runs it, which must have coverage.py."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

DEADLINE = 5400.0
"""A coverage task's deadline. Branch tracing slows a task several times over, and a task cut short leaves
its files' coverage partial, which `report` refuses."""
SHORT = ("deadline", "whole", "stale")
"""The kinds of a red row whose task's run did not finish its file (grading.py's `red_as`): cut at its
deadline, its process failed, or a kept row it never ran; its coverage is partial, which `report` refuses."""
SCENARIO_SUITE = ("tests/test_scenario_runs.py", "tests/test_decision_workflow.py")
"""The scenario suite's files, as docs/design/testing.md names them: a short run a cell of each scenario its
`SCENARIOS` registry holds, and the decision workflow's cells; `coverage items --scenarios` measures their
per-module coverage (row 6118; train 878's review F2)."""


def items(entries: list[dict], artifacts: str, scenarios: bool = False) -> list[str]:
    """Plan tasks `entries` as armada map items: each runs its files under coverage.py with the test cache
    off, its rows and data among its artifacts (`artifacts`), by `DEADLINE`. With `scenarios`, the scenario
    suite's tasks alone (`SCENARIO_SUITE`), each running the suite's files it held."""
    if scenarios:
        def held(entry: dict) -> list[str]:
            tokens = entry["tests"].split()
            return [name for name in SCENARIO_SUITE if any(token.startswith(name) for token in tokens)]

        entries = [{**entry, "tests": " ".join(held(entry))} for entry in entries if held(entry)]
    return [f"mkdir -p {artifacts} && python3 tools/armada/ci.py task --python={entry['python']} "
            f"--tests='{entry['tests']}' --split='{entry['split']}' --out={artifacts}/rows.json --cache=off "
            f"--deadline={DEADLINE:.0f} --coverage={artifacts}" for entry in entries]


def report(directory: Path) -> int:
    """Combine the coverage data `ci.py coverage items`' tasks left under `directory` and write its report:
    coverage.py's own JSON, and a table a module a line, the least covered first. It runs where the
    artifacts were downloaded, under this interpreter, which must have coverage.py."""
    if importlib.util.find_spec("coverage") is None:
        print(f"coverage.py is not importable from {sys.executable}; install the test extra", file=sys.stderr)
        return 1
    python = sys.executable
    data = directory / ".coverage"
    parts = sorted(str(path) for path in directory.glob("*/.coverage.*"))
    if not parts:
        print(f"no coverage data under {directory}", file=sys.stderr)
        return 1
    # A task that wrote no rows or no coverage data, or a row of which its run did not finish, left the
    # coverage of its files' later tests out: cut at its deadline, its process crashed (`whole`), or never
    # started (`stale`, a kept row it did not run; row 10801).
    def ran_short(task: Path) -> bool:
        verdict = task / "rows.json"
        return (not verdict.exists() or not any(task.glob(".coverage.*"))
                or any(found["exitCode"] and found.get("kind") in SHORT
                       for found in json.loads(verdict.read_text())["rows"]))

    # armada keeps each task's artifacts in a directory named by its index.
    short = [task.name for task in sorted(directory.iterdir())
             if task.is_dir() and task.name.isdecimal() and ran_short(task)]
    if short:
        print(f"{len(short)} tasks ran short, so the pass is partial; run them again with a longer deadline: "
              f"{', '.join(short)}", file=sys.stderr)
        return 1
    subprocess.run([python, "-m", "coverage", "combine", "--keep", f"--data-file={data}", *parts], check=True)
    report = directory / "coverage.json"
    subprocess.run([python, "-m", "coverage", "json", f"--data-file={data}", "-o", str(report)], check=True)
    files = json.loads(report.read_text())["files"]
    rows = sorted((measured["summary"]["percent_covered"], name, measured["summary"])
                  for name, measured in files.items())
    lines = ["| module | statements | missed | branches | partial | covered |", "|---|---|---|---|---|---|"]
    lines += [f"| {name} | {summary['num_statements']} | {summary['missing_lines']} "
              f"| {summary['num_branches']} | {summary['num_partial_branches']} | {percent:.1f}% |"
              for percent, name, summary in rows]
    (directory / "coverage.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} modules from {len(parts)} data files: {directory / 'coverage.md'}")
    return 0

