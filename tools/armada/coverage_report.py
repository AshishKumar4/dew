"""The coverage pass's report (tools/armada/ci.py `coverage report`): the coverage data `ci.py coverage items`'
tasks left among their artifacts, combined, and a table a module a line, the least covered first. It runs where
the artifacts were downloaded, under the interpreter that runs it, which must have coverage.py."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


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
    # A task that wrote no rows or was interrupted left the coverage of its files' later tests out.
    def ran_short(task: Path) -> bool:
        verdict = task / "rows.json"
        return not verdict.exists() or any(found["exitCode"] and found["kind"] == "deadline"
                                           for found in json.loads(verdict.read_text())["rows"])

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

