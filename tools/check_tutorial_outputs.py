"""Report tutorials whose outputs are older than a change to the Dew code they import.

Each executed notebook records in its metadata, under `dew.outputs.commit`, the
last commit that had changed src/dew when its outputs were made
(`tools/run_tutorials.py --save` writes it). `tools/run_tutorials.py --imports
FILE` records which of Dew's modules each notebook's kernel had imported by
the end of a run. A notebook is stale when a commit after the recorded one
changed the file of any of those modules, or pyproject.toml, which pins the
dependencies (JAX among them); the script names the commits and files, so outputs that may no longer match the code are flagged instead of
trusted. It also flags a notebook with no recorded commit, with a commit that
is not in this checkout's history (it needs the full history: CI checks out
with fetch-depth 0), or without recorded imports, which happens when the
notebook failed.

It exits 1 when it flags anything. With --report it exits 0 and, under GitHub
Actions, turns each flag into a warning and writes a table to the job summary:
the Tutorials workflow reports stale outputs this way, since the notebooks
import about 190 of Dew's files and nearly every change to the library makes
their outputs stale. The outputs are refreshed by a daily full run; the smoke
run is what fails when a notebook breaks.

    python tools/run_tutorials.py --imports /tmp/imports.json
    python tools/check_tutorial_outputs.py /tmp/imports.json [--report]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def module_file(name: str) -> str | None:
    """The file in this checkout that defines Dew module `name`; None for a namespace package."""
    base = ROOT / "src" / Path(*name.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return str(candidate.relative_to(ROOT))
    return None


def verdict(path: Path, imports: dict[str, list[str]]) -> tuple[str, list[str]]:
    """'current', or why not, with the details to print."""
    record = json.loads(path.read_text()).get("metadata", {}).get("dew", {}).get("outputs", {})
    commit = record.get("commit")
    if not commit:
        return "NO RECORD", ["its metadata has no dew.outputs.commit; save the notebook with tools/run_tutorials.py --save"]
    if git("merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
        return "NO RECORD", [f"its outputs are recorded at {commit[:8]}, which is not in this checkout's history"]
    modules = imports.get(path.name)
    if modules is None:
        return "UNCHECKED", ["the run recorded no imports for it, so it did not finish"]
    files = sorted({file for name in modules if (file := module_file(name))} | {"pyproject.toml"})
    log = git("log", "--format=%h %ad %s", "--date=short", f"{commit}..HEAD", "--", *files).stdout.strip()
    if not log:
        return "current", []
    changed = git("diff", "--name-only", commit, "HEAD", "--", *files).stdout.split()
    return "STALE", [f"its outputs are from {commit[:8]}, and these commits since then changed {len(changed)} of the "
                     f"{len(files)} files it depends on ({', '.join(changed)}):", *log.splitlines()]


def main() -> int:
    report = "--report" in sys.argv[2:]
    source = Path(sys.argv[1])
    imports = json.loads(source.read_text()) if source.is_file() else {}
    if not source.is_file():
        print(f"{source} does not exist: tools/run_tutorials.py --imports did not finish", file=sys.stderr)
    actions = report and os.environ.get("GITHUB_ACTIONS") == "true"
    rows = []
    flagged = 0
    for path in sorted((ROOT / "tutorials").glob("*.ipynb")):
        label, lines = verdict(path, imports)
        print(f"{path.name}: {label}")
        for line in lines:
            print(f"    {line}")
        if label != "current":
            flagged += 1
            if actions:
                print(f"::warning title=Tutorial outputs::{path.name}: {label}, {lines[0]}")
        rows.append(f"| {path.name} | {label} | {lines[0] if lines else ''} |")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if actions and summary:
        with open(summary, "a") as out:
            out.write("### Tutorial outputs\n\n| Notebook | Outputs | Why |\n|---|---|---|\n" + "\n".join(rows) + "\n")
    if flagged:
        print(f"{flagged} notebooks flagged. Execute them again with tools/run_tutorials.py --full --save "
              f"and commit the result.", file=sys.stderr)
    return 0 if report or not flagged else 1


if __name__ == "__main__":
    sys.exit(main())
