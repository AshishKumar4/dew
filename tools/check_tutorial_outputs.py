"""Fail when a tutorial's outputs are older than a change to the Dew code the tutorial imports.

Each executed notebook records in its metadata, under `dew.outputs.commit`, the
last commit that had changed src/dew when its outputs were made
(`tools/run_tutorials.py --save` writes it). `tools/run_tutorials.py --imports
FILE` records which of Dew's modules each notebook's kernel had imported by
the end of a run. This script fails when a commit after the recorded one
changed the file of any of those modules, and names the commits and files, so
outputs that may no longer match the code are caught instead of trusted.

It also fails when a notebook has no recorded commit, when that commit is not
in this checkout's history (it needs the full history: CI checks out with
fetch-depth 0), or when the run recorded no imports for the notebook, which
happens when the notebook failed.

    python tools/run_tutorials.py --imports /tmp/imports.json
    python tools/check_tutorial_outputs.py /tmp/imports.json
"""

from __future__ import annotations

import json
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
    files = sorted({file for name in modules if (file := module_file(name))})
    log = git("log", "--format=%h %ad %s", "--date=short", f"{commit}..HEAD", "--", *files).stdout.strip()
    if not log:
        return "current", []
    changed = git("diff", "--name-only", commit, "HEAD", "--", *files).stdout.split()
    return "STALE", [f"its outputs are from {commit[:8]}, and these commits since then changed {len(changed)} of the "
                     f"{len(files)} files of Dew it imports ({', '.join(changed)}):", *log.splitlines()]


def main() -> int:
    source = Path(sys.argv[1])
    if not source.is_file():
        print(f"{source} does not exist: tools/run_tutorials.py --imports did not finish", file=sys.stderr)
        return 1
    imports = json.loads(source.read_text())
    failed = 0
    for path in sorted((ROOT / "tutorials").glob("*.ipynb")):
        label, lines = verdict(path, imports)
        print(f"{path.name}: {label}")
        for line in lines:
            print(f"    {line}")
        failed += label != "current"
    if failed:
        print(f"{failed} notebooks failed the check. Execute them again with tools/run_tutorials.py --full --save "
              f"and commit the result.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
