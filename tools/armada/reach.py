"""reach.py BASE --target SECONDS --timings FILE: the test rows this checkout's change from BASE reaches,
each Python's, as the test cache's own keys read it (`reached`), JSON on stdout: {"base", "policy",
"changed", "<python>": {"files", "module", "unknown"}}. The plan (ci.py's `plan`) gives each Python's rows,
those its cache keeps and their last runs' inputs; a change that leaves the cache cold or off reaches every
row (Main, 2026-11-15: the tests a diff reaches, as the trains' own planner states them, for run_tests)."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Container
from pathlib import Path

import ci


def reached(python: str, rows: list[tuple[str, str]], kept: Container[tuple[str, str]],
            last: dict[tuple[str, str], list[str]], base: Path) -> dict:
    """Which of a Python's `rows` (each a file and its group) the checkout's change from `base` (a checkout of
    it) reaches, as the cache's keys read it (`reach`): none it keeps (`kept`, green on exactly these inputs);
    one whose key over its last run's inputs (`last`, what `record` kept of its reads) differs between `base`
    and here (`cache_key`, each side's own reading and common: a unit executed, a file read, a lookup, a
    listing, a judge's map inputs, all as a key covers them); and one whose last run recorded none, unknown.
    The same keys read in two tiers (Main, 2026-11-15): `files`, the rows whose key moved on an input other
    than a module's top level (a function or class body its last run executed, a file it read, a lookup),
    and `module`, those it moved through module-level units alone, which every
    importer of a module executes. A row the change leaves unread is no reach of it, whatever the cache's
    state for it. {"files", "module": the files of each tier, "unknown": the files reached only through a row
    with no recorded inputs}."""
    rows = [each for each in rows if each not in kept]
    here = os.getcwd()

    def keyed() -> tuple[dict[tuple[str, str], str], Callable[[str], str]]:
        # A side's own reading: its digests, and whether a path is a build's output, its checkout's (review).
        ci.digests.built_input.cache_clear()
        common, reading = ci.shared(python), ci.Reading()
        alone: dict[str, str] = {}

        def single(item: str) -> str:
            # One input's own key, which says whether that input moved (`moved`).
            if item not in alone:
                alone[item] = ci.cache_key("", "", "", [item], reading)
            return alone[item]
        return ({(name, group): ci.cache_key(common, name, group, last[(name, group)], reading)
                 for name, group in rows if last.get((name, group))}, single)

    now, single_now = keyed()
    try:
        os.chdir(base)
        then, single_then = keyed()
        # Each input as the key covers it (`inputs`: the test's own code whole, the lookups that found each
        # module), so a changed lookup or the test's own change is a tier-one move (review).
        moved = {each: [item for item in ci.inputs(each[0], last[each])
                        if single_then(item) != single_now(item)]
                 for each, key in now.items() if then.get(each) != key}
    finally:
        os.chdir(here)
        ci.digests.built_input.cache_clear()
    top = f"{ci.followed.CODE}<module>"
    deep = {each[0] for each, items in moved.items()
            if not items or any(not item.endswith(top) for item in items)}
    module = {each[0] for each in moved} - deep
    unknown = {name for name, group in rows if not last.get((name, group))} - deep - module
    return {"files": sorted(deep), "module": sorted(module), "unknown": sorted(unknown)}


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    parser.add_argument("base")
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--timings", type=Path, required=True)
    options = parser.parse_args(arguments)
    timings = json.loads(options.timings.read_text()) if options.timings.is_file() else {}
    base = subprocess.run(["git", "merge-base", "HEAD", options.base], capture_output=True, text=True,
                          check=True).stdout.strip()
    changed = set(subprocess.run(["git", "diff", "--name-only", "--no-renames", f"{base}...HEAD"],
                                 capture_output=True, text=True, check=True).stdout.split())
    # The plan's policy, as a run's scope sets it.
    os.environ["DEW_TEST_CACHE"] = policy = ci.policy(base)
    planned: dict = {}
    ci.plan(options.target, timings, reach=planned)
    with tempfile.TemporaryDirectory(prefix="reach-base-") as held:
        # The base's checkout, which each row's key is read against.
        subprocess.run(["git", "worktree", "add", "-q", "--detach", f"{held}/base", base], check=True,
                       capture_output=True)
        try:
            every = {python: {"files": sorted({name for name, _ in rows}), "module": [], "unknown": []}
                     for python, (rows, _, _) in planned.items()}
            found = {python: reached(python, rows, kept, last, Path(held, "base")) if kept is not None
                     else every[python] for python, (rows, kept, last) in planned.items()}
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", f"{held}/base"], capture_output=True)
            ci.digests.built_input.cache_clear()
    print(json.dumps({"base": base, "policy": policy, "changed": sorted(changed), **found}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
