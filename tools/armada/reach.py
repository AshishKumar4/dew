"""reach.py BASE --target SECONDS --timings FILE: the test rows this checkout's change from BASE reaches,
each Python's, as the test cache's own keys read it (`reached`), JSON on stdout: {"base", "policy",
"changed", "<python>": {"files", "module", "unknown"}}. The plan (ci.py's `plan`) gives each Python's rows
and those its cache keeps; each other row's last run's inputs are its last record's (`last_inputs`); a
change that leaves the cache cold or off reaches every row (Main, 2026-11-15: the tests a diff reaches, as
the trains' own planner states them, for run_tests)."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Container
from pathlib import Path

import ci


def last_inputs(python: str, rows: list[tuple[str, str]]) -> dict[tuple[str, str], list[str] | None]:
    """Each row's last run's inputs (its last record's `read`, else its key's `depends`, the first class's
    found), whatever shared inputs it ran on, so a change to those reads no less (review: rows read only on
    the head's own shared inputs left a change to them reaching nothing); [] where its record holds none;
    None where no record was read (no read token, none there, the Worker unread), which runs."""
    token, held = os.environ.get(ci.READ_TOKEN), ci.pinned_classes() or [ci.run_keys.cpu_class()]
    if not token or "DEW_XLA_CACHE_URL" not in os.environ:
        return dict.fromkeys(rows)

    def one(each_row: tuple[str, str]) -> list[str] | None:
        for each in held:
            found = ci.records.recorded(ci.records.record_key(python, ci.row(python, *each_row), each), token)
            if found is not None:
                return found.get("read") or found.get("depends") or []
        return None
    with concurrent.futures.ThreadPoolExecutor(32) as pool:
        return dict(zip(rows, pool.map(one, rows), strict=True))


def reached(python: str, rows: list[tuple[str, str]], kept: Container[tuple[str, str]],
            last: dict[tuple[str, str], list[str] | None], base: Path) -> dict:
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
    with no recorded inputs}. A row whose record was not read (`last_inputs`' None) is `files`'s: it runs."""
    rows = [each for each in rows if each not in kept]
    unread = {name for name, group in rows if last.get((name, group)) is None}
    here = os.getcwd()

    def keyed() -> tuple[dict[tuple[str, str], str], Callable[[str], str]]:
        # A side's own reading: its digests, and whether a path is a build's output, its checkout's (review).
        ci.digests.built_input.cache_clear()
        common, reading = ci.shared(python), ci.run_keys.Reading()
        alone: dict[str, str] = {}

        def single(item: str) -> str:
            # One input's own key, which says whether that input moved (`moved`).
            if item not in alone:
                alone[item] = ci.run_keys.cache_key("", "", "", [item], reading)
            return alone[item]
        return ({(name, group): ci.run_keys.cache_key(common, name, group, depends, reading)
                 for name, group in rows if (depends := last.get((name, group)))}, single)

    now, single_now = keyed()
    try:
        os.chdir(base)
        then, single_then = keyed()
        # Each input as the key covers it (`inputs`: the test's own code whole, the lookups that found each
        # module), so a changed lookup or the test's own change is a tier-one move (review).
        moved = {each: [item for item in ci.run_keys.inputs(each[0], last[each] or [])
                        if single_then(item) != single_now(item)]
                 for each, key in now.items() if then.get(each) != key}
    finally:
        os.chdir(here)
        ci.digests.built_input.cache_clear()
    top = f"{ci.followed.CODE}{ci.followed.TOP}"
    deep = unread | {each[0] for each, items in moved.items()
                     if not items or any(not item.endswith(top) for item in items)}
    module = {each[0] for each in moved} - deep
    unknown = {name for name, group in rows if last.get((name, group)) == []} - deep - module
    return {"files": sorted(deep), "module": sorted(module), "unknown": sorted(unknown)}


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    # The base's checkout beside this one, on its disk: never /tmp, memory on a planner's machine (bible §8).
    with tempfile.TemporaryDirectory(prefix="reach-base-", dir=Path.cwd().parent) as held:
        # The base's checkout, which each row's key is read against.
        subprocess.run(["git", "worktree", "add", "-q", "--detach", f"{held}/base", base], check=True,
                       capture_output=True)
        try:
            every = {python: {"files": sorted({name for name, _ in rows}), "module": [], "unknown": []}
                     for python, (rows, _) in planned.items()}
            found = {python: reached(python, rows, kept, last_inputs(python, [each for each in rows
                                                                              if each not in kept]),
                                     Path(held, "base")) if kept is not None
                     else every[python] for python, (rows, kept) in planned.items()}
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", f"{held}/base"], capture_output=True)
            ci.digests.built_input.cache_clear()
    print(json.dumps({"base": base, "policy": policy, "changed": sorted(changed), **found}))
    return 0


if __name__ == "__main__":
    # Under the Python every key is hashed by (`ci.HASHED_BY`, as ci.py's `pinned`), else run again under it.
    if f"{sys.version_info.major}.{sys.version_info.minor}" != ci.HASHED_BY:
        os.execvp("uv", ["uv", "run", "--no-project", "--python", ci.HASHED_BY, "python",
                         str(Path(__file__).resolve()), *sys.argv[1:]])
    sys.exit(main())
