"""How tools/armada/ci.py's plan cuts a test file heavier than its target into groups (`weighed`): from the
cut armada measured its rows at (`measured_cut`), whose seconds past the file's recorded tests are what each
group of that cut paid besides them (`split`), else from an estimate of its seconds alone, spread over its
tests. Loaded by ci.py by its path, beside session.py, whose runs of consecutive tests (`session.groups`) a
task cuts a file by."""

import importlib.util
import math
import re
import types
from pathlib import Path


def _loaded(path: Path, name: str) -> types.ModuleType:
    """The module at `path`, loaded by its path as ci.py loads its neighbours."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


session = _loaded(Path(__file__).resolve().with_name("session.py"), "armada_session")
rows = _loaded(Path(__file__).resolve().with_name("rows.py"), "armada_rows")
CUT = re.compile(r"(\d+)/(\d+)")
"""A row's group that is one of a file's cut: `G/N`; a collection's, a site's judge's or a GPU lane's
process variant's is none (`measured_cut`)."""


def split(tests: list[float], seconds: float, target: float, cut: int = 1) -> list[float]:
    """The weights of the groups a file that took `seconds` as `cut` groups runs as. Of those
    seconds, what its recorded `tests`' times leave is what each group paid besides its tests (its
    module's fixtures, imports and compiles, which a test's least recorded time does not hold), so
    each group weighs its tests plus that setup. The file is cut (`session.groups`) into the fewest runs
    whose tests fit what the setup leaves of `target`, at least half of it, so a file whose setup
    is most of a group is not cut into slivers that each pay it again; a heavier test runs alone.
    Counting the setup once a group, rather than scaling every test by the file's total, keeps a
    file from being cut finer each run for the setup its last cut added. A file keeps the cut it
    ran as while that cut's heaviest group, setup included, stays within `KEPT` of `target`:
    a group's compiled programs are cached under its cut (tests/remote_cache.py), so a new cut
    compiles every group from nothing, and the setup that measures moves the next cut again
    (armada dew-20261020144644-dceff433: a new cut fetched no programs, and 14 of its 20 split
    files would have been cut again). A file with no recorded time runs as equal counts of its
    tests."""
    if sum(tests) <= 0:
        count = math.ceil(seconds / target)
        return [seconds / count] * count
    setup = max(seconds - sum(tests), 0.0) / cut
    heaviest = max(sum(tests[run.start:run.stop]) for run in session.groups(tests, cut)) if cut > 1 else None
    if heaviest is not None and heaviest + setup <= KEPT * target:
        count = cut
    else:
        room = max(target - setup, target / 2)
        count = len(session.starts(session.milliseconds(tests), round(room * 1000)))
    return [sum(tests[run.start:run.stop]) + setup for run in session.groups(tests, count) if run]


KEPT = 1.5
"""How far past the target a split file's heaviest group may run before the file is cut again."""


def measured_cut(timings: dict, python: str, name: str) -> tuple[float, int] | None:
    """What `name` took under `python` at the cut its rows in `timings` name, and that cut's count: its
    groups' rows summed where every group of a count has its row (the most groups if several counts do),
    else its whole row, else None. A row's median is measured at the cut its name holds, unlike the file's
    median, which mixes the cuts of the runs it remembers. A map task's seconds are kept under its task's
    name, no row's, and a collection's row, a site's judge's and a GPU lane's process variant's are no group
    of a cut (`CUT`)."""
    whole, counts = None, {}
    for row_name, seconds in timings.get("rows", {}).items():
        if ":" not in row_name:
            continue
        python_, name_, group = rows.parsed_row(row_name)
        if (python_, name_) != (python, name):
            continue
        if not group:
            whole = seconds
        elif found := CUT.fullmatch(group):
            counts.setdefault(int(found[2]), {})[int(found[1])] = seconds
    complete = [count for count, held in counts.items() if len(held) == count]
    if complete:
        return sum(counts[max(complete)].values()), max(complete)
    return None if whole is None else (whole, 1)


def weighed(tests: list[float], seconds: float, target: float,
            measured: tuple[float, int] | None) -> list[float]:
    """The weights of the groups the plan runs a file as, from its recorded `tests`: cut from the seconds
    and the cut its rows were `measured` at (`measured_cut`, `split`). Where no run measured it at a cut,
    from the `seconds` it is estimated at (ci.py's `weights`: its record times how much slower armada ran
    the files measured both ways, or a median that mixes the cuts of the runs it remembers), spread over its
    tests as one group that paid nothing besides: an estimate's excess over the record is no one group's
    setup, and taken for one, every group the file was then cut into paid it again (2026-11-12: with one
    file's timing left to scale by, every other file weighed 1.6 times its record and the 0.6 was its setup,
    so tests/test_model_protocols.py's 10,537 s of tests were planned as 195 groups of 6,500 s, 354 task-
    hours; armada sized the run by its longest tasks, gave the sites' 874 map tasks 20 of its 688
    containers, and train 772's CI ran out its deadline)."""
    if measured is not None:
        return split(tests, measured[0], target, measured[1])
    total = sum(tests)
    return split([each * seconds / total for each in tests] if total > 0 else tests, seconds, target)
