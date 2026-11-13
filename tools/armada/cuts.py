"""How tools/armada/ci.py's plan cuts a test file heavier than its target into groups (`weighed`): from
the cut armada measured its rows at (`measured_cut`), whose seconds past the file's recorded tests are
what each group of that cut paid besides them (`split`), else from an estimate of its seconds alone,
spread over its tests. The standard library alone, loaded by ci.py by its path: the runs of consecutive
tests a task cuts a file by are session.py's (`session.groups`), which ci.py hands each function that cuts
(`session`)."""

import math
import re
from collections.abc import Mapping, Sequence

CUT = re.compile(r"(\d+)/(\d+)")
"""A row's group that is one of a file's cut: `G/N`; a collection's, a site's judge's or a GPU lane's process
variant's is none (`measured_cut`)."""


def split(session, tests: list[float], seconds: float, target: float, cut: int = 1) -> list[float]:
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
    if heaviest is not None and heaviest + setup <= session.KEPT * target:
        count = cut
    else:
        room = max(target - setup, target / 2)
        count = len(session.starts(session.milliseconds(tests), round(room * 1000)))
    return [sum(tests[run.start:run.stop]) + setup for run in session.groups(tests, count) if run]


"""How far past the target a split file's heaviest group may run before the file is cut again."""


def measured_cut(groups: Mapping[str, float]) -> tuple[float, int] | None:
    """What a file took at the cut its rows name, and that cut's count, from its rows' seconds by their
    `groups` (a whole file's row under ""; ci.py's `measured_cut` reads them from armada's timings): its
    groups' rows summed where every group of a count has its row (the most groups if several counts do),
    else its whole row, else None. A row's median is measured at the cut its name holds, unlike the file's
    median, which mixes the cuts of the runs it remembers. A collection's row, a site's judge's and a GPU
    lane's process variant's are no group of a cut (`CUT`)."""
    counts: dict[int, dict[int, float]] = {}
    for group, seconds in groups.items():
        if found := CUT.fullmatch(group):
            counts.setdefault(int(found[2]), {})[int(found[1])] = seconds
    complete = [count for count, held in counts.items() if len(held) == count]
    if complete:
        return sum(counts[max(complete)].values()), max(complete)
    return (groups[""], 1) if "" in groups else None


def weighed(session, tests: list[float], seconds: float, target: float,
            measured: tuple[float, int] | None, stored: Sequence[float] = ()) -> list[float]:
    """The weights of the groups the plan runs a file as, from its recorded `tests`: cut from the seconds and
    the cut its rows were `measured` at (`measured_cut`, `split`). Where no run measured it at a cut, from the
    `seconds` it is estimated at (ci.py's `weights`: its record times how much slower armada ran the files
    measured both ways, or a median that mixes the cuts of the runs it remembers), spread over its tests as
    one group that paid nothing besides: an estimate's excess over the record is no one group's setup, and
    taken for one, every group the file was then cut into paid it again (2026-11-12: with one file's timing
    left to scale by, every other file weighed 1.6 times its record and the 0.6 was its setup, so
    tests/test_model_protocols.py's 10,537 s of tests were planned as 195 groups of 6,500 s, 354 task-hours;
    armada sized the run by its longest tasks, gave the sites' 874 map tasks 20 of its 688 containers, and
    train 772's CI ran out its deadline). A file with a stored cut (`stored`: each of its groups' recorded
    seconds, session.CUTS) runs as that cut, each group its tests plus the setup each paid, while every
    group's tests stay within `session.KEPT` of `target` (`session.planned`'s test, so its tasks cut it
    alike): the durations fold alone keeps or replaces a cut (`refolded`), so a group's name means the same
    tests from run to run and its record is found again (Main, 2026-11-13); one outgrown is cut afresh, so a
    group past its deadline, which no green run would fold away, is not run as it is again (train 793's
    review)."""
    if stored and max(stored) <= session.KEPT * target:
        setup = max((measured[0] if measured is not None else seconds) - sum(stored), 0.0) / len(stored)
        return [weight + setup for weight in stored]
    if measured is not None:
        return split(session, tests, measured[0], target, measured[1])
    total = sum(tests)
    return split(session, [each * seconds / total for each in tests] if total > 0 else tests, seconds, target)


def ran(rows: Mapping[str, list[dict]]) -> dict[str, list[float]]:
    """How main's CI ran each test file, from its rows by file (ci.py's `verdict_rows`): each group's seconds
    in order (one, its whole row's, where it ran whole) where every row is green and every group of its count
    is there; none where a row is red or a group is missing."""
    runs: dict[str, list[float]] = {}
    for name, held in rows.items():
        if any(row["exitCode"] != 0 for row in held):
            continue
        groups = {row["name"].partition("#")[2]: row["seconds"] for row in held}
        if set(groups) == {""}:
            runs[name] = [groups[""]]
            continue
        found = {CUT.fullmatch(group) for group in groups}
        counts = {int(each[2]) for each in found if each}
        if None not in found and len(counts) == 1 and len(groups) == (count := counts.pop()):
            runs[name] = [groups[f"{group}/{count}"] for group in range(1, count + 1)]
    return runs


def refolded(session, stored: Mapping[str, list[str]], runs: Mapping[str, list[float]], target: float,
             before: Mapping[str, dict[str, float]],
             after: Mapping[str, dict[str, float]]) -> dict[str, list[str]]:
    """The stored cuts (session.CUTS) after main's CI ran each file as `runs` says (`ran`), every group within
    `KEPT` of `target`, setup and all: a file it ran as its stored cut keeps it; one it ran as another cut
    stores that cut, made from the record it was cut by (`before`: each file's own tests, node id to seconds).
    A group past that cuts the file anew from the folded record (`after`), as `split` cuts a file it has no
    cut for. Each group's first test is its boundary; a file run whole within its bound, or cut into a group
    that holds none of its recorded tests, stores none; a file it did not run keeps its own. The fold is the
    one place a cut changes (Main, 2026-11-13: a group's name means the same tests from run to run, so its
    record is found again)."""
    kept = dict(stored)
    for name, seconds in runs.items():
        bounds = stored.get(name)
        if max(seconds) <= session.KEPT * target:
            if bounds is not None and len(bounds) + 1 == len(seconds):
                continue
            own, count = before.get(name, {}), len(seconds)
        else:
            own = after.get(name, {})
            units = session.unit_seconds(list(own), list(own.values()))
            count = len(split(session, units, sum(seconds), target, len(seconds))) if own else 1
        kept.pop(name, None)
        if count > 1 and own:
            groups = session.cut(list(own), list(own.values()), count)
            if all(groups):
                kept[name] = [group[0] for group in groups[1:]]
    return kept
