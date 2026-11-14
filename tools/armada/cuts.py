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
    train 772's CI ran out its deadline). A file with a stored cut that holds (`stored`: each of its groups'
    recorded seconds, session.CUTS, where ci.py's `stored_weights` finds `session.holds` as `session.planned`
    does, whatever target a plan is asked for, so its tasks cut it alike: review) runs as that cut, each
    group its tests plus the setup each paid: the durations fold alone keeps or replaces a cut (`refolded`),
    so a group's name means the same tests from run to run and its record is found again (Main, 2026-11-13);
    one outgrown is cut afresh, so a group past its deadline, which no green run would fold away, is not run
    as it is again (train 793's review)."""
    if stored:
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
    A group past that, in a file that ran as its stored cut, cuts its nearest groups again alone
    (`rebalanced`); where none fit, or the file ran as another cut, the file is cut anew from the folded
    record (`after`), as `split` cuts a file it has no cut for. Each group's first test is its boundary; a
    file run whole within its bound, or cut into a group that holds none of its recorded tests, stores none; a
    file it did not run keeps its own. The fold is the one place a cut changes (Main, 2026-11-13: a group's
    name means the same tests from run to run, so its record is found again)."""
    kept = dict(stored)
    for name, seconds in runs.items():
        bounds = stored.get(name)
        if max(seconds) <= session.KEPT * target:
            if bounds is not None and len(bounds) + 1 == len(seconds):
                continue
            own, count = before.get(name, {}), len(seconds)
        else:
            own = after.get(name, {})
            # Its stored cut ran: only the outgrown groups' neighbourhoods are cut again (`rebalanced`).
            local = (rebalanced(session, own, bounds, seconds, session.KEPT * target)
                     if own and bounds is not None and len(bounds) + 1 == len(seconds) else None)
            if local is not None:
                kept[name] = local
                continue
            units = session.unit_seconds(list(own), list(own.values()))
            count = len(split(session, units, sum(seconds), target, len(seconds))) if own else 1
        kept.pop(name, None)
        if count > 1 and own:
            groups = session.cut(list(own), list(own.values()), count)
            if all(groups):
                kept[name] = [group[0] for group in groups[1:]]
    return kept


def rebalanced(session, own: Mapping[str, float], bounds: list[str], measured: list[float],
               limit: float) -> list[str] | None:
    """Stored cut `bounds` of a file whose groups ran as `measured` (seconds, setup included), its record
    `own` (node id to seconds), with each group past `limit`, as it ran or as its recorded tests weigh with
    the setup it paid (a test added since falls in the group it sorts into), cut again together with its
    nearest groups alone, as few as bring every group of them within it, their setup shared alike: the
    other groups keep their tests and so their names' records and compiled programs (2026-11-14: one group
    past its bound re-cut test_recurrence_stability.py whole, 9 of 11 bounds moved, every group run cold).
    A group that is one unit (a test heavier than the bound) runs alone however the file is cut, as the plan
    keeps it (`session.holds`). None where a group holds no recorded test or no neighbourhood short of the
    whole file fits, which is then cut anew whole."""
    groups = session.cut_by(list(own), bounds)
    if not all(groups):
        return None
    setup = [max(ran - sum(own[node] for node in group), 0.0)
             for ran, group in zip(measured, groups, strict=True)]
    weighs = [max(ran, sum(own[node] for node in group) + paid)
              for ran, group, paid in zip(measured, groups, setup, strict=True)]

    def outgrown() -> int | None:
        return next((index for index, group in enumerate(groups)
                     if weighs[index] > limit and len(session.units(group)) > 1), None)

    # Each pass brings one more group within the bound and moves none outside its window: it ends.
    while (index := outgrown()) is not None:
        for reach in range(1, len(groups)):
            low, high = max(0, index - reach), min(len(groups), index + reach + 1)
            if (low, high) == (0, len(groups)):
                return None
            tests = [node for group in groups[low:high] for node in group]
            cut = session.cut(tests, [own[node] for node in tests], high - low)
            shared = sum(setup[low:high]) / (high - low)
            if all(cut) and session.holds(cut, own, limit - shared):
                groups[low:high], setup[low:high] = cut, [shared] * (high - low)
                weighs[low:high] = [sum(own[node] for node in group) + shared for group in cut]
                break
        else:
            return None
    return [group[0] for group in groups[1:]]


def stale(session, own: dict[str, float], observed: dict[str, float], seconds: float, cut: int,
          target: float) -> bool:
    """Whether a file's record `own` no longer cuts it as it runs: its tests differ from those its rows
    `observed` (one added, removed or renamed); a file within `target` (it took `seconds`), or one with no
    tests to cut, is recorded past it, so the plan would split what runs whole; or, for a file past it (as
    `cut` groups), the groups the plan cuts from the record (`split`) run a group past `session.KEPT` of
    the target as its tests ran, setup included, and those it would cut from what they took run their
    heaviest shorter. A cut that fits is kept whatever its tests' times drift, as the plan keeps it, and so
    is one whose heaviest group no cut shortens (a test heavier than the budget, which runs alone either
    way): a fold rewrites a file only where its plan changes for the better."""
    if own.keys() != observed.keys():
        return True
    if seconds <= target or not own:
        return sum(own.values()) > target
    nodes = sorted(own)
    recorded = session.unit_seconds(nodes, [own[node] for node in nodes])
    ran = session.unit_seconds(nodes, [observed[node] for node in nodes])
    setup = max(seconds - sum(ran), 0.0) / cut

    def heaviest(weights: list[float]) -> float:
        count = len(split(session, weights, seconds, target, cut))
        return max(sum(ran[run.start:run.stop]) + setup for run in session.groups(weights, count) if run)

    return heaviest(recorded) > session.KEPT * target and heaviest(ran) < heaviest(recorded)
