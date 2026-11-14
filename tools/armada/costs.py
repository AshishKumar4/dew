"""What a check-1 or `shards` site's map task costs (tools/armada/ci.py's `rate`, `site_costs`): its setup,
paid once a task, and each unit it draws after, as its map tasks' own records say (map_judge's
`record_units`, `before` and `draws`), else as armada's measures of its tasks' seconds tell them apart. The
standard library alone, loaded by ci.py by its path."""

import dataclasses
import statistics
from collections.abc import Mapping, Sequence
from typing import NotRequired, TypedDict

SITE_COSTS = ("setup", "unit", "weighed_unit")
"""What a site's judge times of its map tasks' records (ci.py's `site_costs`), each under the site's task
name and its word (`costed`): the seconds a task pays once whatever it draws (map_judge's `before`:
environment, imports, collection and compiles; and what its draws pay once, `timed`), and a unit's seconds
of their draws, of its unweighted check's and of its weighed check's apart; each the median of its tasks.
armada keeps them among the run's file timings, which the next plan reads (ci.py's `rate`)."""


def costed(task: str, word: str) -> str:
    """The timing a site's judge reports its map tasks' `word` of `SITE_COSTS` under, its site's task
    `task`."""
    return f"{task}.{word}"


SITE_TARGETS = 2
"""How many of the plan's targets (a row's seconds, .armada.json's) a site's map task of its declared cut may
take before the plan cuts it finer (ci.py's `cut_draws`). Every map task pays its site's setup again, so cut
down to one target, sites declared at 8 to 64 draws a task ran as 640 tasks of one or two draws: of the
suite's 167 check-1 task-hours 56 were their setups (2026-11-12, the run at ac69e5e9ede; by the 115 sites'
recorded costs, 11,436 map tasks and 483 h cut to one target, 4,647 and 327 h to four). A site's chain is its
longest map task and then its judge, which a train waits for: at four targets, 480 s of draws and a judge of up
to 320 s set the wall of every train past the owner's 15 minutes, whose bound for a row is 530 to 730 s
(2026-11-14: test_narrow's 955 s chain, test_parallelism_matrix's 915 s; Main's target). Two targets, 240 s,
hold a chain within it; the setups they add fall in a pool that is not the limit (300 to 460 containers of
688)."""


@dataclasses.dataclass(frozen=True)
class Cost:
    """What a site's map task takes: its setup, once a task (its environment, imports and compiles before its
    first draw), and each unit it draws after (2026-11-09: a task of one draw and one of two both took 516 s,
    so task seconds over units charged each draw the setup a task pays once)."""

    setup: float
    unit: float

    def seconds(self, units: int) -> float:
        """A task's seconds drawing `units`."""
        return self.setup + self.unit * units

    def fitting(self, declared: int, most: float) -> int:
        """The most units a task draws within `most` seconds past its setup, at most `declared`, at least one;
        where its setup alone fills them, the declared cut, as every finer task pays the setup again."""
        if not self.unit or self.setup >= most:
            return declared
        return max(1, min(declared, int((most - self.setup) // self.unit)))


def recorded(task: str, costs: Mapping[str, float], *, weighed: bool) -> Cost | None:
    """Site task `task`'s cost as its judge timed its map tasks' records (`SITE_COSTS` among `costs`, armada's
    file timings), its weighed check's units' or the rest's; None where it timed none."""
    setup = costs.get(costed(task, "setup"))
    unit = costs.get(costed(task, "weighed_unit" if weighed else "unit"))
    return None if setup is None or unit is None else Cost(setup, unit)


def fitted(timed: Sequence[tuple[int, float]]) -> Cost | None:
    """A site's cost from armada's measures of its map tasks, each (units, seconds): tasks of two sizes or
    more tell its setup from its draws (a least-squares line of seconds over units, its setup and draws none
    below zero); tasks of one size alone no setup, their seconds all the draws'; None where none was
    measured."""
    if not timed:
        return None
    sizes, seconds = [count for count, _ in timed], [taken for _, taken in timed]
    if len(set(sizes)) < 2:
        return Cost(0.0, sum(seconds) / sum(sizes))
    unit, setup = statistics.linear_regression(sizes, seconds)
    if unit < 0:
        return Cost(statistics.fmean(seconds), 0.0)
    # A line through the origin where the fit's setup falls below zero: no task pays less than its draws.
    return Cost(setup, unit) if setup >= 0 else Cost(
        0.0, statistics.linear_regression(sizes, seconds, proportional=True).slope)


class Drawn(TypedDict):
    """One label's record of a site's map task (map_judge's `record_units`): the plan's span its task draws
    (`task_start` to `task_stop`), the seconds from its process's start to its draws (`before`) and its draws'
    (`draws`); and, where its record says them, what its draws paid once: its first draw's seconds (`first`),
    the backend compiles JAX reported within that draw (`compiled`), and its weighed check's weighing
    (`weighing`). A record written before these were kept holds none, and is read as its draws alone."""

    task_start: int
    task_stop: int
    before: float
    draws: float
    first: NotRequired[float]
    compiled: NotRequired[float]
    weighing: NotRequired[float]


def _once(records: Sequence[Drawn], units: int) -> float:
    """What a map task's draws paid once whatever they drew, of `records`, its labels', over `units` units:
    each label's weighing, and of its first draw what its later draws do not repeat. A task of two units or
    more measures a later draw (its draws past the first, over the units past the first), so its first
    draw's excess over one is paid once; a task of one unit has no later draw, and JAX's own account of
    the first one's backend compiles is what it paid once, the rest of it its one draw (an over-estimate of
    the draw where its traces and lowerings are long, never an under-estimate, and never a draw of no
    seconds: compiles reported at the whole draw or more tell nothing of it). Records that keep no first
    draw (`Drawn`) say nothing paid once: every second of their draws is a draw's, as before they did."""
    if not all("first" in record for record in records):
        return 0.0
    weighing = sum(record.get("weighing", 0.0) for record in records)
    drawing = sum(record["draws"] for record in records) - weighing
    first = sum(record["first"] for record in records)
    if units < 2:
        # Compiles reported at the whole draw or more (another thread's among them) leave no draw to tell.
        compiled = sum(record.get("compiled", 0.0) for record in records)
        return weighing + (compiled if compiled < first else 0.0)
    # Each label's first draw less one later draw of it; none where a later draw took longer than the first.
    return weighing + max(first - (drawing - first) / (units - 1), 0.0)


def timed(task: str, records: Sequence[Drawn], units: int) -> dict[str, float]:
    """`SITE_COSTS` of site task `task` from its map tasks' `records`, a label's each: of each map task (the
    plan's span it drew), what it paid once (its setup: all it paid outside its draws up to its last label's
    end, its imports, collection and every compile, a layout's between two labels' draws among them; and
    what its draws paid once, `_once`: a jit compiles at its first call, inside a site's first draw, which
    read as a draw's made each finer task pay it again, row 5492) and the rest of its labels' draws over
    its units; each the median of its tasks, those past `units` (its weighed check's) apart. Nothing where
    no record is given."""
    spans: dict[tuple[int, int], list[Drawn]] = {}
    for record in records:
        spans.setdefault((record["task_start"], record["task_stop"]), []).append(record)
    if not spans:
        return {}
    setups, rates = [], {False: [], True: []}
    for (start, stop), held in spans.items():
        end, draws = max(record["before"] + record["draws"] for record in held), sum(
            record["draws"] for record in held)
        once = _once(held, stop - start)
        setups.append(end - draws + once)
        rates[start >= units].append((draws - once) / (stop - start))
    found = {costed(task, "setup"): statistics.median(setups)}
    for word, weighed in (("unit", False), ("weighed_unit", True)):
        if rates[weighed]:
            found[costed(task, word)] = statistics.median(rates[weighed])
    return found
