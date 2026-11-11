"""What a check-1 or `shards` site's map task costs (tools/armada/ci.py's `rate`, `site_costs`): its setup,
paid once a task, and each unit it draws after, as its map tasks' own records say (map_judge's
`record_units`, `before` and `draws`), else as armada's measures of its tasks' seconds tell them apart. The
standard library alone, loaded by ci.py by its path."""

import dataclasses
import math
import statistics
from collections.abc import Mapping, Sequence

SITE_COSTS = ("setup", "unit", "weighed_unit")
"""What a site's judge times of its map tasks' records (ci.py's `site_costs`), each under the site's task
name and its word (`costed`): their seconds before their first draw (map_judge's `before`: environment,
imports, collection and compiles), and a unit's seconds of their draws (`draws` over their units), of its
unweighted check's and of its weighed check's apart; each the median of its tasks. armada keeps them among
the run's file timings, which the next plan reads (ci.py's `rate`)."""


def costed(task: str, word: str) -> str:
    """The timing a site's judge reports its map tasks' `word` of `SITE_COSTS` under, its site's task
    `task`."""
    return f"{task}.{word}"


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

    def fitting(self, declared: int, target: float) -> int:
        """The most units a task draws within `target` past its setup, at most `declared`, at least one; where
        its setup alone fills the target, the declared cut, as every finer task pays the setup again."""
        if not self.unit or self.setup >= target:
            return declared
        return max(1, min(declared, int((target - self.setup) // self.unit)))


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


def timed(task: str, records: Sequence[Mapping], units: int) -> dict[str, float]:
    """`SITE_COSTS` of site task `task` from its map tasks' `records` (map_judge's: each a span's `start`,
    `stop`, `before` and `draws`, one a label of the site's that span drew): of each span, its least seconds
    before a draw (its task's setup) and its labels' draws summed, over its units; each the median of the
    spans, those past `units` (its weighed check's) apart. Nothing where no record is given."""
    spans: dict[tuple[int, int], tuple[float, float]] = {}
    for record in records:
        span = (record["start"], record["stop"])
        before, draws = spans.get(span, (math.inf, 0.0))
        spans[span] = (min(before, record["before"]), draws + record["draws"])
    if not spans:
        return {}
    found = {costed(task, "setup"): statistics.median(before for before, _ in spans.values())}
    for word, weighed in (("unit", False), ("weighed_unit", True)):
        rates = [draws / (stop - start) for (start, stop), (_, draws) in spans.items()
                 if (start >= units) == weighed]
        if rates:
            found[costed(task, word)] = statistics.median(rates)
    return found
