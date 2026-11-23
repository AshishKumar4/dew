"""What a check-1 or `shards` site's map task costs (tools/armada/ci.py's `rate`, `site_costs`): its setup,
paid once a task, and each unit it draws after, as its map tasks' own records say (map_judge's
`record_units`, `before` and `draws`), else as armada's measures of its tasks' seconds tell them apart; and
a site's mark (`Mark`) and the map tasks its units are cut into by that cost (`cut_draws`). The standard
library alone, loaded by ci.py by its path."""

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
longest map task and then its judge, which a train waits for: at four targets, 480 s of draws and a judge of
up to 320 s set the wall of every train past the owner's 15 minutes, whose bound for a row is 530 to 730 s
(2026-11-14: test_narrow's 955 s chain, test_parallelism_matrix's 915 s; Main's target). Two targets, 240 s,
hold a chain within it; the setups they add fall in a pool that is not the limit (300 to 460 containers of
688)."""


def evenly(first: int, stop: int, each: int) -> list[tuple[int, int]]:
    """Units `first` to `stop` in tasks of `each`, the last of the rest."""
    return [(start, min(start + each, stop)) for start in range(first, stop, each)]


@dataclasses.dataclass(frozen=True)
class Cost:
    """What a site's map task takes: its setup, once a task (its environment, imports and compiles before its
    first draw), and each unit it draws after (2026-11-09: a task of one draw and one of two both took 516 s,
    so task seconds over units charged each draw the setup a task pays once)."""

    setup: float
    unit: float
    spans: tuple[tuple[int, int, float], ...] = ()
    """Each measured span's own seconds a unit (`timed`'s `span_word`), its start and stop: a site's units
    need not cost alike (row 8611: a rung walk's first three places took 192 s, the rest 54 to 83 s)."""

    def unit_at(self, index: int) -> float:
        """Unit `index`'s seconds: its measured span's, else every span's median (`unit`)."""
        return next((rate for start, stop, rate in self.spans if start <= index < stop), self.unit)

    def seconds(self, start: int, stop: int) -> float:
        """A task's seconds drawing units `start` to `stop`."""
        return self.setup + sum(self.unit_at(index) for index in range(start, stop))

    def cut(self, first: int, stop: int, declared: int, most: float, least: int = 1) -> list[tuple[int, int]]:
        """Units `first` to `stop` in tasks of at most `declared` and at least `least` each (the last may be
        less), each as many as fit within `most` seconds by each unit's own seconds (`unit_at`), so a heavy
        span is a task alone and light ones merge; where its setup alone fills them, the declared cut, as
        every finer task pays the setup again. A unit of a span measured past `most` is a task alone: its
        span's rate is its units' mean, which a heavy unit beside light ones hides, so a cut that moved one
        of them into a neighbour's task could swing back and forth past `most` run after run; alone, each is
        measured by itself next."""
        if self.setup >= most:
            return evenly(first, stop, declared)
        alone = {index for start, stop_, rate in self.spans if self.setup + rate * (stop_ - start) > most
                 for index in range(start, stop_)}
        spans, start = [], first
        while start < stop:
            end = start + 1
            while end < stop and end - start < declared and (
                    end - start < least or (self.seconds(start, end + 1) <= most
                                            and not {start, end} & alone)):
                end += 1
            spans.append((start, end))
            start = end
        return spans


def span_word(start: int, stop: int, *, weighed: bool) -> str:
    """The word a site's span's own unit seconds are timed under (`costed`, `timed`), its weighed check's
    apart from the rest's, as their medians are (`SITE_COSTS`): a mark whose weighed check moves takes no rate
    of a span the other stage measured."""
    return f"{'weighed_unit' if weighed else 'unit'}-{start}-{stop}"


def recorded(task: str, costs: Mapping[str, float], *, weighed: bool) -> Cost | None:
    """Site task `task`'s cost as its judge timed its map tasks' records (`SITE_COSTS` among `costs`, armada's
    file timings), its weighed check's units' or the rest's, with each span's own (`span_word`); None where
    it timed none."""
    setup = costs.get(costed(task, "setup"))
    unit = costs.get(costed(task, "weighed_unit" if weighed else "unit"))
    prefix = costed(task, span_word(0, 0, weighed=weighed)).removesuffix("0-0")
    spans = tuple(sorted((int(start), int(stop), rate) for key, rate in costs.items()
                         if key.startswith(prefix) for start, _, stop in [key[len(prefix):].partition("-")]))
    return None if setup is None or unit is None else Cost(setup, unit, spans)


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
    first = sum(record.get("first", 0.0) for record in records)
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
    its units; each the median of its tasks, those past `units` (its weighed check's) apart; and each span's
    own seconds a unit (`span_word`), which the plan cuts by unit (`Cost.cut`). Nothing where no record is
    given."""
    spans: dict[tuple[int, int], list[Drawn]] = {}
    for record in records:
        spans.setdefault((record["task_start"], record["task_stop"]), []).append(record)
    if not spans:
        return {}
    setups, rates, own = [], {False: [], True: []}, {}
    for (start, stop), held in spans.items():
        end, draws = max(record["before"] + record["draws"] for record in held), sum(
            record["draws"] for record in held)
        once = _once(held, stop - start)
        setups.append(end - draws + once)
        rates[start >= units].append((draws - once) / (stop - start))
        own[costed(task, span_word(start, stop, weighed=start >= units))] = (draws - once) / (stop - start)
    found = {costed(task, "setup"): statistics.median(setups), **own}
    for word, weighed in (("unit", False), ("weighed_unit", True)):
        if rates[weighed]:
            found[costed(task, word)] = statistics.median(rates[weighed])
    return found


MAX_INPUTS = 1000
"""The most tasks' artifacts one armada job takes as its inputs (armada's MAX_INPUT_RUNS,
src/protocol.ts): armada cuts the judges of one environment and size into as many judges' jobs as keep
each within it (its `judgeChunks`), so a site's own map tasks, which one judge reads in one job, stay
within it (`refused_sites`)."""


@dataclasses.dataclass(frozen=True)
class Mark:
    """A site's mark (`session.SHARDS`): the units its map tasks together take, 0 to `units`, the most of them
    a map task takes, the line of its test's source that holds the mark, and the units a whole run of its test
    takes at least, over which its recorded seconds weigh one (`site_entries`): a check-1 site's declared
    seeds, of (1 + REDRAW) times as many its map tasks draw; a `shards` test's units, every one. A check-1
    site's weighed check (`conditioning`) draws `weighed` units more, `units` to `units + weighed`, at most
    `weighed_per_task` a map task: its draws cost their own, apart from the unweighted check's."""

    units: int
    per_task: int
    line: int
    timed: int
    weighed: int = 0
    weighed_per_task: int = 0


def stages_of(mark: Mark) -> list[tuple[int, int]]:
    """A site's unweighted units by stage (`draws_of`): a check-1 site's first (0 to its declared seeds,
    `timed`) and its second apart, a `shards` test's units as one."""
    return [(0, mark.timed), (mark.timed, mark.units)] if mark.units > mark.timed else [(0, mark.units)]


def draws_of(mark: Mark, per_task: int | None = None, weighed_per_task: int | None = None
             ) -> list[tuple[int, int]]:
    """The units each map task of a site marked `mark` takes, at most its per_task each (`per_task` where
    given, as `cut_draws` cuts finer): together 0 to its units, every seed both of check 1's stages draw, or
    every unit of a `shards` test's work; a check-1 site's first stage (0 to its declared seeds, `timed`) and
    its second each in tasks of their own, the second's `later` (`site_entries`); and its weighed check's
    units past them (`Mark.weighed`), at most `weighed_per_task` a task, every one later."""
    each = per_task or mark.per_task
    spans = [(start, min(start + each, stop)) for first, stop in stages_of(mark)
             for start in range(first, stop, each)]
    if not mark.weighed:
        return spans
    end, weighed = mark.units + mark.weighed, weighed_per_task or mark.weighed_per_task
    return spans + [(start, min(start + weighed, end)) for start in range(mark.units, end, weighed)]


def cut_draws(mark: Mark, rates: Mapping[bool, Cost | None], target: float) -> list[tuple[int, int]]:
    """The spans of a site's map tasks (`draws_of`'s stages): its declared per_task each, a ceiling, or fewer
    where its measured cost (`site_rate`, the unweighted check's and the weighed check's apart, `rates` by
    whether weighed) puts a task past `SITE_TARGETS` of `target`: as many units as fit past its setup
    by each unit's own measured seconds (`Cost.cut`), at least one, so no map task is the run's longest for
    its mark's cut (row 4376: a 64-seed shard ran 966 s where every other group took 256 s or less), a heavy
    span is a task alone and light ones merge (row 8611: a rung walk's first three places took 192 s and the
    rest 54 to 83 s, and a cut at their median rate was capped by hand), and none is cut down to one target,
    each task paying the site's setup again. Its tasks' spans are among what the cost measures, so a cut
    converges on the one that fills them. A cut that would pass what its judges' job takes (`MAX_INPUTS`, the
    two kinds' tasks together) keeps the finest within it, its tasks at least so many units each."""
    most = target * SITE_TARGETS
    stages = [(start, stop, False) for start, stop in stages_of(mark)]
    stages += [(mark.units, mark.units + mark.weighed, True)] if mark.weighed else []

    def spans(least: int) -> list[tuple[int, int]]:
        found = []
        for first, stop, weighed in stages:
            declared = mark.weighed_per_task if weighed else mark.per_task
            cost = rates[weighed]
            found += (cost.cut(first, stop, declared, most, min(least, declared)) if cost is not None
                      else evenly(first, stop, declared))
        return found

    least = 1
    while len(found := spans(least)) > MAX_INPUTS and least < max(mark.per_task, mark.weighed_per_task):
        least += 1
    return found


def refused_sites(marked: Mapping[str, Mapping[str, Mark]]) -> None:
    """Refuse the marks of `marked` (each test file's sites, `sites`) whose own map tasks pass what one
    judges' job takes (`MAX_INPUTS`): its judge reads them all in one job, which armada would refuse after
    every map task ran, and the run would go ungraded; marks within it pass it together, armada cutting
    their judges into jobs. A mark is refused on the run that adds it, naming its line."""
    over = {(name, test): count for name, held in marked.items() for test, mark in held.items()
            if (count := len(draws_of(mark))) > MAX_INPUTS}
    if over:
        each = "; ".join(f"{name}, line {marked[name][test].line}: {test}, {count} map tasks"
                         for (name, test), count in sorted(over.items()))
        raise SystemExit(f"a site marked check1_shards or shards plans more map tasks than the {MAX_INPUTS} "
                         "tasks' artifacts its judges' job takes; give a map task more units (per_task): "
                         f"{each}")
