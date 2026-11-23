"""Evaluation policies shared by fit and recorded run configurations."""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal

from dew.objectives.base import Metric, Shown, TrainingScalar

if TYPE_CHECKING:
    from dew.checkpoints import Ranking
    from dew.state import TrainState

from dew.scores import Summary
from dew.training.evaluation import Declared, SplitLoss


@dataclasses.dataclass(frozen=True, init=False)
class Best:
    """Keeps the `top` best evaluations by a metric, named, or by a custom `score` of the fit's scores.

    The metric is named as its scores are recorded, `<split>/<name>` (`val/accuracy`, `train/loss`,
    `evaluation/summary/<name>` for a `Summary`), or by its bare name beside `split`, or the only split's;
    a metric object, a training scalar or a summary given in its place is named by its own name and
    split. `mode` sets the direction, 'min' or 'max'; where it is None, a custom score is minimized and a
    named metric takes its direction from its `Shown` declaration. An evaluation whose score is not
    better than `threshold` is not ranked. A custom score is code alone: a run's record names metrics.
    """
    metric: str
    top: int = 1
    mode: Literal['min', 'max'] | None = None
    threshold: float | None = None
    weights_only: bool = False
    split: str | None = None
    score: Callable[[Mapping], float] | None = dataclasses.field(
        default=None, repr=False, compare=False, metadata={'record': False})

    def __init__(self, metric: str | Metric | TrainingScalar | Summary | Callable[[Mapping], float],
                 top: int = 1, mode: Literal['min', 'max'] | None = None, threshold: float | None = None,
                 weights_only: bool = False, split: str | None = None, *,
                 score: Callable[[Mapping], float] | None = None):
        if top < 1 or mode not in (None, 'min', 'max'):
            raise ValueError("Best needs integer top >= 1 and mode min or max")
        if threshold is not None and not math.isfinite(threshold):
            raise ValueError("Best.threshold must be finite")
        if isinstance(metric, TrainingScalar):
            name, split = metric.name, "train"
        elif isinstance(metric, Summary):
            name, split = metric.name, Summary.SPLIT if split is None else split
        elif isinstance(metric, (Metric, SplitLoss)):
            name = metric.name
        elif isinstance(metric, str):
            name = metric
        else:
            name, score = AGGREGATE, metric
        for field, value in (('metric', name), ('top', top), ('mode', mode), ('threshold', threshold),
                             ('weights_only', weights_only), ('split', split), ('score', score)):
            object.__setattr__(self, field, value)

    def resolved(self, scored: Mapping[str, Sequence[Declared]],
                 scalar: Callable[[str], TrainingScalar] | None = None) -> Best:
        """This selection bound to the split it ranks and the bare name of its metric there: `train` for a
        training scalar `scalar` names (the objective's, `Objective.scalars`), else the split of `scored`
        (each split's metrics) it names, by `split` or as its name's prefix, or the only one, its metric found
        among that split's own by name. Its direction, where it names none, is how that metric declares it is
        shown. A custom score binds to nothing."""
        if self.score is not None:
            return self
        selection = self
        # A summary is ranked in its namespace, which no split can be named, so no other selection's
        # split defaults to it. A recorded name's split is all before its last '/', as a namespace is.
        splits = tuple(split for split in scored if split != Summary.SPLIT)
        prefix = selection.metric.rpartition('/')[0]
        if selection.split is None and (prefix in scored or prefix == "train"):
            selection = dataclasses.replace(selection, split=prefix)
        if selection.split is None:
            if len(splits) != 1:
                raise ValueError("Best(metric, split=...) is required with several validation splits")
            selection = dataclasses.replace(selection, split=splits[0])
        elif selection.split not in scored and selection.split != "train":
            raise ValueError(f"unknown validation split {selection.split!r}")
        split = selection.split
        assert split is not None, "a selection is bound to its split above"
        name = selection.metric.removeprefix(f"{split}/")
        declared: Declared | TrainingScalar
        if split == "train":
            if scalar is None:
                raise ValueError(f"best training scalar {name!r} ranks a fit's objective's own")
            declared = scalar(name)
        else:
            found = next((declared for declared in scored[split] if declared.name == name), None)
            if found is None:
                raise ValueError(f"best metric {selection.metric!r} is not among fit's metrics")
            declared = found
        selection = dataclasses.replace(selection, metric=name)
        if selection.mode is None:
            shown = _declared_shown(declared)
            better = None if shown is None else shown.better
            if better not in ('higher', 'lower'):
                raise ValueError("best metric has no declared direction; use Best(metric, mode='min' or "
                                 "'max')")
            selection = dataclasses.replace(selection, mode='max' if better == 'higher' else 'min')
        return selection


AGGREGATE = "<aggregate>"
"""The name a custom score ranks under (`Best.score`), which names no recorded metric."""


def _declared_shown(metric: Declared | TrainingScalar) -> Shown | None:
    """How `metric` itself declares it is shown, which the Metric protocol leaves optional, so it is
    read at this boundary."""
    return getattr(metric, "shown", None)


def rankings(best: Sequence[Best], scores: Mapping[str, float], *, validation: str | None,
             declared: Mapping[str, Sequence[Declared]], objective) -> tuple[Ranking, ...]:
    """How an evaluation's `scores` rank its state: by each `best` policy, a threshold it misses
    ranking nothing (a NaN), or without one by the `validation` split's loss, else the training
    loss. A step no evaluation scored ranks nothing, and asks no aggregate. `declared` (each
    split's metrics) and `objective` read an aggregate's scores (`MetricScores`)."""
    from dew.checkpoints import Ranking
    from dew.training.evaluation import MetricScores

    if not scores:
        return ()
    if not best:
        name = f"{validation}/loss" if validation is not None else "train/loss"
        return (Ranking(name, scores[name]),) if name in scores else ()
    ranks = []
    for index, selection in enumerate(best):
        if selection.score is None:
            name = f'{selection.split}/{selection.metric}'
            value = scores.get(name, float('nan'))
        else:
            name = f'aggregate:{index}'
            value = float(selection.score(MetricScores(declared, scores, objective)))
        mode = selection.mode or 'min'
        threshold = selection.threshold
        if threshold is not None and (value >= threshold if mode == 'min' else value <= threshold):
            value = float('nan')
        ranks.append(Ranking(name, value, mode, selection.top, selection.weights_only))
    return tuple(ranks)


@dataclasses.dataclass
class HeldBest:
    """The model state a run without Checkpoints ranks best, held in host memory for
    `restore_best`: the counters, variables and EMA (`MODEL_STATE`) of the state whose first
    ranking, the one a checkpointed run's best restores by, beats every earlier one, an earlier one
    winning a tie; one copy, replaced as the best moves, and no optimizer state."""

    score: float | None = None
    restore: Callable[[], Mapping[str, object]] | None = None

    def offer(self, ranking: Sequence[Ranking], state: TrainState) -> None:
        """Hold `state`'s model state where its first ranking beats the held one (`Ranking.score`)."""
        score = ranking[0].score if ranking else None
        if score is not None and (self.score is None or score < self.score):
            from dew.distributed.host import snapshot
            from dew.state import MODEL_STATE
            self.score, self.restore = score, snapshot({name: getattr(state, name) for name in MODEL_STATE})

    def state(self, final: TrainState) -> TrainState:
        """`final` with the held model state put back where it was placed."""
        if self.restore is None:
            raise ValueError("restore_best found no evaluation of this run that its ranking ranks; "
                             "no state is the best to return")
        return dataclasses.replace(final, **self.restore())
