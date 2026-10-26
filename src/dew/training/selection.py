"""Evaluation policies shared by fit and recorded run configurations."""
from __future__ import annotations

import dataclasses
import math
import types
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal

from dew.objectives.base import Metric, Shown, TrainingScalar

if TYPE_CHECKING:
    from dew.checkpoints import Ranking
    from dew.state import TrainState

from dew.training.evaluation import Summary


@dataclasses.dataclass(frozen=True, init=False)
class Best:
    """Keeps the `top` best evaluations by a metric, a recorded metric name, or a custom score.

    A name refers to one of the metric objects passed to `fit`, or to a `Summary` as
    `evaluation/summary/<name>`. `mode` sets the direction, 'min' or 'max'. When it is None, a custom
    score is minimized, and a named metric or a metric object takes its direction
    from its `Shown` declaration. An evaluation whose score is not better
    than `threshold` is not ranked.
    """
    metric: str
    top: int = 1
    mode: Literal['min', 'max'] | None = None
    threshold: float | None = None
    weights_only: bool = False
    split: str | None = None
    _source: Metric | TrainingScalar | Summary | Callable[[Mapping], float] | None = dataclasses.field(
        default=None, repr=False, compare=False, metadata={'record': False})

    def __init__(self, metric: str | Metric | TrainingScalar | Summary | types.MethodType
                 | Callable[[Mapping], float], top: int = 1, mode: Literal['min', 'max'] | None = None,
                 threshold: float | None = None, weights_only: bool = False, split: str | None = None,
                 *, _source: Metric | TrainingScalar | Summary | Callable[[Mapping], float] | None = None):
        if top < 1 or mode not in (None, 'min', 'max'):
            raise ValueError("Best needs integer top >= 1 and mode min or max")
        if threshold is not None and not math.isfinite(threshold):
            raise ValueError("Best.threshold must be finite")
        name = (
            metric
            if isinstance(metric, str)
            else metric.name
            if isinstance(metric, (Metric, TrainingScalar, Summary))
            else "<aggregate>"
        )
        source = _source if _source is not None else None if isinstance(metric, str) else metric
        for field, value in (('metric', name), ('top', top), ('mode', mode), ('threshold', threshold),
                             ('weights_only', weights_only), ('split', split), ('_source', source)):
            object.__setattr__(self, field, value)

    def resolved(self, scored: Mapping[str, Sequence[Metric | Summary]]) -> Best:
        """This selection bound to the split it ranks and the metric it names there: `train` for a
        training scalar, else the split of `scored` (each split's metrics) it names, by `split` or as
        its recorded name's prefix, or the only one, and the metric it names among that split's own,
        so each suite's instance of one metric is its own. Its direction, where it names none, is how
        that metric itself declares it is shown."""
        selection = self
        metric = selection._source
        if isinstance(metric, TrainingScalar):
            if selection.mode is None:
                if metric.shown.better is None:
                    raise ValueError("training scalar has no declared direction; use Best(scalar, mode=...)")
                selection = dataclasses.replace(
                    selection, mode="max" if metric.shown.better == "higher" else "min"
                )
            return dataclasses.replace(selection, split='train')
        # A summary is ranked in its namespace, which no split can be named, so no other selection's
        # split defaults to it. A recorded name's split is all before its last '/', as a namespace is.
        splits = tuple(split for split in scored if split != Summary.SPLIT)
        if isinstance(metric, Summary) and selection.split is None:
            selection = dataclasses.replace(selection, split=Summary.SPLIT)
        if metric is None and selection.split is None and selection.metric.rpartition('/')[0] in scored:
            selection = dataclasses.replace(selection, split=selection.metric.rpartition('/')[0])
        if selection.split is None:
            if len(splits) != 1 and (metric is None or isinstance(metric, Metric)):
                raise ValueError("Best(metric, split=...) is required with several validation splits")
            selection = dataclasses.replace(selection, split=splits[0] if len(splits) == 1 else None)
        elif selection.split not in scored:
            raise ValueError(f"unknown validation split {selection.split!r}")
        if metric is None:
            assert selection.split is not None
            name = selection.metric.removeprefix(f"{selection.split}/")
            metric = next((declared for declared in scored[selection.split] if declared.name == name), None)
            if metric is None:
                raise ValueError(f"best metric {selection.metric!r} is not among fit's metrics")
            selection = dataclasses.replace(selection, _source=metric)
        if isinstance(metric, (Metric, Summary)):
            assert selection.split is not None
            if not any(metric is declared for declared in scored[selection.split]):
                raise ValueError("best must be a metric object passed in metrics")
            declared = _declared_shown(metric)
            better = None if declared is None else declared.better
            if selection.mode is None:
                if better not in ('higher', 'lower'):
                    raise ValueError(
                        "best metric has no declared direction; use Best(metric, mode='min' or 'max')"
                    )
                selection = dataclasses.replace(selection, mode='max' if better == 'higher' else 'min')
        return selection


def _declared_shown(metric: Metric | Summary) -> Shown | None:
    """How `metric` itself declares it is shown, which the Metric protocol leaves optional, so it is
    read at this boundary."""
    return getattr(metric, "shown", None)


def rankings(best: Sequence[Best], scores: Mapping[str, float], *, validation: str | None,
             declared: Mapping[str, Sequence[Metric | Summary]], objective) -> tuple[Ranking, ...]:
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
        metric = selection._source
        assert metric is not None
        if isinstance(metric, (Metric, TrainingScalar, Summary)):
            name = f'{selection.split}/{metric.name}'
            value = scores.get(name, float('nan'))
        else:
            name = f'aggregate:{index}'
            value = float(metric(MetricScores(declared, scores, objective)))
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
            from dew.state import MODEL_STATE
            from dew.training.host import snapshot
            self.score, self.restore = score, snapshot({name: getattr(state, name) for name in MODEL_STATE})

    def state(self, final: TrainState) -> TrainState:
        """`final` with the held model state put back where it was placed."""
        if self.restore is None:
            raise ValueError("restore_best found no evaluation of this run that its ranking ranks; "
                             "no state is the best to return")
        return dataclasses.replace(final, **self.restore())
