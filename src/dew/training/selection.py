"""Evaluation policies shared by fit and recorded run configurations."""
from __future__ import annotations

import dataclasses
import math
import types
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from dew.objectives.base import Metric, Shown, TrainingScalar


@dataclasses.dataclass(frozen=True, init=False)
class Best:
    """Keeps the `top` best evaluations by a metric, a recorded metric name, or a custom score.

    A name refers to one of the metric objects passed to `fit`. `mode` sets
    the direction, 'min' or 'max'. When it is None, a custom score is
    minimized, and a named metric or a metric object takes its direction
    from its `Shown` declaration. An evaluation whose score is not better
    than `threshold` is not ranked.
    """
    metric: str
    top: int = 1
    mode: Literal['min', 'max'] | None = None
    threshold: float | None = None
    weights_only: bool = False
    split: str | None = None
    _source: Metric | TrainingScalar | Callable[[Mapping], float] | None = dataclasses.field(
        default=None, repr=False, compare=False, metadata={'record': False})

    def __init__(self, metric: str | Metric | TrainingScalar | types.MethodType | Callable[[Mapping], float],
                 top: int = 1, mode: Literal['min', 'max'] | None = None,
                 threshold: float | None = None, weights_only: bool = False, split: str | None = None,
                 *, _source: Metric | TrainingScalar | Callable[[Mapping], float] | None = None):
        if top < 1 or mode not in (None, 'min', 'max'):
            raise ValueError("Best needs integer top >= 1 and mode min or max")
        if threshold is not None and not math.isfinite(threshold):
            raise ValueError("Best.threshold must be finite")
        name = (
            metric
            if isinstance(metric, str)
            else metric.name
            if isinstance(metric, (Metric, TrainingScalar))
            else "<aggregate>"
        )
        source = _source if _source is not None else None if isinstance(metric, str) else metric
        for field, value in (('metric', name), ('top', top), ('mode', mode), ('threshold', threshold),
                             ('weights_only', weights_only), ('split', split), ('_source', source)):
            object.__setattr__(self, field, value)

    def resolved(self, scored: Mapping[str, Sequence[Metric]]) -> Best:
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
        splits = tuple(scored)
        if metric is None and selection.split is None and selection.metric.partition('/')[0] in scored:
            selection = dataclasses.replace(selection, split=selection.metric.partition('/')[0])
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
        if isinstance(metric, Metric):
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


def _declared_shown(metric: Metric) -> Shown | None:
    """How `metric` itself declares it is shown, which the Metric protocol leaves optional, so it is
    read at this boundary."""
    return getattr(metric, "shown", None)
