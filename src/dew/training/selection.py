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

    def resolved(self, metrics: Sequence[Metric], splits: Sequence[str] = ("val",)) -> Best:
        """This selection bound to the metric it names among `metrics`, with its direction from that
        metric's `Shown` where it names none, and the split it ranks: `train` for a training scalar,
        else the one of `splits` it names or the only one."""
        selection = self
        metric = selection._source
        if metric is None:
            name = selection.metric.removeprefix((selection.split or 'val') + '/')
            metric = next((declared for declared in metrics if declared.name == name), None)
            if metric is None:
                raise ValueError(f"best metric {selection.metric!r} is not among fit's metrics")
            selection = dataclasses.replace(selection, _source=metric)
        if isinstance(metric, TrainingScalar):
            if selection.mode is None:
                if metric.shown.better is None:
                    raise ValueError("training scalar has no declared direction; use Best(scalar, mode=...)")
                selection = dataclasses.replace(
                    selection, mode="max" if metric.shown.better == "higher" else "min"
                )
            return dataclasses.replace(selection, split='train')
        if isinstance(metric, Metric):
            if not any(metric is declared for declared in metrics):
                raise ValueError("best must be a metric object passed in metrics")
            shown = getattr(metric, "shown", None)
            better = shown.better if isinstance(shown, Shown) else None
            if selection.mode is None:
                if better not in ('higher', 'lower'):
                    raise ValueError(
                        "best metric has no declared direction; use Best(metric, mode='min' or 'max')"
                    )
                selection = dataclasses.replace(selection, mode='max' if better == 'higher' else 'min')
        if selection.split is None:
            if len(splits) != 1 and isinstance(metric, Metric):
                raise ValueError("Best(metric, split=...) is required with several validation splits")
            selection = dataclasses.replace(selection, split=splits[0] if len(splits) == 1 else None)
        elif selection.split not in splits:
            raise ValueError(f"unknown validation split {selection.split!r}")
        return selection
