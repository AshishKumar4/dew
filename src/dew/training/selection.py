"""Evaluation policies shared by fit and recorded run configurations."""
from __future__ import annotations

import dataclasses
import math
import types
from collections.abc import Callable, Mapping
from typing import Literal

from dew.objectives.base import Metric, TrainingScalar


@dataclasses.dataclass(frozen=True, init=False)
class Best:
    """Keep top-K evaluations of a metric, a recorded metric name, or a custom score.

    Names bind to fit's metric objects. A custom score minimizes by default;
    named and object metrics take direction from their Shown declaration.
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
        name = metric if isinstance(metric, str) else metric.name if isinstance(metric, (Metric, TrainingScalar)) else '<aggregate>'
        source = _source if _source is not None else None if isinstance(metric, str) else metric
        for field, value in (('metric', name), ('top', top), ('mode', mode), ('threshold', threshold),
                             ('weights_only', weights_only), ('split', split), ('_source', source)):
            object.__setattr__(self, field, value)
