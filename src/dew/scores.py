"""A run's scores by name: a summary of its splits' scores (`Summary`) and the recorded name each score object
a run ranks by names (`recorded_name`), which training's evaluation records them under and a checkpoint's
metrics are read back by (`dew.checkpoints.Metrics`), below both."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol

import numpy as np

from dew.objectives.base import Metric, Objective, Shown, TrainingScalar


class Suite(Protocol):
    """The metrics a split's evaluation scores (`dew.training.evaluation.EvalSuite`), as a summary reads
    them."""

    @property
    def metrics(self) -> Sequence[Metric]: ...


@dataclass(frozen=True)
class Summary:
    """One scalar a run forms from each evaluation's split scores by a declared weighting, a board's
    weighted index: `sum(weight * score) / sum(weight)` over `weights`, each score named as an evaluation
    records it (`val/accuracy`, `test/loss`) and each weight positive. It is formed once a step, where the
    step's scores are joined (`Trainer.fit`), and only at a step that scored every split it weighs, so a
    step a rarer suite skips forms none rather than mix in its stale scores. It is logged as
    `evaluation/summary/<name>`, which `Best` and `Plateau` rank by, `better` the way it improves."""

    name: str
    weights: Mapping[str, float]
    better: Literal["higher", "lower"]

    SPLIT: ClassVar[str] = "evaluation/summary"
    """The namespace a summary's score is recorded under, in a score's split's place:
    `evaluation/summary/<name>`. It holds a '/', which no split's name or metric's name can
    (`Evaluation.run`), so no split's score is named in it."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", dict(self.weights))
        if not self.name or "/" in self.name:
            raise ValueError(f"a summary is named without '/', and {self.name!r} is not")
        if not self.weights:
            raise ValueError(f"summary {self.name!r} weighs no score")
        for score, weight in self.weights.items():
            split, _, metric = score.rpartition("/")
            if not split or not metric:
                raise ValueError(f"summary {self.name!r} weighs {score!r}, which names no split/metric")
            if not (weight > 0 and np.isfinite(weight)):
                raise ValueError(f"summary {self.name!r} weighs {score!r} by {weight}, which is not positive")
        if self.better not in ("higher", "lower"):
            raise ValueError(f"summary {self.name!r} improves 'higher' or 'lower', not {self.better!r}")

    @property
    def shown(self) -> Shown:
        """Its direction, as a metric declares its own (`Best` reads it)."""
        return Shown(better=self.better)

    def formed(self, scores: Mapping[str, float]) -> dict[str, float]:
        """`{evaluation/summary/<name>: value}` from one step's `scores`, where they hold every score it
        weighs, as a step that scored every split it weighs does (`check`), and nothing where not."""
        if not all(score in scores for score in self.weights):
            return {}
        total = sum(self.weights.values())
        value = sum(weight * float(scores[score]) for score, weight in self.weights.items()) / total
        return {f"{self.SPLIT}/{self.name}": value}

    def check(self, suites: Mapping[str, Suite], loss: bool) -> None:
        """Refuse a score no evaluation records: one of a split `suites` does not hold, or a metric its
        suite does not score, its loss where the objective defines none (`loss`)."""
        for score in self.weights:
            split, _, metric = score.rpartition("/")
            if split not in suites:
                raise ValueError(f"summary {self.name!r} weighs {score!r}, and the run validates no split "
                                 f"{split!r} ({sorted(suites)})")
            declared = {scored.name for scored in suites[split].metrics}
            if metric not in declared and not (metric == "loss" and loss):
                raise ValueError(f"summary {self.name!r} weighs {score!r}, which split {split!r} does not "
                                 f"score ({sorted(declared | ({'loss'} if loss else set()))})")


def recorded_name[S, A](key, names: Collection[str], owner: Objective[S, A] | None = None) -> str:
    """The recorded name a run's score object names, one rule for the scores `best=` and `stop=` read
    live and the ones a checkpoint keeps: a `TrainingScalar` the training split's (of the objective
    `owner` where it is given, by identity), a `Metric` the validation split's ('val'), a `Summary` its
    namespace's (`Summary.SPLIT`), and a `(split, metric)` pair that split's. A metric its split did not
    record raises with the splits that did."""
    split = Summary.SPLIT if isinstance(key, Summary) else "val"
    if isinstance(key, tuple) and len(key) == 2:
        split, key = key
    if isinstance(key, TrainingScalar):
        if owner is not None and key.owner is not owner:
            raise KeyError("training score belongs to another objective")
        return f"train/{key.name}"
    if isinstance(key, (Metric, Summary)):
        if f"{split}/{key.name}" in names:
            return f"{split}/{key.name}"
        splits = sorted(name.rpartition("/")[0] for name in names if name.rpartition("/")[2] == key.name)
        raise KeyError(f"metric {key.name!r} has no {split} score" + (
            f"; it is recorded under {', '.join(splits)}: ask (split, metric)" if splits else ""))
    raise KeyError(key)
