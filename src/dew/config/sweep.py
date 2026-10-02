"""Search hyperparameters over the ordinary training path.

`RunConfig.sweep` overrides a config through its own record, hands each trial to
the recipe's train entry point, and records the score that entry point
returns. Trials land in a JSON ledger before they are reported, so an
interrupted sweep resumes at the trial it stopped on instead of retraining
the finished ones. `random_search` and `grid_search` need nothing beyond
numpy; `optuna_search` asks Optuna's sampler for the next point and tells it
the ledger's trials, and needs `dewml[hpo]`.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from dew.telemetry.records import TrialFinished, json_value

if TYPE_CHECKING:
    from dew.config import RunConfig

type Choice = None | bool | int | float | str
"""One candidate value for a swept field: what JSON and Optuna both hold."""

type Space = Mapping[str, Sequence[Choice]]
"""Dotted paths into a run record, each with the values a trial draws from."""

type Point = dict[str, Choice]


class Search(Protocol):
    """Chooses the next point of a space, given the trials already finished.

    `seed` is read by the samplers that draw at random, `random_search` and
    `optuna_search`; `grid_search` walks the product in order and ignores it.
    """

    def __call__(self, space: Space, finished: Sequence[TrialFinished], seed: int) -> Point: ...


def override[C: RunConfig](config: C, point: Point) -> C:
    """Return `config` with each dotted path in `point` replaced, through its record.

    `RunConfig.from_dict` refuses a leaf the class does not declare, so a
    misspelled path raises instead of training the unchanged config.
    """
    record = config.to_dict()
    for path, value in point.items():
        *groups, field = path.split('.')
        node = record
        for name in groups:
            child = node.get(name)
            if not isinstance(child, dict):
                raise KeyError(f'{path} names no group of the run record')
            node = child
        node[field] = value
    return type(config).from_dict(record)


def random_search(space: Space, finished: Sequence[TrialFinished], seed: int) -> Point:
    """Draw one independent value per field, reproducible from the trial's number."""
    rng = np.random.default_rng([seed, len(finished)])
    return {path: values[int(rng.integers(len(values)))] for path, values in space.items()}


def grid_search(space: Space, finished: Sequence[TrialFinished], seed: int) -> Point:
    """Return the next point of the space's cartesian product, in order.

    `seed` is the `Search` protocol's, and this walk draws nothing, so it goes
    unread here.
    """
    points = list(itertools.product(*space.values()))
    if len(finished) >= len(points):
        raise ValueError(f'the grid holds {len(points)} points and trial '
                         f'{len(finished)} was asked for; lower the budget')
    return dict(zip(space, points[len(finished)], strict=True))


def optuna_search(space: Space, finished: Sequence[TrialFinished], seed: int) -> Point:
    """Ask Optuna's sampler for the next point over the same space.

    The study is built from the ledger on every call rather than kept across
    them, so a resumed sweep asks from the same trials a fresh one would.
    """
    import optuna

    distributions: dict[str, optuna.distributions.BaseDistribution] = {
        path: optuna.distributions.CategoricalDistribution(list(values))
        for path, values in space.items()}
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    for trial in finished:
        study.add_trial(optuna.trial.create_trial(
            params=dict(trial.overrides), distributions=distributions, value=trial.value))
    return dict(study.ask(distributions).params)


def _recorded(space: Space) -> dict[str, list[Choice]]:
    """Return the space as the ledger holds it. JSON has no tuple."""
    return {field: list(values) for field, values in space.items()}


def _read(path: Path, space: Space) -> list[TrialFinished]:
    """Read the ledger's finished trials, refusing a ledger of another space."""
    if not path.exists():
        return []
    ledger = json.loads(path.read_text())
    if ledger['space'] != _recorded(space):
        raise ValueError(f'{path} holds a sweep over {sorted(ledger["space"])}, not this space; '
                         f'a resumed sweep continues the space it started')
    return [TrialFinished(int(row['index']), str(row['name']), dict(row['overrides']),
                          float(row['value'])) for row in ledger['trials']]


def _write(path: Path, space: Space, trials: Sequence[TrialFinished]) -> None:
    """Replace the ledger with `trials`, through a temporary file so an
    interrupt cannot leave half a ledger behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps({'space': _recorded(space),
                                   'trials': [json_value(trial) for trial in trials]}, indent=2))
    partial.replace(path)


__all__ = ["Choice", "Point", "Search", "Space", "grid_search", "optuna_search", "random_search"]
