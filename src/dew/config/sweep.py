"""Hyperparameter search that trains each trial through the ordinary training path.

`RunConfig.sweep` overrides fields of a config through its record, passes
each trial to the recipe's train function, and records the score that
function returns. Each trial is written to a JSON ledger before it is
reported, so an interrupted sweep resumes at the trial it stopped on and
does not retrain the finished ones. `RandomSearch` and `GridSearch` need
only numpy. `OptunaSearch` gives Optuna's sampler the trials in the ledger
and asks it for the next point; it needs `dewml[hpo]`.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import tyro

from dew.records import JSON
from dew.registry import Annotation, _declared_type, models, objectives, parameters, wants_tuple
from dew.telemetry.records import TrialFinished, json_value

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from dew.config import RunConfig

type Choice = None | bool | int | float | str
"""One candidate value for a swept field, of a type both JSON and Optuna can hold."""

type Space = Mapping[str, Sequence[Choice]]
"""A mapping from dotted paths in a run record to the values a trial draws from there."""

type Point = dict[str, Choice]


class Search(Protocol):
    """Chooses the next point in a space, given the trials already finished."""

    def __call__(self, space: Space, finished: Sequence[TrialFinished]) -> Point: ...


def _within(node: dict[str, JSON], key: str) -> dict[str, JSON]:
    """The record `node` holds under `key`. A field declaring a base or a
    union records its value's class beside its fields, and this is the fields."""
    held = node[key]
    held = held["fields"] if isinstance(held, dict) and "class" in held else held
    if not isinstance(held, dict):
        raise KeyError(f"{key} holds {held!r}, not a record of fields")
    return held


def _placed(config: RunConfig, path: str) -> tuple[list[str], Annotation]:
    """The keys of `path`'s value in `config`'s record, and the annotation it is read by."""
    from dew.config import ModelConfig, ObjectiveConfig, _member_flags

    *groups, field = path.split(".")
    held: object = config
    for group in groups:
        if not (dataclasses.is_dataclass(held) and group in _names(held)):
            raise KeyError(f"{path} names no group of the run record")
        held = getattr(held, group)
    if isinstance(held, ModelConfig | ObjectiveConfig) and field not in _names(held):
        member = models[held.name] if isinstance(held, ModelConfig) else objectives[held.name]
        # A flag's annotation types the value, as `--model.<field>` reads it;
        # an argument without a flag (the loss a `Supervised` is given) takes JSON.
        annotation = _declared_type(_member_flags(member, held.fields)[0], field)
        if annotation is None and field not in parameters(member)[0]:
            raise KeyError(f"{path} names no argument of {held.name}")
        return [*groups, "fields", field], annotation
    if not (dataclasses.is_dataclass(held) and field in _names(held)):
        raise KeyError(f"{path} names no field of the run record")
    return [*groups, field], _declared_type(type(held), field)


def _names(held: DataclassInstance | type[DataclassInstance]) -> set[str]:
    """The fields a dataclass declares."""
    return {field.name for field in dataclasses.fields(held)}


def _parsed(path: str, text: str, annotation: Annotation) -> JSON:
    """`text` as the flag of `annotation` reads it (`RunConfig.assigned`), a
    tuple as the list its record holds."""
    from dew.config import _flag_json, _scalar

    if annotation is not None and _scalar(annotation):
        holder = dataclasses.make_dataclass("Set", [("value", tyro.conf.Positional[annotation])])
        given = shlex.split(text) if wants_tuple(annotation) else [text]
        return json.loads(json.dumps(tyro.cli(holder, args=given, prog=f"--set {path}").value))
    try:
        return _flag_json(text)
    except json.JSONDecodeError:
        return text


@dataclasses.dataclass(frozen=True)
class RandomSearch(Search):
    """Draws one value per field independently, reproducibly from `seed` and the trial's number."""

    seed: int = 0

    def __call__(self, space: Space, finished: Sequence[TrialFinished]) -> Point:
        rng = np.random.default_rng([self.seed, len(finished)])
        return {path: values[int(rng.integers(len(values)))] for path, values in space.items()}


@dataclasses.dataclass(frozen=True)
class GridSearch(Search):
    """Walks the space's cartesian product in order. Asking for more trials
    than the grid has points raises `ValueError`."""

    def __call__(self, space: Space, finished: Sequence[TrialFinished]) -> Point:
        points = list(itertools.product(*space.values()))
        if len(finished) >= len(points):
            raise ValueError(f'the grid holds {len(points)} points and trial '
                             f'{len(finished)} was asked for; lower the budget')
        return dict(zip(space, points[len(finished)], strict=True))


@dataclasses.dataclass(frozen=True)
class OptunaSearch(Search):
    """Asks Optuna's TPE sampler for the next point in the space.

    Each call builds a new study from the finished trials, seeded with
    `seed`, so a resumed sweep asks from the same trials as a sweep that
    never stopped.
    """

    seed: int = 0

    def __call__(self, space: Space, finished: Sequence[TrialFinished]) -> Point:
        import optuna

        distributions: dict[str, optuna.distributions.BaseDistribution] = {
            path: optuna.distributions.CategoricalDistribution(list(values))
            for path, values in space.items()}
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=self.seed))
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


__all__ = ["Choice", "GridSearch", "OptunaSearch", "Point", "RandomSearch", "Search", "Space"]
