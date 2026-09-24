"""Host-only run records carried by Tracker.artifact, not training pytrees.

`json_value` here runs the opposite direction from `dew.records`: that module
narrows a config field a published file handed over, this one encodes a
finished record on its way out to a tracker.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import platform
import traceback
import typing
from collections.abc import Mapping, Sequence
from typing import Literal

from dew.records import JSON


def json_value(value: object) -> JSON:
    """JSON values, with nonfinite numbers encoded as 'NaN', '+Inf', '-Inf'.

    These strings preserve valid infinite metrics (for example perfect PSNR)
    without emitting invalid JSON numbers. No arbitrary objects are stringified.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else ('NaN' if math.isnan(value) else
                                                   '+Inf' if value > 0 else '-Inf')
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_value(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        encoded: dict[str, JSON] = {}
        for key, entry in value.items():
            if not isinstance(key, str):
                raise TypeError('report metadata keys must be strings')
            encoded[key] = json_value(entry)
        return encoded
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [json_value(entry) for entry in value]
    raise TypeError(f'{type(value).__name__} is not JSON reporting metadata')


def packages_installed() -> dict[str, str]:
    versions = {'python': platform.python_version()}
    for name in ('dew-ml', 'jax', 'jaxlib', 'flax', 'optax', 'orbax-checkpoint', 'grain', 'numpy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


@dataclasses.dataclass(frozen=True)
class RunRecord:
    name: str
    config: JSON
    summary: JSON
    steps: int
    packages: Mapping[str, str]


@dataclasses.dataclass(frozen=True)
class FitStarted:
    start_step: int
    target_steps: int
    resumed_from: str | None
    parameters: int
    devices: int
    device_kind: str
    processes: int
    mesh: Mapping[str, int]


@dataclasses.dataclass(frozen=True)
class StepCompiled:
    """A training step compiled for a new batch shape, and the remat it
    compiled under, which is the model's own or a stronger rung the trainer
    moved to because the step did not fit."""
    seconds: float
    remat: JSON


@dataclasses.dataclass(frozen=True)
class CheckpointRequested:
    """Async save submitted, NOT proof of a durable checkpoint."""
    directory: str
    local: bool = False


@dataclasses.dataclass(frozen=True)
class ProfileWindow:
    directory: str
    steps: int


@dataclasses.dataclass(frozen=True)
class FitEnded:
    status: Literal['completed', 'failed', 'interrupted']
    seconds: float
    error: str | None = None
    traceback: str | None = None

    @classmethod
    def outcome(cls, seconds: float, error: BaseException | None) -> FitEnded:
        if error is None:
            return cls('completed', seconds)
        return cls('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', seconds,
                   f'{type(error).__name__}: {error}', ''.join(traceback.format_exception(error)))


@dataclasses.dataclass(frozen=True)
class TrialFinished:
    """One finished sweep trial: the search point it trained, the run name it
    trained under and the score the sweep recorded for it."""
    index: int
    name: str
    overrides: Mapping[str, JSON]
    value: float


type Record = (RunRecord | FitStarted | StepCompiled | CheckpointRequested | ProfileWindow | FitEnded
               | TrialFinished)
# A PEP 695 alias holds its union in `__value__`; `get_args` of the alias
# itself is empty, and these are the classes `isinstance` is given.
RECORD_TYPES = typing.get_args(Record.__value__)
