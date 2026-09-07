"""Host-only run records carried by Tracker.artifact, not training pytrees."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import platform
import traceback
from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

JSON: TypeAlias = None | bool | int | float | str | list['JSON'] | dict[str, 'JSON']


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
        result: dict[str, JSON] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError('report metadata keys must be strings')
            result[key] = json_value(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [json_value(item) for item in value]
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


Record: TypeAlias = RunRecord | FitStarted | CheckpointRequested | ProfileWindow | FitEnded
RECORD_TYPES = (RunRecord, FitStarted, CheckpointRequested, ProfileWindow, FitEnded)
