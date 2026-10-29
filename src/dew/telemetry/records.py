"""Host-only run records carried by Tracker.artifact, not training pytrees.

`json_value` here runs the opposite direction from `dew._records`: that module
narrows a config field a published file handed over, this one encodes a
finished record on its way out to a tracker.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import logging
import math
import platform
import subprocess
import traceback
import typing
from collections.abc import Mapping, Sequence
from typing import Literal

from dew import _checkout
from dew._records import JSON
from dew.pool import PREEMPTED_EXIT

_log = logging.getLogger(__name__)


_NONFINITE = {'NaN': math.nan, '+Inf': math.inf, '-Inf': -math.inf}
"""The spellings `json_value` writes a nonfinite number in, and the number each reads back as."""


def reported_number(value: object, key: str) -> float:
    """A number a report wrote through `json_value`: a JSON number, or the spelling of a nonfinite
    one, read back as that float."""
    if isinstance(value, str) and value in _NONFINITE:
        return _NONFINITE[value]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key}={value!r}: a report's number is a JSON number, NaN, +Inf or -Inf")
    return float(value)


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
    for name in ('dewml', 'jax', 'jaxlib', 'flax', 'optax', 'orbax-checkpoint', 'grain', 'numpy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def checkout_revision() -> str | None:
    """The Dew checkout's revision, never the training driver's working directory.

    A wheel is not a checkout and records None. A worktree's .git is a file,
    so both files and directories are accepted. Git unavailable or unable
    to read this checkout leaves the revision unknown and reports why.
    """
    try:
        return _checkout.revision()
    except (OSError, subprocess.CalledProcessError) as error:
        _log.warning("Cannot read Dew's checkout revision: %s", error)
        return None


@dataclasses.dataclass(frozen=True)
class RunRecord:
    """The complete run config and the software and hardware that trained it.

    `dew_sha` belongs to the installed Dew source checkout. A driver's own
    revision can be carried separately in W&B's WANDB_TAGS or WANDB_NOTES.
    Hardware fields are filled by the config's training path; a record made
    without a training backend leaves them unknown.
    """
    name: str
    config: JSON
    summary: JSON
    steps: int | None
    packages: Mapping[str, str]
    dew_sha: str | None = dataclasses.field(default_factory=checkout_revision)
    device_kind: str | None = None
    devices: int | None = None
    processes: int | None = None


@dataclasses.dataclass(frozen=True)
class FitStarted:
    start_step: int
    target_steps: int | None
    resumed_from: str | None
    parameters: int
    devices: int
    device_kind: str
    processes: int
    mesh: Mapping[str, int]
    sharded: float
    """The share of the parameters' bytes a parameter axis (fsdp, expert,
    tensor) splits."""
    seed: int | None = None
    """None: no integer seed recorded (a JAX key was supplied, or the record predates the field)."""
    tokens: int | None = None
    cadences: Mapping[str, JSON] = dataclasses.field(default_factory=dict)
    """The run's cadences and schedule warmups as it resolved them, a share of the run among them
    (`dew.training.budget.every_step`, `dew.optim.OptimConfig.warmups`): steps, committed tokens or a
    duration by name."""
    groups: Mapping[str, str] | None = None
    """Where the optimizer's split sent each parameter and why, by its path (`dew.optim.OptimConfig.groups`:
    Muon's matrices beside AdamW's vectors, lookups and filters); None for an optimizer that splits none."""


@dataclasses.dataclass(frozen=True)
class AxisLink:
    """A mesh axis's link as a compiled step used it: the bandwidth its
    all-gather measured (None on a CPU mesh or a device the peak table does
    not name) and whether a projection ran split over the axis
    (`dew.nn.sharding.down_projection`, `split_positions`)."""
    bytes_per_second: float | None
    spread: bool


@dataclasses.dataclass(frozen=True)
class StepCompiled:
    """A training step compiled for a new batch shape, the remat it compiled
    under, which is the model's own or a stronger rung the trainer moved to
    because the step did not fit, the links of the tensor and sequence axes
    the mesh splits, by axis name, the arithmetic of the check that it
    fits this process's devices (`dew.training.memory.memory_record`), and
    its plan: the microbatches it reads each batch in, their rows, every
    compile the search made (`dew.training.memory.Probe`), and its kernels
    (`dew.training._lowering.KernelReport.record`)."""
    seconds: float
    remat: JSON
    links: Mapping[str, AxisLink]
    memory: JSON = None
    plan: JSON = None


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
    status: Literal['completed', 'preempted', 'failed', 'interrupted', 'lowered']
    seconds: float
    error: str | None = None
    traceback: str | None = None

    @property
    def exit_code(self) -> int:
        """Trackers record the same success or preemption a launcher sees."""
        return {'completed': 0, 'lowered': 0, 'preempted': PREEMPTED_EXIT,
                'failed': 1, 'interrupted': 1}[self.status]

    @property
    def terminal_status(self) -> Literal['FINISHED', 'KILLED', 'FAILED']:
        """A service's run state follows the same process exit semantics."""
        statuses: Mapping[int, Literal['FINISHED', 'KILLED', 'FAILED']] = {
            0: 'FINISHED', PREEMPTED_EXIT: 'KILLED', 1: 'FAILED'}
        return statuses[self.exit_code]

    @classmethod
    def outcome(cls, seconds: float, error: BaseException | None, *, preempted: bool = False,
                lowered: bool = False) -> FitEnded:
        """A run's end: its first step lowered for another device in place of
        training (`TrainerConfig.lower_for`), a failure, else stopped at a
        preemption notice with its checkpoint written, else completed. A
        successful SystemExit carries the same outcome as a return."""
        if lowered:
            return cls('lowered', seconds)
        if error is None or (isinstance(error, SystemExit) and error.code in (0, None)):
            return cls('preempted' if preempted else 'completed', seconds)
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
