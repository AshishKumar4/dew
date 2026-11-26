"""The plan an earlier run of a step chose, its rung of the fit ladder and
the microbatches it read each batch in, recorded beside the persistent
compilation cache so a later run's search starts there (`Trainer.compile`),
as a resumed run starts at its checkpoint's.

The fit check reads memory that identical runs can find apart: a process
that compiles the step autotunes it, and a pool that grows keeps the
regions the autotuner took (openxla/xla#50052), where a process that loads
the step from the cache does not. On an RTX 4080 the 99M MoE at 8 x 1024
tiled its head in the first (98.2 ms a step) and kept its whole logits in
the second (78.4 ms, docs/performance.md).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import jax
from jax.sharding import Mesh

from dew import _cache
from dew._code import provenance, provenance_code, stated_class, states_its_code, whole_digest
from dew._devices import device_memory
from dew._records import JSON, json_value, record
from dew._stated import Unestablished, stated
from dew.communicators import loaded_program
from dew.nn.kernels.generation import assumption, gpu_target
from dew.nn.sharding import trace_context
from dew.telemetry.instrumentation import StepFlops, step_flops
from dew.training.ahead import computation_flags

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dew.batch import Batch
    from dew.state import TrainState
    from dew.training._candidates import StepSettings
    from dew.training.trainer import StepFn


def rung_records() -> Path | None:
    """Where a step's rung is recorded: DEW_RUNG_RECORDS when set, else beside the persistent compilation
    cache the process is configured with (`dew._cache.kept_beside`), or nowhere when it keeps none or keeps
    a bucket's."""
    return _cache.kept_beside('dew-rungs', variable='DEW_RUNG_RECORDS', configured=True)


RECORD_VERSION = 4
"""The identity a record names (`rung_key`); a record of another version is another run's."""


def rung_key(modules: Sequence[object], arguments: tuple[TrainState, Batch], splits: Sequence[int],
             mesh: Mesh, step: Callable[..., StepFn] | None = None, remat: str = "auto", *,
             settings: StepSettings) -> dict[str, JSON] | None:
    """What a recorded rung was decided for, read before the step traces: the
    configuration of the `modules` its programs run at the rung it starts from
    (`Objective.program_key`) and a custom `step`, as an identity states them
    (`dew._stated.stated`), the code that builds them (`_code`): the step's compiler, what the run's
    `settings` hold (its objective and optimizer among them) and those modules, each with every module of
    Dew's it can import, all of Dew's where any is a function, the step's `arguments` (their tree's
    structure and each leaf's abstract value and placement, stated alike), the `splits` its plan chooses
    among (one where the run pins it), the mesh and its devices' kind and allocator limit, the kernels their
    generation runs (`gpu_target`), what a model reads from context as it traces
    (`trace_context`), the libraries that compile it and the allocator and
    XLA_FLAGS settings it compiles under, and the recomputation the run's fit policy asks (`remat`,
    `FitPolicy.remat`), whose plans differ. None where an identity states none of the step (a module holding
    a class made inside a function, or an array this process holds only part of): such a run keeps no
    record and decides its rung as it compiles."""
    leaves, structure = jax.tree.flatten(arguments)
    configured = _configured(settings)
    if configured is None:
        _log.info("the step keeps no rung record, its rung decided as it compiles: its objective states no "
                  "trace key")
        return None
    try:
        traced = stated([list(modules), step, structure,
                         [[jax.typeof(leaf), leaf.sharding if isinstance(leaf, jax.Array) else None]
                          for leaf in leaves], trace_context(), configured],
                        "the step a rung record names")
    except Unestablished as unstated:
        _log.info("the step keeps no rung record, its rung decided as it compiles: %s", unstated)
        return None
    devices = [device for device in mesh.devices.flat if device.process_index == jax.process_index()]
    limits = {f"{device.device_kind}, limit {device_memory(device).get('bytes_limit')}"
              for device in devices}
    devices_seen: list[JSON] = [*sorted(limits)]
    # In their order, as XLA reads a flag given twice by its last value (row 10797: sorted, two orders of one
    # flag's two values were one key).
    flags = computation_flags()
    allocator = sorted(f"{name}={value}" for name, value in os.environ.items()
                       if name.startswith(('XLA_PYTHON_CLIENT_', 'TF_GPU_ALLOCATOR')))
    # The GPU its kernels are chosen for: this process's, or the one a CPU host builds for (`assumed`), so two
    # builds for two GPUs of a generation (their multiprocessors) never share a record.
    target = (repr(gpu_target()) if jax.default_backend() == 'gpu' or assumption() is not None
              else jax.default_backend())
    code = _code(settings, modules, step)
    program = json.dumps([RECORD_VERSION, traced, code, list(splits), dict(mesh.shape), target,
                          jax.__version__, importlib.metadata.version('jaxlib'), allocator, remat])
    return {'program': hashlib.sha256(program.encode()).hexdigest(), 'devices': devices_seen,
            'xla_flags': ' '.join(flags)}


def _configured(settings: StepSettings) -> Mapping[str, object] | None:
    """What a run's `settings` state of its step by value, beside the code `_code` states: each setting that
    is no function, mesh or module (its loss scale, whether it skips a step not finite, its inspection, its
    microbatches, its Newton-Schulz group...), its objective's trace key (`stated_trace`: every setting its
    loss reads, its head's chunks and tile among them, its untraced attributes left out by name) and the
    optimizer config it was built from. None where the objective's key states none (a class of a user's
    that states no key of its own): such a run keeps no record. Two runs of one shape and code whose steps
    compute otherwise by a setting name two records (the 2026-11-26 audit: a run skipping a step not
    finite, or scoring its head in other chunks, took another's rung). A raw optimizer's settings are
    functions, which only its code states."""
    from dew.objectives.base import stated_trace
    from dew.optim import OptimConfig

    trace = stated_trace(settings.objective)
    if trace is None:
        return None
    apart = ("objective", "built_optimizer", "device_mesh", "state_mesh", "step", "optimizer_source")
    held = {field.name: getattr(settings, field.name) for field in dataclasses.fields(settings)
            if field.name not in apart}
    source = settings.optimizer_source
    return {**held, "objective": trace, "optimizer": source if isinstance(source, OptimConfig) else None}


def _code(settings: StepSettings, modules: Sequence[object], step: Callable[..., StepFn] | None) -> list[str]:
    """The digests of the code that builds a step (`provenance_code`): the step's compiler and the modules of
    Dew's defining what its `settings` hold (its objective, every base of its class among them, a model-free
    objective's loss its base's, and what its optimizer was made from), its program's `modules` and a custom
    `step`, each with every module of Dew's it can import. All of Dew's where any of them is code no class of
    Dew's states, which may call any of Dew's: the objective's class none of Dew's (a user's subclass), a
    function (an optimizer handed in as a transformation, a user's criterion), an instance of a class of a
    user's or a library's (an optimizer spec or schedule of a user's own), or an optimizer an import path
    names outside Optax (`dew.optim.Optax`, its factory found by the registry, which no closure follows).
    Not every file of Dew's otherwise: a run keying a rung on them all read all 794, so its tests' CI key was
    every file (22% of the suite's rows)."""
    from dew.optim import Optax, OptimConfig

    objective = settings.objective
    # The optimizer as built is functions, what it was made from (`optimizer_source`) states its code; the
    # meshes are devices, which the key states apart (a mesh is callable, a context of JAX's).
    values = [*(getattr(settings, field.name) for field in dataclasses.fields(settings)
                if field.name not in ("built_optimizer", "device_mesh", "state_mesh")), *modules, step]
    pathed = [value.optimizer for value in values if isinstance(value, OptimConfig)]
    if not stated_class(type(objective)) or not states_its_code(values) or any(
            isinstance(optimizer, Optax) and not optimizer.path.startswith("optax:") for optimizer in pathed):
        return [whole_digest()]
    return provenance_code("dew.training.compiler", provenance([*values, *type(objective).__mro__]))


def recorded_rung(modules: Sequence[object], arguments: tuple[TrainState, Batch], splits: Sequence[int],
                  mesh: Mesh, step: Callable[..., StepFn] | None = None, remat: str = "auto", *,
                  settings: StepSettings) -> tuple[Path | None, dict[str, JSON], JSON]:
    """Where the rung of the step over `arguments` on `mesh` is recorded, its key
    (`rung_key`), and the rung the record holds (None if there is none). A
    process of a pool keeps no record: its processes agree on the rung; nor does a step its key states
    none of.

    A record whose contents name another key is refused by its path, as
    a stale or foreign file; deleting it lets the run decide again."""
    directory = rung_records()
    if directory is None or jax.process_count() > 1:
        return None, {}, None
    key = rung_key(modules, arguments, splits, mesh, step, remat, settings=settings)
    if key is None:
        return None, {}, None
    path = directory / f"{hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()}.json"
    found = _cache.read_kept(path)
    if found is None:
        return path, key, None
    stored = record(found, f"rung record {path}")
    if stored.get('key') != key:
        raise ValueError(f"the rung record {path} was written for another step or device "
                         f"({stored.get('key')!r}, not {key!r}); delete it to decide the rung again")
    return path, key, json_value(stored.get('rung'), f"rung record {path}")


def plan_flops(recorded: JSON, executable: jax.stages.Compiled) -> tuple[StepFlops | None, JSON]:
    """A microbatch's FLOPs in the plan `executable` runs, and what a record
    keeps of them: those the `recorded` plan's record holds where they were
    counted for this very executable (its fingerprint, which a
    persistent-cache hit keeps), else counted from it (`step_flops`), which
    reads the module's text. A key read before the step traces cannot tell
    apart two steps that differ in an objective's arguments alone; the
    executable can."""
    # An executable compiled for absent devices is not loaded and has no runtime fingerprint.
    program = loaded_program(executable)
    named = program.fingerprint.hex() if program is not None and program.fingerprint is not None else None
    counted = record(recorded, 'recorded plan').get('flops') if recorded is not None else None
    if named is not None and isinstance(counted, list) and len(counted) == 5 and counted[:1] == [named]:
        flops = StepFlops(*counted[1:])
    else:
        flops = step_flops(executable)
    return flops, None if flops is None else [named, flops.model, flops.hardware, flops.unsized,
                                              flops.bounded]


def keep_rung(path: Path, key: Mapping[str, JSON], rung: JSON) -> None:
    """Record `rung` for `key` at `path`, whole or not at all; a directory that cannot be written keeps
    none, the miss logged (`dew._cache.keep`), and a later run decides its rung again."""
    _cache.keep(path, {'key': dict(key), 'rung': rung}, "a step's rung")
