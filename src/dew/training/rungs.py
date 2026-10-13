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

import functools
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import numpy as np
from jax.sharding import Mesh

from dew.files import write_atomically
from dew.nn.kernels.generation import gpu_target
from dew.nn.sharding import trace_context
from dew.records import JSON, json_value, record
from dew.telemetry.instrumentation import StepFlops, count_compilation_cache, step_flops

if TYPE_CHECKING:
    from dew.objectives.base import Batch
    from dew.training.state import TrainState
    from dew.training.trainer import StepFn


def rung_records() -> Path | None:
    """Where a step's rung is recorded: beside the persistent compilation
    cache, or nowhere when the process keeps none."""
    cache = jax.config.jax_compilation_cache_dir
    return None if not cache else Path(cache) / 'dew-rungs'


RECORD_VERSION = 2
"""The identity a record names (`rung_key`); a record of another version is another run's."""


def rung_key(modules: Sequence[object], arguments: tuple[TrainState, Batch], splits: Sequence[int],
             mesh: Mesh, step: Callable[..., StepFn] | None = None) -> dict[str, JSON]:
    """What a recorded rung was decided for, read before the step traces: the
    configuration of the `modules` its programs run at the rung it starts from
    (`Objective.program_key`) and a custom `step`'s name, the code that builds
    them (`_code`), the step's `arguments` (each leaf's path, shape, dtype and
    placement), the `splits` its plan chooses among (one where the run pins it),
    the mesh and its devices' kind and allocator limit, the kernels their
    generation runs (`gpu_target`), what a model reads from context as it traces
    (`trace_context`), the libraries that compile it and the allocator and
    XLA_FLAGS settings it compiles under."""
    devices = [device for device in mesh.devices.flat if device.process_index == jax.process_index()]
    limits = {f"{device.device_kind}, limit {(device.memory_stats() or {}).get('bytes_limit')}"
              for device in devices}
    devices_seen: list[JSON] = [*sorted(limits)]
    leaves = [f"{jax.tree_util.keystr(path)} {np.shape(leaf)} {leaf.dtype} "
              f"{leaf.sharding if isinstance(leaf, jax.Array) else None}"
              for path, leaf in jax.tree_util.tree_flatten_with_path(arguments)[0]]
    settings = sorted(os.environ.get('XLA_FLAGS', '').split())
    allocator = sorted(f"{name}={value}" for name, value in os.environ.items()
                       if name.startswith(('XLA_PYTHON_CLIENT_', 'TF_GPU_ALLOCATOR')))
    target = repr(gpu_target()) if jax.default_backend() == 'gpu' else jax.default_backend()
    named = step.__qualname__ if isinstance(step, types.FunctionType) else type(step).__qualname__
    # A module's configuration as its repr reads it, a function field by its name: the address
    # Python prints beside one is this process's alone.
    configured = [re.sub(r' at 0x[0-9a-f]+', '', repr(module)) for module in modules]
    program = repr([RECORD_VERSION, configured, named, _code(), leaves,
                    list(splits), dict(mesh.shape), target, trace_context(), jax.__version__,
                    importlib.metadata.version('jaxlib'), allocator])
    return {'program': hashlib.sha256(program.encode()).hexdigest(), 'devices': devices_seen,
            'xla_flags': ' '.join(settings)}


@functools.cache
def _code() -> str:
    """The digest of Dew's source files this process has imported: a plan
    recorded by other code is decided again."""
    digest = hashlib.sha256()
    for name in sorted(module for module in sys.modules if module == 'dew' or module.startswith('dew.')):
        path = vars(sys.modules[name]).get('__file__')
        if path is not None and path.endswith('.py'):
            digest.update(name.encode())
            digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def recorded_rung(modules: Sequence[object], arguments: tuple[TrainState, Batch], splits: Sequence[int],
                  mesh: Mesh, step: Callable[..., StepFn] | None = None,
                  ) -> tuple[Path | None, dict[str, JSON], JSON]:
    """Where the rung of the step over `arguments` on `mesh` is recorded, its key
    (`rung_key`), and the rung the record holds (None if there is none). A
    process of a pool keeps no record: its processes agree on the rung.

    A record whose contents name another key is refused by its path, as
    a stale or foreign file; deleting it lets the run decide again."""
    directory = rung_records()
    if directory is None or jax.process_count() > 1:
        return None, {}, None
    count_compilation_cache()  # whether a search compiled anything (`plan_flops`)
    key = rung_key(modules, arguments, splits, mesh, step)
    path = directory / f"{hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()}.json"
    if not path.exists():
        return path, key, None
    stored = record(json.loads(path.read_text()), f"rung record {path}")
    if stored.get('key') != key:
        raise ValueError(f"the rung record {path} was written for another step or device "
                         f"({stored.get('key')!r}, not {key!r}); delete it to decide the rung again")
    return path, key, json_value(stored.get('rung'), f"rung record {path}")


FACTS = ('flops', 'launches')
"""What a record holds of its plan's program besides the plan: a
microbatch's FLOPs (`plan_flops`) and its kernel launches (`tune`)."""


def recorded_fact(recorded: JSON, plan: Mapping[str, JSON], name: str) -> object:
    """The record's `name` (`FACTS`) of the program `plan` compiles, where
    `plan` is the `recorded` one, else None."""
    kept = dict(record(recorded, 'recorded plan')) if recorded is not None else {}
    facts = {fact: kept.pop(fact, None) for fact in FACTS}
    return facts[name] if kept == plan else None


def plan_flops(recorded: JSON, chosen: Mapping[str, JSON], compiled: bool,
               executable: jax.stages.Compiled) -> StepFlops | None:
    """A microbatch's FLOPs in the `chosen` plan: those its record holds, where
    the plan is the `recorded` one and the search `compiled` nothing (every
    program came from the persistent cache), else counted from its
    `executable` (`step_flops`), which reads the module's text."""
    counted = recorded_fact(recorded, chosen, 'flops')
    if not compiled and isinstance(counted, list):
        return StepFlops(*counted)
    return step_flops(executable)


def keep_rung(path: Path, key: Mapping[str, JSON], rung: JSON) -> None:
    """Record `rung` for `key` at `path`, whole or not at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, json.dumps({'key': dict(key), 'rung': rung}, sort_keys=True))
