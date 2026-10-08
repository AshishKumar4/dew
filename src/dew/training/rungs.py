"""The rung of the fit ladder an earlier run of a step chose, recorded
beside the persistent compilation cache so a later run starts there
(`Trainer.compile`), as a resumed run starts at its checkpoint's rung.

The fit check reads memory that identical runs can find apart: a process
that compiles the step autotunes it, and a pool that grows keeps the
regions the autotuner took (openxla/xla#50052), where a process that loads
the step from the cache does not. On an RTX 4080 the 99M MoE at 8 x 1024
tiled its head in the first (98.2 ms a step) and kept its whole logits in
the second (78.4 ms, docs/performance.md).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import jax
from jax.sharding import Mesh

from dew.files import write_atomically
from dew.records import JSON, json_value, record


def rung_records() -> Path | None:
    """Where a step's rung is recorded: beside the persistent compilation
    cache, or nowhere when the process keeps none."""
    cache = jax.config.jax_compilation_cache_dir
    return None if not cache else Path(cache) / 'dew-rungs'


def rung_key(program: jax.stages.Lowered, mesh: Mesh) -> dict[str, JSON]:
    """What a recorded rung was decided for: the step's program at the rung
    it starts from, the devices' kind and allocator limit, and XLA_FLAGS."""
    devices = [device for device in mesh.devices.flat if device.process_index == jax.process_index()]
    limits = {f"{device.device_kind}, limit {(device.memory_stats() or {}).get('bytes_limit')}"
              for device in devices}
    devices_seen: list[JSON] = [*sorted(limits)]
    return {'program': hashlib.sha256(program.as_text().encode()).hexdigest(), 'devices': devices_seen,
            'xla_flags': os.environ.get('XLA_FLAGS', '')}


def recorded_rung(program: jax.stages.Lowered, mesh: Mesh) -> tuple[Path | None, dict[str, JSON], JSON]:
    """Where the rung of `program` on `mesh` is recorded, its key
    (`rung_key`), and the rung the record holds (None if there is none). A
    process of a pool keeps no record: its processes agree on the rung.

    A record whose contents name another key is refused by its path, as
    a stale or foreign file; deleting it lets the run decide again."""
    directory = rung_records()
    if directory is None or jax.process_count() > 1:
        return None, {}, None
    key = rung_key(program, mesh)
    path = directory / f"{hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()}.json"
    if not path.exists():
        return path, key, None
    stored = record(json.loads(path.read_text()), f"rung record {path}")
    if stored.get('key') != key:
        raise ValueError(f"the rung record {path} was written for another step or device "
                         f"({stored.get('key')!r}, not {key!r}); delete it to decide the rung again")
    return path, key, json_value(stored.get('rung'), f"rung record {path}")


def keep_rung(path: Path, key: Mapping[str, JSON], rung: JSON) -> None:
    """Record `rung` for `key` at `path`, whole or not at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, json.dumps({'key': dict(key), 'rung': rung}, sort_keys=True))
