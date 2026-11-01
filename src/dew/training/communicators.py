"""The memory NCCL's communicators take outside XLA's pool on this host's GPUs, measured before the backend
opens, and the pool share that leaves it (ledger 2901).

XLA's pool takes a fixed share of each GPU as its backend opens (XLA_PYTHON_CLIENT_MEM_FRACTION). NCCL
allocates each communicator's buffers outside it, and nothing draws them from the pool: `ncclConfig_t` carries
no allocator, and XLA passes only its blocking mode, split sharing and channel caps
(xla/backends/gpu/collectives/nccl_collectives.cc, `AsNcclConfig`). XLA keys a communicator by its devices and
its kind, a collective's (communication id 0: all-gather, all-reduce, reduce-scatter, all-to-all) apart from a
permute's (id 1: collective-permute, send, recv; xla/backends/gpu/collectives/gpu_clique_key.h), so a step
that holds both makes two over the same GPUs.

`import dew` has NCCL connect every peer as it creates a communicator (NCCL_RUNTIME_CONNECT=0,
`dew.devices.connect_at_creation`), so the buffers are all allocated there, where XLA returns a failure to
every rank as an error. Connected lazily, as NCCL does by default since 2.22, a rank that fails to allocate at
a collective's first launch leaves its peer waiting inside XLA's first-call rendezvous, which aborts the
process after 40 seconds (xla/service/rendezvous.cc, LOG(FATAL)): a 2xH100 LoRA run, whose first all-to-all
one GPU entered and the other never did.

Before the backend opens, a child process on this host's GPUs (`main`) creates a communicator of each kind
over all of them and reports the bytes each GPU's driver saw go: NCCL's, and the runtime's own beside its
pool. The answer is kept in Dew's cache per configuration (`configuration`: the GPUs, the driver, NCCL's
version and settings), so a host pays for it once. Dew's share of each GPU then leaves that room
(`pool_share`); a share the user set too high for it is refused, naming the bytes. Once the step is compiled,
its collectives' kinds are read from its program and held to the room its GPUs have outside the pool
(`checked_room`), so a step that would fail to connect is refused before it runs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from dew.cache import dew_cache_dir
from dew.devices import cuda_driver_version, gpu_count, gpu_memory, gpu_name, nccl_version

_log = logging.getLogger(__name__)

KINDS = {0: "collective", 1: "permute"}
"""The communicators XLA keys apart over the same GPUs, by communication id."""

PERMUTES = re.compile(r"\b(collective-permute|send|recv)(-start)?\(")
COLLECTIVES = re.compile(r"\b(all-gather|all-reduce|reduce-scatter|all-to-all|ragged-all-to-all)(-start)?\(")


@dataclasses.dataclass(frozen=True)
class Needs:
    """What a child measured on each GPU, by ordinal: each kind's communicator's bytes (`KINDS`), the
    runtime's own beside its pool (cuBLAS's workspaces, the kernels it loads), and the seconds the measurement
    took."""

    communicators: Mapping[int, Mapping[int, int]]
    runtime: Mapping[int, int]
    seconds: float

    def to_json(self) -> dict:
        return {"communicators": {str(kind): {str(gpu): taken for gpu, taken in gpus.items()}
                                  for kind, gpus in self.communicators.items()},
                "runtime": {str(gpu): taken for gpu, taken in self.runtime.items()}, "seconds": self.seconds}

    @classmethod
    def of(cls, written: Mapping) -> Needs:
        communicators = {int(kind): {int(gpu): int(taken) for gpu, taken in gpus.items()}
                         for kind, gpus in written["communicators"].items()}
        runtime = {int(gpu): int(taken) for gpu, taken in written["runtime"].items()}
        return cls(communicators, runtime, float(written["seconds"]))

    def needed(self, kinds: frozenset[int], *, runtime: bool) -> dict[int, int]:
        """Each GPU's bytes outside the pool for communicators of `kinds`, with the runtime's where `runtime`.
        """
        gpus = set(self.runtime) | {gpu for taken in self.communicators.values() for gpu in taken}
        return {gpu: sum(self.communicators.get(kind, {}).get(gpu, 0) for kind in kinds)
                + (self.runtime.get(gpu, 0) if runtime else 0) for gpu in sorted(gpus)}


class CommunicatorRoom(ValueError):
    """A pool share, or a compiled step, that leaves a GPU less room outside XLA's pool than its communicators
    take: `needed` and `room` in bytes, by GPU ordinal."""

    def __init__(self, message: str, needed: Mapping[int, int], room: Mapping[int, int]):
        self.needed, self.room = dict(needed), dict(room)
        short = {gpu: needed[gpu] - room[gpu] for gpu in needed if needed[gpu] > room.get(gpu, 0)}
        super().__init__(f"{message}: each GPU needs {self.needed} bytes outside XLA's pool and has "
                         f"{self.room}, short by {short}")


def configuration() -> dict:
    """What a measurement holds for: this host's GPUs by name, the driver's CUDA version, NCCL's version and
    every NCCL setting in the environment."""
    count = gpu_count()
    return {"gpus": [gpu_name(ordinal) for ordinal in range(count)], "driver": cuda_driver_version(),
            "nccl": nccl_version(), "nccl_env": {name: value for name, value in sorted(os.environ.items())
                                                 if name.startswith("NCCL_")}}


def measured(configured: Mapping, *, measure: Callable[[Mapping], Needs]) -> Needs:
    """The measurement for `configured`, from Dew's cache, or `measure`d and kept there."""
    digest = hashlib.sha256(json.dumps(configured, sort_keys=True).encode()).hexdigest()[:24]
    path = Path(dew_cache_dir()) / "communicators" / f"{digest}.json"
    if path.exists():
        return Needs.of(json.loads(path.read_text())["needs"])
    needs = measure(configured)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = path.with_suffix(f".{os.getpid()}.tmp")
    written.write_text(json.dumps({"configuration": dict(configured), "needs": needs.to_json()}, indent=1))
    written.replace(path)
    _log.info("measured NCCL's communicators on this host's GPUs in %.1f s (kept in %s): %s", needs.seconds,
              path, needs.to_json())
    return needs


def needs_of(gpus: int | None = None) -> Needs | None:
    """This host's measurement (`measured`), measured by a child process where the cache holds none; None on a
    host of fewer than two GPUs, where no communicator is made."""
    if (gpu_count() if gpus is None else gpus) < 2:
        return None
    return measured(configuration(), measure=_child)


def _child(configured: Mapping) -> Needs:
    """The measurement a child process of this interpreter makes (`main`), its environment the parent's."""
    began = time.perf_counter()
    environment = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    done = subprocess.run([sys.executable, "-m", "dew.training.communicators"], capture_output=True,
                          text=True, env=environment, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"measuring NCCL's communicators on this host's GPUs failed (exit "
                           f"{done.returncode}), so no pool share can be sized to leave them room:\n"
                           f"{done.stderr[-4000:]}")
    written = json.loads(done.stdout.strip().splitlines()[-1])
    return Needs.of({**written, "seconds": time.perf_counter() - began})


def pool_share(needs: Needs, memory: Mapping[int, tuple[int, int]], requested: float | None,
               kinds: frozenset[int]) -> float:
    """The share of each GPU XLA's pool takes: `requested` (the user's XLA_PYTHON_CLIENT_MEM_FRACTION), held
    to leave each GPU its communicators of `kinds` and the runtime's room outside it, else the largest share
    that leaves that room, to a thousandth, none above Dew's 0.95. `memory` is each GPU's free and total bytes
    before the backend opens, every process's allocations counted."""
    needed = needs.needed(kinds, runtime=True)
    if requested is not None:
        room = {gpu: free - int(requested * total) for gpu, (free, total) in memory.items()}
        if any(needed.get(gpu, 0) > room[gpu] for gpu in room):
            raise CommunicatorRoom(f"XLA_PYTHON_CLIENT_MEM_FRACTION={requested} leaves too little of a GPU "
                                   "for NCCL's communicators and the runtime (measured on this host, "
                                   "dew.training.communicators)", needed, room)
        return requested
    widest = min((free - needed.get(gpu, 0)) / total for gpu, (free, total) in memory.items())
    return min(0.95, int(widest * 1000) / 1000)


def kinds_of(program: str) -> frozenset[int]:
    """The communicator kinds (`KINDS`) a compiled program's collectives make, read from its text."""
    return frozenset(kind for kind, pattern in ((0, COLLECTIVES), (1, PERMUTES)) if pattern.search(program))


def checked_room(outside: Mapping[int, int | None], needs: Needs, kinds: frozenset[int]) -> None:
    """Refuse a compiled step whose communicators of `kinds` need more than its GPUs have free outside XLA's
    pool (`outside`, by ordinal, with the step's kernels loaded), naming the bytes, before it runs."""
    needed = needs.needed(kinds, runtime=False)
    room = {gpu: free for gpu, free in outside.items() if free is not None}
    if any(needed.get(gpu, 0) > free for gpu, free in room.items()):
        made = sorted(KINDS[kind] for kind in kinds)
        raise CommunicatorRoom(f"the compiled step's collectives make {made} communicators, which its GPUs "
                               "have too little room outside XLA's pool for; lower "
                               "XLA_PYTHON_CLIENT_MEM_FRACTION by the shortfall over the GPU's bytes",
                               {gpu: needed.get(gpu, 0) for gpu in room}, room)


NEEDS: Needs | None = None
"""This process's measurement, where `sized_pool` took one."""


def sized_pool() -> None:
    """Size XLA's pool on a host of several GPUs to leave room for its communicators (`pool_share`), before
    the backend opens: Dew's share where the user named none, the user's held to it where they did. A pool
    that grows (XLA_PYTHON_CLIENT_PREALLOCATE=false) holds no share to size. A pool of processes is not
    measured: its communicators span hosts, whose network buffers one host's child does not make."""
    global NEEDS
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "true").lower() in ("false", "0"):
        return
    NEEDS = needs_of()
    if NEEDS is None:
        return
    memory = {ordinal: answered for ordinal in range(gpu_count())
              if (answered := gpu_memory(ordinal)) is not None}
    named = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
    share = pool_share(NEEDS, memory, None if named is None else float(named), frozenset({0}))
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(share)


def main() -> None:
    """The child's measurement: each GPU's free bytes before the runtime starts, then after a communicator of
    each kind is made over every GPU (an all-gather's, then a permute's), each less the pool's own growth; the
    rest the runtime took beside its pool."""
    from dew.devices import gpu_free_bytes

    ordinals = range(gpu_count())
    before = {ordinal: gpu_free_bytes(ordinal) or 0 for ordinal in ordinals}

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    devices = jax.local_devices()
    mesh = Mesh(np.array(devices), ("all",))
    rows = jax.device_put(jnp.ones((len(devices), 256), jnp.float32),
                          NamedSharding(mesh, PartitionSpec("all")))
    jax.block_until_ready(rows * 2)

    def held() -> dict[int, tuple[int, int]]:
        return {device.local_hardware_id: (gpu_free_bytes(device.local_hardware_id) or 0,
                                           int((device.memory_stats() or {}).get("pool_bytes", 0)))
                for device in devices}

    def taken(collective) -> dict[int, int]:
        start = held()
        jax.block_until_ready(jax.shard_map(collective, mesh=mesh, in_specs=PartitionSpec("all"),
                                            out_specs=PartitionSpec("all"))(rows))
        end = held()
        return {gpu: (start[gpu][0] - end[gpu][0]) - (end[gpu][1] - start[gpu][1]) for gpu in start}

    count = len(devices)
    gathered = taken(lambda part: jax.lax.all_gather(part, "all", tiled=True)[:part.shape[0]])
    permuted = taken(lambda part: jax.lax.ppermute(part, "all", [(i, (i + 1) % count) for i in range(count)]))
    after = held()
    runtime = {gpu: before[gpu] - after[gpu][0] - after[gpu][1] - gathered[gpu] - permuted[gpu]
               for gpu in after}
    sys.stdout.write(json.dumps({"communicators": {"0": gathered, "1": permuted}, "runtime": runtime}) + "\n")


if __name__ == "__main__":
    main()
