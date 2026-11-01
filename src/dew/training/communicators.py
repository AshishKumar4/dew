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
version and settings), so a host pays for it once. The pool's share then leaves that room at NCCL's full
speed, the user's share (or Dew's 95%) its upper bound (`pool_share`): the pool need hold only the compiled
step, which the fit check holds to it, so no channel or buffer of NCCL's is cut to fit. Once the step is
compiled, its communicators are read from its program and held to the room left outside the
pool (`Resident.admit`), with the communicators earlier steps made, so a step that would fail to connect is
refused before it runs."""

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
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np

from dew.cache import dew_cache_dir
from dew.devices import cuda_driver_version, gpu_count, gpu_memory, gpu_name, nccl_version
from dew.files import write_atomically

_log = logging.getLogger(__name__)

KINDS = {0: "collective", 1: "permute"}
"""The communicators XLA keys apart over the same GPUs, by communication id."""

PERMUTES = re.compile(r"\b(collective-permute|send|recv)(-start)?\(")
COLLECTIVES = re.compile(r"\b(all-gather|all-reduce|reduce-scatter|all-to-all|ragged-all-to-all)(-start)?\(")


Clique = tuple[int, tuple[tuple[int, ...], ...]]
"""A communicator as XLA keys it: its kind (`KINDS`) and its device groups, each a tuple of GPU ordinals."""


@dataclasses.dataclass(frozen=True)
class Needs:
    """What a child measured: each kind's communicator's bytes on each of its GPUs, by the size of the device
    group it spans (`KINDS`; the most any GPU of the group took), the runtime's own bytes beside its pool on
    each GPU, by ordinal (cuBLAS's workspaces, the kernels it loads), and the seconds the measurement took."""

    communicators: Mapping[int, Mapping[int, int]]
    runtime: Mapping[int, int]
    seconds: float

    def to_json(self) -> dict:
        return {"communicators": {str(kind): {str(size): taken for size, taken in sizes.items()}
                                  for kind, sizes in self.communicators.items()},
                "runtime": {str(gpu): taken for gpu, taken in self.runtime.items()}, "seconds": self.seconds}

    @classmethod
    def of(cls, written: Mapping) -> Needs:
        communicators = {int(kind): {int(size): int(taken) for size, taken in sizes.items()}
                         for kind, sizes in written["communicators"].items()}
        runtime = {int(gpu): int(taken) for gpu, taken in written["runtime"].items()}
        return cls(communicators, runtime, float(written["seconds"]))

    def taken(self, kind: int, size: int) -> int:
        """A communicator's bytes on each GPU of a group of `size`: as measured, else the next larger size's,
        else the largest measured (a group's buffers grow with its peers)."""
        sizes = self.communicators.get(kind, {})
        larger = sorted(measured for measured in sizes if measured >= size)
        return sizes[larger[0]] if larger else max(sizes.values(), default=0)

    def needed(self, cliques: Iterable[Clique], gpus: Iterable[int], *, runtime: bool) -> dict[int, int]:
        """Each GPU's bytes outside the pool for `cliques` (`cliques_of`), each charged on the GPUs of its
        groups at its group's size, with the runtime's where `runtime`."""
        needed = {gpu: self.runtime.get(gpu, 0) if runtime else 0 for gpu in sorted(gpus)}
        for kind, groups in cliques:
            for group in groups:
                for gpu in group:
                    if gpu in needed:
                        needed[gpu] += self.taken(kind, len(group))
        return needed


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
    record = {"configuration": dict(configured), "needs": needs.to_json()}
    write_atomically(path, json.dumps(record, indent=1))
    _log.info("measured NCCL's communicators on this host's GPUs in %.1f s (kept in %s): %s", needs.seconds,
              path, needs.to_json())
    return needs


def needs_of(gpus: int | None = None) -> Needs | None:
    """This host's measurement (`measured`), measured by a child process where the cache holds none; None on a
    host of fewer than two GPUs, or in a process that asks JAX for no GPU, where no communicator is made."""
    platforms = os.environ.get("JAX_PLATFORMS", "")
    asks_no_gpu = bool(platforms) and not {"cuda", "gpu"} & set(platforms.split(","))
    if (gpu_count() if gpus is None else gpus) < 2 or asks_no_gpu:
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
               cliques: frozenset[Clique]) -> float:
    """The share of each GPU XLA's pool takes: the user's (XLA_PYTHON_CLIENT_MEM_FRACTION, `requested`; Dew's
    0.95 where they named none) as an upper bound, held to leave each GPU the room its `cliques`'
    communicators and the runtime take outside it at NCCL's full speed, the tightest GPU deciding, to a
    thousandth below. The pool need hold only the compiled step, which the fit check holds to it after.
    `memory` is each GPU's free and total bytes before the backend opens, every process's allocations
    counted. A GPU whose communicators leave the pool no room at all is refused, naming the bytes."""
    needed = needs.needed(cliques, memory, runtime=True)
    free = {gpu: answered[0] for gpu, answered in memory.items()}
    if any(needed[gpu] >= free[gpu] for gpu in free):
        raise CommunicatorRoom("NCCL's communicators and the runtime leave a GPU no room for XLA's pool "
                               "(measured on this host, dew.training.communicators)", needed, free)
    widest = min((free[gpu] - needed[gpu]) / total for gpu, (_, total) in memory.items())
    return min(0.95 if requested is None else requested, int(widest * 1000) / 1000)


GROUPS = re.compile(r"replica_groups=(\{[{}0-9,]*\}|\[[0-9,]+\]<=\[[0-9,]+\](?:T\([0-9,]+\))?)")
IOTA = re.compile(r"\[([0-9,]+)\]<=\[([0-9,]+)\](?:T\(([0-9,]+)\))?")


def cliques_of(program: str, gpus: int) -> frozenset[Clique]:
    """The communicators a compiled program's collectives make over this host's `gpus`: each kind (`KINDS`)
    and set of device groups, as XLA keys its cliques, the groups read alike however the program writes them
    (`groups_of`). A permute's spans every GPU."""
    every = (tuple(range(gpus)),)
    found: set[Clique] = set()
    for line in program.splitlines():
        if PERMUTES.search(line):
            found.add((1, every))
        elif COLLECTIVES.search(line):
            written = GROUPS.search(line)
            found.add((0, groups_of(written.group(1), gpus) if written else every))
    return frozenset(found)


def groups_of(written: str, gpus: int) -> tuple[tuple[int, ...], ...]:
    """The device groups XLA's text writes as a list (`{{0,1},{2,3}}`, `{}` for every device) or as an iota
    (`[2,2]<=[4]`, `[2,2]<=[2,2]T(1,0)`: the devices counted out in the second shape, transposed, laid out
    in the first, a group to a row), sorted."""
    iota = IOTA.fullmatch(written)
    if iota is not None:
        shape, laid = ([int(size) for size in part.split(",")] for part in iota.groups()[:2])
        order = [int(axis) for axis in iota.group(3).split(",")] if iota.group(3) else list(range(len(laid)))
        devices = np.arange(int(np.prod(laid))).reshape(laid).transpose(order).reshape(-1, shape[-1])
        return tuple(sorted(tuple(int(device) for device in row) for row in devices))
    listed = [tuple(int(device) for device in group.split(","))
              for group in re.findall(r"\{([0-9,]+)\}", written)]
    return tuple(sorted(listed)) if listed else (tuple(range(gpus)),)


class Resident:
    """The communicators this process has made: XLA keeps each it creates, so every step's are admitted
    together with those of the steps before it (`admit`)."""

    def __init__(self) -> None:
        self.cliques: frozenset[Clique] = frozenset()

    def admit(self, cliques: frozenset[Clique], needs: Needs, room: Mapping[int, int]) -> None:
        """Admit a compiled step's `cliques`, refusing it, naming the bytes, before it runs, where they and
        the process's earlier communicators, each counted once, with the runtime need more than the room Dew
        left outside XLA's pool (`room`, by ordinal, as `sized_pool` left it). Held against the room Dew
        left, not what is free now, which the earlier communicators already took."""
        held = self.cliques | cliques
        needed = needs.needed(held, room, runtime=True)
        if any(needed[gpu] > room[gpu] for gpu in room):
            made = sorted({KINDS[kind] for kind, _ in held})
            raise CommunicatorRoom(f"the compiled step's collectives and the process's earlier ones make "
                                   f"{len(held)} {made} communicators, which the room left outside XLA's "
                                   "pool does not hold; lower XLA_PYTHON_CLIENT_MEM_FRACTION by the "
                                   "shortfall over the GPU's bytes", needed, room)
        self.cliques = held


RESIDENT = Resident()
"""This process's communicators, as the fit check admitted them."""


NEEDS: Needs | None = None
"""This process's measurement, where `sized_pool` took one."""

SIZED: dict | None = None
"""The pool `sized_pool` set: the user's bound (`requested`, None for Dew's own), the `share` it took, each
GPU's bytes its communicators and the runtime need (`needed`) and the room it left them (`room`), which the
fit check records and holds a compiled step to."""


def gpu_memories() -> dict[int, tuple[int, int]]:
    """Each visible GPU's free and total bytes, by ordinal, as the driver reports them (`gpu_memory`)."""
    return {ordinal: answered for ordinal in range(gpu_count())
            if (answered := gpu_memory(ordinal)) is not None}


def sized_pool(*, measure: Callable[[], Needs | None] = lambda: needs_of(),
               memories: Callable[[], Mapping[int, tuple[int, int]]] = gpu_memories) -> None:
    """Size XLA's pool on a host of several GPUs to leave room for its communicators (`pool_share`), before
    the backend opens, the user's share or Dew's its upper bound. A pool
    that grows (XLA_PYTHON_CLIENT_PREALLOCATE=false) holds no share to size. A pool of processes is not
    measured: its communicators span hosts, whose network buffers one host's child does not make."""
    global NEEDS, SIZED
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "true").lower() in ("false", "0"):
        return
    NEEDS = measure()
    if NEEDS is None:
        return
    memory = memories()
    named = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
    requested = None if named is None else float(named)
    # A collective's communicator and a permute's over every GPU, as a step's cliques are known only once it
    # compiles; the fit check holds the compiled step's own to the room left (`Resident.admit`).
    cliques = frozenset((kind, (tuple(sorted(memory)),)) for kind in KINDS)
    share = pool_share(NEEDS, memory, requested, cliques)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(share)
    SIZED = {"requested": requested, "share": share, "needed": NEEDS.needed(cliques, memory, runtime=True),
             "room": {gpu: free - int(share * total) for gpu, (free, total) in memory.items()}}
    _log.info("XLA's pool takes %s of each GPU (the user's bound %s), leaving each GPU %s bytes for NCCL's "
              "communicators at full speed and the runtime", share, requested, SIZED["needed"])


def main() -> None:
    """The child's measurement: each GPU's free bytes before the runtime starts; then, for each group size
    that divides this host's GPUs (the first GPUs of that many), the bytes a communicator of each kind takes
    as it is made over them (an all-gather's, then a permute's), less the pool's own growth, the most any of
    them took; and the rest, the runtime's beside its pool."""
    from dew.devices import gpu_free_bytes

    before = {ordinal: gpu_free_bytes(ordinal) or 0 for ordinal in range(gpu_count())}

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    devices = jax.local_devices()

    def held() -> dict[int, tuple[int, int]]:
        return {device.local_hardware_id: (gpu_free_bytes(device.local_hardware_id) or 0,
                                           int((device.memory_stats() or {}).get("pool_bytes", 0)))
                for device in devices}

    spent = dict.fromkeys(before, 0)

    def taken(group: list, collective) -> int:
        """The most bytes a GPU of `group` gave the communicator `collective` makes, each GPU's kept in
        `spent`."""
        mesh = Mesh(np.array(group), ("group",))
        placed = NamedSharding(mesh, PartitionSpec("group"))
        rows = jax.device_put(jnp.ones((len(group), 256), jnp.float32), placed)
        jax.block_until_ready(rows * 2)
        start = held()
        jax.block_until_ready(jax.shard_map(collective, mesh=mesh, in_specs=PartitionSpec("group"),
                                            out_specs=PartitionSpec("group"))(rows))
        end = held()
        gave = {gpu: (start[gpu][0] - end[gpu][0]) - (end[gpu][1] - start[gpu][1])
                for gpu in (device.local_hardware_id for device in group)}
        for gpu, given in gave.items():
            spent[gpu] += given
        return max(gave.values())

    def gathered(part):
        return jax.lax.all_gather(part, "group", tiled=True)[:part.shape[0]]

    communicators: dict[str, dict[str, int]] = {"0": {}, "1": {}}
    for size in (size for size in range(2, len(devices) + 1) if len(devices) % size == 0):
        group = devices[:size]
        ring = [(index, (index + 1) % size) for index in range(size)]
        communicators["0"][str(size)] = taken(group, gathered)
        communicators["1"][str(size)] = taken(group, lambda part, ring=ring: jax.lax.ppermute(part, "group",
                                                                                                ring))
    after = held()
    # The runtime's share on each GPU: what went that neither the pool's growth nor a communicator took.
    runtime = {gpu: before[gpu] - after[gpu][0] - after[gpu][1] - spent[gpu] for gpu in after}
    sys.stdout.write(json.dumps({"communicators": communicators, "runtime": runtime}) + "\n")


if __name__ == "__main__":
    main()
