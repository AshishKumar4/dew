"""The memory NCCL's communicators take outside XLA's pool on this host's GPUs, measured before the backend
opens, and the pool share that leaves it (ledger 2901).

XLA's pool takes a fixed share of each GPU as its backend opens (XLA_PYTHON_CLIENT_MEM_FRACTION). NCCL
allocates each communicator's buffers outside it, and nothing draws them from the pool: `ncclConfig_t` carries
no allocator, and XLA passes only its blocking mode, split sharing and channel caps
(xla/backends/gpu/collectives/nccl_collectives.cc, `AsNcclConfig`). XLA keys a communicator by its devices and
its kind, a collective's (communication id 0: all-gather, all-reduce, reduce-scatter, all-to-all) apart from a
permute's (id 1: collective-permute, send, recv; xla/backends/gpu/collectives/gpu_clique_key.h), so a program
that holds both, or collectives over two groupings of its GPUs, makes several over the same GPUs.

`import dew` has NCCL connect each communicator's rings, trees and NVLS peers as it creates it
(NCCL_RUNTIME_CONNECT=0, `dew.devices.connect_at_creation`), where XLA returns a failure to every rank as an
error. Connected lazily, as NCCL does by default since 2.22, a rank that fails to allocate at a collective's
first launch leaves its peer waiting inside XLA's first-call rendezvous, which aborts the process after 40
seconds (xla/service/rendezvous.cc, LOG(FATAL)): a 2xH100 LoRA run, whose first all-to-all one GPU entered
and the other never did. Point-to-point connections, an all-to-all's and a permute's, still connect as each
peer is first sent to, whatever that setting; their room is held for them here.

Before the backend opens, a child process on this host's GPUs (`main`) makes, over each group size that
divides them, a collective's communicator, an all-to-all's connections to every peer, and a permute's
communicator to one peer and to every peer, and reports the bytes each GPU's driver saw go: NCCL's, and the
runtime's own beside its pool. The answer is kept in Dew's cache per configuration (`configuration`), so a
host pays for it once. The pool's share then leaves that room at NCCL's full speed, the run's share its upper
bound (`dew.devices.pool_bound`, `pool_share`): the pool need hold only the compiled step, which the fit
check holds to it, so no channel or buffer of NCCL's is cut to fit. Every program Dew compiles passes one
point (`dew.nn.kernels.templates.report.compiled`), where its communicators, read from its HLO by the parser
the FLOP count reads, are held with those of the programs before it to the room left (`admit`), so a program
that would fail to connect is refused before it runs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from dew.cache import dew_cache_dir
from dew.devices import (
    backend_open,
    cuda_driver_version,
    cuda_plugin,
    gpu_count,
    gpu_memory,
    gpu_name,
    gpu_place,
    nccl_version,
    pool_bound,
)
from dew.files import write_atomically
from dew.telemetry.instrumentation import hlo_instructions

_log = logging.getLogger(__name__)

COLLECTIVES = frozenset({"all-gather", "all-gather-start", "all-reduce", "all-reduce-start", "reduce-scatter",
                         "all-to-all", "ragged-all-to-all", "collective-broadcast"})
"""The ops that run on a collective's communicator (XLA's communication id 0)."""

EXCHANGES = frozenset({"all-to-all", "ragged-all-to-all"})
"""The collectives that connect every peer of their group point to point as they first run."""

PERMUTES = frozenset({"collective-permute", "collective-permute-start", "send", "recv"})
"""The ops that run on a permute's communicator (XLA's communication id 1), each connecting the peers it
sends to and receives from as it first runs."""


@dataclasses.dataclass(frozen=True)
class Collectives:
    """The communicators a process's programs make on its GPUs, by GPU ordinal: each distinct device group a
    collective takes (`groups`, a communicator each), each an all-to-all takes (`exchanges`, its connections
    to every peer of the group), and each GPU's permute peers (`peers`, the GPUs it sends to or receives
    from on a permute's communicator)."""

    groups: frozenset[tuple[int, ...]] = frozenset()
    exchanges: frozenset[tuple[int, ...]] = frozenset()
    peers: Mapping[int, frozenset[int]] = dataclasses.field(default_factory=dict)

    def __or__(self, other: Collectives) -> Collectives:
        joined = {gpu: self.peers.get(gpu, frozenset()) | other.peers.get(gpu, frozenset())
                  for gpu in {*self.peers, *other.peers}}
        return Collectives(self.groups | other.groups, self.exchanges | other.exchanges, joined)


def _at(table: Mapping[int, int], size: int) -> int:
    """`table`'s bytes for a group of `size`: as measured, else the next larger size's, else the largest
    measured (a group's buffers grow with its peers)."""
    larger = sorted(measured for measured in table if measured >= size)
    return table[larger[0]] if larger else max(table.values(), default=0)


@dataclasses.dataclass(frozen=True)
class Needs:
    """What a child measured, each the most any GPU of its group took, by the group's size: a collective's
    communicator (`collective`), an all-to-all's connections to every peer beside it (`exchanged`), and a
    permute's communicator connected to one peer and to every peer (`permuted`); the runtime's own bytes
    beside its pool on each GPU, by ordinal (cuBLAS's workspaces, the kernels it loads); and the seconds the
    measurement took."""

    collective: Mapping[int, int]
    exchanged: Mapping[int, int]
    permuted: Mapping[int, tuple[int, int]]
    runtime: Mapping[int, int]
    seconds: float

    def to_json(self) -> dict:
        return {"collective": {str(size): taken for size, taken in self.collective.items()},
                "exchanged": {str(size): taken for size, taken in self.exchanged.items()},
                "permuted": {str(size): list(taken) for size, taken in self.permuted.items()},
                "runtime": {str(gpu): taken for gpu, taken in self.runtime.items()}, "seconds": self.seconds}

    @classmethod
    def of(cls, written: Mapping) -> Needs:
        return cls({int(size): int(taken) for size, taken in written["collective"].items()},
                   {int(size): int(taken) for size, taken in written["exchanged"].items()},
                   {int(size): (int(one), int(every)) for size, (one, every) in written["permuted"].items()},
                   {int(gpu): int(taken) for gpu, taken in written["runtime"].items()},
                   float(written["seconds"]))

    def permute(self, size: int, peers: int) -> int:
        """A permute's communicator over `size` GPUs on a GPU connected to `peers` of them: its bytes with
        one peer, and each further peer's share of the bytes every peer added."""
        if peers == 0 or not self.permuted:
            return 0
        larger = sorted(measured for measured in self.permuted if measured >= size)
        measured = larger[0] if larger else max(self.permuted)
        one, every = self.permuted[measured]
        return one + (every - one) * (peers - 1) // max(1, measured - 2)

    def needed(self, held: Collectives, gpus: Iterable[int], *, runtime: bool) -> dict[int, int]:
        """Each of `gpus`' bytes outside the pool for `held`: every group's communicator and every
        exchange's connections on the GPUs of the group, at its size, and each GPU's permute peers on a
        permute's communicator over all `gpus`; with the runtime's where `runtime`."""
        needed = {gpu: self.runtime.get(gpu, 0) if runtime else 0 for gpu in sorted(gpus)}
        for groups, table in ((held.groups, self.collective), (held.exchanges, self.exchanged)):
            for group in groups:
                for gpu in group:
                    if gpu in needed:
                        needed[gpu] += _at(table, len(group))
        for gpu, peers in held.peers.items():
            if gpu in needed:
                needed[gpu] += self.permute(len(needed), len(peers))
        return needed


class CommunicatorRoom(ValueError):
    """A pool share, or a compiled program, that leaves a GPU less room outside XLA's pool than its
    communicators take: `needed` and `room` in bytes, by GPU ordinal."""

    def __init__(self, message: str, needed: Mapping[int, int], room: Mapping[int, int]):
        self.needed, self.room = dict(needed), dict(room)
        short = {gpu: needed[gpu] - room[gpu] for gpu in needed if needed[gpu] > room.get(gpu, 0)}
        super().__init__(f"{message}: each GPU needs {self.needed} bytes outside XLA's pool and has "
                         f"{self.room}, short by {short}")


def configuration() -> dict:
    """What a measurement holds for: this host's GPUs by name and by place (PCI bus id, which their links
    follow), the driver's CUDA version, NCCL's version and every NCCL setting in the environment, each XLA
    flag that sets NCCL's communicators, and the JAX build that makes them."""
    count = gpu_count()
    flags = os.environ.get("XLA_FLAGS", "").split()
    return {"gpus": [gpu_name(index) for index in range(count)],
            "placement": [gpu_place(index) for index in range(count)], "driver": cuda_driver_version(),
            "nccl": nccl_version(), "nccl_env": {name: value for name, value in sorted(os.environ.items())
                                                 if name.startswith("NCCL_")},
            "xla_nccl": [setting for setting in flags if "nccl" in setting], "jax": jax.__version__}


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
    done = subprocess.run([sys.executable, "-m", "dew.communicators"], capture_output=True,
                          text=True, env=environment, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"measuring NCCL's communicators on this host's GPUs failed (exit "
                           f"{done.returncode}), so no pool share can be sized to leave them room:\n"
                           f"{done.stderr[-4000:]}")
    written = json.loads(done.stdout.strip().splitlines()[-1])
    return Needs.of({**written, "seconds": time.perf_counter() - began})


def everything(gpus: Iterable[int]) -> Collectives:
    """What a program over `gpus` may make before it compiles: a collective's communicator over all of them,
    its all-to-all's connections to every peer, and a permute's to every peer."""
    every = tuple(sorted(gpus))
    return Collectives(frozenset({every}), frozenset({every}),
                       {gpu: frozenset(every) - {gpu} for gpu in every})


def pool_share(needs: Needs, memory: Mapping[int, tuple[int, int]], bound: float, held: Collectives) -> float:
    """The share of each GPU XLA's pool takes: the run's `bound` (`dew.devices.pool_bound`) at most, held to
    leave each GPU the room `held`'s communicators and the runtime take outside it at NCCL's full speed, the
    tightest GPU deciding, to a thousandth below. The pool need hold only the compiled step, which the fit
    check holds to it after. `memory` is each GPU's free and total bytes before the backend opens, every
    process's allocations counted. A GPU whose communicators leave the pool no room at all is refused,
    naming the bytes."""
    needed = needs.needed(held, memory, runtime=True)
    free = {gpu: answered[0] for gpu, answered in memory.items()}
    if any(needed[gpu] >= free[gpu] for gpu in free):
        raise CommunicatorRoom("NCCL's communicators and the runtime leave a GPU no room for XLA's pool "
                               "(measured on this host, dew.communicators)", needed, free)
    widest = min((free[gpu] - needed[gpu]) / total for gpu, (_, total) in memory.items())
    return min(bound, int(widest * 1000) / 1000)


def ordinal(device) -> int:
    """A device's ordinal on its host: a GPU's CUDA ordinal, any other device's id."""
    return device.id if device.local_hardware_id is None else device.local_hardware_id


def assigned(executable: jax.stages.Compiled) -> tuple[int, ...]:
    """The GPUs a compiled program runs on, in the order of its device assignment, which its collectives'
    groups count positions in: its first named sharding's mesh, laid out; else its devices by id."""
    shardings = jax.tree.leaves(executable.input_shardings)
    for sharding in shardings:
        if isinstance(sharding, NamedSharding) and isinstance(sharding.mesh, Mesh):
            return tuple(ordinal(device) for device in sharding.mesh.devices.flat)
    devices = {device for sharding in shardings for device in sharding.device_set}
    return tuple(ordinal(device) for device in sorted(devices, key=lambda device: device.id))


def collectives_of(program: str, gpus: tuple[int, ...]) -> Collectives:
    """The communicators a compiled program's text makes on `gpus` (its device assignment, `assigned`), read
    by the parser the FLOP count reads (`hlo_instructions`): each distinct group of its collectives and of its
    all-to-alls, however XLA prints it, and each GPU's permute peers; a send or a receive of no pairs read
    connects its GPU to every other."""
    groups: set[tuple[int, ...]] = set()
    exchanges: set[tuple[int, ...]] = set()
    peers: dict[int, set[int]] = {}
    for instruction in hlo_instructions(program).values():
        positions = instruction.groups
        if instruction.op in COLLECTIVES:
            taken = [tuple(gpus[position] for position in group) for group in positions] or [gpus]
            groups.update(taken)
            if instruction.op in EXCHANGES:
                exchanges.update(taken)
        elif instruction.op in PERMUTES:
            pairs = [(gpus[source], gpus[target]) for source, target in positions]
            for source, target in pairs or [(gpu, other) for gpu in gpus for other in gpus if other != gpu]:
                peers.setdefault(source, set()).add(target)
                peers.setdefault(target, set()).add(source)
    return Collectives(frozenset(groups), frozenset(exchanges),
                       {gpu: frozenset(connected) for gpu, connected in peers.items()})


class Resident:
    """The communicators this process has made: XLA keeps each it creates, so every program's are admitted
    together with those of the programs before it (`admit`)."""

    def __init__(self) -> None:
        self.held = Collectives()

    def admit(self, made: Collectives, needs: Needs, room: Mapping[int, int]) -> None:
        """Admit a compiled program's communicators (`made`), refusing it, naming the bytes, before it runs,
        where they and the process's earlier ones, each counted once, with the runtime need more than the
        room Dew left outside XLA's pool (`room`, by ordinal, as `sized_pool` left it). Held against the
        room Dew left, not what is free now, which the earlier communicators already took."""
        held = self.held | made
        needed = needs.needed(held, room, runtime=True)
        if any(needed[gpu] > room[gpu] for gpu in room):
            raise CommunicatorRoom(f"the compiled program's collectives and the process's earlier ones make "
                                   f"{len(held.groups)} collective groups, {len(held.exchanges)} exchanges "
                                   f"and {sum(map(len, held.peers.values()))} permute connections, which the "
                                   "room left outside XLA's pool does not hold; lower "
                                   "XLA_PYTHON_CLIENT_MEM_FRACTION by the shortfall over the GPU's bytes",
                                   needed, room)
        self.held = held


@dataclasses.dataclass(frozen=True)
class Sizing:
    """A pool Dew sized: the host's measurement (`needs`), the run's bound, the share the pool took, each
    GPU's bytes its communicators and the runtime need (`needed`) and the room the share left them
    (`room`), by GPU ordinal."""

    needs: Needs
    bound: float
    share: float
    needed: Mapping[int, int]
    room: Mapping[int, int]

    def record(self) -> dict:
        """The sizing as the fit check's record carries it into the run record."""
        return {"bound": self.bound, "share": self.share,
                "needed": {str(gpu): taken for gpu, taken in self.needed.items()},
                "room": {str(gpu): room for gpu, room in self.room.items()}}


def gpu_memories() -> dict[int, tuple[int, int]]:
    """Each visible GPU's free and total bytes, by ordinal, as the driver reports them (`gpu_memory`)."""
    return {index: answered for index in range(gpu_count()) if (answered := gpu_memory(index)) is not None}


def sized_pool(*, measure: Callable[[], Needs | None] = needs_of,
               memories: Callable[[], Mapping[int, tuple[int, int]]] = gpu_memories,
               cuda: bool | None = None, opened: bool | None = None) -> Sizing | None:
    """Size XLA's pool on a host of several GPUs to leave room for its communicators (`pool_share`), before
    the backend opens, the run's share its upper bound, and return the sizing; the process holds it once
    `hold` is given it. None where nothing is sized: a process off CUDA (`cuda`, JAX's CUDA plugin by
    default) or whose backend has opened (`opened`), a pool of no fixed share (`dew.devices.pool_bound`), or
    a host that makes no communicator (`needs_of`). A pool of processes is not measured: its communicators
    span hosts, whose network buffers one host's child does not make."""
    cuda = cuda_plugin() is not None if cuda is None else cuda
    if (backend_open() if opened is None else opened) or (bound := pool_bound(cuda=cuda)) is None:
        return None
    needs = measure()
    if needs is None:
        return None
    memory = memories()
    # Every communicator a program over this host's GPUs may make, as a program's are known only once it
    # compiles; each compiled program's own are held to the room left (`admit`).
    held = everything(memory)
    share = pool_share(needs, memory, bound, held)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(share)
    sizing = Sizing(needs, bound, share, needs.needed(held, memory, runtime=True),
                    {gpu: free - int(share * total) for gpu, (free, total) in memory.items()})
    _log.info("XLA's pool takes %s of each GPU (the run's bound %s), leaving each GPU %s bytes for NCCL's "
              "communicators at full speed and the runtime", share, bound, dict(sizing.needed))
    return sizing


_HELD: Sizing | None = None
_RESIDENT = Resident()


def hold(sizing: Sizing | None) -> None:
    """Have this process admit every program it compiles against `sizing`'s room (`admit`); a process holds
    one sizing, the one its preparation made, and None holds nothing."""
    global _HELD
    _HELD = sizing


def held() -> Sizing | None:
    """The sizing this process holds (`hold`), None where its pool was not sized."""
    return _HELD


def admit(executable: jax.stages.Compiled) -> None:
    """Admit a compiled program's communicators against the room the process's sizing left (`Resident`),
    refusing one that would not fit before it runs; a process that holds no sizing admits every program."""
    if _HELD is not None:
        _RESIDENT.admit(collectives_of(executable.as_text() or "", assigned(executable)), _HELD.needs,
                        _HELD.room)


def main() -> None:
    """The child's measurement: each GPU's free bytes before the runtime starts; then, for each group size
    that divides this host's GPUs (the first GPUs of that many), the bytes a GPU gave, less its pool's
    growth, as a collective's communicator is made over them (an all-gather's), as an all-to-all connects
    every peer beside it, and as a permute's communicator connects one peer and then every peer, the most
    any of them gave; and the rest, the runtime's beside its pool."""
    from dew.devices import gpu_free_bytes

    before = {index: gpu_free_bytes(index) or 0 for index in range(gpu_count())}

    import jax.numpy as jnp
    import numpy as np

    devices = jax.local_devices()

    def held_now() -> dict[int, tuple[int, int]]:
        return {device.local_hardware_id: (gpu_free_bytes(device.local_hardware_id) or 0,
                                           int((device.memory_stats() or {}).get("pool_bytes", 0)))
                for device in devices}

    spent = dict.fromkeys(before, 0)

    def taken(group: list, program: Callable) -> int:
        """The most bytes a GPU of `group` gave as `program` first ran over it, each GPU's kept in
        `spent`."""
        mesh = Mesh(np.array(group), ("group",))
        rows = jax.device_put(jnp.ones((len(group) * len(group), 256), jnp.float32),
                              NamedSharding(mesh, PartitionSpec("group")))
        jax.block_until_ready(rows * 2)
        start = held_now()
        jax.block_until_ready(jax.shard_map(program, mesh=mesh, in_specs=PartitionSpec("group"),
                                            out_specs=PartitionSpec("group"))(rows))
        end = held_now()
        gave = {gpu: (start[gpu][0] - end[gpu][0]) - (end[gpu][1] - start[gpu][1])
                for gpu in (device.local_hardware_id for device in group)}
        for gpu, given in gave.items():
            spent[gpu] += given
        return max(gave.values())

    def every_peer(part, size: int):
        for shift in range(1, size):
            ring = [(index, (index + shift) % size) for index in range(size)]
            part = jax.lax.ppermute(part, "group", ring)
        return part

    collective, exchanged, permuted = {}, {}, {}
    for size in (size for size in range(2, len(devices) + 1) if len(devices) % size == 0):
        group = devices[:size]
        collective[str(size)] = taken(group, lambda part: jax.lax.all_gather(part, "group", tiled=True)
                                      [:part.shape[0]])
        exchanged[str(size)] = taken(group, lambda part: jax.lax.all_to_all(part, "group", 0, 0, tiled=True))
        one = taken(group, lambda part: jax.lax.ppermute(part, "group", [(0, 1), (1, 0)]))
        permuted[str(size)] = [one, one + taken(group, lambda part, size=size: every_peer(part, size))]
    after = held_now()
    # The runtime's share on each GPU: what went that neither the pool's growth nor a communicator took.
    runtime = {gpu: before[gpu] - after[gpu][0] - after[gpu][1] - spent[gpu] for gpu in after}
    sys.stdout.write(json.dumps({"collective": collective, "exchanged": exchanged, "permuted": permuted,
                                 "runtime": runtime}) + "\n")


if __name__ == "__main__":
    main()
