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
communicator connected along each directed pair in turn, as NCCL picks each peer's transport apart, and
reports the bytes each GPU's driver saw go: NCCL's, and the runtime's own beside its pool. The answer is kept
in Dew's cache per configuration (`configuration`), so a host pays for it once. The pool's share then leaves
that room at NCCL's full speed, the run's share its upper bound (`dew.devices.pool_bound`, `pool_share`): the
pool need hold only the compiled step, which the fit check holds to it, so no channel or buffer of NCCL's is
cut to fit. Every program Dew compiles passes one point (`dew.nn.kernels.templates.report.compiled`), where
its communicators, read from its HLO by the parser the FLOP count reads, are held with those of the programs
before it to the room left (`admit`), so a program that would fail to connect is refused before it runs."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxlib.xla_client import LoadedExecutable

from dew import pool
from dew.cache import dew_cache_dir, package_digest
from dew.devices import (
    backend_open,
    cuda_driver_version,
    cuda_plugin,
    fraction_name,
    gpu_free_bytes,
    gpu_memory,
    gpu_name,
    gpu_place,
    nccl_version,
    pool_bound,
)
from dew.files import write_atomically
from dew.telemetry.instrumentation import hlo_instructions
from dew.telemetry.trace import collective

_log = logging.getLogger(__name__)

EXCHANGES = frozenset({"all-to-all", "ragged-all-to-all"})
"""The collectives that connect every peer of their group point to point as they first run."""

PERMUTES = frozenset({"collective-permute", "send", "recv"})
"""The ops that run on a permute's communicator (XLA's communication id 1), each connecting the peers it
sends to and receives from as it first runs."""


@dataclasses.dataclass(frozen=True)
class Collectives:
    """The communicators a process's programs make on its GPUs, by GPU ordinal: each distinct device group a
    collective takes (`groups`, a communicator each, its devices as the program orders them), each an
    all-to-all takes (`exchanges`, its connections to every peer of the group), and each permute's clique
    (the devices of the program that runs it) with the directed pairs it sends along (`permutes`)."""

    groups: frozenset[tuple[int, ...]] = frozenset()
    exchanges: frozenset[tuple[int, ...]] = frozenset()
    permutes: Mapping[tuple[int, ...], frozenset[tuple[int, int]]] = dataclasses.field(default_factory=dict)

    def __or__(self, other: Collectives) -> Collectives:
        joined = {clique: self.permutes.get(clique, frozenset()) | other.permutes.get(clique, frozenset())
                  for clique in {*self.permutes, *other.permutes}}
        return Collectives(self.groups | other.groups, self.exchanges | other.exchanges, joined)

    def to_json(self) -> dict:
        return {"groups": sorted(map(list, self.groups)), "exchanges": sorted(map(list, self.exchanges)),
                "permutes": [[list(clique), sorted(map(list, pairs))]
                             for clique, pairs in self.permutes.items()]}

    @classmethod
    def of(cls, written: Mapping, gpus: Sequence[int]) -> Collectives:
        """Communicators written as positions in a device assignment (`to_json`), on the assignment's
        `gpus`."""
        def placed(positions) -> tuple[int, ...]:
            return tuple(gpus[int(position)] for position in positions)
        return cls(frozenset(map(placed, written["groups"])), frozenset(map(placed, written["exchanges"])),
                   {placed(clique): frozenset((gpus[int(source)], gpus[int(sink)]) for source, sink in pairs)
                    for clique, pairs in written["permutes"]})

    def beyond(self, held: Collectives) -> Collectives:
        """What these communicators add to `held`: the groups and exchanges it has not made, and each
        permute clique's pairs it has not connected, a clique it has not made among them with no pairs."""
        added = {clique: pairs - held.permutes.get(clique, frozenset())
                 for clique, pairs in self.permutes.items()
                 if clique not in held.permutes or pairs - held.permutes[clique]}
        return Collectives(self.groups - held.groups, self.exchanges - held.exchanges, added)


def _at(table: Mapping[int, int], size: int) -> int:
    """`table`'s bytes for a group of `size`: as measured, else the next larger size's, else the largest
    measured (a group's buffers grow with its peers)."""
    larger = sorted(measured for measured in table if measured >= size)
    return table[larger[0]] if larger else max(table.values(), default=0)


@dataclasses.dataclass(frozen=True)
class Needs:
    """What a child measured on a group of each size, each GPU's bytes: a collective's communicator
    (`collective`), an all-to-all's connections to every peer beside it (`exchanged`), and a permute's
    communicator's base on every member (`permuted`); each directed pair's connection, its bytes on its
    source, on its target and the most on any other GPU, the most of any clique it was measured in
    (`connections`, `measured_connections`); the runtime's own bytes beside its pool on each GPU, by ordinal
    (CUDA's context, cuBLAS's and cuDNN's handles and workspaces, as a matrix product and a convolution leave
    them; each program's own code is held as it is admitted, `Loaded`);
    and the seconds the measurement took."""

    collective: Mapping[int, int]
    exchanged: Mapping[int, int]
    permuted: Mapping[int, int]
    connections: Mapping[tuple[int, int], tuple[int, int, int]]
    runtime: Mapping[int, int]
    seconds: float

    def to_json(self) -> dict:
        return {"collective": {str(size): taken for size, taken in self.collective.items()},
                "exchanged": {str(size): taken for size, taken in self.exchanged.items()},
                "permuted": {str(size): taken for size, taken in self.permuted.items()},
                "connections": [[*pair, *gave] for pair, gave in self.connections.items()],
                "runtime": {str(gpu): taken for gpu, taken in self.runtime.items()}, "seconds": self.seconds}

    @classmethod
    def of(cls, written: Mapping) -> Needs:
        return cls({int(size): int(taken) for size, taken in written["collective"].items()},
                   {int(size): int(taken) for size, taken in written["exchanged"].items()},
                   {int(size): int(taken) for size, taken in written["permuted"].items()},
                   {(int(source), int(target)): (int(at_source), int(at_target), int(beside))
                    for source, target, at_source, at_target, beside in written["connections"]},
                   {int(gpu): int(taken) for gpu, taken in written["runtime"].items()},
                   float(written["seconds"]))

    def needed(self, held: Collectives, gpus: Iterable[int], *, runtime: bool,
               made: Collectives | None = None) -> dict[int, int]:
        """Each of `gpus`' bytes outside the pool for `held` beyond what `made` holds already
        (`Collectives.beyond`): every group's communicator and every exchange's connections on the GPUs of
        the group, at its size; every new permute clique's base on its members, and each new pair's
        connection on its source and its target and, as NCCL may route it through any of them, the most a
        GPU beside it took on every other member of its clique; with the runtime's where `runtime`."""
        made = Collectives() if made is None else made
        added = held.beyond(made)
        needed = {gpu: self.runtime.get(gpu, 0) if runtime else 0 for gpu in sorted(gpus)}

        def charge(gpu: int, taken: int) -> None:
            if gpu in needed:
                needed[gpu] += taken

        for groups, table in ((added.groups, self.collective), (added.exchanges, self.exchanged)):
            for group in groups:
                for gpu in group:
                    charge(gpu, _at(table, len(group)))
        for clique, pairs in added.permutes.items():
            if clique not in made.permutes:
                for gpu in clique:
                    charge(gpu, _at(self.permuted, len(clique)))
            for source, target in pairs:
                at_source, at_target, beside = self.connections.get((source, target), (0, 0, 0))
                charge(source, at_source)
                charge(target, at_target)
                for gpu in clique:
                    if gpu not in (source, target):
                        charge(gpu, beside)
        return needed


class CommunicatorRoom(ValueError):
    """A pool share, or a compiled program, that leaves a GPU less room outside XLA's pool than its
    communicators take: `needed` and `room` in bytes, by GPU ordinal."""

    def __init__(self, message: str, needed: Mapping[int, int], room: Mapping[int, int]):
        self.needed, self.room = dict(needed), dict(room)
        short = {gpu: needed[gpu] - room[gpu] for gpu in needed if needed[gpu] > room.get(gpu, 0)}
        super().__init__(f"{message}: each GPU needs {self.needed} bytes outside XLA's pool and has "
                         f"{self.room}, short by {short}")


def local_gpus() -> tuple[int, ...] | None:
    """The GPUs this process's JAX opens, by the driver's ordinals (`dew.pool.local_gpus` under JAX's own
    selection, `jax_cuda_visible_devices`); None where they could not be counted."""
    return pool.local_gpus(jax.config.read("jax_cuda_visible_devices"))


def configuration() -> dict:
    """What a measurement holds for: this host's GPUs by name and by place (PCI bus id, which their links
    follow), the driver's CUDA version, NCCL's version and every NCCL setting in the environment, each XLA
    flag that sets NCCL's communicators, and the JAX build that makes them."""
    gpus = local_gpus() or ()
    flags = os.environ.get("XLA_FLAGS", "").split()
    return {"gpus": [gpu_name(index) for index in gpus],
            "placement": [gpu_place(index) for index in gpus], "driver": cuda_driver_version(),
            "nccl": nccl_version(), "nccl_env": {name: value for name, value in sorted(os.environ.items())
                                                 if name.startswith("NCCL_")},
            "xla_nccl": [setting for setting in flags if "nccl" in setting], "jax": jax.__version__}


FORM = 6
"""The form of the measurements Dew keeps, which names their place in the cache: a change to what the child
measures or how it writes it takes the next, so a host measures afresh rather than read an earlier one, and
a change elsewhere in Dew measures nothing again. A program's communicators are kept under the digest of
Dew's own code instead (`collectives_kept`)."""


def measured(configured: Mapping, *, measure: Callable[[Mapping], Needs]) -> Needs:
    """The measurement for `configured`, from Dew's cache, or `measure`d and kept there, in this `FORM`."""
    digest = hashlib.sha256(json.dumps(configured, sort_keys=True).encode()).hexdigest()[:24]
    path = Path(dew_cache_dir()) / "communicators" / f"form-{FORM}" / f"{digest}.json"
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
    count = len(local_gpus() or ()) if gpus is None else gpus
    if count < 2 or asks_no_gpu:
        return None
    return measured(configuration(), measure=_child)


def _child(configured: Mapping) -> Needs:
    """The measurement a child process of this interpreter makes (`main`), its environment the parent's."""
    began = time.perf_counter()
    environment = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    # The child opens the GPUs this process's JAX selects (`local_gpus`), which JAX reads from no environment.
    selected = jax.config.read("jax_cuda_visible_devices")
    done = subprocess.run([sys.executable, "-m", "dew.communicators", selected], capture_output=True,
                          text=True, env=environment, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"measuring NCCL's communicators on this host's GPUs failed (exit "
                           f"{done.returncode}), so no pool share can be sized to leave them room:\n"
                           f"{done.stderr[-4000:]}")
    written = json.loads(done.stdout.strip().splitlines()[-1])
    return Needs.of({**written, "seconds": time.perf_counter() - began})


def everything(gpus: Iterable[int]) -> Collectives:
    """What a program over `gpus` may make before it compiles: a collective's communicator over all of them,
    its all-to-all's connections to every peer, and a permute's along every pair."""
    every = tuple(sorted(gpus))
    return Collectives(frozenset({every}), frozenset({every}),
                       {every: frozenset((source, target) for source in every for target in every
                                         if source != target)})


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


def assigned(devices: Iterable) -> tuple[int, ...]:
    """The `devices` a program loaded on this process runs on (its `LoadedExecutable`'s, `admit`), as ordinals
    in the order of its device assignment, which its collectives' groups count positions in."""
    return tuple(ordinal(device) for device in devices)


def collectives_of(program: str, gpus: tuple[int, ...]) -> Collectives:
    """The communicators a compiled program's text makes on `gpus` (its device assignment, `assigned`), read
    by the parser the FLOP count reads (`hlo_instructions`) and its collective vocabulary
    (`dew.telemetry.trace.collective`): each distinct group of its collectives and of its all-to-alls,
    however XLA prints it, and its permutes' directed pairs on one clique of all its devices; a group whose
    positions lie past the assignment is taken as every device, and a send or a receive of no pairs read
    as connecting every pair. The communicators it read are kept under the digest of Dew's own code, so
    a change to how it reads a program reads them again (`collectives_kept`)."""
    groups: set[tuple[int, ...]] = set()
    exchanges: set[tuple[int, ...]] = set()
    pairs: set[tuple[int, int]] = set()
    for instruction in hlo_instructions(program).values():
        kind = collective(instruction.op)
        if kind is None:
            continue
        read = [group for group in instruction.groups if all(position < len(gpus) for position in group)]
        if len(read) < len(instruction.groups):
            read = []
        if kind in PERMUTES:
            pairs.update((gpus[source], gpus[target]) for source, target in read)
            if not read:
                pairs.update((source, target) for source in gpus for target in gpus if source != target)
            continue
        taken = [tuple(gpus[position] for position in group) for group in read] or [gpus]
        groups.update(taken)
        if kind in EXCHANGES:
            exchanges.update(taken)
    return Collectives(frozenset(groups), frozenset(exchanges), {gpus: frozenset(pairs)} if pairs else {})


class Resident:
    """The communicators this process has made: XLA keeps each it creates, so every program is admitted with
    those of the programs before it (`admit`), one program at a time, as programs of different keys compile
    at once (`dew.nn.kernels.options.OuterProgram`)."""

    def __init__(self) -> None:
        self.held = Collectives()
        self.lock = threading.Lock()

    def admit(self, made: Collectives, needs: Needs, room: Mapping[int, int],
              free: Callable[[int], int | None], loaded: Loaded | None = None) -> None:
        """Admit a compiled program's communicators (`made`), refusing it, naming the bytes, before it runs,
        where they and the process's earlier ones, each counted once, with the runtime's need more than the
        room Dew left outside XLA's pool (`room`, by ordinal, as `sized_pool` left it), or where those it
        adds (`Needs.needed`) need more than its GPUs have free there now (`free`, by ordinal; a GPU of no
        answer is held to the room alone). What is free now is read after the runtime started and after
        the earlier programs that ran made their communicators, so neither is charged again. NCCL takes a
        communicator's buffers as its program first runs, which no process can see, so free memory cannot
        tell an earlier program not yet run from one that ran: the room holds every program's together,
        and what is free now refuses a program whose own communicators cannot fit even beside the earlier
        ones all made. A program's code (`loaded`) loads as it first runs and leaves as its executable is
        released, which no process can see either, so it is held to what is free now beside the
        communicators it adds, at every admission (an executable admitted again over-counts its code alone),
        and the room holds what stays made. Each refusal names the comparison that failed."""
        with self.lock:
            held = self.held | made
            needed = needs.needed(held, room, runtime=True)
            added = needs.needed(made, room, runtime=False, made=self.held)
            for gpu in set(loaded.gpus if loaded is not None else ()) & set(added):
                added[gpu] += loaded.code if loaded is not None else 0
            now = {gpu: answered for gpu in room if (answered := free(gpu)) is not None}
            new = made.beyond(self.held)
            adds = (f"the compiled program's collectives add {len(new.groups)} collective groups, "
                    f"{len(new.exchanges)} exchanges and {sum(map(len, new.permutes.values()))} permute "
                    "connections to the process's")
            if any(needed[gpu] > room[gpu] for gpu in room):
                raise CommunicatorRoom(f"{adds}, which with the runtime's need more room than Dew left "
                                       f"outside XLA's pool; lower {fraction_name()} by the shortfall over "
                                       "the GPU's bytes", needed, room)
            if any(added[gpu] > now[gpu] for gpu in now):
                raise CommunicatorRoom(f"{adds}, which with its code need more than its GPUs have free "
                                       "outside XLA's pool now", {gpu: added[gpu] for gpu in now},
                                       {gpu: now[gpu] for gpu in now})
            self.held = held


@dataclasses.dataclass(frozen=True)
class Loaded:
    """A program's code as its GPUs hold it outside XLA's pool from its first run on: its executable's
    fingerprint, the GPUs it runs on, and the bytes of its generated code on each (`memory_analysis`)."""

    fingerprint: bytes
    gpus: tuple[int, ...]
    code: int


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
    """Each GPU this process's JAX opens (`local_gpus`), its free and total bytes by the driver's ordinal, as
    the driver reports them (`gpu_memory`)."""
    return {index: answered for index in local_gpus() or ()
            if (answered := gpu_memory(index)) is not None}


def sized_pool(*, measure: Callable[[], Needs | None] = needs_of,
               memories: Callable[[], Mapping[int, tuple[int, int]]] = gpu_memories,
               cuda: bool | None = None, opened: bool | None = None) -> Sizing | None:
    """Size XLA's pool on a host of several GPUs to leave room for its communicators (`pool_share`), before
    the backend opens, the run's share its upper bound, and return the sizing; the process holds it once
    `hold` is given it. None where nothing is sized: a process off CUDA (`cuda`, JAX's CUDA plugin by
    default) or whose backend has opened (`opened`), a pool of no fixed share (`dew.devices.pool_bound`), or
    a host that makes no communicator (`needs_of`). A pool of processes is not measured: its communicators
    span hosts, whose network buffers one host's child does not make. Once the backend has opened, the pool
    keeps the share it took, so a process prepared again (a sweep's next run, a profile after a run) gets the
    sizing it holds (`held`)."""
    cuda = cuda_plugin() if cuda is None else cuda
    if backend_open() if opened is None else opened:
        return _HELD
    if (bound := pool_bound(cuda=cuda)) is None:
        return None
    needs = measure()
    if needs is None:
        return None
    memory = memories()
    # Every communicator a program over this host's GPUs may make, as a program's are known only once it
    # compiles; each compiled program's own are held to the room left (`admit`).
    held = everything(memory)
    share = pool_share(needs, memory, bound, held)
    os.environ[fraction_name()] = str(share)
    sizing = Sizing(needs, bound, share, needs.needed(held, memory, runtime=True),
                    {gpu: free - int(share * total) for gpu, (free, total) in memory.items()})
    _log.info("XLA's pool takes %s of each GPU (the run's bound %s), leaving each GPU %s bytes for NCCL's "
              "communicators at full speed and the runtime", share, bound, dict(sizing.needed))
    return sizing


_HELD: Sizing | None = None
_RESIDENT = Resident()


def hold(sizing: Sizing | None) -> None:
    """Have this process admit every program it compiles or loads (`admit`), its pool sized as `sizing`
    says; a process holds one sizing, the one its preparation made, and None holds nothing."""
    global _HELD
    _HELD = sizing


def held() -> Sizing | None:
    """The sizing this process holds (`hold`), None where its pool was not sized."""
    return _HELD


def collectives_kept(fingerprint: bytes, gpus: tuple[int, ...], text: Callable[[], str]) -> Collectives:
    """The communicators an executable makes on `gpus` (`collectives_of`), read from its `text` once per
    executable and kept in Dew's cache under its `fingerprint` as positions in its device assignment, below
    the digest of Dew's own code (`dew.cache.package_digest`), so a warm start that loads it from the
    compilation cache reads no text, and another Dew's are read again."""
    kept = Path(dew_cache_dir()) / "communicators" / "programs" / package_digest()
    path = kept / f"{fingerprint.hex()}.json"
    if path.exists():
        return Collectives.of(json.loads(path.read_text()), gpus)
    positions = collectives_of(text(), tuple(range(len(gpus))))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, json.dumps(positions.to_json()))
    return Collectives.of(positions.to_json(), gpus)


def admit(executable: jax.stages.Compiled, free: Callable[[int], int | None] = gpu_free_bytes) -> None:
    """Admit a compiled or loaded program's communicators and code against the room the process's sizing left
    them and its GPUs have free outside XLA's pool now (`Resident`), refusing one that would not fit before it
    runs; a process that holds no sizing admits every program, and a program compiled for devices this process
    has not (`dew.training.ahead.compile_only_gpus`) is never loaded here, so it takes none of this host's
    room, as does a program on the host's CPU. Every program Dew compiles passes here
    (`dew.nn.kernels.templates.report`), the RL rollouts' among them (`dew.nn.kernels.options.OuterProgram`),
    and every executable it loads compiled ahead (`dew.training.ahead`)."""
    runtime = executable.runtime_executable()
    if _HELD is None or runtime is None or not isinstance(runtime, LoadedExecutable):
        return
    devices = runtime.local_devices()
    if not devices or any(device.platform != "gpu" for device in devices):
        # A program on the host's CPU makes no NCCL communicator and holds no GPU code.
        return
    gpus = assigned(devices)
    fingerprint = runtime.fingerprint
    if not isinstance(fingerprint, bytes):
        made = collectives_of(executable.as_text() or "", gpus)
        loaded = None
    else:
        made = collectives_kept(fingerprint, gpus, lambda: executable.as_text() or "")
        memory = executable.memory_analysis()
        # A backend that states no memory analysis states no code of the program's either.
        loaded = None if memory is None else Loaded(fingerprint, gpus, memory.generated_code_size_in_bytes)
    _RESIDENT.admit(made, _HELD.needs, _HELD.room, free, loaded)


type Run = tuple[int, int, Mapping[int, int]]
"""A permute along one directed pair as the child ran it: its source, its target and each GPU's bytes."""


def peer_costs(runs: Sequence[Run]) -> tuple[int, dict[tuple[int, int], tuple[int, int, int]]]:
    """A permute communicator's base and each directed pair's connection on one clique, from `runs`
    connecting each directed pair of it in turn, the first making the communicator. NCCL picks each peer's
    transport apart and may route one through a third GPU, whose buffers it then holds
    (`P2P_INTERMEDIATE`), so each later pair keeps its bytes on its source, on its target and the most on
    any other GPU. The first pair's own connection is its reverse's, a transport's buffers being the same
    both ways, and the base is the most a GPU took in the first run beside it, so the clique's runs are
    charged what they took in all."""
    (source, target, first), *later = runs

    def beside(source: int, target: int, gave: Mapping[int, int]) -> int:
        return max((given for gpu, given in gave.items() if gpu not in (source, target)), default=0)

    connections = {(sender, receiver): (max(0, gave.get(sender, 0)), max(0, gave.get(receiver, 0)),
                                        max(0, beside(sender, receiver, gave)))
                   for sender, receiver, gave in later}
    connections[(source, target)] = own = connections.get((target, source), (0, 0, 0))
    base = max(first[source] - own[0], first[target] - own[1],
               *(given - own[2] for gpu, given in first.items() if gpu not in (source, target)))
    return max(0, base), connections


def measured_connections(cliques: Iterable[Sequence[Run]]
                         ) -> tuple[dict[int, int], dict[tuple[int, int], tuple[int, int, int]]]:
    """The permute communicators' bases by clique size and each directed pair's connection, the most of
    every clique the child measured it in (`peer_costs`): the cliques of the host's first GPUs and each pair
    of GPUs alone, where NCCL routes the pair directly, as it must in a clique that holds no GPU to route
    it through."""
    bases: dict[int, int] = {}
    connections: dict[tuple[int, int], tuple[int, int, int]] = {}

    def kept(pair: tuple[int, int], costs: tuple[int, int, int]) -> None:
        held = connections.get(pair, (0, 0, 0))
        connections[pair] = (max(held[0], costs[0]), max(held[1], costs[1]), max(held[2], costs[2]))

    for runs in cliques:
        size = len(runs[0][2])
        base, measured = peer_costs(runs)
        bases[size] = max(bases.get(size, 0), base)
        for pair, costs in measured.items():
            kept(pair, costs)
    return bases, connections


def main() -> None:
    """The child's measurement: each GPU's free bytes before the runtime starts; then, for each group size
    that divides this host's GPUs (the first GPUs of that many), the bytes each GPU gave, less its pool's
    growth, as a collective's communicator is made over them (an all-gather's), as an all-to-all connects
    every peer beside it, each the most a member gave, and as a permute's communicator connects each
    directed pair of them in turn (`peer_costs`); and the rest, after a matrix product and a convolution on
    each GPU (cuBLAS's and cuDNN's handles and workspaces), the runtime's beside its pool."""
    import jax.numpy as jnp
    import numpy as np

    # The GPUs its parent's JAX selects (`_child`), selected before the backend opens.
    jax.config.update("jax_cuda_visible_devices", sys.argv[1] if len(sys.argv) > 1 else pool.ALL_GPUS)
    before = {index: gpu_free_bytes(index) or 0 for index in local_gpus() or ()}
    devices = jax.local_devices()

    def held_now() -> dict[int, tuple[int, int]]:
        return {device.local_hardware_id: (gpu_free_bytes(device.local_hardware_id) or 0,
                                           int((device.memory_stats() or {}).get("pool_bytes", 0)))
                for device in devices}

    spent = dict.fromkeys(before, 0)

    def taken(group: list, program: Callable) -> dict[int, int]:
        """The bytes each GPU of `group` gave as `program` first ran over it, kept in `spent` as well."""
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
        return gave

    def gathered(part):
        return jax.lax.all_gather(part, "group", tiled=True)[:part.shape[0]]

    def exchanged_rows(part):
        return jax.lax.all_to_all(part, "group", 0, 0, tiled=True)

    def permuted_along(pair: tuple[int, int]) -> Callable:
        return lambda part: jax.lax.ppermute(part, "group", [pair])

    def connecting(group: list) -> list[Run]:
        return [(group[source].local_hardware_id, group[target].local_hardware_id,
                 taken(group, permuted_along((source, target))))
                for source, target in itertools.permutations(range(len(group)), 2)]

    collective, exchanged, cliques = {}, {}, []
    sizes = [size for size in range(2, len(devices) + 1) if len(devices) % size == 0]
    for size in sizes:
        group = devices[:size]
        collective[str(size)] = max(taken(group, gathered).values())
        exchanged[str(size)] = max(taken(group, exchanged_rows).values())
        cliques.append(connecting(group))
    # Each pair alone, the first two GPUs' unless a group of two measured them already.
    pairs = itertools.combinations(range(len(devices)), 2)
    alone = [pair for pair in pairs if (*pair, sizes[0]) != (0, 1, 2)]
    cliques += [connecting([devices[one], devices[other]]) for one, other in alone]
    permuted, connections = measured_connections(cliques)
    for device in devices:
        square = jax.device_put(jnp.ones((512, 512), jnp.float32), device)
        image = jax.device_put(jnp.ones((1, 8, 32, 32), jnp.float32), device)
        jax.block_until_ready((square @ square,
                               jax.lax.conv(image, jnp.ones((8, 8, 3, 3), jnp.float32), (1, 1), "SAME")))
    after = held_now()
    # The runtime's share on each GPU: what went that neither the pool's growth nor a communicator took.
    runtime = {gpu: before[gpu] - after[gpu][0] - after[gpu][1] - spent[gpu] for gpu in after}
    sys.stdout.write(json.dumps({"collective": collective, "exchanged": exchanged, "permuted": permuted,
                                 "connections": [[*pair, *gave] for pair, gave in connections.items()],
                                 "runtime": runtime}) + "\n")


if __name__ == "__main__":
    main()
