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
(NCCL_RUNTIME_CONNECT=0, `dew._devices.connect_at_creation`), where XLA returns a failure to every rank as an
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
that room at NCCL's full speed, the run's share its upper bound (`dew._devices.pool_bound`, `pool_share`): the
pool need hold only the compiled step, which the fit check holds to it, so no channel or buffer of NCCL's is
cut to fit. Every program Dew compiles passes one point (`dew.nn.kernels.templates.report.compiled`), where
its communicators, read from its HLO by the parser the FLOP count reads, are held with those of the programs
before it to the room left (`admit`), so a program that would fail to connect is refused before it runs."""

from __future__ import annotations

import collections
import dataclasses
import functools
import hashlib
import itertools
import json
import logging
import operator
import os
import threading
import time
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxlib.xla_client import LoadedExecutable

from dew import _pool
from dew._builds import dew_cache_dir
from dew._cache import keep, kept_beside, read_kept
from dew._code import code_digest
from dew._devices import (
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
    write_share,
)
from dew._isolated import Isolated, IsolatedFailure
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
    """The GPUs this process's JAX opens, by the driver's ordinals (`dew._pool.local_gpus` under JAX's own
    selection, `jax_cuda_visible_devices`); None where they could not be counted."""
    return _pool.local_gpus(jax.config.read("jax_cuda_visible_devices"))


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
    """The measurement for `configured`, from Dew's cache, or `measure`d and kept there (`dew._cache.keep`),
    in this `FORM`."""
    digest = hashlib.sha256(json.dumps(configured, sort_keys=True).encode()).hexdigest()[:24]
    path = Path(dew_cache_dir()) / "communicators" / f"form-{FORM}" / f"{digest}.json"
    found = read_kept(path)
    if isinstance(found, dict) and isinstance(held := found.get("needs"), dict):
        return Needs.of(held)
    needs = measure(configured)
    keep(path, {"configuration": dict(configured), "needs": needs.to_json()},
         "the host's measured communicators")
    _log.info("measured NCCL's communicators on this host's GPUs in %.1f s (kept in %s): %s", needs.seconds,
              path, needs.to_json())
    return needs


def needs_of(gpus: int | None = None) -> Needs | None:
    """This host's measurement (`measured`), measured by a child process where the cache holds none; None on a
    host of fewer than two GPUs, or in a process that asks JAX for no GPU, where no communicator is made."""
    named = _pool.named_platforms(os.environ.get("JAX_PLATFORMS"))
    asks_no_gpu = bool(named) and not named & {"cuda", "gpu"}
    count = len(local_gpus() or ()) if gpus is None else gpus
    if count < 2 or asks_no_gpu:
        return None
    return measured(configuration(), measure=_child)


def _child(configured: Mapping) -> Needs:
    """The measurement a process of its own makes (`measure`, `dew._isolated`), its environment the
    parent's, its pool growing: its reply comes over a pipe of its own, so whatever it or NCCL prints (an
    `NCCL_DEBUG` log on its stdout) reaches none of it, where a reply read from the child's stdout was
    refused for any line printed after it (row 6497's 2xH100 start)."""
    began = time.perf_counter()
    # The child opens the GPUs this process's JAX selects (`local_gpus`), which JAX reads from no environment.
    selected = jax.config.read("jax_cuda_visible_devices")
    try:
        with Isolated({"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}) as child:
            written = child.call(measure, selected)
    except IsolatedFailure as failed:
        raise RuntimeError(f"measuring NCCL's communicators on this host's GPUs failed ({failed}), so no "
                           "pool share can be sized to leave them room") from failed
    return Needs.of({**written, "seconds": time.perf_counter() - began})


def everything(gpus: Iterable[int]) -> Collectives:
    """What a program over `gpus` may make before it compiles: a collective's communicator over all of them,
    its all-to-all's connections to every peer, and a permute's along every pair."""
    every = tuple(sorted(gpus))
    return Collectives(frozenset({every}), frozenset({every}),
                       {every: frozenset((source, target) for source in every for target in every
                                         if source != target)})


def pool_share(needs: Needs, memory: Mapping[int, tuple[int, int]], bound: float, held: Collectives) -> float:
    """The share of each GPU XLA's pool takes: the run's `bound` (`dew._devices.pool_bound`) at most, held to
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


def collectives_of(program: str | jax.stages.Compiled, gpus: tuple[int, ...]) -> Collectives:
    """The communicators a compiled program makes on `gpus` (its device assignment, `assigned`), read from
    its module as the FLOP count reads it (`hlo_instructions`: the executable's own, or one's text) and its
    collective vocabulary (`dew.telemetry.trace.collective`): each distinct group of its collectives and of
    its all-to-alls, however XLA states it, and its permutes' directed pairs on one clique of all its
    devices; a group whose positions lie past the assignment is taken as every device, and a send or a
    receive of no pairs read as connecting every pair. The communicators it read are kept under the digest
    of Dew's own code, so a change to how it reads a program reads them again (`collectives_kept`)."""
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


PROGRAM_FLOOR = 8 << 20
"""The bytes outside XLA's pool a program not yet run is charged on each of its GPUs for what loading it takes
there before any first run measured one (`Resident.ahead`, `Resident.ran`): the most a step of a 1.4B decoder
took past the process's first, 6 MiB on a GH200 (row 12131's probe, its code and its first run), rounded
up."""


@dataclasses.dataclass(frozen=True)
class Pending:
    """A program admitted that has not run (`Resident.pending`): its GPUs, the communicators it makes as it
    first runs (`made`), which no reading of free memory counts until then, and on each GPU the bytes its
    admission charged for those of them no program that ran made (`charged`), which a first run's measure of
    its load leaves out (`Resident.ran`)."""

    gpus: tuple[int, ...]
    made: Collectives
    charged: Mapping[int, int]


class Resident:
    """The communicators this process has made: those of every program that ran (`held`), which XLA keeps,
    each program admitted with them (`admit`), one program at a time, as programs of different keys compile
    at once (`dew.nn.kernels.options.OuterProgram`). Beside them, by the executable's identity while it lives
    (`Loaded.key`), every program admitted that has not run (`pending`), whose communicators and load no
    reading of free memory counts yet, and each that ran (`settled`), which loads nothing again: a program
    whose first run is seen to complete leaves the one for the other, its communicators made (`ran`,
    `dispatched`), and one released leaves both (`watch`), its communicators and load never made, and an
    executable loaded again is charged again. What loading a program took is the most a first run measured
    on each GPU (`cost`, `ran`), `PROGRAM_FLOOR` before any."""

    def __init__(self) -> None:
        self.held = Collectives()
        self.lock = threading.Lock()
        self.cost: dict[int, int] = {}
        self.pending: dict[int, Pending] = {}
        self.settled: set[int] = set()
        self.watched: set[int] = set()
        # The executables released since, which a garbage collection may report inside any of the methods
        # below: each is taken under the lock before what it changes is read (`_forget_released`).
        self.released: collections.deque[int] = collections.deque()

    def watch(self, key: int, executable: jax.stages.Compiled) -> None:
        """Forget the program `key` names once `executable` is released (`released`), which loads nothing
        more and whose communicators an executable not run never makes."""
        with self.lock:
            self._forget_released()
            if key in self.watched:
                return
            self.watched.add(key)
        weakref.finalize(executable, self.released.append, key)

    def _forget_released(self) -> None:
        """Drop what this process held of every executable released since (`watch`), under the lock: a key
        an executable released frees is no other's until it is dropped here, before any is admitted."""
        while self.released:
            key = self.released.popleft()
            self.pending.pop(key, None)
            self.settled.discard(key)
            self.watched.discard(key)

    def ahead(self, gpus: tuple[int, ...], key: int | None) -> tuple[Collectives, dict[int, int]]:
        """What every other program admitted on `gpus` that has not run makes as it first runs, which no
        reading of free memory counts yet: their communicators together, and on each GPU their loads, with
        the program `key` names's own where it has not run (none of a program of no identity); each load the
        most a first run took there (`cost`), `PROGRAM_FLOOR` before any."""
        others = [pending for held, pending in self.pending.items()
                  if held != key and set(pending.gpus) & set(gpus)]
        made = functools.reduce(operator.or_, (pending.made for pending in others), Collectives())
        own = key is not None and key not in self.settled
        return made, {gpu: (own + sum(gpu in pending.gpus for pending in others))
                      * self.cost.get(gpu, PROGRAM_FLOOR) for gpu in gpus}

    def dispatched(self, key: int) -> None:
        """Take the program `key` names as run, its first run seen to complete, its load unmeasured: its
        communicators made (`held`), its load what free memory reads from here on."""
        self.ran(key, {}, {}, {})

    def ran(self, key: int, before: Mapping[int, int], after: Mapping[int, int], grown: Mapping[int, int]
            ) -> None:
        """Take the first run of the program `key` names, seen to complete: its communicators made (`held`),
        and on each of its GPUs whose bytes free outside XLA's pool `before` and `after` it are given, what
        the GPU lost over it less the communicators its admission charged (`Pending.charged`) and the pool's
        growth (`grown`, signed: a pool that gave regions back grew by less than nothing, which the GPU's
        free bytes gained) is what loading it took, the most of which a program not yet run is charged
        (`ahead`). A program taken as run before takes nothing again."""
        if key not in self.pending:
            return
        with self.lock:
            self._forget_released()
            pending = self.pending.pop(key, None)
            self.settled.add(key)
            if pending is None:
                return
            self.held = self.held | pending.made
            for gpu in pending.gpus:
                if gpu in before and gpu in after:
                    took = before[gpu] - after[gpu] - pending.charged.get(gpu, 0) - grown.get(gpu, 0)
                    self.cost[gpu] = max(self.cost.get(gpu, PROGRAM_FLOOR), took)

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
        ones all made. A program (`loaded`) loads as it first runs, which free memory then counts; its code as
        XLA states it is no measure of what that takes (a GPU step's states its host code alone, 2 to 4 KB,
        row 12131's probe), so a program not yet run is charged what a first run measured loading one took,
        beside the communicators and load of every other admitted on its GPUs that has not run (`ahead`),
        which what is free now has not counted either; a program of no `loaded` is charged its communicators
        alone. Each refusal names the comparison that failed, before
        the program runs on any of its GPUs."""
        with self.lock:
            self._forget_released()
            gpus = () if loaded is None else loaded.gpus
            waiting, loads = self.ahead(gpus, None if loaded is None else loaded.key)
            every = functools.reduce(operator.or_, (pending.made for pending in self.pending.values()),
                                     self.held)
            needed = needs.needed(every | made, room, runtime=True)
            # Its communicators and those of every program not yet run on its GPUs that no program that ran
            # made, each counted once, beside their loads: on its own GPUs, which it alone loads on.
            communicating = needs.needed(waiting | made, room, runtime=False, made=self.held)
            if loaded is not None:
                communicating = {gpu: bytes_ for gpu, bytes_ in communicating.items() if gpu in loaded.gpus}
            added = {gpu: communicating.get(gpu, 0) + loads.get(gpu, 0) for gpu in {*communicating, *loads}}
            now = {gpu: answered for gpu in room if (answered := free(gpu)) is not None}
            new = made.beyond(self.held)
            adds = (f"the compiled program's collectives add {len(new.groups)} collective groups, "
                    f"{len(new.exchanges)} exchanges and {sum(map(len, new.permutes.values()))} permute "
                    "connections to the process's")
            if any(needed[gpu] > room[gpu] for gpu in room):
                raise CommunicatorRoom(f"{adds}, which with the runtime's need more room than Dew left "
                                       f"outside XLA's pool; lower {fraction_name()} by the shortfall over "
                                       "the GPU's bytes", needed, room)
            if any(added.get(gpu, 0) > now[gpu] for gpu in now):
                raise CommunicatorRoom(f"{adds}, which with its load and the communicators and load of every "
                                       "program admitted that has not run need more than its GPUs have free "
                                       "outside XLA's pool now", {gpu: added.get(gpu, 0) for gpu in now},
                                       {gpu: now[gpu] for gpu in now})
            if loaded is None:
                # A program of no identity, which no run of it is taken by, is held as made.
                self.held = self.held | made
            elif loaded.key not in self.settled:
                own = needs.needed(made, room, runtime=False, made=self.held)
                self.pending.setdefault(loaded.key, Pending(loaded.gpus, made, {
                    gpu: own.get(gpu, 0) for gpu in loaded.gpus}))


@dataclasses.dataclass(frozen=True)
class Loaded:
    """A program as its GPUs load it outside XLA's pool from its first run on: the GPUs it runs on and its
    executable's identity while it lives (`Resident.watch`), by which its first run is taken
    (`Resident.ran`, `Resident.dispatched`)."""

    gpus: tuple[int, ...]
    key: int


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
    default) or whose backend has opened (`opened`), a pool of no fixed share (`dew._devices.pool_bound`), or
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
    write_share(share, "sized")
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


def collectives_kept(fingerprint: bytes, gpus: tuple[int, ...],
                     program: Callable[[], str | jax.stages.Compiled]) -> Collectives:
    """The communicators an executable makes on `gpus` (`collectives_of`), read from its `program` once per
    executable and kept beside the compilation cache it lives in (`dew._cache.kept_beside`) under its
    `fingerprint` as positions in its device assignment, below the digest of the code that reads it
    (`dew._code.code_digest`), so a warm start that loads it from that cache reads no module, a process naming
    another cache reads none kept for another's programs, and those another code of it kept are read again.
    Where nothing is kept (a bucket's cache, a read-only home: `dew._cache.keep`), the program's text serves
    the process."""
    directory = kept_beside("dew-communicators")
    path = None if directory is None else directory / code_digest(__name__) / f"{fingerprint.hex()}.json"
    found = None if path is None else read_kept(path)
    if isinstance(found, dict):
        # A record kept that does not read as positions in this assignment (a position past it or no
        # integer, a field missing) is a miss, read again from the program.
        try:
            return Collectives.of(found, gpus)
        except (IndexError, KeyError, TypeError, ValueError) as unread:
            _log.info("a program's kept communicators are read again from its text: %s", unread)
    positions = collectives_of(program(), tuple(range(len(gpus)))).to_json()
    if path is not None:
        keep(path, positions, "a program's communicators")
    return Collectives.of(positions, gpus)


@dataclasses.dataclass(frozen=True)
class LoadedProgram:
    """A compiled program as this process loaded it (`loaded_program`): the devices it runs on and its
    runtime's fingerprint, None where the runtime states none."""

    devices: tuple
    """The devices it runs on (JAX's `Device`s)."""
    fingerprint: bytes | None


def loaded_program(executable: jax.stages.Compiled) -> LoadedProgram | None:
    """`executable`'s program as this process loaded it (its runtime a `LoadedExecutable`), None for one
    compiled for devices it has not (`dew.nn.kernels.generation.compile_only_gpus`), which is never loaded
    here: no devices, fingerprint or room of this process's. The one reading of it `admit`, a kept program
    (`dew.training.ahead`) and a step's flops (`dew.training.rungs`) take."""
    runtime = executable.runtime_executable()
    if runtime is None or not isinstance(runtime, LoadedExecutable):
        return None
    fingerprint = runtime.fingerprint
    return LoadedProgram(tuple(runtime.local_devices()),
                         fingerprint if isinstance(fingerprint, bytes) else None)


def admit(executable: jax.stages.Compiled, free: Callable[[int], int | None] = gpu_free_bytes) -> None:
    """Admit a compiled or loaded program's communicators and load against the room the process's sizing left
    them and its GPUs have free outside XLA's pool now (`Resident`), refusing one that would not fit before it
    runs on any of its GPUs. A process that holds no sizing admits every program, and so does every process
    of a pool of processes, whose communicators span hosts no process's sizing measures
    (`dew.training.runtime.pooled` sizes none of them): its programs are admitted on every process or on
    none, never refused on one while its peer goes on into a collective. A program compiled for devices this
    process has not (`dew.nn.kernels.generation.compile_only_gpus`) is never loaded here, so it takes none of
    this host's room, as does a program on the host's CPU. Every program Dew compiles passes here
    (`dew.nn.kernels.templates.report`), the RL rollouts' among them (`dew.nn.kernels.options.OuterProgram`),
    and every executable it loads compiled ahead (`dew.training.ahead`); each of a fingerprint is charged as
    not run until a call of it returns (`dispatched`) or it is released (`Resident.watch`), one of none by
    its communicators alone."""
    program = loaded_program(executable)
    # A pool of processes admits every program on every rank alike: no rank holds a sizing there, and a
    # verdict agreed across ranks needs a round keyed by the program, not by its order (row 12351).
    if _HELD is None or program is None or jax.process_count() > 1:
        return
    devices = program.devices
    if not devices or any(device.platform != "gpu" for device in devices):
        # A program on the host's CPU makes no NCCL communicator and holds no GPU code.
        return
    gpus = assigned(devices)
    if program.fingerprint is None:
        _RESIDENT.admit(collectives_of(executable, gpus), _HELD.needs, _HELD.room, free)
        return
    made = collectives_kept(program.fingerprint, gpus, lambda: executable)
    _RESIDENT.watch(id(executable), executable)
    _RESIDENT.admit(made, _HELD.needs, _HELD.room, free, Loaded(gpus, id(executable)))


def completed(executable: jax.stages.Compiled, outputs) -> None:
    """Take `executable`'s first run as run once it is seen to complete (`Resident.dispatched`): its
    outputs, the call's, waited for where it is admitted and not run, never its dispatch alone, which an
    asynchronous one returns before the run loads the program and makes its communicators
    (`PJRT_GPU_ENABLE_ASYNC_DISPATCH`); a run that fails raises here, leaving it charged as not run. Called
    past each call through Dew's compile point (`dew.nn.kernels.options.OuterProgram`)."""
    if _HELD is None or id(executable) not in _RESIDENT.pending:
        return
    jax.block_until_ready(outputs)
    _RESIDENT.dispatched(id(executable))


def first_ran(executable: jax.stages.Compiled, before: Mapping[int, int], after: Mapping[int, int],
              grown: Mapping[int, int]) -> None:
    """Take `executable`'s first run, seen to complete (`Resident.ran`): each of its GPUs' bytes free outside
    XLA's pool `before` it and `after`, and its pool's growth over it (`grown`), by GPU ordinal (`ordinal`).
    A trainer's newly compiled step's (`dew.training.memory.first_run`)."""
    if _HELD is not None:
        _RESIDENT.ran(id(executable), before, after, grown)


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


def measure(selected: str) -> dict:
    """The child's measurement (`_child`), over the GPUs `selected` names (JAX's own setting): each GPU's
    free bytes before the runtime starts; then, for each group size
    that divides this host's GPUs (the first GPUs of that many), the bytes each GPU gave, less its pool's
    growth, as a collective's communicator is made over them (an all-gather's), as an all-to-all connects
    every peer beside it, each the most a member gave, and as a permute's communicator connects each
    directed pair of them in turn (`peer_costs`); and the rest, after a matrix product and a convolution on
    each GPU (cuBLAS's and cuDNN's handles and workspaces), the runtime's beside its pool."""
    import jax.numpy as jnp
    import numpy as np

    # The GPUs its parent's JAX selects (`_child`), selected before the backend opens.
    jax.config.update("jax_cuda_visible_devices", selected or _pool.ALL_GPUS)
    before = {index: gpu_free_bytes(index) or 0 for index in local_gpus() or ()}
    devices = jax.local_devices()

    def held_now() -> dict[int, tuple[int, int]]:
        return {device.local_hardware_id: (gpu_free_bytes(device.local_hardware_id) or 0,
                                           int((device.memory_stats() or {}).get("pool_bytes", 0)))
                for device in devices}

    spent = dict.fromkeys((device.local_hardware_id for device in devices), 0)

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
    runtime = {gpu: before[gpu] - after[gpu][0] - after[gpu][1] - spent[gpu] for gpu in before}
    return {"collective": collective, "exchanged": exchanged, "permuted": permuted,
            "connections": [[*pair, *gave] for pair, gave in connections.items()], "runtime": runtime}
