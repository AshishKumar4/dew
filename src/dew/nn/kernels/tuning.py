"""Each Pallas launch's blocks, warps and stages, measured per GPU model and
shape, and the timing every benchmark in Dew measures with.

A launch is a `KernelOp`, tokamax's `Op` (tokamax/_src/ops/op.py) for one
`pallas_call` on primal arrays. It declares its config dataclass, the space it
measures (`_get_autotuning_configs`), the rule it takes without a measurement
(`_get_heuristics_config`) and the arguments whose leading axis counts tokens
(`ROWS`), bucketed to a power of two in its key, as fla keys `triton.autotune`
on its heads and widths. The kernel's own `custom_vjp`, shard placement and
remat names stay around it; the op is never differentiated.

A launch's space, heuristic and measurement are read at its key's shapes
(`KernelOp.keyed`), so one pick serves every shape the key covers. At trace
time a launch takes, in order, a config pinned for its kernel (`pinned`),
the pick tokamax's cache holds for its key on its target GPU model (its
overlays, then Dew's picks, `KernelOp.get_autotuning_cache`), the pick
measured at the nearest stored token count of the same launch, and its own
shapes' heuristic, the picks only where the space holds them; Dew resolves
the config itself, so no tokamax policy measures or refuses inside a trace.

Before a program compiles, `tune` reads its launches from a lowering that
describes them (`launches`), and the pool's process 0 decides every launch's
pick, which the others take exactly, so every process compiles the same
program. Only an explicit pass measures (`tune(..., measure=True)`, which
`Trainer(tune_kernels=True)` and tools/kernel_picks.py call), on a GPU the
process has to itself (`dew.cache.measures_picks`): every config of a launch
without a pick compiles on a thread pool and runs alone, its kernel's device
time read by CUPTI after `FLUSH_BYTES` are read, and the fastest is kept.
Any other compile measures nothing and says once how many launches lack a
measurement. Picks persist per GPU model and jaxlib beside XLA's own
(`dew.cache.kernel_picks_path`). `samples` is the wall-clock timing every
benchmark tool in Dew measures with.

Tile choice changes a kernel's reduction order, and its error is not
monotone in tiles, warps or programs, so the space is a declaration of
proven configs: each kernel's GPU tests hold its float64 gate at every config
of its space (`pinned`, tests/tuned.py's `each_config`), and a config that
fails leaves the space. The tuner only times what is proven.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import ClassVar

import filelock
import jax
import jax.numpy as jnp
import tokamax
from jax.extend.backend import get_default_device

# tokamax's writer of launches as JSON, public only as a file writer (`bound_args_to_json`).
from tokamax._src.autotuning.api import dump_bound_args_to_json

# tokamax names the type of one launch's timings by config nowhere public; its picks are read as this type.
from tokamax._src.autotuning.autotuner import AutotuningData

# tokamax's own reading of a launch's target GPU model, which its cache lookups key on, public nowhere.
from tokamax._src.ops.op import infer_device_kind

from dew import cache, coordination
from dew.files import write_atomically

from . import generation

_log = logging.getLogger(__name__)

FLUSH_BYTES = 256 << 20
"""The bytes read before each timed launch: twice the largest L2 among Dew's
GPUs (an RTX PRO 6000's 128 MB), so no launch finds its operands in cache,
as none does in a training step."""

REPEATS = 5
"""The timed launches of each config."""


class _Picks(dict):
    """One op's picks on one GPU model, by key; a key without one reads None,
    tokamax's miss."""

    def __missing__(self, key):
        return None


# tokamax's `get_bound_args` for a lowered program: its annotation's ParamSpec,
# for a callable's arguments, is one pyright cannot solve without any.
_bound_args: Callable[[jax.stages.Lowered], tuple[tokamax.BoundArguments, ...]] = (
    tokamax.autotuning.get_bound_args)

_PICKS: dict[tuple[tokamax.Op, str], _Picks] = {}
_LOADED: set[str] = set()
_LOADING = threading.Lock()
_told: set[str] = set()


def samples[T](fn: Callable[..., T], *args, warmup: int = 3, repeats: int = 10,
               calls: int = 1) -> list[float]:
    """Seconds per call of `fn(*args)`, one sample a synchronized group of
    `calls` calls, after `warmup` groups. With one call a group each sample is
    a call's latency, and its result is released before the next call starts,
    so no two calls' outputs are ever held at once; more calls a group queue
    their dispatches and synchronize once, which measures throughput. The
    caller takes the statistic and units it reports."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    measured = []
    for _ in range(repeats):
        start = time.perf_counter()
        outputs = fn(*args)
        for _ in range(calls - 1):
            outputs = fn(*args)
        jax.block_until_ready(outputs)
        measured.append((time.perf_counter() - start) / calls)
        del outputs
    return measured


@dataclasses.dataclass(frozen=True)
class KernelOp(tokamax.Op):
    """One Pallas launch whose config is measured per GPU model and shape.

    A subclass declares `config_cls` (a frozen dataclass), `_fwd(..., config,
    return_residuals)` returning `(outputs, None)`, `_get_autotuning_configs`
    and `_get_heuristics_config`, and names its token-counted arguments in
    `ROWS`. `revision` is part of its stored picks' identity: a kernel whose
    speed changes raises it, and picks measured for the old revision stop
    matching. A kernel whose inputs must hold more than random values to run
    (group sizes, indices) overrides `operands`.
    """

    ROWS: ClassVar[tuple[str, ...]] = ()
    supports_batched_args_capture: ClassVar[bool] = False
    supports_symbolic_shapes: ClassVar[bool] = False
    revision: int = 1

    def __call__(self, *args, **kwargs):
        config = self.config
        if config is None:
            ba = self.bind(*args, **kwargs)
            keyed = self.keyed(ba)
            pin = dict(cache.KERNEL_PINS.value).get(type(self))
            pick = _pick(keyed) or self._nearest(keyed)
            config = (pin if pin in keyed.autotuning_configs
                      else ba.heuristics_config if pick is None else pick.fastest_config)
        if cache.KERNEL_LAUNCHES.value:
            # Dew chose the launch for this GPU (`generation`); tokamax's device check would ask
            # each traced array for its devices (`keyed`).
            return tokamax.Op.__call__(self.replace(config=config, bypass_device_check=True), *args, **kwargs)
        return self._fwd(*args, config=config, **kwargs)[0]

    def keyed(self, ba: tokamax.BoundArguments) -> tokamax.BoundArguments:
        """`ba` as its key reads it, where its space, heuristic and measurement
        are read: its arrays as their shapes and dtypes, as tokamax keys them,
        the token-counted ones at their key's shapes, so a pick holds for every
        shape a key covers, and the op as its picks name it (`_identity`).
        A launch's GPU model is then read from the context, never from a
        traced array: asking one for its devices raises an error that walks
        the whole trace to say where the array came from, which was 111 s of
        the packed Qwen3.5-0.8B step's 330 s compile on an A100 (ledger 554)."""
        def shaped(name: str, value):
            if name in self.ROWS and value is not None:
                return _bucketed(jax.ShapeDtypeStruct(value.shape, value.dtype))
            return jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)
                                if isinstance(leaf, jax.Array) else leaf, value)
        return tokamax.BoundArguments(_identity(self), {name: shaped(name, value)
                                                        for name, value in ba.arguments.items()})

    def _nearest(self, ba: tokamax.BoundArguments) -> AutotuningData | None:
        """The pick measured for the stored key nearest `ba`'s: every argument
        the same but the token counts (`ROWS`), the fewest powers of two away,
        the smaller on a tie, where `ba`'s space holds its config."""
        key = ba.autotuning_cache_key
        rows = [name for name in self.ROWS if key.get(name) is not None]

        def distance(stored) -> int | None:
            if stored.keys() != key.keys():
                return None
            if any(stored[name] != key[name] for name in key if name not in rows):
                return None
            # A launch without one of the token-counted operands (a norm with no branch) is another launch.
            if any(stored[name] is None or (stored[name].shape[1:], stored[name].dtype)
                   != (key[name].shape[1:], key[name].dtype) for name in rows):
                return None
            return sum(abs(stored[name].shape[0].bit_length() - key[name].shape[0].bit_length())
                       for name in rows)

        picks = self.get_autotuning_cache(_kind(ba)).items() if rows else ()
        near = sorted((((steps, stored[rows[0]].shape[0]), timings) for stored, timings in picks
                       if (steps := distance(stored)) is not None), key=lambda entry: entry[0])
        return next((timings for _, timings in near if timings.fastest_config in ba.autotuning_configs), None)

    def get_autotuning_cache(self, device_kind: str | None = None) -> _Picks:
        """Dew's picks for this op on GPUs of `device_kind` (the default
        device's for None), the cache tokamax reads a launch's pick from; the
        store is read in at the first lookup of each GPU model."""
        kind = device_kind or get_default_device().device_kind
        with _LOADING:
            if kind not in _LOADED:
                for key, picks in _stored_picks(kind).items():
                    _PICKS.setdefault(key, _Picks()).update(picks)
                _LOADED.add(kind)
            return _PICKS.setdefault((_identity(self), kind), _Picks())

    def _get_autotuning_cache_key(self, ba: tokamax.BoundArguments) -> Mapping[str, object]:
        key = super()._get_autotuning_cache_key(ba)
        return type(key)({name: _bucketed(spec) if name in self.ROWS and spec is not None else spec
                          for name, spec in key.items()})

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        """Arrays at `specs` to time the launch on: normal floats, zero integers."""
        keys = jax.random.split(jax.random.key(seed), len(specs))
        return [jax.random.normal(key, spec.shape, spec.dtype) if jnp.issubdtype(spec.dtype, jnp.floating)
                else jnp.zeros(spec.shape, spec.dtype) for key, spec in zip(keys, specs, strict=True)]


def _identity(op: tokamax.Op) -> tokamax.Op:
    """`op` as its picks name it: without the config, the vjp and the device
    check a call sets."""
    return op.replace(config=None, vjp=None, bypass_device_check=False)


def _kind(ba: tokamax.BoundArguments) -> str:
    """The GPU model the keyed launch `ba` (`keyed`) is chosen for, as tokamax
    reads it: the context mesh's (`Trainer.lower_for`'s abstract GPU), else
    the default device's."""
    return infer_device_kind(ba) or get_default_device().device_kind


def _pick(ba: tokamax.BoundArguments) -> AutotuningData | None:
    """The pick tokamax's cache holds for `ba` (its key's shapes), where
    `ba`'s space holds its fastest config; any other is no pick."""
    stored = ba.cached_autotuning_data
    return stored if stored is not None and stored.fastest_config in ba.autotuning_configs else None


def _bucketed(spec: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
    """`spec` with its leading (token) axis at the next power of two."""
    return jax.ShapeDtypeStruct((1 << max(spec.shape[0] - 1, 0).bit_length(), *spec.shape[1:]), spec.dtype)


def _kernel_times(cupti, executable, *args) -> list[tuple[str, float]]:
    """Each kernel's name and device milliseconds over one run of
    `executable`, read by `cupti` (a mosaic `profiler.Cupti`); none where it
    launched none."""
    _, kernels = cupti.measure(executable, aggregate=False)(*args)
    return kernels if isinstance(kernels, list) else []


def _measured(op: KernelOp, ba: tokamax.BoundArguments) -> AutotuningData:
    """Every config of `ba`'s launch (at its key's shapes), each compiled on a
    thread and timed alone: the device time CUPTI reads for the launch's own
    kernels, each run after `FLUSH_BYTES` are read, so every config starts
    with its operands out of cache."""
    from jax.experimental.mosaic.gpu import profiler

    leaves, tree = jax.tree.flatten((ba.args, ba.kwargs),
                                    is_leaf=lambda leaf: isinstance(leaf, jax.ShapeDtypeStruct))
    places = [index for index, leaf in enumerate(leaves) if isinstance(leaf, jax.ShapeDtypeStruct)]
    arrays = op.operands([leaves[index] for index in places], 0)
    flush = jnp.zeros(FLUSH_BYTES // 4, jnp.float32)
    evicting = jax.jit(jnp.sum).lower(flush).compile()

    def compiled(config):
        configured = op.replace(config=config)

        def run(flush, arrays):
            # The launch waits for the flush, which reads the whole buffer through the cache.
            total, arrays = jax.lax.optimization_barrier((jnp.sum(flush), arrays))
            filled = list(leaves)
            for index, array in zip(places, arrays, strict=True):
                filled[index] = array
            args, kwargs = jax.tree.unflatten(tree, filled)
            return total, configured(*args, **kwargs)

        start = time.perf_counter()
        lowered = jax.jit(run).lower(flush, arrays)
        middle = time.perf_counter()
        executable = lowered.compile()
        return executable, (middle - start) * 1e3, (time.perf_counter() - middle) * 1e3

    timings: dict[Hashable, tokamax.BenchmarkData | Exception] = {}
    read = {}
    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as pool:
        futures = {config: pool.submit(compiled, config) for config in ba.autotuning_configs}
    # One CUPTI subscription reads every run, the flush's own first. CUPTI
    # takes one subscriber, so beside an open profile capture none is
    # measured and the launch keeps its heuristic until a later compile.
    cupti = profiler.Cupti(finalize=False)
    try:
        flushes = {name for name, _ in _kernel_times(cupti, evicting, flush)}
    except RuntimeError as error:
        return AutotuningData(dict.fromkeys(ba.autotuning_configs, error))
    try:
        for config, future in futures.items():
            try:
                executable, lowering, compiling = future.result()
                runs = [_kernel_times(cupti, executable, flush, arrays) for _ in range(REPEATS)]
            except Exception as error:  # a config past the GPU's shared memory or registers
                timings[config] = error
                continue
            read[config] = runs, lowering, compiling
    finally:
        # A finalizing read detaches the subscription, so a profiler can subscribe after.
        _kernel_times(profiler.Cupti(), evicting, flush)
    for config, (runs, lowering, compiling) in read.items():
        times = tuple(sum(milliseconds for name, milliseconds in run if name not in flushes) for run in runs)
        timings[config] = tokamax.BenchmarkData(compile_time_ms=compiling, lower_time_ms=lowering,
                                                metadata={}, evaluation_times_ms=times)
    return AutotuningData(timings)


def _config(fastest: AutotuningData | None) -> Hashable:
    return None if fastest is None else fastest.fastest_config


Keyed = dict[tuple[tokamax.Op, Hashable], tuple[tokamax.BoundArguments, AutotuningData]]
"""Launches by their op and cache key, each with its timings (`_keyed`)."""


def _keyed(timed) -> Keyed:
    """Each of the `(launch, timings)` pairs `timed` by its op and cache key, which binds the
    launch's arguments to this checkout's signature of its kernel (ValueError where they do not)."""
    return {(ba.op, ba.autotuning_cache_key): (ba, timings) for ba, timings in timed}


def _kernel_launches(launches: Keyed):
    """`launches`' launches of Dew's kernels, each `(op, key, fastest config's timing)`."""
    for (op, key), (_, timings) in launches.items():
        if isinstance(op, KernelOp) and timings.prune_errors():
            yield op, key, timings.prune()


def _read(path: str) -> tuple[Keyed, list]:
    """The launches stored at `path` that this checkout's kernels read
    (`_keyed`), and the stored entries they do not, as JSON: another
    checkout's launches (a kernel another branch adds, another branch's
    config fields, or another signature of a kernel, with an operand more or
    one fewer), which a write keeps as they are. A launch binds its stored
    arguments by name, dropping any its kernel lacks and leaving out any it
    has, so an entry whose names are not its kernel's signature's is another
    checkout's: re-keyed here, every write from a checkout that dropped an
    operand took that kernel's picks from the checkouts that still have it.
    Each launch is read alone, so one launch
    this checkout cannot read leaves the others readable. A missing file, or
    one that is not JSON, holds nothing."""
    if not os.path.exists(path):
        return {}, []
    try:
        with open(path) as file:
            stored = json.load(file)
        entries = list(stored["data"])
    except (ValueError, KeyError, TypeError) as error:
        _log.warning("the kernel picks at %s are not a store, so they are measured anew: %s", path, error)
        return {}, []
    read: Keyed = {}
    foreign = []
    for entry in entries:
        try:
            launches = _keyed(tokamax.AutotuningResult.loads(json.dumps({**stored, "data": [entry]})).data)
        except ValueError:
            foreign.append(entry)
            continue
        if any(set(entry[0]["arguments"]) != set(ba.signature.parameters) - {"config"}
               for ba, _ in launches.values()):
            foreign.append(entry)
            continue
        read.update(launches)
    return read, foreign


def _stored_picks(kind: str) -> dict[tuple[tokamax.Op, str], _Picks]:
    """The picks stored for GPUs of `kind` that this checkout reads (`_read`), by op,
    none where the process keeps no picks (`dew.cache.keeps_picks`)."""
    if not cache.keeps_picks():
        return {}
    stored, _ = _read(cache.kernel_picks_path(kind))
    picks: dict[tuple[tokamax.Op, str], _Picks] = {}
    for op, key, fastest in _kernel_launches(stored):
        picks.setdefault((_identity(op), kind), _Picks())[key] = fastest
    return picks


def _store(picks: tokamax.AutotuningResult) -> None:
    """Merge `picks` into the stored picks for their GPU model, one writer at a
    time, a launch's newer timings replacing its older, the entries this
    checkout does not read kept as they are (`_read`). The file is replaced
    whole (`write_atomically`), so no reader on the shared filesystem sees a
    part of it."""
    path = cache.kernel_picks_path(picks.device_kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with filelock.FileLock(path + '.lock'):
        stored, foreign = _read(path)
        merged = {**stored, **_keyed(picks.data)}
        written = json.loads(tokamax.AutotuningResult(picks.device_kind, tuple(merged.values()))
                             .dumps(prune_errors=True))
        write_atomically(path, json.dumps({**written, "data": [*written["data"], *foreign]}))


@contextlib.contextmanager
def launches() -> Iterator[None]:
    """Each kernel launch traced within carries tokamax's description of
    itself, which `tokamax.autotuning.get_bound_args` reads out of the
    lowered program (`dew.cache.KERNEL_LAUNCHES`)."""
    with cache.KERNEL_LAUNCHES(new_value=True):
        yield


def tune(lower: Callable[[], jax.stages.Lowered], *, measure: bool = False) -> str:
    """Settle every launch's config in the program `lower()` lowers across the
    pool, read from a lowering of its own (`launches`): process 0 takes its
    stored pick for each, with `measure` measuring and storing one for a
    launch without where it has its GPU to itself, and every process takes
    exactly process 0's picks. Returns the program's launches as JSON, which
    `settled` settles again without a lowering. Without `measure` a launch
    without a pick takes the nearest stored shape's or its heuristic, and the
    process says once how many lack one and how to measure them."""
    if not generation.triton_runs():
        return "[]"
    with launches():
        lowered = lower()
    found = [(op, op.keyed(ba)) for ba in _bound_args(lowered) if isinstance(op := ba.op, KernelOp)]
    _settle(found, measure=measure)
    return dump_bound_args_to_json([ba for _, ba in found])


def settled(launches: str) -> bool:
    """Whether every launch of `launches` (`tune`'s) has a stored pick, with
    every process taking process 0's: what a compile whose launches are
    recorded settles instead of lowering for `tune`."""
    if not generation.triton_runs():
        return True
    found = [(op, ba) for ba in tokamax.autotuning.bound_args_from_json(launches)
             if isinstance(op := ba.op, KernelOp)]
    return _settle(found, measure=False) == 0


def missing(lowered: jax.stages.Lowered) -> int:
    """How many launches of `lowered`, a program lowered under `launches`,
    have no stored pick on this process."""
    return sum(isinstance(op := ba.op, KernelOp) and _pick(op.keyed(ba)) is None
               for ba in _bound_args(lowered))


def _settle(found: list[tuple[KernelOp, tokamax.BoundArguments]], *, measure: bool) -> int:
    """Take process 0's picks for the keyed launches `found` on every process,
    process 0 measuring a launch without one where `measure` and it has its
    GPU to itself; how many still have none, which the process says once,
    with how to measure them."""
    if not found:
        return 0
    kind = _kind(found[0][1])

    def decided() -> str | None:
        if jax.process_index() != 0:
            return None
        decisions, measured = [], []
        for op, ba in found:
            if (fastest := _pick(ba)) is not None:
                decisions.append((ba, fastest))
            elif measure and cache.measures_picks():
                timings = _measured(op, ba)
                if timings.prune_errors():
                    measured.append((ba, timings))
                    decisions.append((ba, timings.prune()))
                else:
                    _log.warning("no config of %s compiled or ran; it keeps its heuristic: %s",
                                 type(op).__name__, next(iter(timings.values()), None))
        if measured:
            _store(tokamax.AutotuningResult(kind, tuple(measured)))
        return tokamax.AutotuningResult(kind, tuple(decisions)).dumps()

    text = coordination.broadcast_from_process_zero(coordination.agreed("settling kernel configs", decided))
    decided_launches = _keyed(tokamax.AutotuningResult.loads(text).data)
    settled: dict[tuple[tokamax.Op, Hashable], AutotuningData] = {
        (op, key): fastest for op, key, fastest in _kernel_launches(decided_launches)}
    changed = False
    for op, ba in found:
        picks, key = op.get_autotuning_cache(kind), ba.autotuning_cache_key
        fastest, held = settled.get((ba.op, key)), _config(_pick(ba))
        picks.pop(key, None)
        if fastest is not None:
            picks[key] = fastest
        # The pick resolution reads, which tokamax's ignore-cache setting keeps None.
        changed |= _config(_pick(ba)) != held
    if any(coordination.from_every_process(changed)):
        # A program lowered again traces again, with the picks it now reads.
        cache.KERNEL_TUNES.set_global(cache.KERNEL_TUNES.get_global() + 1)
    unmeasured = sum(_pick(ba) is None for _, ba in found)
    if unmeasured and kind not in _told:
        _told.add(kind)
        _log.warning("%d kernel launches have no config measured on %s, so they run the nearest measured "
                     "shape's or their heuristic: measure them on a GPU to yourself with "
                     "Trainer(tune_kernels=True) or `python tools/kernel_picks.py`", unmeasured, kind)
    return unmeasured


@contextlib.contextmanager
def pinned(configs: Mapping[type[KernelOp], Hashable]) -> Iterator[None]:
    """Each launch of a kernel `configs` names, traced within, takes that
    config where its space at the launch's key holds it, any stored pick
    or heuristic set aside: for the float64 gate at every config of a space,
    and for a lab that times one."""
    with cache.KERNEL_PINS(tuple(configs.items())):
        yield
