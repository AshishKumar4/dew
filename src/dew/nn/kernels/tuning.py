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
overlays, then Dew's picks, `KernelOp.get_autotuning_cache`), and its own
shapes' heuristic, the first two only where the space holds them; Dew
resolves the config itself, so no tokamax policy measures or refuses inside
a trace. Picks are measured once, where a program is lowered before its
compile (`tune`: the Trainer's step, the serving engine's), on the launches
tokamax finds in the lowered program's metadata. Every config of a missing
launch compiles on a thread pool and runs alone, its kernel's device time
read by CUPTI after `FLUSH_BYTES` are read, and the fastest is kept. A
pool's process 0 decides every launch's pick, stored or measured, and the
others take exactly those, so every process compiles the same program.
Picks persist per GPU model and jaxlib beside XLA's own
(`dew.cache.kernel_picks_path`), and only a process that has its GPU to itself
measures (`dew.cache.measures_picks`); a CPU, an interpreted kernel or a shared
GPU takes stored picks and heuristics. `samples` is the wall-clock timing
every benchmark tool in Dew measures with.

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


_PICKS: dict[tuple[tokamax.Op, str], _Picks] = {}
_LOADED: set[str] = set()
_LOADING = threading.Lock()


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
        if self.config is not None:
            return tokamax.Op.__call__(self, *args, **kwargs)
        ba = self.bind(*args, **kwargs)
        keyed = self.keyed(ba)
        pin = dict(cache.KERNEL_PINS.value).get(type(self))
        pick = _pick(keyed)
        config = (pin if pin in keyed.autotuning_configs
                  else ba.heuristics_config if pick is None else pick.fastest_config)
        return tokamax.Op.__call__(self.replace(config=config), *args, **kwargs)

    def keyed(self, ba: tokamax.BoundArguments) -> tokamax.BoundArguments:
        """`ba` with its token-counted arguments at its key's shapes, where its
        space, heuristic and measurement are read: every shape a key covers
        shares one space, so a pick holds for each."""
        return tokamax.BoundArguments(self.replace(config=None, vjp=None), {
            name: _bucketed(jax.ShapeDtypeStruct(value.shape, value.dtype))
            if name in self.ROWS and value is not None else value for name, value in ba.arguments.items()})

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
            return _PICKS.setdefault((self.replace(config=None, vjp=None), kind), _Picks())

    def _get_autotuning_cache_key(self, ba: tokamax.BoundArguments) -> Mapping[str, object]:
        key = super()._get_autotuning_cache_key(ba)
        return type(key)({name: _bucketed(spec) if name in self.ROWS and spec is not None else spec
                          for name, spec in key.items()})

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        """Arrays at `specs` to time the launch on: normal floats, zero integers."""
        keys = jax.random.split(jax.random.key(seed), len(specs))
        return [jax.random.normal(key, spec.shape, spec.dtype) if jnp.issubdtype(spec.dtype, jnp.floating)
                else jnp.zeros(spec.shape, spec.dtype) for key, spec in zip(keys, specs, strict=True)]


def _kind(ba: tokamax.BoundArguments) -> str:
    """The GPU model `ba`'s launch is chosen for, as tokamax reads it: its
    arrays', else the context mesh's (`Trainer.lower_for`'s abstract GPU),
    else the default device's."""
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
    # One CUPTI subscription reads every run; the flush's own read, last,
    # detaches it, so a profiler can subscribe after.
    cupti = profiler.Cupti(finalize=False)
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
        flushes = {name for name, _ in _kernel_times(profiler.Cupti(), evicting, flush)}
    for config, (runs, lowering, compiling) in read.items():
        times = tuple(sum(milliseconds for name, milliseconds in run if name not in flushes) for run in runs)
        timings[config] = tokamax.BenchmarkData(compile_time_ms=compiling, lower_time_ms=lowering,
                                                metadata={}, evaluation_times_ms=times)
    return AutotuningData(timings)


def _config(fastest: AutotuningData | None) -> Hashable:
    return None if fastest is None else fastest.fastest_config


def _kernel_launches(picks: tokamax.AutotuningResult):
    """`picks`' launches of Dew's kernels, each with its fastest config's timing."""
    for ba, timings in picks.data:
        if isinstance(ba.op, KernelOp) and timings.prune_errors():
            yield ba, timings.prune()


def _stored_picks(kind: str) -> dict[tuple[tokamax.Op, str], _Picks]:
    """The picks stored for GPUs of `kind`, by op. A store that no longer
    parses (a kernel left Dew) is measured anew."""
    path = cache.kernel_picks_path(kind)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as file:
            stored = tokamax.AutotuningResult.load(file)
    except ValueError as error:
        _log.warning("the kernel picks at %s no longer parse, so they are measured anew: %s", path, error)
        return {}
    picks: dict[tuple[tokamax.Op, str], _Picks] = {}
    for ba, fastest in _kernel_launches(stored):
        bare = ba.op.replace(config=None, vjp=None)
        picks.setdefault((bare, kind), _Picks())[ba.autotuning_cache_key] = fastest
    return picks


def _store(picks: tokamax.AutotuningResult) -> None:
    """Merge `picks` into the stored picks for their GPU model, one writer at a
    time, a launch's newer timings replacing its older (`write_atomically`)."""
    path = cache.kernel_picks_path(picks.device_kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with filelock.FileLock(path + '.lock'):
        launches = []
        if os.path.exists(path):
            with open(path) as file, contextlib.suppress(ValueError):
                launches = list(tokamax.AutotuningResult.load(file).data)
        merged = {(ba.op, ba.autotuning_cache_key): (ba, timings) for ba, timings in (*launches, *picks.data)}
        write_atomically(path, tokamax.AutotuningResult(picks.device_kind, tuple(merged.values()))
                         .dumps(prune_errors=True))


def tune(lowered: jax.stages.Lowered) -> bool:
    """Settle every launch's config in the `lowered` program across the pool:
    process 0 takes its stored pick, or measures one where it has its GPU to
    itself (storing it), and every process takes exactly process 0's picks
    for the program's launches. True on every process when any process's
    picks changed, so the caller lowers again to compile with them."""
    if not generation.triton_runs():
        return False
    found = [(op, op.keyed(ba)) for ba in tokamax.autotuning.get_bound_args(lowered)
             if isinstance(op := ba.op, KernelOp)]
    if not found:
        return False
    kind = _kind(found[0][1])

    def decided() -> str | None:
        if jax.process_index() != 0:
            return None
        decisions, measured = [], []
        for op, ba in found:
            if (fastest := _pick(ba)) is not None:
                decisions.append((ba, fastest))
            elif cache.measures_picks():
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
    settled = {(ba.op, ba.autotuning_cache_key): fastest
               for ba, fastest in _kernel_launches(tokamax.AutotuningResult.loads(text))}
    changed = False
    for op, ba in found:
        picks, key = op.get_autotuning_cache(kind), ba.autotuning_cache_key
        fastest = settled.get((ba.op, key))
        if _config(_pick(ba)) != _config(fastest):
            changed = True
            picks.pop(key, None)
            if fastest is not None:
                picks[key] = fastest
    if not any(coordination.from_every_process(changed)):
        return False
    cache.KERNEL_TUNES.set_global(cache.KERNEL_TUNES.get_global() + 1)
    return True


@contextlib.contextmanager
def pinned(configs: Mapping[type[KernelOp], Hashable]) -> Iterator[None]:
    """Each launch of a kernel `configs` names, traced within, takes that
    config where its space at the launch's key holds it, any stored pick
    or heuristic set aside: for the float64 gate at every config of a space,
    and for a lab that times one."""
    with cache.KERNEL_PINS(tuple(configs.items())):
        yield
