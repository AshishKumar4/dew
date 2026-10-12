"""Each Pallas launch's blocks, warps and stages, measured per GPU model and
shape, and the timing every benchmark in Dew measures with.

A launch is a `KernelOp`, tokamax's `Op` (tokamax/_src/ops/op.py) for one
`pallas_call` on primal arrays. It declares its config dataclass, the space it
measures (`_get_autotuning_configs`), the rule it takes without a measurement
(`_get_heuristics_config`) and the arguments whose leading axis counts tokens
(`ROWS`), bucketed to a power of two in its key, as fla keys `triton.autotune`
on its heads and widths. The kernel's own `custom_vjp`, shard placement and
remat names stay around it; the op is never differentiated.

At trace time a launch takes, in order, a config pinned for its kernel
(`pinned`), the pick measured for its key on this GPU model, and its
heuristic, each only where the launch's declared space holds it; Dew
resolves the config itself, so no tokamax policy measures or refuses inside
a trace. Picks are measured once, where a program is lowered before its
compile (`tune`: the Trainer's step, the serving engine's), on the launches
tokamax finds in the lowered program's metadata. Every config of a missing
launch compiles on a thread pool and runs alone, timed by `samples` over
operand sets rotating through `ROTATION_BYTES`, and the fastest is kept; a
launch too small to fill the rotation would be timed on cache hits, so it
keeps its heuristic. A pool's process 0 decides every launch's pick, stored
or measured, and the others take exactly those, so every process compiles
the same program. Picks persist per GPU model and jaxlib beside XLA's own
(`dew.cache.kernel_picks_path`), and only a process that has its GPU to itself
measures (`dew.cache.measures_picks`); a CPU, an interpreted kernel or a shared
GPU takes stored picks and heuristics.

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
import functools
import logging
import os
import time
from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import ClassVar

import filelock
import jax
import jax.numpy as jnp
import tokamax

# tokamax names the type of one launch's timings by config nowhere public; its picks are read as this type.
from tokamax._src.autotuning.autotuner import AutotuningData

from dew import cache, coordination
from dew.files import write_atomically

from . import generation

_log = logging.getLogger(__name__)

ROTATION_BYTES = 256 << 20
"""The operand bytes a measurement cycles through: twice the largest L2 among
Dew's GPUs (an RTX PRO 6000's 128 MB), so no call finds the last one's
operands in cache, as no training step does."""

ROTATION_SETS = 16
"""The most operand sets one timed program holds. A launch whose sets cannot
fill `ROTATION_BYTES` at this count keeps its heuristic."""

_PICKS: dict[tuple[tokamax.Op, str], dict[Hashable, AutotuningData]] = {}
_LOADED: set[str] = set()


@functools.cache
def _state():
    """`(tunes, pins)`: how many times `tune` changed this process's picks,
    and the pinned `(kernel, config)` pairs, as a context JAX keys its trace
    and compile caches on, so a program lowered again after picks change, or
    under other pins, is traced again. Made at the first launch, pin or tune,
    not at import."""
    return jax.make_user_context((0, ()))


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
        space = ba.autotuning_configs
        pin = dict(_state().value[1]).get(type(self))
        picked = _picks(self, jax.devices()[0].device_kind).get(ba.autotuning_cache_key)
        stored = None if picked is None else picked.fastest_config
        config = next((choice for choice in (pin, stored) if choice in space), ba.heuristics_config)
        return tokamax.Op.__call__(self.replace(config=config), *args, **kwargs)

    def _get_autotuning_cache_key(self, ba: tokamax.BoundArguments) -> Mapping[str, object]:
        key = super()._get_autotuning_cache_key(ba)
        return type(key)({name: _bucketed(spec) if name in self.ROWS and spec is not None else spec
                          for name, spec in key.items()})

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        """Arrays at `specs` to time the launch on: normal floats, zero integers."""
        keys = jax.random.split(jax.random.key(seed), len(specs))
        return [jax.random.normal(key, spec.shape, spec.dtype) if jnp.issubdtype(spec.dtype, jnp.floating)
                else jnp.zeros(spec.shape, spec.dtype) for key, spec in zip(keys, specs, strict=True)]


def _picks(op: tokamax.Op, kind: str) -> dict[Hashable, AutotuningData]:
    """`op`'s picks on GPUs of `kind` by key, each its fastest config's
    timing, the store read in at the first lookup."""
    _load(kind)
    return _PICKS.setdefault((op.replace(config=None, vjp=None), kind), {})


def _bucketed(spec: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
    """`spec` with its leading (token) axis at the next power of two."""
    return jax.ShapeDtypeStruct((1 << max(spec.shape[0] - 1, 0).bit_length(), *spec.shape[1:]), spec.dtype)


def _measured(op: KernelOp, ba: tokamax.BoundArguments) -> AutotuningData:
    """Every config of `op`'s launch at `ba`, each compiled on a thread and
    timed alone over operand sets that rotate through `ROTATION_BYTES`; the
    heuristic alone where the sets cannot fill it."""
    leaves, tree = jax.tree.flatten((ba.args, ba.kwargs),
                                    is_leaf=lambda leaf: isinstance(leaf, jax.ShapeDtypeStruct))
    places = [index for index, leaf in enumerate(leaves) if isinstance(leaf, jax.ShapeDtypeStruct)]
    specs = [leaves[index] for index in places]
    held = max(1, sum(spec.size * spec.dtype.itemsize for spec in specs))
    rotation = min(ROTATION_SETS, -(-ROTATION_BYTES // held))
    fills = held * ROTATION_SETS >= ROTATION_BYTES
    configs = list(ba.autotuning_configs) if fills else [ba.heuristics_config]
    sets = [op.operands(specs, seed) for seed in range(rotation)]

    def compiled(config):
        configured = op.replace(config=config)

        def run(sets):
            outputs = []
            for arrays in sets:
                filled = list(leaves)
                for index, array in zip(places, arrays, strict=True):
                    filled[index] = array
                args, kwargs = jax.tree.unflatten(tree, filled)
                outputs.append(configured(*args, **kwargs))
            return outputs

        start = time.perf_counter()
        lowered = jax.jit(run).lower(sets)
        middle = time.perf_counter()
        executable = lowered.compile()
        return executable, (middle - start) * 1e3, (time.perf_counter() - middle) * 1e3

    timings: dict[Hashable, tokamax.BenchmarkData | Exception] = {}
    with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as pool:
        futures = {config: pool.submit(compiled, config) for config in configs}
    for config, future in futures.items():
        try:
            executable, lowering, compiling = future.result()
            times = samples(executable, sets, warmup=2, repeats=5)
        except Exception as error:  # a config past the GPU's shared memory or registers
            timings[config] = error
            continue
        timings[config] = tokamax.BenchmarkData(
            compile_time_ms=compiling, lower_time_ms=lowering, metadata={},
            evaluation_times_ms=tuple(seconds * 1e3 / len(sets) for seconds in times))
    return AutotuningData(timings)


def _config(fastest: AutotuningData | None) -> Hashable:
    return None if fastest is None else fastest.fastest_config


def _kernel_launches(picks: tokamax.AutotuningResult):
    """`picks`' launches of Dew's kernels, each with its fastest config's timing."""
    for ba, timings in picks.data:
        if isinstance(ba.op, KernelOp) and timings.prune_errors():
            yield ba, timings.prune()


def _load(kind: str) -> None:
    """Read the picks stored for GPUs of `kind`, once a process, at its first
    lookup: no program traced before them, so none is traced again. A store
    that no longer parses (a kernel left Dew) is measured anew."""
    if kind in _LOADED:
        return
    _LOADED.add(kind)
    path = cache.kernel_picks_path(kind)
    if not os.path.exists(path):
        return
    try:
        with open(path) as file:
            stored = tokamax.AutotuningResult.load(file)
    except ValueError as error:
        _log.warning("the kernel picks at %s no longer parse, so they are measured anew: %s", path, error)
        return
    for ba, fastest in _kernel_launches(stored):
        _picks(ba.op, kind)[ba.autotuning_cache_key] = fastest


def _stored(picks: tokamax.AutotuningResult) -> None:
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


def tune(jitted: jax.stages.Wrapped, *args) -> bool:
    """Settle every launch's config in `jitted(*args)` across the pool: process
    0 takes its stored pick, or measures one where it has its GPU to itself
    (storing it), and every process takes exactly process 0's picks for the
    program's launches. True on every process when any process's picks
    changed, so the caller lowers again to compile with them. The program is
    lowered as the caller lowers it, which JAX's cache answers."""
    if not generation.triton_runs():
        return False
    kind = jax.devices()[0].device_kind
    state = _state()
    found = [(ba, ba.op) for ba in tokamax.autotuning.get_bound_args(jitted, *args)
             if isinstance(ba.op, KernelOp)]

    def decided() -> str | None:
        if jax.process_index() != 0:
            return None
        decisions, measured = [], []
        for ba, op in found:
            if (fastest := _picks(op, kind).get(ba.autotuning_cache_key)) is not None:
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
            _stored(tokamax.AutotuningResult(kind, tuple(measured)))
        return tokamax.AutotuningResult(kind, tuple(decisions)).dumps()

    text = coordination.broadcast_from_process_zero(coordination.agreed("settling kernel configs", decided))
    settled = {(ba.op, ba.autotuning_cache_key): fastest
               for ba, fastest in _kernel_launches(tokamax.AutotuningResult.loads(text))}
    changed = False
    for ba, op in found:
        picks, key = _picks(op, kind), ba.autotuning_cache_key
        fastest = settled.get((op, key))
        held = picks.get(key)
        if _config(held) != _config(fastest):
            changed = True
            if fastest is None:
                del picks[key]
            else:
                picks[key] = fastest
    if not any(coordination.from_every_process(changed)):
        return False
    tunes, pins = state.get_global()
    state.set_global((tunes + 1, pins))
    return True


@contextlib.contextmanager
def pinned(configs: Mapping[type[KernelOp], Hashable]) -> Iterator[None]:
    """Each launch of a kernel `configs` names, traced within, takes that
    config where its space at the launch's shapes holds it, any stored pick
    or heuristic set aside: for the float64 gate at every config of a space,
    and for a lab that times one."""
    state = _state()
    tunes, _ = state.value
    with state((tunes, tuple(configs.items()))):
        yield
