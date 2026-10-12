"""Each Pallas launch's blocks, warps and stages, measured per GPU model and
shape, and the timing every benchmark in Dew measures with.

A launch is a `KernelOp`, tokamax's `Op` (tokamax/_src/ops/op.py) for one
`pallas_call` on primal arrays. It declares its config dataclass, the space it
measures (`_get_autotuning_configs`), the rule it takes without a measurement
(`_get_heuristics_config`) and the arguments whose leading axis counts tokens
(`ROWS`), bucketed to a power of two in its key, as fla keys `triton.autotune`
on its heads and widths. The kernel's own `custom_vjp`, shard placement and
remat names stay around it; the op is never differentiated.

At trace time a launch takes, in order, a pinned end of its space (`pinned`),
the pick measured for its key on this GPU model, and its heuristic. Picks are
measured once, where a program is lowered before its compile (`tune`: the
Trainer's step, the serving engine's), on the launches tokamax finds in the
lowered program's metadata. Every config of a missing launch compiles on a
thread pool and runs alone, timed by `samples` over rotated operands that
together outgrow any L2 Dew targets, and the fastest is kept. A pool's
process 0 measures and the others take its picks, so every process compiles
the same program. Picks persist per GPU model and jaxlib beside XLA's own
(`dew.cache.kernel_picks_path`), and only a process that has its GPU to itself
measures (`dew.cache.measures_picks`); a CPU, an interpreted kernel or a shared
GPU takes stored picks and heuristics.

Tile choice changes a kernel's reduction order, so a kernel's float64 gate
holds at both ends of its space, not only at its pick: its tests run under
each of `EXTREMES` (`pinned`), as tests/conftest.py's `config_extreme` fixture
does.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import logging
import os
import tempfile
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

from . import generation

_log = logging.getLogger(__name__)

EXTREMES = ('lowest', 'highest')
"""The ends of a space in its config's field order: a kernel declares the
fields that shape its reductions first."""

ROTATION_BYTES = 256 << 20
"""The operand bytes a measurement cycles through: twice the largest L2 among
Dew's GPUs (an RTX PRO 6000's 128 MB), so no call finds the last one's
operands in cache, as no training step does."""

_STATE = jax.make_user_context((0, None))
"""(loads, pin): how many times picks arrived in this process, and the pinned
end. JAX keys its trace and compile caches on it, so a program lowered again
after picks arrive, or under another pin, is traced again."""

_PICKS: dict[tuple[tokamax.Op, str], _Picks] = {}
_LOADED: set[str] = set()


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


class _Picks(dict):
    """One op's picks on one GPU model, by key. A missing key reads as None,
    which tokamax takes as a miss (`BoundArguments.cached_autotuning_data`)
    without its warning, and is said once an op and GPU model."""

    def __init__(self, name: str, kind: str):
        super().__init__()
        self.name, self.kind, self.said = name, kind, False

    def __missing__(self, key) -> None:
        if not self.said:
            self.said = True
            _log.info("%s has no measured config for a shape on %s; it takes its heuristic until a process "
                      "with the GPU to itself lowers the step (dew.nn.kernels.tuning.tune)",
                      self.name, self.kind)


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
        _, pin = _STATE.value
        op = self
        if pin is not None and self.config is None:
            ordered = sorted(self.bind(*args, **kwargs).autotuning_configs, key=dataclasses.astuple)
            op = self.replace(config=ordered[0] if pin == 'lowest' else ordered[-1])
        return tokamax.Op.__call__(op, *args, **kwargs)

    def get_autotuning_cache(self, device_kind: str | None = None) -> _Picks:
        kind = device_kind or jax.devices()[0].device_kind
        _load(kind)
        op = self.replace(config=None, vjp=None)
        return _PICKS.setdefault((op, kind), _Picks(type(self).__name__, kind))

    def _get_autotuning_cache_key(self, ba: tokamax.BoundArguments) -> Mapping[str, object]:
        key = super()._get_autotuning_cache_key(ba)
        return type(key)({name: _bucketed(spec) if name in self.ROWS and spec is not None else spec
                          for name, spec in key.items()})

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        """Arrays at `specs` to time the launch on: normal floats, zero integers."""
        keys = jax.random.split(jax.random.key(seed), len(specs))
        return [jax.random.normal(key, spec.shape, spec.dtype) if jnp.issubdtype(spec.dtype, jnp.floating)
                else jnp.zeros(spec.shape, spec.dtype) for key, spec in zip(keys, specs, strict=True)]


def _bucketed(spec: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
    """`spec` with its leading (token) axis at the next power of two."""
    return jax.ShapeDtypeStruct((1 << max(spec.shape[0] - 1, 0).bit_length(), *spec.shape[1:]), spec.dtype)


def _measured(op: KernelOp, ba: tokamax.BoundArguments) -> AutotuningData:
    """Every config of `op`'s launch at `ba`, each compiled on a thread and
    timed alone over operand sets that rotate through `ROTATION_BYTES`."""
    leaves, tree = jax.tree.flatten((ba.args, ba.kwargs),
                                    is_leaf=lambda leaf: isinstance(leaf, jax.ShapeDtypeStruct))
    places = [index for index, leaf in enumerate(leaves) if isinstance(leaf, jax.ShapeDtypeStruct)]
    specs = [leaves[index] for index in places]
    held = sum(spec.size * spec.dtype.itemsize for spec in specs)
    rotation = max(1, min(16, -(-ROTATION_BYTES // max(held, 1))))
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

    configs = list(ba.autotuning_configs)
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


def _added(picks: tokamax.AutotuningResult) -> None:
    """Take `picks` into each launch's in-process cache, and trace anew."""
    for ba, timings in picks.data:
        ba.op.get_autotuning_cache(picks.device_kind)[ba.autotuning_cache_key] = timings.prune()
    loads, pin = _STATE.get_global()
    _STATE.set_global((loads + 1, pin))


def _load(kind: str) -> None:
    """Read the picks stored for GPUs of `kind`, once a process."""
    if kind in _LOADED:
        return
    _LOADED.add(kind)
    path = cache.kernel_picks_path(kind)
    if os.path.exists(path):
        with open(path) as file:
            _added(tokamax.AutotuningResult.load(file))


def _stored(picks: tokamax.AutotuningResult) -> None:
    """Merge `picks` into the stored picks for their GPU model, one writer at a time."""
    path = cache.kernel_picks_path(picks.device_kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with filelock.FileLock(path + '.lock'):
        if os.path.exists(path):
            with open(path) as file:
                picks = tokamax.AutotuningResult.load(file) | picks
        with tempfile.NamedTemporaryFile('w', dir=os.path.dirname(path), delete=False) as file:
            file.write(picks.dumps(prune_errors=True))
        os.replace(file.name, path)


def tune(jitted: jax.stages.Wrapped, *args) -> bool:
    """Measure the launches of `jitted(*args)` that have no stored pick for
    this GPU model, store the picks and take them on every process. True when
    picks arrived, so the caller lowers again to compile with them. Only
    process 0 measures; the others take what it measured. The program is
    lowered as the caller lowers it, which JAX's cache answers."""
    kind = jax.devices()[0].device_kind
    _load(kind)
    if not (generation.triton_runs() and cache.measures_picks()):
        return False

    def measured() -> str | None:
        if jax.process_index() != 0:
            return None
        launches = []
        for ba in tokamax.autotuning.get_bound_args(jitted, *args):
            op = ba.op
            if not isinstance(op, KernelOp) or ba.cached_autotuning_data is not None:
                continue
            timings = _measured(op, ba)
            if timings.prune_errors():
                launches.append((ba, timings))
            else:
                _log.warning("no config of %s compiled or ran; it keeps its heuristic: %s",
                             type(ba.op).__name__, next(iter(timings.values()), None))
        if not launches:
            return None
        _stored(tokamax.AutotuningResult(kind, tuple(launches)))
        # The pool takes the fastest configs alone; the store keeps every time.
        fastest = tuple((ba, timings.prune()) for ba, timings in launches)
        return tokamax.AutotuningResult(kind, fastest).dumps()

    text = coordination.broadcast_from_process_zero(coordination.agreed("measuring kernel configs", measured))
    if text is None:
        return False
    _added(tokamax.AutotuningResult.loads(text))
    return True


@contextlib.contextmanager
def pinned(extreme: str | None) -> Iterator[None]:
    """Every launch traced within takes `extreme` of its space, one of
    `EXTREMES` (None for its usual config), any stored pick or heuristic set
    aside: for a kernel's float64 gate at the ends of its space, and for a lab
    that times one end."""
    if extreme not in (None, *EXTREMES):
        raise ValueError(f"extreme must be one of {EXTREMES} or None, not {extreme!r}")
    loads, _ = _STATE.value
    with _STATE((loads, extreme)):
        yield
