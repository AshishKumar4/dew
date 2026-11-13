"""Wall-clock timing of calls: `samples`, which every benchmark tool in Dew measures independent
calls with, `carried`, which times calls that carry a state from one to the next, `benchmark`,
which lays out many calls' timings as a table, and `compared`, which reads two calls against each
other by their paired ratio; a GPU call's own kernels' device time, as CUPTI reads it
(`device_time`), which the kernel tuner and the kernel benchmarks read; the process's resident
memory as the kernel counts it page by page (`resident_memory`) and the most a region holds
(`resident_peak`), which the memory benchmarks read; and
`exclusive_window`, which holds a job's measured region alone on its node's GPUs where a node's
session runs the job.

A run is timed from the call to the moment its output is ready on the devices
(`jax.block_until_ready`), so an asynchronous dispatch is timed as the work it
starts, and reading the output back to the host is not timed. The warm-up runs,
which compile what the call traces, are not kept. The caller passes what is
timed, a jitted function bound to its inputs: nothing is jitted here.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import html
import math
import os
import select
import statistics
import threading
import time
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class Benchmark[T]:
    """What `benchmark` measured: each call's timed runs in seconds, in order, and its last output.

    `table()` and a notebook's display show each call's median. Labels that are all pairs,
    `(row, column)`, lay out as a grid of rows by columns, a sampler against its step counts say.
    """

    seconds: Mapping[Hashable, tuple[float, ...]]
    outputs: Mapping[Hashable, T]

    def median(self, label: Hashable) -> float:
        return statistics.median(self.seconds[label])

    def _grid(self) -> tuple[list[str], list[tuple[str, list[float | None]]]]:
        """The column names and each row's medians: a grid for pair labels, one column otherwise."""
        labels = list(self.seconds)
        pairs = [label for label in labels if isinstance(label, tuple) and len(label) == 2]
        if labels and len(pairs) == len(labels):
            rows = list(dict.fromkeys(row for row, _ in pairs))
            columns = list(dict.fromkeys(column for _, column in pairs))
            return [str(column) for column in columns], [
                (str(row), [self.median((row, column)) if (row, column) in self.seconds else None
                            for column in columns]) for row in rows]
        return ["median"], [(str(label), [self.median(label)]) for label in labels]

    def _cells(self) -> tuple[list[str], list[tuple[str, list[str]]]]:
        """The grid's medians written in one unit: milliseconds, or seconds once any reaches ten."""
        columns, rows = self._grid()
        known = [value for _, values in rows for value in values if value is not None]
        scale, unit = (1.0, "s") if known and max(known) >= 10 else (1000.0, "ms")
        return columns, [(name, ["" if value is None else f"{value * scale:,.1f} {unit}" for value in values])
                         for name, values in rows]

    def table(self) -> str:
        """The medians as aligned plain text, one row a label (or a row of pair labels)."""
        columns, rows = self._cells()
        first = max([len(name) for name, _ in rows] + [0])
        widths = [max([len(column)] + [len(values[index]) for _, values in rows])
                  for index, column in enumerate(columns)]
        def line(name: str, cells: list[str]) -> str:
            return f"{name:<{first}}" + "".join(f"  {cell:>{width}}" for cell, width in zip(cells, widths,
                                                                                            strict=True))

        return "\n".join([line("", columns), *(line(name, values) for name, values in rows)])

    def _repr_html_(self) -> str:
        columns, rows = self._cells()
        head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
        body = "".join(f"<tr><th>{html.escape(name)}</th>" + "".join(
            f'<td style="text-align:right">{html.escape(value)}</td>' for value in values) + "</tr>"
            for name, values in rows)
        return f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>"

    def __str__(self) -> str:
        return self.table()


def resident_memory() -> dict[str, int]:
    """This process's resident memory as the kernel counts it page by page, /proc/self/smaps_rollup's
    fields (Rss, Anonymous, Pss_File, Swap, ...), each in bytes under the kernel's names: a walk of its
    page tables, exact where /proc/self/status's running counters (VmRSS, RssAnon) lag the pages just
    written, since Linux 6.2 by a batch for each CPU the writing threads ran on, on a machine of many CPUs
    tens of MiB (a 31 MB tree held home read as 9 MB, train 742). A benchmark keeps the fields and unit
    it reports; the most a region holds is `resident_peak`'s."""
    fields = {}
    for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
        name, _, value = line.partition(":")
        match value.split():
            case [amount, "kB"]:
                fields[name] = int(amount) * 1024
    return fields


@dataclasses.dataclass
class ResidentPeak:
    """What `resident_peak` read of the process's resident memory (`resident_memory`'s fields): at the
    block's start (`before`), and the most of each field it saw within it (`most`), whole once the
    block ends."""

    before: dict[str, int]
    most: dict[str, int]

    def rise(self, field: str = "Rss") -> int:
        """How far `field` rose above the block's start at its most."""
        return self.most[field] - self.before[field]


@contextlib.contextmanager
def resident_peak(interval: float = 0.002) -> Iterator[ResidentPeak]:
    """The process's resident memory at the block's start and the most of each field within it
    (`ResidentPeak`), read exactly (`resident_memory`) every `interval` seconds on a thread of its own and
    once as the block ends, however it ends: a region's high-water, which the kernel's own (VmHWM) keeps
    for the process's whole life alone. What the region holds for less than an interval may be missed."""
    before = resident_memory()
    peak = ResidentPeak(before, dict(before))
    finished = threading.Event()

    def held() -> None:
        now = resident_memory()
        peak.most.update({name: max(peak.most.get(name, value), value) for name, value in now.items()})

    def sample() -> None:
        while not finished.wait(interval):
            held()

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        yield peak
    finally:
        finished.set()
        sampler.join()
        held()


def largest_difference(left, right) -> float:
    """The largest elementwise difference between two pytrees of arrays, in fp32: how far one
    implementation's outputs sit from another's."""
    return max(float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))
               for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True))


class KernelTime(NamedTuple):
    """One kernel a call launched and its device time, as CUPTI read it (`DeviceTime.kernels`)."""

    name: str
    milliseconds: float


class CuptiHeld(RuntimeError):
    """CUPTI's one subscription is another subscriber's, a profile capture's or Nsight's, so no device
    time is read (`device_time`): the kernel tuner keeps a launch's derived config until a later compile,
    and a benchmark reports it."""


EVICTION_BYTES = 256 << 20
"""The bytes an eviction reads (`eviction`, `behind`): over twice the L2 of every GPU Dew targets (an
H100's 50 MB, a B200's 126 MB), so a read of them leaves none of a timed call's operands in it."""


def eviction() -> jax.Array:
    """The buffer an eviction reads: `EVICTION_BYTES` of zeros, on the default device."""
    return jnp.zeros(EVICTION_BYTES // 4, jnp.float32)


EVICTION_KERNEL = "dew_eviction"
"""The name of the one kernel an eviction reads its buffer by (`evicting_read`), which no call's kernel takes,
so a timing leaves it out by name (`DeviceTime.behind`): a call's own sum is named as an XLA sum of the buffer
would be (a B200's read of 512 MiB timed at 4.9 us, its own sum's kernel left out with the eviction's)."""

EVICTION_BLOCK = 4096
"""The elements of the buffer each program of the eviction's kernel reads (`evicting_read`)."""


def evicting_read(evicting: jax.Array) -> jax.Array:
    """Every element of `evicting` read by one kernel of its own name (`EVICTION_KERNEL`), each of its
    programs summing its block into its element of the result."""
    from jax.experimental import pallas as pl

    from dew.nn.kernels.launches import pallas_call

    def summed(block, total):
        total[...] = jnp.sum(block[...], keepdims=True)

    blocks = evicting.size // EVICTION_BLOCK
    return pallas_call(summed, out_shape=jax.ShapeDtypeStruct((blocks,), jnp.float32), grid=(blocks,),
                       in_specs=[pl.BlockSpec((EVICTION_BLOCK,), lambda i: (i,))],
                       out_specs=pl.BlockSpec((1,), lambda i: (i,)), name=EVICTION_KERNEL)(evicting)


def behind(call: Callable) -> Callable:
    """The program that runs `call(*args)` behind an eviction, a read of `evicting` (`eviction`,
    `evicting_read`): called as `(evicting, args)`, the call waits for the read (an optimization barrier)
    and both outputs are returned, so the compiler keeps it. A timing that needs its operands out of the
    cache times this program and leaves the read's kernel out (`DeviceTime.behind`): CUPTI reads a call
    after an untimed run of it, which refills a cache an eviction run before the call emptied."""
    def run(evicting, args):
        evicted, args = jax.lax.optimization_barrier((evicting_read(evicting), args))
        return evicted, call(*args)
    return run


class NoKernels(ValueError):
    """A call CUPTI read none of its own kernels of (`DeviceTime`), whose time would read as zero."""


class Ungraphed(ValueError):
    """A call whose program holds a conditional XLA:GPU may run as a command buffer's conditional node,
    whose taken branch's kernels CUPTI does not record (`DeviceTime.kernels`): a B32 decode's attention on an
    RTX PRO 6000 read 1.47 ms with command buffers, its reduces alone, and 4.27 ms without, the kernel's 2.81
    ms among them (G4-LF-2). Its process keeps conditionals out of command buffers (`UNGRAPHED`)."""


UNGRAPHED = "FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CUDNN"
"""The command buffer kinds a timing process's XLA_FLAGS name (`--xla_gpu_enable_command_buffer`), every
default's but the conditional and the loop, whose kernels CUPTI reads only as plain launches (`Ungraphed`)."""


def graphs_conditionals() -> bool:
    """Whether this process's XLA:GPU may run a conditional as a command buffer's node: its XLA_FLAGS name
    no command buffer kinds, or name CONDITIONAL among them."""
    from dew._devices import xla_flag

    kinds = xla_flag("xla_gpu_enable_command_buffer")
    return kinds is None or "CONDITIONAL" in kinds.upper().split(",")


class DeviceTime:
    """The device time of each GPU call `device_time`'s session reads: its own kernels' as CUPTI times
    them, from their start to their end on the device, which no host dispatch or other process's work
    enters. The caller composes the operands each call reads and the statistic it reports.

    CUPTI reads a call after running it once untimed (Mosaic GPU's profiler warms what it times), so a
    call is timed warm; one timed out of the cache runs behind an eviction in one program (`behind`)."""

    def __init__(self, cupti):
        self._cupti = cupti
        self._subscribed = False
        self._joined: dict[Callable, Callable] = {}
        self._evicting: jax.Array | None = None
        self._branched: dict[int, tuple[object, bool]] = {}

    def kernels(self, fn: Callable, *args, evicted: bool = False) -> tuple[KernelTime, ...]:
        """Each kernel one call of `fn(*args)` launched, in launch order, with its device milliseconds,
        once its output is ready. A call that launched none is refused (`NoKernels`): its time would read
        as zero; so is a jitted call holding a conditional in a process that graphs them (`Ungraphed`),
        whose branch CUPTI may not read. `evicted` runs the call behind the session's eviction in one
        program (`behind`), its kernels left out, so its operands are read from memory, not the cache."""
        if graphs_conditionals() and isinstance(fn, jax.stages.Wrapped) and self._conditional(fn, args):
            raise Ungraphed(f"{fn} holds a conditional, which this process's XLA may graph as a command "
                            f"buffer's node whose branch CUPTI does not record: set XLA_FLAGS="
                            f"--xla_gpu_enable_command_buffer={UNGRAPHED} before JAX starts")
        if not evicted:
            return self._read(fn, *args)
        if fn not in self._joined:
            from dew.nn.kernels.options import owned_program

            self._joined[fn] = owned_program(behind(fn))
        if self._evicting is None:
            self._evicting = eviction()
        return self.behind(self._joined[fn], self._evicting, *args)

    def behind(self, program: Callable, evicting: jax.Array, *args) -> tuple[KernelTime, ...]:
        """The kernels of one call of `program`, a call run behind a read of `evicting` (`behind`,
        compiled by its caller: the kernel tuner compiles one a config), but the read's own, which its name
        gives (`EVICTION_KERNEL`). A call none of whose own CUPTI read is refused (`NoKernels`)."""
        timed = tuple(kernel for kernel in self._read(program, evicting, args)
                      if EVICTION_KERNEL not in kernel.name)
        if not timed:
            raise NoKernels(f"{program} launched no kernel CUPTI timed apart from its eviction's")
        return timed

    def _conditional(self, fn: jax.stages.Wrapped, args: tuple) -> bool:
        """Whether `fn` compiled for `args` holds a conditional (`Ungraphed`), read once for each of its
        specializations a session times: the program JAX traces for them (its static arguments' values among
        what it is traced for), which JAX's cache gives again for the same ones."""
        traced = fn.trace(*args)
        found = self._branched.get(id(traced.jaxpr))
        if found is None or found[0] is not traced.jaxpr:
            found = traced.jaxpr, " conditional(" in (traced.lower().compile().as_text() or "")
            self._branched[id(traced.jaxpr)] = found
        return found[1]

    def seconds(self, fn: Callable, *args, evicted: bool = False) -> float:
        """One call's device seconds, its kernels' summed (`kernels`)."""
        return sum(kernel.milliseconds for kernel in self.kernels(fn, *args, evicted=evicted)) / 1e3

    def _read(self, fn: Callable, *args) -> tuple[KernelTime, ...]:
        """CUPTI's reading of one call (`kernels`), refusing a call of no kernel."""
        # Each read subscribes CUPTI, which only a finalizing read detaches from the process (`_detach`),
        # whether this one reads, launches nothing or fails.
        self._subscribed = True
        try:
            _, read = self._cupti.measure(fn, aggregate=False)(*args)
        except RuntimeError as error:
            if "another subscriber" in str(error):
                raise CuptiHeld(str(error)) from error
            raise
        if not read:
            raise NoKernels(f"{fn} launched no kernel CUPTI timed, so it has no device time")
        return tuple(KernelTime(name, milliseconds) for name, milliseconds in read)

    def _detach(self) -> None:
        """Detach CUPTI from the process with a finalizing read of a call that runs nothing, so a profiler
        can subscribe after and no call of the caller's runs again; a session that read nothing, or whose
        subscription another holds (`CuptiHeld`), has nothing to detach."""
        from jax.experimental.mosaic.gpu import profiler

        if not self._subscribed:
            return
        try:
            profiler.Cupti().measure(_nothing, aggregate=False)()
        except RuntimeError as error:
            if "another subscriber" not in str(error):
                raise


def _nothing() -> None:
    """A call that launches no kernel, which a session's finalizing read runs (`DeviceTime._detach`)."""


@contextlib.contextmanager
def device_time() -> Iterator[DeviceTime]:
    """A session of GPU device timing (`DeviceTime`): CUPTI subscribed for each call it reads and
    detached from the process when it ends, however it ends, so a profile capture or Nsight can
    subscribe after. CUPTI takes one subscriber; where another holds it, a read raises `CuptiHeld`.
    Dew's one reader of CUPTI (Mosaic GPU's profiler's `Cupti`)."""
    from jax.experimental.mosaic.gpu import profiler

    session = DeviceTime(profiler.Cupti(finalize=False))
    try:
        yield session
    finally:
        session._detach()


class WindowRefused(RuntimeError):
    """The node's session cannot give the window a job asked for (`exclusive_window`), and says why."""


IN_CREATE, IN_DELETE, IN_MOVED_TO = 0x100, 0x200, 0x80
"""Linux's inotify events for a file made, removed or renamed into a watched directory (`sys/inotify.h`)."""


def _synchronize() -> None:
    """Every device of this process done with the work dispatched to it: each array still held, and one more
    computation on each device, which a GPU runs after the work queued before it on its one compute stream, so
    work whose output was dropped is waited for as well."""
    jax.block_until_ready([array for array in jax.live_arrays()
                           if not any(shard.data.is_deleted() for shard in array.addressable_shards)])
    jax.block_until_ready([jax.device_put(0.0, device) + 0.0 for device in jax.local_devices()])


def _wake(session: str) -> None:
    """A byte to the session's wake FIFO, which has the session read its jobs' windows again."""
    with open(Path(session) / "wake", "wb") as wake:
        wake.write(b"x")


class _Watch:
    """The files of a directory another process writes, watched by inotify so that a wait on one is woken by
    the change, not repeated until it shows."""

    def __init__(self, directory: Path):
        libc = ctypes.CDLL(None, use_errno=True)
        self.fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0 or libc.inotify_add_watch(self.fd, os.fsencode(directory),
                                                 IN_CREATE | IN_DELETE | IN_MOVED_TO) < 0:
            raise OSError(ctypes.get_errno(), f"inotify cannot watch {directory}")

    def until(self, done: Callable[[], bool]) -> None:
        """Waits until `done()` holds, looking again at each change to the directory. The watch is made before
        the first look, so no change is missed."""
        while not done():
            select.select([self.fd], [], [])
            os.read(self.fd, 4096)

    def close(self) -> None:
        os.close(self.fd)


@contextlib.contextmanager
def exclusive_window(cores: Iterable[int] | None = None) -> Iterator[None]:
    """The block of a job a node's session runs that is measured alone on the node's GPUs, which the job holds
    exclusively only for the block, not for its whole run. The job writes its request (`$OUT/window`, naming
    `cores`, the CPU cores it measures on, if it names any) and wakes the session (`$SESSION/wake`). The
    session admits no other job from then. Once nothing else runs on the GPUs it makes the job's GPU lock
    exclusive, locks the GPUs' clocks, pins the job's processes to `cores` and every other process it runs off
    them, and writes `$OUT/window.granted`. This waits for that file and synchronizes the device before it
    runs the block. On exit it synchronizes, removes the request, wakes the session and waits until the
    session has put all of it back and removed its grant. A session that cannot give the window (cores that
    are no CPUs of the node or all of them) refuses it, and this raises `WindowRefused`. It asks only where
    the session says the job runs packed and may (`DEW_WINDOW`); a job that holds its GPU for its whole run,
    or runs off a node (a laptop, CI), has the block to itself already, and this does nothing."""
    if os.environ.get("DEW_WINDOW") != "1":
        yield
        return
    session = os.environ["SESSION"]
    out = Path(os.environ["OUT"])
    request, granted, refused = out / "window", out / "window.granted", out / "window.refused"
    with contextlib.closing(_Watch(out)) as watch:
        written = out / "window.part"
        written.write_text(",".join(str(core) for core in sorted(cores or ())))
        written.replace(request)
        _wake(session)
        watch.until(lambda: granted.exists() or refused.exists())
        if not granted.exists():
            reason = refused.read_text().strip()
            refused.unlink()
            raise WindowRefused(reason)
        try:
            _synchronize()
            yield
        finally:
            try:
                _synchronize()
            finally:
                request.unlink()
                _wake(session)
                watch.until(lambda: not granted.exists())


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


def carried[S](step: Callable[[S], S], state: S, count: int, *, each: bool = False) -> tuple[S, list[float]]:
    """`count` calls of `step`, each handed the state the last returned (a training step's state, a
    decoder's cache), and the seconds they took, with the state the last call returned. With `each`,
    one sample a call, synchronized before the next starts, which times every call alone; otherwise
    one sample the window, its calls dispatched as a run dispatches them and synchronized once at its
    end, which times their throughput. A carried state is never released between calls, so its
    windows are timed apart from `samples`, whose calls are independent. Refuses no call."""
    if count < 1:
        raise ValueError(f"a measured window takes at least one call, not {count}")
    if each:
        seconds = []
        for _ in range(count):
            start = time.perf_counter()
            state = jax.block_until_ready(step(state))
            seconds.append(time.perf_counter() - start)
        return state, seconds
    start = time.perf_counter()
    for _ in range(count):
        state = step(state)
    state = jax.block_until_ready(state)
    return state, [time.perf_counter() - start]


@dataclasses.dataclass(frozen=True)
class Comparison:
    """A call's paired ratios to its reference judged (`judged`): each pair's ratio of the call's seconds to
    the reference's, the median ratio and its interval (`_median_rank`), and the reference against itself
    (its A/A), whose interval is the placement's own noise: a difference is one where `interval` lies apart
    from 1 and outside `alone` (`differs`)."""

    ratios: tuple[float, ...]
    ratio: float
    interval: tuple[float, float]
    alone: tuple[float, float]

    @property
    def differs(self) -> bool:
        """Whether the call's time differs from its reference's: its interval apart from 1 and outside the
        A/A's, so neither chance nor the placement's own noise accounts for it."""
        low, high = self.interval
        return (low > 1 or high < 1) and (low > self.alone[1] or high < self.alone[0])


def _median_rank(count: int, level: float) -> int:
    """The rank k whose k-th smallest and k-th largest of `count` values hold their median with chance at
    least `level`: the largest k with 2 P(Bin(count, 1/2) < k) <= 1 - level, the sign test inverted, which
    holds wherever each value falls on either side of the median as an independent fair coin. Raises
    `ValueError` where no rank does: `count` values hold their median at most at 1 - 2^(1 - count)."""
    allowed = (1 - Fraction(level)) * 2 ** count
    below, rank = 0, 0
    for k in range(1, (count + 1) // 2 + 1):
        below += math.comb(count, k - 1)
        if 2 * below > allowed:
            break
        rank = k
    if rank == 0:
        raise ValueError(f"{count} values hold their median at most at level {1 - 2.0 ** (1 - count):.6g} "
                         f"(their least and greatest), short of {level}: take more of them")
    return rank


def judged(ratios: Sequence[float], alone: Sequence[float], *, level: float = 0.95) -> Comparison:
    """Paired ratios of a call to its reference (each pair's seconds of the call over the reference's), judged
    against the reference's own A/A ratios at `level`: their median and the interval the pairs' order
    statistics give it (`_median_rank`), beside the A/A's, the placement's noise (`Comparison.differs`). The
    one judgment of a time against another: `compared`'s, and the timing harness's windows'
    (tools/timing.py), whose pairs a window's interleaved rounds make. Raises `ValueError` where `ratios` or
    `alone` hold too few pairs to reach an interval at `level`."""

    def bounds(values: Sequence[float]) -> tuple[float, float]:
        rank, ordered = _median_rank(len(values), level), sorted(values)
        return float(ordered[rank - 1]), float(ordered[len(ordered) - rank])

    interval, noise = bounds(ratios), bounds(alone)
    return Comparison(tuple(float(ratio) for ratio in ratios), float(np.median(ratios)), interval, noise)


def compared[T](first: Callable[[], T], second: Callable[[], T], *, pairs: int = 400, warmup: int = 3,
                level: float = 0.95) -> Comparison:
    """`first` against `second`, each timed as `samples` times one call, in `pairs` interleaved pairs whose
    order a fair coin picks each pair (a call that runs second in a pair runs on whatever the first left),
    and `second` against itself in as many pairs, read as the median of each pair's ratio of seconds and
    the interval its pairs' order statistics give it at `level` (`_median_rank`): only paired ratios carry
    from one machine to another, and one-sided runs of an unchanged call moved -0.6% to +0.6% (the kernel
    challenge's measurements, row 3616). The coin makes the A/A's ratios fall above and below 1 as
    independent fair coins, however the timings depend on one another, so its interval holds 1 with chance
    at least `level`, exactly; the pair's holds its median ratio so where its pairs are independent. The
    caller binds each call to fresh inputs where its time depends on its data (a MoE's routing). Raises
    `ValueError` where `pairs` reach no interval at `level`."""
    _median_rank(pairs, level)
    coins = np.random.default_rng()
    for _ in range(warmup):
        jax.block_until_ready(first())
        jax.block_until_ready(second())

    def timed(call: Callable[[], T]) -> float:
        start = time.perf_counter()
        jax.block_until_ready(call())
        return time.perf_counter() - start

    def paired(left: Callable[[], T], right: Callable[[], T]) -> np.ndarray:
        ratios = []
        for flipped in coins.integers(0, 2, pairs):
            if flipped:
                right_seconds, left_seconds = timed(right), timed(left)
            else:
                left_seconds, right_seconds = timed(left), timed(right)
            ratios.append(left_seconds / right_seconds)
        return np.asarray(ratios)

    ratios = paired(first, second)
    return judged(ratios.tolist(), paired(second, second).tolist(), level=level)


def benchmark[T](calls: Mapping[Hashable, Callable[[], T]], *, warmup: int = 1,
                 repeats: int = 3) -> Benchmark[T]:
    """Time each of `calls`, in order, `repeats` times after `warmup` untimed runs (`samples`), and
    return the runs' seconds and each call's last output (`Benchmark`); no two of a call's outputs
    are held at once. Raises `ValueError` for no timed run or a negative warm-up."""
    if repeats < 1 or warmup < 0:
        raise ValueError(f"a benchmark times at least one run after none or more warm-ups, not "
                         f"repeats={repeats} and warmup={warmup}")
    seconds: dict[Hashable, tuple[float, ...]] = {}
    outputs: dict[Hashable, T] = {}
    for label, call in calls.items():
        ran = 0

        def kept(label=label, call=call):
            # The last run's output stays: `samples` releases each one before the next run.
            nonlocal ran
            ran += 1
            output = call()
            if ran == warmup + repeats:
                outputs[label] = output
            return output

        seconds[label] = tuple(samples(kept, warmup=warmup, repeats=repeats))
    return Benchmark(seconds, outputs)


__all__ = ["EVICTION_BYTES", "Benchmark", "Comparison", "CuptiHeld", "DeviceTime", "KernelTime", "NoKernels",
           "ResidentPeak", "WindowRefused", "behind", "benchmark", "carried", "compared", "device_time",
           "eviction", "exclusive_window", "resident_memory", "resident_peak", "samples"]
