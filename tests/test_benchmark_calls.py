"""`dew.telemetry.benchmark`: calls timed from the call until their output is ready, after warm-up
runs that compile what they trace, laid out by label as tutorial 04 lays out its samplers; and the
process's resident memory as the memory benchmarks read it."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sampling_gate import FALSE_RED

from dew.telemetry.benchmark import CuptiHeld, behind, benchmark, device_time, eviction


class Slow:
    """An output that is ready a known time after it is returned, as a dispatched computation is."""

    def __init__(self, seconds):
        self.seconds = seconds

    def block_until_ready(self):
        time.sleep(self.seconds)
        return self


def test_a_benchmark_times_each_call_until_its_output_is_ready_after_untimed_warm_ups():
    traced = []

    @jax.jit
    def step(x):
        traced.append(1)
        return jnp.cumsum(x) * 2

    x = jnp.arange(8.0)
    measured = benchmark({"cumsum": lambda: step(x), "slow": lambda: Slow(0.05)}, warmup=2, repeats=3)
    assert len(traced) == 1, "the warm-up compiles once and no timed run traces again"
    assert [len(measured.seconds[label]) for label in ("cumsum", "slow")] == [3, 3]
    assert min(measured.seconds["slow"]) >= 0.05, "the wait for the output is timed"
    np.testing.assert_array_equal(np.asarray(measured.outputs["cumsum"]), np.cumsum(np.arange(8.0)) * 2)
    assert measured.median("slow") == sorted(measured.seconds["slow"])[1]
    with pytest.raises(ValueError):
        benchmark({"cumsum": lambda: step(x)}, repeats=0)


def test_a_carried_window_hands_each_call_the_last_state_and_times_the_wait_for_it():
    """Calls that carry a state get the state the last returned; one window waits once at its end,
    each call's wait inside it, and a window of samples waits on every call, one sample apiece."""
    from dew.telemetry.benchmark import carried

    def step(carried_state):
        count, _ = carried_state
        return count + 1, Slow(0.02)

    (count, _), (window,) = carried(step, (0, None), 3)
    assert count == 3 and window >= 0.02
    (count, _), each = carried(step, (count, None), 2, each=True)
    assert count == 5 and len(each) == 2 and min(each) >= 0.02
    with pytest.raises(ValueError):
        carried(step, (0, None), 0)


def test_pair_labels_lay_out_as_rows_by_columns_in_one_unit():
    """Medians laid out by label: pairs as a grid of rows by columns, a pair that never ran left
    empty, and one unit a table, milliseconds until a median reaches ten seconds."""
    from dew.telemetry.benchmark import Benchmark

    measured = Benchmark({("euler", "4 steps"): (0.002, 0.001, 0.003), ("euler", "8 steps"): (0.004,),
                          ("heun", "8 steps"): (1.0115,)}, {})
    # Each column right-aligned: every cell ends where its column's label ends, at least two spaces
    # past the column before; a pair that never ran leaves its cell blank.
    header, euler, heun = measured.table().splitlines()
    ends = [header.index(label) + len(label) for label in ("4 steps", "8 steps")]
    assert euler.startswith("euler") and heun.startswith("heun")
    assert [euler.index(cell) + len(cell) for cell in ("2.0 ms", "4.0 ms")] == ends
    assert heun.index("1,011.5 ms") + len("1,011.5 ms") == ends[1] and not heun[len("heun"):ends[0]].strip()
    # Two spaces at least between columns: each cell starts two past the end of the column before it,
    # the row labels' column as wide as its widest label, a blank cell's column included.
    bounds = [max(len("euler"), len("heun")), *ends]
    for line, cells in ((header, ("4 steps", "8 steps")), (euler, ("2.0 ms", "4.0 ms")),
                        (heun, (None, "1,011.5 ms"))):
        for column, cell in enumerate(cells):
            if cell is not None:
                assert line.index(cell) >= bounds[column] + 2, line
    html = measured._repr_html_()
    assert "<th>heun</th>" in html and "1,011.5 ms" in html and html.count("<td") == 4
    slow = Benchmark({"sample": (12.0, 14.0, 13.0), "<script>": (1.0,)}, {})
    # A median of ten seconds or more and its column in seconds, the column right-aligned.
    header, *rows = slow.table().splitlines()
    assert [row.split()[0] for row in rows] == ["sample", "<script>"]
    assert [row.rstrip()[-len(cell):] for row, cell in zip(rows, ("13.0 s", "1.0 s"), strict=True)] == [
        "13.0 s", "1.0 s"]
    assert {len(row.rstrip()) for row in rows} == {len(header.rstrip())}
    assert "&lt;script&gt;" in slow._repr_html_()


class Scheduled:
    """The host's clock as `dew.telemetry.benchmark` reads it (`time.perf_counter`, its seam), advanced by
    the calls it times alone: each by its sleep and the delay a seeded scheduler adds past it, as a busy
    host's do (row 3793's measurements). Outside a burst a call is late by 0.1 to 1.5 ms; a burst, which
    one call in 30 begins, makes 20 calls 1 to 8 ms late and one in ten of them 12 to 40 ms, at most
    `long` such a run. No delay is negative: a sleep returns no sooner than asked."""

    def __init__(self, seed: int, long: int = 15):
        self.rng, self.now, self.burst, self.long = np.random.default_rng(seed), 0.0, 0, long

    def read(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        if not self.burst and self.rng.random() < 1 / 30:
            self.burst = 20
        if not self.burst:
            late = self.rng.uniform(1e-4, 1.5e-3)
        elif self.long and self.rng.random() < 0.1:
            self.long, late = self.long - 1, self.rng.uniform(0.012, 0.040)
        else:
            late = self.rng.uniform(1e-3, 8e-3)
        self.burst = max(self.burst - 1, 0)
        self.now += seconds + late


def scheduled(monkeypatch, seed: int) -> Scheduled:
    """A seeded `Scheduled` clock that `compared` reads in place of the host's."""
    from dew.telemetry import benchmark

    clock = Scheduled(seed)
    monkeypatch.setattr(benchmark.time, "perf_counter", clock.read)
    return clock


def test_a_comparison_reads_two_arms_paired_and_role_swapped_against_their_own_noise(monkeypatch):
    """Two calls compared (`compared`): one taking four times the other's 12 ms, timed in role-swapped
    pairs, read as their paired ratio, and the faster call against itself (its A/A), each interval at a
    level of 1 - FALSE_RED / 2 over 64 pairs (`_median_rank`: the 16th smallest and largest), on a host
    whose scheduler bursts (`Scheduled`, seed 3, where 24 pairs at 95% left 1 out of the A/A's interval).
    The A/A's holds 1 but for 2.4e-5 of the coins that order its pairs, its ratios' sides fair coins
    whatever the delays. The pair's starts at most at 4 but for 1.2e-5 of them: a ratio passes 4 only
    where the coin gave the slower call the larger delay, so 49 do with chance P(Bin(64, 1/2) >= 49). And
    the pair's lies above the A/A's whatever the coins: a ratio reaches 2, the arms' geometric middle,
    only where a faster call is 12 ms late, and an A/A's only where either is, so the intervals meet only
    where 16 calls are, one more than the scheduler makes a run: an interval apart from 1 and from the
    A/A's is a difference the placement does not make (row 3616, the kernel challenge's method: one-sided
    runs moved -0.6% to +0.6% whatever the change)."""
    from dew.telemetry.benchmark import compared

    clock = scheduled(monkeypatch, 3)

    def slow():
        clock.sleep(0.048)
        return jnp.zeros(())

    def fast():
        clock.sleep(0.012)
        return jnp.zeros(())

    result = compared(slow, fast, pairs=64, warmup=1, level=1 - FALSE_RED / 2)
    low, _ = result.interval
    alone_low, alone_high = result.alone
    assert alone_low <= 1.0 <= alone_high < low <= 4.0 and result.differs


@pytest.mark.parametrize(("ratios", "alone", "level", "ratio", "interval", "noise", "differs"), [
    # Six pairs at 0.95 hold their median between their least and greatest (rank 1).
    ((1.31, 1.29, 1.30, 1.33, 1.28, 1.32), (0.99, 1.01, 1.00, 1.02, 0.98, 1.00), 0.95,
     1.305, (1.28, 1.33), (0.98, 1.02), True),
    # Ten pairs at 0.95 between their second least and second greatest (rank 2): the tenth's outlier at 0.97
    # leaves the interval apart from 1.
    ((1.10, 1.12, 1.11, 1.13, 0.97, 1.09, 1.12, 1.10, 1.11, 1.14),
     (1.00, 0.99, 1.01, 1.00, 1.02, 0.98, 1.00, 1.01, 0.99, 1.00), 0.95, 1.11, (1.09, 1.13), (0.99, 1.01),
     True),
    # The same ten at 0.99 (rank 1): the outlier's 0.97 holds 1, so nothing differs.
    ((1.10, 1.12, 1.11, 1.13, 0.97, 1.09, 1.12, 1.10, 1.11, 1.14),
     (1.00, 0.99, 1.01, 1.00, 1.02, 0.98, 1.00, 1.01, 0.99, 1.00), 0.99, 1.11, (0.97, 1.14), (0.98, 1.02),
     False),
    # Apart from 1 but within the A/A's noise: the placement's own, not a difference, however far its median.
    ((1.03, 1.04, 1.05, 1.03, 1.06, 1.04), (0.95, 1.07, 1.00, 0.97, 1.03, 1.01), 0.95, 1.04, (1.03, 1.06),
     (0.95, 1.07), False),
    # Holding 1, though apart from the A/A's: chance accounts for it.
    ((0.99, 1.02, 1.03, 1.04, 1.05, 1.03), (0.97, 0.98, 0.98, 0.97, 0.98, 0.98), 0.95, 1.03, (0.99, 1.05),
     (0.97, 0.98), False),
    # Ties at the ends.
    ((1.2, 1.2, 1.2, 1.2, 1.2, 1.2), (1.0, 1.0, 1.0, 1.0, 1.0, 1.0), 0.95, 1.2, (1.2, 1.2), (1.0, 1.0), True),
])
def test_paired_ratios_are_judged_against_their_a_a_by_one_rule(ratios, alone, level, ratio, interval,
                                                                  noise, differs):
    """`judged`, the one judgment `compared` and a timing window (tools/timing.py) read a time by: the paired
    ratios' median, the interval their order statistics hold it in at `level` (`_median_rank`), the A/A's
    interval beside it, and a difference only where the interval lies apart from 1 and outside the A/A's."""
    from dew.telemetry.benchmark import judged

    judgment = judged(ratios, alone, level=level)
    assert (judgment.ratio, judgment.interval, judgment.alone, judgment.differs) == (
        pytest.approx(ratio), pytest.approx(interval), pytest.approx(noise), differs)


def test_ratios_too_few_for_an_interval_at_their_level_are_refused_and_six_are_not():
    """Five pairs hold their median at 0.9375 at most, so `judged` refuses them at 0.95, either side short
    (the call's or the A/A's), as `_median_rank` does; six it judges."""
    from dew.telemetry.benchmark import judged

    six = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
    with pytest.raises(ValueError):
        judged(six[:5], six)
    with pytest.raises(ValueError):
        judged(six, six[:5])
    assert judged(six, six).interval == (1.0, 1.5)


def test_a_medians_rank_holds_it_at_its_level_and_no_higher_rank_does():
    """`_median_rank`'s k-th smallest and largest of n values hold their median with chance
    1 - 2 P(Bin(n, 1/2) < k) (the sign test, any continuous distribution): at least the level, and the next
    rank's short of it, at every count from the fewest that reach the level to 400; fewer refuse."""
    from scipy import stats

    from dew.telemetry.benchmark import _median_rank

    for level in (0.95, 0.999, 1 - FALSE_RED / 2):
        fewest = next(count for count in range(1, 64) if 1 - 2.0 ** (1 - count) >= level)
        with pytest.raises(ValueError):
            _median_rank(fewest - 1, level)
        for count in range(fewest, 401):
            rank = _median_rank(count, level)
            assert 1 - 2 * stats.binom.cdf(rank - 1, count, 0.5) >= level
            assert rank == (count + 1) // 2 or 1 - 2 * stats.binom.cdf(rank, count, 0.5) < level


NEIGHBOURED = """
import ctypes
import json
import mmap
import os

import numpy as np

from dew.telemetry.benchmark import resident_memory

# The resident counters lag by a batch for each CPU the writer ran on (`resident_memory`): written from
# one CPU, the pages reach the reading but for one CPU's batch, however many CPUs the machine has.
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

# The heap serves 64 MiB (glibc's M_MMAP_THRESHOLD, -3), and a freed 64 MiB block stays resident under
# a live one: memory a process's earlier work left for its allocator to hand out again.
ctypes.CDLL(None).mallopt(-3, 1 << 30)
held = np.ones(64 * 2**20, np.uint8)
guard = np.ones(2**20, np.uint8)
del held
# A new private mapping is pages no allocator held: the kernel hands each in as it is written.
with mmap.mmap(-1, 64 * 2**20, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS) as pages:
    before = resident_memory()
    for offset in range(0, len(pages), mmap.PAGESIZE):
        pages[offset] = 1
    after = resident_memory()
    print(json.dumps({"before": before, "after": after, "size": len(pages)}))
"""


def test_resident_memory_reads_the_kernel_fields_in_bytes_and_sees_an_allocation():
    """Each of the kernel's memory fields in bytes under its own name: 64 MiB of new pages written
    through, from one CPU so the kernel's per-CPU batching holds back one CPU's batch at most, raises
    the anonymous resident memory by more than half its size, which a field read in the kernel's KiB
    would not show, even in a process whose allocator holds a freed block of that size resident,
    which an array of that size would be served from without a new page."""
    import json
    import subprocess
    import sys

    measured = json.loads(subprocess.run([sys.executable, "-c", NEIGHBOURED], check=True, capture_output=True,
                                         text=True).stdout.splitlines()[-1])
    before, after = measured["before"], measured["after"]
    assert {"VmRSS", "RssAnon", "RssFile", "VmHWM"} <= set(before)
    assert after["RssAnon"] > before["RssAnon"] + measured["size"] // 2
    assert after["VmHWM"] >= after["VmRSS"] >= after["RssAnon"]


@pytest.mark.gpu(reason="CUPTI times a GPU's kernels")
def test_device_time_reads_a_calls_kernels_and_hands_cupti_to_a_profiler_however_it_ends(tmp_path):
    """A session reads each kernel a call launched with its device time and refuses a call that launched
    none, whose time would read as zero; a call timed behind an eviction in its program reads its own
    kernels, the eviction's left out; the session detaches CUPTI however it ends, after a read or only a
    refused one, so a profile capture after it records the call's kernels, and a read inside that capture
    is refused as held. Leaving a session runs none of its calls again: an input deleted inside it stays
    deleted."""
    from dew.telemetry.profile import Profiler, program_kernels

    def squared(a):
        return a @ a
    x = jnp.ones((1024, 1024))
    matmul = jax.jit(squared)
    used = jnp.ones((1024, 1024))
    with pytest.raises(ZeroDivisionError), device_time() as device:
        kernels = device.kernels(matmul, used)
        assert kernels and all(kernel.milliseconds > 0 for kernel in kernels)
        assert device.seconds(matmul, used) > 0
        used.delete()
        raise ZeroDivisionError
    with device_time() as device:
        with pytest.raises(ValueError):
            device.kernels(lambda: None)
        buffer = eviction()
        evicting = {kernel.name for kernel in device.kernels(jax.jit(jnp.sum), buffer)}
        # The eviction runs in the program its call is timed in, which the compiler keeps.
        assert evicting & {kernel.name for kernel in device.kernels(jax.jit(behind(matmul)), buffer, (x,))}
        timed = device.kernels(matmul, x, evicted=True)
        assert timed and not evicting & {kernel.name for kernel in timed}
    profiler = Profiler(tmp_path)
    with profiler, device_time() as device:
        jax.block_until_ready(matmul(x))
        with pytest.raises(CuptiHeld):
            device.kernels(matmul, x)
    assert program_kernels(profiler.captures[-1], ["squared"])["squared"]
