#!/usr/bin/env python3
"""A GPU's memory facts as its kernels meet them, measured rather than read off its datasheet (row 3612): the
device seconds one kernel takes to stream N bytes from cold HBM, against N, and the fit `seconds = fixed +
bytes / plateau` over them, with the least kernel's seconds (a launch's floor on the device), and the wall
seconds a launch adds to a program that runs kernels one after another and a hand-off's between two programs
of one launch (row 3613's fixed costs of a launch and of the balanced tail's partials).

    python tools/device_facts.py --out facts.json

Each point reads an array of N bytes once (its sum, `read`) or reads and writes it (`copy`), after a write of
twice the L2 that evicts it (`cold`); CUPTI times the kernel alone (`jax.experimental.mosaic.gpu.profiler`),
the median of `--repeats` after `--warmup`. An H200's read, measured so, ramps from 2.6 TB/s at 16.8 MB to
4.5 at 805 MB over a 2.5-2.8 us fixed cost, which a peak bandwidth alone prices as free at small transfers.

A chain is one program of N tiny kernels, each reading the last's output and held apart from it by an
optimization barrier (so XLA fuses none), timed whole by the wall clock (`dew.telemetry.benchmark.samples`,
as XLA launches it, its command buffers at JAX's default) for N from 1 to 64; the line's slope is a launch's
seconds in a program, its intercept the call's own. A hand-off is one GPU-scope semaphore signal another
program waits on, as the balanced tail's partials (`templates.mosaic.RingFamily.fixed_up`): two programs of
one kernel each signal the other and wait on their own K times, CUPTI timing it for K from 16 to 1024; the
slope over the K rounds is one hand-off's seconds, measured where the GPU's ring families run the tail (TMA,
sm90 on).
"""

import argparse
import json
import statistics

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.mosaic.gpu import profiler

from dew.nn.kernels import generation
from dew.telemetry.benchmark import samples

SIZES = tuple(2 ** power for power in range(20, 31))
"""The bytes each point streams: 1 MiB to 1 GiB, by doubling."""

CHAINS = (1, 2, 4, 8, 16, 32, 64)
"""The kernels each chain launches one after another."""

ROUND_TRIPS = (16, 64, 256, 1024)
"""The rounds each hand-off kernel's two programs make, each one hand-off."""


def timed(fn, args, *, flush, warmup: int, repeats: int) -> float:
    """The median device seconds of `fn`'s kernels over `repeats` calls after `warmup`, each call after
    `flush` evicts the L2 (CUPTI times `fn`'s kernels alone)."""
    cupti = profiler.Cupti(finalize=False)
    seconds = []
    for index in range(warmup + repeats):
        jax.block_until_ready(flush())
        _, read = cupti.measure(fn, aggregate=False)(*args)
        if index >= warmup:
            seconds.append(sum(ms for _, ms in read or []) / 1e3)
    profiler.Cupti().measure(fn, aggregate=False)(*args)
    return statistics.median(seconds)


def fitted(points: list[tuple[int, float]]) -> tuple[float, float]:
    """The least-squares `seconds = fixed + bytes / plateau` over `points`: (fixed seconds, bytes a
    second)."""
    sizes, seconds = np.asarray([b for b, _ in points], float), np.asarray([s for _, s in points], float)
    slope, fixed = np.polyfit(sizes, seconds, 1)
    return float(fixed), float(1.0 / slope)


def chained(length: int):
    """A program of `length` kernels one after another: each adds one to the last's output, an optimization
    barrier between them, so each is a kernel of its own."""
    def program(x):
        for _ in range(length):
            x = jax.lax.optimization_barrier(x + 1.0)
        return x
    return jax.jit(program)


def handed_off(count: int):
    """One kernel of two programs passing a GPU-scope semaphore back and forth `count` times."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu

    def body(_):
        program = jax.lax.axis_index("x")
        ready = pl.get_global(plgpu.SemaphoreType.REGULAR((2,)))

        # Each round both programs signal the other's semaphore and wait on their own: one hand-off a round.
        def round_trip(_, carry):
            plgpu.semaphore_signal(ready.at[1 - program], memory_scope="gpu")
            plgpu.semaphore_wait(ready.at[program], memory_scope="gpu")
            return carry

        jax.lax.fori_loop(0, count, round_trip, 0)

    return jax.jit(plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((2, 128), jnp.float32), grid=(2,),
                                grid_names=("x",)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    device = jax.devices()[0]
    l2 = int(getattr(device, "l2_cache_size", 0) or 64 * 2 ** 20)
    evictor = jnp.zeros((2 * l2 // 4,), jnp.float32)
    flush = jax.jit(lambda x: x + 1.0)
    read = jax.jit(lambda x: jnp.sum(x, dtype=jnp.float32))
    copy = jax.jit(lambda x: x + jnp.asarray(1, x.dtype))
    rows = []
    for size in SIZES:
        x = jnp.ones((size // 2,), jnp.bfloat16)
        row = {"bytes": size}
        for name, fn, moved in (("read", read, size), ("copy", copy, 2 * size)):
            seconds = timed(fn, (x,), flush=lambda: flush(evictor), warmup=args.warmup, repeats=args.repeats)
            row[name] = {"seconds": seconds, "bytes_per_second": moved / seconds}
        rows.append(row)
        print(json.dumps(row), flush=True)
    tiny = jnp.ones((256,), jnp.float32)
    least = timed(copy, (tiny,), flush=lambda: tiny, warmup=args.warmup, repeats=args.repeats)
    chains = [(length, statistics.median(samples(chained(length), tiny, warmup=args.warmup,
                                                 repeats=args.repeats)))
              for length in CHAINS]
    slope, intercept = np.polyfit([length for length, _ in chains], [seconds for _, seconds in chains], 1)
    facts = {"device": device.device_kind, "l2_bytes": l2, "least_kernel_seconds": least, "points": rows,
             "chain": {"launch_seconds": float(slope), "call_seconds": float(intercept),
                       "points": [{"kernels": length, "seconds": seconds} for length, seconds in chains]}}
    if generation.gpu_target().architecture.tma:
        trips = [(count, timed(handed_off(count), (), flush=lambda: tiny, warmup=args.warmup,
                               repeats=args.repeats)) for count in ROUND_TRIPS]
        hop, _ = np.polyfit([count for count, _ in trips], [seconds for _, seconds in trips], 1)
        facts["handoff"] = {"seconds": float(hop), "points": [{"round_trips": count, "seconds": seconds}
                                                              for count, seconds in trips]}
    for name, moved in (("read", 1), ("copy", 2)):
        fixed, plateau = fitted([(row["bytes"] * moved, row[name]["seconds"]) for row in rows])
        facts[name] = {"fixed_seconds": fixed, "plateau_bytes_per_second": plateau}
    print(json.dumps({key: value for key, value in facts.items() if key != "points"}), flush=True)
    with open(args.out, "w") as handle:
        json.dump(facts, handle, indent=1)


if __name__ == "__main__":
    main()
