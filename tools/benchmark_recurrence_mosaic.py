#!/usr/bin/env python3
"""The recurrence template's Mosaic GPU forward (`chunk(..., mainloop='mosaic')`, the device's family's
bundle) against the chunked op's port (fla's kernels, `generation.chosen(gated_delta_rule='fla')`) and the
portable math in XLA (`generation.chosen`'s 'xla'), at the scoreboard's delta-rule shapes
(`benchmark_recurrence.SHAPES`), on the same operands: device milliseconds by CUPTI (every kernel of a call
summed, the median of `--repeats` calls after `--warmup`) and the heaviest kernels, and memory: the compiled
program's arguments, outputs and temporaries, and the device's peak in use. The forward, or with `--pass
training` the forward and its gradient under a fixed cotangent (the template's fast backward, the port's,
XLA's). One arm a process
(`--arm`), so the device's peak is its own; one JSON line an arm, pass and shape.

With `--profile DIR` the arms named (`--arm mosaic,port`) run in one process under one capture of Dew's
profiler (`dew.telemetry.profile.Profiler`, XProf's reports), each `--repeats` times after `--warmup`, and
each arm's kernels come out of XProf's kernel_stats: time, registers, shared memory, occupancy and
tensor-core use a kernel, attributed to the arm whose program ran it. Profile with
`XLA_FLAGS=--xla_gpu_enable_command_buffer=` (the profiler's note), and time without it.

    PYTHONPATH=src:tools python tools/benchmark_recurrence_mosaic.py --arm mosaic --out runs.jsonl
    XLA_FLAGS=--xla_gpu_enable_command_buffer= PYTHONPATH=src:tools \
        python tools/benchmark_recurrence_mosaic.py --arm mosaic,port --profile captures --out kernels.jsonl
"""

import argparse
import itertools
import json
import statistics
import sys
import time
import traceback

import jax
import jax.numpy as jnp
from benchmark_recurrence import SHAPES, operands
from jax.experimental.mosaic.gpu import profiler

from dew.nn.kernels import generation, recurrence
from dew.nn.kernels.recurrence_chunk import Chunk, chunk, recomputing

DELTA = ('qwen3.5-27b-gdn', 'qwen3-next-gdn', 'kimi-linear-kda')
"""The delta-rule shapes the template's forward takes: a decay a head (Qwen3.5-27B's and Qwen3-Next's
gated delta nets) and a key dimension's (Kimi Linear's)."""

ROUTES = {'mosaic': {'mainloop': 'mosaic'}, 'mosaic-recompute': {'mainloop': 'mosaic'}, 'port': {}, 'xla': {}}
"""Each arm's route through the chunked op; the port's and the portable math's under
`generation.chosen(gated_delta_rule=...)` (`NAMED`)."""

NAMED = {'port': 'fla', 'xla': 'xla'}
"""The kernel each arm names to the delta rule's selector."""


KERNELS = 40
"""The heaviest kernels a row lists: every launch a pass's glue takes as well as its mainloop's."""


def measured(compiled, args, warmup: int, repeats: int) -> tuple[float, list]:
    """The median of `repeats` calls' device milliseconds after `warmup`, every kernel of a call summed, and
    the heaviest kernels' medians."""
    cupti = profiler.Cupti(finalize=False)
    times, kernels = [], {}
    for index in range(warmup + repeats):
        _, read = cupti.measure(compiled, aggregate=False)(*args)
        if index < warmup:
            continue
        times.append(sum(ms for _, ms in read or []))
        summed = {}
        for name, ms in read or []:
            summed[name] = summed.get(name, 0.0) + ms
        for name, ms in summed.items():
            kernels.setdefault(name, []).append(ms)
    # A finalizing read detaches CUPTI from the process.
    profiler.Cupti().measure(compiled, aggregate=False)(*args)
    top = sorted(((name[:80], round(statistics.median(ms), 4)) for name, ms in kernels.items()),
                 key=lambda item: -item[1])[:KERNELS]
    return statistics.median(times), top


def compiled_arm(name: str, arm: str, dtype, training: bool = False):
    """`(compiled, args)`: the arm's forward, or forward and gradient, at the shape, compiled, its program
    named for the arm (`forward_mosaic`, `trained_port`, ...) and its operands."""
    shape = SHAPES[name]
    given = {key: jnp.asarray(value) for key, value in operands(shape).items()
             if key in ('q', 'k', 'v', 'g', 'beta')}
    # Kimi delta attention's declared floor (its layer's `lower_bound`), which the draws' decays lie above.
    hooks = recurrence.Hooks(decay='vector' if shape.kind == 'kda' else 'scalar',
                             scale=shape.key_width ** -0.5, floor=-5.0 if shape.kind == 'kda' else None)
    args = (*(given[key].astype(dtype) for key in ('q', 'k', 'v')), given['g'], given['beta'].astype(dtype))

    def forward(q, k, v, g, beta):
        return chunk(hooks, Chunk(q, k, v, log_decay=g, beta=beta), chunk_size=shape.chunk, **ROUTES[arm])

    def trained(*args):
        # The forward's results returned with the gradients, as a training step reads both: a pass
        # returning the gradients alone leaves XLA to drop every forward launch the backward does not
        # read (each arm's output launch). The recomputing arm keeps no memories for its backward, as the
        # remat rung 'delta_memory' asks (`recurrence_chunk.recomputing`).
        with recomputing(('delta_memory',) if arm.endswith('-recompute') else ()):
            out, pull = jax.vjp(forward, *args)
        return out, pull(jax.tree.map(jnp.ones_like, out))

    timed = trained if training else forward
    timed.__name__ = f"{'trained' if training else 'forward'}_{arm}"
    if arm in NAMED:
        with generation.chosen(gated_delta_rule=NAMED[arm]):
            return jax.jit(timed).lower(*args).compile(), args
    return jax.jit(timed).lower(*args).compile(), args


def rows(name: str, arm: str, dtype, warmup: int, repeats: int, training: bool = False,
         rounds: int = 1) -> list[dict]:
    """The arm's pass at the shape compiled once and timed `rounds` times (`measured`), a row a round: its
    compile time (the first round's), its program's memory and the device's peak so far."""
    found = []
    base = {'shape': name, 'arm': arm, 'pass': 'training' if training else 'forward',
            'dtype': jnp.dtype(dtype).name, 'device': jax.devices()[0].device_kind}
    try:
        start = time.perf_counter()
        compiled, args = compiled_arm(name, arm, dtype, training)
        base['compile_s'] = round(time.perf_counter() - start, 1)
        memory = compiled.memory_analysis()
        base.update({f'{part}_mb': round(getattr(memory, f'{part}_size_in_bytes') / 2**20, 1)
                     for part in ('argument', 'output', 'temp')})
        for index in range(rounds):
            row = {**base, 'round': index}
            row['ms'], row['kernels'] = measured(compiled, args, warmup, repeats)
            stats = jax.devices()[0].memory_stats() or {}
            row['peak_mb'] = round(stats.get('peak_bytes_in_use', 0) / 2**20, 1)
            found.append(row)
    except Exception as error:  # an arm's failure is its row's
        found.append({**base, 'error': f"{type(error).__name__}: {str(error)[:2000]}",
                      'trace': traceback.format_exc()[-3000:]})
    return found


def profiled(name: str, arms: list[str], dtype, warmup: int, repeats: int, training: bool,
             directory: str) -> list[dict]:
    """The arms' programs run `repeats` times each, after `warmup`, in one capture of Dew's profiler under
    `directory`, and a row an arm: its kernels from XProf's kernel_stats, each attributed to the arm whose
    program's name its op names (`compiled_arm`'s), with its calls a pass, its time a call (microseconds,
    the mean), registers, shared memory, occupancy and tensor-core use."""
    from dew.telemetry.profile import Profiler, program_kernels

    compiled = {arm: compiled_arm(name, arm, dtype, training) for arm in arms}
    for program, args in compiled.values():
        for _ in range(warmup):
            jax.block_until_ready(program(*args))
    profiler = Profiler(directory)
    with profiler:
        for arm, (program, args) in compiled.items():
            with profiler.region(arm):
                for _ in range(repeats):
                    jax.block_until_ready(program(*args))
    passed = 'trained' if training else 'forward'
    capture = profiler.captures[-1]
    kernels = program_kernels(capture, [f"{passed}_{arm}" for arm in arms])
    rows = [{'shape': name, 'arm': arm, 'pass': passed, 'dtype': jnp.dtype(dtype).name,
             'device': jax.devices()[0].device_kind, 'capture': str(capture),
             'kernels': kernels[f"{passed}_{arm}"], 'unattributed': kernels['']} for arm in arms]
    for found in rows:
        found['us'] = round(sum(float(kernel.get('total_duration_us') or 0) for kernel in found['kernels'])
                            / repeats, 1)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--arm', required=True, help=f"one of {tuple(ROUTES)}, or several comma-separated "
                                                     "with --profile")
    parser.add_argument('--profile', help="capture the arms' kernels under this directory (kernel_stats)")
    parser.add_argument('--shape', default='all', help=f"one of {DELTA} or 'all'")
    parser.add_argument('--dtype', default='bfloat16', help="bfloat16, float32, or both comma-separated")
    parser.add_argument('--pass', dest='training', default='forward',
                        help="forward, training, or both comma-separated (in that order: the device's peak "
                             "is the process's so far, so the forward's stays its own)")
    parser.add_argument('--rounds', type=int, default=1,
                        help="time each compiled program this many times, a row each, compiling it once")
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--out', help='append JSON lines here as well as printing them')
    args = parser.parse_args(argv)
    arms = args.arm.split(',')
    if any(arm not in ROUTES for arm in arms) or (len(arms) > 1 and not args.profile):
        parser.error(f"--arm takes one of {tuple(ROUTES)}, several only with --profile")
    passes = args.training.split(',')
    if not set(passes) <= {'forward', 'training'} or passes != sorted(passes):
        parser.error("--pass takes forward, training, or forward,training")
    failed = []
    for name in DELTA if args.shape == 'all' else (args.shape,):
        for dtype, passed in itertools.product(args.dtype.split(','), passes):
            training = passed == 'training'
            kind, timing = jnp.dtype(dtype), (args.warmup, args.repeats, training)
            if args.profile:
                found = profiled(name, arms, kind, *timing, args.profile)
            else:
                found = rows(name, arms[0], kind, *timing, args.rounds)
            failed += [f"{row['arm']} {row['shape']} {row['pass']} {row['dtype']}: {row['error'][:200]}"
                       for row in found if 'error' in row]
            for line in map(json.dumps, found):
                print(line, flush=True)
                if args.out:
                    with open(args.out, 'a') as handle:
                        handle.write(line + '\n')
    # Every row is written, then an arm that failed fails the run: a receipt never reads as a timing of an
    # arm that did not run.
    if failed:
        sys.exit("arms failed:\n" + "\n".join(failed))


if __name__ == '__main__':
    main()
