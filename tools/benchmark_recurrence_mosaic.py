#!/usr/bin/env python3
"""The recurrence template's Mosaic GPU forward (`chunk(..., mainloop='mosaic')`, the device's family's
bundle) against the chunked op's port (fla's kernels, `generation.chosen(gated_delta_rule='fla')`) and the
portable math in XLA (`ported=False`), at the scoreboard's delta-rule shapes (`benchmark_recurrence.SHAPES`),
on the same operands: device milliseconds by CUPTI (every kernel of a call summed, the median of `--repeats`
calls after `--warmup`) and the heaviest kernels, and memory: the compiled program's arguments, outputs and
temporaries, and the device's peak in use. One arm a process (`--arm`), so the device's peak is its own; one
JSON line an arm and shape.

    PYTHONPATH=src:tools python tools/benchmark_recurrence_mosaic.py --arm mosaic --out runs.jsonl
"""

import argparse
import json
import statistics
import time
import traceback

import jax
import jax.numpy as jnp
from benchmark_recurrence import SHAPES, operands
from jax.experimental.mosaic.gpu import profiler

from dew.nn.kernels import generation, recurrence
from dew.nn.kernels.recurrence_chunk import Chunk, chunk

DELTA = ('qwen3.5-27b-gdn', 'qwen3-next-gdn', 'kimi-linear-kda')
"""The delta-rule shapes the template's forward takes: a decay a head (Qwen3.5-27B's and Qwen3-Next's
gated delta nets) and a key dimension's (Kimi Linear's)."""

ROUTES = {'mosaic': {'mainloop': 'mosaic'}, 'port': {}, 'xla': {'ported': False}}
"""Each arm's route through the chunked op; the port's under `generation.chosen(gated_delta_rule='fla')`."""


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
                 key=lambda item: -item[1])[:8]
    return statistics.median(times), top


def row(name: str, arm: str, dtype, warmup: int, repeats: int) -> dict:
    shape = SHAPES[name]
    given = {key: jnp.asarray(value) for key, value in operands(shape).items()
             if key in ('q', 'k', 'v', 'g', 'beta')}
    hooks = recurrence.Hooks(decay='vector' if shape.kind == 'kda' else 'scalar',
                             scale=shape.key_width ** -0.5)
    args = (*(given[key].astype(dtype) for key in ('q', 'k', 'v')), given['g'], given['beta'].astype(dtype))

    def forward(q, k, v, g, beta):
        return chunk(hooks, Chunk(q, k, v, log_decay=g, beta=beta), chunk_size=shape.chunk, **ROUTES[arm])

    found = {'shape': name, 'arm': arm, 'dtype': jnp.dtype(dtype).name,
             'device': jax.devices()[0].device_kind}
    try:
        start = time.perf_counter()
        if arm == 'port':
            with generation.chosen(gated_delta_rule='fla'):
                compiled = jax.jit(forward).lower(*args).compile()
        else:
            compiled = jax.jit(forward).lower(*args).compile()
        found['compile_s'] = round(time.perf_counter() - start, 1)
        memory = compiled.memory_analysis()
        found.update({f'{part}_mb': round(getattr(memory, f'{part}_size_in_bytes') / 2**20, 1)
                      for part in ('argument', 'output', 'temp')})
        found['ms'], found['kernels'] = measured(compiled, args, warmup, repeats)
        stats = jax.devices()[0].memory_stats() or {}
        found['peak_mb'] = round(stats.get('peak_bytes_in_use', 0) / 2**20, 1)
    except Exception as error:  # an arm's failure is its row's
        found['error'] = f"{type(error).__name__}: {str(error)[:2000]}"
        found['trace'] = traceback.format_exc()[-3000:]
    return found


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--arm', choices=tuple(ROUTES), required=True)
    parser.add_argument('--shape', default='all', help=f"one of {DELTA} or 'all'")
    parser.add_argument('--dtype', default='bfloat16', help="bfloat16, float32, or both comma-separated")
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--out', help='append JSON lines here as well as printing them')
    args = parser.parse_args(argv)
    for name in DELTA if args.shape == 'all' else (args.shape,):
        for dtype in args.dtype.split(','):
            line = json.dumps(row(name, args.arm, jnp.dtype(dtype), args.warmup, args.repeats))
            print(line, flush=True)
            if args.out:
                with open(args.out, 'a') as handle:
                    handle.write(line + '\n')


if __name__ == '__main__':
    main()
