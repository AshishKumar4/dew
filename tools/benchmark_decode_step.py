#!/usr/bin/env python3
"""A gated delta net's decode token, the whole layer (its projections, conv, gates, update and gated norm),
under each of the decode selector's kernels (`linear.DELTA_DECODE`): the fused step (one launch taking the
conv's history, the gates, the delta rule's update and the output's gated norm, `recurrence.step` with the
layer's hooks), the step kernel between the layer's own conv and norm, and the portable recurrence in XLA;
at 1, 8, 32 and 128 rows, bf16 weights and the cache as the layer allocates it. Device milliseconds
by CUPTI (every kernel of a token summed, the median of `--repeats` tokens after `--warmup`) and the
heaviest kernels. One JSON line a kernel and row count.

    PYTHONPATH=src:tools python tools/benchmark_decode_step.py --out decode.jsonl
"""

import argparse
import json
import sys
import traceback

import jax
import jax.numpy as jnp
from benchmark_recurrence_mosaic import measured

from dew.nn.inputs import MixerCall
from dew.nn.kernels import generation
from dew.nn.linear import GatedDeltaNet

SHAPES = {
    'qwen3.5-27b-gdn': {'emb_features': 5120, 'num_k_heads': 16, 'num_v_heads': 48, 'head_k_dim': 128,
                        'head_v_dim': 128},
    'qwen3-next-gdn': {'emb_features': 2048, 'num_k_heads': 16, 'num_v_heads': 32, 'head_k_dim': 128,
                       'head_v_dim': 128},
}
"""The gated delta nets of Qwen3.5-27B and Qwen3-Next, as the scoreboard's prefill shapes have them."""

KERNELS = ('fused', 'step', 'xla')
"""The decode selector's kernels (`linear.DELTA_DECODE`)."""

ROWS = (1, 8, 32, 128)
"""The rows a decode step serves (row 1424's)."""

PREFILL = 4
"""Tokens through the cache before the timed one: the conv's history full."""


def decode_token(name: str, rows: int):
    """The layer's decode step for `rows` rows (`MixerCall(decode=True)`), jitted, and its arguments: the
    cache after a prefill of `PREFILL` tokens, and the next token."""
    module = GatedDeltaNet(**SHAPES[name], dtype=jnp.bfloat16)
    hidden = jax.random.normal(jax.random.key(0), (rows, PREFILL + 1, SHAPES[name]['emb_features']),
                               jnp.bfloat16)
    # Weights in bf16, as a serving checkpoint holds them: at a few rows the projections' reads are most
    # of a token's bytes.
    variables = jax.tree.map(lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
                             module.init(jax.random.key(1), hidden[:, :1]))
    call = MixerCall(decode=True)
    allocated = module.apply(variables, hidden[:, :1], call, mutable=['cache'])[1]
    _, cache = module.apply({**variables, **allocated}, hidden[:, :PREFILL], call, mutable=['cache'])

    @jax.jit
    def step(cache, token):
        return module.apply({**variables, **cache}, token, call, mutable=['cache'])

    return step, (cache, hidden[:, PREFILL:])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--shape', default='qwen3.5-27b-gdn', choices=sorted(SHAPES))
    parser.add_argument('--kernel', default=','.join(KERNELS))
    parser.add_argument('--rows', default=','.join(map(str, ROWS)))
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    failed = []
    for kernel in args.kernel.split(','):
        for rows in map(int, args.rows.split(',')):
            row = {'shape': args.shape, 'kernel': kernel, 'rows': rows,
                   'device': jax.devices()[0].device_kind}
            try:
                # The choice is made where the step traces: inside the measurement's first call.
                with generation.chosen(delta_decode=kernel):
                    step, given = decode_token(args.shape, rows)
                    row['ms'], row['kernels'] = measured(step, given, args.warmup, args.repeats)
            except Exception as error:  # a kernel the token refuses is a row of its own
                row['error'] = f"{type(error).__name__}: {error}"[:400]
                failed.append(f"{kernel} {rows}: {row['error'][:200]}")
                traceback.print_exc()
            line = json.dumps(row)
            print(line, flush=True)
            if args.out:
                with open(args.out, 'a') as handle:
                    handle.write(line + '\n')
    if failed:
        sys.exit("kernels failed:\n" + "\n".join(failed))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
