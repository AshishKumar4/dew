#!/usr/bin/env python3
"""Time the attention kernel alone, one implementation at a time.

`tools/benchmark_step.py` is the arbiter of what gets adopted: an
optimization counts only when the full training step is faster. This tool is
the cheaper question under that one, and the reason a candidate step change
is worth measuring at all: which kernel wins at this shape, forward-only and
forward+backward.

The four paths are the four attention_impl values the trainer can log:
the flax reference einsum, xla, cudnn, and the triton 'pallas' kernel.
Batch and head counts are picked so every implementation sees the same
token x head count, which is the fair comparison at a fixed sequence length.

Usage:
    python tools/benchmark_attention.py
    python tools/benchmark_attention.py --implementations pallas cudnn \\
        --sequence-lengths 4096 --head-dims 128
"""

import dataclasses
import json
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from dew.nn.attention import pallas_supports

JsonList = Annotated[
    list[dict[str, Any]],
    tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="JSON",
        instance_from_str=lambda args: json.loads(args[0]),
        is_instance=lambda value: isinstance(value, list),
        str_from_instance=lambda value: [json.dumps(value)],
    ),
]
"""A list of dicts, written as one JSON string on the command line."""

IMPLEMENTATIONS = ('reference', 'xla', 'cudnn', 'pallas')
# B chosen so batch * sequence * heads is constant across the sequence sweep:
# 512k query tokens at every row, so a slower row is a slower kernel, not a
# smaller one.
TOKEN_HEAD_BUDGET = 2 ** 19


@dataclass(frozen=True)
class BenchmarkConfig:
    """Which shapes to time, and how."""

    sequence_lengths: tuple[int, ...] = (256, 1024, 4096)
    head_dims: tuple[int, ...] = (64, 128)
    implementations: tuple[str, ...] = IMPLEMENTATIONS
    causal: tuple[bool, ...] = (False, True)
    forward_only: bool = False
    """Skip the gradient timings; the fwd row is the cheaper question."""
    warmup: int = 3
    steps: int = 20
    json_out: Optional[str] = None


def attention_fn(implementation: str, causal: bool):
    from dew.nn.attention import scaled_dot_product_attention
    return lambda q, k, v: scaled_dot_product_attention(
        q, k, v, implementation=None if implementation == 'reference' else implementation,
        causal=causal)


def measure(case: dict, config: BenchmarkConfig) -> Optional[dict]:
    """One row: implementation x shape x causality, forward and backward.

    Returns None when the kernel cannot serve the shape, which is itself a
    row of the answer rather than an error.
    """
    seq, head_dim = case['sequence_length'], case['head_dim']
    heads = max(1, TOKEN_HEAD_BUDGET // (seq * case['batch_size']))
    batch = case['batch_size']
    qkv = [jax.random.normal(
        jax.random.PRNGKey(i), (batch, seq, heads, head_dim), jnp.bfloat16)
        for i in range(3)]

    if case['implementation'] == 'pallas':
        refused = pallas_supports(qkv[0], qkv[1])
        if refused:
            return {**case, 'heads': heads, 'served': False, 'reason': refused}

    fn = jax.jit(attention_fn(case['implementation'], case['causal']))
    try:
        out = fn(*qkv)
        out.block_until_ready()
    except Exception as e:  # a kernel refusing this shape is a result, not a crash
        return {**case, 'heads': heads, 'served': False,
                'reason': f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"}

    def time_call(callable_, *args):
        for _ in range(config.warmup):
            jax.block_until_ready(callable_(*args))
        start = time.perf_counter()
        for _ in range(config.steps):
            jax.block_until_ready(callable_(*args))
        return (time.perf_counter() - start) / config.steps * 1e3

    row = {**case, 'heads': heads, 'served': True,
           'forward_ms': time_call(fn, *qkv)}

    if not config.forward_only:
        def loss(q, k, v):
            out = attention_fn(case['implementation'], case['causal'])(q, k, v)
            return jnp.sum(out.astype(jnp.float32))
        grad = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
        row['forward_backward_ms'] = time_call(grad, *qkv)
    return row


def cases(config: BenchmarkConfig) -> list[dict]:
    unknown = sorted(set(config.implementations) - set(IMPLEMENTATIONS))
    if unknown:
        raise ValueError(f"Unknown implementations {unknown}; "
                         f"valid: {list(IMPLEMENTATIONS)}")
    # B=2 at 4096 doubles the budget of the S=256 row; smaller batch at longer
    # sequence is what keeps activation memory flat across the sweep.
    batch_for = {256: 16, 1024: 4, 4096: 2}
    rows = []
    for seq in config.sequence_lengths:
        for head_dim in config.head_dims:
            for causal in config.causal:
                for implementation in config.implementations:
                    rows.append({
                        'implementation': implementation,
                        'sequence_length': seq,
                        'head_dim': head_dim,
                        'causal': causal,
                        'batch_size': batch_for.get(seq, 2),
                    })
    return rows


def format_table(rows: list[dict]) -> str:
    """One block per (sequence, head_dim, causal): implementations as rows."""
    keys = sorted({(r['sequence_length'], r['head_dim'], r['causal']) for r in rows})
    lines = [f"{'implementation':<11}{'fwd ms':>9}{'fwd+bwd ms':>11}   notes"]
    for seq, head_dim, causal in keys:
        block = [r for r in rows
                 if (r['sequence_length'], r['head_dim'], r['causal']) == (seq, head_dim, causal)]
        lines.append("")
        lines.append(f"S={seq} D={head_dim} causal={causal}, "
                     f"tokens*heads fixed at {TOKEN_HEAD_BUDGET}")
        for row in sorted(block, key=lambda r: IMPLEMENTATIONS.index(r['implementation'])):
            if not row.get('served'):
                lines.append(f"{row['implementation']:<11}{'-':>9}{'-':>11}   "
                              f"unsupported: {row['reason'][:60]}")
                continue
            fwd = row['forward_ms']
            bwd = row.get('forward_backward_ms')
            bwd_text = f"{bwd:.3f}" if bwd is not None else "-"
            lines.append(f"{row['implementation']:<11}{fwd:>9.3f}{bwd_text:>11}")
    return "\n".join(lines)


def main(config: BenchmarkConfig) -> list[dict]:
    print(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    rows = []
    for case in cases(config):
        row = measure(case, config)
        if row is None:
            continue
        rows.append(row)
        if row.get('served'):
            print(f"{row['implementation']} S={row['sequence_length']} "
                  f"D={row['head_dim']} causal={row['causal']}: "
                  f"fwd {row['forward_ms']:.3f} ms"
                  + (f", fwd+bwd {row['forward_backward_ms']:.3f} ms"
                     if 'forward_backward_ms' in row else ""))
        else:
            print(f"{row['implementation']} S={row['sequence_length']} "
                  f"D={row['head_dim']} causal={row['causal']}: unsupported")
        if config.json_out:
            with open(config.json_out, "w") as handle:
                json.dump(rows, handle, indent=2)
    print()
    print(format_table(rows))
    if config.json_out:
        print(f"\nWrote {config.json_out}")
    return rows


if __name__ == "__main__":
    main(tyro.cli(BenchmarkConfig))
