#!/usr/bin/env python3
"""Time the attention kernel alone, one implementation at a time.

`tools/benchmark_step.py` is the arbiter of what gets adopted: an
optimization counts only when the full training step is faster. This tool is
the cheaper question under that one, and the reason a candidate step change
is worth measuring at all: which kernel wins at this shape, forward-only and
forward+backward.

Three of the paths are the attention kernels the trainer can log for a GPU
run: the flax reference einsum, xla, and cudnn. The fourth, `triton`, is
tokamax's Pallas-Triton flash attention, called directly through
`tokamax.dot_product_attention`; Dew has no route to it, and this tool is
where the case for one is measured. Batch and head counts are picked so
every implementation sees the same token x head count, which is the fair
comparison at a fixed sequence length.

Usage:
    python tools/benchmark_attention.py
    python tools/benchmark_attention.py --implementations cudnn xla \\
        --sequence-lengths 4096 --head-dims 128
    python tools/benchmark_attention.py --implementations xla triton \\
        --head-dims 256 --kv-groups 2 --causal True --reference-error \\
        --triton-device-kind "NVIDIA GeForce RTX 4090"
"""

import contextlib
import importlib
import json
import time
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import tyro

IMPLEMENTATIONS = ('reference', 'xla', 'cudnn', 'triton')
# B chosen so batch * sequence * heads is constant across the sequence sweep:
# 512k query tokens at every row, so a slower row is a slower kernel, not a
# smaller one.
TOKEN_HEAD_BUDGET = 2 ** 19


@dataclass(frozen=True)
class BenchmarkConfig:
    """Which shapes to time, and how."""

    sequence_lengths: tuple[int, ...] = (256, 1024, 4096)
    head_dims: tuple[int, ...] = (64, 128)
    implementations: tuple[str, ...] = IMPLEMENTATIONS[:3]
    causal: tuple[bool, ...] = (False, True)
    kv_groups: int = 1
    """Query heads per key/value head; 1 is multi-head attention."""
    sliding_window: Optional[int] = None
    softcap: Optional[float] = None
    """Gemma 2's tanh logit cap. Only the reference, xla and triton paths take it."""
    forward_only: bool = False
    """Skip the gradient timings; the fwd row is the cheaper question."""
    reference_error: bool = False
    """Also run the reference path in fp32 on the same inputs and record each
    kernel's largest output and gradient difference from it. The fp32 logits
    are materialized, so this fits only where the reference row does."""
    triton_device_kind: Optional[str] = None
    """A device kind from JAX's Pallas-Triton table to compile the triton
    rows for, when the local card is missing from that table. JAX 0.11.1
    lists the RTX 4090 but not the RTX 4080; both are compute capability
    8.9. Applied through jax.sharding.use_abstract_mesh."""
    warmup: int = 3
    steps: int = 20
    json_out: Optional[str] = None


def attention_fn(implementation: str, config: BenchmarkConfig, causal: bool, dtype=None):
    from dew.nn.attention import scaled_dot_product_attention
    window, softcap = config.sliding_window, config.softcap
    if implementation == 'triton':
        # tokamax is not a dependency (docs/concepts/moe.md), so it is
        # imported at the call.
        tokamax = importlib.import_module('tokamax')
        local = None if window is None else (window - 1, 0)
        return lambda q, k, v: tokamax.dot_product_attention(
            q, k, v, is_causal=causal, local_window_size=local,
            logits_soft_cap=softcap, implementation='triton')
    return lambda q, k, v: scaled_dot_product_attention(
        q, k, v, implementation=None if implementation == 'reference' else implementation,
        causal=causal, sliding_window=window, softcap=softcap, dtype=dtype)


def lowering_target(implementation: str, config: BenchmarkConfig):
    if implementation != 'triton' or config.triton_device_kind is None:
        return contextlib.nullcontext()
    device = jax.sharding.AbstractDevice(config.triton_device_kind, None, 'cuda')
    return jax.sharding.use_abstract_mesh(
        jax.sharding.AbstractMesh((), (), abstract_device=device))


def failure(error: BaseException) -> str:
    return f"{type(error).__name__}: {str(error).splitlines()[0][:160]}"


def largest_difference(left, right) -> float:
    return max(float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))
               for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True))


def measure(case: dict, config: BenchmarkConfig) -> dict:
    """One row: implementation x shape x causality, forward and backward.

    A kernel that cannot serve the shape is a row with served=False and the
    reason; the row is part of the answer, not an error. jax raises
    NotImplementedError for a shape its cudnn checks refuse, a
    JaxRuntimeError when the materialized logits do not fit in memory or a
    Pallas kernel exceeds the card's shared memory, and ImportError when
    tokamax is not installed. A forward that runs but a backward that does
    not is a served row with `backward_reason`.
    """
    seq, head_dim = case['sequence_length'], case['head_dim']
    heads = max(1, TOKEN_HEAD_BUDGET // (seq * case['batch_size']))
    if heads % config.kv_groups:
        raise ValueError(f"{heads} query heads at S={seq} do not split into "
                         f"kv_groups={config.kv_groups}")
    batch = case['batch_size']
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    q = jax.random.normal(keys[0], (batch, seq, heads, head_dim), jnp.bfloat16)
    k = jax.random.normal(keys[1], (batch, seq, heads // config.kv_groups, head_dim), jnp.bfloat16)
    v = jax.random.normal(keys[2], (batch, seq, heads // config.kv_groups, head_dim), jnp.bfloat16)
    cotangent = jax.random.normal(keys[3], q.shape, jnp.float32)
    causal = case['causal']
    row = {**case, 'heads': heads, 'kv_heads': heads // config.kv_groups}

    with lowering_target(case['implementation'], config):
        try:
            kernel = attention_fn(case['implementation'], config, causal)
            fn = jax.jit(kernel)
            out = fn(q, k, v)
            out.block_until_ready()
        except (ImportError, NotImplementedError, ValueError, jax.errors.JaxRuntimeError) as e:
            return {**row, 'served': False, 'reason': failure(e)}

        def time_call(callable_, *args):
            for _ in range(config.warmup):
                jax.block_until_ready(callable_(*args))
            start = time.perf_counter()
            for _ in range(config.steps):
                jax.block_until_ready(callable_(*args))
            return (time.perf_counter() - start) / config.steps * 1e3

        row.update(served=True, forward_ms=time_call(fn, q, k, v))

        def loss(q, k, v):
            return jnp.sum(kernel(q, k, v).astype(jnp.float32) * cotangent)

        grads = None
        if not config.forward_only:
            grad = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
            try:
                grads = grad(q, k, v)
                jax.block_until_ready(grads)
                row['forward_backward_ms'] = time_call(grad, q, k, v)
            except (NotImplementedError, jax.errors.JaxRuntimeError) as e:
                row['backward_reason'] = failure(e)

    if config.reference_error:
        # The reference einsum in fp32 on the same bf16 values, the number
        # every kernel row is measured against.
        reference = attention_fn('reference', config, causal, dtype=jnp.float32)
        expected = jax.jit(reference)(q, k, v)
        row['forward_error'] = largest_difference(out, expected)
        if grads is not None:
            expected_grads = jax.jit(jax.grad(
                lambda q, k, v: jnp.sum(reference(q, k, v) * cotangent), argnums=(0, 1, 2)))(q, k, v)
            row['gradient_error'] = largest_difference(grads, expected_grads)
            row['gradient_scale'] = float(max(jnp.max(jnp.abs(g)) for g in expected_grads))
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
    lines = [f"{'implementation':<11}{'fwd ms':>9}{'fwd+bwd ms':>11}{'fwd err':>10}{'grad err':>10}   notes"]
    for seq, head_dim, causal in keys:
        block = [r for r in rows
                 if (r['sequence_length'], r['head_dim'], r['causal']) == (seq, head_dim, causal)]
        lines.append("")
        lines.append(f"S={seq} D={head_dim} causal={causal}, "
                     f"tokens*heads fixed at {TOKEN_HEAD_BUDGET}")
        for row in sorted(block, key=lambda r: IMPLEMENTATIONS.index(r['implementation'])):
            if not row.get('served'):
                lines.append(f"{row['implementation']:<11}{'-':>9}{'-':>11}{'-':>10}{'-':>10}   "
                             f"unsupported: {row['reason'][:60]}")
                continue
            bwd = row.get('forward_backward_ms')
            cells = (f"{row['forward_ms']:.3f}",
                     f"{bwd:.3f}" if bwd is not None else "-",
                     f"{row['forward_error']:.4f}" if 'forward_error' in row else "-",
                     f"{row['gradient_error']:.4f}" if 'gradient_error' in row else "-")
            note = f"   backward: {row['backward_reason'][:60]}" if 'backward_reason' in row else ""
            lines.append(f"{row['implementation']:<11}{cells[0]:>9}{cells[1]:>11}"
                         f"{cells[2]:>10}{cells[3]:>10}{note}")
    return "\n".join(lines)


def main(config: BenchmarkConfig) -> list[dict]:
    print(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    rows = []
    for case in cases(config):
        row = measure(case, config)
        rows.append(row)
        label = (f"{row['implementation']} S={row['sequence_length']} "
                 f"D={row['head_dim']} causal={row['causal']}")
        if row.get('served'):
            print(f"{label}: fwd {row['forward_ms']:.3f} ms"
                  + (f", fwd+bwd {row['forward_backward_ms']:.3f} ms"
                     if 'forward_backward_ms' in row else "")
                  + (f", backward failed: {row['backward_reason']}"
                     if 'backward_reason' in row else ""))
        else:
            print(f"{label}: unsupported: {row['reason']}")
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
