#!/usr/bin/env python3
"""Time the Mamba-2 SSD chunked scan: the template's portable math in XLA against the Pallas kernel.

`dew.nn.mixers.mamba2.chunk_ssd` picks between the recurrence template's
portable math (`portable_scan`) and `dew.nn.kernels.ssd.ssd_chunk_scan` from
the backend and the geometry. This
tool times the two on the same chunked operands, forward and forward+backward,
so the selection rule in `ssd_kernel_runs` can be checked against numbers
instead of against the tile arithmetic its constants are written from. Every
row also says whether that rule would have chosen the kernel for its geometry.

The scan alone is what is timed. The padding, the group expansion, the `dt`
scaling and the `D` skip around it are the same operations on both paths, so
they would add the same constant to both rows; the transposes the kernel needs
are inside `ssd_chunk_scan` and are timed with it.

This needs a TPU to mean anything. On CPU the kernel runs through
pallas' interpreter, which is orders of magnitude slower than either path and
is only good for `--parity`, where the row carries the kernel's largest
difference from the XLA path instead of a time.

Usage:
    python tools/benchmark_ssd.py
    python tools/benchmark_ssd.py --sequence-lengths 4096 --chunk-size 128
    JAX_PLATFORMS=cpu python tools/benchmark_ssd.py --parity --no-timing \\
        --sequence-lengths 1024 --batch-size 1 --heads 2
"""

import json
import statistics
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import tyro

from dew.nn.kernels.recurrence_chunk import chunk_delta_rule
from dew.nn.kernels.ssd import ssd_chunk_scan, ssd_kernel_runs
from dew.telemetry.benchmark import largest_difference, samples

IMPLEMENTATIONS = ('xla', 'kernel')


@dataclass(frozen=True)
class BenchmarkConfig:
    """Which scans to time, and how."""

    sequence_lengths: tuple[int, ...] = (4096, 16384, 65536)
    batch_size: int = 1
    heads: int = 8
    head_dim: int = 64
    """The `P` of the `[P, N]` state each head carries."""
    state_size: int = 128
    """The `N` of it, which `B` and `C` project into and out of."""
    chunk_size: int = 256
    """Steps per chunk, `Mamba2.chunk_size`. Sequence lengths are multiples of
    it here, so no row pays for a ragged tail the others do not."""
    platform: str | None = None
    """The backend to build the kernel for; the default is the one running."""
    timing: bool = True
    """Off for a correctness-only run, which is all CPU can give."""
    forward_only: bool = False
    parity: bool = False
    """Also record the kernel's largest difference from the XLA path."""
    warmup: int = 3
    steps: int = 10
    json_out: str | None = None


def operands(sequence_length: int, config: BenchmarkConfig):
    """The scan's operands as `chunk_ssd` hands them over, with the groups
    already expanded: `x` `[B, T, H, P]` carrying its step, `B` and `C`
    `[B, T, H, N]`, `A dt` `[B, T, H]` and the state `[B, H, P, N]`."""
    rows = (config.batch_size, sequence_length, config.heads)
    keys = jax.random.split(jax.random.key(0), 5)
    step = jax.nn.softplus(jax.random.normal(keys[3], rows))
    decay = -jnp.exp(jax.random.normal(keys[4], (config.heads,))) * step
    return (jax.random.normal(keys[0], (*rows, config.head_dim), jnp.float32),
            jax.random.normal(keys[1], (*rows, config.state_size), jnp.float32),
            jax.random.normal(keys[2], (*rows, config.state_size), jnp.float32),
            jnp.asarray(decay, jnp.float32),
            jnp.zeros((config.batch_size, config.heads, config.head_dim, config.state_size),
                      jnp.float32))


def kernel_platform(config: BenchmarkConfig) -> str:
    """The backend the kernel is built for. A cpu host has no kernel to build,
    so one has to be named and pallas interprets it."""
    platform = config.platform or jax.default_backend()
    if platform != 'tpu':
        raise ValueError(f"the ssd kernel is written for tpu, not {platform!r}; name it "
                         f"with --platform tpu to read it through pallas' interpreter "
                         f"on this host")
    return platform


def portable_scan(x, B, C, a, carried, chunk_size: int):
    """The scan as the recurrence template's portable math runs it, on `ssd_chunk_scan`'s operands
    (`x` scaled by `dt`, `B` and `C` a head each, `a` the per-step `A dt`, `carried` `[B, H, P, N]`):
    `chunk_delta_rule` without the delta rule, every write gate one."""
    out, final = chunk_delta_rule(C, B, x, a[..., None], jnp.ones(a.shape, a.dtype),
                                  jnp.swapaxes(carried, -1, -2), chunk_size, delta=False, scale=1.0)
    return out, jnp.swapaxes(final, -1, -2)


def scan_fn(implementation: str, platform: str, chunk_size: int):
    if implementation == 'kernel':
        return lambda *args: ssd_chunk_scan(*args, chunk_size, platform)
    return lambda *args: portable_scan(*args, chunk_size)


def failure(error: BaseException) -> str:
    return f"{type(error).__name__}: {str(error).splitlines()[0][:160]}"


def measure(case: dict, config: BenchmarkConfig) -> dict:
    """One row: implementation x sequence length, forward and forward+backward.

    A path that cannot serve the shape is a row with served=False and the
    reason it gave; that is part of the answer. Pallas raises a ValueError for
    a block shape its backend will not tile and a JaxRuntimeError when the
    compiled kernel asks for more shared memory than the card has.
    """
    platform = kernel_platform(config)
    built = operands(case['sequence_length'], config)
    cotangents = tuple(jax.random.normal(key, shape, jnp.float32)
                       for key, shape in zip(jax.random.split(jax.random.key(1)),
                                             (built[0].shape, built[4].shape), strict=True))
    scan = scan_fn(case['implementation'], platform, config.chunk_size)
    row = {**case, 'platform': platform,
           'selected': ssd_kernel_runs(config.chunk_size, config.head_dim,
                                       config.state_size, platform, dtype=jnp.float32)}
    try:
        forward = jax.jit(scan)
        scanned = jax.block_until_ready(forward(*built))
    except (ValueError, NotImplementedError, jax.errors.JaxRuntimeError) as error:
        return {**row, 'served': False, 'reason': failure(error)}

    def trained(*args):
        # The scan's results returned with their gradients, as a training step reads both: a pass
        # returning the gradients alone leaves XLA to drop every forward launch the backward does not
        # read (the forward's readout).
        out, pull = jax.vjp(scan, *args)
        return out, pull(cotangents)

    row['served'] = True
    gradient = jax.jit(trained)
    gradients = None
    if not config.forward_only:
        try:
            gradients = jax.block_until_ready(gradient(*built))[1]
        except (ValueError, NotImplementedError, jax.errors.JaxRuntimeError) as error:
            row['backward_reason'] = failure(error)

    def time_call(callable_):
        return statistics.mean(samples(callable_, *built, warmup=config.warmup, repeats=config.steps)) * 1e3

    if config.timing:
        row['forward_ms'] = time_call(forward)
        if gradients is not None:
            row['forward_backward_ms'] = time_call(gradient)
    if config.parity and case['implementation'] == 'kernel':
        reference = scan_fn('xla', platform, config.chunk_size)
        expected = jax.jit(reference)(*built)
        row['forward_error'] = largest_difference(scanned, expected)
        row['forward_scale'] = float(jnp.max(jnp.abs(expected[0])))
        if gradients is not None:
            wanted = jax.jit(jax.grad(
                lambda *a: (jnp.sum(reference(*a)[0] * cotangents[0])
                            + jnp.sum(reference(*a)[1] * cotangents[1])),
                argnums=(0, 1, 2, 3, 4)))(*built)
            row['gradient_error'] = largest_difference(gradients, wanted)
            row['gradient_scale'] = float(max(jnp.max(jnp.abs(g)) for g in wanted))
    return row


def cases(config: BenchmarkConfig) -> list[dict]:
    ragged = [length for length in config.sequence_lengths if length % config.chunk_size]
    if ragged:
        raise ValueError(f"sequence lengths {ragged} are not whole chunks of "
                         f"{config.chunk_size}; the scan pads them and the rows stop comparing")
    return [{'implementation': implementation, 'sequence_length': length}
            for length in config.sequence_lengths for implementation in IMPLEMENTATIONS]


def column(row: dict, key: str, width: int, digits: str) -> str:
    """One measurement in its column, or a dash where the row has none."""
    return f"{'-':>{width}}" if key not in row else f"{row[key]:>{width}{digits}}"


def format_table(rows: list[dict], config: BenchmarkConfig) -> str:
    """One block per sequence length, the two paths as its rows. The error
    columns are the kernel's distance from the XLA path, so they are the XLA
    row's own reference and empty there."""
    header = f"{'path':<8}{'fwd ms':>10}{'fwd+bwd ms':>12}{'fwd err':>11}{'grad err':>11}   notes"
    lines = [f"batch {config.batch_size}, {config.heads} heads of {config.head_dim}, "
             f"state {config.state_size}, chunk {config.chunk_size}", header]
    for length in dict.fromkeys(row['sequence_length'] for row in rows):
        block = [row for row in rows if row['sequence_length'] == length]
        lines.append("")
        lines.append(f"S={length}, {length // config.chunk_size} chunks, "
                     f"kernel selected by the rule: {block[0]['selected']}")
        for row in sorted(block, key=lambda r: IMPLEMENTATIONS.index(r['implementation'])):
            if not row.get('served'):
                lines.append(f"{row['implementation']:<8}{'-':>10}{'-':>12}{'-':>11}{'-':>11}   "
                             f"refused: {row['reason'][:60]}")
                continue
            notes = [row['backward_reason']] if 'backward_reason' in row else []
            if 'forward_scale' in row:
                notes.append(f"against outputs up to {row['forward_scale']:.1f}")
            if 'gradient_scale' in row:
                notes.append(f"and gradients up to {row['gradient_scale']:.1f}")
            lines.append(
                f"{row['implementation']:<8}{column(row, 'forward_ms', 10, '.3f')}"
                f"{column(row, 'forward_backward_ms', 12, '.3f')}"
                f"{column(row, 'forward_error', 11, '.2e')}"
                f"{column(row, 'gradient_error', 11, '.2e')}   {' '.join(notes)}")
    return "\n".join(lines)


def main(config: BenchmarkConfig) -> None:
    rows = [measure(case, config) for case in cases(config)]
    print(format_table(rows, config))
    if config.json_out:
        with open(config.json_out, 'w') as out:
            json.dump(rows, out, indent=2)
        print(f"\nwrote {config.json_out}")


if __name__ == '__main__':
    main(tyro.cli(BenchmarkConfig))
