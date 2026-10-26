#!/usr/bin/env python3
"""The recurrence template's scoreboard: each real recurrent shape's chunked forward and forward plus
backward, timed in one GPU session for every arm that computes it, and one markdown row a shape.

Arms, each on the same operands:

- `template`: `recurrence_chunk.chunk`, its route as the GPU's row picks it (the conv is not in the
  template yet: it has no template arm);
- `hand`: the kernel the template replaces, called directly: the delta rule's port (`fla_rule`,
  `fla_gdn` or `fla_kda`), mamba_ssm's port (`mamba_ssd.ssd_scan`, a refusal where it does not take
  the shape), the short conv's kernel (`linear.causal_conv1d`);
- `xla`: the portable math in XLA (`chunk` with `generation.chosen`'s 'xla', `linear.xla_conv1d`);
- `fla`: fla's own torch/Triton kernels (`chunk_gated_delta_rule` on the queries and keys repeated
  over the value heads, as the layers hand them to it, `chunk_kda`, `causal_conv1d`);
- `mamba_ssm`: mamba_ssm's `mamba_chunk_scan_combined`;
- `flashqla`: FlashQLA's TileLang kernels (`flash_qla.chunk_gated_delta_rule`, Hopper and Blackwell only),
  the gated delta rule's queries and keys a key head each, as its API takes them.

An arm that runs out of memory records the error in its row.

The JAX arms run in Dew's environment (`jax` mode) and the torch arms in their own (`torch` mode, which
imports neither JAX nor Dew), one process each, one after another on the same GPU; `row` mode reads
their JSON lines and prints the markdown rows. Each time is the median of `--repeats` synchronized
calls after `--warmup`, in milliseconds; forward plus backward is one call returning the outputs and
their VJP under the cotangents `do` and `dfinal`, as torch's backward runs after its forward.

MFU is the shape's model FLOPs (`flops`: `dew.nn.flops`' recurrent function, the same count for every
arm, chunk work excluded) over the time and the GPU's dense bf16 peak (`dew.nn.kernels.generation`'s
table); forward plus backward is counted as three times the forward.

    PYTHONPATH=src python tools/benchmark_recurrence.py jax --shape all --out runs.jsonl
    $FLA_ENV/bin/python tools/benchmark_recurrence.py torch --shape all --arm fla --out runs.jsonl
    $MAMBA_ENV/bin/python tools/benchmark_recurrence.py torch --shape all --arm mamba_ssm --out runs.jsonl
    python tools/benchmark_recurrence.py row runs.jsonl --sha <sha>
"""

import argparse
import dataclasses
import datetime
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np


@dataclasses.dataclass(frozen=True)
class Shape:
    """A model's recurrent layer at one row of `length` tokens: `kind` 'gdn' (a decay a head),
    'kda' (a decay a key dimension), 'ssd' (Mamba2's, `key_heads` its groups, `key_width` its state,
    `value_width` its head width) or 'conv' (the short conv over the gated delta net's projection)."""
    kind: str
    key_heads: int
    value_heads: int
    key_width: int
    value_width: int
    chunk: int = 64
    taps: int = 4
    batch: int = 1
    length: int = 4096


SHAPES = {
    # tests/fixtures/hf/bonsai-27b-mlx-1bit/config.json (qwen3_5_text): 16 key heads, 48 value heads.
    'qwen3.5-27b-gdn': Shape('gdn', 16, 48, 128, 128),
    # tests/fixtures/hf/qwen3-coder-next-fp8/config.json (qwen3_next): 16 key heads, 32 value heads.
    'qwen3-next-gdn': Shape('gdn', 16, 32, 128, 128),
    # tests/fixtures/hf/kimi-linear-source/config.json: 32 KDA heads of 128.
    'kimi-linear-kda': Shape('kda', 32, 32, 128, 128),
    # tests/fixtures/hf/nemotron-h-4b/config.json: 96 heads of 80, 8 groups, state 128, chunk 256.
    'nemotron-h-4b-ssd': Shape('ssd', 8, 96, 128, 80, chunk=256),
    # tests/fixtures/hf/nemotron-h-30b-a3b/config.json: 64 heads of 64, 8 groups, state 128, chunk 128.
    'nemotron-h-30b-ssd': Shape('ssd', 8, 64, 128, 64, chunk=128),
    # The Qwen3.5-27B gated delta net's conv over its queries, keys and values.
    'qwen3.5-27b-conv': Shape('conv', 16, 48, 128, 128),
}
"""The real shapes. Dory 1.0 4B has none: its layers are attention and MLP (its config.json)."""

ARMS = {'gdn': ('template', 'hand', 'xla', 'fla', 'flashqla'), 'kda': ('template', 'hand', 'xla', 'fla'),
        'ssd': ('template', 'hand', 'xla', 'mamba_ssm'), 'conv': ('hand', 'xla', 'fla')}


def flops(shape: Shape) -> float:
    """The forward's model FLOPs, the recurrent function's (`dew.nn.flops`): the delta rule's
    `6 H K V` a token, the SSD's `4 H N P`, the depthwise conv's `2 C K_taps` over its channels."""
    from dew.nn import flops as model

    lengths = np.full(shape.batch, shape.length)
    if shape.kind == 'conv':
        channels = 2 * shape.key_heads * shape.key_width + shape.value_heads * shape.value_width
        return model.conv_flops(lengths, channels, channels, shape.taps, groups=channels)
    if shape.kind == 'ssd':
        return model.ssd_flops(lengths, shape.value_heads, shape.key_width, shape.value_width)
    return model.delta_flops(lengths, shape.value_heads, shape.key_width, shape.value_width)


def operands(shape: Shape, seed: int = 0) -> dict[str, np.ndarray]:
    """The arms' shared operands as fp32 host arrays, which each arm casts to bf16 as it takes them:
    l2-normalised queries and keys, values, a log decay in (-inf, 0) a head or a key dimension, a
    write gate in (0, 1), and the cotangents; for the SSD `x`, a step through softplus, `A`, `B`,
    `C` and `D`; for the conv a projection and its taps."""
    rng = np.random.default_rng(seed)
    batch, length = shape.batch, shape.length

    def normal(*dims):
        return rng.standard_normal(dims).astype(np.float32)

    if shape.kind == 'conv':
        channels = 2 * shape.key_heads * shape.key_width + shape.value_heads * shape.value_width
        return {'x': normal(batch, length, channels), 'taps': 0.5 * normal(channels, shape.taps),
                'dy': normal(batch, length, channels)}
    if shape.kind == 'ssd':
        heads, groups = shape.value_heads, shape.key_heads
        return {'x': normal(batch, length, heads, shape.value_width),
                'dt': np.log1p(np.exp(normal(batch, length, heads) - 2.0)),
                'A': -np.exp(rng.uniform(0.0, np.log(16.0), heads)).astype(np.float32),
                'B': normal(batch, length, groups, shape.key_width),
                'C': normal(batch, length, groups, shape.key_width), 'D': normal(heads),
                'dy': normal(batch, length, heads, shape.value_width),
                'dfinal': normal(batch, heads, shape.value_width, shape.key_width)}
    q, k = (normal(batch, length, shape.key_heads, shape.key_width) for _ in range(2))
    q, k = (x / np.linalg.norm(x, axis=-1, keepdims=True) for x in (q, k))
    decay = (batch, length, shape.value_heads) + ((shape.key_width,) if shape.kind == 'kda' else ())
    return {'q': q, 'k': k, 'v': normal(batch, length, shape.value_heads, shape.value_width),
            'g': -np.log1p(np.exp(normal(*decay) + 2.0)).astype(np.float32) / 8,
            'beta': 1 / (1 + np.exp(-normal(batch, length, shape.value_heads))),
            'do': normal(batch, length, shape.value_heads, shape.value_width),
            'dfinal': normal(batch, shape.value_heads, shape.key_width, shape.value_width)}


def jax_arms(name: str, shape: Shape, warmup: int, repeats: int) -> list[dict]:
    """The JAX arms' times on `shape`, each forward and forward plus backward."""
    import jax
    import jax.numpy as jnp

    from dew.nn import linear
    from dew.nn.kernels import generation, mamba_ssd, recurrence
    from dew.nn.kernels.recurrence_chunk import SCAN, Chunk, chunk, fla_rule, scan_case
    from dew.nn.mixers.mamba2 import SSD
    from dew.telemetry.benchmark import samples

    given = {key: jnp.asarray(value) for key, value in operands(shape).items()}
    bf16 = jnp.bfloat16
    refusals = {}
    if shape.kind == 'conv':
        x, taps = given['x'].astype(bf16), given['taps'].astype(bf16)
        arms = {'hand': lambda x, taps: linear.causal_conv1d(x, taps),
                'xla': lambda x, taps: linear.xla_conv1d(x, taps, activation=True)}
        inputs, cotangents = (x, taps), given['dy'].astype(bf16)
    else:
        if shape.kind == 'ssd':
            x, B, C = (given[key].astype(bf16) for key in ('x', 'B', 'C'))
            inputs = (x, given['dt'], given['A'], B, C, given['D'])
            state = jnp.zeros((shape.batch, shape.value_heads, shape.value_width, shape.key_width))

            def template(x, dt, A, B, C, D):
                out, final = chunk(SSD, Chunk(C, B, x, beta=dt, rate=A, skip=D), chunk_size=shape.chunk,
                                   dtype=jnp.float32)
                return out, jnp.swapaxes(final, -1, -2)

            def hand(x, dt, A, B, C, D):
                return mamba_ssd.ssd_scan(x, dt, A, B, C, D, state, None, shape.chunk,
                                          not generation.triton_runs())

            # What the kernels take, whatever this GPU's row prefers.
            case = scan_case(x, B, shape.chunk)
            refusals['hand'] = '; '.join(requirement.refusal(case).detail
                                         for requirement in SCAN.kernel('pallas').failed(case))
            cotangents = (given['dy'], given['dfinal'])
        else:
            hooks = recurrence.Hooks(decay='vector' if shape.kind == 'kda' else 'scalar',
                                     scale=shape.key_width ** -0.5)
            inputs = (*(given[key].astype(bf16) for key in ('q', 'k', 'v')), given['g'],
                      given['beta'].astype(bf16))

            def template(q, k, v, g, beta):
                return chunk(hooks, Chunk(q, k, v, log_decay=g, beta=beta), chunk_size=shape.chunk)

            def hand(q, k, v, g, beta):
                return fla_rule(q, k, v, g, beta, scale=shape.key_width ** -0.5)

            cotangents = (given['do'], given['dfinal'])

        def xla(*args):
            with generation.chosen(gated_delta_rule='xla', ssd='xla'):
                return template(*args)

        arms = {'template': template, 'hand': hand, 'xla': xla}

    def forward_backward(fn):
        def run(*args):
            outputs, pull = jax.vjp(fn, *args)
            return outputs, pull(jax.tree.map(lambda c, o: c.astype(o.dtype), cotangents, outputs))
        return run

    rows = []
    for arm, fn in arms.items():
        row = {'shape': name, 'arm': arm}
        rows.append(row)
        if refusals.get(arm):
            row.update(fwd_ms=None, fwd_bwd_ms=None, error=f"refuses: {refusals[arm]}")
            continue
        for part, compiled in (('fwd_ms', jax.jit(fn)), ('fwd_bwd_ms', jax.jit(forward_backward(fn)))):
            try:
                measured = samples(compiled, *inputs, warmup=warmup, repeats=repeats)
                row[part] = 1e3 * statistics.median(measured)
            except jax.errors.JaxRuntimeError as error:
                row[part], row['error'] = None, _failure(error)
    return rows


def _failure(error: BaseException) -> str:
    return f"{type(error).__name__}: {str(error).splitlines()[0][:200]}"


def torch_arm(name: str, shape: Shape, arm: str, warmup: int, repeats: int) -> dict:
    """fla's, FlashQLA's or mamba_ssm's kernels on `shape`, timed by CUDA events. fla's arm runs fla's own
    Triton kernels: fla dispatches its gated delta rule to FlashQLA where that is installed and the GPU
    takes it (`fla.ops.gated_delta_rule.backends.flash_qla`), which `FLA_DISABLE_BACKEND_DISPATCH` turns
    off before fla is imported. An arm whose kernels refuse this GPU records the refusal."""
    import os

    if arm == 'fla':
        os.environ['FLA_DISABLE_BACKEND_DISPATCH'] = '1'
    import torch

    given = {key: torch.tensor(value, device='cuda') for key, value in operands(shape).items()}
    bf16 = torch.bfloat16
    if arm == 'mamba_ssm':
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        leaves = [given['x'].to(bf16), given['dt'], given['A'], given['B'].to(bf16), given['C'].to(bf16),
                  given['D']]

        def run(x, dt, A, B, C, D):
            return mamba_chunk_scan_combined(x, dt, A, B, C, shape.chunk, D=D, return_final_states=True)

        cotangents = (given['dy'].to(bf16), given['dfinal'])
    elif arm == 'flashqla':
        major, minor = torch.cuda.get_device_capability()
        try:
            from flash_qla import chunk_gated_delta_rule as rule
        except ValueError as error:  # FlashQLA refuses an architecture it has no kernels for at import
            return {'shape': name, 'arm': arm, 'length': shape.length, 'device': torch.cuda.get_device_name(),
                    'fwd_ms': None, 'fwd_bwd_ms': None, 'error': f"refuses sm{major}{minor}: {error}"}

        leaves = [given['q'].to(bf16), given['k'].to(bf16), given['v'].to(bf16), given['g'],
                  given['beta'].to(bf16)]

        def run(q, k, v, g, beta):
            return rule(q=q, k=k, v=v, g=g, beta=beta, scale=shape.key_width ** -0.5, output_final_state=True)

        cotangents = (given['do'].to(bf16), given['dfinal'])
    elif shape.kind == 'conv':
        from fla.modules.convolution import causal_conv1d

        leaves = [given['x'].to(bf16), given['taps'].to(bf16)]

        def run(x, taps):
            return (causal_conv1d(x, taps, activation='silu')[0],)

        cotangents = (given['dy'].to(bf16),)
    else:
        if shape.kind == 'kda':
            from fla.ops.kda import chunk_kda as rule
        else:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule as rule
        group = shape.value_heads // shape.key_heads
        leaves = [given['q'].to(bf16), given['k'].to(bf16), given['v'].to(bf16), given['g'],
                  given['beta'].to(bf16)]

        def run(q, k, v, g, beta):
            # The layers' repeat, inside the timed call so its transpose sums each key head's gradients.
            q, k = (x.repeat_interleave(group, dim=2) for x in (q, k))
            return rule(q, k, v, g, beta, scale=shape.key_width ** -0.5, output_final_state=True)

        cotangents = (given['do'].to(bf16), given['dfinal'])
    leaves = [leaf.requires_grad_() for leaf in leaves]

    def forward():
        with torch.no_grad():
            return run(*leaves)

    def backward():
        outputs = run(*leaves)
        torch.autograd.backward(list(outputs),
                                [c.to(o.dtype) for o, c in zip(outputs, cotangents, strict=True)])
        for leaf in leaves:
            leaf.grad = None

    row = {'shape': name, 'arm': arm, 'device': torch.cuda.get_device_name()}
    for part, fn in (('fwd_ms', forward), ('fwd_bwd_ms', backward)):
        try:
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            measured = []
            for _ in range(repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                measured.append(start.elapsed_time(end))
            row[part] = statistics.median(measured)
        except torch.OutOfMemoryError as error:
            row[part], row['error'] = None, _failure(error)
            torch.cuda.empty_cache()
    return row


def markdown(rows: list[dict], sha: str) -> list[str]:
    """One row a shape: each arm's forward and forward plus backward in ms and their MFU."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from dew.nn.kernels.generation import gpu_target

    lines = []
    runs = sorted({(row['shape'], row['length'], row['device']) for row in rows if row['shape'] in SHAPES})
    for name, length, device in runs:
        # The last measurement of each arm at this shape, length and device.
        found = {row['arm']: row for row in rows
                 if (row['shape'], row['length'], row['device']) == (name, length, device)}
        shape = dataclasses.replace(SHAPES[name], length=length)
        peak = gpu_target(device).bf16_peak
        work = flops(shape)
        cells = []
        for arm in ARMS[shape.kind]:
            row = found.get(arm)
            if row is None or row.get('fwd_ms') is None:
                cells.append(f"{arm} {row.get('error', 'failed') if row else 'not run'}")
                continue
            parts = []
            for key, factor in (('fwd_ms', 1), ('fwd_bwd_ms', 3)):
                ms = row.get(key)
                mfu = '' if ms is None or not peak else f" ({100 * factor * work / (ms * 1e-3) / peak:.1f}%)"
                parts.append('n/a' if ms is None else f"{ms:.3f}{mfu}")
            cells.append(f"{arm} {parts[0]} / {parts[1]}")
        lines.append(f"- **Recurrence scoreboard, {name}** ({shape.kind}, B={shape.batch} T={shape.length} "
                     f"Hk={shape.key_heads} Hv={shape.value_heads} K={shape.key_width} V={shape.value_width} "
                     f"C={shape.chunk}, bf16) · {device} · {datetime.date.today().isoformat()} · `{sha}` · "
                     f"fwd / fwd+bwd ms (MFU of {flops(shape) / 1e9:.1f} GFLOP fwd at the bf16 peak): "
                     + '; '.join(cells))
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('mode', choices=('jax', 'torch', 'row'))
    parser.add_argument('runs', nargs='?', help="row mode: the JSON lines the other modes wrote")
    parser.add_argument('--shape', default='all', help=f"one of {sorted(SHAPES)} or 'all'")
    parser.add_argument('--arm', choices=('fla', 'flashqla', 'mamba_ssm'), help='torch mode: whose kernels')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--out', help='append JSON lines here as well as printing them')
    parser.add_argument('--sha', default='unknown')
    parser.add_argument('--length', type=int, help="every shape's tokens, for a smoke run of the tool")
    args = parser.parse_args(argv)
    if args.mode == 'row':
        rows = [json.loads(line) for line in Path(args.runs).read_text().splitlines() if line.strip()]
        print('\n'.join(markdown(rows, args.sha)))
        return
    names = sorted(SHAPES) if args.shape == 'all' else [args.shape]
    for name in names:
        shape = SHAPES[name] if args.length is None else dataclasses.replace(SHAPES[name], length=args.length)
        if args.mode == 'jax':
            import jax
            found = jax_arms(name, shape, args.warmup, args.repeats)
            for row in found:
                row['device'] = jax.devices()[0].device_kind
        elif args.arm in ARMS[shape.kind]:
            found = [torch_arm(name, shape, args.arm, args.warmup, args.repeats)]
        else:
            continue
        for row in found:
            row.update(time=time.time(), length=shape.length)
            line = json.dumps(row)
            print(line, flush=True)
            if args.out:
                with open(args.out, 'a') as handle:
                    handle.write(line + '\n')


if __name__ == '__main__':
    main()
