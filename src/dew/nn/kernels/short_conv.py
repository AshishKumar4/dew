"""The short causal conv of the recurrent mixers, one Pallas-Triton kernel each way.

`linear.causal_conv1d` is the depthwise causal conv over a `[B, T, C]` projection with `[C, K]` taps
(K a few tokens), its bias and silu, which the gated delta net, Kimi's delta attention and Mamba run
before their scans; the delta rules then l2-normalise each query and key head. As XLA computes it, the
backward reads and writes the `[B, T, C]` activation once for each of the norm's, silu's and the conv's
transposes, several of them in fp32. Here the forward reads the projection once and writes each output
piece once (the queries, keys and values as the scans take them, so no copy splits them), and the
backward reads the projection and the cotangents once and writes the projection's gradient once,
everything between in fp32 registers. Neither saves anything but the projection. Each piece is its
own call, so a program knows statically whether it normalises and which cotangent it reads; the
backward's calls write their columns of one gradient, each taking the last one's as its output.

A kernel that reads a time block's inputs at the K shifts of the conv, as fla's causal_conv1d does
(fla/modules/conv/triton/kernels.py, v0.5.2), would need the backward's pre-activation gradient at
the K-1 rows past its block, and a Triton program cannot shift a register tile: fla writes the
pre-activation to memory in a separate forward pass and reads it back at each shift, and fusing the
norm on top would recompute the conv, silu and norm K times a row, about 80 fp32 operations an element
against the 6 bytes it moves. Each program here instead walks time, as causal-conv1d's channel-last
backward does (Dao-AILab/causal-conv1d, csrc/causal_conv1d_bwd.cu): a tile is `segments` rows, each
the next token of its own run of `steps` tokens, and a loop steps all of them forward one token at a
time, carrying the last K-1 inputs (and in the backward the last K-1 pre-activation gradients) from
step to step, so each token's conv, silu, norm and their gradients are computed once. A run reads the
K-1 tokens before it and the backward computes K-1 past its end, `(K - 1) / steps` extra.

The arithmetic is `causal_conv1d`'s in fp32 with one rounding at each output: a query or key head is
`a * rsqrt(sum(a^2) + eps)` of its silu output `a`, `linear.l2norm`'s.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from . import generation

SEGMENTS = 8
"""Rows of a program's tile, each its own run of tokens: with a 128-channel block, 1024 elements, eight
an fp32 register of each of four warps' threads."""

STEPS = 64
"""Tokens of a run: its K-1 extra tokens are 5% of 64 at the K of 4 every mixer uses."""

WARPS = 4

WIDEST_TAPS = 8
"""The longest conv the kernels unroll: each tap is a register tile the walk carries."""

DTYPES = (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))


def block_width(width: int, head: int | None) -> int:
    """The channels a program of a `width`-channel piece holds: a normalised piece's `head`, which its
    row sum needs whole, else the widest power of two up to 128 that divides the piece."""
    if head is not None:
        return head
    block = 128
    while block > 1 and width % block:
        block //= 2
    return block


def _fits(width: int, head: int | None) -> bool:
    """Whether a piece's block tiles it: a power of two of 16 to 256 channels dividing it, at most 128
    where it is not normalised."""
    block = block_width(width, head)
    return block & (block - 1) == 0 and 16 <= block <= (128 if head is None else 256) and width % block == 0


def refusal(x, taps, parts, dtype) -> str | None:
    """Why the kernels do not take a conv of `x` `[B, T, C]` by `taps` `[C, K]` into `parts` in `dtype`,
    or None where they do."""
    channels = x.shape[-1]
    return generation.first_refusal(
        (x.ndim == 3 and x.shape[1] > 0, f"a conv over {x.shape}; the kernels take [B, T, C]"),
        (jnp.dtype(x.dtype) in DTYPES and jnp.dtype(dtype) in DTYPES,
         f"{x.dtype} inputs into {jnp.dtype(dtype)}; the kernels read and write float32, bfloat16 or "
         "float16, summing in float32"),
        (2 <= taps.shape[-1] <= WIDEST_TAPS, f"{taps.shape[-1]} taps; the kernels unroll 2 to {WIDEST_TAPS}"),
        (sum(width for width, _ in parts) == channels, f"pieces {parts} of {channels} channels"),
        (all(_fits(width, head) for width, head in parts),
         f"pieces {parts}: each a multiple of its block, a normalised piece's power-of-two head up to "
         "256, another's 16 to 128 channels"),
    )


def _program(first: int, block: int, segments: int, steps: int):
    """This program's channels of the projection `[block]`, from the piece's `first`, and the first
    token of each of its runs `[segments]`."""
    columns = first + pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    starts = (pl.program_id(1) * segments + jnp.arange(segments, dtype=jnp.int32)) * steps
    return columns, starts


def _rows(ref, tokens, columns, length: int):
    """`ref`'s `[segments, block]` tile at `tokens`, zero before the first token and past the last."""
    from jax.experimental.pallas import triton as plgpu

    inside = (tokens >= 0) & (tokens < length)
    clipped = jnp.clip(tokens, 0, length - 1)
    return plgpu.load(ref.at[clipped[:, None], columns[None, :]], mask=inside[:, None],
                      other=0).astype(jnp.float32)


def _kept(valid_ref, tokens, length: int):
    """Whether each of `tokens` is a real token: inside the row and, with `valid_ref`, not padding."""
    from jax.experimental.pallas import triton as plgpu

    inside = tokens < length
    if valid_ref is None:
        return inside
    held = plgpu.load(valid_ref.at[jnp.clip(tokens, 0, length - 1)], mask=inside, other=0)
    return inside & (held != 0)


def _taps(w_ref, b_ref, columns, count: int):
    taps = [w_ref[tap, columns].astype(jnp.float32) for tap in range(count)]
    bias = None if b_ref is None else b_ref[columns].astype(jnp.float32)
    return taps, bias


def _preactivation(window, taps, bias):
    total = window[0] * taps[0][None, :]
    for inputs, tap in zip(window[1:], taps[1:], strict=True):
        total = total + inputs * tap[None, :]
    return total if bias is None else total + bias[None, :]


def _unpack(refs, has_bias: bool, has_valid: bool):
    refs = list(refs)
    x_ref, w_ref = refs.pop(0), refs.pop(0)
    b_ref = refs.pop(0) if has_bias else None
    valid_ref = refs.pop(0) if has_valid else None
    return x_ref, w_ref, b_ref, valid_ref, refs


def _forward_kernel(*refs, first: int, normalised: bool, block: int, segments: int, steps: int, count: int,
                    length: int, has_bias: bool, has_valid: bool, activation: bool, eps: float):
    from jax.experimental.pallas import triton as plgpu

    x_ref, w_ref, b_ref, valid_ref, (out_ref,) = _unpack(refs, has_bias, has_valid)
    columns, starts = _program(first, block, segments, steps)
    local = columns - first
    taps, bias = _taps(w_ref, b_ref, columns, count)
    history = tuple(_rows(x_ref, starts - count + 1 + tap, columns, length) for tap in range(count - 1))

    def load(tokens):
        return _rows(x_ref, tokens, columns, length), _kept(valid_ref, tokens, length)

    def step(index, carry):
        history, (current, kept) = carry
        tokens = starts + index
        # The next token's loads go out before this one's arithmetic, which hides their latency.
        ahead = load(tokens + 1)
        window = (*history, current)
        y = _preactivation(window, taps, bias)
        if activation:
            y = y * jax.nn.sigmoid(y)
        if normalised:
            y = y * jax.lax.rsqrt(jnp.sum(y * y, axis=1, keepdims=True) + eps)
        y = jnp.where(kept[:, None], y, 0.0)
        plgpu.store(out_ref.at[tokens[:, None], local[None, :]], y.astype(out_ref.dtype),
                    mask=(tokens < length)[:, None])
        return window[1:], ahead

    jax.lax.fori_loop(0, steps, step, (history, load(starts)))


def _backward_kernel(*refs, first: int, normalised: bool, block: int, segments: int, steps: int,
                     count: int, length: int, has_bias: bool, has_valid: bool, aliased: bool,
                     activation: bool, eps: float):
    from jax.experimental.pallas import triton as plgpu

    x_ref, w_ref, b_ref, valid_ref, rest = _unpack(refs, has_bias, has_valid)
    # The gradient the calls before wrote, aliased to this call's, is read by none.
    dy_ref, dx_ref, dw_ref, *db_ref = rest[1:] if aliased else rest
    columns, starts = _program(first, block, segments, steps)
    local = columns - first
    taps, bias = _taps(w_ref, b_ref, columns, count)
    history = tuple(_rows(x_ref, starts - count + 1 + tap, columns, length) for tap in range(count - 1))
    zero = jnp.zeros((segments, block), jnp.float32)

    def load(tokens):
        kept = _kept(valid_ref, tokens, length)[:, None]
        grad = plgpu.load(dy_ref.at[jnp.clip(tokens, 0, length - 1)[:, None], local[None, :]], mask=kept,
                          other=0).astype(jnp.float32)
        return _rows(x_ref, tokens, columns, length), grad

    def step(index, carry):
        history, later, sums, lost, (current, grad) = carry
        tokens = starts + index
        # The next token's loads go out before this one's arithmetic, which hides their latency.
        ahead = load(tokens + 1)
        window = (*history, current)
        pre = _preactivation(window, taps, bias)
        gate = jax.nn.sigmoid(pre) if activation else None
        out = pre if gate is None else pre * gate
        if normalised:
            scale = jax.lax.rsqrt(jnp.sum(out * out, axis=1, keepdims=True) + eps)
            along = jnp.sum(out * grad, axis=1, keepdims=True)
            grad = scale * grad - (scale * scale * scale * along) * out
        if gate is not None:
            # silu's derivative as XLA's autodiff rounds it, the closer of the forms to float64.
            grad = grad * gate + (grad * pre) * (gate * (1.0 - gate))
        # The token's own run sums its taps' and bias's gradients, compensated (Kahan) so a run's tokens
        # round no worse than XLA's sum over the batch; the K-1 past the run are the next run's.
        owned = jnp.where(((index < steps) & (tokens < length))[:, None], grad, 0.0)
        terms = (*(owned * inputs for inputs in window), owned)
        corrected = tuple(term - error for term, error in zip(terms, lost, strict=True))
        added = tuple(total + term for total, term in zip(sums, corrected, strict=True))
        lost = tuple((new - total) - term for new, total, term in zip(added, sums, corrected, strict=True))
        # Token `tokens - (K - 1)` reads this one through the last tap and the K-1 before through the others.
        gradients = (*later, grad)
        dx = gradients[0] * taps[count - 1][None, :]
        for shift in range(1, count):
            dx = dx + gradients[shift] * taps[count - 1 - shift][None, :]
        emitted = tokens - (count - 1)
        plgpu.store(dx_ref.at[jnp.maximum(emitted, 0)[:, None], columns[None, :]], dx.astype(dx_ref.dtype),
                    mask=((index >= count - 1) & (emitted < length))[:, None])
        return window[1:], gradients[1:], added, lost, ahead

    zeros = (zero,) * (count + 1)
    carry = (history, (zero,) * (count - 1), zeros, zeros, load(starts))
    _, _, totals, lost, _ = jax.lax.fori_loop(0, steps + count - 1, step, carry)
    totals = tuple(total - error for total, error in zip(totals, lost, strict=True))
    for tap, total in enumerate(totals[:count]):
        dw_ref[tap, local] = jnp.sum(total, axis=0)
    if db_ref:
        db_ref[0][local] = jnp.sum(totals[count], axis=0)


def _config(length: int, segments: int | None, steps: int | None) -> tuple[int, int]:
    segments = SEGMENTS if segments is None else segments
    steps = min(STEPS, pl.cdiv(length, segments)) if steps is None else steps
    return segments, steps


def _specs(x, w, bias, valid, width: int, block: int, segments: int, steps: int):
    """The grid over a piece's channel blocks, time tiles and rows, and the inputs' specs: each program
    sees its row of `x` and `valid` whole and indexes its tokens."""
    batch, length, channels = x.shape
    grid = (width // block, pl.cdiv(length, segments * steps), batch)
    specs = [pl.BlockSpec((None, length, channels), lambda c, t, b: (b, 0, 0)),
             pl.BlockSpec(w.shape, lambda c, t, b: (0, 0))]
    operands = [x, w]
    if bias is not None:
        specs.append(pl.BlockSpec(bias.shape, lambda c, t, b: (0,)))
        operands.append(bias)
    if valid is not None:
        specs.append(pl.BlockSpec((None, length), lambda c, t, b: (b, 0)))
        operands.append(valid)
    return grid, specs, operands


def _pieces(parts):
    """Each piece's first channel, width and head."""
    first = 0
    for width, head in parts:
        yield first, width, head
        first += width


def _params(warps: int | None, stages: int | None):
    from jax.experimental.pallas import triton as plgpu

    return plgpu.CompilerParams(num_warps=WARPS if warps is None else warps,
                                num_stages=1 if stages is None else stages)


def _forward(x, taps, bias, valid, parts, activation: bool, eps: float, dtype, interpret: bool,
             segments: int | None = None, steps: int | None = None, warps: int | None = None,
             stages: int | None = None):
    batch, length, _ = x.shape
    segments, steps = _config(length, segments, steps)
    out = []
    for first, width, head in _pieces(parts):
        block = block_width(width, head)
        grid, specs, operands = _specs(x, taps.T, bias, valid, width, block, segments, steps)
        kernel = functools.partial(
            _forward_kernel, first=first, normalised=head is not None, block=block, segments=segments,
            steps=steps, count=taps.shape[1], length=length, has_bias=bias is not None,
            has_valid=valid is not None, activation=activation, eps=eps)
        out.append(pl.pallas_call(
            kernel, grid=grid, in_specs=specs,
            out_specs=pl.BlockSpec((None, length, width), lambda c, t, b: (b, 0, 0)),
            out_shape=jax.ShapeDtypeStruct((batch, length, width), dtype),
            compiler_params=_params(warps, stages), interpret=interpret, name="short_conv_forward",
        )(*operands))
    return tuple(out)


def _backward(x, taps, bias, valid, cotangents, parts, activation: bool, eps: float, interpret: bool,
              segments: int | None = None, steps: int | None = None, warps: int | None = None,
              stages: int | None = None):
    batch, length, channels = x.shape
    count = taps.shape[1]
    segments, steps = _config(length, segments, steps)
    dx, dw, db = None, [], []
    for (first, width, head), dy in zip(_pieces(parts), cotangents, strict=True):
        block = block_width(width, head)
        grid, specs, operands = _specs(x, taps.T, bias, valid, width, block, segments, steps)
        tiles = grid[1]
        kernel = functools.partial(
            _backward_kernel, first=first, normalised=head is not None, block=block, segments=segments,
            steps=steps, count=count, length=length, has_bias=bias is not None, has_valid=valid is not None,
            aliased=dx is not None, activation=activation, eps=eps)
        whole = pl.BlockSpec((None, length, channels), lambda c, t, b: (b, 0, 0))
        out_specs = [whole, pl.BlockSpec((None, None, count, width), lambda c, t, b: (b, t, 0, 0))]
        out_shape = [jax.ShapeDtypeStruct(x.shape, x.dtype),
                     jax.ShapeDtypeStruct((batch, tiles, count, width), jnp.float32)]
        if bias is not None:
            out_specs.append(pl.BlockSpec((None, None, width), lambda c, t, b: (b, t, 0)))
            out_shape.append(jax.ShapeDtypeStruct((batch, tiles, width), jnp.float32))
        # Each piece writes its columns of one gradient: the calls after the first take the one before's.
        before = [] if dx is None else [dx]
        found = pl.pallas_call(
            kernel, grid=grid,
            in_specs=[*specs, *([whole] if before else []),
                      pl.BlockSpec((None, length, width), lambda c, t, b: (b, 0, 0))],
            out_specs=out_specs, out_shape=out_shape,
            input_output_aliases={len(operands): 0} if before else {},
            compiler_params=_params(warps, stages), interpret=interpret, name="short_conv_backward",
        )(*operands, *before, dy)
        dx = found[0]
        dw.append(jnp.sum(found[1], axis=(0, 1)))
        if bias is not None:
            db.append(jnp.sum(found[2], axis=(0, 1)))
    dw = jnp.concatenate(dw, axis=1).T.astype(taps.dtype)
    return dx, dw, None if bias is None else jnp.concatenate(db).astype(bias.dtype)


def _conv(x, taps, bias, valid, parts, activation, eps, dtype, interpret):
    return _forward(x, taps, bias, valid, parts, activation, eps, dtype, interpret)


def _conv_fwd(x, taps, bias, valid, parts, activation, eps, dtype, interpret):
    out = _forward(x, taps, bias, valid, parts, activation, eps, dtype, interpret)
    return out, (x, taps, bias, valid)


def _conv_bwd(parts, activation, eps, dtype, interpret, residual, cotangents):
    x, taps, bias, valid = residual
    dx, dw, db = _backward(x, taps, bias, valid, cotangents, parts, activation, eps, interpret)
    return dx, dw, db, None


conv = jax.custom_vjp(_conv, nondiff_argnums=(4, 5, 6, 7, 8))
"""`linear.causal_conv1d` on the kernels: `x` `[B, T, C]` convolved causally by `taps` `[C, K]` behind
K-1 zero tokens, plus `bias` `[C]` or None, silu'd where `activation`, as the `parts` pieces
`((width, head), ...)` of its channels in order, each `[B, T, width]` in `dtype`, a piece with a `head`
l2-normalised (`eps`) over each `head` channels. `valid` `[B, T]` int8 or None zeroes padding tokens'
outputs. Gradients for `x`, `taps` and `bias`."""
conv.defvjp(_conv_fwd, _conv_bwd)
