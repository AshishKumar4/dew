"""The short causal conv of the recurrent mixers, one Pallas-Triton kernel each way.

`linear.causal_conv1d` is the depthwise causal conv over a `[B, T, C]` projection with `[C, K]` taps
(K a few tokens), its bias and silu, which the gated delta net, Kimi's delta attention and Mamba run
before their scans; the delta rules then l2-normalise each query and key head. As XLA computes it, the
backward reads and writes the `[B, T, C]` activation once for each of the norm's, silu's and the conv's
transposes, several of them in fp32. Here the forward reads the projection once and writes each output
piece once (the queries, keys and values as the scans take them, so no copy splits them), and the
backward reads the projection and the cotangents once and writes the projection's gradient once,
everything between in fp32 registers. Neither saves anything but the projection. The forward writes
every piece in one call, each program branching statically on the piece its blocks belong to, so that
a decoding step, whose conv is a few launch-bound programs, launches it once (a Qwen3.5-0.8B gated delta
net's decode step on an H100 at one row, bf16 parameters: 45.4 against 38.1 us); the backward's pieces
are each their own call, which knows statically whether it normalises and which cotangent it reads,
and writes its columns of one gradient, each taking the last one's as its output.

The forward needs nothing past a token's window, so each of its programs takes a tile of tokens
(`Tile`) and loads their windows at the conv's K shifts at once (the overlap stays in L1 and L2), as
fla's causal_conv1d does (fla/modules/conv/triton/kernels.py, v0.5.2). The backward needs the
pre-activation gradient at the K-1 rows past a tile, and a Triton program cannot shift a register
tile: fla writes the pre-activation to memory in a separate pass and reads it back at each shift, and
fusing the norm on top would recompute the conv, silu and norm K times a row. Each backward program
instead walks time, as causal-conv1d's channel-last backward does (Dao-AILab/causal-conv1d,
csrc/causal_conv1d_bwd.cu): a tile is `segments` rows, each the next token of its own run of `steps`
tokens (`Walk`), and a loop steps all of them forward one token at a time, carrying the last K-1 inputs and
pre-activation gradients, so each token's conv, silu, norm and their gradients are computed once. A
run reads the K-1 tokens before it and computes K-1 past its end, `(K - 1) / steps` extra.

The conv selector (`CONV`) runs them where their requirements hold and XLA's conv elsewhere, reported
as the call's route ('short_conv'). Each launch takes its one config, `TILE` and `WALK`
(`dew.nn.kernels.tuning`), whose float64 gate tests/test_short_conv.py holds.

The arithmetic is `causal_conv1d`'s in fp32, each output rounded once to its dtype: the conv's
products summed tap by tap, as XLA sums them, and the sigmoid's quotient and the norm's root and
inverse rounded once (`div.rn`, `sqrt.rn`, which cost nothing measurable), where Triton's and XLA's
fp32 division is `div.full` and the root `sqrt.approx`. Every output and gradient is within the
float64 gate of XLA's conv (tests/test_short_conv.py). Summing the products compensated and the tap
gradients Kahan's way brought them nearer float64 than that (the 0.8B's fp32 queries 7.0e-8 from
float64 against 8.3e-8 relative RMS) and cost 19% of an H100's forward and 26% of its backward, more
than the gate asks for.
"""

import dataclasses
import functools
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from . import generation, tuning
from .generation import INHERENT
from .selection import Kernel, Op, Requirement

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


@dataclasses.dataclass(frozen=True)
class ConvCase:
    """A conv as the conv selector reads it (`CONV`): whether the kernels run here (`placed`), whether no
    mesh axis splits its operand (`whole`), the operand's `shape`
    and `dtype`, the dtype it is written in (`out`), its `taps` and its pieces (`parts`, `(width, head)`
    each)."""

    placed: bool
    whole: bool
    shape: tuple[int, ...]
    dtype: np.dtype
    out: np.dtype
    taps: int
    parts: tuple[tuple[int, int | None], ...]


SUPPORTS: tuple[Requirement[ConvCase], ...] = (
    Requirement('capability', lambda case: case.placed, "the kernels need a GPU of sm80 or later",
                row=INHERENT),
    Requirement('placement', lambda case: case.whole,
                "a mesh axis splits the conv's operands, and the kernels see whole arrays",
                row='ledger 1306'),
    Requirement('shape', lambda case: len(case.shape) == 3 and case.shape[1] > 0,
                lambda case: f"a conv over {case.shape}; the kernels take [B, T, C]", row=INHERENT),
    Requirement('dtype', lambda case: case.dtype in DTYPES and case.out in DTYPES,
                lambda case: f"{case.dtype} inputs into {case.out}; the kernels read and write float32, "
                             "bfloat16 or float16, summing in float32", row=INHERENT),
    Requirement('taps', lambda case: 2 <= case.taps <= WIDEST_TAPS,
                lambda case: f"{case.taps} taps; the kernels unroll 2 to {WIDEST_TAPS}", row=INHERENT),
    Requirement('pieces', lambda case: sum(width for width, _ in case.parts) == case.shape[-1],
                lambda case: f"pieces {case.parts} of {case.shape[-1]} channels", row=INHERENT),
    Requirement('blocks', lambda case: all(_fits(width, head) for width, head in case.parts),
                lambda case: f"pieces {case.parts}: each a multiple of its block, a normalised piece's "
                             "power-of-two head up to 256, another's 16 to 128 channels", row=INHERENT),
)
"""What the kernels need of a conv: a GPU they compile for, reverse mode, an operand no mesh axis splits,
and the shapes and dtypes they take."""

CONV: Op[ConvCase] = Op('short_conv', (Kernel('pallas', SUPPORTS), Kernel('xla', (), generic=True)),
                        lambda case: 'pallas' if generation.backend() == 'gpu' else 'xla')
"""The conv selector: the kernels on a GPU where their requirements hold, else XLA's conv
(`dew.nn.linear.xla_conv1d`)."""


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


def _documents(documents_ref, tokens, length: int):
    """The packed document of each of `tokens`, clipped into the row; None without `documents_ref`."""
    from jax.experimental.pallas import triton as plgpu

    if documents_ref is None:
        return None
    return plgpu.load(documents_ref.at[jnp.clip(tokens, 0, length - 1)])


def _alone(tiles, documents, document):
    """Each of `tiles` `[segments, block]` zeroed where its token, in `documents`, is not in `document`:
    a packed document's tap reads none of another's tokens."""
    if document is None:
        return tiles
    return tuple(jnp.where((theirs == document)[:, None], tile, 0.0)
                 for theirs, tile in zip(documents, tiles, strict=True))


def _kept(valid_ref, tokens, length: int):
    """Whether each of `tokens` is a real token: inside the row and, with `valid_ref`, not padding."""
    from jax.experimental.pallas import triton as plgpu

    inside = tokens < length
    if valid_ref is None:
        return inside
    held = plgpu.load(valid_ref.at[jnp.clip(tokens, 0, length - 1)], mask=inside, other=0)
    return inside & (held != 0)


def _taps(w_ref, b_ref, columns, count: int):
    taps = [w_ref[columns, tap].astype(jnp.float32) for tap in range(count)]
    bias = None if b_ref is None else b_ref[columns].astype(jnp.float32)
    return taps, bias


def _inline(asm: str, *args):
    from jax.experimental.pallas import triton as plgpu

    shape = jax.ShapeDtypeStruct(args[0].shape, jnp.float32)
    return plgpu.elementwise_inline_asm(asm, args=args, constraints=",".join(["=f", *"f" * len(args)]),
                                        pack=1, result_shape_dtypes=(shape,))[0]


def _divide(numerator, denominator, *, interpret: bool):
    """The quotient rounded once: Triton divides fp32 approximately (`div.full`, 2 ulp)."""
    if interpret:
        return numerator / denominator
    numerator, denominator = jnp.broadcast_arrays(numerator, denominator)
    return _inline("div.rn.f32 $0, $1, $2;", numerator, denominator)


def _sqrt(x, *, interpret: bool):
    """The root rounded once: Triton's fp32 root is approximate (`sqrt.approx`)."""
    return jnp.sqrt(x) if interpret else _inline("sqrt.rn.f32 $0, $1;", x)


def preactivation(window, taps, bias):
    """The conv's sum, tap by tap from the oldest token, then the bias: `window` and `taps` each tap's
    inputs `[..., C]` and weights `[C]`, oldest first, in the sum's dtype (fp32 in the kernels; the portable
    conv's and the decode reference's at least fp32), the one order every conv path adds in."""
    total = window[0] * taps[0][None, :]
    for inputs, tap in zip(window[1:], taps[1:], strict=True):
        total = total + inputs * tap[None, :]
    return total if bias is None else total + bias[None, :]


def _sigmoid(pre, *, interpret: bool):
    """The sigmoid, one rounding of its quotient."""
    return _divide(1.0, 1.0 + jnp.exp(-pre), interpret=interpret)


def _inverse_norm(y, eps: float, *, interpret: bool):
    """`1 / sqrt(sum(y^2) + eps)` over each row `[segments, 1]`, its root and quotient each rounded once:
    `l2norm`'s scale without rsqrt's approximation."""
    root = _sqrt(jnp.sum(y * y, axis=1, keepdims=True) + eps, interpret=interpret)
    return _divide(1.0, root, interpret=interpret)


def _unpack(refs, has_bias: bool, has_valid: bool, has_documents: bool):
    refs = list(refs)
    x_ref, w_ref = refs.pop(0), refs.pop(0)
    b_ref = refs.pop(0) if has_bias else None
    valid_ref = refs.pop(0) if has_valid else None
    documents_ref = refs.pop(0) if has_documents else None
    return x_ref, w_ref, b_ref, valid_ref, documents_ref, refs


def _forward_kernel(*refs, pieces: tuple[tuple[int, int, int, bool], ...], rows: int, count: int, length: int,
                    has_bias: bool, has_valid: bool, has_documents: bool, activation: bool, eps: float,
                    interpret: bool):
    """A tile of `rows` tokens of one block of one piece, each token computed from its K-window loaded
    at the conv's shifts, all at once: the forward needs nothing from the tokens past the tile, so it
    walks nothing. `pieces` holds each piece's first program, first channel, block and whether it
    normalises; the programs of grid axis 0 run through the pieces' blocks in order, so one call
    writes every piece."""
    x_ref, w_ref, b_ref, valid_ref, documents_ref, out_refs = _unpack(refs, has_bias, has_valid,
                                                                       has_documents)
    program = pl.program_id(0)
    tokens = pl.program_id(1) * rows + jnp.arange(rows, dtype=jnp.int32)
    ends = [lo for lo, *_ in pieces[1:]] + [None]
    documents = tuple(_documents(documents_ref, tokens - count + 1 + tap, length) for tap in range(count))

    for (lo, first, block, normalised), end, out_ref in zip(pieces, ends, out_refs, strict=True):
        def piece(lo=lo, first=first, block=block, normalised=normalised, out_ref=out_ref):
            from jax.experimental.pallas import triton as plgpu

            local = (program - lo) * block + jnp.arange(block, dtype=jnp.int32)
            columns = first + local
            taps, bias = _taps(w_ref, b_ref, columns, count)
            window = tuple(_rows(x_ref, tokens - count + 1 + tap, columns, length) for tap in range(count))
            y = preactivation(_alone(window, documents, documents[-1]), taps, bias)
            if activation:
                y = _divide(y, 1.0 + jnp.exp(-y), interpret=interpret)
            if normalised:
                y = y * _inverse_norm(y, eps, interpret=interpret)
            y = jnp.where(_kept(valid_ref, tokens, length)[:, None], y, 0.0)
            plgpu.store(out_ref.at[tokens[:, None], local[None, :]], y.astype(out_ref.dtype),
                        mask=(tokens < length)[:, None])

        if len(pieces) == 1:
            piece()
        else:
            pl.when((program >= lo) & (program < end) if end is not None else program >= lo)(piece)


def _backward_kernel(*refs, first: int, normalised: bool, block: int, segments: int, steps: int,
                     count: int, length: int, has_bias: bool, has_valid: bool, has_documents: bool,
                     aliased: bool, activation: bool, eps: float, interpret: bool):
    from jax.experimental.pallas import triton as plgpu

    x_ref, w_ref, b_ref, valid_ref, documents_ref, rest = _unpack(refs, has_bias, has_valid, has_documents)
    # The gradient the calls before wrote, aliased to this call's, is read by none.
    dy_ref, dx_ref, dw_ref, *db_ref = rest[1:] if aliased else rest
    columns, starts = _program(first, block, segments, steps)
    local = columns - first
    taps, bias = _taps(w_ref, b_ref, columns, count)
    history = tuple(_rows(x_ref, starts - count + 1 + tap, columns, length) for tap in range(count - 1))
    owners = tuple(_documents(documents_ref, starts - count + 1 + tap, length) for tap in range(count - 1))
    zero = jnp.zeros((segments, block), jnp.float32)

    def load(tokens):
        kept = _kept(valid_ref, tokens, length)[:, None]
        grad = plgpu.load(dy_ref.at[jnp.clip(tokens, 0, length - 1)[:, None], local[None, :]], mask=kept,
                          other=0).astype(jnp.float32)
        return _rows(x_ref, tokens, columns, length), grad, _documents(documents_ref, tokens, length)

    def step(index, carry):
        history, owners, later, sums, (current, grad, document) = carry
        tokens = starts + index
        # The next token's loads go out before this one's arithmetic, which hides their latency.
        ahead = load(tokens + 1)
        stream, documents = (*history, current), (*owners, document)
        window = _alone(stream, documents, document)
        pre = preactivation(window, taps, bias)
        gate = _sigmoid(pre, interpret=interpret) if activation else None
        out = pre if gate is None else pre * gate
        if normalised:
            # l2norm's transpose, scale * dy - scale^3 (out . dy) out, as scale * (dy - (u . dy) u) for the
            # normalised u.
            scale = _inverse_norm(out, eps, interpret=interpret)
            unit = out * scale
            grad = scale * (grad - jnp.sum(unit * grad, axis=1, keepdims=True) * unit)
        if gate is not None:
            # silu's derivative as XLA's autodiff rounds it, the closer of the forms to float64.
            grad = grad * gate + (grad * pre) * (gate * (1.0 - gate))
        # The token's own run sums its taps' and bias's gradients; the K-1 past the run are the next run's.
        owned = jnp.where(((index < steps) & (tokens < length))[:, None], grad, 0.0)
        terms = (*(owned * inputs for inputs in window), owned)
        added = tuple(total + term for total, term in zip(sums, terms, strict=True))
        # Token `tokens - (K - 1)` reads this one through the last tap and the K-1 before through the others,
        # those of its own document.
        gradients = (*later, grad)
        reads = _alone(gradients, documents, documents[0])
        dx = reads[0] * taps[count - 1][None, :]
        for shift in range(1, count):
            dx = dx + reads[shift] * taps[count - 1 - shift][None, :]
        emitted = tokens - (count - 1)
        plgpu.store(dx_ref.at[jnp.maximum(emitted, 0)[:, None], columns[None, :]], dx.astype(dx_ref.dtype),
                    mask=((index >= count - 1) & (emitted < length))[:, None])
        return stream[1:], documents[1:], gradients[1:], added, ahead

    carry = (history, owners, (zero,) * (count - 1), (zero,) * (count + 1), load(starts))
    _, _, _, totals, _ = jax.lax.fori_loop(0, steps + count - 1, step, carry)
    for tap, total in enumerate(totals[:count]):
        dw_ref[local, tap] = jnp.sum(total, axis=0)
    if db_ref:
        db_ref[0][local] = jnp.sum(totals[count], axis=0)


@dataclasses.dataclass(frozen=True)
class Tile:
    """The forward's tokens a program loads at once, and its warps."""
    rows: int
    warps: int


@dataclasses.dataclass(frozen=True)
class Walk:
    """The backward's runs a program walks side by side, the tokens of each run, and its warps: runs
    and tokens first, which order the tap gradients' sums."""
    segments: int
    steps: int
    warps: int


TILE = Tile(rows=16, warps=4)
"""The forward's launch: 16 tokens a tile, each window loaded at once."""

WALK = Walk(segments=8, steps=32, warps=4)
"""The backward's launch, its runs no longer than a row's share: eight runs a tile, with a 128-channel
block 1024 elements, eight an fp32 register of each of four warps' threads, and 32 tokens a run, whose
K-1 extra tokens are 9% of it at the K of 4 every mixer uses. A run sums its taps' products token by
token, so its rounding grows with its steps: runs of 128 put the 0.8B's bf16 tap gradients 2.06 times
as far from float64 as XLA's on an A100, and runs of 64 1.57 times at the 0.8B's and 9B's training
shapes on an H100 (row 2411), past the gate."""


def _inputs(x, w, bias, valid, documents):
    """The inputs' specs over a grid of channel blocks, time tiles and rows: each program sees its row
    of `x`, `valid` and `documents` whole and indexes its tokens."""
    _, length, channels = x.shape
    specs = [pl.BlockSpec((None, length, channels), lambda c, t, b: (b, 0, 0)),
             pl.BlockSpec(w.shape, lambda c, t, b: (0, 0))]
    operands = [x, w]
    if bias is not None:
        specs.append(pl.BlockSpec(bias.shape, lambda c, t, b: (0,)))
        operands.append(bias)
    for tokens in (valid, documents):
        if tokens is not None:
            specs.append(pl.BlockSpec((None, length), lambda c, t, b: (b, 0)))
            operands.append(tokens)
    return specs, operands


def _pieces(parts):
    """Each piece's first channel, width and head."""
    first = 0
    for width, head in parts:
        yield first, width, head
        first += width


def _params(warps: int):
    from jax.experimental.pallas import triton as plgpu

    return plgpu.CompilerParams(num_warps=warps, num_stages=1)


class _ConvLaunch(tuning.KernelOp):
    """A launch of the conv, timed on rows whose every token is real (`valid` int8 ones) and in one
    document (`documents` int32 zeros)."""

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        return [jnp.ones(spec.shape, spec.dtype) if spec.dtype == jnp.int8 else array
                for spec, array in zip(specs, super().operands(specs, seed), strict=True)]


@dataclasses.dataclass(frozen=True)
class ConvForward(_ConvLaunch):
    """Every piece's forward in one launch (`_forward_kernel`): at one decoding row the conv's cost is
    its launches. A program takes a `Tile`, `TILE`."""

    config_cls: ClassVar[type[Tile]] = Tile

    ROWS: ClassVar[tuple[str, ...]] = ('x', 'valid', 'documents')

    def _fwd(self, x: jax.Array, taps: jax.Array, bias: jax.Array | None, valid: jax.Array | None,
             documents: jax.Array | None, *, parts: tuple[tuple[int, int | None], ...], activation: bool,
             eps: float, dtype: str, interpret: bool, config: Tile, return_residuals: bool = False):
        batch, length, _ = x.shape
        pieces, programs = [], 0
        for first, width, head in _pieces(parts):
            block = block_width(width, head)
            pieces.append((programs, first, block, head is not None))
            programs += width // block
        specs, operands = _inputs(x, taps, bias, valid, documents)
        kernel = functools.partial(
            _forward_kernel, pieces=tuple(pieces), rows=config.rows, count=taps.shape[1], length=length,
            has_bias=bias is not None, has_valid=valid is not None, has_documents=documents is not None,
            activation=activation, eps=eps, interpret=interpret)
        widths = [width for width, _ in parts]
        return tuple(pl.pallas_call(
            kernel, grid=(programs, pl.cdiv(length, config.rows), batch), in_specs=specs,
            out_specs=[pl.BlockSpec((None, length, width), lambda c, t, b: (b, 0, 0)) for width in widths],
            out_shape=[jax.ShapeDtypeStruct((batch, length, width), jnp.dtype(dtype)) for width in widths],
            compiler_params=_params(config.warps), interpret=interpret, name="short_conv_forward",
        )(*operands)), None

    def _get_heuristics_config(self, ba) -> Tile:
        return TILE


@dataclasses.dataclass(frozen=True)
class ConvBackward(_ConvLaunch):
    """One piece's backward (`_backward_kernel`): the projection's gradient at the piece's columns,
    written into the gradient the piece before wrote (`before`, aliased) where there is one, and each
    tile's tap and bias gradients. A program walks a `Walk`, `WALK` at the call's length (`_walk`)."""

    config_cls: ClassVar[type[Walk]] = Walk

    ROWS: ClassVar[tuple[str, ...]] = ('x', 'valid', 'documents', 'before', 'dy')

    def _fwd(self, x: jax.Array, taps: jax.Array, bias: jax.Array | None, valid: jax.Array | None,
             documents: jax.Array | None, before: jax.Array | None, dy: jax.Array, *, first: int,
             head: int | None, activation: bool, eps: float, interpret: bool, config: Walk,
             return_residuals: bool = False):
        batch, length, channels = x.shape
        count, width = taps.shape[1], dy.shape[-1]
        block = block_width(width, head)
        grid = (width // block, pl.cdiv(length, config.segments * config.steps), batch)
        specs, operands = _inputs(x, taps, bias, valid, documents)
        tiles = grid[1]
        kernel = functools.partial(
            _backward_kernel, first=first, normalised=head is not None, block=block, segments=config.segments,
            steps=config.steps, count=count, length=length, has_bias=bias is not None,
            has_valid=valid is not None, has_documents=documents is not None, aliased=before is not None,
            activation=activation, eps=eps, interpret=interpret)
        whole = pl.BlockSpec((None, length, channels), lambda c, t, b: (b, 0, 0))
        out_specs = [whole, pl.BlockSpec((None, None, width, count), lambda c, t, b: (b, t, 0, 0))]
        out_shape = [jax.ShapeDtypeStruct(x.shape, x.dtype),
                     jax.ShapeDtypeStruct((batch, tiles, width, count), jnp.float32)]
        if bias is not None:
            out_specs.append(pl.BlockSpec((None, None, width), lambda c, t, b: (b, t, 0)))
            out_shape.append(jax.ShapeDtypeStruct((batch, tiles, width), jnp.float32))
        held = [] if before is None else [before]
        return tuple(pl.pallas_call(
            kernel, grid=grid,
            in_specs=[*specs, *([whole] if held else []),
                      pl.BlockSpec((None, length, width), lambda c, t, b: (b, 0, 0))],
            out_specs=out_specs, out_shape=out_shape,
            input_output_aliases={len(operands): 0} if held else {},
            compiler_params=_params(config.warps), interpret=interpret, name="short_conv_backward",
        )(*operands, *held, dy)), None

    def _get_heuristics_config(self, ba) -> Walk:
        return self._walk(ba, WALK)

    @staticmethod
    def _walk(ba, walk: Walk) -> Walk:
        """`walk` with its runs no longer than a row's tokens shared among its segments."""
        length = ba.arguments['x'].shape[1]
        return dataclasses.replace(walk, steps=min(walk.steps, pl.cdiv(length, walk.segments)))


def _forward(x, taps, bias, valid, documents, parts, activation: bool, eps: float, dtype, interpret: bool):
    """Every piece in one call (`ConvForward`)."""
    return ConvForward()(x, taps, bias, valid, documents, parts=tuple(parts), activation=activation, eps=eps,
                         dtype=jnp.dtype(dtype).name, interpret=interpret)


def _backward(x, taps, bias, valid, documents, cotangents, parts, activation: bool, eps: float,
              interpret: bool):
    """Each piece's call (`ConvBackward`), each writing its columns of one gradient."""
    dx, dw, db = None, [], []
    for (first, _, head), dy in zip(_pieces(parts), cotangents, strict=True):
        found = ConvBackward()(x, taps, bias, valid, documents, dx, dy, first=first, head=head,
                               activation=activation, eps=eps, interpret=interpret)
        dx = found[0]
        dw.append(jnp.sum(found[1], axis=(0, 1)))
        if bias is not None:
            db.append(jnp.sum(found[2], axis=(0, 1)))
    dw = jnp.concatenate(dw).astype(taps.dtype)
    return dx, dw, None if bias is None else jnp.concatenate(db).astype(bias.dtype)


def conv(x, taps, bias, valid, documents, parts, *, activation: bool, eps: float, dtype, interpret: bool):
    """`linear.causal_conv1d` on the kernels: `x` `[B, T, C]` convolved causally by `taps` `[C, K]` behind
    K-1 zero tokens, plus `bias` `[C]` or None, silu'd where `activation`, as the `parts` pieces
    `((width, head), ...)` of its channels in order, each `[B, T, width]` in `dtype`, a piece with a `head`
    l2-normalised (`eps`) over each `head` channels. `valid` `[B, T]` int8 or None zeroes padding tokens'
    outputs; `documents` `[B, T]` int32 or None numbers packed documents, a tap reading a token only in
    its output token's. Differentiated at every order through XLA's conv (`linear.xla_conv1d`, the same
    arithmetic; `generation.linear_tangent`): reverse mode runs the backward kernels, and forward mode takes
    XLA's tangent however a jit, a scan or a remat staged the call (row 2395)."""
    from dew.nn.linear import xla_conv1d

    def forward(x, taps, bias, valid, documents):
        return _forward(x, taps, bias, valid, documents, parts, activation, eps, dtype, interpret), ()

    def backward(saved, operands, cotangents):
        del saved
        x, taps, bias, valid, documents = operands
        dx, dw, db = _backward(x, taps, bias, valid, documents, cotangents, parts, activation, eps, interpret)
        # The masks have no tangent space: float0's zero, as JAX takes it of them.
        masks = (None if mask is None else np.zeros(mask.shape, jax.dtypes.float0)
                 for mask in (valid, documents))
        return dx, dw, db, *masks

    def portable(x, taps, bias, valid, documents):
        return xla_conv1d(x, taps, activation, bias, valid=valid, segments=documents, parts=parts,
                          dtype=dtype)

    return generation.linear_tangent(forward, backward, portable)(x, taps, bias, valid, documents)

