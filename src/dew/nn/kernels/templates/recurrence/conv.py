"""The short causal conv of the recurrent mixers on Mosaic GPU (`short_conv.conv`'s kernels): a program a
row, a block of `TOKENS` tokens and a warpgroup's `scan.THREADS` channels of a piece, each thread one channel
walking its tokens in order, each token's window read at the conv's K shifts (their overlap in L1).

The forward reads each token's channels once and writes each output once; a normalised piece (a delta rule's
queries and keys, one head a program, `scan.THREADS` wide) sums its head's squares across the warpgroup a
token (`scan.summed`). The backward walks its tokens from the block's last, and the K-1 past it which the
next block owns, carrying the pre-activation gradients of the K-1 tokens after a token, so each token's conv,
activation, norm and their gradients are computed once and its input's gradient written once; each program
sums its own tokens' tap and bias gradients, which XLA adds. The arithmetic is `short_conv`'s: the products
summed tap by tap from the oldest (`short_conv.preactivation`'s order), the SiLU's sigmoid one quotient, the
norm's root and inverse each rounded once, its transpose `scale (dy - (u . dy) u)`.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import short_conv, tuning
from dew.nn.kernels.templates import mosaic, scan
from dew.nn.precision import compensated_add

TOKENS = 64
"""The most tokens a program walks: a backward program computes K-1 past them, `(K - 1) / TOKENS` extra."""


def _tokens(length: int) -> int:
    """The tokens a program of a call of `length` walks: `TOKENS`, or the call's own where shorter. Fixed at
    `TOKENS`, a decode token's conv walked 64 masked tokens, 29 us a piece on an A100 against the Triton
    conv's 10 (M-JS-18)."""
    return min(TOKENS, length)


UNROLLED = 8
"""The most tokens a turn of a walk reads at once, their inputs (and in the backward their output gradients)
loaded before the first is used, so the loads overlap: a token a turn, each load's latency was the walk's
(an A100's conv training step 2.11 ms against the Triton conv's 1.81, M-JS-20)."""


def _unrolled(steps: int) -> int:
    """The tokens a turn of a walk of `steps` reads at once: `UNROLLED`'s widest power of two dividing it."""
    return next(unrolled for unrolled in (UNROLLED, 4, 2, 1) if steps % unrolled == 0)


def lanes(parts) -> int:
    """The channels a program of a conv of `parts` (`(width, head)` each) takes, a vector of `lanes //
    THREADS` a thread: the widest of 8, 4, 2 and 1 channels a thread (a 128-bit transfer of 16-bit
    channels at 8) every piece's width holds whole, a normalised piece's head summed in at most a warp's
    threads (`scan.grouped_summed`) or across the warpgroup at one a thread (`scan.summed`). A channel a
    thread, each transfer a warp's 64 bytes, a GH200's conv training step took 1.07 ms against the Triton
    conv's 0.77 (M-JS-20)."""
    for vector in (8, 4, 2, 1):
        span = scan.THREADS * vector
        if all(width % span == 0 and (head is None or vector == 1 or head // vector <= scan.WARP)
               for width, head in parts):
            return span
    raise ValueError(f"no program's channels divide the conv's pieces {parts}")


def _group(head: int | None, lanes_: int) -> int | None:
    """The threads a normalised head's channels take at `lanes_` a program (None: the warpgroup's, one a
    thread)."""
    vector = lanes_ // scan.THREADS
    return None if head is None or vector == 1 else head // vector


def fits(width: int, head: int | None) -> bool:
    """Whether a piece of `width` channels steps here: whole warpgroups of them, a normalised piece's head
    one (its row sum the warpgroup's)."""
    return width % scan.THREADS == 0 and head in (None, scan.THREADS)



def _kept(kept, value: jax.Array) -> jax.Array:
    """`value` where the scalar `kept` holds, else zeros (`mosaic.laid`)."""
    return jnp.where(mosaic.laid(kept, value), value, jnp.zeros_like(value))


def _loaded(ref, row, token, columns, length: int):
    """`ref`'s program's channels (`columns`) at `token` of `row`, fp32, zero before the row's first and
    past its last."""
    inside = (token >= 0) & (token < length)
    at = ref.at[row, jnp.clip(token, 0, length - 1), columns]
    value = plgpu.load(at, layout=scan.blocked(columns.size), optimized=False).astype(jnp.float32)
    return _kept(inside, value)


def _owner(documents_ref, row, token, length: int):
    """The packed document `token` of `row` is in, -1 outside the row; 0 without documents."""
    if documents_ref is None:
        return jnp.int32(0)
    inside = (token >= 0) & (token < length)
    return jnp.where(inside, documents_ref[row, jnp.clip(token, 0, length - 1)], -1)


def _stored(ref, at: tuple, value, when) -> None:
    """`value` stored at `ref[at]` in its dtype where `when` holds (a token inside the row, or its own)."""
    @pl.when(when)
    def _():
        plgpu.store(ref.at[at], value.astype(ref.dtype), optimized=False)


def _activated(pre, activation: bool):
    """The SiLU of the conv's sums, its sigmoid one quotient, and the sigmoid (None without one)."""
    if not activation:
        return pre, None
    gate = 1.0 / (1.0 + jnp.exp(-pre))
    return pre / (1.0 + jnp.exp(-pre)), gate


class _Piece:
    """A program's view of its piece: its row, its tokens' first, its channels in the operand and the
    piece, the taps and bias it reads, and the masks."""

    def __init__(self, refs, *, first: int, local, length: int, count: int, tokens: int, lanes_: int):
        self.x_ref, self.w_ref, self.b_ref, self.valid_ref, self.documents_ref = refs
        self.row, self.length, self.count = lax.axis_index("rows"), length, count
        self.start = lax.axis_index("tokens") * tokens
        self.local, self.columns = pl.ds(local, lanes_), pl.ds(first + local, lanes_)
        self.lanes = lanes_
        layout = scan.blocked(lanes_)
        self.taps = [plgpu.load(self.w_ref.at[tap, self.columns], layout=layout, optimized=False)
                     .astype(jnp.float32) for tap in range(count)]
        self.bias = (None if self.b_ref is None else plgpu.load(
            self.b_ref.at[self.columns], layout=layout, optimized=False).astype(jnp.float32))

    def input(self, token):
        return _loaded(self.x_ref, self.row, token, self.columns, self.length)

    def owner(self, token):
        return _owner(self.documents_ref, self.row, token, self.length)

    def kept(self, token):
        """Whether `token` is a real one: inside the row and, with a validity mask, not padding."""
        inside = (token >= 0) & (token < self.length)
        if self.valid_ref is None:
            return inside
        return inside & (self.valid_ref[self.row, jnp.clip(token, 0, self.length - 1)] != 0)

    def read(self, token):
        """`token`'s input and its document, as a window slides over them (`window`)."""
        return self.input(token), self.owner(token)

    @staticmethod
    def window(read):
        """A token's window from its `count` tokens' reads (`read`), oldest first, the token last: each input
        of another document zero. A walk slides it a token at a time, one token's read a step: read anew, a
        window's `count` inputs and documents a token put an A100's conv training step at 2.65 ms against
        the Triton conv's 1.81 (M-JS-18)."""
        document = read[-1][1]
        return [_kept(theirs == document, value) for value, theirs in read]

    def preactivation(self, window):
        return short_conv.preactivation(window, self.taps, self.bias)


def _named(kernel: str, parts: tuple) -> str:
    """A launch's name, which its call's location carries (`lowered.kernel_calls`): its kernel's and its
    pieces' widths, a normalised piece's marked."""
    return kernel + "_" + "_".join(f"{width}{'n' if head else ''}" for width, head in parts)


@tuning.lowered_once(("parts", "activation", "eps", "dtype", "family", "interpret"))
def forward(x, taps, bias, valid, documents, *, parts: tuple, activation: bool, eps: float, dtype,
            family: mosaic.Family, interpret: bool = False):
    """Every piece's output in one launch, `[B, T, width]` each of `parts` (each `(width, head)` in channel
    order): `x`'s channels convolved by `taps` `[C, K]` plus `bias`, activated and, a piece with a `head`,
    l2-normalised over each head, each program's channels its piece's (`lanes`). Lowered once a configuration
    (`tuning.lowered_once`): a model's layers' identical convs were each lowered again (two Mamba layers six
    Mosaic calls to one's four, jsf-r06p). A launch a piece, a decode token's conv took three, 10.1 us on an
    H100 against the Triton conv's one at 4.8 (M-JS-25)."""
    rows, length, _ = x.shape
    count, tokens, lanes_ = taps.shape[1], _tokens(length), lanes(parts)
    unrolled = _unrolled(tokens)
    operands = [x, taps.T, bias, valid, documents]
    bounds = [(first // lanes_, (first + width) // lanes_, first, head)
              for first, width, head in pieces(parts)]

    def walk(piece: _Piece, out_ref, exchanged, head: int | None):
        group = _group(head, lanes_)

        def turn(i, held):
            first_token = piece.start + i * unrolled
            reads = [piece.read(first_token + u) for u in range(unrolled)]
            for u, latest in enumerate(reads):
                t = first_token + u
                read = [*held, latest]
                out = _activated(piece.preactivation(piece.window(read)), activation)[0]
                if head is not None:
                    out = out * scan.inverse_norm(out, eps, exchanged, group=group, interpret=interpret)
                _stored(out_ref, (piece.row, t, piece.local), _kept(piece.kept(t), out), t < length)
                held = read[1:]
            return held

        lax.fori_loop(0, tokens // unrolled, turn,
                      [piece.read(piece.start - count + 1 + j) for j in range(count - 1)])

    def entry(*refs):
        *given, exchanged = refs
        given, out_refs = given[:-len(parts)], given[-len(parts):]
        given = iter(given)
        found = [next(given) if operand is not None else None for operand in operands]
        program = lax.axis_index("channels")
        for (lo, hi, first, head), out_ref in zip(bounds, out_refs, strict=True):
            @pl.when((program >= lo) & (program < hi))
            def _(lo=lo, first=first, head=head, out_ref=out_ref):
                walk(_Piece(found, first=first, local=(program - lo) * lanes_, length=length, count=count,
                            tokens=tokens, lanes_=lanes_), out_ref, exchanged, head)

    out_type = tuple(jax.ShapeDtypeStruct((rows, length, width), dtype) for width, _ in parts)
    grid = (rows, pl.cdiv(length, tokens), x.shape[-1] // lanes_)
    launched = mosaic.kernel(entry, out_type=out_type, grid=grid,
                             grid_names=("rows", "tokens", "channels"), warpgroups=1, interpret=interpret,
                             family=family.name, name=_named("short_conv", parts), scratch=(scan.scratch(),))
    return tuple(launched(*(operand for operand in operands if operand is not None)))


@tuning.lowered_once(("parts", "activation", "eps", "family", "interpret"))
def backward(x, taps, bias, valid, documents, dys, *, parts: tuple, activation: bool, eps: float,
             family: mosaic.Family, interpret: bool = False):
    """Every piece's gradients in one launch: the input's gradient `[B, T, C]` in `x`'s dtype, each program's
    channels its piece's (`parts`, each `(width, head)` in channel order, `dys` their output gradients), and
    the tap and bias gradients `[K, C]` and `[C]` fp32 (the bias's zero without one). A launch a piece wrote
    its piece's input gradient apart, which XLA joined: 0.66 of an A100's 2.65 ms conv training step at
    Qwen3.5-0.8B's widths (M-JS-18)."""
    rows, length, channels = x.shape
    count, tokens, lanes_ = taps.shape[1], _tokens(length), lanes(parts)
    blocks = pl.cdiv(length, tokens)
    operands = [x, taps.T, bias, valid, documents]
    bounds = [(first // lanes_, (first + width) // lanes_, first, head)
              for first, width, head in pieces(parts)]

    def entry(*refs):
        *given, dx_ref, dw_ref, db_ref, exchanged = refs
        given, dy_refs = given[:-len(dys)], given[-len(dys):]
        given = iter(given)
        found = [next(given) if operand is not None else None for operand in operands]
        program = lax.axis_index("channels")
        for (lo, hi, first, head), dy_ref in zip(bounds, dy_refs, strict=True):
            @pl.when((program >= lo) & (program < hi))
            def _(lo=lo, first=first, head=head, dy_ref=dy_ref):
                piece = _Piece(found, first=first, local=(program - lo) * lanes_, length=length,
                               count=count, tokens=tokens, lanes_=lanes_)
                _backward_walk(piece, dy_ref, (dx_ref, dw_ref, db_ref), exchanged, head=head,
                               activation=activation, eps=eps, tokens=tokens, interpret=interpret)

    f32 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.float32)
    out_type = (jax.ShapeDtypeStruct((rows, length, channels), x.dtype), f32((rows, blocks, count, channels)),
                f32((rows, blocks, channels)))
    launched = mosaic.kernel(entry, out_type=out_type, grid=(rows, blocks, channels // lanes_),
                             grid_names=("rows", "tokens", "channels"), warpgroups=1, interpret=interpret,
                             family=family.name, name=_named("short_conv_backward", parts),
                             scratch=(scan.scratch(),))
    dx, dw, db = launched(*(operand for operand in operands if operand is not None), *dys)
    return dx, jnp.sum(dw, axis=(0, 1)), jnp.sum(db, axis=(0, 1))


def pieces(parts):
    """Each of a conv's `parts` (`(width, head)` in channel order) as its first channel, width and head."""
    first = 0
    for width, head in parts:
        yield first, width, head
        first += width


def _backward_walk(piece: _Piece, dy_ref, outs, exchanged, *, head: int | None, activation: bool, eps: float,
                   tokens: int, interpret: bool):
    """A program's tokens' gradients, walked from its block's last and the K-1 past it which the next block
    owns, carrying the pre-activation gradients of the K-1 tokens after a token: its input's gradient written
    once at its channels (`piece.columns`), its tap and bias gradients summed over its own tokens compensated
    (`precision.compensated_add`), each block's rounded once: summed plainly a token at a time, the taps'
    gradient was 1.36 times XLA's conv's distance from float64 interpreted and 1.51 natively (row 5441)."""
    dx_ref, dw_ref, db_ref = outs
    count, length = piece.count, piece.length
    block = lax.axis_index("tokens")
    zero = jnp.zeros((piece.lanes,), jnp.float32)
    group = _group(head, piece.lanes)
    # The walk's tokens a whole number of turns: those past the block's last K-1 add only to tokens no
    # program here owns. A turn's tokens hold their inputs and output gradients a vector each beside the
    # walk's compensated tap and bias sums (two vectors each), so fewer a wider vector, `UNROLLED` channels
    # a thread at most: eight tokens of eight channels spilled 2140 bytes a thread, and two of eight beside
    # the compensated sums 24 and 36 bytes on a B200 and 16 on an sm120, where one of eight holds every
    # training case in 250 to 255 registers on sm90, sm100 and sm120 (the conv's resource gate).
    unrolled = max(1, UNROLLED // (piece.lanes // scan.THREADS))
    steps = -(-(tokens + count - 1) // unrolled) * unrolled
    last = piece.start + steps - 1

    def gradient(t, window, dy):
        """`t`'s pre-activation gradient from its output's `dy`, through the norm and the SiLU."""
        pre = piece.preactivation(window)
        out, gate = _activated(pre, activation)
        grad = _kept(piece.kept(t), dy)
        if head is not None:
            scale = scan.inverse_norm(out, eps, exchanged, group=group, interpret=interpret)
            unit = out * scale
            projected = (scan.grouped_summed(unit * grad, group, interpret=interpret) if group is not None
                         else scan.summed(unit * grad, exchanged, interpret=interpret))
            grad = scale * (grad - projected * unit)
        if gate is not None:
            grad = grad * gate + (grad * pre) * (gate * (1.0 - gate))
        return grad

    def turn(i, carry):
        held, later, owners, sums = carry
        newest = last - i * unrolled
        tokens_ = [newest - u for u in range(unrolled)]
        reads = [piece.read(t - count + 1) for t in tokens_]
        dys = [_loaded(dy_ref, piece.row, t, piece.local, length) for t in tokens_]
        for t, oldest, dy in zip(tokens_, reads, dys, strict=True):
            read = [oldest, *held]
            window = piece.window(read)
            grad, document = gradient(t, window, dy), read[-1][1]
            owned = (t < piece.start + tokens) & (t < length)
            sums = [compensated_add(total, (_kept(owned, term), zero))
                    for total, term in zip(sums, [*(grad * inputs for inputs in window), grad], strict=True)]
            dx = zero
            reads = zip([grad, *later], [document, *owners], strict=True)
            for shift, (value, theirs) in enumerate(reads):
                dx = dx + _kept(theirs == document, value) * piece.taps[count - 1 - shift]
            _stored(dx_ref, (piece.row, t, piece.columns), dx, owned)
            held, later, owners = read[:-1], [grad, *later[:-1]], [document, *owners[:-1]]
        return held, later, owners, sums

    held = [piece.read(last - count + 2 + j) for j in range(count - 1)]
    carry = (held, [zero] * (count - 1), [jnp.int32(-2)] * (count - 1), [(zero, zero)] * (count + 1))
    *_, sums = lax.fori_loop(0, steps // unrolled, turn, carry)
    # Each compensated sum's two halves added once, as it is stored.
    totals = [high + low for high, low in sums]
    for tap in range(count):
        plgpu.store(dw_ref.at[piece.row, block, tap, piece.columns], totals[tap], optimized=False)
    plgpu.store(db_ref.at[piece.row, block, piece.columns], totals[count], optimized=False)
