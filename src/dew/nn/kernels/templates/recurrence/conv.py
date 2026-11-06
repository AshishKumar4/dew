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

from dew.nn.kernels.templates import mosaic, scan

TOKENS = 64
"""The tokens a program walks: a backward program computes K-1 past them, `(K - 1) / TOKENS` extra."""


def fits(width: int, head: int | None) -> bool:
    """Whether a piece of `width` channels steps here: whole warpgroups of them, a normalised piece's head
    one (its row sum the warpgroup's)."""
    return width % scan.THREADS == 0 and head in (None, scan.THREADS)



def _kept(kept, value: jax.Array) -> jax.Array:
    """`value` where the scalar `kept` holds, else zeros (`mosaic.laid`)."""
    return jnp.where(mosaic.laid(kept, value), value, jnp.zeros_like(value))


def _loaded(ref, row, token, columns, length: int):
    """`ref`'s `[THREADS]` channels at `token` of `row`, fp32, zero before the row's first and past its
    last."""
    inside = (token >= 0) & (token < length)
    at = ref.at[row, jnp.clip(token, 0, length - 1), columns]
    value = plgpu.load(at, layout=scan.blocked(scan.THREADS), optimized=False).astype(jnp.float32)
    return _kept(inside, value)


def _owner(documents_ref, row, token, length: int):
    """The packed document `token` of `row` is in, -1 outside the row; 0 without documents."""
    if documents_ref is None:
        return jnp.int32(0)
    inside = (token >= 0) & (token < length)
    return jnp.where(inside, documents_ref[row, jnp.clip(token, 0, length - 1)], -1)


def _activated(pre, activation: bool):
    """The SiLU of the conv's sums, its sigmoid one quotient, and the sigmoid (None without one)."""
    if not activation:
        return pre, None
    gate = 1.0 / (1.0 + jnp.exp(-pre))
    return pre / (1.0 + jnp.exp(-pre)), gate


def _scale(out, eps: float, exchanged, interpret: bool):
    """`1 / sqrt(sum(out^2) + eps)` over the program's head, its root and quotient each rounded once."""
    return 1.0 / jnp.sqrt(scan.summed(out * out, exchanged, interpret=interpret) + eps)


class _Piece:
    """A program's view of its piece: its row, its tokens' first, its channels in the operand and the
    piece, the taps and bias it reads, and the masks."""

    def __init__(self, refs, *, first: int, length: int, count: int):
        self.x_ref, self.w_ref, self.b_ref, self.valid_ref, self.documents_ref = refs
        self.row, self.length, self.count = lax.axis_index("rows"), length, count
        self.start = lax.axis_index("tokens") * TOKENS
        local = lax.axis_index("channels") * scan.THREADS
        self.local, self.columns = pl.ds(local, scan.THREADS), pl.ds(first + local, scan.THREADS)
        layout = scan.blocked(scan.THREADS)
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

    def window(self, token):
        """`token`'s inputs, oldest first, each of another document zero, and its document."""
        document = self.owner(token)
        found = [self.input(token - self.count + 1 + tap) for tap in range(self.count)]
        owners = [self.owner(token - self.count + 1 + tap) for tap in range(self.count)]
        return [_kept(theirs == document, value)
                for value, theirs in zip(found, owners, strict=True)]

    def preactivation(self, window):
        total = window[0] * self.taps[0]
        for inputs, tap in zip(window[1:], self.taps[1:], strict=True):
            total = total + inputs * tap
        return total if self.bias is None else total + self.bias


def _named(kernel: str, first: int, width: int) -> str:
    """A piece's launch's name, which its call's location carries (`lowered.kernel_calls`): its kernel's and
    its channels'. A call's pieces are each a module of its own (`first` static), which a profile and the
    compile gate tell apart by name (`kernel_compile.kernel_modules`)."""
    return f"{kernel}_{first}_{width}"


def forward(x, taps, bias, valid, documents, *, first: int, width: int, head: int | None, activation: bool,
            eps: float, dtype, family: mosaic.Family, interpret: bool = False):
    """A piece's output `[B, T, width]` of `x`'s channels `first` to `first + width`, convolved by `taps`
    `[C, K]` plus `bias`, activated and, with a `head`, l2-normalised over each head."""
    rows, length, _ = x.shape
    count = taps.shape[1]
    operands = [x, taps.T, bias, valid, documents]

    def entry(*refs):
        *given, out_ref, exchanged = refs
        given = iter(given)
        piece = _Piece([next(given) if operand is not None else None for operand in operands], first=first,
                       length=length, count=count)

        def token(i, carry):
            t = piece.start + i
            out, _ = _activated(piece.preactivation(piece.window(t)), activation)
            if head is not None:
                out = out * _scale(out, eps, exchanged, interpret)
            out = _kept(piece.kept(t), out)

            @pl.when(t < length)
            def _():
                plgpu.store(out_ref.at[piece.row, t, piece.local], out.astype(out_ref.dtype),
                            optimized=False)
            return carry

        lax.fori_loop(0, TOKENS, token, None)

    name = _named("short_conv", first, width)
    launched = mosaic.kernel(entry, out_type=jax.ShapeDtypeStruct((rows, length, width), dtype),
                             grid=(rows, pl.cdiv(length, TOKENS), width // scan.THREADS),
                             grid_names=("rows", "tokens", "channels"), warpgroups=1, interpret=interpret,
                             family=family.name, name=name, scratch=(scan.scratch(),))
    return launched(*(operand for operand in operands if operand is not None))


def backward(x, taps, bias, valid, documents, dy, *, first: int, width: int, head: int | None,
             activation: bool, eps: float, family: mosaic.Family, interpret: bool = False):
    """A piece's gradients: its channels' input gradient `[B, T, width]` fp32, and its tap and bias
    gradients `[K, width]` and `[width]` fp32 (the bias's zero without one)."""
    rows, length, _ = x.shape
    count = taps.shape[1]
    blocks = pl.cdiv(length, TOKENS)
    operands = [x, taps.T, bias, valid, documents]

    def entry(*refs):
        *given, dy_ref, dx_ref, dw_ref, db_ref, exchanged = refs
        given = iter(given)
        piece = _Piece([next(given) if operand is not None else None for operand in operands], first=first,
                       length=length, count=count)
        block = lax.axis_index("tokens")
        zero = jnp.zeros((scan.THREADS,), jnp.float32)

        def gradient(t, window):
            """`t`'s pre-activation gradient from its output's, through the norm and the SiLU."""
            pre = piece.preactivation(window)
            out, gate = _activated(pre, activation)
            grad = _kept(piece.kept(t), _loaded(dy_ref, piece.row, t, piece.local, length))
            if head is not None:
                scale = _scale(out, eps, exchanged, interpret)
                unit = out * scale
                grad = scale * (grad - scan.summed(unit * grad, exchanged, interpret=interpret) * unit)
            if gate is not None:
                grad = grad * gate + (grad * pre) * (gate * (1.0 - gate))
            return grad

        def token(i, carry):
            later, owners, sums = carry
            t = piece.start + TOKENS + count - 2 - i
            window = piece.window(t)
            grad, document = gradient(t, window), piece.owner(t)
            owned = (t < piece.start + TOKENS) & (t < length)
            sums = [*(total + _kept(owned, grad * inputs)
                      for total, inputs in zip(sums[:-1], window, strict=True)),
                    sums[-1] + _kept(owned, grad)]
            reads = [grad, *later]
            dx = zero
            for shift, (value, theirs) in enumerate(zip(reads, [document, *owners], strict=True)):
                dx = dx + _kept(theirs == document, value) * piece.taps[count - 1 - shift]

            @pl.when(owned)
            def _():
                plgpu.store(dx_ref.at[piece.row, t, piece.local], dx, optimized=False)
            return [grad, *later[:-1]], [document, *owners[:-1]], sums

        carry = ([zero] * (count - 1), [jnp.int32(-2)] * (count - 1), [zero] * (count + 1))
        _, _, sums = lax.fori_loop(0, TOKENS + count - 1, token, carry)
        for tap in range(count):
            plgpu.store(dw_ref.at[piece.row, block, tap, piece.local], sums[tap], optimized=False)
        plgpu.store(db_ref.at[piece.row, block, piece.local], sums[count], optimized=False)

    f32 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.float32)
    out_type = (f32((rows, length, width)), f32((rows, blocks, count, width)), f32((rows, blocks, width)))
    name = _named("short_conv_backward", first, width)
    launched = mosaic.kernel(entry, out_type=out_type, grid=(rows, blocks, width // scan.THREADS),
                             grid_names=("rows", "tokens", "channels"), warpgroups=1, interpret=interpret,
                             family=family.name, name=name, scratch=(scan.scratch(),))
    dx, dw, db = launched(*(operand for operand in operands if operand is not None), dy)
    return dx, jnp.sum(dw, axis=(0, 1)), jnp.sum(db, axis=(0, 1))
