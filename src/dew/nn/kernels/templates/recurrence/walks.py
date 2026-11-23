"""A channel walk on Mosaic GPU, the recurrences whose state is a few values a channel (the diagonal scan,
`diagonal`; RWKV4's WKV, `wkv4`): a program takes a row, a span of its tokens and a block of channels, each
thread its channels walking the span's tokens in order, the short conv's layout (`conv`): each token's
channels read in whole lines, no exchange between threads. A call over several spans summarises each,
XLA's associative scan joins the summaries for each span's entering state, and a launch walks each span
from its own.
"""

import dataclasses
from collections.abc import Callable

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation
from dew.nn.kernels.templates import mosaic, outputs, scan
from dew.nn.kernels.templates.recurrence import conv

TURN = conv.UNROLLED
"""The tokens a turn of a walk reads at once, their loads issued before the first is used (`conv.UNROLLED`);
a backward keeps its state a turn."""


@dataclasses.dataclass(frozen=True)
class Walk:
    """A call's launches' facts: its rows, tokens and channels, the tokens a program walks (`span`, whole
    turns), the channels it takes (`lanes`), its bundle's family and whether it is interpreted."""

    rows: int
    length: int
    channels: int
    span: int
    lanes: int
    family: mosaic.Family
    interpret: bool

    @property
    def spans(self) -> int:
        return -(-self.length // self.span)

    @property
    def grid(self) -> tuple[int, int, int]:
        return self.rows, self.spans, self.channels // self.lanes

    def tokens(self) -> list[int]:
        """Each span's tokens in the row, the last span's those short of its end."""
        return [min(self.span, self.length - first) for first in range(0, self.length, self.span)]


def walk(rows: int, length: int, channels: int, lanes: int, most: int, family: mosaic.Family,
         interpret: bool) -> Walk:
    """The launches' facts for a call of `rows` rows of `length` tokens and `channels` channels, `lanes` a
    program: its tokens as many as leave the fewest programs filling the device once (`conv.span`), at most
    `most` (what a backward keeps of its turns), whole turns."""
    target = generation.interpreted_target() if interpret else generation.gpu_target()
    span = min(conv.span(length, rows, channels // lanes, target), most)
    return Walk(rows, length, channels, -(-span // TURN) * TURN, lanes, family, interpret)


class Program:
    """A program's view: its row, span and channels, and its loads and stores at them."""

    def __init__(self, shape: Walk, starts_ref):
        self.shape, self.starts_ref = shape, starts_ref
        self.row, self.block = lax.axis_index("rows"), lax.axis_index("tokens")
        self.start = self.block * shape.span
        self.columns = pl.ds(lax.axis_index("channels") * shape.lanes, shape.lanes)

    def channels(self, ref, token):
        """`ref`'s channels at `token`, fp32, zero outside the row."""
        return conv.channels_at(ref, self.row, token, self.columns, self.shape.length)

    def parameter(self, ref):
        """A `[D]` parameter's channels, fp32."""
        return plgpu.load(ref.at[self.columns], layout=scan.blocked(self.shape.lanes),
                          optimized=False).astype(jnp.float32)

    def inside(self, token, like):
        """Whether `token` is in the row, laid over `like`'s channels (`mosaic.laid`)."""
        return mosaic.laid(token < self.shape.length, like)

    def first(self, token, like):
        """Whether `token` is a document's first, laid over `like`'s channels; None without documents."""
        if self.starts_ref is None:
            return None
        length = self.shape.length
        marked = self.starts_ref[self.row, jnp.clip(token, 0, length - 1)]
        return mosaic.laid((token >= 0) & (token < length) & (marked != 0), like)

    def summary(self, ref):
        """`ref`'s channels at this program's span (`[B, spans, D]`)."""
        return conv.channels_at(ref, self.row, self.block, self.columns, self.shape.spans)

    def summarised(self, ref, value) -> None:
        plgpu.store(ref.at[self.row, self.block, self.columns], value, optimized=False)

    def stored(self, ref, token, value) -> None:
        conv.stored_where(ref, (self.row, token, self.columns), value, token < self.shape.length)

    def turns(self, body: Callable, carried, *, reverse: bool = False):
        """`body(turn, tokens, carried)` over the span's turns, `turn` its index and `tokens` its `TURN`
        in order; from the last turn where `reverse`."""
        count = self.shape.span // TURN

        def turn(i, carried):
            at = count - 1 - i if reverse else i
            first = self.start + at * TURN
            return body(at, [first + u for u in range(TURN)], carried)

        return lax.fori_loop(0, count, turn, carried)


def cut(first, value, replaced):
    """`value`, `replaced` where `first` holds (a document's first token, `Program.first`'s, None for none),
    selected: never a product with zero, which an overflowed value made NaN."""
    return value if first is None else jnp.where(first, replaced, value)


def refs_of(refs, operands) -> tuple[list, list]:
    """A launch's refs as its `operands` (None where one is absent), then its outputs' and scratch's."""
    refs = iter(refs)
    return [None if operand is None else next(refs) for operand in operands], list(refs)


def launch(entry, shape: Walk, name: str, out_type, operands, scratch=()):
    """`entry` launched over `shape`'s grid on `operands` (an absent one None)."""
    return mosaic.kernel(entry, out_type=out_type, grid=shape.grid, grid_names=("rows", "tokens", "channels"),
                         warpgroups=1, interpret=shape.interpret, family=shape.family.name,
                         compiled=mosaic.compiler_params(), name=name, scratch=scratch)(
        *(operand for operand in operands if operand is not None))


def per_token(x, dtype) -> outputs.Output:
    """A `[B, T, D]` output of `x`'s axes in `dtype`."""
    return outputs.output(dtype, outputs.axis(x, 0), outputs.axis(x, 1), outputs.axis(x, 2))


def per_span(x, shape: Walk) -> outputs.Output:
    """A `[B, spans, D]` fp32 output of `x`'s rows and channels."""
    return outputs.output(jnp.float32, outputs.axis(x, 0), outputs.axis(x, 1).ceil_div(shape.span),
                          outputs.axis(x, 2))
