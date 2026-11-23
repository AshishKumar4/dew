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


RAISES = (64, 32, 16, 8, 4, 2, 1)
"""The powers of two a product's mantissa is raised by in turn where it falls below `2^(1 - power)`
(`normalised`), which bring any normal mantissa back to `[1, 2)`."""


def normalised(mantissa, exponent):
    """A product's mantissa raised back to `[1, 2)` by `RAISES`' powers of two, its exponent (fp32, whole)
    lowered with each, every step exact: a factor at most one that is normal alone then multiplies a mantissa
    of at least one, and the product stays normal (a mantissa raised only to `[1/2, 1]` times a decay of
    e^-87.25 fell below float32's least normal, ReviewRecurrenceBatch). Zero stays zero, its exponent lowered,
    which `extended` reads as zero."""
    for power in RAISES:
        low = mantissa < 2.0 ** (1 - power)
        mantissa = jnp.where(low, mantissa * 2.0 ** power, mantissa)
        exponent = jnp.where(low, exponent - float(power), exponent)
    return mantissa, exponent


LARGE = 2.0 ** 64
"""The magnitude past which a running sum is lowered by 2^64 (`summed_on`)."""


def summed_on(carried, decayed, term, first):
    """A running sum `(value, scale)`, its value `2^(64 scale)` times its `value`, walked on by one token:
    `decayed` (its value times the token's decay, at its scale) plus `term` lowered to its scale (2^-64 a
    step, a term so lowered below the value's precision a part it no longer holds), a document's first token
    `term` alone at scale zero (selected), and a value past `LARGE` lowered by 2^64, its scale raised. A sum
    from zero is a span's last value less its entering one's decayed share, at most twice float32's largest,
    so it is held in range at two steps and none of its terms is lowered unless the sum outweighs it:
    summed whole, two writes of 2^127 after a state of -2^127 were infinite; halved, a least normal write was
    lost (ReviewRecurrenceBatch)."""
    _, scale = carried
    lowered = jnp.where(scale >= 1.0, term * 2.0 ** -64, term)
    lowered = jnp.where(scale >= 2.0, lowered * 2.0 ** -64, lowered)
    value = cut(first, decayed + lowered, term)
    scale = cut(first, scale, jnp.zeros_like(scale))
    large = jnp.abs(value) >= LARGE
    return jnp.where(large, value * 2.0 ** -64, value), jnp.where(large, scale + 1.0, scale)


ZERO_EXPONENT = -(2 ** 20)
"""An extended float's exponent for zero (`extended`), far below any other's: a sum's terms aligned to the
larger exponent leave a zero none of its own."""

LEAST_EXPONENT = -1024
"""The least exponent an extended float keeps (`extended`): any value past it is zero against every value a
float32 sum holds."""


def extended(mantissa, exponent):
    """`mantissa 2^exponent` as an extended float `(fraction, exponent)`: the fraction in `[1/2, 1)` and the
    exponent int32 (`jnp.frexp`), so a span's decays and sums compose (`multiplied`, `added`) whatever their
    exponents, each step rounded once, where held as float32 they left its range; zero, or a value below
    2^`LEAST_EXPONENT`, as `(0, ZERO_EXPONENT)`."""
    fraction, shift = jnp.frexp(mantissa)
    exponent = jnp.asarray(exponent).astype(jnp.int32) + shift
    zero = (fraction == 0.0) | (exponent < LEAST_EXPONENT)
    return jnp.where(zero, 0.0, fraction), jnp.where(zero, ZERO_EXPONENT, exponent)


def multiplied(first, second):
    """Two extended floats' product."""
    return extended(first[0] * second[0], first[1] + second[1])


def added(first, second):
    """Two extended floats' sum: each fraction aligned to the larger exponent (`jnp.ldexp`), summed, rounded
    once."""
    top = jnp.maximum(first[1], second[1])
    return extended(jnp.ldexp(first[0], first[1] - top) + jnp.ldexp(second[0], second[1] - top), top)


def valued(value):
    """An extended float as float32 (`jnp.ldexp`): zero below its least, infinite past its largest."""
    return jnp.ldexp(value[0], value[1])
