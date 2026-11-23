"""The diagonal recurrence on Mosaic GPU (`kernels.diagonal`'s kernels): `h_t = a_t h_{t-1} + x_t` a
channel, `a_t = exp(g_t)`, each token's output its state, a packed document's first token `x_t` alone (HGRN's,
fla's `chunk_hgrn`; RG-LRU's and MinGRU's carry). A program takes a row, a span of its tokens and a
warpgroup's channels, each thread its channels walking the span's tokens in order, the short conv's layout
(`conv`): each token's channels read in whole lines, no exchange between threads.

A call of one span walks each row from its entering state in one launch. Over more, a launch summarises each
span as its affine map, its decay `prod a_t` and its last state from zero; XLA's scan over the spans' maps
gives each its entering state; and a launch walks each span from it, writing its outputs and its last state.
The backward mirrors it from the last token, a token's state's cotangent `q_t = dy_t + a_{t+1} q_{t+1}`: a
launch summarises each span's map from its last state's cotangent to its entering state's, XLA's scan gives
each span the cotangent its last state takes from the later ones and the final state, and a launch walks each
span back, writing `dx_t = q_t` and `dg_t = (q_t h_{t-1}) a_t`. Its states `h_{t-1}` are recomputed from the
span's entering state, which the forward keeps: a walk forward keeps each turn's entering state in shared
memory, and each turn's states are recomputed in registers as the walk back reaches it. Nothing `[B, T, D]` is
kept but the operands (fla keeps each token's state in fp32, its output, fla/ops/hgrn/chunk.py).

The arithmetic is the recurrence's, a token at a time, every state and cotangent fp32, a token's decay its own
exponential, as the token recurrence and fla take it; `dg_t`'s product in the order reverse mode forms the
token recurrence's, `q_t h_{t-1}` first. A span's decay is the product of its tokens' as its walk applies
them, kept as a mantissa and an exponent (`_decayed`, each scaling exact) and applied by `ldexp` (`_scaled`):
held as one float32 it underflowed where the states it scaled were normal (`g = -2` over 64 tokens from a
state of 2^100, ReviewRecurrenceBatch); a log decay summed and exponentiated errs by `|sum g|` times float32's
epsilon, past the product's. Its write from zero is carried in halves, doubled once its entering value is
added (`_entering`). A document's first token selects its own write, and a cut selects zero, never multiplying
by a zero decay: an overflowed document's infinite state times zero was NaN in the next.
"""

import dataclasses

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn import derivatives
from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import mosaic, outputs, scan
from dew.nn.kernels.templates.recurrence import conv

TURN = conv.UNROLLED
"""The tokens a turn of a walk reads at once, their loads issued before the first is used (`conv.UNROLLED`);
the backward keeps a state a turn."""

MOST = 256
"""The most tokens a program walks: the backward keeps `MOST / TURN` states of its channels in shared
memory, 32 KB at 256 channels a program."""


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


def walk(rows: int, length: int, channels: int, dtype, family: mosaic.Family, interpret: bool) -> Walk:
    """The launches' facts for a call of `rows` rows of `length` tokens and `channels` channels in `dtype`:
    a program's channels the conv's (`conv.lanes`, whole lines a warp), its tokens as many as leave the
    fewest programs filling the device once (`conv.span`), at most `MOST`, whole turns."""
    lanes = conv.lanes(((channels, None),), dtype)
    target = generation.interpreted_target() if interpret else generation.gpu_target()
    span = min(conv.span(length, rows, channels // lanes, target), MOST)
    return Walk(rows, length, channels, -(-span // TURN) * TURN, lanes, family, interpret)


class _Program:
    """A program's view: its row, span and channels, and its loads and stores at them."""

    def __init__(self, shape: Walk, starts_ref):
        self.shape, self.starts_ref = shape, starts_ref
        self.row, self.block = lax.axis_index("rows"), lax.axis_index("tokens")
        self.start = self.block * shape.span
        self.columns = pl.ds(lax.axis_index("channels") * shape.lanes, shape.lanes)

    def channels(self, ref, token):
        """`ref`'s channels at `token`, fp32, zero outside the row."""
        return conv.channels_at(ref, self.row, token, self.columns, self.shape.length)

    def gate(self, g_ref, token):
        """`(g_t, a_t, first)`: the log decay and its exponential (zero and one outside the row) and whether
        `token` is a document's first, laid over the channels (None without documents)."""
        g = self.channels(g_ref, token)
        a = jnp.exp(g)
        if self.starts_ref is None:
            return g, a, None
        length = self.shape.length
        marked = self.starts_ref[self.row, jnp.clip(token, 0, length - 1)]
        return g, a, mosaic.laid((token >= 0) & (token < length) & (marked != 0), a)

    def summary(self, ref):
        """`ref`'s channels at this program's span (`[B, spans, D]`)."""
        return conv.channels_at(ref, self.row, self.block, self.columns, self.shape.spans)

    def summarised(self, ref, value) -> None:
        plgpu.store(ref.at[self.row, self.block, self.columns], value, optimized=False)

    def stored(self, ref, token, value) -> None:
        conv.stored_where(ref, (self.row, token, self.columns), value, token < self.shape.length)

    def turns(self, body, carried, *, reverse: bool = False):
        """`body(turn, tokens, carried)` over the span's turns, `turn` its index and `tokens` its `TURN`
        in order; from the last turn where `reverse`."""
        count = self.shape.span // TURN

        def turn(i, carried):
            at = count - 1 - i if reverse else i
            first = self.start + at * TURN
            return body(at, [first + u for u in range(TURN)], carried)

        return lax.fori_loop(0, count, turn, carried)


def _cut(first, value, replaced):
    """`value`, `replaced` at a document's first token (`first`, `_Program.gate`'s), selected: never a
    product with zero, which an overflowed state made NaN."""
    return value if first is None else jnp.where(first, replaced, value)


def _refs(refs, operands) -> tuple[list, list]:
    """A launch's refs as its `operands` (None where one is absent), then its outputs' and scratch's."""
    refs = iter(refs)
    return [None if operand is None else next(refs) for operand in operands], list(refs)


def _launched(entry, shape: Walk, name: str, out_type, operands, scratch=()):
    """`entry` launched over `shape`'s grid on `operands` (an absent one None)."""
    return mosaic.kernel(entry, out_type=out_type, grid=shape.grid, grid_names=("rows", "tokens", "channels"),
                         warpgroups=1, interpret=shape.interpret, family=shape.family.name,
                         compiled=mosaic.compiler_params(), name=name, scratch=scratch)(
        *(operand for operand in operands if operand is not None))


def _tokens_of(x, dtype) -> outputs.Output:
    """A `[B, T, D]` output of `x`'s axes in `dtype`."""
    return outputs.output(dtype, outputs.axis(x, 0), outputs.axis(x, 1), outputs.axis(x, 2))


def _spans_of(x, shape: Walk) -> outputs.Output:
    """A `[B, spans, D]` fp32 output of `x`'s rows and channels."""
    return outputs.output(jnp.float32, outputs.axis(x, 0), outputs.axis(x, 1).ceil_div(shape.span),
                          outputs.axis(x, 2))


RAISES = (64, 32, 16, 8, 4, 2, 1)
"""The powers of two a span's decay's mantissa is raised by in turn where it falls below their inverse
(`_decayed`), which bring any normal mantissa back to `[1/2, 1]`: a token's decay then multiplies a mantissa
of at least a half, so the product stays normal wherever the decay alone is (raised only below 2^-64, a
mantissa of e^-44 times a decay of e^-64 flushed to zero, ReviewRecurrenceBatch)."""


def _decayed(decay, a_t, first):
    """A span's decay, `(mantissa, exponent)`, past one more token's `a_t`: the mantissa times it (zero from a
    document's first token on, selected), then raised back to `[1/2, 1]` by `RAISES`' powers of two, the
    exponent lowered with each, every step exact."""
    mantissa, exponent = decay
    mantissa = _cut(first, mantissa * a_t, jnp.zeros_like(mantissa))
    for power in RAISES:
        low = mantissa < 2.0 ** -power
        mantissa = jnp.where(low, mantissa * 2.0 ** power, mantissa)
        exponent = jnp.where(low, exponent - float(power), exponent)
    return mantissa, exponent


@tuning.lowered_once(("shape",))
def summaries(x, g, starts, *, shape: Walk):
    """Each span's map from its entering state to its last, `(mantissa, exponent, state)` `[B, spans, D]`
    fp32: the product of its tokens' decays (`_decayed`, the mantissa zero where a document starts in the
    span, which keeps none of its entering state) and half its last state from zero (`_entering`)."""
    operands = (x, g, starts)

    def entry(*refs):
        (x_ref, g_ref, starts_ref), (mantissa_ref, exponent_ref, state_ref) = _refs(refs, operands)
        program = _Program(shape, starts_ref)

        def body(_, tokens, carried):
            decay, h = carried
            xs = [program.channels(x_ref, t) for t in tokens]
            gates = [program.gate(g_ref, t) for t in tokens]
            for x_t, (_, a_t, first) in zip(xs, gates, strict=True):
                decay = _decayed(decay, a_t, first)
                half = x_t * 0.5
                h = _cut(first, a_t * h + half, half)
            return decay, h

        zero = jnp.zeros((shape.lanes,), jnp.float32)
        (mantissa, exponent), h = program.turns(body, ((zero + 1.0, zero), zero))
        program.summarised(mantissa_ref, mantissa)
        program.summarised(exponent_ref, exponent)
        program.summarised(state_ref, h)

    out_type = (_spans_of(x, shape), _spans_of(x, shape), _spans_of(x, shape))
    return _launched(entry, shape, "diagonal_summaries", out_type, operands)


@tuning.lowered_once(("shape",))
def walked(x, g, starts, entering, *, shape: Walk):
    """Each token's state from its span's entering state (`entering` `[B, spans, D]` fp32), `[B, T, D]` in
    `x`'s dtype, and each span's last state, `[B, spans, D]` fp32."""
    operands = (x, g, starts, entering)

    def entry(*refs):
        (x_ref, g_ref, starts_ref, entering_ref), (out_ref, last_ref) = _refs(refs, operands)
        program = _Program(shape, starts_ref)

        def body(_, tokens, h):
            xs = [program.channels(x_ref, t) for t in tokens]
            gates = [program.gate(g_ref, t) for t in tokens]
            for t, x_t, (_, a_t, first) in zip(tokens, xs, gates, strict=True):
                h = _cut(first, a_t * h + x_t, x_t)
                program.stored(out_ref, t, h)
            return h

        program.summarised(last_ref, program.turns(body, program.summary(entering_ref)))

    return _launched(entry, shape, "diagonal_walk", (_tokens_of(x, x.dtype), _spans_of(x, shape)), operands)


@tuning.lowered_once(("shape",))
def reverse_summaries(g, starts, dy, *, shape: Walk):
    """Each span's map from its last state's cotangent to its entering state's, `(mantissa, exponent,
    cotangent)` `[B, spans, D]` fp32: the product of its tokens' decays (`_decayed`) and half the cotangent
    its own outputs' (`dy`) give its entering state (`_entering`)."""
    operands = (g, starts, dy)

    def entry(*refs):
        (g_ref, starts_ref, dy_ref), (mantissa_ref, exponent_ref, carried_ref) = _refs(refs, operands)
        program = _Program(shape, starts_ref)
        zero = jnp.zeros((shape.lanes,), jnp.float32)

        def body(_, tokens, carried):
            decay, c = carried
            gates = [program.gate(g_ref, t) for t in tokens]
            dys = [program.channels(dy_ref, t) for t in tokens]
            for (_, a_t, first), dy_t in zip(reversed(gates), reversed(dys), strict=True):
                decay = _decayed(decay, a_t, first)
                c = _cut(first, a_t * (dy_t * 0.5 + c), zero)
            return decay, c

        (mantissa, exponent), c = program.turns(body, ((zero + 1.0, zero), zero), reverse=True)
        program.summarised(mantissa_ref, mantissa)
        program.summarised(exponent_ref, exponent)
        program.summarised(carried_ref, c)

    out_type = (_spans_of(dy, shape), _spans_of(dy, shape), _spans_of(dy, shape))
    return _launched(entry, shape, "diagonal_summaries_backward", out_type, operands)


@tuning.lowered_once(("shape",))
def walked_back(x, g, starts, dy, entering, later, *, shape: Walk):
    """The gradients of `x` (`[B, T, D]` in its dtype) and `g` (in its), and each span's entering state's
    cotangent `[B, spans, D]` fp32, from the outputs' cotangent `dy`, each span's entering state (`entering`)
    and its last state's cotangent from the later tokens and the final state (`later`), each `[B, spans, D]`
    fp32. A walk forward keeps each turn's entering state in shared memory; the walk back recomputes a
    turn's states from it."""
    operands = (x, g, starts, dy, entering, later)
    kept = plgpu.SMEM((shape.span // TURN, shape.lanes), jnp.float32)

    def entry(*refs):
        given, made = _refs(refs, operands)
        x_ref, g_ref, starts_ref, dy_ref, entering_ref, later_ref = given
        dx_ref, dg_ref, leaving_ref, kept_ref = made
        program = _Program(shape, starts_ref)
        layout = scan.blocked(shape.lanes)
        zero = jnp.zeros((shape.lanes,), jnp.float32)

        def forward(turn, tokens, h):
            plgpu.store(kept_ref.at[turn], h, optimized=False)
            xs = [program.channels(x_ref, t) for t in tokens]
            gates = [program.gate(g_ref, t) for t in tokens]
            for x_t, (_, a_t, first) in zip(xs, gates, strict=True):
                h = _cut(first, a_t * h + x_t, x_t)
            return h

        program.turns(forward, program.summary(entering_ref))

        def back(turn, tokens, c):
            xs = [program.channels(x_ref, t) for t in tokens]
            gates = [program.gate(g_ref, t) for t in tokens]
            dys = [program.channels(dy_ref, t) for t in tokens]
            h, before = plgpu.load(kept_ref.at[turn], layout=layout, optimized=False), []
            for x_t, (_, a_t, first) in zip(xs, gates, strict=True):
                before.append(h)
                h = _cut(first, a_t * h + x_t, x_t)
            walked = list(zip(tokens, gates, dys, before, strict=True))
            for t, (_, a_t, first), dy_t, h_t in reversed(walked):
                q = dy_t + c
                program.stored(dx_ref, t, q)
                program.stored(dg_ref, t, _cut(first, (q * h_t) * a_t, zero))
                c = _cut(first, a_t * q, zero)
            return c

        program.summarised(leaving_ref, program.turns(back, program.summary(later_ref), reverse=True))

    out_type = (_tokens_of(x, x.dtype), _tokens_of(g, g.dtype), _spans_of(x, shape))
    return _launched(entry, shape, "diagonal_walk_backward", out_type, operands, scratch=(kept,))


def _scaled(values, decay):
    """`values` times a span's decay, `(mantissa, exponent)` (`_decayed`): their mantissas multiplied and
    the exponents applied by `ldexp`, exact where the product is normal; zero at a cut (a zero mantissa),
    selected, as an infinite value times zero is NaN."""
    mantissa, exponent = decay
    fraction, shift = jnp.frexp(mantissa)
    scaled = jnp.ldexp(values * fraction, exponent.astype(jnp.int32) + shift)
    return jnp.where(mantissa == 0.0, jnp.zeros_like(values), scaled)


def _composed(first, then):
    """Two spans' maps, `(decay, value)` each, as one, `first`'s applied first: their decays multiplied (each
    mantissa's own exponent taken into the sum, `frexp`, so the product stays normal), `first`'s value scaled
    by `then`'s decay plus `then`'s (`associative_scan`'s operator either way)."""
    (mantissa, exponent), value = first
    (then_mantissa, then_exponent), then_value = then
    fraction, shift = jnp.frexp(mantissa)
    then_fraction, then_shift = jnp.frexp(then_mantissa)
    decay = (fraction * then_fraction, exponent + then_exponent + (shift + then_shift).astype(exponent.dtype))
    return decay, _scaled(value, (then_mantissa, then_exponent)) + then_value


def _entering(mantissa, exponent, halves, given, *, reverse: bool = False):
    """Each span's value as its walk enters it, `[B, spans, D]`: `given` `[B, D]` (the state entering the
    first span; `reverse`, the final state's cotangent entering the last) carried through the spans' maps
    (`mantissa`, `exponent`, `halves`, as `summaries` or `reverse_summaries` give them) by XLA's associative
    scan, in halves through it and doubled once out. A map's value from zero is its last value less its
    entering value's decayed share, of two finite values' difference, so its half is finite: whole, writes of
    2^127 and 2^127 after a state of -2^127 summed to infinity where every state was finite
    (ReviewRecurrenceBatch)."""
    through, written = lax.associative_scan(_composed, ((mantissa, exponent), halves), axis=1,
                                            reverse=reverse)
    after = (_scaled(given[:, None] * 0.5, through) + written) * 2.0
    if reverse:
        return jnp.concatenate([after[:, 1:], given[:, None]], axis=1)
    return jnp.concatenate([given[:, None], after[:, :-1]], axis=1)


def _forward(x, g, state, starts, shape: Walk):
    """`(outputs, final state, each span's entering state)`."""
    entering = state[:, None]
    if shape.spans > 1:
        mantissa, exponent, written = summaries(x, g, starts, shape=shape)
        entering = _entering(mantissa, exponent, written, state)
    out, last = walked(x, g, starts, entering, shape=shape)
    return out, last[:, -1], entering


def _primal(x, g, state, starts, shape: Walk):
    return _forward(x, g, state, starts, shape)[:2]


def _forward_rule(x, g, state, starts, shape: Walk):
    out, final, entering = _forward(x, g, state, starts, shape)
    return (out, final), (x, g, starts, entering)


def _backward_rule(shape: Walk, residuals, cotangents):
    x, g, starts, entering = residuals
    dy, dfinal = cotangents
    dy, dfinal = dy.astype(x.dtype), dfinal.astype(jnp.float32)
    later = dfinal[:, None]
    if shape.spans > 1:
        mantissa, exponent, carried = reverse_summaries(g, starts, dy, shape=shape)
        later = _entering(mantissa, exponent, carried, dfinal, reverse=True)
    dx, dg, leaving = walked_back(x, g, starts, dy, entering, later, shape=shape)
    return dx, dg, leaving[:, 0], None


_scan = derivatives.contracted("test_the_diagonal_scan_holds_its_orders")(
    derivatives.declared_derivative(derivatives.KernelGlue(), _primal, _forward_rule, _backward_rule,
                                    nondiff_argnums=(4,)))


def diagonal_scan(x, g, state, starts, *, family: mosaic.Family, interpret: bool = False):
    """`kernels.diagonal.diagonal_scan` on the kernels: `x`, `g` `[B, T, D]` (`D` whole warpgroups of
    channels), `state` `[B, D]` the state entering, `starts` `[B, T]` packed documents' first tokens or None;
    `(output [B, T, D] in x's dtype, final state [B, D] fp32)`."""
    rows, length, channels = x.shape
    shape = walk(rows, length, channels, x.dtype, family, interpret)
    flags = None if starts is None else jnp.asarray(starts, jnp.int8)
    return _scan(x, g, jnp.asarray(state, jnp.float32), flags, shape)
