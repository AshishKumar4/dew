"""The diagonal recurrence on Mosaic GPU (`kernels.diagonal_scan`'s kernels): `h_t = a_t h_{t-1} +
x_t` a channel, `a_t = exp(g_t)`, zero at a packed document's first token, each token's output its state
(HGRN's, fla's `chunk_hgrn`; RG-LRU's and MinGRU's carry). A program takes a row, a span of its tokens and a
warpgroup's channels, each thread its channels walking the span's tokens in order, the short conv's layout
(`conv`): each token's channels read in whole lines, no exchange between threads.

A call of one span walks each row from its entering state in one launch. Over more, a launch summarises each
span as its affine map, its decay `prod a_t` and its last state from zero; XLA's scan over the spans' maps
gives each its entering state; and a launch walks each span from it, writing its outputs and its last state.
The backward mirrors it from the last token, a token's state's cotangent `q_t = dy_t + a_{t+1} q_{t+1}`: a
launch summarises each span's map from its last state's cotangent to its entering state's, XLA's scan gives
each span the cotangent its last state takes from the later ones and the final state, and a launch walks each
span back, writing `dx_t = q_t` and `dg_t = q_t a_t h_{t-1}`. Its states `h_{t-1}` are recomputed from the
span's entering state, which the forward keeps: a walk forward keeps each turn's entering state in shared
memory, and each turn's states are recomputed in registers as the walk back reaches it. Nothing `[B, T, D]`
is kept but the operands (fla keeps each token's state in fp32, its output, fla/ops/hgrn/chunk.py).

The arithmetic is the recurrence's, a token at a time, every state and cotangent fp32: a span's decay the
product of its tokens', as its walk applies them.
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

    def decay(self, g_ref, token):
        """`a_t`: `exp(g_t)`, zero at a document's first token; one outside the row, where `g` reads zero."""
        a = jnp.exp(self.channels(g_ref, token))
        if self.starts_ref is None:
            return a
        length = self.shape.length
        flag = self.starts_ref[self.row, jnp.clip(token, 0, length - 1)]
        first = (token >= 0) & (token < length) & (flag != 0)
        return jnp.where(mosaic.laid(first, a), jnp.zeros_like(a), a)

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


def _ones(shape: Walk):
    return jnp.ones((shape.lanes,), jnp.float32)


def _zeros(shape: Walk):
    return jnp.zeros((shape.lanes,), jnp.float32)


@tuning.lowered_once(("shape",))
def summaries(x, g, starts, *, shape: Walk):
    """Each span's map from its entering state to its last, `(decay, state)` `[B, spans, D]` fp32: the
    product of its tokens' decays and its last state from zero."""
    operands = (x, g, starts)

    def entry(*refs):
        (x_ref, g_ref, starts_ref), (decay_ref, state_ref) = _refs(refs, operands)
        program = _Program(shape, starts_ref)

        def body(_, tokens, carried):
            decay, h = carried
            xs = [program.channels(x_ref, t) for t in tokens]
            decays = [program.decay(g_ref, t) for t in tokens]
            for x_t, a_t in zip(xs, decays, strict=True):
                decay, h = decay * a_t, a_t * h + x_t
            return decay, h

        decay, h = program.turns(body, (_ones(shape), _zeros(shape)))
        program.summarised(decay_ref, decay)
        program.summarised(state_ref, h)

    out_type = (_spans_of(x, shape), _spans_of(x, shape))
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
            decays = [program.decay(g_ref, t) for t in tokens]
            for t, x_t, a_t in zip(tokens, xs, decays, strict=True):
                h = a_t * h + x_t
                program.stored(out_ref, t, h)
            return h

        program.summarised(last_ref, program.turns(body, program.summary(entering_ref)))

    return _launched(entry, shape, "diagonal_walk", (_tokens_of(x, x.dtype), _spans_of(x, shape)), operands)


@tuning.lowered_once(("shape",))
def reverse_summaries(g, starts, dy, *, shape: Walk):
    """Each span's map from its last state's cotangent to its entering state's, `(decay, cotangent)`
    `[B, spans, D]` fp32: the product of its tokens' decays and the cotangent its own outputs' (`dy`) give
    its entering state."""
    operands = (g, starts, dy)

    def entry(*refs):
        (g_ref, starts_ref, dy_ref), (decay_ref, carried_ref) = _refs(refs, operands)
        program = _Program(shape, starts_ref)

        def body(_, tokens, carried):
            decay, c = carried
            decays = [program.decay(g_ref, t) for t in tokens]
            dys = [program.channels(dy_ref, t) for t in tokens]
            for a_t, dy_t in zip(reversed(decays), reversed(dys), strict=True):
                decay, c = decay * a_t, a_t * (dy_t + c)
            return decay, c

        decay, c = program.turns(body, (_ones(shape), _zeros(shape)), reverse=True)
        program.summarised(decay_ref, decay)
        program.summarised(carried_ref, c)

    out_type = (_spans_of(dy, shape), _spans_of(dy, shape))
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

        def forward(turn, tokens, h):
            plgpu.store(kept_ref.at[turn], h, optimized=False)
            xs = [program.channels(x_ref, t) for t in tokens]
            decays = [program.decay(g_ref, t) for t in tokens]
            for x_t, a_t in zip(xs, decays, strict=True):
                h = a_t * h + x_t
            return h

        program.turns(forward, program.summary(entering_ref))

        def back(turn, tokens, c):
            xs = [program.channels(x_ref, t) for t in tokens]
            decays = [program.decay(g_ref, t) for t in tokens]
            dys = [program.channels(dy_ref, t) for t in tokens]
            h, before = plgpu.load(kept_ref.at[turn], layout=layout, optimized=False), []
            for x_t, a_t in zip(xs, decays, strict=True):
                before.append(h)
                h = a_t * h + x_t
            for t, a_t, dy_t, h_t in reversed(list(zip(tokens, decays, dys, before, strict=True))):
                q = dy_t + c
                program.stored(dx_ref, t, q)
                program.stored(dg_ref, t, q * a_t * h_t)
                c = a_t * q
            return c

        program.summarised(leaving_ref, program.turns(back, program.summary(later_ref), reverse=True))

    out_type = (_tokens_of(x, x.dtype), _tokens_of(g, g.dtype), _spans_of(x, shape))
    return _launched(entry, shape, "diagonal_walk_backward", out_type, operands, scratch=(kept,))


def _entering(decay, state, entered):
    """Each span's entering state `[B, spans, D]` from the spans' maps (`decay`, `state`, as `summaries`
    gives them) and the call's (`entered`, `[B, D]`): their prefix by XLA's associative scan."""
    def composed(earlier, later):
        return earlier[0] * later[0], later[0] * earlier[1] + later[1]

    through, written = lax.associative_scan(composed, (decay, state), axis=1)
    after = through * entered[:, None] + written
    return jnp.concatenate([entered[:, None], after[:, :-1]], axis=1)


def _later(decay, carried, final):
    """Each span's last state's cotangent `[B, spans, D]` from the later spans' maps (`decay`, `carried`, as
    `reverse_summaries` gives them) and the final state's cotangent (`final`, `[B, D]`)."""
    def composed(later, earlier):
        return later[0] * earlier[0], earlier[0] * later[1] + earlier[1]

    through, given = lax.associative_scan(composed, (decay, carried), axis=1, reverse=True)
    before = through * final[:, None] + given
    return jnp.concatenate([before[:, 1:], final[:, None]], axis=1)


def _forward(x, g, state, starts, shape: Walk):
    """`(outputs, final state, each span's entering state)`."""
    entering = state[:, None]
    if shape.spans > 1:
        entering = _entering(*summaries(x, g, starts, shape=shape), state)
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
        later = _later(*reverse_summaries(g, starts, dy, shape=shape), dfinal)
    dx, dg, leaving = walked_back(x, g, starts, dy, entering, later, shape=shape)
    return dx, dg, leaving[:, 0], None


_scan = derivatives.contracted("test_the_diagonal_scan_holds_its_orders")(
    derivatives.declared_derivative(derivatives.KernelGlue(), _primal, _forward_rule, _backward_rule,
                                    nondiff_argnums=(4,)))


def diagonal_scan(x, g, state, starts, *, family: mosaic.Family, interpret: bool = False):
    """`kernels.diagonal_scan.diagonal_scan` on the kernels: `x`, `g` `[B, T, D]` (`D` whole warpgroups of
    channels), `state` `[B, D]` the state entering, `starts` `[B, T]` packed documents' first tokens or None;
    `(output [B, T, D] in x's dtype, final state [B, D] fp32)`."""
    rows, length, channels = x.shape
    shape = walk(rows, length, channels, x.dtype, family, interpret)
    flags = None if starts is None else jnp.asarray(starts, jnp.int8)
    return _scan(x, g, jnp.asarray(state, jnp.float32), flags, shape)
