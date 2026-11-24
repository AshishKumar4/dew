"""The diagonal recurrence on Mosaic GPU (`kernels.diagonal`'s kernels): `h_t = a_t h_{t-1} + x_t` a
channel, `a_t = exp(g_t)`, each token's output its state, a packed document's first token `x_t` alone (HGRN's,
fla's `chunk_hgrn`; RG-LRU's and MinGRU's carry), on the channel walk (`walks`): a program a row, a span of
its tokens and a warpgroup's channels, each thread its channels walking the span's tokens in order.

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
them, its mantissa held in `[1, 2)` and its exponent apart (`walks.normalised`, each scaling exact), and its
state from zero a running sum held below 2^64 by exact scalings (`walks.summed_on`); XLA composes the spans'
maps in extended floats (`walks.extended`). Held as one float32 a span's decay underflowed where the states it
scaled were normal (`g = -2` over 64 tokens from a state of 2^100, ReviewRecurrenceBatch); a log decay summed
and exponentiated errs by `|sum g|` times float32's epsilon, past the product's. A document's first token
selects its own write, and a cut selects zero, never multiplying by a zero decay: an overflowed document's
infinite state times zero was NaN in the next.
"""

import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn import derivatives
from dew.nn.kernels import tuning
from dew.nn.kernels.templates import mosaic, scan
from dew.nn.kernels.templates.recurrence import conv, walks
from dew.nn.kernels.templates.recurrence.walks import (
    TURN,
    Program,
    Walk,
    added,
    chosen,
    cut,
    extended,
    launch,
    multiplied,
    normalised,
    per_span,
    per_token,
    refs_of,
    summed_on,
    valued,
    zero_decay,
)

MOST = 256
"""The most tokens a program walks: the backward keeps `MOST / TURN` states of its channels in shared
memory, 32 KB at 256 channels a program."""


def walk(rows: int, length: int, channels: int, dtype, family: mosaic.Family, interpret: bool) -> Walk:
    """The launches' facts for a call of `rows` rows of `length` tokens and `channels` channels in `dtype`:
    a program's channels the conv's (`conv.lanes`, whole lines a warp), its tokens `walks.walk`'s, at most
    `MOST`."""
    return walks.walk(rows, length, channels, conv.lanes(((channels, None),), dtype), MOST, family, interpret)


def _gate(program: Program, g_ref, token):
    """`(a_t, first)`: `exp(g_t)` (one outside the row, where `g` reads zero) and whether `token` is a
    document's first, laid over the channels (`Program.first`)."""
    a = jnp.exp(program.channels(g_ref, token))
    return a, program.first(token, a)


def _decayed(decay, a_t, first):
    """A span's decay, `(mantissa, exponent)`, past one more token's `a_t`: the mantissa times it (zero from a
    document's first token on, selected), raised back to `[1, 2)` (`walks.normalised`)."""
    mantissa, exponent = decay
    return normalised(cut(first, mantissa * a_t, jnp.zeros_like(mantissa)), exponent)


@tuning.lowered_once(("shape",))
def summaries(x, g, starts, *, shape: Walk):
    """Each span's map from its entering state to its last, `(mantissa, exponent, state, scale, cut)` `[B,
    spans, D]` fp32: the product of its tokens' decays (`_decayed`, the mantissa zero where a document starts
    in the span, which keeps none of its entering state), its last state from zero, a running sum
    (`walks.summed_on`, `state 2^(64 scale)`), and whether a document starts in it (one, else zero: a cut,
    apart from a decay zero by its gates)."""
    operands = (x, g, starts)

    def entry(*refs):
        (x_ref, g_ref, starts_ref), made = refs_of(refs, operands)
        program = Program(shape, starts_ref)

        def body(_, tokens, carried):
            decay, h, cutting = carried
            xs = [program.channels(x_ref, t) for t in tokens]
            gates = [_gate(program, g_ref, t) for t in tokens]
            for x_t, (a_t, first) in zip(xs, gates, strict=True):
                decay = _decayed(decay, a_t, first)
                h = summed_on(h, a_t, x_t, first)
                cutting = cut(first, cutting, cutting + 1.0)
            return decay, h, cutting

        zero = jnp.zeros((shape.lanes,), jnp.float32)
        decay, h, cutting = program.turns(body, ((zero + 1.0, zero), (zero, zero), zero))
        for ref, value in zip(made, (*decay, *h, jnp.minimum(cutting, 1.0)), strict=True):
            program.summarised(ref, value)

    out_type = tuple(per_span(x, shape) for _ in range(5))
    return launch(entry, shape, "diagonal_summaries", out_type, operands)


@tuning.lowered_once(("shape",))
def walked(x, g, starts, entering, *, shape: Walk):
    """Each token's state from its span's entering state (`entering` `[B, spans, D]` fp32), `[B, T, D]` in
    `x`'s dtype, and each span's last state, `[B, spans, D]` fp32."""
    operands = (x, g, starts, entering)

    def entry(*refs):
        (x_ref, g_ref, starts_ref, entering_ref), (out_ref, last_ref) = refs_of(refs, operands)
        program = Program(shape, starts_ref)

        def body(_, tokens, h):
            xs = [program.channels(x_ref, t) for t in tokens]
            gates = [_gate(program, g_ref, t) for t in tokens]
            for t, x_t, (a_t, first) in zip(tokens, xs, gates, strict=True):
                h = cut(first, a_t * h + x_t, x_t)
                program.stored(out_ref, t, h)
            return h

        program.summarised(last_ref, program.turns(body, program.summary(entering_ref)))

    return launch(entry, shape, "diagonal_walk", (per_token(x, x.dtype), per_span(x, shape)), operands)


@tuning.lowered_once(("shape",))
def reverse_summaries(g, starts, dy, *, shape: Walk):
    """Each span's map from its last state's cotangent to its entering state's, `(mantissa, exponent,
    cotangent, scale, cut)` `[B, spans, D]` fp32: the product of its tokens' decays (`_decayed`), the
    cotangent its own outputs' (`dy`) give its entering state, a running sum (`walks.summed_on`), and whether
    a document starts in it (`summaries`')."""
    operands = (g, starts, dy)

    def entry(*refs):
        (g_ref, starts_ref, dy_ref), made = refs_of(refs, operands)
        program = Program(shape, starts_ref)
        zero = jnp.zeros((shape.lanes,), jnp.float32)

        def body(_, tokens, carried):
            decay, c, cutting = carried
            gates = [_gate(program, g_ref, t) for t in tokens]
            dys = [program.channels(dy_ref, t) for t in tokens]
            for (a_t, first), dy_t in zip(reversed(gates), reversed(dys), strict=True):
                decay = _decayed(decay, a_t, first)
                # The cotangent the state before a document's first token takes of it: none.
                c = tuple(cut(first, value, zero) for value in summed_on(c, a_t, a_t * dy_t, None))
                cutting = cut(first, cutting, cutting + 1.0)
            return decay, c, cutting

        decay, c, cutting = program.turns(body, ((zero + 1.0, zero), (zero, zero), zero), reverse=True)
        for ref, value in zip(made, (*decay, *c, jnp.minimum(cutting, 1.0)), strict=True):
            program.summarised(ref, value)

    out_type = tuple(per_span(dy, shape) for _ in range(5))
    return launch(entry, shape, "diagonal_summaries_backward", out_type, operands)


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
        given, made = refs_of(refs, operands)
        x_ref, g_ref, starts_ref, dy_ref, entering_ref, later_ref = given
        dx_ref, dg_ref, leaving_ref, kept_ref = made
        program = Program(shape, starts_ref)
        layout = scan.blocked(shape.lanes)
        zero = jnp.zeros((shape.lanes,), jnp.float32)

        def forward(turn, tokens, h):
            plgpu.store(kept_ref.at[turn], h, optimized=False)
            xs = [program.channels(x_ref, t) for t in tokens]
            gates = [_gate(program, g_ref, t) for t in tokens]
            for x_t, (a_t, first) in zip(xs, gates, strict=True):
                h = cut(first, a_t * h + x_t, x_t)
            return h

        program.turns(forward, program.summary(entering_ref))

        def back(turn, tokens, c):
            xs = [program.channels(x_ref, t) for t in tokens]
            gates = [_gate(program, g_ref, t) for t in tokens]
            dys = [program.channels(dy_ref, t) for t in tokens]
            h, before = plgpu.load(kept_ref.at[turn], layout=layout, optimized=False), []
            for x_t, (a_t, first) in zip(xs, gates, strict=True):
                before.append(h)
                h = cut(first, a_t * h + x_t, x_t)
            walked = list(zip(tokens, gates, dys, before, strict=True))
            for t, (a_t, first), dy_t, h_t in reversed(walked):
                q = dy_t + c
                program.stored(dx_ref, t, q)
                program.stored(dg_ref, t, cut(first, (q * h_t) * a_t, zero))
                c = cut(first, a_t * q, zero)
            return c

        program.summarised(leaving_ref, program.turns(back, program.summary(later_ref), reverse=True))

    out_type = (per_token(x, x.dtype), per_token(g, g.dtype), per_span(x, shape))
    return launch(entry, shape, "diagonal_walk_backward", out_type, operands, scratch=(kept,))


def _composed(first, then):
    """Two spans' maps, `(decay, value, cut)` (the decay and value scaled values, `walks.extended`; the cut a
    flag) as one, `first`'s applied first: their decays multiplied, `first`'s value times `then`'s decay plus
    `then`'s, or `then`'s alone where it cuts (selected: an infinite value times a cut's zero decay was NaN),
    and either's cut (`associative_scan`'s operator either way). The cut a flag apart from the decay: read
    off a zero decay, a gate of -inf a document did not start at cut an infinite state the walk makes NaN
    (ReviewRecurrenceBatch)."""
    decay, value, cuts = first
    then_decay, then_value, then_cuts = then
    joined = added(multiplied(value, then_decay), then_value)
    return multiplied(decay, then_decay), chosen(then_cuts, then_value, joined), cuts | then_cuts


def _entering(summary: tuple, given, *, reverse: bool = False):
    """Each span's value as its walk enters it, `[B, spans, D]`: `given` `[B, D]` (the state entering the
    first span; `reverse`, the final state's cotangent entering the last) carried through the spans' maps
    (`summary`, `(mantissa, exponent, value, scale)` as `summaries` or `reverse_summaries` give them) by XLA's
    associative scan in scaled values (`walks.extended`, `dew.algebra.scaled`), so a decay below float32's
    range scaling a value past it, or a sum from zero past its largest, composes exactly where the value
    entering a span is a float32's (as float32 a span's decay flushed, ReviewRecurrenceBatch; halved, a least
    normal state), and a span a document starts in takes none of the value before it, selected."""
    mantissa, exponent, value, scale, cutting = summary
    maps = (extended(mantissa, exponent), extended(value, 64.0 * scale), cutting != 0)
    through, written, cuts = lax.associative_scan(_composed, maps, axis=1, reverse=reverse)
    entered = added(multiplied(extended(given[:, None], 0), through), written)
    after = valued(chosen(cuts, written, entered))
    # A value entering that no float holds (an infinite state) is the walk's through any span a document does
    # not cut: itself through a nonzero decay, NaN through a zero one (`0 inf`); the scaled algebra holds
    # finite values alone.
    kept = jnp.where(zero_decay(through), 0.0, 1.0)
    lasting = ~jnp.isfinite(given[:, None]) & ~cuts
    after = jnp.where(lasting, given[:, None] * kept + valued(written), after)
    if reverse:
        return jnp.concatenate([after[:, 1:], given[:, None]], axis=1)
    return jnp.concatenate([given[:, None], after[:, :-1]], axis=1)


def _forward(x, g, state, starts, shape: Walk):
    """`(outputs, final state, each span's entering state)`."""
    entering = state[:, None]
    if shape.spans > 1:
        entering = _entering(summaries(x, g, starts, shape=shape), state)
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
        later = _entering(reverse_summaries(g, starts, dy, shape=shape), dfinal, reverse=True)
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
