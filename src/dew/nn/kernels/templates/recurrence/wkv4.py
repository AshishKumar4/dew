"""RWKV4's WKV on Mosaic GPU (`kernels.rwkv4`'s kernels, row 1455), on the channel walk (`walks`): a program a
row, a span of its tokens and a warpgroup's channels, each thread its channel walking the span's tokens with
fla's carry, the numerator `alpha` and denominator `beta` scaled by `e^-eps`, `eps` their running maximum
(fla/ops/rwkv4/fused_recurrent.py; `rwkv4.recurrent_wkv4`, its arithmetic token by token):

    out_t = (e^{eps - tau} alpha + e^{u + k_t - tau} v_t) / (e^{eps - tau} beta + e^{u + k_t - tau}),
    tau = max(u + k_t, eps), then eps' = max(w + eps, k_t) and alpha', beta' rescaled to it.

A call over several spans summarises each span's carry from an empty past, XLA joins the summaries by the
portable scan's operator (`rwkv4.joined`) for each span's entering carry, and a launch walks each span from
its own, writing its outputs, its last carry and whether the carry entering it kept the running maximum
through it.

The backward is fla's carried form (fused_recurrent.py's backward): the cotangent of the carry `(g_alpha,
g_beta, g_eps)` walked back a token at a time at each carry's own maximum's frame, each step's key and decay
gradients formed net of the frame's shift (`_token_back`), the running maximum's tie the later key's (`_won`).
Spans as the forward: a launch walks each span back from none for the cotangent its own outputs give its
entering carry; the linear part of each span's map back follows from the forward's facts (`_maps_back`: its
decay `e^(n w + eps_in - eps_out)`, and `g_eps`'s share by whether the entering maximum held through it); XLA
composes the maps from the final carry's cotangent in scaled values; and a launch walks each span back from
its own, recomputing the forward from its entering carry, a walk forward keeping each turn's carry in shared
memory.

A document's first token starts from an empty past, selected; outside the row a token leaves every carry as
it is. Every value is fp32, `w` and `u` gradients summed compensated a program (`precision.compensated_add`)
and across them by XLA.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.algebra import scaled
from dew.nn import derivatives
from dew.nn.kernels import tuning
from dew.nn.kernels.templates import mosaic, scan
from dew.nn.kernels.templates.recurrence import walks
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
    per_span,
    per_token,
    refs_of,
    summed_on,
    valued,
)
from dew.nn.precision import compensated_add, compensated_sum

MOST = 128
"""The most tokens a program walks: the backward keeps a carry of three values a turn in shared memory,
`MOST / TURN` of them, 24 KB at a warpgroup's channels."""


def walk(rows: int, length: int, channels: int, family: mosaic.Family, interpret: bool) -> Walk:
    """The launches' facts for a call of `rows` rows of `length` tokens and `channels` channels: a channel a
    thread (the backward's walk holds a turn's carries and loads, three values a token), its tokens
    `walks.walk`'s, at most `MOST`."""
    return walks.walk(rows, length, channels, scan.THREADS, MOST, family, interpret)


def _empty(like):
    """An empty past's carry: no numerator or denominator, the running maximum -inf."""
    zero = jnp.zeros_like(like)
    return zero, zero, zero - jnp.inf


def _held(inside, new, old):
    """`new` where the token is inside the row, else `old`, each value of their trees selected."""
    return jax.tree.map(lambda one, other: jnp.where(inside, one, other), new, old)


def _entered(program: Program, token, carry: tuple, like) -> tuple:
    """The carry a token reads: an empty past's at a document's first token, selected."""
    first = program.first(token, like)
    return tuple(cut(first, value, empty) for value, empty in zip(carry, _empty(like), strict=True))


def _read(carry: tuple, k, v, u):
    """A token's output from the carry it reads, and the terms its backward reuses: `(out, past, present,
    denominator)`, `past = e^{eps - tau}` and `present = e^{u + k - tau}` at `tau = max(u + k, eps)`."""
    alpha, beta, eps = carry
    own = u + k
    tau = jnp.maximum(own, eps)
    past, present = jnp.exp(eps - tau), jnp.exp(own - tau)
    denominator = past * beta + present
    return (past * alpha + present * v) / denominator, past, present, denominator


def _weights(eps, k, w) -> tuple:
    """A token's step of the running maximum: `(kept, added, top)`, the past's weight `e^(w + eps - top)`
    and the token's `e^(k - top)` at the new maximum `top = max(w + eps, k)`, a tie the key's (`_won`)."""
    aged = w + eps
    top = jnp.where(_won(k, aged), k, aged)
    return jnp.exp(aged - top), jnp.exp(k - top), top


def _won(k, aged):
    """Whether a token's key sets the running maximum past it: at least the aged maximum, a tie the key's,
    as fla's backward takes it (fused_recurrent.py: the maximum's cotangent the decay's where `w + eps > k`):
    the later of tied maxima takes it, alike token by token and in any grouping of spans (`rwkv4.joined`),
    where a tie split in halves token by token and by a scan's tree split three tied keys apart."""
    return k >= aged


def _aged(carry: tuple, k, v, w) -> tuple:
    """The carry past a token: its past decayed by `e^w` and the token's `e^k v` and `e^k` added, rescaled
    to the larger maximum."""
    alpha, beta, eps = carry
    kept, added, top = _weights(eps, k, w)
    return kept * alpha + added * v, kept * beta + added, top


def _stepped(program: Program, token, carry: tuple, k, v, w) -> tuple:
    """`(carry read, carry after)` at `token`: its document's empty past at a first token, the carry aged by
    it inside the row and left as it is outside."""
    read = _entered(program, token, carry, k)
    return read, _held(program.inside(token, k), _aged(read, k, v, w), carry)


@tuning.lowered_once(("shape",))
def summaries(k, v, w, starts, *, shape: Walk):
    """Each span's carry from an empty past, `(alpha, alpha scale, beta, eps)` `[B, spans, D]` fp32, its
    numerator a running sum (`walks.summed_on`)."""
    operands = (k, v, w, starts)

    def entry(*refs):
        (k_ref, v_ref, w_ref, starts_ref), made = refs_of(refs, operands)
        program = Program(shape, starts_ref)
        decay = program.parameter(w_ref)

        def body(_, tokens, carry):
            ks = [program.channels(k_ref, t) for t in tokens]
            vs = [program.channels(v_ref, t) for t in tokens]
            for t, k_t, v_t in zip(tokens, ks, vs, strict=True):
                (numerator, scale), beta, eps = carry
                first = program.first(t, k_t)
                zero = jnp.zeros_like(k_t)
                numerator, scale, beta = (cut(first, value, zero) for value in (numerator, scale, beta))
                kept, added, top = _weights(cut(first, eps, zero - jnp.inf), k_t, decay)
                stepped = (summed_on((numerator, scale), kept * numerator, added * v_t, None),
                           kept * beta + added, top)
                carry = _held(program.inside(t, k_t), stepped, carry)
            return carry

        zero = jnp.zeros_like(decay)
        (numerator, scale), beta, eps = program.turns(body, ((zero, zero), zero, zero - jnp.inf))
        for ref, value in zip(made, (numerator, scale, beta, eps), strict=True):
            program.summarised(ref, value)

    return launch(entry, shape, "wkv4_summaries", tuple(per_span(k, shape) for _ in range(4)), operands)


@tuning.lowered_once(("shape",))
def walked(k, v, w, u, starts, alpha, beta, eps, *, shape: Walk):
    """Each token's output from its span's entering carry (`alpha`, `beta`, `eps` `[B, spans, D]` fp32),
    `[B, T, D]` in `k`'s dtype, each span's last carry, and whether the entering carry's maximum held
    through the span (one, else zero: a key set it, `_won`, or a document started)."""
    operands = (k, v, w, u, starts, alpha, beta, eps)

    def entry(*refs):
        (k_ref, v_ref, w_ref, u_ref, starts_ref, *entering), made = refs_of(refs, operands)
        out_ref, *last_refs, survived_ref = made
        program = Program(shape, starts_ref)
        decay, bonus = program.parameter(w_ref), program.parameter(u_ref)

        def body(_, tokens, carried):
            carry, survived = carried
            ks = [program.channels(k_ref, t) for t in tokens]
            vs = [program.channels(v_ref, t) for t in tokens]
            for t, k_t, v_t in zip(tokens, ks, vs, strict=True):
                read, carry = _stepped(program, t, carry, k_t, v_t, decay)
                program.stored(out_ref, t, _read(read, k_t, v_t, bonus)[0])
                set_by_key = program.inside(t, k_t) & _won(k_t, decay + read[2])
                survived = jnp.where(set_by_key, jnp.zeros_like(survived), survived)
            return carry, survived

        carry = tuple(program.summary(ref) for ref in entering)
        carry, survived = program.turns(body, (carry, jnp.ones_like(carry[0])))
        for ref, value in zip(last_refs, carry, strict=True):
            program.summarised(ref, value)
        program.summarised(survived_ref, survived)

    out_type = (per_token(k, k.dtype), *(per_span(k, shape) for _ in range(4)))
    return launch(entry, shape, "wkv4_walk", out_type, operands)


def _back(program: Program, refs: tuple, shape: Walk, kept_ref, carried, emit):
    """A span's walk back: the forward recomputed from its entering carry, each turn's carry kept in shared
    memory (`kept_ref`), then each turn's tokens from its last, each token handed to `emit(t, k, v, dy, read,
    terms, decay, carried)` (`read` the carry it reads, `terms` `_read`'s), which returns the reverse carry
    before it."""
    k_ref, v_ref, w_ref, u_ref, dy_ref, entering = refs
    decay, bonus = program.parameter(w_ref), program.parameter(u_ref)
    layout = scan.blocked(shape.lanes)

    def forward(turn, tokens, carry):
        for slot, value in enumerate(carry):
            plgpu.store(kept_ref.at[turn, slot], value, optimized=False)
        ks = [program.channels(k_ref, t) for t in tokens]
        vs = [program.channels(v_ref, t) for t in tokens]
        for t, k_t, v_t in zip(tokens, ks, vs, strict=True):
            carry = _stepped(program, t, carry, k_t, v_t, decay)[1]
        return carry

    program.turns(forward, tuple(program.summary(ref) for ref in entering))

    def back(turn, tokens, carried):
        ks = [program.channels(k_ref, t) for t in tokens]
        vs = [program.channels(v_ref, t) for t in tokens]
        dys = [program.channels(dy_ref, t) for t in tokens]
        carry = tuple(plgpu.load(kept_ref.at[turn, slot], layout=layout, optimized=False)
                      for slot in range(3))
        reads = []
        for t, k_t, v_t in zip(tokens, ks, vs, strict=True):
            read, carry = _stepped(program, t, carry, k_t, v_t, decay)
            reads.append(read)
        walked = list(zip(tokens, ks, vs, dys, reads, strict=True))
        for t, k_t, v_t, dy_t, read in reversed(walked):
            carried = emit(t, k_t, v_t, dy_t, read, _read(read, k_t, v_t, bonus), decay, carried)
        return carried

    return program.turns(back, carried, reverse=True)


def _token_back(program: Program, t, k_t, v_t, dy_t, read, terms, decay, after):
    """One token walked back in fla's carried form (fused_recurrent.py's backward), contracted: `(cotangent of
    the carry before it, (dk, dv, dw, du) of it)` from `after`, the cotangent `(g_alpha, g_beta, g_eps)` of
    the carry past it, each at its own maximum's frame. A step's key and decay gradients are formed net of the
    frame's shift: the maximum's cotangent less the past's share where the key sets the maximum, less the
    token's own where the past does (`_won`), each share weighted by its step's factor first (`kept`, `added`,
    at most one): taken apart, a reset token's key gradient was the difference of two terms each past
    float32's largest (2^200 - 2^200, ReviewRecurrenceBatch); the output's `v - out` and `alpha - v beta`
    products are each weighted first. A document's first token leaves the carry before it none; outside the
    row, the carry as it is and no gradient."""
    alpha, beta, eps = read
    out, past, present, denominator = terms
    g_alpha, g_beta, g_eps = after
    zero = jnp.zeros_like(g_eps)
    c = dy_t / denominator
    by_present, by_past = c * present, c * past
    du = by_present * v_t - by_present * out
    weighted = by_past * (present / denominator)
    eps_read = weighted * alpha - (weighted * v_t) * beta
    aged = decay + eps
    won = _won(k_t, aged)
    top = jnp.where(won, k_t, aged)
    kept, added = jnp.exp(aged - top), jnp.exp(k_t - top)
    past_share = (kept * g_alpha) * alpha + (kept * g_beta) * beta
    own_share = (added * g_alpha) * v_t + added * g_beta
    dk = du + jnp.where(won, g_eps - past_share, own_share)
    dw = jnp.where(won, past_share, g_eps - own_share)
    dv = by_present + added * g_alpha
    before = (kept * g_alpha + by_past, kept * g_beta - by_past * out, dw + eps_read)
    first = program.first(t, k_t)
    before = tuple(cut(first, value, zero) for value in before)
    inside = program.inside(t, k_t)
    return _held(inside, before, after), tuple(jnp.where(inside, x, zero) for x in (dk, dv, dw, du))


def _kept(shape: Walk):
    """A backward's shared memory: each turn's carry, three values of its channels."""
    return plgpu.SMEM((shape.span // TURN, 3, shape.lanes), jnp.float32)


@tuning.lowered_once(("shape",))
def reverse_summaries(k, v, w, u, starts, alpha, beta, eps, dy, *, shape: Walk):
    """The cotangent each span's own outputs (`dy`) give the carry entering it, `(g_alpha, g_beta, g_eps)`
    `[B, spans, D]` fp32 (`_token_back` from none), from each span's entering carry (`alpha`, `beta`,
    `eps`): the constant of the span's map back, whose linear part is the forward's (`_maps_back`)."""
    operands = (k, v, w, u, starts, alpha, beta, eps, dy)

    def entry(*refs):
        given, (*made, kept_ref) = refs_of(refs, operands)
        k_ref, v_ref, w_ref, u_ref, starts_ref, *entering, dy_ref = given
        program = Program(shape, starts_ref)
        zero = jnp.zeros((shape.lanes,), jnp.float32)

        def emit(t, k_t, v_t, dy_t, read, terms, decay, carried):
            return _token_back(program, t, k_t, v_t, dy_t, read, terms, decay, carried)[0]

        carried = _back(program, (k_ref, v_ref, w_ref, u_ref, dy_ref, entering), shape, kept_ref,
                        (zero, zero, zero), emit)
        for ref, value in zip(made, carried, strict=True):
            program.summarised(ref, value)

    out_type = tuple(per_span(k, shape) for _ in range(3))
    return launch(entry, shape, "wkv4_summaries_backward", out_type, operands, scratch=(_kept(shape),))


@tuning.lowered_once(("shape",))
def walked_back(k, v, w, u, starts, alpha, beta, eps, dy, later_alpha, later_beta, later_eps, *,
                shape: Walk):
    """The gradients of `k` and `v` (`[B, T, D]` in their dtype), each program's of `w` and `u` (`[B, spans,
    D]` fp32, XLA sums them), and the cotangent of each span's entering carry (`[B, spans, D]` fp32 each),
    from the outputs' cotangent `dy`, each span's entering carry and the cotangent of its last carry from the
    later tokens and the final carry (`later_*`)."""
    operands = (k, v, w, u, starts, alpha, beta, eps, dy, later_alpha, later_beta, later_eps)

    def entry(*refs):
        given, made = refs_of(refs, operands)
        k_ref, v_ref, w_ref, u_ref, starts_ref, *carries, dy_ref = given[:9]
        dk_ref, dv_ref, dw_ref, du_ref, *leaving_refs, kept_ref = made
        program = Program(shape, starts_ref)
        zero = jnp.zeros((shape.lanes,), jnp.float32)

        def emit(t, k_t, v_t, dy_t, read, terms, decay, carried):
            after, (dw_sum, du_sum) = carried
            before, (dk, dv, dw, du) = _token_back(program, t, k_t, v_t, dy_t, read, terms, decay, after)
            program.stored(dk_ref, t, dk)
            program.stored(dv_ref, t, dv)
            return before, (compensated_add(dw_sum, (dw, zero)), compensated_add(du_sum, (du, zero)))

        later = tuple(program.summary(ref) for ref in given[9:])
        leaving, sums = _back(program, (k_ref, v_ref, w_ref, u_ref, dy_ref, carries), shape, kept_ref,
                              (later, ((zero, zero), (zero, zero))), emit)
        for ref, (high, low) in zip((dw_ref, du_ref), sums, strict=True):
            program.summarised(ref, high + low)
        for ref, value in zip(leaving_refs, leaving, strict=True):
            program.summarised(ref, value)

    out_type = (per_token(k, k.dtype), per_token(v, v.dtype), *(per_span(k, shape) for _ in range(5)))
    return launch(entry, shape, "wkv4_walk_backward", out_type, operands, scratch=(_kept(shape),))


def _spans(shape: Walk, starts, like):
    """Each span's tokens in its row and whether a document starts in it, `[B, spans, D]` each as `like`
    (`rwkv4.SpanCarry`'s)."""
    tokens = jnp.broadcast_to(jnp.asarray(shape.tokens(), like.dtype)[None, :, None], like.shape)
    if starts is None:
        return tokens, jnp.zeros(like.shape, bool)
    padded = jnp.pad(jnp.asarray(starts, bool), ((0, 0), (0, shape.spans * shape.span - shape.length)))
    fresh = padded.reshape(shape.rows, shape.spans, shape.span).any(axis=-1)
    return tokens, jnp.broadcast_to(fresh[..., None], like.shape)


class _Carry(NamedTuple):
    """`rwkv4.SpanCarry` as the template's join holds it: its numerator a scaled value (`walks.extended`,
    `dew.algebra.scaled.Scaled`), `rwkv4.joined` reading each field by its name alike."""

    numerator: jax.Array | scaled.Scaled
    denominator: jax.Array | scaled.Scaled
    top: jax.Array
    tokens: jax.Array
    fresh: jax.Array


def _weighed(kept, earlier, added_, later):
    """Two spans' scaled values (`walks.extended`) at their larger maximum, `rwkv4.joined`'s weighing."""
    return added(multiplied(earlier, extended(kept, 0)), multiplied(later, extended(added_, 0)))


def _joined(carries: tuple, given: tuple, decay, shape: Walk, starts) -> tuple:
    """Each span's carry as its walk enters it, `(alpha, beta, eps)` `[B, spans, D]`: `given` (the call's
    entering carry, `[B, D]` each) joined with the spans' (`carries`, from an empty past, the numerator a
    running sum's value and scale) by the portable scan's operator (`rwkv4.joined`) over XLA's associative
    scan, the numerators scaled (`walks.extended`): a span's from none is its last less its entering one's
    decayed share, past float32's largest where every value is in it."""
    from dew.nn.kernels import rwkv4

    numerator, scale, beta, top = carries
    tokens, fresh = _spans(shape, starts, top)
    spans = _Carry(extended(numerator, 64.0 * scale), beta, top, tokens, fresh)
    edge = _Carry(extended(given[0][:, None], 0), given[1][:, None], given[2][:, None],
                  jnp.zeros_like(tokens[:, :1]), jnp.zeros_like(fresh[:, :1]))
    ordered = jax.tree.map(lambda *parts: jnp.concatenate(parts, axis=1), edge, spans)
    joined = jax.lax.associative_scan(rwkv4.joined(decay, numerators=_weighed), ordered, axis=1)
    entered = jax.tree.map(lambda value: value[:, :-1], joined)
    return valued(entered.numerator), entered.denominator, entered.top


class _Back(NamedTuple):
    """A span's map back from the cotangent of its last carry to its entering carry's, as scaled values:
    `(g_alpha, g_beta)` scaled by `kept`, `g_eps` by `on_alpha`, `on_beta` and (where `held`, the entering
    maximum held through the span) itself, plus the cotangent its own outputs give (`constant`, `(alpha,
    beta, eps)`)."""

    kept: scaled.Scaled
    on_alpha: scaled.Scaled
    on_beta: scaled.Scaled
    held: jax.Array
    constant: tuple


def _then(first: _Back, then: _Back) -> _Back:
    """`first`'s map back, then `then`'s (`associative_scan`'s operator, reversed: `first` the later span)."""
    alpha, beta, eps = first.constant
    passed = then.held != 0
    return _Back(
        multiplied(then.kept, first.kept),
        added(multiplied(then.on_alpha, first.kept), chosen(passed, first.on_alpha, _nothing(first.kept))),
        added(multiplied(then.on_beta, first.kept), chosen(passed, first.on_beta, _nothing(first.kept))),
        then.held * first.held,
        (added(multiplied(then.kept, alpha), then.constant[0]),
         added(multiplied(then.kept, beta), then.constant[1]),
         added(added(multiplied(then.on_alpha, alpha), multiplied(then.on_beta, beta)),
               added(chosen(passed, eps, _nothing(eps)), then.constant[2]))))


def _nothing(like: scaled.Scaled) -> scaled.Scaled:
    """A scaled zero of `like`'s shape."""
    return extended(jnp.zeros_like(like.significand), 0)


def _exponential(x) -> scaled.Scaled:
    """`e^x` as a scaled value (`dew.algebra.scaled`): its power of two apart from its significand, so a
    span's decay below float32's range keeps its value; zero at `x = -inf`."""
    exponent = jnp.floor(x * _LOG2E)
    significand = jnp.exp2(x * _LOG2E - exponent)
    finite = jnp.isfinite(x)
    return extended(jnp.where(finite, significand, 0.0), jnp.where(finite, exponent, 0.0))


_LOG2E = 1.4426950408889634
"""`log2(e)`."""


def _maps_back(forward: tuple, constants: tuple, decay, shape: Walk, starts) -> _Back:
    """Each span's map back (`_Back`), `[B, spans, D]`, from the forward's facts (`forward`: each span's
    entering carry, its last carry, whether its entering maximum held through it, and its carry from an
    empty past, its numerator a running sum) and the cotangents its own outputs give its entering carry
    (`constants`, `reverse_summaries`'). A span's decay is `e^(n w + eps_in - eps_out)` over its `n` tokens
    (zero where a document starts in it); its last carry's numerator is its entering one's decayed share plus
    its own from none, so the map takes `g_eps` to `-own alpha` where the entering maximum held through it and
    to `kept alpha_in` where a key set it."""
    (alpha_in, beta_in, eps_in), (_, _, eps_out), held, (numerator, scale, beta_own, eps_own) = forward
    tokens, fresh = _spans(shape, starts, eps_in)
    kept = chosen(fresh, _nothing(extended(eps_in, 0)), _exponential(tokens * decay + eps_in - eps_out))
    own = _exponential(eps_own - eps_out)
    keeps = held != 0
    on_alpha = chosen(keeps, scaled.neg(multiplied(extended(numerator, 64.0 * scale), own)),
                      multiplied(kept, extended(alpha_in, 0)))
    on_beta = chosen(keeps, scaled.neg(multiplied(extended(beta_own, 0), own)),
                     multiplied(kept, extended(beta_in, 0)))
    return _Back(kept, on_alpha, on_beta, held, tuple(extended(value, 0) for value in constants))


def _later(maps: _Back, final: tuple) -> tuple:
    """The cotangent of each span's last carry, `(alpha, beta, eps)` `[B, spans, D]`: `final` (the final
    carry's, `[B, D]` each) carried back through the later spans' maps (`maps`) by XLA's associative scan."""
    through = jax.lax.associative_scan(_then, maps, axis=1, reverse=True)
    g_alpha, g_beta, g_eps = (extended(value[:, None], 0) for value in final)
    entered = (added(multiplied(through.kept, g_alpha), through.constant[0]),
               added(multiplied(through.kept, g_beta), through.constant[1]),
               added(added(added(multiplied(through.on_alpha, g_alpha), multiplied(through.on_beta, g_beta)),
                           chosen(through.held != 0, g_eps, _nothing(g_eps))), through.constant[2]))
    return tuple(jnp.concatenate([valued(value)[:, 1:], given[:, None]], axis=1)
                 for value, given in zip(entered, final, strict=True))


def _forward(k, v, w, u, state, starts, shape: Walk):
    """`(outputs, final carry [B, 3, D], the forward's facts its backward reads)`."""
    given = (state[:, 0], state[:, 1], state[:, 2])
    alpha, beta, eps = state[:, 0, None], state[:, 1, None], state[:, 2, None]
    own = None
    if shape.spans > 1:
        own = summaries(k, v, w, starts, shape=shape)
        alpha, beta, eps = _joined(own, given, w, shape, starts)
    out, *last, held = walked(k, v, w, u, starts, alpha, beta, eps, shape=shape)
    final = jnp.stack([value[:, -1] for value in last], axis=1)
    return out, final, ((alpha, beta, eps), tuple(last), held, own)


def _primal(k, v, w, u, state, starts, shape: Walk):
    return _forward(k, v, w, u, state, starts, shape)[:2]


def _forward_rule(k, v, w, u, state, starts, shape: Walk):
    out, final, facts = _forward(k, v, w, u, state, starts, shape)
    return (out, final), (k, v, w, u, starts, facts)


def _backward_rule(shape: Walk, residuals, cotangents):
    k, v, w, u, starts, facts = residuals
    entering = facts[0]
    dy, dfinal = cotangents
    final = tuple(value.astype(jnp.float32) for value in (dfinal[:, 0], dfinal[:, 1], dfinal[:, 2]))
    later = tuple(value[:, None] for value in final)
    if shape.spans > 1:
        constants = reverse_summaries(k, v, w, u, starts, *entering, dy, shape=shape)
        later = _later(_maps_back(facts, constants, w, shape, starts), final)
    dk, dv, dw, du, *leaving = walked_back(k, v, w, u, starts, *entering, dy, *later, shape=shape)
    dstate = jnp.stack([value[:, 0] for value in leaving], axis=1)
    return dk, dv, compensated_sum(dw, (0, 1)), compensated_sum(du, (0, 1)), dstate, None


_scan = derivatives.contracted("test_the_wkv4_scan_holds_its_orders")(
    derivatives.declared_derivative(derivatives.KernelGlue(), _primal, _forward_rule, _backward_rule,
                                    nondiff_argnums=(6,)))


def wkv4(w, u, k, v, state, starts, *, family: mosaic.Family, interpret: bool = False):
    """`kernels.rwkv4.wkv4` on the kernels: the decay `w` and bonus `u` `[D]` fp32, keys and values `[B, T,
    D]` in one dtype (`D` whole warpgroups of channels), the carry entering `state` `[B, 3, D]` fp32,
    `starts` `[B, T]` packed documents' first tokens or None; `(out [B, T, D] in the keys' dtype, final carry
    [B, 3, D] fp32)`."""
    rows, length, channels = k.shape
    shape = walk(rows, length, channels, family, interpret)
    flags = None if starts is None else jnp.asarray(starts, jnp.int8)
    return _scan(k, v, w, u, state, flags, shape)
