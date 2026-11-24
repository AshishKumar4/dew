"""RWKV4's WKV on Mosaic GPU (`kernels.rwkv4`'s kernels, row 1455), on the channel walk (`walks`): a program a
row, a span of its tokens and a warpgroup's channels, each thread its channel walking the span's tokens with
fla's carry, the numerator `alpha` and denominator `beta` scaled by `e^-eps`, `eps` their running maximum
(fla/ops/rwkv4/fused_recurrent.py; `rwkv4.recurrent_wkv4`, its arithmetic token by token):

    out_t = (e^{eps - tau} alpha + e^{u + k_t - tau} v_t) / (e^{eps - tau} beta + e^{u + k_t - tau}),
    tau = max(u + k_t, eps), then eps' = max(w + eps, k_t) and alpha', beta' rescaled to it.

A call over several spans summarises each span's carry from an empty past, XLA joins the summaries by the
portable scan's operator (`rwkv4.joined`) for each span's entering carry, and a launch walks each span from
its own, writing its outputs and its last carry.

The backward is fla's carried form (fused_recurrent.py's backward): the cotangent of the carry `(g_alpha,
g_beta, g_eps)` walked back a token at a time at each carry's own maximum's frame, each step's key and decay
gradients formed net of the frame's shift (`_token_back`), the running maximum's tie the later key's (`_won`).
A program a row and a block of channels walks the row's spans back in turn from the final carry's cotangent
(fla's backward's parallelism), recomputing each span's forward from its entering carry, a walk forward
keeping each turn's carry in shared memory.

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
from dew.nn.kernels.templates import mosaic, outputs, scan
from dew.nn.kernels.templates.recurrence import walks
from dew.nn.kernels.templates.recurrence.walks import (
    TURN,
    Program,
    Walk,
    added,
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
                stepped = (summed_on((numerator, scale), kept, added * v_t, None),
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
    `[B, T, D]` in `k`'s dtype, and each span's last carry."""
    operands = (k, v, w, u, starts, alpha, beta, eps)

    def entry(*refs):
        (k_ref, v_ref, w_ref, u_ref, starts_ref, *entering), (out_ref, *last_refs) = refs_of(refs, operands)
        program = Program(shape, starts_ref)
        decay, bonus = program.parameter(w_ref), program.parameter(u_ref)

        def body(_, tokens, carry):
            ks = [program.channels(k_ref, t) for t in tokens]
            vs = [program.channels(v_ref, t) for t in tokens]
            for t, k_t, v_t in zip(tokens, ks, vs, strict=True):
                read, carry = _stepped(program, t, carry, k_t, v_t, decay)
                program.stored(out_ref, t, _read(read, k_t, v_t, bonus)[0])
            return carry

        carry = program.turns(body, tuple(program.summary(ref) for ref in entering))
        for ref, value in zip(last_refs, carry, strict=True):
            program.summarised(ref, value)

    out_type = (per_token(k, k.dtype), *(per_span(k, shape) for _ in range(3)))
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
    """One token walked back in fla's carried form (fused_recurrent.py's backward): `(cotangent of the carry
    before it, (dk, dv, dw, du) of it)` from `after`, the cotangent `(g_alpha, g_beta, g_eps)` of the carry
    past it, each at its own maximum's frame, each term in the reverse mode's own order
    (`rwkv4.recurrent_wkv4`'s), whose rounding check 1 holds it to, and contracted where that order parts past
    float32's range (`_finite_or`): a step's key and decay gradients net of the frame's shift, the maximum's
    cotangent less the past's share where the key sets the maximum (`_won`) and less the token's own where the
    past does, each share weighted by its step's factor first (`_contracted`), and the output's `v - out` and
    `alpha - v beta` weighted first. In the reverse mode's order a reset token's key gradient was the
    difference of two terms each past float32's largest (2^200 - 2^200, ReviewRecurrenceBatch); contracted
    throughout, a small call's decay and bonus gradients were 1.7 times the recurrence's error. A document's
    first token leaves the carry before it none; outside the row, the carry as it is and no gradient."""
    alpha, beta, eps = read
    out, past, present, denominator = terms
    g_alpha, g_beta, g_eps = after
    zero = jnp.zeros_like(g_eps)
    c = dy_t / denominator
    by_present, by_past = c * present, c * past
    du = _finite_or(by_present * (v_t - out), lambda: by_present * v_t - by_present * out)
    weighted = by_past * (present / denominator)
    eps_read = _finite_or(weighted * (alpha - v_t * beta), lambda: weighted * alpha - (weighted * v_t) * beta)
    aged = decay + eps
    won = _won(k_t, aged)
    top = jnp.where(won, k_t, aged)
    kept, added = jnp.exp(aged - top), jnp.exp(k_t - top)
    # Each share in the reverse mode's grouping (the recurrence's, `rwkv4.recurrent_wkv4`'s), weighted first
    # where that parts past float32's range; the maximum's cotangent net of them, never added and taken back.
    # A share whose step's factor is zero (the past's at a document's first token) is none, selected: an
    # infinite cotangent times it was NaN (ReviewRecurrenceBatch).
    past_share = _finite_or((g_alpha * alpha + g_beta * beta) * kept,
                            lambda: _contracted(kept, g_alpha, alpha, g_beta, beta))
    own_share = _finite_or((g_alpha * v_t + g_beta) * added,
                           lambda: _contracted(added, g_alpha, v_t, g_beta, 1.0))
    past_share = jnp.where(kept == 0.0, zero, past_share)
    own_share = jnp.where(added == 0.0, zero, own_share)
    dk = du + jnp.where(won, g_eps - past_share, own_share)
    dw = jnp.where(won, past_share, g_eps - own_share)
    dv = by_present + added * g_alpha
    before = (kept * g_alpha + by_past, kept * g_beta - by_past * out, dw + eps_read)
    first = program.first(t, k_t)
    before = tuple(cut(first, value, zero) for value in before)
    inside = program.inside(t, k_t)
    return _held(inside, before, after), tuple(jnp.where(inside, x, zero) for x in (dk, dv, dw, du))


def _finite_or(value, contracted):
    """`value` where finite, else `contracted()`, the same quantity formed so no term passes float32's range
    where it does not: `value` the reverse mode's own order, whose rounding check 1 holds the kernels to."""
    return jnp.where(jnp.isfinite(value), value, contracted())


def _contracted(factor, g_alpha, alpha, g_beta, beta):
    """`factor (g_alpha alpha + g_beta beta)`, each cotangent weighted by `factor` (at most one) first."""
    return (factor * g_alpha) * alpha + (factor * g_beta) * beta


def _kept(shape: Walk):
    """A backward's shared memory: each turn's carry, three values of its channels."""
    return plgpu.SMEM((shape.span // TURN, 3, shape.lanes), jnp.float32)


@tuning.lowered_once(("shape",))
def walked_back(k, v, w, u, starts, alpha, beta, eps, dy, final_alpha, final_beta, final_eps, *, shape: Walk):
    """The gradients of `k` and `v` (`[B, T, D]` in their dtype), each span's of `w` and `u` (`[B, spans, D]`
    fp32, XLA sums them) and the cotangent of the carry entering the call (`[B, 1, D]` fp32 each), from the
    outputs' cotangent `dy`, each span's entering carry (`alpha`, `beta`, `eps` `[B, spans, D]`) and the final
    carry's cotangent (`final_*` `[B, 1, D]`): a program a row and a block of channels walks the row's spans
    back in turn from its last (fla's fused recurrent backward's parallelism), each span's forward recomputed
    from its entering carry. Its spans' maps composed apart from their own, each rounding its own way, a
    walk's cotangent was lost where the forward's frame rounded a decay away and a span's own constant passed
    float32's range where the walk's did not (ReviewRecurrenceBatch)."""
    operands = (k, v, w, u, starts, alpha, beta, eps, dy, final_alpha, final_beta, final_eps)

    def entry(*refs):
        given, made = refs_of(refs, operands)
        k_ref, v_ref, w_ref, u_ref, starts_ref, *carries, dy_ref = given[:9]
        dk_ref, dv_ref, dw_ref, du_ref, *leaving_refs, kept_ref = made
        first = Program(shape, starts_ref, block=0)
        zero = jnp.zeros((shape.lanes,), jnp.float32)

        def span(i, after):
            program = Program(shape, starts_ref, block=shape.spans - 1 - i)

            def emit(t, k_t, v_t, dy_t, read, terms, decay, carried):
                after, (dw_sum, du_sum) = carried
                before, grads = _token_back(program, t, k_t, v_t, dy_t, read, terms, decay, after)
                dk, dv, dw, du = grads
                program.stored(dk_ref, t, dk)
                program.stored(dv_ref, t, dv)
                return before, (compensated_add(dw_sum, (dw, zero)), compensated_add(du_sum, (du, zero)))

            given_refs = (k_ref, v_ref, w_ref, u_ref, dy_ref, carries)
            before, sums = _back(program, given_refs, shape, kept_ref, (after, ((zero, zero), (zero, zero))),
                                 emit)
            # Each span's own partial sums, XLA's to sum: a row's summed through would pass float32's range
            # where its spans' did not (2^127 + 2^127 before -2^127, ReviewRecurrenceBatch).
            for ref, (high, low) in zip((dw_ref, du_ref), sums, strict=True):
                program.summarised(ref, high + low)
            return before

        final = tuple(first.summary(ref) for ref in given[9:])
        leaving = jax.lax.fori_loop(0, shape.spans, span, final)
        for ref, value in zip(leaving_refs, leaving, strict=True):
            first.summarised(ref, value)

    once = outputs.output(jnp.float32, outputs.axis(k, 0), 1, outputs.axis(k, 2))
    out_type = (per_token(k, k.dtype), per_token(v, v.dtype), per_span(k, shape), per_span(k, shape),
                *(once for _ in range(3)))
    return launch(entry, shape, "wkv4_walk_backward", out_type, operands, scratch=(_kept(shape),), whole=True)


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


def _forward(k, v, w, u, state, starts, shape: Walk):
    """`(outputs, final carry [B, 3, D], each span's entering carry)`."""
    given = (state[:, 0], state[:, 1], state[:, 2])
    alpha, beta, eps = state[:, 0, None], state[:, 1, None], state[:, 2, None]
    if shape.spans > 1:
        alpha, beta, eps = _joined(summaries(k, v, w, starts, shape=shape), given, w, shape, starts)
    out, *last = walked(k, v, w, u, starts, alpha, beta, eps, shape=shape)
    final = jnp.stack([value[:, -1] for value in last], axis=1)
    return out, final, (alpha, beta, eps)


def _primal(k, v, w, u, state, starts, shape: Walk):
    return _forward(k, v, w, u, state, starts, shape)[:2]


def _forward_rule(k, v, w, u, state, starts, shape: Walk):
    out, final, entering = _forward(k, v, w, u, state, starts, shape)
    return (out, final), (k, v, w, u, starts, entering)


def _backward_rule(shape: Walk, residuals, cotangents):
    k, v, w, u, starts, entering = residuals
    dy, dfinal = cotangents
    final = tuple(dfinal[:, slot, None].astype(jnp.float32) for slot in range(3))
    dk, dv, dw, du, *leaving = walked_back(k, v, w, u, starts, *entering, dy, *final, shape=shape)
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
