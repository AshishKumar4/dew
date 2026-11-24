"""RWKV4's WKV on Mosaic GPU (`kernels.rwkv4`'s kernels, row 1455), on the channel walk (`walks`): a program a
row, a span of its tokens and a warpgroup's channels, each thread its channel walking the span's tokens with
fla's carry, the numerator `alpha` and denominator `beta` scaled by `e^-eps`, `eps` their running maximum
(fla/ops/rwkv4/fused_recurrent.py; `rwkv4.recurrent_wkv4`, its arithmetic token by token):

    out_t = (e^{eps - tau} alpha + e^{u + k_t - tau} v_t) / (e^{eps - tau} beta + e^{u + k_t - tau}),
    tau = max(u + k_t, eps), then eps' = max(w + eps, k_t) and alpha', beta' rescaled to it.

A call over several spans summarises each span's carry from an empty past, XLA joins the summaries by the
portable scan's operator (`rwkv4.joined`) for each span's entering carry, and a launch walks each span from
its own, writing its outputs, its last carry and whether the carry entering it kept the running maximum
through it (`survived`).

The backward is the numerator's and denominator's own, `A = alpha e^eps` and `B = beta e^eps`, which the
outputs read alone: a token's output's cotangent `g_t` gives them `c_t e^-tau` and `-c_t out_t e^-tau`,
`c_t = g_t / (e^{eps - tau} beta + e^{u + k_t - tau})`, carried to the tokens before by `e^w` a token. Their
carry is held as fla's forward holds its own, two values scaled by `e^mu`, `mu` the running maximum walked
back (`mu_t = max(-tau_t, w + mu_{t+1})`), so a token's terms `e^{mu + k}` and `e^{w + mu + eps}` are at
most one. The final carry's `eps` cotangent, net of what `alpha` and `beta` take of it, runs the running
maximum's own path back: to the key that set it, or to `w` each token it aged. Spans as the forward: a
launch summarises each span's carry walked back from none, XLA joins them by the same operator from the
final carry's, and a launch walks each span back from its own. Each recomputes the forward from its span's
entering carry, a walk forward keeping each turn's carry in shared memory.

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
    """A token's output from the carry it reads, and the terms its backward reuses: `(out, tau, present,
    denominator)`, `present = e^{u + k - tau}`."""
    alpha, beta, eps = carry
    own = u + k
    tau = jnp.maximum(own, eps)
    past, present = jnp.exp(eps - tau), jnp.exp(own - tau)
    denominator = past * beta + present
    return (past * alpha + present * v) / denominator, tau, present, denominator


def _weights(eps, k, w) -> tuple:
    """A token's step of the running maximum: `(kept, added, top)`, the past's weight `e^(w + eps - top)`
    and the token's `e^(k - top)` at the new maximum `top = max(w + eps, k)`."""
    aged = w + eps
    top = jnp.maximum(aged, k)
    return jnp.exp(aged - top), jnp.exp(k - top), top


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
    `[B, T, D]` in `k`'s dtype, each span's last carry, and the share of the maximum's cotangent its end
    passes back to the carry entering it (`_shares`: one where the entering maximum held through the span,
    halved at each tie, zero where a key set it or a document started)."""
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
                kept = _shares(k_t, decay + read[2], survived)[1]
                survived = jnp.where(program.inside(t, k_t), kept, survived)
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


def _shares(k_t, aged, e):
    """The running maximum's cotangent `e` at a token's `max(aged, k_t)` as `(the key's, the aged past's)`:
    all to the larger, half to each at a tie, as JAX's `maximum` takes it a token at a time (fla's recurrence,
    `rwkv4.recurrent_wkv4`; the portable scan's associative grouping splits a tie of three or more tokens by
    its own tree), each selected, never a product with zero."""
    zero, half = jnp.zeros_like(e), e * 0.5
    won, tied = k_t > aged, k_t == aged
    return jnp.where(won, e, jnp.where(tied, half, zero)), jnp.where(won, zero, jnp.where(tied, half, e))


def _reverse_step(program: Program, t, k_t, v_t, dy_t, read, terms, decay, carried, *, summed: bool = False):
    """One token walked back: `(carried before it, (dk, dv, dw, du) of it)`, `carried` the reverse carry
    `(a, b, mu)` of the state after it (`A`'s and `B`'s cotangents scaled by `e^-mu`) and the cotangent `e`
    of its running maximum; outside the row, the carry as it is and no gradient. `summed`, a span's summary
    walked back from none: `a` and `b` each a running sum `(value, scale)` (`walks.summed_on`), the
    gradients of none."""
    a, b, mu, e = carried
    if summed:
        a, b = a[0], b[0]
    alpha, beta, eps = read
    out, tau, present, denominator = terms
    c = dy_t / denominator
    direct = c * present
    weight_k, weight_w = jnp.exp(mu + k_t), jnp.exp(decay + mu + eps)
    zero = jnp.zeros_like(e)
    to_key, to_age = _shares(k_t, decay + eps, e)
    dk = direct * (v_t - out) + weight_k * (a * v_t + b) + to_key
    dv = direct + a * weight_k
    dw = weight_w * (a * alpha + b * beta) + to_age
    du = direct * (v_t - out)
    top = jnp.maximum(-tau, decay + mu)
    own, later = jnp.exp(-tau - top), jnp.exp(decay + mu - top)
    if summed:
        before = (summed_on(carried[0], later * a, own * c, None),
                  summed_on(carried[1], later * b, -(own * c) * out, None), top, to_age)
        emptied = ((zero, zero), (zero, zero), zero - jnp.inf, zero)
    else:
        before = (own * c + later * a, later * b - own * c * out, top, to_age)
        emptied = (*_empty(k_t), zero)
    first = program.first(t, k_t)
    before = jax.tree.map(lambda value, empty: cut(first, value, empty), before, emptied)
    inside = program.inside(t, k_t)
    return _held(inside, before, carried), tuple(jnp.where(inside, x, zero) for x in (dk, dv, dw, du))


def _kept(shape: Walk):
    """A backward's shared memory: each turn's carry, three values of its channels."""
    return plgpu.SMEM((shape.span // TURN, 3, shape.lanes), jnp.float32)


@tuning.lowered_once(("shape",))
def reverse_summaries(k, v, w, u, starts, alpha, beta, eps, dy, *, shape: Walk):
    """Each span's reverse carry walked back from none, `(a, a scale, b, b scale, mu)` `[B, spans, D]` fp32:
    the cotangents its own outputs' (`dy`) give the state entering it, each a running sum
    (`walks.summed_on`), from each span's entering carry (`alpha`, `beta`, `eps`)."""
    operands = (k, v, w, u, starts, alpha, beta, eps, dy)

    def entry(*refs):
        given, (*made, kept_ref) = refs_of(refs, operands)
        k_ref, v_ref, w_ref, u_ref, starts_ref, *entering, dy_ref = given
        program = Program(shape, starts_ref)
        zero = jnp.zeros((shape.lanes,), jnp.float32)

        def emit(t, k_t, v_t, dy_t, read, terms, decay, carried):
            return _reverse_step(program, t, k_t, v_t, dy_t, read, terms, decay, carried, summed=True)[0]

        (a, b, mu, _) = _back(program, (k_ref, v_ref, w_ref, u_ref, dy_ref, entering), shape, kept_ref,
                              ((zero, zero), (zero, zero), zero - jnp.inf, zero), emit)
        for ref, value in zip(made, (*a, *b, mu), strict=True):
            program.summarised(ref, value)

    out_type = tuple(per_span(k, shape) for _ in range(5))
    return launch(entry, shape, "wkv4_summaries_backward", out_type, operands, scratch=(_kept(shape),))


@tuning.lowered_once(("shape",))
def walked_back(k, v, w, u, starts, alpha, beta, eps, dy, later_a, later_b, later_mu, later_e, *,
                shape: Walk):
    """The gradients of `k` and `v` (`[B, T, D]` in their dtype), each program's of `w` and `u` (`[B, spans,
    D]` fp32, XLA sums them), and each span's entering carry's reverse carry and maximum's cotangent (`(a, b,
    mu, e)` `[B, spans, D]` fp32), from the outputs' cotangent `dy`, each span's entering carry and the
    reverse carry and maximum's cotangent its last state takes from the later tokens and the final carry
    (`later_*`)."""
    operands = (k, v, w, u, starts, alpha, beta, eps, dy, later_a, later_b, later_mu, later_e)

    def entry(*refs):
        given, made = refs_of(refs, operands)
        k_ref, v_ref, w_ref, u_ref, starts_ref, *carries, dy_ref = given[:9]
        dk_ref, dv_ref, dw_ref, du_ref, *leaving_refs, kept_ref = made
        program = Program(shape, starts_ref)
        zero = jnp.zeros((shape.lanes,), jnp.float32)

        def emit(t, k_t, v_t, dy_t, read, terms, decay, carried):
            reverse, (dw_sum, du_sum) = carried
            reverse, (dk, dv, dw, du) = _reverse_step(program, t, k_t, v_t, dy_t, read, terms, decay,
                                                      reverse)
            program.stored(dk_ref, t, dk)
            program.stored(dv_ref, t, dv)
            return reverse, (compensated_add(dw_sum, (dw, zero)), compensated_add(du_sum, (du, zero)))

        later = tuple(program.summary(ref) for ref in given[9:])
        reverse, sums = _back(program, (k_ref, v_ref, w_ref, u_ref, dy_ref, carries), shape, kept_ref,
                              (later, ((zero, zero), (zero, zero))), emit)
        for ref, (high, low) in zip((dw_ref, du_ref), sums, strict=True):
            program.summarised(ref, high + low)
        for ref, value in zip(leaving_refs, reverse, strict=True):
            program.summarised(ref, value)

    out_type = (per_token(k, k.dtype), per_token(v, v.dtype), *(per_span(k, shape) for _ in range(6)))
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
    """`rwkv4.SpanCarry` as the template's join holds it: a signed value (the numerator; reversed, both) a
    scaled value (`walks.extended`, `dew.algebra.scaled.Scaled`), `rwkv4.joined` reading each field by its
    name alike."""

    numerator: jax.Array | scaled.Scaled
    denominator: jax.Array | scaled.Scaled
    top: jax.Array
    tokens: jax.Array
    fresh: jax.Array


def _weighed(kept, earlier, added_, later):
    """Two spans' scaled values (`walks.extended`) at their larger maximum, `rwkv4.joined`'s weighing."""
    return added(multiplied(earlier, extended(kept, 0)), multiplied(later, extended(added_, 0)))


def _joined(carries: tuple, given: tuple, decay, shape: Walk, starts, *, reverse: bool = False) -> tuple:
    """Each span's carry as its walk enters it, `(alpha, beta, eps)` `[B, spans, D]`: `given` (the call's
    entering carry, `[B, D]` each; `reverse`, the final carry's reverse carry, entering the last span from its
    end) joined with the spans' (`carries`, from an empty past, each signed value a running sum's value and
    scale: the numerator; reversed, both) by the portable scan's operator (`rwkv4.joined`) over XLA's
    associative scan, the signed values scaled (`walks.extended`): a span's from none is its
    last less its entering one's decayed share, past float32's largest where every value is in it."""
    from dew.nn.kernels import rwkv4

    if reverse:
        a, a_scale, b, b_scale, top = carries
        values = (extended(a, 64.0 * a_scale), extended(b, 64.0 * b_scale))
        edges = (extended(given[0][:, None], 0), extended(given[1][:, None], 0))
        weighing = {"numerators": _weighed, "denominators": _weighed}
    else:
        numerator, scale, beta, top = carries
        values = (extended(numerator, 64.0 * scale), beta)
        edges = (extended(given[0][:, None], 0), given[1][:, None])
        weighing = {"numerators": _weighed}
    tokens, fresh = _spans(shape, starts, top)
    spans = _Carry(*values, top, tokens, fresh)
    edge = _Carry(*edges, given[2][:, None], jnp.zeros_like(tokens[:, :1]), jnp.zeros_like(fresh[:, :1]))
    ordered = (spans, edge) if reverse else (edge, spans)
    joined = jax.lax.associative_scan(rwkv4.joined(decay, **weighing),
                                      jax.tree.map(lambda *parts: jnp.concatenate(parts, axis=1), *ordered),
                                      axis=1, reverse=reverse)
    entered = jax.tree.map(lambda value: value[:, 1:] if reverse else value[:, :-1], joined)
    numerator = valued(entered.numerator)
    denominator = valued(entered.denominator) if reverse else entered.denominator
    return numerator, denominator, entered.top


def _forward(k, v, w, u, state, starts, shape: Walk):
    """`(outputs, final carry [B, 3, D], residuals)`."""
    given = (state[:, 0], state[:, 1], state[:, 2])
    alpha, beta, eps = state[:, 0, None], state[:, 1, None], state[:, 2, None]
    if shape.spans > 1:
        alpha, beta, eps = _joined(summaries(k, v, w, starts, shape=shape), given, w, shape, starts)
    out, last_alpha, last_beta, last_eps, survived = walked(k, v, w, u, starts, alpha, beta, eps, shape=shape)
    final = jnp.stack([last_alpha[:, -1], last_beta[:, -1], last_eps[:, -1]], axis=1)
    return out, final, ((alpha, beta, eps), survived)


def _primal(k, v, w, u, state, starts, shape: Walk):
    return _forward(k, v, w, u, state, starts, shape)[:2]


def _forward_rule(k, v, w, u, state, starts, shape: Walk):
    out, final, (entering, survived) = _forward(k, v, w, u, state, starts, shape)
    return (out, final), (k, v, w, u, state, starts, entering, survived, final)


def _backward_rule(shape: Walk, residuals, cotangents):
    k, v, w, u, state, starts, entering, survived, final = residuals
    dy, dfinal = cotangents
    dfinal = dfinal.astype(jnp.float32)
    # The final carry's cotangents as the reverse carry of `A` and `B` past the last token, and its maximum's
    # net of what `alpha` and `beta` take of it.
    given = (dfinal[:, 0], dfinal[:, 1], -final[:, 2])
    raised = dfinal[:, 2] - dfinal[:, 0] * final[:, 0] - dfinal[:, 1] * final[:, 1]
    alpha, beta, eps = entering
    later_a, later_b, later_mu = (value[:, None] for value in given)
    if shape.spans > 1:
        local = reverse_summaries(k, v, w, u, starts, alpha, beta, eps, dy, shape=shape)
        later_a, later_b, later_mu = _joined(local, given, w, shape, starts, reverse=True)
    # The maximum's cotangent reaching each span's end: the later spans' shares of it (`walked`), selected
    # zero where a later span cut it (an infinite cotangent times zero was NaN, ReviewRecurrenceBatch).
    held = jnp.cumprod(survived[:, ::-1], axis=1)[:, ::-1]
    passed = jnp.concatenate([held[:, 1:], jnp.ones_like(held[:, :1])], axis=1)
    later_e = jnp.where(passed == 0.0, 0.0, raised[:, None] * passed)
    dk, dv, dw, du, a, b, mu, e = walked_back(k, v, w, u, starts, alpha, beta, eps, dy, later_a, later_b,
                                              later_mu, later_e, shape=shape)
    # The entering carry's cotangents from `A`'s and `B`'s before the first token (`A = alpha e^eps`).
    scale = jnp.exp(mu[:, 0] + state[:, 2])
    dstate = jnp.stack([a[:, 0] * scale, b[:, 0] * scale,
                        scale * (a[:, 0] * state[:, 0] + b[:, 0] * state[:, 1]) + e[:, 0]], axis=1)
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
