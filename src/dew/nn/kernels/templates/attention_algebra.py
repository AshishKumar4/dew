"""The attention template's algebra (docs/design/kernel-templates.md §3): the pure functions on tile
values that lie between a mainloop's two products, `S = Q·Kᵀ` and `O += P·V`. A mainloop issues the
products, loads the tiles and supplies their coordinates; it calls these functions in whatever order
and placement its pipeline needs, and holds no masking, hook or softmax arithmetic of its own, so every
mainloop computes the same thing (tests/test_attention_algebra.py runs one algebra under the Triton
mainloop and under the plain-JAX one, `attention_portable`, and asks for the same bits).

A forward step over one key tile, in a mainloop's order:

    s = modify(spec, S, coords, params, masked)    # scale, score hook, mask
    m, alpha = peak(spec, state.m, s)
    p = weights(spec, s, m)
    l = total(state.total, alpha, p)
    acc = rescale(state.acc, alpha) + narrow(p, V.dtype) · V

`Partial` is that running state, `merge` joins two of them in order (split keys, ring attention,
chunked prefill), and `finish` normalizes the last one and applies the output hook.

The functions specialize at trace time on what the call holds (`Forward`): with no score hook the
logits stay in base 2 (log2(e) folded into the scale) for exp2; a full or causal call's mask comes from
its coordinates alone, only spans read intervals and ids, and a tile no query's interval cuts takes no
mask at all (`masked`); and only a call where a query can see nothing (spans, a score hook) guards its
maxima and denominators against an empty row."""

import dataclasses
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from dew.nn.kernels.templates.attention import Kind

LOG2E = 1.4426950408889634
"""log2(e): a natural exponent times it is a base-2 one."""


@dataclasses.dataclass(frozen=True)
class Forward:
    """What the forward's functions specialize on, all static: the spans' `form` ('full', 'causal' or
    'spans', `KeySpans.form`), the call's key count `keys`, whether it packs documents (`ids`), the
    softmax `scale`, and the score hook `score_mod(s, q_pos, k_pos, *params)` on natural logits."""

    form: str
    keys: int
    ids: bool
    scale: float
    score_mod: Callable | None = None

    @property
    def base2(self) -> bool:
        """Whether the logits stay in base 2 (log2(e) folded into the scale, then exp2, jax's mha's and
        FlashAttention 2's arithmetic): no hook reads them. Its known limits, where the reference stays
        finite and apart: a scale of 1/log2(e) (0.69) or more takes a product near fp32's largest past
        it, and the fold's rounding can tie two logits one ulp apart (1610613120 and 1610612992 at scale
        1, which the reference weighs 1 and 0)."""
        return self.score_mod is None

    @property
    def guarded(self) -> bool:
        """Whether a query can see no key, so its maxima and denominators need the empty row's guard: a
        call of no keys, one whose spans narrow, or one a score hook may mask."""
        return self.keys == 0 or self.form == "spans" or self.score_mod is not None

    def power(self, x: jax.Array) -> jax.Array:
        """The exponential the logits are in: exp2 in base 2, exp otherwise."""
        return jnp.exp2(x) if self.base2 else jnp.exp(x)


class Coords(NamedTuple):
    """A tile's coordinates as the mainloop provides them (in the layout its products give): the query
    indices `q_pos` `[bq, 1]` and key indices `k_pos` `[1, bk]`, and where the spans narrow, each
    query's interval `start`, `end` `[bq]` and, with documents, the queries' and keys' ids."""

    q_pos: jax.Array
    k_pos: jax.Array
    start: jax.Array | None = None
    end: jax.Array | None = None
    query_ids: jax.Array | None = None
    key_ids: jax.Array | None = None


class Partial(NamedTuple):
    """A query block's softmax state, unhooked: the weighted values `acc` `[..., bq, Dv]`, the running
    maximum `m` `[..., bq]` (in base 2 where the step's logits are) and the denominator `total`
    `[..., bq]`, fp32. `rescale`, `merge` and `finish` take any leading axes, so states a launch wrote
    per slice merge in XLA after it."""

    acc: jax.Array
    m: jax.Array
    total: jax.Array


def empty(rows: int, width: int) -> Partial:
    """The softmax before any key: `merge`'s identity."""
    return Partial(jnp.zeros((rows, width), jnp.float32), jnp.full((rows,), -jnp.inf, jnp.float32),
                   jnp.zeros((rows,), jnp.float32))


def resumed(spec: Forward, m: jax.Array, total: jax.Array, acc: jax.Array) -> Partial:
    """The softmax resumed from a state `(m, total, acc)` (`Hooks.initial`: sinks, a resumed partial),
    its maximum taken to base 2 where the step's logits are."""
    m, total, acc = (array.astype(jnp.float32) for array in (m, total, acc))
    return Partial(acc, m * LOG2E if spec.base2 else m, total)


def modify(spec: Forward, s: jax.Array, coords: Coords, params: tuple[jax.Array, ...] = (),
           masked: bool = True) -> jax.Array:
    """A tile's logits from its products `s` `[bq, bk]` (fp32): scaled (by log2(e) too in base 2), then
    the score hook, then -inf at each key the query does not see. `masked` False is a tile every query
    of the block sees whole (a full tile of the block map), which takes no mask."""
    logits = s * (spec.scale * LOG2E if spec.base2 else spec.scale)
    if spec.score_mod is not None:
        logits = spec.score_mod(logits, coords.q_pos, coords.k_pos, *params)
    if not masked:
        return logits
    seen = None
    if spec.form == "spans":
        start, end = _given(coords.start, coords.end, "its queries' intervals")
        seen = (coords.k_pos >= start[:, None]) & (coords.k_pos < end[:, None])
    elif spec.form == "causal":
        seen = coords.k_pos <= coords.q_pos
    if spec.form != "spans" and spec.keys % s.shape[1]:
        # A tile past the call's last key reads padding.
        seen = coords.k_pos < spec.keys if seen is None else seen & (coords.k_pos < spec.keys)
    if spec.ids:
        query_ids, key_ids = _given(coords.query_ids, coords.key_ids, "its queries' and keys' ids")
        same = query_ids[:, None] == key_ids[None, :]
        seen = same if seen is None else seen & same
    return logits if seen is None else jnp.where(seen, logits, -jnp.inf)


def _given(first: jax.Array | None, second: jax.Array | None, what: str) -> tuple[jax.Array, jax.Array]:
    """Two of a tile's coordinates its call's form reads, which its mainloop gives it (`Coords`)."""
    if first is None or second is None:
        raise ValueError(f"a mainloop gives each tile of this call {what} (`Coords`)")
    return first, second


def _safe(spec: Forward, m: jax.Array) -> jax.Array:
    """`m`, with 0 for a query that has seen nothing where one can (`Forward.guarded`)."""
    return jnp.where(m == -jnp.inf, 0.0, m) if spec.guarded else m


def peak(spec: Forward, m: jax.Array, s: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The running maximum after a tile's logits `s`, and `alpha`, the factor the state before it
    takes to the new maximum."""
    new = jnp.maximum(m, s.max(axis=-1))
    return new, spec.power(m - _safe(spec, new))


def weights(spec: Forward, s: jax.Array, m: jax.Array) -> jax.Array:
    """A tile's unnormalized weights `[bq, bk]` under the running maximum `m`, fp32."""
    return spec.power(s - _safe(spec, m)[..., None])


def narrow(p: jax.Array, dtype) -> jax.Array:
    """The weights at the values' dtype, the operand of `P·V`: the reference rounds them so too."""
    return p.astype(dtype)


def total(denominator: jax.Array, alpha: jax.Array, p: jax.Array) -> jax.Array:
    """The denominator after a tile."""
    return alpha * denominator + p.sum(axis=-1)


def rescale(acc: jax.Array, alpha: jax.Array) -> jax.Array:
    """The weighted values before a tile, taken to the new maximum; `P·V` adds the tile's to them."""
    return alpha[..., None] * acc


class Argmax(NamedTuple):
    """The keys' running maximum logit `peak` `[bq]` (the initial state's apart) and `key`, the first
    key that holds it `[bq]` int32: the backward's anchor key (`off_anchor`), a residual of the forward."""

    peak: jax.Array
    key: jax.Array


def no_argmax(rows: int) -> Argmax:
    """`Argmax` before any key."""
    return Argmax(jnp.full((rows,), -jnp.inf, jnp.float32), jnp.zeros((rows,), jnp.int32))


def argmax(state: Argmax, logits: jax.Array, k_pos: jax.Array) -> Argmax:
    """`Argmax` after a tile's logits (-inf where unseen) at its keys' indices `k_pos` `[1, bk]`: one
    reduction to each row's first maximum (`jnp.argmax`'s ties to the first)."""
    tile_peak = logits.max(axis=-1)
    first = k_pos[0, 0] + jnp.argmax(logits, axis=-1).astype(jnp.int32)
    return Argmax(jnp.maximum(state.peak, tile_peak), jnp.where(tile_peak > state.peak, first, state.key))


def anchor_split(spec: Forward, narrowed: jax.Array, at_anchor: jax.Array, anchor: "Argmax",
                 m: jax.Array) -> tuple[jax.Array, jax.Array]:
    """A tile's narrowed weights split at the anchor key (`at_anchor`): the others, the left operand of
    the product that adds to `B = sum_{j != c} e_j v_j`, and the anchor's own weight `[bq]`, fp32, its
    logit the anchor's peak, so no reduction finds it (where the anchor is not in the tile it is unread,
    `split_accumulate`)."""
    own = narrow(weights(spec, anchor.peak[:, None], m), narrowed.dtype).astype(jnp.float32)[:, 0]
    return jnp.where(at_anchor, jnp.zeros_like(narrowed), narrowed), own


def split_accumulate(others: jax.Array, own: jax.Array, others_total: jax.Array, alpha: jax.Array,
                     moved: jax.Array, old_value: jax.Array, product: jax.Array, rest: jax.Array,
                     own_tile: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A training forward's weighted values kept apart from the anchor key's (`others`, `B`, with the
    initial state's), the anchor's own weight, and the other keys' weights' sum (`others_total`, keys
    alone), after a tile: where the anchor moved to this tile (`moved`), the old anchor's term (its
    weight times its value row, `old_value`, which the mainloop gathers, and its weight) joins `B` and
    the sum before the rescale, and the new anchor's weight is the tile's (`own_tile`); the tile's other
    keys add their `product` and their narrowed weights `rest`. So neither is summed with a key's weight
    cancelled out of it."""
    joined = jnp.where(moved[:, None], own[:, None] * old_value.astype(jnp.float32), 0.0)
    total = alpha * (others_total + jnp.where(moved, own, 0.0)) + rest.astype(jnp.float32).sum(axis=-1)
    return rescale(others + joined, alpha) + product, jnp.where(moved, own_tile, alpha * own), total


def split_output(others: jax.Array, own: jax.Array, anchor_value: jax.Array) -> jax.Array:
    """The weighted values of every key, `B` and the anchor's own term (`anchor_value`, its value row)."""
    return others + own[:, None] * anchor_value.astype(jnp.float32)


def merge(spec: Forward, first: Partial, second: Partial) -> Partial:
    """Two states over disjoint keys as one, `first`'s keys before `second`'s: each taken to the larger
    maximum and summed, in that order. Either may hold no key whatever the call (a slice past a row's
    keys), so `empty` is its identity under every spec."""
    m = jnp.maximum(first.m, second.m)
    safe = jnp.where(m == -jnp.inf, 0.0, m)
    a, b = spec.power(first.m - safe), spec.power(second.m - safe)
    acc = a[..., None] * first.acc + b[..., None] * second.acc
    return Partial(acc, m, a * first.total + b * second.total)


def finish(spec: Forward, state: Partial, *, initial: bool, output_mod: Callable | None = None,
           output_params: tuple[jax.Array, ...] = ()) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The block's output `[bq, Dv]` fp32 (its weighted values over its denominator, zero for a query
    that saw nothing and started from no state, then the output hook) and its residuals: the natural
    maximum (0 for such a query, as the reference's) and the denominator."""
    m = state.m / LOG2E if spec.base2 else state.m
    denominator = state.total
    if spec.guarded or initial:
        out = jnp.where(denominator[..., None] > 0,
                        state.acc / jnp.where(denominator > 0, denominator, 1.0)[..., None], 0.0)
        m = jnp.where(m == -jnp.inf, 0.0, m)
    else:
        out = state.acc / denominator[..., None]
    if output_mod is not None:
        out = output_mod(out, *output_params)
    return out, m, denominator


# The backward's algebra: what lies between its products (S = Q·Kᵀ, dP = dO·Vᵀ, then dV, dK and dQ),
# on natural logits.


def inverse(total: jax.Array) -> jax.Array:
    """Each query's `1 / total`, zero where it saw nothing (total 0)."""
    return jnp.where(total > 0, 1.0 / jnp.where(total > 0, total, 1.0), 0.0)


def floating_params(params) -> list[int]:
    """The places of a hook's floating parameters, the ones it differentiates along."""
    return [index for index, param in enumerate(params) if jnp.issubdtype(param.dtype, jnp.floating)]


def differentiable(score_mod: Callable, q_pos, k_pos, params, floating: list[int]) -> Callable:
    """`score_mod` at the tile's coordinates as a function of its logits and its floating parameters
    (`floating`) alone, the others held: what `jax.vjp` takes."""
    def hooked(s, *wide):
        values = list(params)
        for index, value in zip(floating, wide, strict=True):
            values[index] = value
        return score_mod(s, q_pos, k_pos, *values)

    return hooked


def scored(score_mod, logits, q_pos, k_pos, params, *, slopes: bool):
    """`score_mod` on the scaled logits, in fp32, and with `slopes` each entry's derivative of it along
    the logits and along each parameter, a tile apiece: `score_mod` being pointwise in the entries, they
    are the whole of its pullback (`pulled`). They are its reverse-mode pullback's transpose applied to
    each input's ones, on the hook's own tile and layouts, so a hook through a `jax.custom_vjp`
    differentiates as the reference's does and a hook reads its arguments at the ranks it is called
    with. An integer parameter has no derivative: its slopes are zeros. Without a hook, the logits and
    no slopes."""
    if score_mod is None:
        return logits, None
    if not slopes:
        return score_mod(logits, q_pos, k_pos, *params).astype(jnp.float32), None
    floating = floating_params(params)
    inputs = (logits, *(params[index] for index in floating))
    out, pull = jax.vjp(differentiable(score_mod, q_pos, k_pos, params, floating), *inputs)
    along = jax.linear_transpose(pull, out)

    def slope(index):
        [entries] = along(tuple(jnp.ones_like(x) if place == index else jnp.zeros_like(x)
                                for place, x in enumerate(inputs)))
        return entries.astype(jnp.float32)

    differentiated = {index: slope(place + 1) for place, index in enumerate(floating)}
    return out.astype(jnp.float32), (slope(0), *(differentiated.get(index, jnp.zeros_like(logits))
                                                  for index in range(len(params))))


def pulled(cotangents, slopes, seen, kinds: tuple[Kind, ...]):
    """The score cotangents `cotangents` through `score_mod`, its entries' derivatives `slopes`
    (`scored`; None without a hook): the logits' tile, and each parameter's, a head's or a query's its
    queries' sums `[block_q]`, a key's its keys' `[block_k]`, a tile's the tile. Each entry is masked
    before any sum, so one the call never sees (padding, another document) adds nothing even where the
    hook's derivative there is not finite. fp32."""
    if slopes is None:
        return cotangents, ()
    logits = jnp.where(seen, cotangents * slopes[0], 0.0)
    pieces = []
    for slope, kind in zip(slopes[1:], kinds, strict=True):
        entries = jnp.where(seen, cotangents * slope, 0.0)
        pieces.append(entries.sum(axis=0) if kind == Kind.KEY else entries if kind == Kind.TILE
                      else entries.sum(axis=-1))
    return logits, tuple(pieces)


class OffAnchor(NamedTuple):
    """The backward's first pass weighed against the forward's maximum m, so no running maximum rescales
    a sum: the denominator `total` `[bq]`, and of every key but the forward's maximum key c
    (`Argmax.key`) the weighted values `others` `B = sum_{j != c} e_j v_j` `[bq, Dv]` (the mainloop's
    product of `off_anchor_weights`' narrowed weights with the values), their weights' sum
    `others_total`, and c's own weight `own`, each `e_j = exp(s_j - m)` rounded as the forward rounds it,
    fp32."""

    total: jax.Array
    others: jax.Array
    others_total: jax.Array
    own: jax.Array


def off_anchor_start(peak0: jax.Array, total0: jax.Array, term0: jax.Array, maximum: jax.Array,
                     width: int) -> tuple[OffAnchor, jax.Array, jax.Array]:
    """The first pass from the softmax's initial state (its maximum, denominator and share of D's
    numerator) weighed at the forward's `maximum`, and that state's denominator and share there."""
    carried = jnp.where(peak0 == -jnp.inf, 0.0, jnp.exp(jnp.minimum(peak0 - maximum, 0.0)))
    state_total, state_share = total0 * carried, term0 * carried
    nothing = jnp.zeros_like(total0)
    return OffAnchor(state_total, jnp.zeros((total0.shape[0], width), jnp.float32), nothing, nothing), \
        state_total, state_share


def off_anchor_weights(logits: jax.Array, maximum: jax.Array, at_anchor: jax.Array,
                       dtype) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A tile's weights `exp(s - m)` at the forward's maximum, fp32; them narrowed to the values'
    `dtype` as the forward rounds them, zero at the anchor key (`at_anchor`), the left operand of `B`'s
    product; and the anchor key's narrowed weight, fp32."""
    exponentials = below(logits, maximum)
    rounded = exponentials.astype(dtype)
    rest = jnp.where(at_anchor, jnp.zeros_like(rounded), rounded)
    own = jnp.where(at_anchor, rounded, jnp.zeros_like(rounded)).astype(jnp.float32).sum(axis=-1)
    return exponentials, rest, own


def off_anchor(state: OffAnchor, exponentials: jax.Array, rest: jax.Array, own: jax.Array,
               others: jax.Array) -> OffAnchor:
    """The first pass after a tile (`off_anchor_weights`), `others` the tile's `rest · V`."""
    return OffAnchor(state.total + exponentials.sum(axis=-1), state.others + others,
                     state.others_total + rest.astype(jnp.float32).sum(axis=-1), state.own + own)


def off_anchor_delta(state: OffAnchor, anchor_value: jax.Array, dout: jax.Array, state_total: jax.Array,
                     state_share: jax.Array, key_index: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Each query's anchor c (`v_c · dO`, or 0), `D - c`, and the anchor key's index (-1 where c is 0),
    from the first pass. D's numerator two ways: about 0, `sum_j e_j v_j · dO`, as the output's own
    product with dO rounds it, which suits a query whose weights spread over many keys; and about
    `c = v_c · dO`, `sum_{j != c} e_j (v_j - v_c) · dO`, where c enters as exactly 0, which suits one
    whose weight sits on c. Each query takes the one nearer its D, the one whose terms cancel less."""
    cotangent = dout.astype(jnp.float32)
    anchor_value = anchor_value.astype(jnp.float32)
    anchor = jnp.sum(anchor_value * cotangent, axis=-1)
    about_zero = (jnp.sum((state.others + state.own[:, None] * anchor_value) * cotangent, axis=-1)
                  + state_share)
    about_anchor = (jnp.sum((state.others - state.others_total[:, None] * anchor_value) * cotangent, axis=-1)
                    + state_share - state_total * anchor)
    anchored = jnp.abs(about_anchor) <= jnp.abs(about_zero)
    numerator = jnp.where(anchored, about_anchor, about_zero)
    return (jnp.where(anchored, anchor, 0.0), numerator * inverse(state.total),
            jnp.where(anchored, key_index, -1))


def split_delta(others: jax.Array, own: jax.Array, others_total: jax.Array, total: jax.Array,
                state_total: jax.Array, anchor_value: jax.Array, dout: jax.Array,
                key_index: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`off_anchor_delta` from a training forward's split sums (`split_accumulate`), the backward's first
    pass taken by the forward: `B` with the initial state's share inside it, the other keys' weights'
    sum, the anchor's own weight, the denominator `total` and the initial state's own `state_total`, all
    at the forward's maximum. D's numerator about 0 is `(B + own v_c) · dO`, about `c = v_c · dO` it is
    `(B - others_total v_c) · dO - state_total c`, its terms kept apart so a state or keys that weigh
    little beside the anchor are not lost from a total; each query takes the one nearer its D."""
    cotangent = dout.astype(jnp.float32)
    anchor_value = anchor_value.astype(jnp.float32)
    anchor = jnp.sum(anchor_value * cotangent, axis=-1)
    about_zero = jnp.sum(split_output(others, own, anchor_value) * cotangent, axis=-1)
    about_anchor = (jnp.sum((others - others_total[:, None] * anchor_value) * cotangent, axis=-1)
                    - state_total * anchor)
    anchored = jnp.abs(about_anchor) <= jnp.abs(about_zero)
    numerator = jnp.where(anchored, about_anchor, about_zero)
    return jnp.where(anchored, anchor, 0.0), numerator * inverse(total), jnp.where(anchored, key_index, -1)


class DqSums(NamedTuple):
    """A dQ pass's sums over its keys: dQ `[bq, D]` and the score cotangents' row sums and the HEAD and
    QUERY hook parameters' pullbacks of them (`grads`), each cotangent taken over a `prior` denominator
    (the forward's, or the first pass's; `grad_scores`); and the unnormalized weights' row sums and the
    hook parameters' pullbacks of them (`weighed`), fp32."""

    dq: jax.Array
    residual: jax.Array
    mass: jax.Array
    grads: tuple[jax.Array, ...]
    weighed: tuple[jax.Array, ...]


def dq_finish(sums: DqSums, prior: jax.Array, state_total: jax.Array, peak0: jax.Array,
              shares: tuple[jax.Array, jax.Array], peak: jax.Array, anchor: jax.Array, delta: jax.Array):
    """A dQ pass's results from its sums, each taken to the pass's own denominator, the initial state's
    `state_total` and its keys' weights, so the weights it differentiates sum to one over its own logits
    whichever pass gave D: the cotangents' sums from the prior by their ratio (one division, about 1),
    the weights' over it. D is then refined once from its own sums (`refinement`), and the HEAD and
    QUERY gradients with it. Returns dQ, the refined `D - c`, the denominator, the keys' share of the
    weights, and the hook gradients."""
    total = state_total + sums.mass
    ratio = jnp.where(total > 0, prior / jnp.where(total > 0, total, 1.0), 0.0)
    over = inverse(total)
    mass, residual = sums.mass * over, sums.residual * ratio
    refined = refinement(residual, mass, peak0, shares, peak, total, anchor, delta)
    grads = tuple(grad * ratio - refined * (weighed * over)
                  for grad, weighed in zip(sums.grads, sums.weighed, strict=True))
    return sums.dq * ratio[:, None], delta + refined, total, mass, grads


def below(logits: jax.Array, peak: jax.Array) -> jax.Array:
    """`exp(s - m)` for a tile's logits recomputed in the backward against the forward's maximum `m`,
    fp32: `m` is every key's maximum, so a logit past it by the rounding the two passes do not share (the
    forward's maximum taken in base 2 and back, or another tile shape's product) is the maximum itself,
    its weight 1, never past it."""
    return jnp.exp(jnp.minimum(logits - peak[:, None], 0.0))


def normalized(logits: jax.Array, peak: jax.Array, total: jax.Array) -> tuple[jax.Array, jax.Array]:
    """A tile's unnormalized weights `exp(s - m)` (`below`) and its weights `p`, fp32."""
    exponentials = below(logits, peak)
    return exponentials, exponentials * inverse(total)[:, None]


def grad_scores(exponentials: jax.Array, products: jax.Array, anchor: jax.Array, delta: jax.Array,
                seen: jax.Array, at_anchor: jax.Array | None, inverse_total: jax.Array) -> jax.Array:
    """The score cotangents `p (dP - D)` of a tile in the reference's own order, as JAX differentiates
    `exp(s - m) / T`: each query's `(dP - c) - (D - c)` about its anchor c (`off_anchor_delta`) over its
    denominator (`inverse_total`), then times the unnormalized weights `exp(s - m)`, so every
    intermediate lies where the reference's does (a weight below fp32's range keeps its share beside a
    large product, and none overflows that the reference keeps); zero where unseen. The anchor key's
    own `dP - c` is 0 by definition where `at_anchor` names it, whatever its product's rounding."""
    relative = products - anchor[:, None]
    if at_anchor is not None:
        relative = jnp.where(at_anchor, 0.0, relative)
    return jnp.where(seen, exponentials * ((relative - delta[:, None]) * inverse_total[:, None]), 0.0)


def refinement(residual: jax.Array, mass: jax.Array, peak0: jax.Array, shares: tuple[jax.Array, jax.Array],
               peak: jax.Array, total: jax.Array, anchor: jax.Array, delta: jax.Array) -> jax.Array:
    """D's one refinement from the main pass's own sums: the score cotangents of a query and of its
    initial state (`shares`: its denominator and D's numerator at the state's maximum `peak0`) sum to 0
    by the softmax's shift, and what they sum to here (`residual`, over the keys' weights `mass`) is D's
    rounding. Moving D by it over the weights' total lets dK/dV, the initial state's gradients and the
    head and query parameters' take the D whose cotangents do."""
    carried = jnp.where(peak0 == -jnp.inf, 0.0, jnp.exp(jnp.minimum(peak0 - peak, 0.0))) * inverse(total)
    whole_residual = residual + (shares[1] - shares[0] * (anchor + delta)) * carried
    whole_mass = mass + shares[0] * carried
    return jnp.where(whole_mass > 0, whole_residual / jnp.where(whole_mass > 0, whole_mass, 1.0), 0.0)


def value_operands(exponentials: jax.Array, weights: jax.Array, dout: jax.Array,
                   total: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The operands `(a, b)` of a tile's share of dV, `aᵀ·b` `[block_k, Dv]` in fp32: the unnormalized
    weights `exp(s - m)` narrowed to the operands' dtype as the reference's forward rounds them, against
    each query's `dout / total` in fp32, the reference's derivative's own order, so no small weight or
    share is lost to normalizing first. bf16 alone takes FlashAttention's form, its normalized weights
    `p` narrowed against the output's cotangent as it is, so the product stays on bf16's tensor cores: a
    query's cotangents that cancel across a key still cancel, and a normalized weight below bf16's least
    subnormal (about 9e-41) is dropped, which only a cotangent past about 1e28 would make count."""
    if dout.dtype == jnp.bfloat16:
        return weights.astype(dout.dtype).T, dout
    scaled = dout.astype(jnp.float32) * inverse(total)[:, None]
    return exponentials.astype(dout.dtype).astype(jnp.float32).T, scaled
