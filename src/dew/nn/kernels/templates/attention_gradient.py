"""The attention backward's tile algebra, what lies between its products (S = Q·Kᵀ, dP = dO·Vᵀ, then dV, dK
and dQ) on natural logits: a tile's logits recomputed under the score hook and their derivatives
(`tile_logits`, `scored`, `pulled`), its weights and score cotangents (`grad_scores`), each query's dQ sums
about its anchor key (`DqSums`, `dq_step`, `dq_finish`), the anchors themselves (`off_anchor`, `centre_step`)
and the dropout's (`dropped_products`). The forward's algebra and the products' operand parts every launch
shares are `attention_algebra`'s; the declared backward and its derived forms
`attention_backward_algebra`'s."""

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dew.algebra import rewritten
from dew.nn.kernels.templates.attention import Kind, differentiable, floating_params
from dew.nn.kernels.templates.attention_algebra import (
    PLAIN,
    WEIGHT_SHIFT,
    Parts,
    Recompute,
    cotangent_product,
    expand,
    exponent,
    finite,
    hooked,
    inverse,
    power2,
    split_output,
    summed_products,
    weights_parts,
)
from dew.nn.precision import Running, added_to, compensated_sum, compensates_loops, running_sum, running_total


def tile_logits(score_mod, product: jax.Array, seen: jax.Array | None, q_pos, k_pos, params, *, scale: float,
                slopes: bool = False):
    """A backward tile's logits from its product `Q·Kᵀ` taken by `scale`, under `score_mod`, -inf where
    unseen (`seen`; None, a tile every query sees whole), and with `slopes` its entries' derivatives
    (`scored`)."""
    logits, derivatives = scored(score_mod, product * scale, q_pos, k_pos, params, slopes=slopes)
    return where_seen(seen, logits, -jnp.inf), derivatives


def where_seen(seen: jax.Array | None, entries: jax.Array, unseen: float) -> jax.Array:
    """A tile's `entries` where `seen`, `unseen` elsewhere; every entry where `seen` is None (a tile every
    query sees whole, which a mainloop gives no mask)."""
    return entries if seen is None else jnp.where(seen, entries, unseen)


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
        return hooked(score_mod, logits, q_pos, k_pos, params).astype(jnp.float32), None
    floating = floating_params(params)

    def sloped(*inputs):
        # The hook's value and its slopes, one program the algebra pass rewrites before the tile places it
        # (docs/design/algebra.md §2): a hook's custom derivative is its rule's, applied in the trace.
        out, pull = jax.vjp(differentiable(score_mod, q_pos, k_pos, params, floating), *inputs)
        along = jax.linear_transpose(pull, out)

        def slope(index):
            [entries] = along(tuple(jnp.ones_like(x) if place == index else jnp.zeros_like(x)
                                    for place, x in enumerate(inputs)))
            return entries.astype(jnp.float32)

        differentiated = {index: slope(place + 1) for place, index in enumerate(floating)}
        return out.astype(jnp.float32), (slope(0), *(differentiated.get(index, jnp.zeros_like(logits))
                                                      for index in range(len(params))))
    return rewritten(sloped)(logits, *(params[index] for index in floating))


ROW_SUMS: Mapping[Kind, Callable[[jax.Array], jax.Array]] = MappingProxyType({
    Kind.HEAD: lambda x: x.sum(axis=-1), Kind.QUERY: lambda x: x.sum(axis=-1),
    Kind.KEY: lambda x: x.sum(axis=0)})
"""`pulled`'s sums of a tile `[bq, bk]`: a head's and a query's over its keys, a key's over its queries."""


def pulled(cotangents, slopes, seen, kinds: tuple[Kind, ...],
           sums: Mapping[Kind, Callable[[jax.Array], jax.Array]] = ROW_SUMS):
    """The score cotangents `cotangents` through `score_mod`, its entries' derivatives `slopes`
    (`scored`; None without a hook): the logits' tile, and each parameter's, a head's or a query's its
    queries' sums `[block_q]`, a key's its keys' `[block_k]`, each by its kind's of `sums` (the mainloop's
    own where it takes one, `Family.column_sums`, or a tile whose rows are its keys'), a tile's the tile;
    None for a kind `sums` omits, whose gradient the mainloop does not write. Each entry is masked before
    any sum, so one the call never sees (padding, another document) adds nothing even where the hook's
    derivative there is not finite. fp32."""
    if slopes is None:
        return cotangents, ()
    logits = where_seen(seen, cotangents * slopes[0], 0.0)
    pieces = []
    for slope, kind in zip(slopes[1:], kinds, strict=True):
        entries = where_seen(seen, cotangents * slope, 0.0)
        pieces.append(entries if kind == Kind.TILE else sums[kind](entries) if kind in sums else None)
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


def off_anchor_start(peak0: jax.Array, total0: jax.Array, maximum: jax.Array, width: int,
                     shift: jax.Array) -> tuple[OffAnchor, jax.Array]:
    """The first pass from the softmax's initial state (its maximum and denominator) weighed at the
    forward's `maximum`, its sums at each row's `2^-shift` (the forward's, `room`), and that state's
    denominator there, unshifted; its share of D enters as that weight over the total times its values'
    mean's product with the cotangent (`off_anchor_delta`), never `acc0 · dO`, which overflows where the
    state's sums are near fp32's largest."""
    # The fold's maximum is the state's taken to base 2 and back, which may round it an ulp below.
    carried = jnp.where(peak0 == -jnp.inf, 0.0, jnp.exp(jnp.minimum(peak0 - maximum, 0.0)))
    state_total = total0 * carried
    nothing = jnp.zeros_like(total0)
    first = OffAnchor(power2(state_total, -shift), jnp.zeros((total0.shape[0], width), jnp.float32),
                      nothing, nothing)
    return first, state_total


def off_anchor_weights(logits: jax.Array, maximum: jax.Array, at_anchor: jax.Array,
                       dtype, shift: jax.Array, how: Recompute = PLAIN
                       ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """A tile's weights `exp(s - m)` at the forward's maximum, fp32, at the forward's weights' shift,
    `2^-min(shift, WEIGHT_SHIFT)` (`room`); them narrowed to the values' `dtype` as the forward rounds
    them, zero at the anchor key (`at_anchor`), the left operand of `B`'s product; the anchor key's
    narrowed weight, fp32; and the weights that shift would push below fp32's least normal, unshifted
    and narrowed, whose product the mainloop issues apart (`off_anchor`), as the forward's (`room`)."""
    exponentials = below(logits, maximum, how)
    weight = expand(jnp.minimum(shift, WEIGHT_SHIFT), exponentials)
    small = jnp.where(exponentials < power2(jnp.float32(1.0), weight - 126.0), exponentials, 0.0)
    exponentials = power2(exponentials - small, -weight)
    rounded = exponentials.astype(dtype)
    rest = jnp.where(at_anchor, jnp.zeros_like(rounded), rounded)
    own = jnp.where(at_anchor, rounded, jnp.zeros_like(rounded)).astype(jnp.float32).sum(axis=-1)
    return exponentials, rest, own, small.astype(dtype)


def off_anchor(state: OffAnchor, exponentials: jax.Array, rest: jax.Array, own: jax.Array,
               others: jax.Array, shift: jax.Array, small: jax.Array, small_others: jax.Array) -> OffAnchor:
    """The first pass after a tile (`off_anchor_weights`), `others` the tile's `rest · V`, its sums taken
    from the weights' shift to each row's `shift`, as the forward's (`beyond_weights`), and the small
    weights and their product `small_others` taken there after the product: they feed D alone, whose
    small terms are below an ulp of it."""
    beyond = jnp.minimum(shift, WEIGHT_SHIFT) - shift

    def taken(x, by=beyond):
        return power2(x, by if x.ndim == by.ndim else expand(by, x))

    small_total = taken(small.astype(jnp.float32).sum(axis=-1), -shift)
    return OffAnchor(state.total + taken(exponentials.sum(axis=-1)) + small_total,
                     state.others + taken(others) + taken(small_others, -shift),
                     state.others_total + taken(rest.astype(jnp.float32).sum(axis=-1)) + small_total,
                     state.own + taken(own))


def nearer_centre(anchor: jax.Array, about_anchor: jax.Array, about_zero: jax.Array, key_index: jax.Array
                  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """D's centre, the nearer to D of the anchor key's `dP` (`anchor`) and zero, each sum's difference from
    it (`about_anchor`, `about_zero`): the centre, D less it, and the anchor key's index (`key_index`), -1
    where zero is the nearer."""
    anchored = jnp.abs(about_anchor) <= jnp.abs(about_zero)
    return (jnp.where(anchored, anchor, 0.0), jnp.where(anchored, about_anchor, about_zero),
            jnp.where(anchored, key_index, -1))


def off_anchor_delta(state: OffAnchor, anchor_value: jax.Array, dout: jax.Array, state_total: jax.Array,
                     mean0: jax.Array, key_index: jax.Array,
                     shift: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Each query's anchor c (`v_c · dO`, or 0), `D - c`, and the anchor key's index (-1 where c is 0),
    from the first pass. D two ways: about 0, `sum_j p_j v_j · dO`, as the output's own product with dO
    rounds it, which suits a query whose weights spread over many keys; and about `c = v_c · dO`,
    `sum_{j != c} p_j (v_j - v_c) · dO`, where c enters as exactly 0, which suits one whose weight sits on
    c. Each query takes the one nearer its D, the one whose terms cancel less. The pass's sums are at each
    row's `2^-shift` (`room`), and each is taken over the total before its product with dO, as the
    reference's output is, so none overflows where the sums are near fp32's largest; the initial state's
    share is its weight over the total times its values' mean's product with dO (`mean0`)."""
    cotangent = dout.astype(jnp.float32)
    anchor_value = anchor_value.astype(jnp.float32)
    anchor = jnp.sum(anchor_value * cotangent, axis=-1)
    over = inverse(state.total)
    state_weight = power2(state_total, -shift) * over
    others = state.others * expand(over, state.others)
    own = expand(state.own * over, anchor_value)
    others_total = expand(state.others_total * over, anchor_value)
    about_zero = jnp.sum((others + own * anchor_value) * cotangent, axis=-1) + state_weight * mean0
    about_anchor = (jnp.sum((others - others_total * anchor_value) * cotangent, axis=-1)
                    + state_weight * (mean0 - anchor))
    return nearer_centre(anchor, about_anchor, about_zero, key_index)


def split_delta(others: jax.Array, own: jax.Array, others_total: jax.Array, total: jax.Array,
                state_total: jax.Array, anchor_value: jax.Array, dout: jax.Array,
                key_index: jax.Array, shift: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`off_anchor_delta` from a training forward's split sums (`training_step`), the backward's first
    pass taken by the forward: `B` with the initial state's share inside it, the other keys' weights'
    sum, the anchor's own weight, the denominator `total` and the initial state's own `state_total`, all
    at the forward's maximum. D's numerator about 0 is `(B + own v_c) · dO`, about `c = v_c · dO` it is
    `(B - others_total v_c) · dO - state_total c`, its terms kept apart so a state or keys that weigh
    little beside the anchor are not lost from a total; each query takes the one nearer its D. The split
    sums are at each row's `2^-shift` (`room`), and `total` is taken there; a row whose shift is negative
    holds normalized sums (`split_residuals`), its total 1. Each sum is taken over the total before its
    product with dO, as the reference's output is, so none overflows where the sums are near fp32's
    largest (a state's)."""
    over = jnp.where(shift < 0, 1.0, inverse(power2(total, -shift)))
    # The state's weight in one division: past 2^126 a total's inverse is subnormal, which flushes to 0,
    # and the state's share of D about the anchor with it (row 2466).
    state_weight = jnp.where(total > 0, state_total / jnp.where(total > 0, total, 1.0), 0.0)
    cotangent = dout.astype(jnp.float32)
    anchor_value = anchor_value.astype(jnp.float32)
    anchor = jnp.sum(anchor_value * cotangent, axis=-1)
    others = others * expand(over, others)
    own, others_total = own * over, others_total * over
    about_zero = jnp.sum(split_output(others, own, anchor_value) * cotangent, axis=-1)
    about_anchor = (jnp.sum((others - expand(others_total, anchor_value) * anchor_value) * cotangent, axis=-1)
                    - state_weight * anchor)
    return nearer_centre(anchor, about_anchor, about_zero, key_index)


def split_mass(own: jax.Array, others_total: jax.Array, total: jax.Array, shift: jax.Array) -> jax.Array:
    """The keys' share of a query's weights from a training forward's split sums (`split_delta`'s), the
    initial state's the rest: 0 exactly where no key weighs anything (`dq_finish`'s mass, for a backward
    with no dQ pass)."""
    over = jnp.where(shift < 0, 1.0, inverse(power2(total, -shift)))
    return (own + others_total) * over


class DqSums(NamedTuple):
    """A dQ pass's sums over its keys: dQ `[bq, D]` and the score cotangents' row sums and the HEAD and
    QUERY hook parameters' pullbacks of them (`grads`), each cotangent taken over a `prior` denominator
    (the forward's, or the first pass's; `grad_scores`); and the unnormalized weights' row sums, the hook
    parameters' pullbacks of them (`weighed`) and their product with the keys as dQ's, scaled and pulled
    through the hook over the prior denominator, held as its coefficients' magnitude
    `keyed_mass · 2^keyed_shift` (`[bq]` each, the mantissa at most the keys' count, the exponent the
    largest coefficient's) times a mean
    of the keys under coefficients of magnitudes summing to at most one, `keyed` `[bq, D]`: each finite
    wherever the keys are, whatever the scale and the hook's slopes, fp32."""

    dq: jax.Array
    residual: jax.Array
    mass: jax.Array
    grads: tuple[jax.Array, ...]
    weighed: tuple[jax.Array, ...]
    keyed: jax.Array
    keyed_mass: jax.Array
    keyed_shift: jax.Array


NO_SHIFT = -1024.0
"""The exponent of an empty sum of coefficients (`DqSums.keyed_shift`): below every composed coefficient's
(a weight's, a scale's and an inverse denominator's exponents together reach about -430), so any tile's
exponent is the larger, and `power2` of a zero mantissa by it stays zero."""


def no_dq_sums(rows: int, width: int, kinds: int) -> DqSums:
    """A dQ pass's sums before its first key (`DqSums`) over `rows` queries `width` wide and `kinds` HEAD and
    QUERY hook parameters: zeros, and an empty sum of coefficients (`NO_SHIFT`)."""
    nothing = jnp.zeros((rows,), jnp.float32)
    zeros = jnp.zeros((rows, width), jnp.float32)
    params = (nothing,) * kinds
    return DqSums(zeros, nothing, nothing, params, params, zeros, nothing,
                  jnp.full((rows,), NO_SHIFT, jnp.float32))


def _split(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Each entry of `x` as a mantissa of magnitude in (1/2, 1] and its power of two's exponent (0 for a
    zero), so a product of several taken mantissa by mantissa and exponent by exponent neither overflows
    nor flushes where the product itself is in range (`exponent_product`)."""
    exponent = jnp.where(x != 0, jnp.ceil(jnp.log2(jnp.where(x != 0, jnp.abs(x), 1.0))), 0.0)
    return power2(x, -exponent), exponent


def _static_split(x: float) -> tuple[np.float32, np.float32]:
    """`_split` of a static fp32 value, taken in Python, so a kernel holds it as constants (a mainloop's
    lowering may take no ceil or log2 of its own)."""
    value = np.float32(x)
    if value == 0:
        return value, np.float32(0.0)
    exponent = math.ceil(math.log2(abs(float(value))))
    # In Python's double, where 2^-exponent of a subnormal scale is no fp32 overflow.
    return np.float32(math.ldexp(float(value), -exponent)), np.float32(exponent)


def exponent_product(mantissa: jax.Array, exponent: jax.Array) -> jax.Array:
    """`mantissa * 2^exponent`, zero where the mantissa is, whatever the exponent (`power2` of a zero by an
    exponent past fp32's would multiply it by an infinite factor)."""
    return jnp.where(mantissa == 0, 0.0, power2(mantissa, exponent))


def dq_rows(sums: DqSums, weights: jax.Array, cotangents: jax.Array, slopes, seen: jax.Array | None, *,
             kinds: tuple, summed: tuple) -> tuple[DqSums, jax.Array]:
    """A dQ tile's row sums (`dq_step`): the HEAD and QUERY parameters' pullbacks of the cotangents and of
    the weights, the cotangents' and the weights' sums; and the weights pulled through the score hook."""
    _, pieces = pulled(cotangents, slopes, seen, kinds)
    grads = tuple(grad + piece if row else grad for grad, row, piece in zip(sums.grads, summed, pieces,
                                                                           strict=True))
    pulled_weights, pieces = pulled(weights, slopes, seen, kinds)
    weighed = sums.weighed
    if any(summed):
        weighed = tuple(total + piece if row else total
                        for total, row, piece in zip(sums.weighed, summed, pieces, strict=True))
    residual, mass = sums.residual + cotangents.sum(axis=-1), sums.mass + weights.sum(axis=-1)
    return sums._replace(residual=residual, mass=mass, grads=grads, weighed=weighed), pulled_weights


def dq_terms(sums: DqSums, weights: jax.Array, cotangents: jax.Array, slopes, seen: jax.Array | None, *,
             scale: float, kinds: tuple, summed: tuple) -> tuple[DqSums, jax.Array]:
    """`dq_step`'s row sums after a tile and dQ's score cotangents pulled and scaled, whose parts
    (`cotangent_parts`, `formed_parts`) a mainloop adds to dQ, into its accumulator in place where it can: the
    whole of the fast form's step, and the accurate form's with `keyed_terms` where it takes no score
    hook."""
    terms, _ = dq_rows(sums, weights, cotangents, slopes, seen, kinds=kinds, summed=summed)
    return terms, pulled(cotangents, slopes, seen, kinds)[0] * scale


def keyed_terms(sums: DqSums, weights: jax.Array, prior: jax.Array, *, scale: float, dtype,
                rows: bool = False) -> tuple[DqSums, Parts]:
    """The accurate form's keyed sums after a tile of a call without a score hook: their mass and exponent,
    and the parts of the weights' product with the keys (`weights_parts`, `DqSums.keyed`) a mainloop adds
    to them unscaled. Each coefficient is the scale times a weight over the prior denominator, which holds
    every weight at its maximum: the weights over it sum to at most one, so their product with the keys is a
    mean of them, finite wherever they are, and the scale is the mass's, its mantissa and exponent apart
    (`dq_finish`). One product, no exponent a row; a score hook's coefficients rescale a row's sums each
    tile instead (`dq_step`)."""
    scale_mantissa, scale_exponent = _static_split(scale)
    keyed = weights_parts(weights * expand(inverse(prior), weights), dtype, rows)
    return (sums._replace(keyed_mass=jnp.full_like(sums.keyed_mass, scale_mantissa),
                          keyed_shift=jnp.full_like(sums.keyed_shift, scale_exponent)), keyed)


def dq_issued(sums: DqSums, weights: jax.Array, cotangents: jax.Array, slopes, seen: jax.Array | None, dtype,
              prior: jax.Array, *, scale: float, kinds: tuple, summed: tuple, accurate: bool,
              rows: bool = False) -> tuple[DqSums, jax.Array, tuple[jax.Array | None, Parts] | None]:
    """A dQ pass's row sums after a key tile and the products a mainloop issues for it, for keys of `dtype`:
    the sums with their row fields updated (the cotangents' and the weights' sums, the summed parameters'
    pullbacks and, in the accurate form, the keyed mass and exponent); dQ's parts (`cotangent_parts`), whose
    products with the keys join dQ; and in the accurate form the keyed parts (`weights_parts`), whose products
    join the keyed sums behind the row factor `(factor, parts)` the sums take first (None: none); None in the
    fast form. `dq_step` takes these products as the Triton and plain-JAX mainloops do; a mainloop that issues
    them in place takes the same parts. dQ's come as its scaled cotangents, whose parts a mainloop forms
    (`cotangent_parts`) or forms a part at a time (`formed_parts`)."""
    if not accurate or slopes is None:
        terms, parts = dq_terms(sums, weights, cotangents, slopes, seen, scale=scale, kinds=kinds,
                                summed=summed)
        if not accurate:
            return terms, parts, None
        terms, keyed = keyed_terms(terms, weights, prior, scale=scale, dtype=dtype, rows=rows)
        return terms, parts, (None, keyed)
    terms, pulled_weights = dq_rows(sums, weights, cotangents, slopes, seen, kinds=kinds, summed=summed)
    parts = pulled(cotangents, slopes, seen, kinds)[0] * scale
    # The tile's coefficients against the keys, `pulled weight · scale / prior`, joined into the running
    # mean as a softmax's sums are joined under a running maximum. Each is held at its row's power of two,
    # never formed whole where it would overflow (a hook's slope near fp32's largest): the row's largest
    # weight's exponent plus the scale's and the prior's, its mantissa at most one; the running mass and
    # the tile's are brought to the larger exponent and the mean taken to their sum. One exponent a row,
    # not one an entry: an entry far below its row's largest is far below the correction's rounding.
    row_top = jnp.max(jnp.abs(pulled_weights), axis=-1)
    _, top_exponent = _split(row_top)
    scale_mantissa, scale_exponent = _static_split(scale)
    prior_mantissa, prior_exponent = _split(inverse(prior))
    tile_shift = jnp.where(row_top > 0, top_exponent + scale_exponent + prior_exponent, NO_SHIFT)
    keyed_shift = jnp.maximum(sums.keyed_shift, tile_shift)
    before = power2(sums.keyed_mass, sums.keyed_shift - keyed_shift)
    taken = expand(scale_exponent + prior_exponent - keyed_shift, pulled_weights)
    coefficients = power2(pulled_weights, taken) * expand(scale_mantissa * prior_mantissa, pulled_weights)
    keyed_mass = before + jnp.abs(coefficients).sum(axis=-1)
    over = inverse(keyed_mass)
    return (terms._replace(keyed_mass=keyed_mass, keyed_shift=keyed_shift), parts,
            (before * over, weights_parts(coefficients * expand(over, coefficients), dtype, rows)))


def dq_step(sums: DqSums, weights: jax.Array, cotangents: jax.Array, slopes, seen: jax.Array, key: jax.Array,
            prior: jax.Array, *, scale: float, kinds: tuple, summed: tuple,
            dot: Callable[[jax.Array, jax.Array], jax.Array], accurate: bool) -> DqSums:
    """A dQ pass's sums after a key tile (`DqSums`): its score cotangents pulled through the score hook
    (`pulled`) and scaled, times the keys (`cotangent_product`, in the form `accurate` names); the HEAD and
    QUERY parameters' pullbacks of them (`summed` marks which `kinds` sum per row) and of the weights,
    which take those gradients to the refined D (`dq_finish`); the weights' sums unnormalized; and in the
    accurate form the weights' product with the keys over the prior denominator, which takes dQ there too.
    The fast form leaves dQ at the first pass's D (the forward's anchored one, `split_delta`, or the dQ
    launch's own, `off_anchor_delta`), FlashAttention's arithmetic, its `keyed` sums untouched: a
    parameter's gradient sums its cotangents over every key and query, a position's or a scale's times
    each, so the first pass's D error enters it multiplied (ALiBi's slope 2.3 to 6.7 times the reference's
    error at its first D), where dQ's per query is FlashAttention's."""
    terms, parts, keyed = dq_issued(sums, weights, cotangents, slopes, seen, key.dtype, prior, scale=scale,
                                    kinds=kinds, summed=summed, accurate=accurate)
    terms = terms._replace(dq=sums.dq + cotangent_product(parts, key, dot, accurate=accurate))
    if keyed is None:
        return terms
    factor, keyed_parts = keyed
    scaled = sums.keyed if factor is None else expand(factor, sums.keyed) * sums.keyed
    return terms._replace(keyed=scaled + summed_products(keyed_parts, key, dot))


def dq_finish(sums: DqSums, prior: jax.Array, state_total: jax.Array, peak0: jax.Array,
              shares: tuple[jax.Array, jax.Array], peak: jax.Array, anchor: jax.Array, delta: jax.Array, *,
              accurate: bool):
    """A dQ pass's results from its sums, each taken to the pass's own denominator, the initial state's
    `state_total` and its keys' weights, so the weights it differentiates sum to one over its own logits
    whichever pass gave D: the cotangents' sums from the prior by their ratio (one division, about 1),
    the weights' over it. D is then refined once from its own sums (`refinement`), the D the dK/dV launch
    takes, and the HEAD and QUERY gradients take it, each less the refinement times its weights' own sum
    (`weighed`). In the accurate form dQ takes it too, less the refinement times its weights' product with
    the keys (`keyed`): a query's dQ cancels across its keys exactly as its cotangents sum to 0, so the D it
    takes must be the one whose cotangents do, not the first pass's; the fast form leaves dQ at the first
    pass's (`dq_step`). Returns dQ, the refined `D - c`, the denominator, the keys' share of the weights,
    and the hook gradients."""
    total = state_total + sums.mass
    ratio = jnp.where(total > 0, prior / jnp.where(total > 0, total, 1.0), 0.0)
    over = inverse(total)
    mass, residual = sums.mass * over, sums.residual * ratio
    refined = refinement(residual, mass, peak0, shares, peak, total, anchor, delta)
    grads = tuple(grad * ratio - refined * (weighed * over)
                  for grad, weighed in zip(sums.grads, sums.weighed, strict=True))
    if not accurate:
        return sums.dq * expand(ratio, sums.dq), delta + refined, total, mass, grads
    # refined · mass · 2^shift · keyed, each factor's mantissa and exponent apart (`_split`), so no partial
    # product overflows or flushes where the correction is in range.
    refined_mantissa, refined_exponent = _split(refined)
    keyed_mantissa, keyed_exponent = _split(sums.keyed)
    correction = exponent_product(
        expand(refined_mantissa * sums.keyed_mass, sums.keyed) * keyed_mantissa,
        expand(refined_exponent + sums.keyed_shift, sums.keyed) + keyed_exponent)
    dq = (sums.dq - correction) * expand(ratio, sums.dq)
    return dq, delta + refined, total, mass, grads


def safe_peak(peak: jax.Array) -> jax.Array:
    """A query's forward maximum as `exponent` reads it unclamped (`safe`): +inf where it saw no key (its
    -inf), so each of its weights is exp(-inf) = 0 whatever its logit, a row's select in place of each
    entry's clamp (the fast backward's dQ launch: FMNMX 128 to 64 of 2080 instructions, its dK/dV's 48 to
    32, at Qwen3-8B's widths, ptxas 12.9 sm90a)."""
    return jnp.where(peak == -jnp.inf, jnp.inf, peak)


def below(logits: jax.Array, peak: jax.Array, how: Recompute = PLAIN, *, safe: bool = False) -> jax.Array:
    """`exp(s - m)` for a tile's logits recomputed in the backward against the forward's maximum `m`,
    fp32, in the forward's units (`Recompute`): `m` is every key's maximum, so a logit past it by the
    rounding the two passes do not share (the forward's maximum taken in base 2 and back, or another tile
    shape's product) is the maximum itself, its weight 1, never past it; in the accurate form one at it
    weighs 1 however a compiler contracts its scaling into the subtraction (`weights`)."""
    return jnp.exp(exponent(logits, peak, how, safe=safe))


def normalized(logits: jax.Array, peak: jax.Array, inverse_total: jax.Array, how: Recompute = PLAIN, *,
               safe: bool = False) -> tuple[jax.Array, jax.Array]:
    """A tile's unnormalized weights `exp(s - m)` (`below`, unclamped where `safe`) and its weights `p`,
    those over each query's denominator (`inverse_total`, its `inverse`), fp32."""
    exponentials = below(logits, peak, how, safe=safe)
    return exponentials, exponentials * expand(inverse_total, exponentials)


def dropped_products(products: jax.Array, multiplier: jax.Array | None) -> jax.Array:
    """A tile's score products as a dropping call takes them, `products * multiplier`, each rounded once
    before it is centred: fused into its subtraction (a contracted multiply-add) it would round differently
    against a centre that rounded it, so every centre and cotangent takes this one rounding."""
    if multiplier is None:
        return products
    return jax.lax.optimization_barrier(products * multiplier)


def grad_scores(exponentials: jax.Array, products: jax.Array, anchor: jax.Array, delta: jax.Array,
                seen: jax.Array | None, at_anchor: jax.Array | None, inverse_total: jax.Array,
                multiplier: jax.Array | None = None, weights: jax.Array | None = None) -> jax.Array:
    """The score cotangents `p (dP - D)` of a tile in the reference's own order, as JAX differentiates
    `exp(s - m) / T`: each query's `(dP - c) - (D - c)` about its anchor c (`off_anchor_delta`) over its
    denominator (`inverse_total`), then times the unnormalized weights `exp(s - m)`, so every intermediate
    lies where the reference's does (a weight below fp32's range keeps its share beside a large product, and
    none overflows that the reference keeps); zero where unseen (`where_seen`). The anchor key's own
    `dP - c` is 0 by definition where `at_anchor` names it, whatever its product's rounding, so a query that
    sees one key has no cotangent, as the reference's: FlashAttention's order, `(dP - D) / T` with no anchor
    test, fails check 1 there in both dQ and dK (a query of one key, the stability suite's ordinary "one
    key" and causal windows' first queries: rows whose float64 gradient is exactly 0). That is the accurate
    form's. The fast form's (`weights` given, the normalized weights its launch forms anyway) is
    FlashAttention's order about the anchor, `weights ((dP - c) - (D - c))`, its anchor test the accurate
    form's alone (`at_anchor` None)."""
    products = dropped_products(products, multiplier)
    relative = products - expand(anchor, products)
    if at_anchor is not None:
        relative = jnp.where(at_anchor, 0.0, relative)
    if weights is not None:
        # The weights a launch forms anyway times `(dP - c) - (D - c)`: one product fewer an entry (the fast
        # backward's dK/dV launch: FMUL 112 to 96 at Qwen3-8B's widths, ptxas 12.9 sm90a).
        return where_seen(seen, weights * (relative - expand(delta, relative)), 0.0)
    centred = (relative - expand(delta, relative)) * expand(inverse_total, relative)
    return where_seen(seen, exponentials * centred, 0.0)


def refinement(residual: jax.Array, mass: jax.Array, peak0: jax.Array, shares: tuple[jax.Array, jax.Array],
               peak: jax.Array, total: jax.Array, anchor: jax.Array, delta: jax.Array) -> jax.Array:
    """D's one refinement from the main pass's own sums: the score cotangents of a query and of its
    initial state (`shares`: its denominator and its values' mean's term at the state's maximum `peak0`,
    `off_anchor_start`) sum to 0 by the softmax's shift, and what they sum to here (`residual`, over the
    keys' weights `mass`) is D's rounding. Moving D by it over the weights' total lets dK/dV, the initial
    state's gradients and the head and query parameters' take the D whose cotangents do. The state's
    weight is taken whole before any product, so a state whose sums are near fp32's largest stays finite."""
    carried = jnp.where(peak0 == -jnp.inf, 0.0, jnp.exp(jnp.minimum(peak0 - peak, 0.0)))
    weight = (shares[0] * carried) * inverse(total)
    whole_residual = residual + (shares[1] - (anchor + delta)) * weight
    whole_mass = mass + weight
    return jnp.where(whole_mass > 0, whole_residual / jnp.where(whole_mass > 0, whole_mass, 1.0), 0.0)


class Centre(NamedTuple):
    """Dropout's first pass over a query's keys (`centre_step`): `centre`, the least-magnitude product in
    the query's first visible tile, the shift each dropped product is centred about; `seen`, whether a key
    is visible yet; and the centred running sums (`running_sum`) at the scale `centre_scale` gives:
    `residual`, the weights' sum of the products less the centre, and `mass`, the weights' sum. A product
    equal to the centre contributes exactly zero, so identical products cancel before any reduction, and
    the sums stay finite for finite products."""

    centre: jax.Array
    seen: jax.Array
    residual: Running
    mass: Running


def centre_scale(keys: int) -> float:
    """The exponent `centre_step` takes products at, `-(2 + ceil(log2 keys))`. A centred product is at most
    twice the largest product's magnitude, and a sum of `keys` weighted ones at most `2 keys` times it, so
    `2^-k >= 2 keys` keeps every sum, the state's share included, below the largest for any finite input."""
    return -(2.0 + math.ceil(math.log2(max(keys, 1))))


def centre_start(rows: int) -> Centre:
    """The centre of `rows` queries that have seen no key."""
    zero = jnp.zeros((rows,), jnp.float32)
    return Centre(zero, jnp.zeros((rows,), bool), running_sum((rows,), jnp.float32),
                  running_sum((rows,), jnp.float32))


def centre_step(state: Centre, weights: jax.Array, products: jax.Array, seen: jax.Array, positions: jax.Array,
                exponent: float) -> Centre:
    """A key tile's part of dropout's centre (`Centre`): `weights` `[bq, bk]` (unnormalized, `exp(s - m)`),
    the tile's dropped `products` (`dropped_products`), `seen` where a key counts, and each entry's key index
    `positions` (broadcasting). A query's centre is its first visible tile's product nearest zero, ties
    broken by position, selected exactly: the sum of one nonzero term and zeros is that term. D is the
    centre plus the weighted mean about it whatever the centre, and the least one keeps the mean's two
    rounded sums from carrying a large common offset that their quotient cancels back (a centre of -21.3
    left D 2.8e-6 from float64's at -2.43); identical products still subtract exactly. The sums run as
    the loop's running sums (`added_to`), each tile's rows summed compensated where loops are
    (`compensates_loops`). Each product is taken at `2^exponent` before the subtraction; no difference or
    sum overflows, and a power of two is exact where the product is normal."""
    # The scale a Python float, exact (`centre_scale`'s exponent is within fp32's normal range), so no
    # scalar arithmetic runs in a kernel (Mosaic GPU holds no rank-0 vector).
    factor = 2.0 ** exponent
    nearest = jnp.min(jnp.where(seen, jnp.abs(products), jnp.inf), axis=-1)
    first = jnp.min(jnp.where(seen & (jnp.abs(products) == expand(nearest, products)), positions,
                              jnp.iinfo(jnp.int32).max), axis=-1)
    picked = seen & (positions == expand(first, products))
    tile_centre = jnp.sum(jnp.where(picked, products, 0.0), axis=-1)
    centre = jnp.where(state.seen, state.centre, tile_centre)
    # Whether a key of the tile counts, a row's maximum (a kernel's rows reduce no booleans).
    counted = jnp.max(jnp.where(seen, 1, 0), axis=-1) > 0
    scaled = products * factor - expand(centre * factor, products)

    def row_sum(values):
        return (compensated_sum(values, (values.ndim - 1,)) if compensates_loops()
                else values.sum(axis=-1))

    return Centre(centre, state.seen | counted,
                  added_to(state.residual, row_sum(jnp.where(seen, weights * scaled, 0.0))),
                  added_to(state.mass, row_sum(jnp.where(seen, weights, 0.0))))


def centre_finish(state: Centre, state_total: jax.Array, initial: jax.Array, exponent: float) -> jax.Array:
    """Dropout's D from its centre: the centre plus the mean of the centred products over the weights, with
    the initial state's share (`state_total` its weight, `initial` its values' mean's product with the
    cotangent) centred the same way. The mean is taken at the scale, so no sum overflows: D is
    `centre + mean` where that is finite, the scaled sum otherwise, and 0 where a row saw no key and has
    no initial state. Where the products agree the mean is exactly zero, so D is their product. The keys'
    sum and the state's share are each taken over the total before they add: the state's weight over the
    total is at most 1, so its centred value is never multiplied by a denominator that overflows it."""
    k = jnp.asarray(exponent, jnp.float32)
    scaled_centre = power2(state.centre, k)
    scaled_initial = power2(initial, k) - scaled_centre
    total = running_total(state.mass) + state_total
    divisor = jnp.where(total > 0, total, 1.0)
    mean = jnp.where(total > 0, running_total(state.residual) / divisor
                     + (state_total / divisor) * scaled_initial, 0.0)
    shifted = state.centre + power2(mean, -k)
    scaled = power2(scaled_centre + mean, -k)
    return jnp.where(total > 0, jnp.where(finite(shifted), shifted, scaled), 0.0)


def dropout_centre(lo, hi, terms: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array, jax.Array]],
                   keys: int, rows: int, state_total: jax.Array, initial: jax.Array
                   ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Dropout's first pass over a query block (`Centre`): `terms(j)` gives key tile `j`'s unnormalized
    weights `exp(s - m)`, its dropped products (`dropped_products`), where its entries are seen, and their
    key indices. Each product is centred about its query's least-magnitude product in its first visible
    tile and scaled before any sum (`centre_step`), and D is rebuilt from that centre (`centre_finish`).
    Returns the second sweep's `(anchor, delta, chosen)`: no anchor, D about zero, and no key chosen, so
    identical products cancel exactly in its cotangents."""
    exponent = centre_scale(keys)

    def step(j, state):
        weights, products, seen, positions = terms(j)
        return centre_step(state, weights, products, seen, positions, exponent)

    centred = jax.lax.fori_loop(lo, hi, step, centre_start(rows))
    return (jnp.zeros((rows,), jnp.float32), centre_finish(centred, state_total, initial, exponent),
            jnp.full((rows,), -1, jnp.int32))
