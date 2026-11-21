"""The attention backward's tile algebra, what lies between its products (S = Q·Kᵀ, dP = dO·Vᵀ, then dV, dK
and dQ) on natural logits: a tile's logits recomputed under the score hook and their derivatives
(`tile_logits`, `scored`, `pulled`), its weights and score cotangents
(`attention_backward_algebra.softmax_cotangents`), each query's dQ sums about its anchor key (`DqSums`,
`dq_issued`, `dq_finish`), the anchors themselves (`off_anchor`, `centre_step`) and the dropout's
(`dropped_products`). The forward's algebra and the products' operand parts every launch shares are
`attention_algebra`'s; the declared backward and its derived forms `attention_backward_algebra`'s."""

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
    anchor_split,
    expand,
    exponent,
    finite,
    hooked,
    inverse,
    normalized_rows,
    split_output,
    tiles_walked,
    weights_parts,
)
from dew.nn.precision import (
    Running,
    added_to,
    compensated_sum,
    compensates_loops,
    power2,
    running_sum,
    running_total,
    two_sum,
    uncontracted,
)


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
    forward's `maximum`, its sums at each row's `2^-shift` (the forward's scale, `Scale`), and that state's
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
    `2^-min(shift, WEIGHT_SHIFT)`; them narrowed to the values' `dtype` as the forward rounds
    them, zero at the anchor key (`at_anchor`), the left operand of `B`'s product; the anchor key's
    weight as it is, fp32, as the forward keeps it (`algebra.anchor_split`); and the weights that shift
    would push below fp32's least normal, unshifted and narrowed, whose product the mainloop issues apart
    (`off_anchor`), as the forward's small weights' (`at_scale`)."""
    exponentials = below(logits, maximum, how)
    weight = expand(jnp.minimum(shift, WEIGHT_SHIFT), exponentials)
    small = jnp.where(exponentials < power2(jnp.float32(1.0), weight - 126.0), exponentials, 0.0)
    exponentials = power2(exponentials - small, -weight)
    rest, own = anchor_split(exponentials, at_anchor, dtype)
    return exponentials, rest, own, small.astype(dtype)


def off_anchor(state: OffAnchor, exponentials: jax.Array, rest: jax.Array, own: jax.Array,
               others: jax.Array, shift: jax.Array, small: jax.Array, small_others: jax.Array) -> OffAnchor:
    """The first pass after a tile (`off_anchor_weights`), `others` the tile's `rest · V`, its sums taken
    from the weights' shift to each row's `shift`, as the forward's (`carry_factor`), and the small
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


def summed_rows(a: jax.Array, b: jax.Array) -> jax.Array:
    """Each row's `a · b` over the last axis, as a kernel's body sums it: D's and the anchor's products with
    dO where the body's issuer forms dP (`off_anchor_delta`'s `rows`)."""
    return jnp.sum(a * b, axis=-1)


def read(sums: jax.Array, cotangent: jax.Array) -> jax.Array:
    """A row's `sums` as its cotangent reads them for D (`split_delta`, `off_anchor_delta`): nothing of a
    column the cotangent is exactly 0 in. D is `sum_j p_j (v_j · dO)`, each key's product its own, so a
    column an unseen key's infinite value poisons in the sums (its weight 0 times the value, a NaN in the
    forward's product) adds nothing where dO reads none of it, as the reference's D: the derived forms'
    gradients at fp16's largest were NaN where the reference's are finite."""
    return jnp.where(cotangent == 0, 0.0, sums)


def nearer_centre(anchor: jax.Array, about_anchor: jax.Array, about_zero: jax.Array, key_index: jax.Array
                  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """D's centre, the nearer to D of the anchor key's `dP` (`anchor`) and zero, each sum's difference from
    it (`about_anchor`, `about_zero`): the centre, D less it, and the anchor key's index (`key_index`), -1
    where zero is the nearer."""
    anchored = jnp.abs(about_anchor) <= jnp.abs(about_zero)
    return (jnp.where(anchored, anchor, 0.0), jnp.where(anchored, about_anchor, about_zero),
            jnp.where(anchored, key_index, -1))


def off_anchor_delta(state: OffAnchor, anchor_value: jax.Array, dout: jax.Array, state_total: jax.Array,
                     mean0: jax.Array, key_index: jax.Array, shift: jax.Array, *,
                     rows: Callable[[jax.Array, jax.Array], jax.Array]
                     ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Each query's anchor c (`v_c · dO`, or 0), `D - c`, and the anchor key's index (-1 where c is 0),
    from the first pass. D two ways: about 0, `sum_j p_j v_j · dO`, as the output's own product with dO
    rounds it, which suits a query whose weights spread over many keys; and about `c = v_c · dO`,
    `sum_{j != c} p_j (v_j - v_c) · dO`, where c enters as exactly 0, which suits one whose weight sits on
    c. Each query takes the one nearer its D, the one whose terms cancel less. The pass's sums are at each
    row's `2^-shift` (`Scale`), and each is taken over the total before its product with dO, as the
    reference's output is, so none overflows where the sums are near fp32's largest; the initial state's
    share is its weight over the total times its values' mean's product with dO (`mean0`). Each row's
    product with dO is `rows`', the issuer's own that forms dP (`summed_rows` in a kernel's body,
    `interpreted_rows` off a GPU), so c and the anchor key's dP sum alike and cancel exactly; the sums as dO
    reads them (`read`)."""
    cotangent = dout.astype(jnp.float32)
    anchor_value = anchor_value.astype(jnp.float32)
    anchor = rows(anchor_value, cotangent)
    over = inverse(state.total)
    state_weight = power2(state_total, -shift) * over
    others = state.others * expand(over, state.others)
    own = expand(state.own * over, anchor_value)
    others_total = expand(state.others_total * over, anchor_value)
    about_zero = rows(read(others + own * anchor_value, cotangent), cotangent) + state_weight * mean0
    about_anchor = (rows(read(others - others_total * anchor_value, cotangent), cotangent)
                    + state_weight * (mean0 - anchor))
    return nearer_centre(anchor, about_anchor, about_zero, key_index)


def split_delta(others: jax.Array, own: jax.Array, others_total: jax.Array, total: jax.Array,
                state_total: jax.Array, anchor_value: jax.Array, dout: jax.Array,
                key_index: jax.Array, shift: jax.Array, *,
                rows: Callable[[jax.Array, jax.Array], jax.Array], dropped: float = 1.0, power: float = 1.0
                ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`off_anchor_delta` from a training forward's split sums (`training_step`), the backward's first
    pass taken by the forward: `B` with the initial state's share inside it, the other keys' weights'
    sum, the anchor's own weight, the denominator `total` and the initial state's own `state_total`, all
    at the forward's maximum. D's numerator about 0 is `(B + own v_c) · dO`, about `c = v_c · dO` it is
    `(B - others_total v_c) · dO - state_total c`, its terms kept apart so a state or keys that weigh
    little beside the anchor are not lost from a total; each query takes the one nearer its D. The split
    sums are at each row's `2^-shift` (`Scale`), and `total` is taken there; a row at the shift
    `NORMALIZED` holds normalized sums (an output residual's, `normalized_rows`), its total 1. Each sum is
    taken over the total before its product with dO, as the reference's output is, so none overflows where
    the sums are near fp32's largest (a state's). A dropping call's `B` holds its kept keys' weights alone
    and its anchor's value row arrives where its draw keeps it (`algebra.kept`), its sums at the
    multiplier's reach `power` (`algebra.Forward.units`): its D and centre are taken without the
    draw's multiplier `dropped`, which multiplies them once (`algebra.Forward.dropped`), as nearer centre is
    chosen alike at any scale; a normalized row's sums (an output's) took it already. Taken
    with each weight over the total, sums near fp32's largest times 4/3 overflowed where D is finite (row
    8652). Each row's product with dO is `rows`' (`off_anchor_delta`'s), of the sums as dO reads them
    (`read`)."""
    multiplier = np.float32(dropped)
    over = jnp.where(normalized_rows(shift), 1.0, inverse(power2(total, -shift)))
    # The state's weight in one division: past 2^126 a total's inverse is subnormal, which flushes to 0,
    # and the state's share of D about the anchor with it (row 2466).
    state_weight = jnp.where(total > 0, state_total / jnp.where(total > 0, total, 1.0), 0.0)
    cotangent = dout.astype(jnp.float32)
    anchor_value = anchor_value.astype(jnp.float32)
    mean = others * expand(over, others)

    def centred(others: jax.Array, unit) -> tuple[jax.Array, jax.Array, jax.Array]:
        # D's centre and numerators about it and about 0, each term in the units `unit` of `others`, the
        # anchor's product scaled on whichever factor cannot overflow `unit` (row 9096: 2^80 * 2^53 -> inf
        # before a zero or tiny dO, while 2^-126 * 2^-40 rounded to 0 before 2^53, 6a57b66de F1).
        own_held, others_held = own * over * unit, others_total * over * unit
        if isinstance(unit, float):
            anchor = rows(read(anchor_value, cotangent), cotangent)
        else:
            scale = expand(unit, anchor_value)
            small = jnp.abs(anchor_value) < 1.0
            lhs = jnp.where(small, anchor_value * scale, anchor_value)
            rhs = jnp.where(small, cotangent, cotangent * scale)
            anchor = rows(read(lhs, rhs), rhs)
        others_dot = rows(read(others, cotangent), cotangent)
        about_zero_combined = rows(read(split_output(others, own_held, anchor_value), cotangent), cotangent)
        about_zero = jnp.where(finite(about_zero_combined), about_zero_combined,
                               others_dot + (own * over) * anchor)
        off_anchor_combined = rows(
            read(others - expand(others_held, anchor_value) * anchor_value, cotangent), cotangent)
        off_anchor = jnp.where(finite(off_anchor_combined), off_anchor_combined,
                               others_dot - (others_total * over) * anchor)
        return anchor, off_anchor - state_weight * anchor, about_zero

    if dropped == 1.0:
        return nearer_centre(*centred(mean, 1.0), key_index)
    # A dropping call's: in the units its sums hold (its power, `algebra.Forward.units`; a normalized row's,
    # an output's, the multiplier), contracted with dO before they come out, so a mean they take below fp32's
    # least subnormal keeps its bits (2^-126 at a weight of e^-22 and a power of 2^53, dQ 0 where it is
    # -2.79e-10, the base review of eb3ab2670, F1; a normalized row's divided by 2^53 first, the range audit
    # of row 8652, #1); where those pass fp32's largest, the units taken out first, after the total where the
    # mean is finite, else in one division with it (18 2^126 over 4, the base review of e2b671ce9, F2).
    held = jnp.where(normalized_rows(shift), multiplier, np.float32(power))
    divisor = jnp.where(normalized_rows(shift), multiplier, power2(total, -shift) * np.float32(power))
    divided = others / expand(jnp.where(finite(divisor) & (divisor > 0), divisor, 1.0), others)
    in_units = centred(mean, held)
    taken = jnp.where(normalized_rows(shift), np.float32(1 / dropped), np.float32(1 / power))
    plain = centred(jnp.where(finite(mean), mean * expand(taken, mean), divided), 1.0)
    kept = finite(in_units[0]) & finite(in_units[1]) & finite(in_units[2])
    centre, about_anchor, about_zero = (jnp.where(kept, held_term, plain_term)
                                        for held_term, plain_term in zip(in_units, plain, strict=True))
    anchor, delta, chosen = nearer_centre(centre, about_anchor, about_zero, key_index)
    factor = jnp.where(kept, multiplier / held, multiplier)
    return anchor * factor, delta * factor, chosen


def split_mass(own: jax.Array, others_total: jax.Array, total: jax.Array, shift: jax.Array) -> jax.Array:
    """The keys' share of a query's weights from a training forward's split sums (`split_delta`'s), the
    initial state's the rest: 0 exactly where no key weighs anything (`dq_finish`'s mass, for a backward
    with no dQ pass)."""
    over = jnp.where(normalized_rows(shift), 1.0, inverse(power2(total, -shift)))
    return (own + others_total) * over


class DqSums(NamedTuple):
    """A dQ pass's sums over its keys: dQ `[bq, D]` and the score cotangents' row sums and the HEAD and QUERY
    hook parameters' pullbacks of them (`grads`), each cotangent taken over a `prior` denominator (the
    forward's, or the first pass's; `attention_backward_algebra.softmax_cotangents`); and the unnormalized
    weights' row sums, the hook parameters' pullbacks of them (`weighed`) and their product with the keys as
    dQ's, scaled and pulled through the hook over the prior denominator, held as its coefficients' magnitude
    `keyed_mass · 2^keyed_shift` (`[bq]` each, the mantissa at most the keys' count, the exponent the largest
    coefficient's) times a mean of the keys under coefficients of magnitudes summing to at most one, `keyed`
    `[bq, D]`: each finite wherever the keys are, whatever the scale and the hook's slopes, fp32."""

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
    """A dQ tile's row sums (`dq_issued`): the HEAD and QUERY parameters' pullbacks of the cotangents and of
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
    """`dq_issued`'s row sums after a tile and dQ's score cotangents pulled and scaled, whose parts
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
    tile instead (`dq_issued`)."""
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
    fast form. Every mainloop issues these products (`attention_program.dq_sums`, the Mosaic dQ launch), in
    place or by its own dot. dQ's come as its scaled cotangents, whose parts a mainloop forms
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
    pass's (`dq_issued`). Returns dQ, the refined `D - c` rounded and what its rounding lost (`two_sum`: a
    refinement below the rounded D's ulp, which an initial state's maximum's gradient reads, row 6302), the
    denominator, the keys' share of the weights, and the hook gradients."""
    total = state_total + sums.mass
    ratio = jnp.where(total > 0, prior / jnp.where(total > 0, total, 1.0), 0.0)
    over = inverse(total)
    mass, residual = sums.mass * over, sums.residual * ratio
    refined = refinement(residual, mass, peak0, shares, peak, total, anchor, delta)
    grads = tuple(grad * ratio - refined * (weighed * over)
                  for grad, weighed in zip(sums.grads, sums.weighed, strict=True))
    refined_delta, lost = two_sum(delta, refined)
    if not accurate:
        return sums.dq * expand(ratio, sums.dq), refined_delta, lost, total, mass, grads
    # refined · mass · 2^shift · keyed, each factor's mantissa and exponent apart (`_split`), so no partial
    # product overflows or flushes where the correction is in range.
    refined_mantissa, refined_exponent = _split(refined)
    keyed_mantissa, keyed_exponent = _split(sums.keyed)
    correction = exponent_product(
        expand(refined_mantissa * sums.keyed_mass, sums.keyed) * keyed_mantissa,
        expand(refined_exponent + sums.keyed_shift, sums.keyed) + keyed_exponent)
    dq = (sums.dq - correction) * expand(ratio, sums.dq)
    return dq, refined_delta, lost, total, mass, grads


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
    (dropped,) = uncontracted(products * multiplier)
    return dropped


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
    """Dropout's first pass over a query's keys (`centre_step`): `centre`, the dropped product at the
    forward's anchor, the row's heaviest key; and the running sums (`running_sum`) at the scale
    `centre_scale` gives: `residual`, the weights' sum of the products less the centre, `zero`, of the
    products themselves, and `mass`, the weights' sum. About the centre the heaviest key's term is exactly
    zero and identical products cancel before any reduction, so a query whose weight one key holds takes D
    as that key's product; about zero a query whose weights spread over many keys keeps its products' own
    rounding; each query takes the nearer (`nearer_centre`). The sums stay finite for finite products."""

    centre: jax.Array
    residual: Running
    zero: Running
    mass: Running


def centre_scale(keys: int) -> float:
    """The exponent `centre_step` takes products at, `-(2 + ceil(log2 keys))`. A centred product is at most
    twice the largest product's magnitude, and a sum of `keys` weighted ones at most `2 keys` times it, so
    `2^-k >= 2 keys` keeps every sum, the state's share included, below the largest for any finite input."""
    return -(2.0 + math.ceil(math.log2(max(keys, 1))))


def centre_start(centre: jax.Array) -> Centre:
    """The centre of queries that have seen no key, about `centre` `[rows]` (`Centre.centre`)."""
    return Centre(centre, *(running_sum(centre.shape, jnp.float32) for _ in range(3)))


def centre_step(state: Centre, weights: jax.Array, products: jax.Array, seen: jax.Array,
                at_anchor: jax.Array, exponent: float) -> Centre:
    """A key tile's part of dropout's centre (`Centre`): `weights` `[bq, bk]` (unnormalized, `exp(s - m)`),
    the tile's dropped `products` (`dropped_products`) and `seen` where a key counts, each product taken at
    `2^exponent`, about the row's anchor's and about zero, before any sum, so no difference or sum overflows
    and a power of two is exact where the product is normal. About a lighter key's product the heaviest's
    term rounded at the scale of the products (a product near 2^123 against a key near 2^30 made dQ infinite
    where it is 1e-127, row 8652), and about the first tile's heaviest a later tile's dominant one was lost
    (the base review of 4e2eb287e, F4). The sums run as the loop's running sums (`added_to`), each tile's
    rows summed compensated where loops are (`compensates_loops`). The anchor's own centred product is
    zero, as in the score cotangents, even when the separate row and tile contractions round it
    differently."""
    # The scale a Python float, exact (`centre_scale`'s exponent is within fp32's normal range), so no
    # scalar arithmetic runs in a kernel (Mosaic GPU holds no rank-0 vector).
    factor = 2.0 ** exponent
    scaled = products * factor

    def row_sum(values):
        values = jnp.where(seen, values, 0.0)
        return compensated_sum(values, (values.ndim - 1,)) if compensates_loops() else values.sum(axis=-1)

    centred = jnp.where(at_anchor, 0.0, weights * (scaled - expand(state.centre * factor, scaled)))
    return Centre(state.centre, added_to(state.residual, row_sum(centred)),
                  added_to(state.zero, row_sum(weights * scaled)), added_to(state.mass, row_sum(weights)))


def centre_finish(state: Centre, state_total: jax.Array, initial: jax.Array, exponent: float,
                  key_index: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Dropout's D from its first pass (`Centre`) as the second sweep takes it, `(anchor, delta, chosen)`:
    about the nearer of the centre and zero (`nearer_centre`), the centre, D less it and the anchor's key
    `key_index`, or zero, D and -1. Each mean is its sum over the weights with the initial state's share
    (`state_total` its weight, `initial` its values' mean's product with the cotangent) centred the same
    way, at the scale, the keys' sum and the state's share each over the total before they add (the state's
    weight over the total at most 1), taken back where finite; 0 where a row saw no key and has no initial
    state. D less the centre is passed apart, never added back into it: a residual of 3.9e-9 beside a centre
    of 4/3 was lost (the base review of a648b3279, F4)."""
    k = jnp.asarray(exponent, jnp.float32)
    total = running_total(state.mass) + state_total
    divisor = jnp.where(total > 0, total, 1.0)
    share = state_total / divisor

    def mean(summed: Running, centre: jax.Array) -> jax.Array:
        # Taken back in the division where the total holds it (a scaled mean below fp32's least normal lost
        # bits, a state of 2^126 beside products of 1), at the scale where it does not.
        direct = running_total(summed) / power2(divisor, k) + share * (initial - centre)
        scaled = running_total(summed) / divisor + share * (power2(initial, k) - power2(centre, k))
        scaled = power2(scaled, -k)
        return jnp.where(total > 0, jnp.where(finite(direct), direct, scaled), 0.0)

    about_anchor, about_zero = mean(state.residual, state.centre), mean(state.zero, jnp.zeros_like(total))
    # A residual past fp32's largest about the centre takes zero's, which holds D whole.
    return nearer_centre(state.centre, jnp.where(finite(about_anchor), about_anchor, jnp.inf), about_zero,
                         jnp.where(total > 0, key_index, -1))


def dropout_centre(lo, hi, terms: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array, jax.Array]],
                   keys: int, centre: jax.Array, key_index: jax.Array, state_total: jax.Array,
                   initial: jax.Array, *,
                   most: int | None = None) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Dropout's first pass over a query block (`Centre`): `terms(j)` gives key tile `j`'s unnormalized
    weights `exp(s - m)`, its dropped products (`dropped_products`), its visibility and its key indices.
    Each product is taken about its query's `centre` (the dropped product at the forward's anchor key
    `key_index`) and about zero, scaled, before any sum (`centre_step`), and D is rebuilt about the nearer
    (`centre_finish`): the second sweep's `(anchor, delta, chosen)`, identical products cancelling exactly in
    its cotangents. Its tiles at most `most` (`tiles_walked`)."""
    exponent = centre_scale(keys)

    def step(j, state):
        weights, products, seen, columns = terms(j)
        return centre_step(state, weights, products, seen, columns == key_index[:, None], exponent)

    centred = tiles_walked(lo, hi, step, centre_start(centre), most)
    return centre_finish(centred, state_total, initial, exponent, key_index)
