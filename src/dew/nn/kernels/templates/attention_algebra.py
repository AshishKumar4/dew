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

The functions specialize at trace time on what the call holds (`Forward`): the logits fold log2(e) into
the scale for exp2 where the scale allows it, unless the call keeps them natural (`Forward.base2`); a
full or causal call's mask comes from its coordinates alone, only spans read intervals and ids, and a
tile no query's interval cuts takes no mask at all (`masked`); and only a call where a query can see
nothing (spans, a score hook) guards its maxima and denominators against an empty row."""

import dataclasses
import functools
import math
from collections.abc import Callable, Iterator
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.kernels.templates.attention import (
    LOG2E,
    NATURAL,
    Draw,
    Kind,
    differentiable,
    floating_params,
    within,
)
from dew.nn.precision import (
    added_to,
    compensated_sum,
    compensates_loops,
    running_sum,
    running_total,
    two_product,
    two_sum,
)


@dataclasses.dataclass(frozen=True)
class Forward:
    """What the forward's functions specialize on, all static: the spans' `form` ('full', 'causal' or
    'spans', `KeySpans.form`), the call's key count `keys`, whether it packs documents (`ids`), the
    softmax `scale`, the score hook `score_mod(s, q_pos, k_pos, *params)` on natural logits, whether
    the call keeps its logits `natural` (`base2`), the accurate form, whether its weighted values can pass
    fp32's largest, so a program's walk is `guarded` (`headroom`), whether its softmax resumes from an
    `initial` state (`raw`), whether natural logits stay `scaled` whatever the scale (`raw`; the fold's
    rewalk, `unfolded`), the call's query count `queries` where a mainloop walks query rows past it
    without spans (a full or causal call padded to whole tiles or whole programs, whose padding rows, a
    tile's last ones or a whole tile of them, `seen` keeps from every key; None where none does),
    whether its walks keep what their running sums' additions drop (`compensated`, `compensates`), and the
    power of two the queries took before their products (`prescale`, `attention.prescale`), which `scale`
    is the call's scale over."""

    form: str
    keys: int
    ids: bool
    scale: float
    score_mod: Callable | None = None
    natural: bool = NATURAL
    headroom: bool = False
    initial: bool = False
    scaled: bool = False
    queries: int | None = None
    compensated: bool = False
    prescale: float = 1.0

    @property
    def raw(self) -> bool:
        """Whether the accurate form (`natural`) keeps its logits as the raw products, each exponent a
        difference of them scaled after it is taken, `(s - m) * scale * log2(e)`: no multiply precedes the
        subtraction for a compiler to contract into it, so the largest logit's weight is 1 exactly and
        every difference rounds once. Where a score hook reads the scaled logits, or an initial state's
        natural maximum would have to be divided into raw units, or the scale is not positive (whose raw
        maximum is not the scaled one's), the logits are scaled, the largest's weight held to 1 by a select
        (`weights`). The backward recomputes in the same units (`below`)."""
        low, high = RAW_SCALES
        return (self.natural and not self.scaled and self.score_mod is None and not self.initial
                and low <= self.scale <= high)

    @property
    def base2(self) -> bool:
        """Whether the logits fold log2(e) into the scale for exp2, the fast form and the default
        (FlashAttention 2's and jax's mha's arithmetic): unless the call keeps them `natural`, where no
        hook reads them and the scale times log2(e) is below 1 in magnitude, so the fold takes no logit the
        reference keeps finite past fp32's largest, of either sign (a negative scale of magnitude 1 takes a
        finite -3e38 to -inf, its weight NaN), and every folded logit is at least an ulp of fp32's largest
        below it (`base2_maximum`). Otherwise they stay natural, each exponential taken
        of their difference (`natural_exp`), as the reference's. The fold rounds each logit at log2(e)
        times its size, so logits the reference keeps apart can tie: keys of 2 * 1610613120 and 2 * 1610612992
        under a query of 1 at scale 0.5 weigh 1 and 0 in the reference and 0.5 each folded, and a
        cotangent of 2**125 then overflows its backward (docs/design/kernel-templates.md, §3, the
        exponent's forms). On drawn inputs both forms pass check 1."""
        # The factor as the kernel would multiply the call's own products by it, rounded to fp32: the call's
        # scale, its products' power of two (`prescale`) times the scale left after it.
        factor = abs(float(np.float32(self.scale * self.prescale * LOG2E)))
        return not self.natural and self.score_mod is None and factor < 1.0

    @property
    def factor(self) -> float:
        """The factor the logits are the products times (`modify`): 1 raw, the scale times log2(e) folded,
        the scale natural, rounded to fp32 as the kernels multiply by it (a scale below fp32's range is 0)."""
        return float(np.float32(1.0 if self.raw else self.scale * LOG2E if self.base2 else self.scale))

    @property
    def guarded(self) -> bool:
        """Whether a query can see no key, so its maxima and denominators need the empty row's guard: a
        call of no keys, one whose spans narrow, one a score hook may mask, or one walked past its `queries`,
        whose padding rows see none (their weights otherwise NaN, which a mainloop rerunning overflowed rows
        would rerun, `walked_launches`)."""
        return (self.keys == 0 or self.form == "spans" or self.score_mod is not None
                or self.queries is not None)

    def power(self, x: jax.Array) -> jax.Array:
        """The exponential the logits are in, a key's weight and the factor a softmax state takes from one
        maximum to another `x` below it alike: exp2 folded; natural, `natural_exp` of the raw form's scaled
        difference or of the scaled logits', as exact as the reference's `jnp.exp` on every backend, whose
        state (a resumed partial, a sink) may be any size: a Mosaic launch of approximate math lowers
        `jnp.exp` to the hardware's base-2 exponential of `x * log2(e)` rounded, about `|x|` ulps off for a
        state carried far, where `natural_exp` keeps that product's residual."""
        if self.base2:
            return jnp.exp2(x)
        return natural_exp(x, self.scale) if self.raw else natural_exp(x)

    def naturalized(self, m: jax.Array) -> jax.Array:
        """A maximum in the logits' units taken to natural ones: from base 2 folded, from raw products
        raw."""
        if self.base2:
            return m / LOG2E
        return m * self.scale if self.raw else m


class Recompute(NamedTuple):
    """How the backward recomputes a tile's weights, as its forward computed them: from raw products
    whose difference it scales by `factor` (`Forward.raw`; None, from scaled logits), and whether the
    largest logit weighs 1 exactly (`exact`, the accurate form's, `Forward.natural`)."""

    factor: float | None = None
    exact: bool = False


def recompute(spec: Forward) -> Recompute:
    """The backward's recomputation for a forward of `spec`."""
    return Recompute(spec.scale if spec.raw else None, spec.natural)


PLAIN = Recompute()
"""The default fold's recomputation: scaled logits, the largest logit's weight as its arithmetic gives it."""


def finite(x: jax.Array) -> jax.Array:
    """Whether each entry of `x` is finite, `|x| < inf` (NaN not), as `jnp.isfinite`: a mainloop's lowering
    may take no `is_finite` (Mosaic GPU's warpgroup lowering, jax 0.11.2)."""
    return jnp.abs(x) < jnp.inf


def infinite(x: jax.Array) -> jax.Array:
    """Whether each entry of `x` is infinite, `|x| == inf`, as `jnp.isinf` (`finite`)."""
    return jnp.abs(x) == jnp.inf


def expand(x: jax.Array, like: jax.Array) -> jax.Array:
    """A per-row `x` (each query's, `[..., bq]`) broadcast over `like`'s trailing axis, as `x[..., None]`
    broadcasts, by `lax.broadcast_in_dim`: the one broadcast every mainloop lowers, Mosaic GPU's
    register layouts included (a row vector of the WGMMA layout to its tile)."""
    return jax.lax.broadcast_in_dim(x, like.shape, tuple(range(x.ndim)))


def across(x: jax.Array, like: jax.Array) -> jax.Array:
    """A per-key `x` (`[bk]`) broadcast over `like`'s rows `[bq, bk]`, as `x[None, :]` broadcasts."""
    return jax.lax.broadcast_in_dim(x, like.shape, (like.ndim - 1,))


class Coords(NamedTuple):
    """A tile's coordinates as the mainloop provides them (in the layout its products give): the query
    indices `q_pos` `[bq, 1]` and key indices `k_pos` `[1, bk]`, and where the spans narrow, each
    query's intervals `spans`, one a run (a call's runs, `KeySpans`; one where it has none), each its
    first and past-last keys `[bq]` (`intervals`), with documents the queries' and keys' ids `ids`,
    `[bq]` and `[bk]`, and with dropout its entries' draw `drop` (`Draw`)."""

    q_pos: jax.Array
    k_pos: jax.Array
    spans: tuple[tuple[jax.Array, jax.Array], ...] | None = None
    ids: tuple[jax.Array, jax.Array] | None = None
    drop: Draw | None = None


def intervals(start: jax.Array, end: jax.Array) -> tuple[tuple[jax.Array, jax.Array], ...]:
    """A tile's queries' intervals as `Coords.spans` holds them: their first and past-last keys `[bq]`, or
    a call's runs `[bq, R]`, one interval a run."""
    if start.ndim == 1:
        return ((start, end),)
    return tuple((start[:, run], end[:, run]) for run in range(start.shape[-1]))


class Guard(NamedTuple):
    """A softmax state's headroom on the guarded path (`guarded`): its sums' `2^-shift` each row
    `[..., bq]` (`room`), a bound on the magnitude of the values the row has weighed, `reach` `[..., bq]`,
    the same sums kept unshifted (`plain` `[..., bq, Dv]` and `plain_total` `[..., bq]`, infinite where they
    overflow), and whether the row may shift, `live` (one whose plain walk overflowed): the output takes
    each element from the unshifted sums where they are finite, so no column of a row is shifted for
    another's sake (`finish`)."""

    shift: jax.Array
    reach: jax.Array
    plain: jax.Array
    plain_total: jax.Array
    live: jax.Array


class Low(NamedTuple):
    """What a compensated walk's running sums' additions dropped (`compensated`, `kept_in_turn`), each
    beside its sum: the weighted values' `acc` and the denominator's `total`, and on the guarded walk the
    unshifted sums' (`Guard.plain`, `plain_total`; None on the plain walk)."""

    acc: jax.Array
    total: jax.Array
    plain: Any = None
    plain_total: Any = None


class Partial(NamedTuple):
    """A query block's softmax state, unhooked: the weighted values `acc` `[..., bq, Dv]` (or, where a
    mainloop's accumulators hold them, the blocks of their columns it holds, `Keys.accumulate`), the running
    maximum `m` `[..., bq]` (in base 2 where the step's logits are) and the denominator `total`
    `[..., bq]`, fp32, both sums at each row's shift on the guarded path, whose `Guard` it carries (None on
    the plain path), and within a compensated walk what its sums' additions dropped (`low`, `compensated`;
    None elsewhere: a walk's state is `settled` when it ends). `rescale`, `merge` and `finish` take any
    leading axes, so states a launch wrote per slice merge in XLA after it (`exported`, `imported`)."""

    acc: Any
    m: jax.Array
    total: jax.Array
    guard: Guard | None = None
    low: Low | None = None


def exported(state: Partial) -> tuple[jax.Array, ...]:
    """A state's arrays as a launch writes them: its sums, maximum and denominator, and its guard's but
    which rows may shift, which a merge finds again (`merge`)."""
    if state.guard is None:
        return state.acc, state.m, state.total
    return state.acc, state.m, state.total, *state.guard[:4]


def imported(parts: tuple[jax.Array, ...]) -> Partial:
    """A state from the arrays a launch wrote (`exported`): on the guarded path no row of it shifts but by
    its own merge."""
    acc, m, total = parts[:3]
    if len(parts) == 3:
        return Partial(acc, m, total)
    shift, reach, plain, plain_total = parts[3:]
    return Partial(acc, m, total, Guard(shift, reach, plain, plain_total, jnp.zeros(total.shape, jnp.bool_)))


def empty(rows: int, width: int) -> Partial:
    """The softmax before any key: `merge`'s identity."""
    return Partial(jnp.zeros((rows, width), jnp.float32), jnp.full((rows,), -jnp.inf, jnp.float32),
                   jnp.zeros((rows,), jnp.float32))


def headroom_of(state: Partial, live: jax.Array | None = None) -> Guard:
    """`state`'s guard: its own on the guarded path, else the one a plain state takes there, no shift yet,
    its rows `live` (None, none), and a bound on its values' magnitudes, its weighted values' largest over
    its denominator (the weighted mean bounds the merged sums a slice's state joins, `merge`)."""
    if state.guard is not None:
        return state.guard
    rows = state.total
    largest = jnp.max(jnp.abs(state.acc), axis=-1)
    reach = jnp.where(rows > 0, largest / jnp.where(rows > 0, rows, 1.0), 0.0)
    live = jnp.zeros(rows.shape, jnp.bool_) if live is None else live
    return Guard(jnp.zeros_like(rows), reach, state.acc, rows, live)


def headroomed(state: Partial, live: jax.Array | None = None) -> Partial:
    """`state` on the guarded path (`headroom_of`)."""
    return state._replace(guard=headroom_of(state, live))


def overflowed(acc: jax.Array) -> jax.Array:
    """Which rows of a program's weighted values `acc` `[bq, Dv]` passed fp32's largest, `[bq]`: the one
    way a row of finite inputs turns non-finite (an infinite sum stays infinite under every later
    factor)."""
    # A maximum, the reduction every mainloop lowers (Pallas Triton takes no `jnp.all`).
    return jnp.max(jnp.where(finite(acc), 0.0, 1.0), axis=-1) > 0


def guarded[C: (Partial, tuple)](plain: Callable[[], C], shifted: Callable[[jax.Array], C],
                                 weighted: Callable[[C], jax.Array] | None = None) -> C:
    """A program's softmax walk under headroom: `plain()`, its walk as ever, its carry a `Partial` or a
    tuple whose first element is one; where that walk's weighted values (its state's, else `weighted(carry)`:
    a training walk's split sums joined, `split_output`) overflowed on some row (`overflowed`),
    `shifted(live)` instead, the walk again with those rows alone shifting (`room`), every other row's
    bits the plain walk's. Ordinary inputs take the plain walk alone, their bits and their cost unchanged
    but for this check; both return the guarded carry (`headroomed`)."""
    carry = plain()
    state, *rest = (carry,) if isinstance(carry, Partial) else carry
    live = overflowed(state.acc if weighted is None else weighted(carry))

    def kept(live):
        held = headroomed(state)
        return (held, *rest) if rest else held

    return jax.lax.cond(jnp.max(live.astype(jnp.int32)) > 0, shifted, kept, live)


FOLD_REACH = 2.0 ** 16
"""The largest folded logit a program's rows take the fold at (`walked`): below it the fold's rounding of a
logit, and of a row's maximum taken back to natural units for the backward, is under 2^-8 in its exponent,
a weight moved by under 0.3%. Past it an ulp of a folded logit nears or passes 1, so two logits the
reference keeps apart can tie and a row's maximum moves its weights by a factor (keys 1 to 4 ulps apart
near 2^30 weigh 1 and 0 in the reference, 1/2 each folded, and a cotangent near 2^125 then overflows the
backward's query gradient where the reference's is 0): such a program walks again in the natural form."""


def unfolded(spec: Forward) -> Forward:
    """The fold's rewalk form (`walked`): natural logits, scaled whatever the scale (`Forward.scaled`), so
    the backward recomputes its rows' weights as it recomputes the fold's (`PLAIN`: both from the scaled
    logits and a natural maximum)."""
    return dataclasses.replace(spec, natural=True, scaled=True)


def walked[C: (Partial, tuple), O](spec: Forward, walk: Callable[[Forward, jax.Array | None], C],
                                   finished: Callable[[Forward, C], O],
                                   weighted: Callable[[C], jax.Array] | None = None) -> O:
    """A program's walk and its outputs under the guards its form takes: `walk(form, live)` its walk in
    `form`, plain where `live` is None, else those rows shifting (`guarded`), its carry a `Partial` or a
    tuple whose first element is one, and `finished(form, carry)` its outputs. Under headroom the walk is
    guarded (`guarded`, `weighted` as it reads it). Folded, a program one of whose rows' maximum passes
    `FOLD_REACH` walks again in the natural form (`unfolded`), guarded as the call's form is, its outputs
    that form's (its maximum natural, as the fold's are once finished). Ordinary inputs take the plain walk
    alone, their bits and cost unchanged but for these checks."""
    def guarded_walk(form: Forward, plain: Callable[[], C]) -> C:
        def shifted(live: jax.Array) -> C:
            return walk(form, live)

        return guarded(plain, shifted, weighted) if form.headroom else plain()

    def walked_plain(form: Forward) -> C:
        return walk(form, None)

    if not spec.base2:
        return finished(spec, guarded_walk(spec, functools.partial(walked_plain, spec)))
    carry = walk(spec, None)
    state = carry if isinstance(carry, Partial) else carry[0]
    reach = reach_of(state.m)
    natural = unfolded(spec)

    def folded() -> C:
        return carry

    def unfolded_walk() -> C:
        return guarded_walk(natural, functools.partial(walked_plain, natural))

    return jax.lax.cond(reach >= FOLD_REACH, lambda: finished(natural, unfolded_walk()),
                        lambda: finished(spec, guarded_walk(spec, folded)))


def reach_of(maxima: jax.Array) -> jax.Array:
    """How far a folded walk's row `maxima` reach (`walked`, against `FOLD_REACH`): their largest magnitude,
    a row that saw no key (-inf) none, and one whose maximum the fold took past fp32's largest, or to no
    number, past every reach, so the call walks again natural, where its logits are finite (a product near
    fp32's largest before a small scale, times log2(e))."""
    reached = jnp.where(jnp.isnan(maxima), jnp.inf, jnp.abs(maxima))
    return jnp.max(jnp.where(maxima == -jnp.inf, 0.0, reached))


type Launch[O] = Callable[[Forward, jax.Array | None], tuple[O, jax.Array, jax.Array]]
"""A whole launch of a call in a form (`walked_launches`): its outputs, each row's running maximum and
whether its weighted values overflowed."""


def walked_launches[O](spec: Forward, launch: Launch[O], kept: Callable[[jax.Array, O, O], O]) -> O:
    """`walked`'s guards over whole launches, for a mainloop whose program walks its keys once (a producer
    streams them, so no program walks again): `launch(form, live)` runs the call in `form`, plain where
    `live` is None, else with its rows `live` shifting (`guarded`), and returns its outputs, each row's
    running maximum in the walk's units and whether its weighted values overflowed (`overflowed`), `[..., Q]`
    each; `kept(live, guarded, plain)` takes the guarded launch's outputs at the rows `live`, the plain
    one's elsewhere, so every other row's bits are the plain walk's. The call walks again where `walked`'s
    program would, decided over the whole call: in the natural form (`unfolded`) where some row's maximum
    passes `FOLD_REACH` folded, guarded as the call's form is; under headroom, the rows whose weighted values
    overflowed. Ordinary inputs take the one plain launch, their cost unchanged but for these checks."""
    def guarded_launch(form: Forward, outputs: O, over: jax.Array) -> O:
        if not form.headroom:
            return outputs
        return jax.lax.cond(jnp.max(over.astype(jnp.int32)) > 0,
                            lambda: kept(over, launch(form, over)[0], outputs), lambda: outputs)

    outputs, maxima, over = launch(spec, None)
    if not spec.base2:
        return guarded_launch(spec, outputs, over)
    reach = reach_of(maxima)
    natural = unfolded(spec)

    def unfolded_launch() -> O:
        unfolded_outputs, _, unfolded_over = launch(natural, None)
        return guarded_launch(natural, unfolded_outputs, unfolded_over)

    return jax.lax.cond(reach >= FOLD_REACH, unfolded_launch, lambda: guarded_launch(spec, outputs, over))


POWER2_REACH = 381.0
"""The largest `|k|` `power2` takes apart: past it every fp32 `x * 2^k` is zero or infinite (`2^-381` of
fp32's largest flushes, `2^381` of its least overflows), so `k` is held there."""


def exact_power2(k: jax.Array) -> jax.Array:
    """`2^k` exactly for an integral `k` in [-126, 127], built from its fp32 exponent bits: no `exp2`,
    which XLA computes as `exp(k ln 2)`, inexactly, and whose products it fuses into one `exp` of their
    sum (its algebraic simplifier's `exp(a) exp(b) -> exp(a + b)`), past fp32's range where each factor
    is in it."""
    biased = jnp.asarray(k, jnp.float32).astype(jnp.int32) + jnp.int32(127)
    bits = jax.lax.shift_left(biased, jnp.full_like(biased, 23))
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def power2(x: jax.Array, k: jax.Array) -> jax.Array:
    """`x * 2^k` for an integral `k` of any size, exactly wherever the result is a normal fp32: three exact
    factors each within 2^±127 (`exact_power2`), of `k`'s sign, so each partial product lies between `x`
    and the result; `k` held within `POWER2_REACH`, past which the result is zero or infinite."""
    k = jnp.clip(jnp.asarray(k, jnp.float32), -POWER2_REACH, POWER2_REACH)
    # k's third by its product with fp32's 1/3, the integer quotient for every integral |k| the clip leaves
    # (POWER2_REACH): k * fl(1/3) exceeds k / 3 by under |k| * 2^-25, far below the 1/3 by which a
    # non-multiple of 3 falls short of the next integer, so the truncation lands where the division's would.
    # An integer division costs the Mosaic mainloops registers they don't have (sm80's GDN training spilled).
    first = jnp.trunc(k * np.float32(1 / 3))
    second = jnp.trunc((k - first) / 2)
    return x * exact_power2(first) * exact_power2(second) * exact_power2(k - first - second)


def natural_exp(x: jax.Array, scale: float = 1.0) -> jax.Array:
    """The natural form's forward exponential (`Forward.power`): `exp(x * scale)` as the hardware's
    base-2 exponential of `x * scale * log2(e)`, the logits staying natural. The conversion keeps its
    rounded product and its residual (`two_product`), the factor's own rounding among them, and the
    residual joins the reduced exponent's fractional part (`power2` takes the integral one), so the
    conversion's rounding does not reach a large value under a small weight, where XLA had folded
    `exp2(x * log2(e))` back to `exp(x)` until JAX 0.12 stopped reassociating constant products
    (jax-ml/jax 2d3744395). The joined exponent lies within an ulp of the conversion of [-1/2, 1/2] and
    rounds at its own ulp, at most 2^-25 there, which moves the weight by at most a quarter of its ulp: the
    weight's error is about the base-2 exponential's on that interval, as the reference's `exp` is
    (libdevice's reduces its argument so and takes the same exponential), where a correction added after
    the exponential rounded the weight once more, half an ulp (1.55 times the reference's error on an
    A100's and an H100's edge weights). The factors that carry a
    state take it too (`Forward.power`); the backward's weights keep `jnp.exp`, as the reference does."""
    # Normalize the factor before splitting: at a tiny raw scale its residual would be subnormal.
    factor, exponent = np.frexp(scale * LOG2E)
    x = x * np.float32(2.0 ** int(exponent))
    high = np.float32(factor)
    low = np.float32(factor - float(high))
    # Bit splitting avoids overflowing Dekker's splitter where raw logits are large and scale is small.
    hi, error = two_product(x, high, bit_split=True)
    lo = error + x * low
    # XLA's CPU exp2 is exp(hi * ln(2)), so reduce first: that multiply must not round a large exponent.
    # Outside these bounds every fp32 exponential is zero or infinite; infinities have no residual.
    bounded = jnp.clip(hi, -150.0, 128.0)
    exponent = jnp.floor(bounded)
    exponent += (bounded - exponent >= 0.5).astype(x.dtype)
    live = (jnp.abs(x) < jnp.inf) & (hi >= -150.0) & (hi <= 128.0)
    lo = jnp.where(live, lo, 0.0)
    return power2(jnp.exp2((bounded - exponent) + lo), exponent)


def headroom(dtype) -> bool:
    """Whether a call's weighted values can pass fp32's largest: values of `dtype` two of which reach
    `2^HEADROOM` (bf16's and fp32's). An initial state's sums are finite, and a walk adds to them at most its
    keys' count times the values' largest, which for fp16's values over any key count a call holds is far
    below the 2^103 that rounds fp32's largest past it; so an fp16 call compiles no guard (`guarded`),
    from a state or none."""
    return float(jnp.finfo(dtype).max) * 2 >= 2.0 ** HEADROOM


def compensates(dtype) -> bool:
    """Whether a walk over values of `dtype` keeps what its running sums' additions drop
    (`Forward.compensated`): fp32's, whose products round at fp32's own precision, so where a row holds many
    comparable terms the order of their sum leads the output's error, and a chain of them errs past the
    reference's, which sums a row as a tree (jax's attention on an A100 and an H100: 1.7 to 2.1 times its
    error over equal weights, whose products are exact, a tile's chain of 32 keys then the tiles' chain). A
    16-bit dtype's weights round to it first, as the reference's do, an error beside which the sums' order
    is nothing."""
    return jnp.dtype(dtype) == jnp.float32


CHUNK = 16
"""The keys a compensated walk sums in one chain (`products_of`): the fewest a Triton dot takes, a chain of
fused multiply-adds, each chunk's product then added with its rounding kept (`kept_in_turn`). Against jax's
attention on an A100 over 40 seeds of the three equal-weight fp32 families: 1.20, 1.32 and 1.23 times its
error, where chunks of 8 keys would give 0.82 to 1.01 and the tile's one chain 2.08, 1.67 and 1.86."""


RAW_SCALES = (2.0 ** -120, 2.0 ** 100)
"""The scales the raw form takes (`Forward.raw`), where its arithmetic weighs every key as the scaled
logits do. Below: a raw difference past fp32's largest (its products near it, of opposite signs) is -inf,
its weight 0; the scaled difference is then past `2^128 * 2^-120 = 2^8`, whose exponential is below fp32's
least subnormal, so both weigh 0, where at a smaller scale the scaled one would be finite. Above: the
factor `scale * log2(e)` stays finite in fp32, so the largest key's exponent `0 * factor` is 0; a larger
scale's factor is infinite and that exponent NaN. Outside, the accurate form scales its logits."""


HEADROOM = 126
"""The exponent a state's weighted values stay within (`needed`): one below fp32's largest binade, so a
tile's product adds at most as much again and the sum stays finite."""


def needed(total: jax.Array, reach: jax.Array, bound: float = 1.0) -> jax.Array:
    """The shift a row's sums need, `[...]` fp32 integers: none while its natural denominator `total` times
    the largest value it weighs, `reach`, is below `2^HEADROOM`, the weighted values' bound; past it, as
    many binades as that product passes it by (log2 rounded up, so the bound holds whatever their
    rounding). A dropping call's draw multiplies each weight by at most `bound`, which counts in the log
    rather than in a product: `reach * bound` may itself pass fp32's largest."""
    log = jnp.log2(total) + jnp.log2(reach)
    if bound != 1.0:
        log = log + math.log2(bound)
    binades = jnp.ceil(log) - HEADROOM
    return jnp.where(finite(binades) & (binades > 0), binades, 0.0)


WEIGHT_SHIFT = 64
"""The most of a row's shift its weights take (`room`): the rest scales the tile's product, so no weight
of 2^-62 or more is pushed below fp32's least normal however large the shift."""


class Room(NamedTuple):
    """A guarded tile's arithmetic (`room`): its weights at `2^-min(shift, WEIGHT_SHIFT)`, the change of
    the row's shift its state's sums take after its factor (`drop`, at most 0; `carry_factor`), the row's
    new shift and value bound, and `beyond`, the shift past its weights', which the tile's product and
    weight sums take (`beyond_weights`), the tile's weights unshifted, `plain`, whose product the mainloop
    issues too (`stepped`), the factor `alpha` the state's sums take, and the state's guard before the
    tile, `before`."""

    weights: jax.Array
    drop: jax.Array
    shift: jax.Array
    reach: jax.Array
    beyond: jax.Array
    plain: jax.Array
    alpha: jax.Array
    before: Guard


def beyond_weights(x: jax.Array, room: Room | None) -> jax.Array:
    """A tile's product `[bq, Dv]` or weight sum `[bq]` taken from its weights' shift to the row's; as it
    is on the plain walk (`room` None)."""
    if room is None:
        return x
    return power2(x, room.beyond if x.ndim == room.beyond.ndim else expand(room.beyond, x))


def carry_factor(x: jax.Array, alpha: jax.Array, held: Room | None) -> jax.Array:
    """A state's sum before a tile (`[bq, Dv]` or `[bq]`) taken by its factor `alpha` to the tile's
    maximum and, on the guarded walk, by the change of the row's shift after it (`Room.drop`), a power of
    two on the product: no factor below fp32's least normal is formed, so a shift that grows by 126
    binades or more at once keeps a sum the factor alone would flush."""
    taken = rescale(x, alpha) if x.ndim > alpha.ndim else alpha * x
    if held is None:
        return taken
    return power2(taken, held.drop if taken.ndim == held.drop.ndim else expand(held.drop, taken))


def tile(state: Partial, alpha: jax.Array, p: jax.Array, largest: Callable[[], jax.Array],
         bound: float = 1.0) -> tuple[jax.Array, jax.Array, Room | None]:
    """A tile's weights for its product and the factor the state's sums take (`carry_factor`), with the
    guarded walk's `Room` (`room`, its keys' values' largest magnitudes `largest()`), None on the plain walk,
    whose weights are `p` as ever."""
    if state.guard is None:
        return p, alpha, None
    held = room(state.guard, state.total, alpha, p, largest(), bound)
    return held.weights, alpha, held


def denominator_after(total: jax.Array, factor: jax.Array, p: jax.Array, held: Room | None = None
                      ) -> jax.Array:
    """A row's denominator after a tile: the state's taken by `factor` (`carry_factor`) and the tile's
    weights `p` added at the row's shift (`beyond_weights`)."""
    return carry_factor(total, factor, held) + beyond_weights(p.sum(axis=-1), held)


def stepped(state: Partial, m: jax.Array, factor: jax.Array, p: jax.Array, products: tuple[jax.Array, ...],
            held: Room | None, times: Callable[[jax.Array], tuple[jax.Array, ...]]) -> Partial:
    """The state after a tile: its sums taken by `factor` (`carry_factor`) and the tile's `products` of its
    weights `p` with its values (a chunk of keys each in a compensated walk, `products_of`) and their sum
    added at the row's shift (`beyond_weights`; `tile`), with the tile's unshifted weights' products, which
    the mainloop issues as `times` of them, kept unshifted (`Guard.plain`); each addition's rounding kept
    where the state carries them (`kept_in_turn`); the plain walk's arithmetic where `held` is None, which
    issues no other product."""
    low = state.low
    total, acc = carry_factor(state.total, factor, held), carry_factor(state.acc, factor, held)
    weights = (beyond_weights(p.sum(axis=-1), held),)
    sums = tuple(beyond_weights(product, held) for product in products)
    kept = None
    if low is None:
        total, acc = in_turn(total, weights), in_turn(acc, sums)
    else:
        total, total_low = kept_in_turn(total, carry_factor(low.total, factor, held), weights)
        acc, acc_low = kept_in_turn(acc, carry_factor(low.acc, factor, held), sums)
        kept = Low(acc_low, total_low)
    if held is None:
        return Partial(acc, m, total, low=kept)
    # The unshifted sums beside them: finite in every column the plain walk kept finite (`finish`).
    plain, plain_total = rescale(held.before.plain, held.alpha), held.alpha * held.before.plain_total
    plain_sums, plain_weights = times(held.plain), (held.plain.sum(axis=-1),)
    if kept is None or low is None:
        plain, plain_total = in_turn(plain, plain_sums), in_turn(plain_total, plain_weights)
    else:
        plain, plain_low = kept_in_turn(plain, rescale(low.plain, held.alpha), plain_sums)
        plain_total, plain_total_low = kept_in_turn(plain_total, held.alpha * low.plain_total, plain_weights)
        kept = kept._replace(plain=plain_low, plain_total=plain_total_low)
    return Partial(acc, m, total, Guard(held.shift, held.reach, plain, plain_total, held.before.live), kept)


def in_turn(total: jax.Array, terms: tuple[jax.Array, ...]) -> jax.Array:
    """`total` with each of `terms` added in turn."""
    for term in terms:
        total = total + term
    return total


def kept_in_turn(total: jax.Array, dropped: jax.Array, terms: tuple[jax.Array, ...]
                 ) -> tuple[jax.Array, jax.Array]:
    """`total` with each of `terms` added in turn and what earlier additions dropped, `dropped` (a compensated
    walk's, `Low`), with each one's rounding added (Knuth's TwoSum, `two_sum`). An infinite sum's rounding is
    no number (`settled`)."""
    for term in terms:
        total, error = two_sum(total, term)
        dropped = dropped + error
    return total, dropped


def compensated(spec: Forward, state: Partial) -> Partial:
    """A walk's start in `spec`'s form: where it compensates its running sums (`Forward.compensated`), with
    nothing dropped yet beside each (`Low`), the guarded walk's unshifted sums' too; else as it is."""
    if not spec.compensated:
        return state
    guard = state.guard
    return state._replace(low=Low(jnp.zeros_like(state.acc), jnp.zeros_like(state.total),
                                  None if guard is None else jnp.zeros_like(guard.plain),
                                  None if guard is None else jnp.zeros_like(guard.plain_total)))


def settled(state: Partial) -> Partial:
    """A walk's end: each running sum with what its additions dropped added back (`compensated`), rounded
    once, and none carried; a sum that passed fp32's largest stays infinite, its residual no number."""
    low = state.low
    if low is None:
        return state

    def joined(total: jax.Array, dropped: jax.Array) -> jax.Array:
        return total + jnp.where(finite(dropped), dropped, 0.0)

    guard = state.guard
    if guard is not None:
        guard = guard._replace(plain=joined(guard.plain, low.plain),
                               plain_total=joined(guard.plain_total, low.plain_total))
    return Partial(joined(state.acc, low.acc), state.m, joined(state.total, low.total), guard)


def room(guard: Guard, total: jax.Array, alpha: jax.Array, p: jax.Array, largest: jax.Array,
         bound: float = 1.0) -> Room:
    """A tile's weights `p` (`weights`) and the factor the sums of its state (`guard`, denominator
    `total`) take (`alpha`, `peak`'s) with each live row's shift (`needed`; `Guard.live`). The weighted
    values of a row can pass fp32's largest only where its values are near it (a sum of up to `total` of
    them); a shift of `2^-k` on its sums keeps them within it, exactly, as a power of two scales, and
    leaves every ratio the softmax takes, so the output is the unshifted one. A row whose plain walk did
    not overflow never shifts, its bits the plain walk's. The weights take at most `WEIGHT_SHIFT` of a
    shift, the product and weight sums the rest (`beyond_weights`), so a weight lost below fp32's least
    normal is under `2^-62`: a live row's weighted values pass `2^128` in some column, so its output there
    is at least `2^128 / total`, and such a weight on a value at most fp32's largest moves its output by
    under `2^66 / total`, `2^-62` of that, in a column that overflowed: every other column's output comes
    from the unshifted sums (`Guard.plain`), so a small column of a row whose other column overflows keeps
    the reference's resolution."""
    # The largest value magnitude of the keys each row weighs (`largest`, `[bk]`, each key's own), as a key
    # the row does not see (padding, a stale cache slot, another document) would shift it for nothing.
    reach = jnp.maximum(guard.reach, jnp.where(p > 0, across(largest, p), 0.0).max(axis=-1))
    natural = power2(alpha * total, guard.shift) + p.sum(axis=-1)
    shift = jnp.where(guard.live, jnp.maximum(guard.shift, needed(natural, reach, bound)), guard.shift)
    weight = jnp.minimum(shift, WEIGHT_SHIFT)
    return Room(power2(p, expand(-weight, p)), guard.shift - shift, shift, reach, weight - shift, p, alpha,
                guard)


def base2_maximum(m: jax.Array) -> jax.Array:
    """A natural initial maximum (a sink's logit, a partial's) in base 2, `m * log2(e)`, saturated at fp32's
    largest where that product passes it. Exact as far as any key reads it: a folded logit is at most fp32's
    largest less an ulp (`Forward.base2`), so a maximum past fp32's largest in base 2 exceeds every key's
    by over an ulp there, 2^104, as its saturation does, and every key's weight is zero under either, the
    state's own one; one past its negative is below every key's alike."""
    folded = m * LOG2E
    largest = float(jnp.finfo(jnp.float32).max)
    return jnp.where(infinite(folded) & finite(m), jnp.sign(m) * largest, folded)


def resumed(spec: Forward, m: jax.Array, total: jax.Array, acc: jax.Array, *,
            live: jax.Array | None = None) -> Partial:
    """The softmax resumed from a state `(m, total, acc)` (`Hooks.initial`: sinks, a resumed partial),
    its maximum taken to base 2 where the step's logits are; on the guarded path (`live`, the rows that
    may shift, `guarded`) those rows' sums shifted as their size needs (`needed`: `acc`'s largest
    magnitude is its denominator times a bound on the values it weighed)."""
    m, total, acc = (array.astype(jnp.float32) for array in (m, total, acc))
    m = base2_maximum(m) if spec.base2 else m
    if live is None:
        return Partial(acc, m, total)
    held = headroom_of(Partial(acc, m, total), live)
    shift = jnp.where(live, needed(total, held.reach), 0.0)
    return Partial(power2(acc, expand(-shift, acc)), m, power2(total, -shift), held._replace(shift=shift))


def modify(spec: Forward, s: jax.Array, coords: Coords, params: tuple[jax.Array, ...] = (),
           masked: bool | jax.Array = True) -> jax.Array:
    """A tile's logits from its products `s` `[bq, bk]` (fp32): scaled (by log2(e) too in base 2; raw in
    the raw form, `Forward.raw`), then the score hook, then -inf at each key the query does not see.
    `masked` False is a tile every query of the block sees whole (a full tile of the block map), which
    takes no mask; a traced `masked` chooses at run time. Without a hook the mask is taken on the products,
    at the infinity the scale takes to -inf, and the scale after it, so a mainloop's multiply feeds the
    weights' subtraction whole, which its compiler contracts to one FMA (`weights`), whichever the tile."""
    return logits_of(spec, s, coords, params, masked)[0]


def logits_of(spec: Forward, s: jax.Array, coords: Coords, params: tuple[jax.Array, ...] = (),
              masked: bool | jax.Array = True) -> tuple[jax.Array, jax.Array | None]:
    """`modify`'s logits and, where they are the products (masked) times `Forward.factor` alone, those
    products, whose extreme the tile's maximum takes exactly (`peak`), so the logits' multiply feeds only
    the weights' subtraction, one FMA (`weights`); None under a hook or a zero factor."""
    factor = spec.factor
    if spec.score_mod is None and factor != 0:
        # The infinity the factor takes to -inf (a zero factor takes none, so it masks after scaling).
        unseen = -jnp.inf if factor > 0 else jnp.inf
        products = _chosen(masked, lambda: _masked(spec, s, coords, unseen), lambda: s)
        return (products if spec.raw else products * factor), products
    logits = s * factor
    if spec.score_mod is not None:
        logits = hooked(spec.score_mod, logits, coords.q_pos, coords.k_pos, params)
    return _chosen(masked, lambda: _masked(spec, logits, coords, -jnp.inf), lambda: logits), None


def _chosen(masked: bool | jax.Array, mask: Callable[[], jax.Array], whole: Callable[[], jax.Array]):
    """`mask()` where `masked`, else `whole()`: at trace time for a Python flag, at run time for a traced
    one."""
    if isinstance(masked, bool):
        return mask() if masked else whole()
    return jax.lax.cond(masked, mask, whole)


def _masked(spec: Forward, s: jax.Array, coords: Coords, unseen: float) -> jax.Array:
    """`s` `[bq, bk]` at `unseen` at each key the query does not see (`seen`)."""
    visible = seen(spec, coords, s)
    return s if visible is None else jnp.where(visible, s, unseen)


def seen(spec: Forward, coords: Coords, like: jax.Array) -> jax.Array | None:
    """Whether each query of a tile `like` `[bq, bk]` sees each key (`within`), every mainloop's one
    predicate, forward and backward; None where each sees every key. A spans call's intervals, the union
    of its runs' where it has them, and, with documents, ids (its mainloops give each tile its queries'
    intervals and both blocks' ids); a causal call's diagonal; the call's keys, where a tile reaches past
    them: no walk visits a tile wholly past them, so a call whose keys are whole tiles reads no padding;
    and, where the spec names them, its queries (a padding row sees no key, so a score hook's derivative
    there, undefined at its padded parameters, reaches no reduction over the rows)."""
    visible = None
    if coords.spans is not None:
        ids = None if coords.ids is None else (expand(coords.ids[0], like), across(coords.ids[1], like))
        visible = functools.reduce(jnp.logical_or, (within(coords.k_pos, expand(start, like),
                                                           expand(end, like), ids)
                                                    for start, end in coords.spans))
    elif spec.form == "causal":
        visible = coords.k_pos <= coords.q_pos
    if spec.form != "spans" and spec.keys % like.shape[1]:
        # A tile past the call's last key reads padding.
        visible = coords.k_pos < spec.keys if visible is None else visible & (coords.k_pos < spec.keys)
    if spec.form != "spans" and spec.queries is not None:
        rows = coords.q_pos < spec.queries
        visible = rows if visible is None else visible & rows
    return visible


def _safe(spec: Forward, m: jax.Array) -> jax.Array:
    """`m`, with 0 for a query that has seen nothing where one can (`Forward.guarded`)."""
    return jnp.where(m == -jnp.inf, 0.0, m) if spec.guarded else m


def peak(spec: Forward, m: jax.Array, s: jax.Array, products: jax.Array | None = None
         ) -> tuple[jax.Array, jax.Array]:
    """The running maximum after a tile's logits `s`, and `alpha`, the factor the state before it
    takes to the new maximum. In the accurate form the exponent `m - new` is compensated (its subtraction's
    rounding, by TwoSum, carried to first order): every weight before the new maximum shares this one
    factor, so its exponent's rounding would not average out over them as the reference's per-key exponents'
    do, an error some sqrt(keys) times the reference's where the maximum jumps far. Where the logits are
    `products` times `Forward.factor` (`logits_of`), the tile's maximum is their extreme times it, the
    same bits (fp32's rounding is monotonic: the largest of `fl(p c)` is `fl(max(p) c)` for a positive `c`,
    `fl(min(p) c)` for a negative one), taken once a row."""
    if products is None:
        tile = s.max(axis=-1)
    else:
        tile = (products.max(axis=-1) if spec.factor > 0 else products.min(axis=-1)) * spec.factor
    new = jnp.maximum(m, tile)
    return new, carried(spec, m, _safe(spec, new))


def carried(spec: Forward, m: jax.Array, safe: jax.Array) -> jax.Array:
    """The factor `exp(m - safe)` a state takes from its maximum `m` to `safe` (`peak`, `merge`), in the
    accurate form with its exponent compensated (`peak`)."""
    if not spec.natural:
        return spec.power(m - safe)
    # TwoSum: m - safe is difference + error exactly; raw, the scale's product with it is split exactly too
    # (TwoProduct over truncated halves, whose products are exact, so a compiler's contraction of any of
    # them into a fused multiply-add changes no bit); exp's derivative is itself.
    difference, error = two_sum(m, -safe)
    alpha = spec.power(difference)
    if spec.raw:
        product, rounding = two_product(difference, np.float32(spec.scale), bit_split=True)
        alpha = natural_exp(product)
        error = rounding + error * spec.scale
    corrected = alpha * (1.0 + error)
    return jnp.where(finite(corrected), corrected, spec.power(difference))


def weights(spec: Forward, s: jax.Array, m: jax.Array) -> jax.Array:
    """A tile's unnormalized weights `[bq, bk]` under the running maximum `m`, fp32. The fold computes
    `exp2(s c - m)`, which a compiler contracts to `fma(s, c, -m)` (FlashAttention 2's by design): the
    largest logit's exponent is then its product's rounding, up to an ulp of the logit, so at logits near
    fp32's largest its weight is 0 or infinite, one of the fold's documented divergences. In the accurate
    form (`Forward.natural`) the largest logit weighs 1 exactly: raw logits subtract before they scale
    (`Forward.raw`), and scaled ones (under a hook or an initial state) hold it to 1 by a select."""
    peak = expand(_safe(spec, m), s)
    if spec.natural and not spec.raw:
        # The accurate form's scaled logits: the largest's weight held to 1 against the contraction.
        return spec.power(jnp.where(s < peak, s - peak, 0.0))
    return spec.power(s - peak)


def narrow(p: jax.Array, dtype) -> jax.Array:
    """The weights at the values' dtype, the operand of `P·V`: the reference rounds them so too."""
    return p.astype(dtype)


def total(denominator: jax.Array, alpha: jax.Array, p: jax.Array) -> jax.Array:
    """The denominator after a tile."""
    return alpha * denominator + p.sum(axis=-1)


def rescale(acc: jax.Array, alpha: jax.Array) -> jax.Array:
    """The weighted values before a tile, taken to the new maximum; `P·V` adds the tile's to them."""
    return expand(alpha, acc) * acc


class Argmax(NamedTuple):
    """The keys' running maximum logit `peak` `[bq]` (the initial state's apart) and `key`, the first
    key that holds it `[bq]` int32: the backward's anchor key (`off_anchor`), a residual of the forward. Under
    dropout, `multiplier` `[bq]` is that key's draw, which the anchor's own weight takes (`joined`)."""

    peak: jax.Array
    key: jax.Array
    multiplier: jax.Array | None = None


def no_argmax(rows: int, dropping: bool = False) -> Argmax:
    """`Argmax` before any key, with a multiplier of zeros where a call drops (`dropping`)."""
    multiplier = jnp.zeros((rows,), jnp.float32) if dropping else None
    return Argmax(jnp.full((rows,), -jnp.inf, jnp.float32), jnp.zeros((rows,), jnp.int32), multiplier)


def started(spec: Forward, initial: tuple[jax.Array, jax.Array, jax.Array] | None, rows: int, width: int,
            live: jax.Array | None) -> Partial:
    """A program's softmax before its first tile: resumed from `initial` (`resumed`), else `empty`; on the
    guarded path (`live`, `guarded`) headroomed."""
    if initial is not None:
        return resumed(spec, *initial, live=live)
    state = empty(rows, width)
    return state if live is None else headroomed(state, live)


Parts = tuple[tuple[jax.Array, jax.Array | None], ...]
"""A product's parts as a mainloop issues them (`cotangent_parts`, `weights_parts`): each a narrowed
operand with the power of two its product takes back (None: none)."""


class Keys(NamedTuple):
    """A key tile as a mainloop gives it the algebra: its coordinates and (modified) logits `[bq, bk]`,
    `load()` its values `[bk, Dv]`, which a step calls after the softmax (where jax's mha loads them), and on
    the guarded walk their largest magnitude a key (`magnitudes`, or `largest()` where the mainloop holds the
    values where it cannot read them, a Mosaic slot's shared memory), the
    mainloop's product `dot(a, b)`, `a @ b` in fp32, where the mainloop holds the running sums of the
    weighted values itself (an MMA's accumulator), `accumulate(acc, scale, parts, value)`, which returns
    the running sums `acc` (the state's, in the mainloop's own form) taken by `scale(acc)` (None: no factor,
    `anchored_step`) with the weights' `parts`' products with the values added (`summed_products`), and the
    products the logits are a factor of, where they are (`logits_of`), whose extreme the tile's maximum
    takes (`peak`)."""

    coords: Coords
    logits: jax.Array
    load: Callable[[], jax.Array]
    dot: Callable[[jax.Array, jax.Array], jax.Array]
    accumulate: Callable[[Any, Callable | None, Parts, jax.Array], Any] | None = None
    products: jax.Array | None = None
    largest: Callable[[], jax.Array] | None = None


Carry = tuple[Partial, Argmax, jax.Array, jax.Array]
"""A training walk's carry: its state, whose values are `B`, every key's but the anchor's
(`training_step`), its anchor key, the anchor's own weight and the other keys' weights' sum."""


def drawn(coords: Coords, queries: jax.Array, keys: jax.Array) -> jax.Array | None:
    """A tile's dropout multipliers at `queries` and `keys` (`Coords.drop`), None where no call drops."""
    return None if coords.drop is None else coords.drop(queries, keys)



def _times(dot: Callable[[jax.Array, jax.Array], jax.Array], value: jax.Array, p: jax.Array,
           multiplier: jax.Array | None = None, chunk: int | None = None) -> tuple[jax.Array, ...]:
    """Weights `p` narrowed to the values' dtype times the tile's `value`s, as the mainloop's `dot`, a
    `chunk` of keys at a time (`products_of`); with a dropout `multiplier`, the dropped weights
    `p * multiplier` narrowed at the scale `weights_parts` gives them."""
    if multiplier is not None:
        return products_of(weights_parts(p * multiplier, value.dtype), value, dot, chunk)
    return products_of(((narrow(p, value.dtype), None),), value, dot, chunk)


def products_of(parts: Parts, value: jax.Array, dot: Callable[[jax.Array, jax.Array], jax.Array],
                chunk: int | None) -> tuple[jax.Array, ...]:
    """The weights' `parts` `[bq, bk]` times a tile's `value`s `[bk, Dv]` (`summed_products`): one product,
    or in a compensated walk one a `chunk` of keys (`chunk_of`), each a chain of `chunk` additions, whose
    sum the walk takes with each addition's rounding kept (`kept_in_turn`)."""
    if chunk is None or value.shape[0] <= chunk:
        return (summed_products(parts, value, dot),)
    count = value.shape[0] // chunk
    values = jnp.split(value, count, axis=0)
    pieces = [jnp.split(part, count, axis=-1) for part, _ in parts]
    return tuple(summed_products(tuple((piece[index], exponent) for piece, (_, exponent) in zip(pieces, parts,
                                                                                               strict=True)),
                                 values[index], dot)
                 for index in range(count))


def chunk_of(spec: Forward) -> int | None:
    """The keys a walk in `spec`'s form sums in one chain: `CHUNK` where it compensates its sums, else a
    tile's whole (None)."""
    return CHUNK if spec.compensated else None


def _softmax(spec: Forward, state: Partial, keys: Keys):
    """A tile's softmax: the running maximum, the factor the state's sums take, the weights for the
    product (the guarded walk's at the row's shift, `tile`), the tile's values and the guarded `Room`."""
    m, alpha = peak(spec, state.m, keys.logits, keys.products)
    p = weights(spec, keys.logits, m)
    value = keys.load()
    largest = keys.largest if keys.largest is not None else functools.partial(magnitudes, value)
    bound = 1.0 if keys.coords.drop is None else keys.coords.drop.bound
    p, factor, held = tile(state, alpha, p, largest, bound)
    return m, factor, p, value, held


def magnitudes(value: jax.Array) -> jax.Array:
    """Each key's values' largest magnitude, `[bk]` of a tile's `[bk, Dv]`, fp32 (`room`)."""
    return jnp.max(jnp.abs(value.astype(jnp.float32)), axis=-1)


def plain_step(spec: Forward, state: Partial, keys: Keys) -> Partial:
    """The state after a tile `keys`: its softmax and the product of its weights with its values, the
    mainloop's own running sums taking it where it holds them (`Keys.accumulate`, the state's `acc` in its
    form)."""
    m, factor, p, value, held = _softmax(spec, state, keys)
    if held is None and keys.accumulate is not None:
        acc = keys.accumulate(state.acc, lambda acc: carry_factor(acc, factor, None),
                              ((narrow(p, value.dtype), None),), value)
        return Partial(acc, m, denominator_after(state.total, factor, p))
    multiplier = drawn(keys.coords, keys.coords.q_pos, keys.coords.k_pos)
    times = functools.partial(_times, keys.dot, value, multiplier=multiplier, chunk=chunk_of(spec))
    return stepped(state, m, factor, p, times(p), held, times)


Gathered = Callable[[jax.Array], jax.Array]
"""A training walk's value rows gathered by key: each row's `[bq, Dv]` at its anchor key `[bq]`."""


def _split_tile(keys: Keys, p: jax.Array, dtype, anchor: Argmax, multiplier: jax.Array | None
                ) -> tuple[Parts, jax.Array, jax.Array, Argmax]:
    """A training step's tile split at each row's `anchor` key (`anchor_split`): the other keys' weights as
    the parts of their product with the values (`dtype`), the weights themselves narrowed, and the anchor's
    own weight; under dropout (the tile's draw's `multiplier`) the other keys' dropped weights, the anchor's
    own weight undropped (it is `own`), and the anchor takes its draw's multiplier, which the forward's output
    and the backward read."""
    at_anchor = keys.coords.k_pos == expand(anchor.key, keys.logits)
    others, own_tile = anchor_split(narrow(p, dtype), at_anchor)
    if multiplier is None:
        return ((others, None),), others, own_tile, anchor
    parts = weights_parts(jnp.where(at_anchor, 0.0, p) * multiplier, dtype)
    anchor_draw = drawn(keys.coords, keys.coords.q_pos, anchor.key[:, None])
    anchor = anchor._replace(multiplier=None if anchor_draw is None else anchor_draw[:, 0])
    return parts, others, own_tile, anchor


def training_step(spec: Forward, carry: Carry, keys: Keys, rows: Gathered) -> tuple[Carry, Gathered]:
    """A training walk's carry after a tile `keys`: its state, whose sums are the weighted values kept apart
    from the anchor key's (`B`, with the initial state's), the anchor key, its own weight, and the other
    keys' weights' sum (keys alone). Where the anchor moved to this tile, the old anchor's term (its weight
    times its value row, `rows(key)` each row's at its key `[bq]`) joins `B` and the sum before the rescale
    (`split_taken`, `split_rows`), and the new anchor's weight is the tile's; the tile's other keys add their
    product and their narrowed weights. So neither is summed with a key's weight cancelled out of it. On the
    guarded walk the tile's product, sum and own weight take the
    row's shift past its weights' (`beyond_weights`). Where the mainloop holds the sums (`Keys.accumulate`,
    the plain walk's), they take the tile there. Returns the carry and `rows`."""
    state, arg, own, others_total = carry
    m, factor, p, value, held = _softmax(spec, state, keys)
    anchor = argmax(arg, keys.logits, keys.coords.k_pos)
    multiplier = drawn(keys.coords, keys.coords.q_pos, keys.coords.k_pos)
    parts, others, own_tile, anchor = _split_tile(keys, p, value.dtype, anchor, multiplier)
    moved = anchor.peak > arg.peak

    def taken(acc):
        return split_taken(acc, own, factor, moved, rows(arg.key), held, arg.multiplier)

    accumulate = keys.accumulate if held is None else None
    if accumulate is None:
        chunk = chunk_of(spec)
        times = functools.partial(_times, keys.dot, value, multiplier=multiplier, chunk=chunk)
        # `B` takes the old anchor's term before the factor (`split_taken`), its rounding kept where the walk
        # compensates (`kept_in_turn`), then the tile's other keys' products as a step takes them.
        low, term = state.low, (split_term(own, moved, rows(arg.key), arg.multiplier),)
        if low is None:
            joined = state._replace(acc=in_turn(state.acc, term))
        else:
            acc, acc_low = kept_in_turn(state.acc, low.acc, term)
            joined = state._replace(acc=acc, low=low._replace(acc=acc_low))
        after = stepped(joined, m, factor, p, products_of(parts, value, keys.dot, chunk), held, times)
        acc = after.acc
    else:
        # The tile's fp32 weights done with before its products, so they hold no registers through them.
        after = Partial(state.acc, m, denominator_after(state.total, factor, p))
        acc = accumulate(state.acc, taken, parts, value)
    own, others_total = split_rows(own, others_total, factor, moved, others, own_tile, held)
    return (after._replace(acc=acc), anchor, own, others_total), rows


def anchored(spec: Forward, start: Partial, arg: Argmax) -> Carry:
    """A training walk's carry before its first tile at each row's anchor known (`arg`: the first key at the
    keys' largest logit, from a walk of `argmax` over the logits alone, `anchored_step`): the start taken to
    the walk's final maximum, the larger of its own and the keys', which no tile moves, and no key's weight
    yet."""
    m = jnp.maximum(start.m, arg.peak)
    factor = carried(spec, start.m, jnp.where(m == -jnp.inf, 0.0, m))
    nothing = jnp.zeros_like(start.total)
    return Partial(expand(factor, start.acc) * start.acc, m, factor * start.total), arg, nothing, nothing


def anchored_step(spec: Forward, carry: Carry, keys: Keys) -> Carry:
    """`training_step` at the anchor and the final maximum known before the walk (`anchored`'s carry): the
    tile's weights at that maximum, so the sums take no factor and no old anchor's term ever joins them; the
    anchor's own weight from its own tile and the other keys' product and narrowed weights, each summed
    apart, as the gathering walk keeps them. No reduction finds a maximum or an anchor, and the sums the
    mainloop holds take the product in place, unscaled (`Keys.accumulate`'s `scale` None)."""
    state, arg, own, others_total = carry
    p = weights(spec, keys.logits, state.m)
    value = keys.load()
    parts, others, own_tile, arg = _split_tile(keys, p, value.dtype, arg,
                                               drawn(keys.coords, keys.coords.q_pos, keys.coords.k_pos))
    if keys.accumulate is None:
        acc = state.acc + summed_products(parts, value, keys.dot)
    else:
        acc = keys.accumulate(state.acc, None, parts, value)
    others_total = others_total + others.astype(jnp.float32).sum(axis=-1)
    return Partial(acc, state.m, state.total + p.sum(axis=-1)), arg, own + own_tile, others_total


def plain_walk(spec: Forward, lo, hi, keys_at: Callable[[jax.Array], Keys], start: Partial) -> Partial:
    """A program's walk of its key tiles `lo` to `hi` (`keys_at(index)`) from `start`, its sums compensated
    where its form takes them so (`compensated`, `settled`)."""
    return settled(jax.lax.fori_loop(lo, hi, lambda index, state: plain_step(spec, state, keys_at(index)),
                                     compensated(spec, start)))


def training_walk(spec: Forward, lo, hi, keys_at: Callable[[jax.Array], Keys],
                  rows: Callable[[jax.Array], jax.Array], start: Partial, dropping: bool = False) -> Carry:
    """A training forward's walk (`plain_walk`), its carry also holding the anchor key and the split
    sums, from none (`training_step`)."""
    nothing = jnp.zeros(start.total.shape, jnp.float32)
    state, arg, own, others_total = jax.lax.fori_loop(
        lo, hi, lambda index, carry: training_step(spec, carry, keys_at(index), rows)[0],
        (compensated(spec, start), no_argmax(start.total.shape[0], dropping), nothing, nothing))
    return settled(state), arg, own, others_total


def joined(carry: Carry, rows: Callable[[jax.Array], jax.Array]) -> Partial:
    """A training walk's state with its split sums joined, the anchor's term added back (`split_output`),
    the weighted values the output normalizes."""
    split, arg, own, _ = carry
    anchored_own = own if arg.multiplier is None else own * arg.multiplier
    return split._replace(acc=split_output(split.acc, anchored_own, rows(arg.key)))


def normalized_pass(spec: Forward, carry: Carry, keys_at: Callable[[jax.Array], Keys],
                    rows: Callable[[jax.Array], jax.Array], bounds,
                    initial: tuple[jax.Array, jax.Array] | None) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A guarded training walk's live rows' split sums, normalized (`split_residuals`): a pass over its key
    tiles `bounds` (`keys_at`) at the walk's own maximum and natural denominator (`normalized_terms`), from
    the `initial` state's share (`normalized_start`, `(m0, acc0)`): the other keys' weighted values, the
    anchor's weight and the other keys' weights, over the denominator; a weight below its dtype's least
    normal once normalized takes its product raised and in fp32, taken back after it (`SMALL_SHIFT`)."""
    split, arg, _, _ = carry
    state = joined(carry, rows)
    _, maximum, total = finish(spec, state, initial=initial is not None)
    share = None
    if initial is not None:
        # The initial state's factor to the walk's maximum, in the walk's own units, at most 1: folded, its
        # maximum taken to base 2 as the walk resumed from it (`resumed`), whose own may round back below it.
        m0 = initial[0].astype(jnp.float32)
        carried = jnp.where(m0 == -jnp.inf, 0.0,
                            jnp.exp2(jnp.minimum(base2_maximum(m0) - state.m, 0.0)) if spec.base2
                            else jnp.exp(jnp.minimum(m0 - maximum, 0.0)))
        share = carried, initial[1]

    def difference(logits):
        # Each exponent in the walk's own units: folded logits against its base-2 maximum, natural ones
        # against the natural maximum as the backward recomputes them.
        if spec.base2:
            return jnp.minimum(logits - expand(state.m, logits), 0.0)
        return exponent(logits, maximum, recompute(spec))

    def walk(index, sums):
        keys = keys_at(index)
        value = keys.load()
        rest, own_tile, small = normalized_terms(difference(keys.logits), total,
                                                 keys.coords.k_pos == arg.key[:, None], value.dtype,
                                                 jnp.exp2 if spec.base2 else jnp.exp)
        multiplier = drawn(keys.coords, keys.coords.q_pos, keys.coords.k_pos)
        if multiplier is None:
            weighed, weighed_small = rest, small
        else:
            # The product takes the dropped weights and the others' sums the undropped ones.
            weighed, _, weighed_small = normalized_terms(
                difference(keys.logits), total, keys.coords.k_pos == arg.key[:, None], value.dtype,
                jnp.exp2 if spec.base2 else jnp.exp, multiplier)
        lowered = jnp.float32(-SMALL_SHIFT)
        product = keys.dot(weighed, value) + power2(
            keys.dot(weighed_small, value.astype(jnp.float32)), lowered)
        others = rest.astype(jnp.float32).sum(axis=-1) + power2(small.sum(axis=-1), lowered)
        return sums[0] + product, sums[1] + own_tile, sums[2] + others

    nothing = jnp.zeros_like(total)
    lo, hi = bounds
    start = normalized_start(share, total, split.acc.shape[-1])
    return jax.lax.fori_loop(lo, hi, walk, (start, nothing, nothing))


def split_written(carry: Carry, rows: Callable[[jax.Array], jax.Array],
                  normalized: Callable[[], tuple[jax.Array, jax.Array, jax.Array]]
                  ) -> tuple[Partial, jax.Array, tuple[jax.Array, ...]]:
    """A training walk's state with its split sums joined (`joined`), its anchor keys, and its split sums
    and their shift for the backward (`split_residuals`, a guarded walk's live rows' from `normalized()`)."""
    split, arg, own, others_total = carry
    return joined(carry, rows), arg.key, split_residuals(split, own, others_total, normalized)


def argmax(state: Argmax, logits: jax.Array, k_pos: jax.Array) -> Argmax:
    """`Argmax` after a tile's logits (-inf where unseen) at its keys' indices `k_pos` `[1, bk]`."""
    tile_peak = logits.max(axis=-1)
    first = jnp.min(jnp.where(logits == expand(tile_peak, logits), k_pos, jnp.iinfo(jnp.int32).max), axis=-1)
    return Argmax(jnp.maximum(state.peak, tile_peak),
                  jnp.where(tile_peak > state.peak, first.astype(jnp.int32), state.key))


def anchor_split(narrowed: jax.Array, at_anchor: jax.Array) -> tuple[jax.Array, jax.Array]:
    """A tile's narrowed weights split at the anchor key (`at_anchor`): the others, the left operand of
    the product that adds to `B = sum_{j != c} e_j v_j`, and the anchor's own weight `[bq]`, fp32."""
    zero = jnp.zeros_like(narrowed)
    return (jnp.where(at_anchor, zero, narrowed),
            jnp.where(at_anchor, narrowed, zero).astype(jnp.float32).sum(axis=-1))


def split_taken(others: jax.Array, own: jax.Array, alpha: jax.Array, moved: jax.Array, old_value: jax.Array,
                held: Room | None = None, old_multiplier: jax.Array | None = None) -> jax.Array:
    """A training step's split sums before the tile's product joins them (`training_step`): where the
    anchor moved, the old anchor's term joined (`split_term`), then taken to the new maximum by `alpha`. A
    mainloop that accumulates the product in place applies this to its accumulator first."""
    return carry_factor(others + split_term(own, moved, old_value, old_multiplier), alpha, held)


def split_term(own: jax.Array, moved: jax.Array, old_value: jax.Array,
               old_multiplier: jax.Array | None = None) -> jax.Array:
    """The old anchor's term a training step's split sums join where the anchor moved (`moved` `[bq]`): its
    weight `own` before the tile (under dropout times its draw's multiplier `old_multiplier`) times its value
    row `old_value` `[bq, Dv]`; zero elsewhere."""
    old = old_value.astype(jnp.float32)
    anchored = own if old_multiplier is None else own * old_multiplier
    return jnp.where(expand(moved, old), expand(anchored, old) * old, 0.0)


def split_rows(own: jax.Array, others_total: jax.Array, alpha: jax.Array, moved: jax.Array, rest: jax.Array,
               own_tile: jax.Array, held: Room | None = None) -> tuple[jax.Array, jax.Array]:
    """A training step's row sums after a tile (`training_step`): the anchor's own weight (the tile's where
    the anchor moved to it), and the other keys' weights' sum, the old anchor's weight joining it where it
    moved."""
    total = (carry_factor(others_total + jnp.where(moved, own, 0.0), alpha, held)
             + beyond_weights(rest.astype(jnp.float32).sum(axis=-1), held))
    own_tile = beyond_weights(own_tile.astype(jnp.float32), held)
    return jnp.where(moved, own_tile, carry_factor(own, alpha, held)), total


SMALL_SHIFT = 63
"""The power of two `normalized_terms` raises a normalized weight below its values' dtype's least normal
by, half fp32's least normal's exponent: raised, such a weight on a value of any dtype is below `2^-63`
of fp32's largest (`2^49` of fp16's), so a sum of them stays finite over any key count a call holds, and
it is a normal fp32 down to `2^-189`."""


def normalized_terms(difference: jax.Array, total: jax.Array, at_anchor: jax.Array, dtype,
                     power: Callable[[jax.Array], jax.Array], multiplier: jax.Array | None = None
                     ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A tile's normalized weights `power(s - m) / T` (`difference`, each logit's exponent against the walk's
    maximum, at most 0, in the units `power` takes: `exp2` folded, `exp` natural) over the natural
    denominator (divided, as the reference's softmax divides), the reference's probabilities, narrowed to
    the values' `dtype`:
    zero at the anchor key (`at_anchor`) and where below the dtype's least normal, the left operand of the
    other keys' sum; the anchor's own, fp32; and the other keys' weights that normalizing takes below the
    least normal, raised by `2^SMALL_SHIFT` in fp32, `small`, whose product with the values the pass takes
    in fp32 and back after it (`normalized_pass`): each exponential raised as two halves, `e^(x/2) 2^k
    e^(x/2)`, so none passes through fp32's subnormals either. Normalized, no sum of them overflows (each
    is at most the values' largest), and a small weight on a large value keeps fp32's resolution of its
    share, so a column the reference resolves is resolved here, whatever another column's size or the
    denominator's."""
    seen = expand(total > 0, difference)
    denominator = expand(jnp.where(total > 0, total, 1.0), difference)
    p = jnp.where(seen, power(difference) / denominator, 0.0)
    if multiplier is not None:
        p = p * multiplier
    tiny = (p < float(jnp.finfo(dtype).tiny)) & jnp.logical_not(at_anchor)
    rounded = p.astype(dtype)
    zero = jnp.zeros_like(rounded)
    rest = jnp.where(at_anchor | tiny, zero, rounded)
    half = power(difference / 2)
    raised = jnp.where(seen, half * jnp.float32(2.0 ** SMALL_SHIFT) * half / denominator, 0.0)
    if multiplier is not None:
        raised = raised * multiplier
    small = jnp.where(tiny, raised, 0.0)
    return rest, jnp.where(at_anchor, rounded, zero).astype(jnp.float32).sum(axis=-1), small


def normalized_start(initial: tuple[jax.Array, jax.Array] | None, total: jax.Array, width: int) -> jax.Array:
    """The normalized split sums' start (`normalized_terms`): the initial state's weighted values at the
    walk's maximum over its natural denominator, `acc0 exp(m0 - m) / T` from `initial` `(exp(m0 - m), acc0)`,
    the state's factor to the walk's maximum (in the walk's units, `normalized_pass`) and its values (its
    share of `B`, which `split_delta` reads inside it), zero without one. The factor is at most 1 and the
    product is taken before the division, so it stays within `acc0`'s magnitude."""
    if initial is None:
        return jnp.zeros((total.shape[0], width), jnp.float32)
    carried, acc0 = initial
    acc0 = acc0.astype(jnp.float32)
    return expand(carried, acc0) * acc0 * expand(inverse(total), acc0)


def split_residuals(split: Partial, own: jax.Array, others_total: jax.Array,
                    normalized: Callable[[], tuple[jax.Array, jax.Array, jax.Array]]
                    ) -> tuple[jax.Array, ...]:
    """A training walk's split sums for the backward and their shift: as the walk left them (shift 0
    unguarded), and on the guarded walk a live row's normalized ones, which the mainloop's pass
    `normalized()` gives (the other keys' weighted values, the anchor's weight and the other keys' weights
    over the natural denominator, `normalized_terms`), with shift -1, which tells `split_delta` they are
    normalized; the plain walk takes no such pass."""
    sums = split.acc, own, others_total, jnp.zeros_like(own)
    if split.guard is None:
        return sums
    live = split.guard.live
    # The pass only where some row is live: a guarded program none of whose rows overflowed takes none.
    terms = jax.lax.cond(jnp.max(live.astype(jnp.int32)) > 0, normalized,
                         lambda: (jnp.zeros_like(split.acc), jnp.zeros_like(own), jnp.zeros_like(own)))
    fresh = (*terms, jnp.full_like(own, -1.0))

    def chosen(new, old):
        return jnp.where(expand(live, old) if old.ndim > own.ndim else live, new, old)
    return tuple(chosen(new, old) for new, old in zip(fresh, (*sums[:3], split.guard.shift), strict=True))


def split_output(others: jax.Array, own: jax.Array, anchor_value: jax.Array) -> jax.Array:
    """The weighted values of every key, `B` and the anchor's own term (`anchor_value`, its value row)."""
    return others + expand(own, others) * anchor_value.astype(jnp.float32)


def merge(spec: Forward, first: Partial, second: Partial) -> Partial:
    """Two states over disjoint keys as one, `first`'s keys before `second`'s: each taken to the larger
    maximum and summed, in that order. Either may hold no key whatever the call (a slice past a row's
    keys), so `empty` is its identity under every spec."""
    m = jnp.maximum(first.m, second.m)
    safe = jnp.where(m == -jnp.inf, 0.0, m)
    if first.guard is None and second.guard is None:
        a, b = carried(spec, first.m, safe), carried(spec, second.m, safe)
        acc = expand(a, first.acc) * first.acc + expand(b, second.acc) * second.acc
        return Partial(acc, m, a * first.total + b * second.total)
    ones, others = headroom_of(first), headroom_of(second)
    reach = jnp.maximum(ones.reach, others.reach)
    factors = carried(spec, first.m, safe), carried(spec, second.m, safe)
    natural = (power2(factors[0] * first.total, ones.shift)
               + power2(factors[1] * second.total, others.shift))

    def joined(shift):
        # Each state's factor, then its shift's change as a power of two on the product (`carry_factor`).
        a, b = factors
        drops = ones.shift - shift, others.shift - shift
        acc = (power2(expand(a, first.acc) * first.acc, expand(drops[0], first.acc))
               + power2(expand(b, second.acc) * second.acc, expand(drops[1], second.acc)))
        return acc, power2(a * first.total, drops[0]) + power2(b * second.total, drops[1])

    a, b = factors
    plain = expand(a, ones.plain) * ones.plain + expand(b, others.plain) * others.plain
    plain_total = a * ones.plain_total + b * others.plain_total

    # Joined at the larger shift; a row whose join overflows (two finite slices can) shifts as it needs,
    # every other row's bits the plain join's.
    held = jnp.maximum(ones.shift, others.shift)
    acc, _ = joined(held)
    live = overflowed(acc) | ones.live | others.live
    shift = jnp.where(live, jnp.maximum(held, needed(natural, reach)), held)
    acc, total = joined(shift)
    return Partial(acc, m, total, Guard(shift, reach, plain, plain_total, live))


def output_of(acc: jax.Array, denominator: jax.Array, *, guarded: bool) -> jax.Array:
    """The weighted values `acc` over their denominator, any leading axes: zero for a query that saw nothing
    where it may (`guarded`: a guarded walk's or one from a state)."""
    if not guarded:
        return acc / expand(denominator, acc)
    safe = expand(jnp.where(denominator > 0, denominator, 1.0), acc)
    return jnp.where(expand(denominator > 0, acc), acc / safe, 0.0)


def finished_rows(spec: Forward, state: Partial, *, initial: bool) -> tuple[jax.Array, jax.Array]:
    """The block's residual rows (`finish`'s): the maximum (0 for a query that saw nothing and started from no
    state, as the reference's) in the units the backward recomputes its logits in, natural, or raw products
    where the forward's are (`Forward.raw`), the products of the query its call's power of two took
    (`Forward.prescale`), and the denominator, unshifted (`room`)."""
    m = state.m if spec.raw else spec.naturalized(state.m)
    if spec.guarded or initial:
        m = jnp.where(m == -jnp.inf, 0.0, m)
    fields = state.guard
    if fields is None:
        return m, state.total
    # The natural denominator: the unshifted one where finite, else the shifted one taken back.
    return m, jnp.where(finite(fields.plain_total), fields.plain_total, power2(state.total, fields.shift))


def finish(spec: Forward, state: Partial, *, initial: bool, output_mod: Callable | None = None,
           output_params: tuple[jax.Array, ...] = ()) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The block's output `[bq, Dv]` fp32 (its weighted values over its denominator, zero for a query
    that saw nothing and started from no state, then the output hook) and its residual rows
    (`finished_rows`); the output is the shifted sums' ratio, which the shift leaves."""
    m, denominator = finished_rows(spec, state, initial=initial)
    out = output_of(state.acc, state.total, guarded=spec.guarded or initial)
    fields = state.guard
    if fields is not None:
        bounded = finite(fields.plain_total)
        # A live row's elements from the unshifted sums where they and their denominator are finite, as
        # the plain walk's; the shifted sums' ratio where they overflowed (`Guard.plain`). Every other
        # row shifted nothing, so its sums are the plain walk's own (a training walk's joined split sums).
        kept = (finite(fields.plain)
                & expand(fields.live & bounded & (fields.plain_total > 0), out))
        safe = expand(jnp.where(bounded & (fields.plain_total > 0), fields.plain_total, 1.0), out)
        out = jnp.where(kept, fields.plain / safe, out)
    if output_mod is not None:
        out = output_mod(out, *output_params)
    return out, m, denominator


# The backward's algebra: what lies between its products (S = Q·Kᵀ, dP = dO·Vᵀ, then dV, dK and dQ),
# on natural logits.


def inverse(total: jax.Array) -> jax.Array:
    """Each query's `1 / total`, zero where it saw nothing (total 0)."""
    return jnp.where(total > 0, 1.0 / jnp.where(total > 0, total, 1.0), 0.0)


def hooked(score_mod: Callable, logits: jax.Array, q_pos, k_pos, params) -> jax.Array:
    """`score_mod` on a tile; a custom-derivative call in it lowers as its primal function (Dew's JAX
    build, constraints.txt)."""
    return score_mod(logits, q_pos, k_pos, *params)


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


def pulled(cotangents, slopes, seen, kinds: tuple[Kind, ...],
           column_sums: Callable[[jax.Array], jax.Array] | None = None):
    """The score cotangents `cotangents` through `score_mod`, its entries' derivatives `slopes`
    (`scored`; None without a hook): the logits' tile, and each parameter's, a head's or a query's its
    queries' sums `[block_q]`, a key's its keys' `[block_k]` (`column_sums`, the mainloop's sum over a
    tile's rows where it takes one: `Family.column_sums`), a tile's the tile. Each entry is masked before
    any sum, so one the call never sees (padding, another document) adds nothing even where the hook's
    derivative there is not finite. fp32."""
    if slopes is None:
        return cotangents, ()
    logits = where_seen(seen, cotangents * slopes[0], 0.0)
    pieces = []
    for slope, kind in zip(slopes[1:], kinds, strict=True):
        entries = where_seen(seen, cotangents * slope, 0.0)
        key_sums = (lambda x: x.sum(axis=0)) if column_sums is None else column_sums
        pieces.append(key_sums(entries) if kind == Kind.KEY else entries if kind == Kind.TILE
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
    anchored = jnp.abs(about_anchor) <= jnp.abs(about_zero)
    return (jnp.where(anchored, anchor, 0.0), jnp.where(anchored, about_anchor, about_zero),
            jnp.where(anchored, key_index, -1))


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
    state_weight = state_total * inverse(total)
    cotangent = dout.astype(jnp.float32)
    anchor_value = anchor_value.astype(jnp.float32)
    anchor = jnp.sum(anchor_value * cotangent, axis=-1)
    others = others * expand(over, others)
    own, others_total = own * over, others_total * over
    about_zero = jnp.sum(split_output(others, own, anchor_value) * cotangent, axis=-1)
    about_anchor = (jnp.sum((others - expand(others_total, anchor_value) * anchor_value) * cotangent, axis=-1)
                    - state_weight * anchor)
    anchored = jnp.abs(about_anchor) <= jnp.abs(about_zero)
    return (jnp.where(anchored, anchor, 0.0), jnp.where(anchored, about_anchor, about_zero),
            jnp.where(anchored, key_index, -1))


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


def _dq_rows(sums: DqSums, weights: jax.Array, cotangents: jax.Array, slopes, seen: jax.Array | None, *,
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
    terms, _ = _dq_rows(sums, weights, cotangents, slopes, seen, kinds=kinds, summed=summed)
    return terms, pulled(cotangents, slopes, seen, kinds)[0] * scale


def keyed_terms(sums: DqSums, weights: jax.Array, prior: jax.Array, *, scale: float, dtype
                ) -> tuple[DqSums, Parts]:
    """The accurate form's keyed sums after a tile of a call without a score hook: their mass and exponent,
    and the parts of the weights' product with the keys (`weights_parts`, `DqSums.keyed`) a mainloop adds
    to them unscaled. Each coefficient is the scale times a weight over the prior denominator, which holds
    every weight at its maximum: the weights over it sum to at most one, so their product with the keys is a
    mean of them, finite wherever they are, and the scale is the mass's, its mantissa and exponent apart
    (`dq_finish`). One product, no exponent a row; a score hook's coefficients rescale a row's sums each
    tile instead (`dq_step`)."""
    scale_mantissa, scale_exponent = _static_split(scale)
    keyed = weights_parts(weights * expand(inverse(prior), weights), dtype)
    return (sums._replace(keyed_mass=jnp.full_like(sums.keyed_mass, scale_mantissa),
                          keyed_shift=jnp.full_like(sums.keyed_shift, scale_exponent)), keyed)


def dq_issued(sums: DqSums, weights: jax.Array, cotangents: jax.Array, slopes, seen: jax.Array | None, dtype,
              prior: jax.Array, *, scale: float, kinds: tuple, summed: tuple, accurate: bool
              ) -> tuple[DqSums, jax.Array, tuple[jax.Array | None, Parts] | None]:
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
        terms, keyed = keyed_terms(terms, weights, prior, scale=scale, dtype=dtype)
        return terms, parts, (None, keyed)
    terms, pulled_weights = _dq_rows(sums, weights, cotangents, slopes, seen, kinds=kinds, summed=summed)
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
            (before * over, weights_parts(coefficients * expand(over, coefficients), dtype)))


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


def below(logits: jax.Array, peak: jax.Array, how: Recompute = PLAIN) -> jax.Array:
    """`exp(s - m)` for a tile's logits recomputed in the backward against the forward's maximum `m`,
    fp32, in the forward's units (`Recompute`): `m` is every key's maximum, so a logit past it by the
    rounding the two passes do not share (the forward's maximum taken in base 2 and back, or another tile
    shape's product) is the maximum itself, its weight 1, never past it; in the accurate form one at it
    weighs 1 however a compiler contracts its scaling into the subtraction (`weights`)."""
    return jnp.exp(exponent(logits, peak, how))


def exponent(logits: jax.Array, peak: jax.Array, how: Recompute = PLAIN) -> jax.Array:
    """The natural exponent `s - m` (at most 0) of `below`'s weights."""
    peak = expand(peak, logits)
    if how.factor is not None:
        # Raw products: the difference is taken first, so the largest's is exactly 0.
        return jnp.minimum(logits - peak, 0.0) * how.factor
    if how.exact:
        return jnp.where(logits < peak, logits - peak, 0.0)
    return jnp.minimum(logits - peak, 0.0)


def normalized(logits: jax.Array, peak: jax.Array, total: jax.Array, how: Recompute = PLAIN
               ) -> tuple[jax.Array, jax.Array]:
    """A tile's unnormalized weights `exp(s - m)` (`below`) and its weights `p`, fp32."""
    exponentials = below(logits, peak, how)
    return exponentials, exponentials * expand(inverse(total), exponentials)


def dropped_products(products: jax.Array, multiplier: jax.Array | None) -> jax.Array:
    """A tile's score products as a dropping call takes them, `products * multiplier`, each rounded once
    before it is centred: fused into its subtraction (a contracted multiply-add) it would round differently
    against a centre that rounded it, so every centre and cotangent takes this one rounding."""
    if multiplier is None:
        return products
    return jax.lax.optimization_barrier(products * multiplier)


def grad_scores(exponentials: jax.Array, products: jax.Array, anchor: jax.Array, delta: jax.Array,
                seen: jax.Array | None, at_anchor: jax.Array | None, inverse_total: jax.Array,
                multiplier: jax.Array | None = None) -> jax.Array:
    """The score cotangents `p (dP - D)` of a tile in the reference's own order, as JAX differentiates
    `exp(s - m) / T`: each query's `(dP - c) - (D - c)` about its anchor c (`off_anchor_delta`) over its
    denominator (`inverse_total`), then times the unnormalized weights `exp(s - m)`, so every intermediate
    lies where the reference's does (a weight below fp32's range keeps its share beside a large product, and
    none overflows that the reference keeps); zero where unseen (`where_seen`). The anchor key's own
    `dP - c` is 0 by definition where `at_anchor` names it, whatever its product's rounding, so a query that
    sees one key has no cotangent, as the reference's: FlashAttention's order, `(dP - D) / T` with no anchor
    test, fails check 1 there in both dQ and dK (a query of one key, the stability suite's ordinary "one
    key" and causal windows' first queries: rows whose float64 gradient is exactly 0)."""
    products = dropped_products(products, multiplier)
    relative = products - expand(anchor, products)
    if at_anchor is not None:
        relative = jnp.where(at_anchor, 0.0, relative)
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
    residual: jax.Array
    mass: jax.Array


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
    k = jnp.asarray(exponent, jnp.float32)
    nearest = jnp.min(jnp.where(seen, jnp.abs(products), jnp.inf), axis=-1)
    first = jnp.min(jnp.where(seen & (jnp.abs(products) == expand(nearest, products)), positions,
                              jnp.iinfo(jnp.int32).max), axis=-1)
    picked = seen & (positions == first[:, None])
    tile_centre = jnp.sum(jnp.where(picked, products, 0.0), axis=-1)
    centre = jnp.where(state.seen, state.centre, tile_centre)
    scaled = power2(products, k) - expand(power2(centre, k), products)

    def row_sum(values):
        return (compensated_sum(values, (values.ndim - 1,)) if compensates_loops()
                else values.sum(axis=-1))

    return Centre(centre, state.seen | jnp.any(seen, axis=-1),
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


def _scaled(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`x` (fp32) times the power of two that puts its largest magnitude at 2^14, fp32, and that power's
    negated exponent, which takes a product of it back (`_fp16_scaled` before its rounding)."""
    top = jnp.max(jnp.abs(x))
    exponent = jnp.where(top > 0, jnp.ceil(jnp.log2(jnp.where(top > 0, top, 1.0))), 0.0)
    return power2(x, 14.0 - exponent), exponent - 14.0


def _fp16_scaled(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`_scaled`'s `x` in fp16 and its exponent: within fp16's range whatever `x`'s, an entry far below the
    largest losing only what is far below the product's own rounding."""
    scaled, exponent = _scaled(x)
    return scaled.astype(jnp.float16), exponent


BF16_LARGEST = float(jnp.finfo(jnp.bfloat16).max)
"""bf16's largest finite value, `(2 - 2^-7) 2^127`, an fp32 value."""


def _bf16(x: jax.Array) -> jax.Array:
    """fp32 `x` narrowed to bf16, saturating at its largest: an fp32 value past it by more than half its ulp
    (to fp32's largest, `(2 - 2^-23) 2^127`) rounds to infinity in bf16, so a finite score cotangent there
    would make an infinite product, or a NaN against a zero operand. Every value bf16 holds narrows as
    before; one past it narrows to its largest, the accurate form's later parts taking the difference whole
    (`cotangent_parts`), the fast form's within 2^-8 of it."""
    return jnp.clip(x, -BF16_LARGEST, BF16_LARGEST).astype(jnp.bfloat16)


def cotangent_parts(cotangents: jax.Array, dtype, *, accurate: bool) -> Parts:
    """The parts a tile's score cotangents `[bq, bk]` (fp32, scaled) take to their product with an operand of
    `dtype` (`cotangent_product`), each narrowed to it, with the power of two its product takes back (None:
    none), so a mainloop may issue their products into one accumulator. An fp32 operand's part is the
    cotangents themselves. The fast form (`accurate` False) narrows bf16's cotangents to bf16 for one product,
    FlashAttention's. fp16's take the accurate form's two products in both forms: one fp16 product of
    cotangents that cancel against equal keys near fp16's largest leaves a gradient past it once it returns in
    fp16 (three keys of 65504, values -65504, 16384 and 65504: 131008 where the exact one is 0 and the
    reference's finite), the stability rule's breach. The accurate form takes them to fp32's accuracy, as the
    reference differentiates (its score cotangents are fp32), so a key's or a query's gradient whose
    cotangents cancel across the other axis loses no more than the reference's fp32 product does, where one
    16-bit product loses every bit the cancellation exposes (bf16 small weights on large values: dK's worst
    row 3700x the reference's). A bf16 operand takes three bf16 products, the cotangents split exactly into
    three bf16 parts (each what the parts before it lost, rounded), 24 bits in all, with no scaling: bf16
    holds fp32's range, so no part leaves it (measured on the H100: 28% over one product, where a scaled
    bf16-and-fp16 pair cost 3x). An fp16 one takes two fp16 products at power-of-two scales that keep each
    part in its range: the cotangents' largest at 2^14, and what their rounding lost likewise, 21 bits."""
    return tuple(formed_parts(cotangents, dtype, accurate=accurate))


def formed_parts(cotangents: jax.Array, dtype, *, accurate: bool
                 ) -> Iterator[tuple[jax.Array, jax.Array | None]]:
    """`cotangent_parts` a part at a time, each formed only as it is taken: a mainloop that retires each
    part's product before taking the next (`Family.register_products`' `retired`) holds one part's registers
    beside the cotangents, not every part's."""
    if dtype == jnp.float32:
        yield cotangents, None
        return
    if dtype == jnp.bfloat16 and not accurate:
        yield _bf16(cotangents), None
        return
    if dtype == jnp.bfloat16:
        high = _bf16(cotangents)
        yield high, None
        rest = cotangents - high.astype(jnp.float32)
        middle = rest.astype(jnp.bfloat16)
        yield middle, None
        yield (rest - middle.astype(jnp.float32)).astype(jnp.bfloat16), None
        return
    scaled, high_exponent = _scaled(cotangents)
    high = scaled.astype(dtype)
    yield high, high_exponent
    low, low_exponent = _fp16_scaled(scaled - high.astype(jnp.float32))
    yield low, low_exponent + high_exponent


def cotangent_product(cotangents: jax.Array, operand: jax.Array,
                      dot: Callable[[jax.Array, jax.Array], jax.Array], *, accurate: bool) -> jax.Array:
    """A tile's score cotangents `[bq, bk]` (fp32, scaled) times the call's queries or keys `operand` by the
    mainloop's `dot` (dQ's `dS K`, `dot(a, b) = a @ b`; dK's `dSᵀ Q`, its transposed product
    `dot(a, b) = aᵀ @ b`, which a mainloop issues from where the tile lies), fp32: its parts' products
    (`cotangent_parts`), each taken back by its power of two, summed in order."""
    return summed_products(cotangent_parts(cotangents, operand.dtype, accurate=accurate), operand, dot)


def summed_products(parts: Parts, operand: jax.Array,
                    dot: Callable[[jax.Array, jax.Array], jax.Array]) -> jax.Array:
    """The sum of `parts`' products with `operand` by `dot` (`cotangent_parts`, `weights_parts`), each taken
    back by its power of two, in order."""
    def taken(part: jax.Array, exponent: jax.Array | None) -> jax.Array:
        product = dot(part, operand)
        return product if exponent is None else power2(product, exponent)

    (first, first_exponent), *rest = parts
    total = taken(first, first_exponent)
    for part, exponent in rest:
        total = total + taken(part, exponent)
    return total


def weights_parts(weights: jax.Array, dtype) -> Parts:
    """Weights `[bq, bk]` (fp32) narrowed for their product with an operand of `dtype` (`weights_product`):
    fp16's at a power-of-two scale that keeps them in its range (`_fp16_scaled`), which its product takes
    back."""
    if dtype == jnp.float16:
        return (_fp16_scaled(weights),)
    return ((weights.astype(dtype), None),)


def weights_product(weights: jax.Array, operand: jax.Array,
                    dot: Callable[[jax.Array, jax.Array], jax.Array]) -> jax.Array:
    """Weights `[bq, bk]` (fp32) times the queries or keys `operand` in its dtype, fp32 (`weights_parts`)."""
    return summed_products(weights_parts(weights, operand.dtype), operand, dot)


def narrowed(x: jax.Array, dtype) -> jax.Array:
    """`x` (fp32) narrowed to `dtype` at the scale `weights_parts` gives it, held in fp32: the values a
    product takes, which a dropped weight's (`x * multiplier`) overflow at the 16-bit scale never meets."""
    (part, exponent), = weights_parts(x, dtype)
    return part.astype(jnp.float32) if exponent is None else power2(part.astype(jnp.float32), exponent)



def value_operands(exponentials: jax.Array, weights: jax.Array, dout: jax.Array,
                   total: jax.Array, multiplier: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """The operands `(a, b)` of a tile's share of dV, `aᵀ·b` `[block_k, Dv]` in fp32 (a mainloop's
    transposed product, `cotangent_product`'s for dK): the unnormalized
    weights `exp(s - m)` narrowed to the operands' dtype as the reference's forward rounds them, against
    each query's `dout / total` in fp32, the reference's derivative's own order, so no small weight or
    share is lost to normalizing first. bf16 alone takes FlashAttention's form, its normalized weights
    `p` narrowed against the output's cotangent as it is, so the product stays on bf16's tensor cores: a
    query's cotangents that cancel across a key still cancel, and a normalized weight below bf16's least
    subnormal (about 9e-41) is dropped, which only a cotangent past about 1e28 would make count."""
    dropped = multiplier is not None
    if dropped:
        exponentials, weights = exponentials * multiplier, weights * multiplier
    if dout.dtype == jnp.bfloat16:
        return value_weights(exponentials, weights, dout.dtype, dropped), dout
    scaled = dout.astype(jnp.float32) * expand(inverse(total), dout)
    return value_weights(exponentials, weights, dout.dtype, dropped), scaled


def value_weights(exponentials: jax.Array, weights: jax.Array, dtype, dropped: bool = False) -> jax.Array:
    """`value_operands`' left operand for an output cotangent of `dtype`: bf16's normalized weights narrowed,
    else the unnormalized ones narrowed to `dtype` and held in fp32."""
    if dtype == jnp.bfloat16:
        return weights.astype(dtype)
    if dropped:
        return narrowed(exponentials, dtype)
    return exponentials.astype(dtype).astype(jnp.float32)
