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
from collections.abc import Callable, Iterator, Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dew.algebra import derive, rewritten
from dew.algebra.forms import Accumulated, Added, Merged, Operand
from dew.nn.kernels.templates import layout, product
from dew.nn.kernels.templates.attention import LOG2E, NATURAL, Capped, Draw, Initial, within
from dew.nn.precision import float32_einsum, two_product, two_sum


def folds(scale: float, prescale: float, *, natural: bool, score_mod) -> bool:
    """Whether a call folds log2(e) into its logits' scale for exp2 (`Forward.base2`, which says why), from
    the scale left past its queries' power of two (`attention.prescale`) and that power: the factor as the
    kernel multiplies the call's own products by it, rounded to fp32, below 1 in magnitude, the call not
    keeping its logits `natural` and no score hook reading them. The one owner of the choice, which a
    mainloop's derivation reads too (`attention_mosaic.MosaicForward`, row 2583)."""
    factor = abs(float(np.float32(scale * prescale * LOG2E)))
    return not natural and score_mod is None and factor < 1.0


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
    whether its walks keep what their running sums' additions drop (`compensated`, `compensates`), the
    power of two the queries took before their products (`prescale`, `attention.prescale`), which `scale`
    is the call's scale over, the keys a launch's walk reaches where past the call's though they are
    whole tiles (`walked_keys`: a holder of a common walk wholly past them, `seen` keeps every query from
    it; None where none does), and whether the launch lowers its exponentials to the hardware's
    approximations (`approximate`, `attention_mosaic_program.approximate`; false where a hook's tanh makes it
    lower every transcendental exactly)."""

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
    walked_keys: int | None = None
    approximate: bool = True
    subtracted: bool = False
    """Whether the fold (`base2`) takes its exponent as the accurate form's raw one does (`raw`), a difference
    of the raw products scaled after it is taken, `exp2((s - m) * scale * log2(e))`: no logit the reference
    keeps apart ties at the fold's size (`reaches`), so no walk walks again natural, for one more operation
    an element than the fold's contracted multiply-add, which a memory-bound walk (the decode kernel's,
    row 1955) does not feel."""

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
        # A raw fold resumes from an initial state's maximum taken to raw units (`raw_maximum`), which the
        # accurate form's exactness forgoes.
        kept = (self.subtracted and self.base2) or (self.natural and not self.initial)
        return kept and not self.scaled and self.score_mod is None and low <= self.scale <= high

    @property
    def reaches(self) -> bool:
        """Whether a walk's logits can pass the fold's reach (`FOLD_REACH`), past which the fold's rounding
        of each logit at log2(e) times its size ties logits the reference keeps apart, so a walk whose rows'
        maxima pass it walks again natural (`walked`, `walked_launches`): folded logits scaled before their
        difference (`base2`, not `raw`), but for a cap's (`Capped`, whose logits stay within it)."""
        cap = self.score_mod
        bounded = isinstance(cap, Capped) and abs(float(cap.softcap)) * LOG2E < FOLD_REACH
        return self.base2 and not self.raw and not bounded

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
        return folds(self.scale, self.prescale, natural=self.natural, score_mod=self.score_mod)

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
        state carried far, where `natural_exp` keeps that product's residual. A launch that lowers its
        exponentials exactly (`approximate` false) takes the fast form's `jnp.exp` as it is: its
        range reduction would repeat the lowering's own (the softcap's forward, 68 of its conversions to
        integers and back a tile)."""
        if self.base2:
            return fold_exp(x, self.scale * LOG2E) if self.raw else jnp.exp2(x)
        if not self.natural and not self.approximate:
            return jnp.exp(x)
        return natural_exp(x, self.scale) if self.raw else natural_exp(x)

    def naturalized(self, m: jax.Array) -> jax.Array:
        """A maximum in the logits' units taken to natural ones: from base 2 folded, from raw products
        raw."""
        if self.raw:
            return m * self.scale
        return m / LOG2E if self.base2 else m


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


def exponent(logits: jax.Array, peak: jax.Array, how: Recompute = PLAIN, *, safe: bool = False) -> jax.Array:
    """The natural exponent `s - m` (at most 0) of a tile's weights recomputed against the forward's maximum
    (`attention_gradient.below`'s, the forward's normalized pass's); unclamped where `peak` is
    `safe_peak`'s and the recompute plain, a logit past its maximum by the passes' rounding then weighing at
    most that rounding past 1."""
    peak = expand(peak, logits)
    if safe and how == PLAIN:
        return logits - peak
    if how.factor is not None:
        # Raw products: the difference is taken first, so the largest's is exactly 0.
        return jnp.minimum(logits - peak, 0.0) * how.factor
    if how.exact:
        return jnp.where(logits < peak, logits - peak, 0.0)
    return jnp.minimum(logits - peak, 0.0)


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
    return jax.lax.broadcast_in_dim(x, like.shape, tuple(range(like.ndim - x.ndim, like.ndim)))


class Coords(NamedTuple):
    """A tile's coordinates as the mainloop provides them (in the layout its products give): the query
    indices `q_pos` `[bq, 1]` and key indices `k_pos` `[1, bk]`, and where the spans narrow, each
    query's intervals `spans`, one a run (a call's runs, `KeySpans`; one where it has none), each its
    first and past-last keys `[bq]` (`intervals`), with documents the queries' and keys' ids `ids`,
    `[bq]` and `[bk]`, and with dropout its entries' draw `drop` (`Draw`). A tile whose rows are its keys
    (`keys_first`: `[bk, bq]`, a program's `K Qᵀ`) holds them so too, `q_pos` `[1, bq]` and `k_pos`
    `[bk, 1]`; its queries' and keys' vectors broadcast along its own axes (`queries`, `keys`). A
    mainloop that reads its walk's tile a part at a time states the tile's keys (`tile_keys`), which a
    part past the call's keys lies within (`key_count`)."""

    q_pos: jax.Array
    k_pos: jax.Array
    spans: tuple[tuple[jax.Array, jax.Array], ...] | None = None
    ids: tuple[jax.Array, jax.Array] | None = None
    drop: Draw | None = None
    keys_first: bool = False
    tile_keys: int | None = None
    key_start: jax.Array | None = None
    """Where given, `k_pos` as `key_start + key_offsets`: a start each thread holds across a row (its first
    key and the key bits its lane decides) and offsets its registers decide, constants the compiler folds
    into each compare, so a row's bound takes the start once (`seen`; FA3's causal mask, against a key place
    formed whole per entry: 36 bit operations a diagonal tile's trip at width 128 on an H100)."""
    key_offsets: jax.Array | None = None

    def queries(self, x: jax.Array, like: jax.Array) -> jax.Array:
        """A per-query `x` (`[bq]`) broadcast over the tile `like`."""
        return across(x, like) if self.keys_first else expand(x, like)

    def keys(self, x: jax.Array, like: jax.Array) -> jax.Array:
        """A per-key `x` (`[bk]`) broadcast over the tile `like`."""
        return expand(x, like) if self.keys_first else across(x, like)

    def query_places(self) -> jax.Array:
        """The tile's queries' places, one a query (`[bq]`): `q_pos` reduced along the tile's key axis (exact:
        a query's entries hold one place), which every mainloop's layouts take, a slice of a column not."""
        tile = jnp.broadcast_to(self.q_pos, jnp.broadcast_shapes(self.q_pos.shape, self.k_pos.shape))
        return jnp.max(tile, axis=0 if self.keys_first else -1)

    @property
    def key_axis(self) -> int:
        """The tile's axis of keys, which a softmax reduces along: its rows where they are its keys."""
        return 0 if self.keys_first else -1

    def key_count(self, like: jax.Array) -> int:
        """The keys the walk's tile holds: `like`'s, or the tile's whole where `like` is a part of it
        (`tile_keys`)."""
        return like.shape[self.key_axis] if self.tile_keys is None else self.tile_keys

    def hooked(self, score_mod: Callable | None) -> Callable | None:
        """`score_mod`, written over `[bq, bk]` tiles, as this tile's layout runs it: taken axis for axis on a
        tile held keys first (`layout.transposed`)."""
        return layout.transposed(score_mod) if self.keys_first and score_mod is not None else score_mod


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
    plain: jax.Array | None = None
    plain_total: jax.Array | None = None


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


LEAST = float(jnp.finfo(jnp.float32).min)
"""A walk's running maximum before any key: fp32's least finite value, the derived forms' floor of it
(`dew.algebra.forms`' init), which every state a step or a merge meets is at or above, so no factor between
two states is a NaN. A real logit may be it too (a hook's mask by fp32's least, as transformers' are), so a
row that saw no key is one of no denominator (`no_keys`), never one at `LEAST`: its factor onto a key's
maximum underflows to 0 as -inf's did, onto another such state's is 1 over zero sums."""


def no_keys(total: jax.Array) -> jax.Array:
    """Whether each row of a state's denominator `total` saw no key: none of its sums, as any key's weight
    at the row's maximum is 1 (a guarded walk's shifted by its room, never to 0)."""
    return total == 0


def empty(rows: int, width: int) -> Partial:
    """The softmax before any key: `merge`'s identity, its maximum `LEAST`."""
    return Partial(jnp.zeros((rows, width), jnp.float32), jnp.full((rows,), LEAST, jnp.float32),
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

    if not spec.reaches:
        return finished(spec, guarded_walk(spec, functools.partial(walked_plain, spec)))
    carry = walk(spec, None)
    state = carry if isinstance(carry, Partial) else carry[0]
    reach = reach_of(state.m, unseen=spec.form == "spans")
    natural = unfolded(spec)

    def folded() -> C:
        return carry

    def unfolded_walk() -> C:
        return guarded_walk(natural, functools.partial(walked_plain, natural))

    return jax.lax.cond(reach >= FOLD_REACH, lambda: finished(natural, unfolded_walk()),
                        lambda: finished(spec, guarded_walk(spec, folded)))


def reach_of(maxima: jax.Array, *, unseen: bool) -> jax.Array:
    """How far a folded walk's row `maxima` reach (`walked`, against `FOLD_REACH`): the largest of their
    reaches (`reached`)."""
    return jnp.max(reached(maxima, unseen=unseen))


def reached(maxima: jax.Array, *, unseen: bool) -> jax.Array:
    """How far each of a folded walk's row `maxima` reaches: its magnitude, and one whose maximum the fold
    took past fp32's largest, or to no number, past every reach, so the row walks again natural, where its
    logits are finite (a product near fp32's largest before a small scale, times log2(e)). A -inf maximum is
    a row that saw no key, none, where a row may see none (`unseen`, a 'spans' call); elsewhere every row
    sees a key, so it is a row whose products all passed fp32 below zero, past every reach
    (`attention_mosaic_forward._prepared`'s raw products)."""
    reach = jnp.where(jnp.isnan(maxima), jnp.inf, jnp.abs(maxima))
    return jnp.where(maxima == -jnp.inf, 0.0, reach) if unseen else reach


type Launch[O, F] = Callable[..., tuple[O, jax.Array, jax.Array] | tuple[O, None, F]]
"""A whole launch of a call in a form (`walked_launches`), `launch(form, live, again=False, into=None)`: its
outputs, each row's running maximum and whether its weighted values overflowed; or, a launch whose programs
state their rows' rewalk flags themselves (`F`, its mainloop's own), its outputs, no maxima and those flags,
which it takes back as `live`; `again` a plain launch run
once more inside the guard's rare branch, a recomputation (its kernel named apart, its products counted
once); `into` refs holding the plain launch's outputs, which it writes its rows `live` into in place
(`walked_launches`' `in_place`)."""


def finished_form(spec: Forward) -> Forward:
    """The form a walk's finished states merge in (`slice_state`, `finished_merged`, `merged_slices`): its
    own where `finish` leaves its maxima in its units (natural, or a raw fold's products), else its fold's
    natural form (`unfolded`), whose units a folded walk's finished maxima are in (`finished_rows`)."""
    return spec if spec.natural or spec.raw else unfolded(spec)


def slice_state(spec: Forward, out: jax.Array, maximum: jax.Array, total: jax.Array) -> Partial:
    """A finished slice of a row's keys as a state of their merge (`finished_merged`): its output `out`
    normalized (`finish`'s), its maximum (in `spec`'s units, `finished_rows`', raw products where its fold
    keeps them) and its denominator folded into its maximum (`m + log(total)`, in those units) and its mass
    1 where it saw a key, so no output multiplied back by its denominator passes fp32's largest; a slice of
    no key the walk's start (`LEAST`), its mass 0. A row or a launch's block of rows alike."""
    seen = total > 0
    units = spec.scale if spec.raw else 1.0
    folded = jnp.where(seen, maximum + jnp.log(jnp.where(seen, total, 1.0)) / units, LEAST)
    return Partial(out, folded, seen.astype(jnp.float32))


def initial_state(spec: Forward, initial: Partial) -> Partial:
    """A call's initial state (its sums unnormalized, its maximum natural) as a finished slice's
    (`slice_state`): normalized where it holds mass, its maximum in `spec`'s units."""
    held = initial.total > 0
    out = jnp.where(held[..., None], initial.acc / jnp.where(held, initial.total, 1.0)[..., None], 0.0)
    return slice_state(spec, out, initial.m / spec.scale if spec.raw else initial.m, initial.total)


def merged_slices(spec: Forward, out: jax.Array, maximum: jax.Array, total: jax.Array,
                  initial: Initial | None = None) -> jax.Array:
    """A split walk's slices merged (`merged` of `slice_state`s, then `finish`): their outputs `out`
    `[B, S, Q, KV, Dv]` (fp32, each normalized), maxima (in natural `spec`'s units, `finished_rows`', a
    fold's `unfolded`, raw products where it keeps them) and denominators `[B, KV, S, Q]`, and the call's
    `initial` state `[B, KV, Q]` (its sums unnormalized): the rows' output `[B, Q, KV, Dv]` fp32, a row of
    no mass zero, as the reference's."""
    if finished_form(spec) is not spec:
        raise ValueError("a split walk's merge takes its finished form's states (`finished_form`)")
    state = slice_state(spec, out, *(jnp.transpose(array, (0, 2, 3, 1)) for array in (maximum, total)))
    if initial is not None:
        own = initial_state(spec, Partial(jnp.swapaxes(initial.acc, 1, 2), jnp.swapaxes(initial.m, 1, 2),
                                          jnp.swapaxes(initial.total, 1, 2)))
        acc, m, total = (jnp.concatenate([mine, theirs[:, None]], axis=1)
                         for mine, theirs in zip(state[:3], own[:3], strict=True))
        state = Partial(acc, m, total)
    return finish(spec, finished_merged(spec, state, axis=1), initial=True, normalized=True)[0]


def finished_merge(spec: Forward, first: Partial, second: Partial) -> Partial:
    """Two finished states over disjoint keys (`slice_state`'s, or merges of them) as one merge of them,
    `first`'s keys before `second`'s: `finished_merged`'s convex combination of two, elementwise, as a
    program's registers hold them (the in-launch merge's, `attention_mosaic_forward._Merge`)."""
    m = jnp.maximum(first.m, second.m)
    a, b = carried(spec, first.m, m) * first.total, carried(spec, second.m, m) * second.total
    total = a + b
    over = 1.0 / jnp.where(total > 0, total, 1.0)
    out = expand(a * over, first.acc) * first.acc + expand(b * over, second.acc) * second.acc
    return Partial(out, m, total)


def finished_merged(spec: Forward, stacked: Partial, axis: int) -> Partial:
    """Finished states over disjoint keys (`slice_state`'s, their outputs normalized, or merges of them)
    stacked on `axis` as one merge of them: their outputs' convex combination, each weighed by its mass at
    the largest of their maxima (`merge`'s factors) over those weights' sum before its product, so the merged
    output is bounded by its states' (their factors' products summed first passed fp32's largest where 8
    slices' outputs near 3e38 merged, row 4526), its denominator those weights' sum, never folded into its
    maximum again (a fold a merge rounds away at a large maximum loses the mass). Its output is its sums
    (`finish`'s `normalized`); a state of no key adds nothing."""
    m = jnp.max(stacked.m, axis=axis)
    weights = carried(spec, stacked.m, jnp.expand_dims(m, axis)) * stacked.total
    total = jnp.sum(weights, axis=axis)
    shares = weights * jnp.expand_dims(1.0 / jnp.where(total > 0, total, 1.0), axis)
    out = jnp.sum(expand(shares, stacked.acc) * stacked.acc, axis=axis)
    return Partial(out, m, total)


def rewalks(spec: Forward, maxima: jax.Array, over: jax.Array) -> tuple[jax.Array, Forward]:
    """The rows a plain launch's guards walk again (`walked_launches`), from its rows' running `maxima` and
    whether their weighted values `over`flowed, and the form they walk in (`rewalked_form`): where the fold
    reaches (`Forward.reaches`), those whose maximum passes `FOLD_REACH` (`reached`: a padded query's -inf, a
    row that reads nothing, as a slice's of no keys) or that overflowed; else those that overflowed. A
    launch's programs state them so of their own rows."""
    if spec.reaches:
        return (reached(maxima, unseen=spec.form == "spans") >= FOLD_REACH) | over, rewalked_form(spec)
    return over, rewalked_form(spec)


def rewalked_form(spec: Forward) -> Forward:
    """The form a plain launch's guards walk its rows again in (`rewalks`): the natural form where the fold
    reaches (`unfolded`), else the call's."""
    return unfolded(spec) if spec.reaches else spec


def walked_launches[O, F](spec: Forward, launch: Launch[O, F], kept: Callable[[jax.Array, O, O], O], *,
                          masked: bool = False, in_place: bool = False) -> O:
    """`walked`'s guards over whole launches, for a mainloop whose program walks its keys once (a producer
    streams them, so no program walks again): `launch(form, live)` runs the call in `form`, plain where
    `live` is None, else with its rows `live` shifting (`guarded`), and returns its outputs, each row's
    running maximum in the walk's units and whether its weighted values overflowed (`overflowed`), `[..., Q]`
    each; `kept(live, guarded, plain)` takes the guarded launch's outputs at the rows `live`, the plain
    one's elsewhere, so every other row's bits are the plain walk's. The call walks again where `walked`'s
    program would, decided over the whole call: in the natural form (`unfolded`) where some row's maximum
    passes `FOLD_REACH` folded, guarded as the call's form is; under headroom, the rows whose weighted values
    overflowed. Ordinary inputs take the one plain launch, their cost unchanged but for these checks.

    `masked` (a spans call's, whose launch walks only its live rows' keys, every other row's none:
    `launch(form, live)`'s program over its rows' spans) takes one more launch every time, no branch: a
    branch's predicate is read on the host before its launch may run, a decode step's round trip (row 1978:
    0.280 ms wall against 0.015 on the device), where a launch of empty walks costs its prologue. It walks
    in the natural form (`unfolded`) the rows the fold's reach passes `FOLD_REACH` a row (`reached`) and
    those whose weighted values overflowed, guarded as that form is, and `kept` takes each of its outputs
    (its rows' finished maxima and denominators too) at those rows: the fold's plain launch costs its
    exponential's two instructions an element where the natural form's costs about sixty
    (ET-DECSASS-h100, ET-FOLDED-h100: Qwen3-8B's decode at 32 rows of 32768 keys, 1.35 ms folded against
    2.11 natural, the H100's bandwidth floor 1.28).

    `in_place` walks so, masked, every call, its rewalk written into the plain launch's outputs held in refs
    (`launch(form, live, into=refs)`: the launch writes its rows `live` and leaves every other row's bits),
    no branch and no `kept` over them: the branch's common side copied the plain launch's outputs into its
    result every call (ET-MEMCPY-h100: the output and its rows, 45 us of Qwen3-8B's causal forward on an
    H100), and the masked launch's `kept` selected over them every call; its empty launch costs its
    prologue."""
    if masked or in_place:
        plain = launch(spec, None)
        outputs = plain[0]
        if not (spec.reaches or spec.headroom):
            return outputs
        # A launch whose programs state its rows' rewalk flags themselves (`rewalks`) gives them in its
        # overflows' place and no maxima, and rewalks in place: no array operation between the launches
        # (ET-DPROF-h100: the two fusions forming them took 6.3 us of a batch-1 decode step's 30.7).
        if plain[1] is None or in_place:
            live, form = ((plain[2], rewalked_form(spec)) if plain[1] is None
                          else rewalks(spec, plain[1], plain[2]))
            held = jax.tree.map(jax.new_ref, outputs)
            launch(form, live, into=held)
            return jax.tree.map(jax.freeze, held)
        live, form = rewalks(spec, plain[1], plain[2])
        return kept(live, launch(form, live)[0], outputs)

    def plain_launch(form: Forward) -> tuple[O, jax.Array, jax.Array]:
        result = launch(form, None)
        assert result[1] is not None, "a launch states its rows' rewalk flags in a masked walk alone"
        return result

    def guarded_launch(form: Forward, outputs: O, over: jax.Array) -> O:
        if not form.headroom:
            return outputs
        # The rows kept plain read from the plain launch run again inside the rare branch (`again`), so the
        # plain launch's outputs reach one branch alone, whose result their buffer then is: read by both,
        # XLA copied them in the common branch (ET-GUARDHLO-h100: `copy.7`, the whole output, 0.047 ms of
        # Qwen3-8B's 0.66 ms causal forward).
        return jax.lax.cond(jnp.max(over.astype(jnp.int32)) > 0,
                            lambda: kept(over, launch(form, over)[0], launch(form, None, again=True)[0]),
                            lambda: outputs)

    outputs, maxima, over = plain_launch(spec)
    if not spec.reaches:
        return guarded_launch(spec, outputs, over)
    # The call's own rows alone: a padded query's maximum is -inf, a row that reads nothing (`seen`).
    reach = reach_of(maxima if spec.queries is None else maxima[..., :spec.queries],
                     unseen=spec.form == "spans")
    natural = unfolded(spec)

    def unfolded_launch() -> O:
        unfolded_outputs, _, unfolded_over = plain_launch(natural)
        return guarded_launch(natural, unfolded_outputs, unfolded_over)

    return jax.lax.cond(reach >= FOLD_REACH, unfolded_launch, lambda: guarded_launch(spec, outputs, over))


NATURAL_EXPONENT_REACH = 150.0
"""The largest `|k|` `natural_exp` hands `power2`: its reduced exponent's integral part, its argument held to
[-150, 128] (every fp32 exponential outside is zero or infinite), rounded up by at most one."""

POWER2_REACH = 381.0
"""The largest `|k|` `power2` takes apart: past it every fp32 `x * 2^k` is zero or infinite (`2^-381` of
fp32's largest flushes, `2^381` of its least overflows), so `k` is held there."""


def exact_power2(k: jax.Array) -> jax.Array:
    """`2^k` exactly for an integral `k` in [-126, 127], built from its fp32 exponent bits: no `exp2`,
    which XLA computes as `exp(k ln 2)`, inexactly, and whose products it fuses into one `exp` of their
    sum (its algebraic simplifier's `exp(a) exp(b) -> exp(a + b)`), past fp32's range where each factor
    is in it."""
    biased = jnp.asarray(k).astype(jnp.int32) + jnp.int32(127)
    bits = jax.lax.shift_left(biased, jnp.full_like(biased, 23))
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def power2(x: jax.Array, k: jax.Array, reach: float = POWER2_REACH) -> jax.Array:
    """`x * 2^k` for an integral `k` of any size, exactly wherever the result is a normal fp32: three exact
    factors each within 2^±127 (`exact_power2`), of `k`'s sign, so each partial product lies between `x`
    and the result; `k` held within `reach` (`POWER2_REACH`, past which the result is zero or infinite, by
    default). `k` is carried as the integer its contract makes it, so its thirds and halves are integer
    divisions, exact, floored, which keeps each part of `k`'s sign, and no factor converts it again. A caller
    whose `k` stays within twice 127 states its `reach` and takes two factors, each built as its product
    takes it: one integer tile live, not two (et-livehook1: power2's factor tiles 66 of the natural forward's
    281 registers at its peak)."""
    k = jnp.clip(jnp.asarray(k, jnp.float32), -reach, reach)
    if reach <= 2 * 127:
        # The half truncated from an exact multiply by its conversion, `k` an integer only at each factor's
        # bias: an int32 `k` halved (a shift or a floored division alike) spilled an A100's recurrence
        # query_key launch 32 bytes at the 27B gated delta net's training tiles, the float half none
        # (gate_one, ledger 2519); a float `trunc` compares each entry's sign, whose predicates over a
        # 32-entry fragment pass a thread's seven and ptxas parks them in general registers, holding the
        # exponent's parts live to their selects (an A100's hooked training forwards at width 128 spilled
        # 496 to 588 bytes; the conversion's own truncation none).
        half = (k * np.float32(0.5)).astype(jnp.int32)
        return x * exact_power2(half) * exact_power2((k - half.astype(jnp.float32)).astype(jnp.int32))
    k = k.astype(jnp.int32)
    first = k // 3
    second = (k - first) // 2
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
    # The exponent within [-150, 129]: two factors take it (`power2`'s `reach`).
    return power2(jnp.exp2((bounded - exponent) + lo), exponent, reach=NATURAL_EXPONENT_REACH)


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


def denominator_after(total: jax.Array, factor: jax.Array, p: jax.Array, *, axis: int = -1) -> jax.Array:
    """A row's denominator after a tile on the plain walk: the state's taken by `factor` (`carry_factor`)
    and the tile's weights `p` added, their keys along `axis` (the guarded walk's, `stepped`)."""
    return carry_factor(total, factor, None) + p.sum(axis=axis)


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
    if kept is None or low is None or low.plain is None or low.plain_total is None:
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
    if guard is not None and low.plain is not None and low.plain_total is not None:
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


def raw_maximum(m: jax.Array, scale: float) -> jax.Array:
    """A natural initial maximum (a sink's logit) in a raw fold's units (`Forward.subtracted`), `m / scale`,
    saturated at fp32's largest where that quotient passes it, as `base2_maximum` saturates: a raw product is
    finite, so a maximum past fp32's largest exceeds every key's and weighs them all zero, as its saturation
    does."""
    raw = m / np.float32(scale)
    largest = float(jnp.finfo(jnp.float32).max)
    return jnp.where(infinite(raw) & finite(m), jnp.sign(m) * largest, raw)


def resumed(spec: Forward, m: jax.Array, total: jax.Array, acc: jax.Array, *,
            live: jax.Array | None = None) -> Partial:
    """The softmax resumed from a state `(m, total, acc)` (`Hooks.initial`: sinks, a resumed partial),
    its maximum taken to base 2 where the step's logits are; on the guarded path (`live`, the rows that
    may shift, `guarded`) those rows' sums shifted as their size needs (`needed`: `acc`'s largest
    magnitude is its denominator times a bound on the values it weighed). A row of no key (a maximum of
    -inf: a padded row, a slice carrying no sink) resumes at the walk's start (`LEAST`), exact for every
    other."""
    m, total, acc = (array.astype(jnp.float32) for array in (m, total, acc))
    if spec.base2:
        m = raw_maximum(m, spec.scale) if spec.raw else base2_maximum(m)
    m = jnp.maximum(m, LEAST)
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
    them or a launch's walk reaches a tile wholly past them (`Forward.walked_keys`: a common walk's holder);
    and, where the spec names them, its queries (a padding row sees no key, so a score hook's derivative
    there, undefined at its padded parameters, reaches no reduction over the rows)."""
    visible = None
    if coords.spans is not None:
        ids = None if coords.ids is None else (coords.queries(coords.ids[0], like),
                                               coords.keys(coords.ids[1], like))
        visible = functools.reduce(jnp.logical_or, (within(coords.k_pos, coords.queries(start, like),
                                                           coords.queries(end, like), ids)
                                                    for start, end in coords.spans))
    elif spec.form == "causal":
        visible = _keys_before(coords, coords.q_pos + 1)
    if spec.form != "spans" and (spec.keys % coords.key_count(like) or spec.walked_keys is not None):
        # A tile past the call's last key reads padding, a holder of a common walk wholly past it too.
        past = _keys_before(coords, spec.keys)
        visible = past if visible is None else visible & past
    if spec.form != "spans" and spec.queries is not None:
        rows = coords.q_pos < spec.queries
        visible = rows if visible is None else visible & rows
    return visible


def _keys_before(coords: Coords, bound) -> jax.Array:
    """Whether each key's place lies before `bound`: against its offsets past the row's start where the
    coordinates split its places (`Coords.key_start`), one subtraction a row."""
    if coords.key_offsets is None:
        return coords.k_pos < bound
    return coords.key_offsets < bound - coords.key_start


def places_read(spec: Forward, *, masked: bool, hooked: tuple[bool, bool], dropped: bool,
                padded: bool) -> tuple[bool, bool]:
    """Whether a tile's step surely reads its queries' places and its keys' (`Coords.q_pos`, `k_pos`): its
    score hook where it reads them (`hooked`, queries' and keys'), its dropout's draw (`dropped`, `drawn`),
    and where `masked` (`seen`) a causal call's diagonal, a spans call's intervals (its keys'), a call's
    padded queries and, where its keys pass a whole tile's (`padded`), its keys."""
    queries = hooked[0] or dropped or (masked and spec.form != "spans"
                                       and (spec.form == "causal" or spec.queries is not None))
    keys = hooked[1] or dropped or (masked and (spec.form in ("spans", "causal") or padded))
    return queries, keys


def peak(spec: Forward, m: jax.Array, s: jax.Array, products: jax.Array | None = None, *, axis: int = -1
         ) -> tuple[jax.Array, jax.Array]:
    """The running maximum after a tile's logits `s`, and `alpha`, the factor the state before it
    takes to the new maximum. In the accurate form the exponent `m - new` is compensated (its subtraction's
    rounding, by TwoSum, carried to first order): every weight before the new maximum shares this one
    factor, so its exponent's rounding would not average out over them as the reference's per-key exponents'
    do, an error some sqrt(keys) times the reference's where the maximum jumps far. Where the logits are
    `products` times `Forward.factor` (`logits_of`), the tile's maximum is their extreme times it, the
    same bits (fp32's rounding is monotonic: the largest of `fl(p c)` is `fl(max(p) c)` for a positive `c`,
    `fl(min(p) c)` for a negative one), taken once a row. The tile's keys are along `axis` (its rows where
    they are its keys, `Coords.key_axis`)."""
    if products is None:
        tile = s.max(axis=axis)
    else:
        tile = (products.max(axis=axis) if spec.factor > 0 else products.min(axis=axis)) * spec.factor
    return peaked(spec, m, jnp.maximum(m, tile))


def peaked(spec: Forward, m: jax.Array, new: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`peak`'s running maximum `new` (at least `m`) and the factor the state before it takes from `m` to
    it: a mainloop that takes a tile's maximum a part of the tile at a time (`peak` of each) states the
    tile's whole by it."""
    return new, carried(spec, m, new)


def carried(spec: Forward, m: jax.Array, safe: jax.Array) -> jax.Array:
    """The factor `exp(m - safe)` a state takes from its maximum `m` to `safe` (`peak`, `merge`), in the
    accurate form with its exponent compensated (`natural_carried`)."""
    if not spec.natural:
        return spec.power(m - safe)
    return natural_carried(m, safe, raw=spec.raw, scale=spec.scale)


def fold_exp(x: jax.Array, scale: float = 1.0) -> jax.Array:
    """The fold's exponential of `x` times `scale` (`exp2(x c)`, a raw fold's weight of its products'
    difference `x`, `Forward.subtracted`), the factor rounded to fp32 as the kernels multiply by it."""
    return jnp.exp2(x * np.float32(scale))


def natural_carried(m: jax.Array, safe: jax.Array, *, raw: bool, scale: float) -> jax.Array:
    """The accurate form's rescale `exp(m - safe)` (`carried`; the derived forms' declared rescale,
    `declaration`), its exponent compensated: of raw products (`Forward.raw`) at their `scale`, else of
    scaled logits."""
    power = functools.partial(natural_exp, scale=scale) if raw else natural_exp
    # TwoSum: m - safe is difference + error exactly; raw, the scale's product with it is split exactly too
    # (TwoProduct over truncated halves, whose products are exact, so a compiler's contraction of any of
    # them into a fused multiply-add changes no bit); exp's derivative is itself.
    difference, error = two_sum(m, -safe)
    alpha = power(difference)
    if raw:
        product, rounding = two_product(difference, np.float32(scale), bit_split=True)
        alpha = natural_exp(product)
        error = rounding + error * scale
    corrected = alpha * (1.0 + error)
    return jnp.where(finite(corrected), corrected, power(difference))


def weights(spec: Forward, s: jax.Array, m: jax.Array, coords: Coords | None = None) -> jax.Array:
    """A tile's unnormalized weights `[bq, bk]` under the running maximum `m`, fp32. The fold computes
    `exp2(s c - m)`, which a compiler contracts to `fma(s, c, -m)` (FlashAttention 2's by design): the
    largest logit's exponent is then its product's rounding, up to an ulp of the logit, so at logits near
    fp32's largest its weight is 0 or infinite, one of the fold's documented divergences. In the accurate
    form (`Forward.natural`) the largest logit weighs 1 exactly: raw logits subtract before they scale
    (`Forward.raw`), and scaled ones (under a hook or an initial state) hold it to 1 by a select. The
    maximum a query's broadcast over its keys as `coords` hold them (its rows by default)."""
    peak = expand(m, s) if coords is None else coords.queries(m, s)
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


Parts = tuple[jax.Array | product.Scaled, ...]
"""A product's parts as a mainloop issues them (`cotangent_parts`, `weights_parts`): each a narrowed
operand with the power of two its product takes back (None: none)."""


class Keys(NamedTuple):
    """A key tile as a mainloop gives it the algebra: its coordinates and (modified) logits `[bq, bk]`, its
    values `[bk, Dv]` as an operand of their type (`Operand`), which a step loads after the softmax (where
    jax's mha loads them; a derived step's evaluation where its form first reads them), and on
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
    values: Operand
    dot: Callable[[jax.Array, jax.Array], jax.Array]
    accumulate: Callable[[Any, Callable | None, Parts, jax.Array], Any] | None = None
    products: jax.Array | None = None
    largest: Callable[[], jax.Array] | None = None


Carry = tuple[Partial, Argmax, jax.Array, jax.Array]
"""A training walk's carry: its state, whose values are `B`, every key's but the anchor's
(`training_step`), its anchor key, the anchor's own weight and the other keys' weights' sum."""


def drawn(coords: Coords, queries: jax.Array, keys: jax.Array) -> jax.Array | None:
    """A tile's dropout multipliers at `queries` and `keys` (`Coords.drop`), None where no call drops. A
    tile whose coordinates are its entries (a Mosaic program's, `queries` its `q_pos`) mixes each query's
    word once, on its vector (`Draw.words`), its keys entry for entry: the same bits, half the hashing."""
    if coords.drop is None:
        return None
    # A tile whose rows are its queries reduces along its own rows; one whose rows are its keys (the keys
    # first, `Coords.keys_first`) would reduce across its warps, through scratch its launch holds none of.
    if (queries is coords.q_pos and jnp.shape(queries) == jnp.shape(keys) and jnp.ndim(keys) == 2
            and not coords.keys_first):
        words = coords.queries(coords.drop.words(coords.query_places()), keys)
        return coords.drop.multiplied(coords.drop.kept(words, keys))
    return coords.drop(queries, keys)



def _times(dot: Callable[[jax.Array, jax.Array], jax.Array], value: jax.Array, p: jax.Array,
           multiplier: jax.Array | None = None, chunk: int | None = None) -> tuple[jax.Array, ...]:
    """Weights `p` narrowed to the values' dtype times the tile's `value`s, as the mainloop's `dot`, a
    `chunk` of keys at a time (`products_of`); with a dropout `multiplier`, the dropped weights
    `p * multiplier` narrowed at the scale `weights_parts` gives them."""
    if multiplier is not None:
        return products_of(weights_parts(p * multiplier, value.dtype), value, dot, chunk)
    return products_of((narrow(p, value.dtype),), value, dot, chunk)


def products_of(parts: Parts, value: jax.Array, dot: Callable[[jax.Array, jax.Array], jax.Array],
                chunk: int | None) -> tuple[jax.Array, ...]:
    """The weights' `parts` `[bq, bk]` times a tile's `value`s `[bk, Dv]` (`summed_products`): one product,
    or in a compensated walk one a `chunk` of keys (`chunk_of`), each a chain of `chunk` additions, whose
    sum the walk takes with each addition's rounding kept (`kept_in_turn`)."""
    if chunk is None or value.shape[0] <= chunk:
        return (summed_products(parts, value, dot),)
    count = value.shape[0] // chunk
    values = jnp.split(value, count, axis=0)
    def pieces(part):
        # A part's chunks of keys, a scaled one's each at its scale.
        if isinstance(part, product.Scaled):
            return [product.Scaled(piece, part.exponent) for piece in jnp.split(part.value, count, axis=-1)]
        return jnp.split(part, count, axis=-1)

    split = [pieces(part) for part in parts]
    return tuple(summed_products(tuple(chunks[index] for chunks in split), values[index], dot)
                 for index in range(count))


def chunk_of(spec: Forward) -> int | None:
    """The keys a walk in `spec`'s form sums in one chain: `CHUNK` where it compensates its sums, else a
    tile's whole (None)."""
    return CHUNK if spec.compensated else None


def _softmax(spec: Forward, state: Partial, keys: Keys):
    """A tile's softmax: the running maximum, the factor the state's sums take, the weights for the
    product (the guarded walk's at the row's shift, `tile`), the tile's values and the guarded `Room`."""
    m, alpha = peak(spec, state.m, keys.logits, keys.products, axis=keys.coords.key_axis)
    p = weights(spec, keys.logits, m, keys.coords)
    value = keys.values.load()
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
    form). A tile held keys first (`Coords.keys_first`, a decode program's `K Qᵀ`) takes its sums so: the
    mainloop's, `[Dv, bq]`, its queries their columns, on the plain walk. The derived forms'
    (`_derived_step`), but for a dropping call's: the derivation narrows a
    dropped weight unnormalized, past fp16's largest at a rate near 1 (a multiplier of 2^16), where this
    step's scaled parts (`weights_parts`) keep it, a representation the declaration states none of."""
    # A compensated or guarded walk's step takes its dropped roundings and its shifts through the derived
    # step's own hook (`_derived_step`).
    if keys.coords.drop is None:
        return _derived_step(spec, state, keys)
    m, factor, p, value, held = _softmax(spec, state, keys)
    multiplier = drawn(keys.coords, keys.coords.q_pos, keys.coords.k_pos)
    if held is None and keys.accumulate is not None:
        # The weights the walk accumulates dropped where the call drops (`_times`' parts); the denominator
        # the undropped weights'.
        parts = ((narrow(p, value.dtype),) if multiplier is None
                 else weights_parts(p * multiplier, value.dtype))
        # The denominator before the product, so the tile's fp32 weights die at their narrowing and hold no
        # registers through it (FS-4324-PEAK1: on an A100 they held 64 a thread past P·V, its peak 299 at
        # 128 x 64, where FlashAttention 2's sums them before; its bits the same either way).
        total = denominator_after(state.total, factor, p, axis=keys.coords.key_axis)
        # `carry_factor`'s plain factor, each query's broadcast over its sums as the mainloop holds them.
        acc = keys.accumulate(state.acc, lambda acc: keys.coords.queries(factor, acc) * acc, parts, value)
        return Partial(acc, m, total)
    assert not keys.coords.keys_first, "a tile held keys first takes the plain walk's mainloop sums"
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
        return (others,), others, own_tile, anchor
    parts = weights_parts(jnp.where(at_anchor, 0.0, p) * multiplier, dtype)
    # The anchor's multiplier by its query's vector and its key's, no unit axis (a Mosaic program's lowers
    # none beside a tile).
    anchor_draw = drawn(keys.coords, keys.coords.query_places(), anchor.key)
    anchor = anchor._replace(multiplier=anchor_draw)
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
    factor = carried(spec, start.m, m)
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
    value = keys.values.load()
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
        value = keys.values.load()
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


NORMALIZED = -1.0
"""A split row's shift (`Split.shift`) where its sums are normalized over its natural denominator, a guarded
walk's live row's (`split_residuals`), not at `2^-shift`: every shift a walk takes (`room`) is at least 0,
so the sentinel needs no residual beside the shift (`normalized_rows`)."""


def normalized_rows(shift: jax.Array) -> jax.Array:
    """Which rows of a split's `shift` hold normalized sums (`NORMALIZED`)."""
    return shift < 0


def split_residuals(split: Partial, own: jax.Array, others_total: jax.Array,
                    normalized: Callable[[], tuple[jax.Array, jax.Array, jax.Array]]
                    ) -> tuple[jax.Array, ...]:
    """A training walk's split sums for the backward and their shift: as the walk left them (shift 0
    unguarded), and on the guarded walk a live row's normalized ones, which the mainloop's pass
    `normalized()` gives (the other keys' weighted values, the anchor's weight and the other keys' weights
    over the natural denominator, `normalized_terms`), at the shift `NORMALIZED`, which tells `split_delta`
    they are normalized; the plain walk takes no such pass."""
    sums = split.acc, own, others_total, jnp.zeros_like(own)
    if split.guard is None:
        return sums
    live = split.guard.live
    # The pass only where some row is live: a guarded program none of whose rows overflowed takes none.
    terms = jax.lax.cond(jnp.max(live.astype(jnp.int32)) > 0, normalized,
                         lambda: (jnp.zeros_like(split.acc), jnp.zeros_like(own), jnp.zeros_like(own)))
    fresh = (*terms, jnp.full_like(own, NORMALIZED))

    def chosen(new, old):
        return jnp.where(expand(live, old) if old.ndim > own.ndim else live, new, old)
    return tuple(chosen(new, old) for new, old in zip(fresh, (*sums[:3], split.guard.shift), strict=True))


def split_output(others: jax.Array, own: jax.Array, anchor_value: jax.Array) -> jax.Array:
    """The weighted values of every key, `B` and the anchor's own term (`anchor_value`, its value row)."""
    return others + expand(own, others) * anchor_value.astype(jnp.float32)


def merge(spec: Forward, first: Partial, second: Partial) -> Partial:
    """Two states over disjoint keys as one, `first`'s keys before `second`'s: each taken to the larger
    maximum and summed, in that order. Either may hold no key whatever the call (a slice past a row's
    keys), so `empty` is its identity under every spec. The derived forms' (`_derived_merge`), a guarded
    state's join of its sums the template's (`_guarded_merge`) over the factors they evaluate."""
    return _derived_merge(spec, first, second)


def _guarded_merge(first: Partial, second: Partial, m: jax.Array, factors: tuple[jax.Array, jax.Array]
                   ) -> Partial:
    """`merge`'s join of two states one of which is guarded (`Guard`), each taken to the joined maximum `m` by
    its factor of `factors`, its shifted sums at the larger shift, or more where the join overflows, and its
    unshifted ones beside them."""
    ones, others = headroom_of(first), headroom_of(second)
    reach = jnp.maximum(ones.reach, others.reach)
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


def merged(spec: Forward, stacked: Partial, axis: int) -> Partial:
    """States over disjoint keys stacked on `axis` (a split launch's slices, `imported`) as one, at once:
    `merge`'s fold over them, each taken to the largest maximum and summed along `axis`, so a combine of
    many slices traces one sum, not a merge a slice (a decode step's 32 slices traced 44 thousand
    operations an attention, which at Dory 4B's depth compiled for over eight minutes; row 2491). Each
    state takes one factor to the largest maximum where the fold takes one a merge, so the sums round in
    another order, no less accurately. A slice of no key (a maximum of -inf) adds nothing. On the guarded
    path the sums join at the largest shift, and a row whose join overflows shifts as its natural sums
    need, as `merge` shifts them. Behind the switch, the derived forms' (`_derived_merged`)."""
    return _derived_merged(spec, stacked, axis)


def output_of(acc: jax.Array, denominator: jax.Array, *, guarded: bool) -> jax.Array:
    """The weighted values `acc` over their denominator, any leading axes: zero for a query that saw nothing
    where it may (`guarded`: a guarded walk's or one from a state)."""
    if not guarded:
        return acc / expand(denominator, acc)
    safe = expand(jnp.where(denominator > 0, denominator, 1.0), acc)
    return jnp.where(expand(denominator > 0, acc), acc / safe, 0.0)


def finished_rows(spec: Forward, state: Partial, *, initial: bool) -> tuple[jax.Array, jax.Array]:
    """The block's residual rows (`finish`'s): the maximum (0 for a query that saw nothing and started from no
    state, as the reference's; -inf for one that saw nothing elsewhere, the walk's start `LEAST` given back
    as the residual's convention) in the units the backward recomputes its logits in, natural, or raw
    products where the forward's are (`Forward.raw`), the products of the query its call's power of two took
    (`Forward.prescale`), and the denominator, unshifted (`room`)."""
    m = state.m if spec.raw else spec.naturalized(state.m)
    m = jnp.where(no_keys(state.total), 0.0 if spec.guarded or initial else -jnp.inf, m)
    fields = state.guard
    if fields is None:
        return m, state.total
    # The natural denominator: the unshifted one where finite, else the shifted one taken back.
    return m, jnp.where(finite(fields.plain_total), fields.plain_total, power2(state.total, fields.shift))


def finish(spec: Forward, state: Partial, *, initial: bool, output_mod: Callable | None = None,
           output_params: tuple[jax.Array, ...] = (), normalized: bool = False
           ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The block's output `[bq, Dv]` fp32 (its weighted values over its denominator, zero for a query
    that saw nothing and started from no state, then the output hook) and its residual rows
    (`finished_rows`); the output is the shifted sums' ratio, which the shift leaves, the derived forms'
    (`_derived_output`), a guarded state's live rows taken from its unshifted sums as ever. A `normalized`
    state's sums are its output already (a merge of finished states, `finished_merged`): zero where it holds
    no mass."""
    m, denominator = finished_rows(spec, state, initial=initial)
    out = (jnp.where(expand(state.total > 0, state.acc), state.acc, 0.0) if normalized
           else _derived_output(spec, state))
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


def inverse(total: jax.Array) -> jax.Array:
    """Each query's `1 / total`, zero where it saw nothing (total 0)."""
    return jnp.where(total > 0, 1.0 / jnp.where(total > 0, total, 1.0), 0.0)


def hooked(score_mod: Callable, logits: jax.Array, q_pos, k_pos, params) -> jax.Array:
    """`score_mod` on a tile, rewritten by the algebra pass before the tile places it (docs/design/algebra.md
    §2, `dew.algebra.rewritten`; the hook as written where the pass is off), each window's derivation held
    across a launch's tiles (`program._derived`); a custom-derivative call in it lowers as its primal
    function (Dew's JAX build, constraints.txt)."""
    return rewritten(score_mod)(logits, q_pos, k_pos, *params)


def _scaled(x: jax.Array, rows: bool = False) -> tuple[jax.Array, jax.Array]:
    """`x` (fp32) times the power of two that puts its largest magnitude at 2^14, fp32, and that power's
    negated exponent, which takes a product of it back (`_fp16_scaled` before its rounding); with `rows`, a
    power a row (its exponents `[rows]`), which a product of the rows (`a @ b`) takes back a row of its own
    (`restored`): a tile's whole largest is a reduction to a scalar, which a tensor-core mainloop's lowering
    has no form of (Mosaic GPU's: "Scalars must be represented by non-vector types", ET-FP16-h100)."""
    top = jnp.max(jnp.abs(x), axis=-1) if rows else jnp.max(jnp.abs(x))
    exponent = jnp.where(top > 0, jnp.ceil(jnp.log2(jnp.where(top > 0, top, 1.0))), 0.0)
    return power2(x, expand(14.0 - exponent, x) if rows else 14.0 - exponent), exponent - 14.0


def restored(value: jax.Array, exponent: jax.Array) -> jax.Array:
    """A scaled part's product `value` taken back by its power of two (`product.Scaled`): a tile's one, or a
    row's where the part is scaled a row at a time (`_scaled`'s `rows`)."""
    return power2(value, exponent if exponent.ndim == 0 else expand(exponent, value))


def _fp16_scaled(x: jax.Array, rows: bool = False) -> tuple[jax.Array, jax.Array]:
    """`_scaled`'s `x` in fp16 and its exponent: within fp16's range whatever `x`'s, an entry far below the
    largest losing only what is far below the product's own rounding."""
    scaled, exponent = _scaled(x, rows)
    return scaled.astype(jnp.float16), exponent


def cotangent_parts(cotangents: jax.Array, dtype, *, accurate: bool) -> Parts:
    """The parts a tile's score cotangents `[bq, bk]` (fp32, scaled) take to their product with an operand of
    `dtype` (`cotangent_product`), each narrowed to it, a `product.Scaled` where its product takes a power of
    two back, so a mainloop may issue the others' products into one accumulator. An fp32 operand's part is the
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


def formed_parts(cotangents: jax.Array, dtype, *, accurate: bool, rows: bool = False
                 ) -> Iterator[jax.Array | product.Scaled]:
    """`cotangent_parts` a part at a time, each formed only as it is taken: a mainloop that retires each
    part's product before taking the next (`Family.register_products`' `retired`) holds one part's registers
    beside the cotangents, not every part's; fp16's scaled a row at a time with `rows` (`_scaled`)."""
    if dtype == jnp.float32:
        yield cotangents
        return
    if dtype == jnp.bfloat16:
        # The leading part saturates at bf16's largest: an fp32 cotangent past it by more than half its ulp
        # rounds to infinity in bf16, an infinite product, or a NaN against a zero operand. The accurate
        # form's later parts take the difference whole, the fast form's one part is within 2^-8 of it.
        yield from product.bf16_parts(cotangents, 3 if accurate else 1, saturating=1)
        return
    scaled, high_exponent = _scaled(cotangents, rows)
    high = scaled.astype(dtype)
    yield product.Scaled(high, high_exponent)
    low, low_exponent = _fp16_scaled(scaled - high.astype(jnp.float32), rows)
    yield product.Scaled(low, low_exponent + high_exponent)


def cotangent_product(cotangents: jax.Array, operand: jax.Array,
                      dot: Callable[[jax.Array, jax.Array], jax.Array], *, accurate: bool) -> jax.Array:
    """A tile's score cotangents `[bq, bk]` (fp32, scaled) times the call's queries or keys `operand` by the
    mainloop's `dot` (dQ's `dS K`, `dot(a, b) = a @ b`; dK's `dSᵀ Q`, its transposed product
    `dot(a, b) = aᵀ @ b`, which a mainloop issues from where the tile lies), fp32: its parts' products
    (`cotangent_parts`), each taken back by its power of two, summed in order."""
    return summed_products(cotangent_parts(cotangents, operand.dtype, accurate=accurate), operand, dot)


def summed_products(parts: Parts, operand: jax.Array,
                    dot: Callable[[jax.Array, jax.Array], jax.Array]) -> jax.Array:
    """The sum of `parts`' products with `operand` by `dot` (`cotangent_parts`, `weights_parts`), a scaled
    one's taken back by its power of two, in order."""
    def taken(part) -> jax.Array:
        if isinstance(part, product.Scaled):
            return power2(dot(part.value, operand), part.exponent)
        return dot(part, operand)

    first, *rest = parts
    total = taken(first)
    for part in rest:
        total = total + taken(part)
    return total


def weights_parts(weights: jax.Array, dtype, rows: bool = False) -> Parts:
    """Weights `[bq, bk]` (fp32) narrowed for their product with an operand of `dtype` (`summed_products`):
    fp16's at a power-of-two scale that keeps them in its range (`_fp16_scaled`, a row's with `rows`),
    which its product takes back."""
    if dtype == jnp.float16:
        return (product.Scaled(*_fp16_scaled(weights, rows)),)
    return (weights.astype(dtype),)


def narrowed(x: jax.Array, dtype) -> jax.Array:
    """`x` (fp32) narrowed to `dtype` at the scale `weights_parts` gives it, held in fp32: the values a
    product takes, which a dropped weight's (`x * multiplier`) overflow at the 16-bit scale never meets."""
    (part,) = weights_parts(x, dtype)
    if isinstance(part, product.Scaled):
        return power2(part.value.astype(jnp.float32), part.exponent)
    return part.astype(jnp.float32)



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


def value_parts(weights: jax.Array, dtype) -> Parts:
    """A tensor-core mainloop's dV left operand, `pᵀ·dO` against the output cotangent as it is in shared
    memory (FlashAttention's form, `value_operands`' bf16 one): the normalized weights narrowed, bf16's as is,
    fp16's at a power of two of their row's (`weights_parts`), which keeps a weight down to 2^-24 of the
    row's largest where fp16's least subnormal (about 6e-8) would drop a normalized one."""
    return weights_parts(weights, dtype, rows=True)


def softmax_of(logits: jax.Array, values: jax.Array, *, base2: bool, keys_first: bool = False,
               held: bool = False, raw: float | None = None, guarded: bool = True) -> jax.Array:
    """Attention's math on a template's formed tiles, whole (theme A's declaration, `declaration`): each
    query's softmax over its `logits` (scaled, in the exponent's base, -inf at each key it does not see, as
    `modify` forms them; `[rows, keys]`, or `[keys, rows]` where the tile holds its keys first, a decode
    program's `K Qᵀ`), its weights at the values' dtype (the reference's, as `narrow` takes them), summed over
    the `values` `[keys, width]` by Dew's product in fp32 (`float32_einsum`, `[rows, width]`, or `[width,
    rows]` keys first), over their total; its weights anchored at its largest logit, at least the dtype's
    least finite value where it sees no key, so such a query's weights and output are zeros, with no mask an
    element, and the anchor grows with the walk (a walked correction `exp(k_carried - k_now)` at most 1). The
    accurate form's (`declaration`'s `natural`) holds its largest's weight to 1 (`held`, `weights`), or, over
    raw products (`Forward.raw`), scales each difference by their `raw` scale after the subtraction, its
    exponential the one its declaration names. A query's total held from 0 where one may see no key
    (`guarded`: `Forward.guarded`, a state from an initial one), its weighted sums over it plainly elsewhere,
    as the hand finish divides (`output_of`)."""
    exponent = jnp.exp2 if base2 else jnp.exp
    keys_axis = 0 if keys_first else 1
    m = jnp.max(logits, axis=keys_axis)
    safe = jnp.maximum(m, jnp.finfo(logits.dtype).min)
    anchor = safe[None, :] if keys_first else safe[:, None]
    difference = logits - anchor
    if raw is not None:
        difference = difference * np.float32(raw)
    p = exponent(jnp.where(logits < anchor, difference, 0.0) if held else difference)
    total = jnp.sum(p, axis=keys_axis)
    over = jnp.where(total == 0, 1.0, total) if guarded else total
    subscripts, per_query = ("kq,kd->dq", over[None, :]) if keys_first else ("qk,kd->qd", over[:, None])
    summed = float32_einsum(subscripts, p.astype(values.dtype), values, preferred_element_type=jnp.float32)
    return summed / per_query


def declaration(*, base2: bool, keys_first: bool = False, factor: float | None = None, natural: bool = False,
                raw: float | None = None, guarded: bool = True):
    """Attention's forward as theme A's declaration (`dew.algebra.derive.Reduction`): `softmax_of` over the
    template's formed tiles, walked along the keys, its exponent the base the template folds (`folds`: 2
    where it folds `log2(e)` into its logits), its tile keys first where the mainloop holds it so; over the
    tile's raw products times `factor` where the mainloop gives them (`Keys.products`, `logits_of`), so the
    derivation may take their maximum and scale it once (`rules`' positive scale out of a maximum), the
    hand step's 62 fewer multiplies a step at Qwen3's tile, its exponent's argument one fused multiply-add a
    logit (`rules._contracted`). Its derived forms (`derived_forms`) are what this module's walk computes
    by (`plain_step`, `merge`, `finish`), on par with XLA's against float64 (docs/design/algebra.md). The
    accurate form's (`natural`, `Forward.natural`) declares its exponential (`natural_exp`) and its rescale
    (`natural_carried`, its exponent compensated), which the derivation evaluates as they are and rewrites
    none: over scaled logits its largest weight held to 1 (`softmax_of`'s `held`), over raw products
    (`Forward.raw`) each difference scaled by their `raw` scale, which the exponential carries and the
    rescale compensates. Its total is held from 0 where a query may see no key (`guarded`, `softmax_of`).
    """
    tile = ("keys", "rows") if keys_first else ("rows", "keys")

    def walked(logits, values):
        scaled = logits if factor is None else logits * np.float32(factor)
        return softmax_of(scaled, values, base2=base2, keys_first=keys_first, held=natural and raw is None,
                          raw=raw, guarded=guarded)

    if natural:
        stated = {"exponential": natural_exp,
                  "rescale": functools.partial(natural_carried, raw=raw is not None,
                                               scale=1.0 if raw is None else raw)}
    else:
        # A raw fold's weights are its products' difference times `raw`, which its exponential and rescale
        # carry (`fold_exp`).
        stated = {} if raw is None else {"exponential": fold_exp,
                                         "rescale": lambda m, safe: fold_exp(m - safe, raw)}
    return derive.Reduction(walked, {"logits": tile, "values": ("keys", "width")}, walked="keys",
                            exponent=2 if base2 else "e", **stated)


@functools.cache
def derived_forms(base2: bool, keys_first: bool = False, factor: float | None = None, natural: bool = False,
                  raw: float | None = None, guarded: bool = True) -> derive.Derived:
    """The forward's derived forms (`declaration`'s), which `plain_step`, `merge` and `finish` compute by; a
    step over raw products, their `factor` (`declaration`). The fast form's are the fast class's; the
    accurate form's (`natural`) the accurate class's, which admits no reordering past the error-free rules."""
    from dew.algebra import SATURATION
    declared = declaration(base2=base2, keys_first=keys_first, factor=factor, natural=natural, raw=raw,
                           guarded=guarded)
    return derive.derive(declared, accurate=natural, saturation=SATURATION)


def _form_key(spec: Forward) -> tuple:
    """The derived forms' key's part past the fold and the products' factor (`derived_forms`): the accurate
    form's natural and a raw product's scale (`Forward.raw`), a raw fold's the factor its difference takes
    (`Forward.subtracted`), the fast form's none; and whether a query may see no key (`Forward.guarded`, or a
    state from an initial one), whose finish holds its total from 0, as the hand finish's (`finish`)."""
    guarded = spec.guarded or spec.initial
    if spec.natural:
        return (True, spec.scale if spec.raw else None, guarded)
    return (False, float(np.float32(spec.scale * LOG2E)) if spec.raw else None, guarded)


def _slotted(key: tuple, shapes: Mapping) -> tuple[str, ...]:
    """The `Partial` field each of the derived forms' (`derived_forms(*key)`) carried states is
    (`Derived.slots`) at a walk of tiles of `shapes`, by its role: its maximum `m`, its sum `total`, its
    contraction (the weights' product with the values) `acc`."""
    slots = derived_forms(*key).slots
    assert slots is not None, "a reduction's derived forms name their slots"
    fields = tuple({"max": "m", "sum": "total", "contraction": "acc"}[role] for role, _ in slots(shapes))
    if sorted(fields) != ["acc", "m", "total"]:
        raise derive.Refused(f"the derived forms carry {fields}, not a softmax's maximum, total and sums")
    return fields


def _stated(spec: Forward, state: Partial) -> tuple[tuple, dict]:
    """The derived forms a merge or a finish of `state` reads, and the walk's tiles it names them by: its
    base's walk, queries first, over tiles of a key per query of the state's own dtypes beside its rows (fp32
    a kernel's; float64 a truth's run, its forms derived apart): a merge's and a finish's forms read the state
    alone, a maximum, a total and a row's sums a query, the same at any keys a tile and any values' dtype (a
    product's sums, `softmax_of`), so the forms of any walk of the state's are these, whatever walk came
    before (a hand-written step's, a dropping walk's, a supplied state)."""
    rows, width = state.m.shape[-1], state.acc.shape[-1]
    tiles = {"logits": ((rows, rows), state.m.dtype), "values": ((rows, width), state.acc.dtype)}
    return (spec.base2, False, None, *_form_key(spec)), tiles


def _derived_step(spec: Forward, state: Partial, keys: Keys) -> Partial:
    """`plain_step` by the derived forms: the tile's logits (-inf where a query does not see a key, as
    `modify` writes them) and values, its maximum, rescale and weights the forms'; the weighted values' sums
    the mainloop's own where it holds them (`Keys.accumulate`), every product the mainloop's (`Keys.dot`). A
    compensated walk's (the state's `Low`) and a guarded one's (its `Guard`) take their sums as the template's
    step takes them (`stepped`, the one owner of both): the tile's products a chunk of keys each, each
    addition's rounding kept, and on the guarded walk each live row's shift (`room`, from the forms' weights
    and the tile's values) with the unshifted sums beside them."""
    # Over the raw products where the mainloop gives them (`logits_of`), their factor the declaration's own.
    raw = keys.products is not None
    key = (spec.base2, bool(keys.coords.keys_first), spec.factor if raw else None, *_form_key(spec))
    forms = derived_forms(*key)
    tiles = {"logits": keys.products if raw else keys.logits, "values": keys.values}
    fields = _slotted(key, {name: (value.shape, value.dtype) for name, value in tiles.items()})
    held_by, plain = keys.accumulate, state.low is None and state.guard is None
    after: list[Partial] = []

    def own(states, parts):
        taken = dict(zip(fields, parts, strict=True))
        accumulated, added = taken["acc"], taken["total"]
        assert isinstance(accumulated, Accumulated)
        if plain:
            # The denominator from the forms' terms before the product, as `plain_step` takes it, so the
            # tile's fp32 weights die at their narrowing (FS-4324-PEAK1); the weighted values' sums the
            # mainloop's accumulators hold.
            assert isinstance(added, Added) and held_by is not None
            total = states[fields.index("total")] * added.factor + added.terms.sum(axis=keys.coords.key_axis)
            factor = accumulated.factor

            def scale(held):
                # A sum the walk rescales (the weighted values': its factor a row's), else as it is.
                return held if factor is None else keys.coords.queries(factor, held) * held

            acc = held_by(states[fields.index("acc")], scale, (accumulated.weights,), accumulated.values)
            return tuple(acc if field == "acc" else total if field == "total" else None for field in fields)
        assert isinstance(added, Added)
        # The template's step over the forms' weights (the denominator's terms, before their narrowing for
        # the product) and rescale.
        value, p, factor = accumulated.values, added.terms, added.factor
        held = None
        if state.guard is not None:
            largest = keys.largest if keys.largest is not None else functools.partial(magnitudes, value)
            held = room(state.guard, state.total, factor, p, largest())
        weights = p if held is None else held.weights
        times = functools.partial(_times, keys.dot, value, multiplier=None, chunk=chunk_of(spec))
        after.append(stepped(state, state.m, factor, weights, times(weights), held, times))
        return tuple(getattr(after[0], field) if field in ("acc", "total") else None for field in fields)

    derived = forms.step(tuple(getattr(state, field) for field in fields), tiles, product=keys.dot,
                         stepped=None if held_by is None and plain else own)
    found = Partial(**dict(zip(fields, derived, strict=True)))
    return found if not after else found._replace(guard=after[0].guard, low=after[0].low)


def _rows(state: Partial) -> Partial:
    """`state` with its leading axes and its rows one axis of rows (`acc` `[rows, Dv]`, `m` and `total`
    `[rows]`, a guard's alike): each row's softmax is its own, so a merge or a finish of a split decode's
    states (`[..., splits, ..., bq]`, `decode_attention._combined`) is the declaration's over all their
    rows."""
    leading = state.m.ndim
    return jax.tree.map(lambda part: part.reshape(-1, *part.shape[leading:]), state)


def _derived_merge(spec: Forward, first: Partial, second: Partial) -> Partial:
    """`merge` by the derived forms (`_stated`'s), over the states' rows whatever their leading axes
    (`_rows`): a guarded state's sums joined as the template joins them (`_guarded_merge`) over the
    factors the forms evaluate, through the forms' merge hook."""
    # A state's maximum at the forms' floor (`LEAST`): a residual's row of no key gives it back -inf
    # (`finished_rows`), the forms' arithmetic NaN from it, as `resumed` takes an initial state's.
    floored = [state._replace(m=jnp.maximum(state.m, LEAST).astype(state.m.dtype))
               for state in (first, second)]
    flat, other = _rows(floored[0]), _rows(floored[1])
    key, shapes = _stated(spec, flat)
    fields = _slotted(key, shapes)
    joined: list[Partial] = []

    def own(firsts, seconds, parts):
        del firsts, seconds
        factor = dict(zip(fields, parts, strict=True))["total"]
        assert isinstance(factor, Merged)
        m = jnp.maximum(flat.m, other.m)
        joined.append(_guarded_merge(flat, other, m, (factor.first, factor.second)))
        return tuple(getattr(joined[0], field) if field in ("acc", "total") else None for field in fields)

    guarded = first.guard is not None or second.guard is not None
    merged = derived_forms(*key).merge(tuple(getattr(flat, field) for field in fields),
                                       tuple(getattr(other, field) for field in fields), shapes=shapes,
                                       merged=own if guarded else None)
    found = Partial(**dict(zip(fields, merged, strict=True)))
    if joined:
        found = found._replace(guard=joined[0].guard)
    leading = first.m.shape
    return jax.tree.map(lambda part: part.reshape(*leading, *part.shape[1:]), found)


def _derived_merged(spec: Forward, stacked: Partial, axis: int) -> Partial:
    """`merged` by the derived forms: the slices merged a pair at a time, a level's pairs at once
    (`_derived_merge` over their rows whatever their leading axes), so n slices trace about log2(n) merges,
    not n - 1. The forms' merge is associative (tests/test_algebra_laws.py), so any tree of merges is the
    fold's. A level of an odd count carries its last slice to the next."""
    def taken(state: Partial, start: int, stop: int, stride: int = 1) -> Partial:
        return jax.tree.map(lambda part: jax.lax.slice_in_dim(part, start, stop, stride, axis), state)

    while (count := stacked.m.shape[axis]) > 1:
        pairs = 2 * (count // 2)
        level = _derived_merge(spec, taken(stacked, 0, pairs, 2), taken(stacked, 1, pairs, 2))
        if count % 2:
            level = jax.tree.map(lambda merged, last: jnp.concatenate([merged, last], axis), level,
                                 taken(stacked, pairs, count))
        stacked = level
    return jax.tree.map(lambda part: jnp.squeeze(part, axis), stacked)


def _derived_output(spec: Forward, state: Partial) -> jax.Array:
    """`finish`'s output before its hook by the derived forms (`_stated`'s): the weighted values over their
    total, zero for a query that saw nothing, over the state's rows whatever its leading axes (`_rows`)."""
    flat = _rows(state)
    key, shapes = _stated(spec, flat)
    fields = _slotted(key, shapes)
    finished = derived_forms(*key).finish(tuple(getattr(flat, field) for field in fields), shapes=shapes)
    return finished.reshape(state.acc.shape)
