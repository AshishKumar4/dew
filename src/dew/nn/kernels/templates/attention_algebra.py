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
from collections.abc import Callable, Iterator, Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dew.algebra import derive, rewritten
from dew.algebra.forms import Accumulated, Added, Merged, Operand
from dew.nn.kernels.templates import layout, product
from dew.nn.kernels.templates.attention import (
    LN2,
    LOG2E,
    NATURAL,
    Capped,
    Draw,
    Dropout,
    Initial,
    coded,
    within,
)
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
    fp32's largest, so a program's walk is scaled (`headroom`, `SCALED_WALK`), whether its softmax
    resumes from an `initial` state (`raw`), whether natural logits stay `scaled` whatever the scale (`raw`;
    the fold's rewalk, `unfolded`), the call's query count `queries` where a mainloop walks query rows past it
    without spans (a full or causal call padded to whole tiles or whole programs, whose padding rows, a
    tile's last ones or a whole tile of them, `seen` keeps from every key; None where none does),
    whether its walks keep what their running sums' additions drop (`compensated`, `compensates`), the
    power of two the queries took before their products (`prescale`, `attention.prescale`), which `scale`
    is the call's scale over, the keys a launch's walk reaches where past the call's though they are
    whole tiles (`walked_keys`: a holder of a common walk wholly past them, `seen` keeps every query from
    it; None where none does), whether the launch lowers its exponentials to the hardware's
    approximations (`approximate`, `attention_mosaic_program.approximate`; false where a hook's tanh makes it
    lower every transcendental exactly), and the bound a quantized model's weights round to (`rounds`,
    `attention.Hooks.rounds`: each weight its code `coded(p, rounds)` at the row's known maximum in the
    value product, the anchored walk's, `anchored_step`; None where they do not)."""

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
    rounds: float | None = None
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

    def raised(self, x: jax.Array, k: jax.Array) -> jax.Array:
        """`power(x)` times `2^k`, `k` whole numbers from 0 to 63 (a scaled tile's small weights,
        `at_scale`), formed without passing through fp32's subnormals wherever it is normal: the fold's and
        the natural form's exponent taken `k` binades up before their one rounding, the fast natural form's
        as two halves (`e^(x/2) 2^k e^(x/2)`, base's normalized pass's)."""
        if self.base2:
            return jnp.exp2((x * np.float32(self.scale * LOG2E) if self.raw else x) + k)
        if not self.natural and not self.approximate:
            half = jnp.exp(x * np.float32(0.5))
            return power2(half, k, reach=2 * 127) * half
        return natural_exp(x, self.scale if self.raw else 1.0, offset=k)

    def naturalized(self, m: jax.Array) -> jax.Array:
        """A maximum in the logits' units taken to natural ones: from base 2 folded, from raw products
        raw."""
        if self.raw:
            return m * self.scale
        return m * LN2 if self.base2 else m


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
    (`attention_gradient.below`'s); unclamped where `peak` is
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


class Low(NamedTuple):
    """What a compensated walk's running sums' additions dropped (`compensated`, `kept_in_turn`), each
    beside its sum: the weighted values' `acc` and the denominator's `total`."""

    acc: jax.Array
    total: jax.Array


class Scale(NamedTuple):
    """A row's scale in the scaled walk (`SCALED_WALK`): the exponent `exponent` `[..., bq]` its weights and
    both its sums are held at, `2^-exponent`, a whole number at least 0 that never decreases, and `bound`
    `[..., bq]`, a bound on its weighted values' magnitude unscaled at its running maximum, held at
    `2^-BOUND_UNIT` so it never passes fp32's largest: the sum over the keys it has seen of each one's
    weight times its values' largest magnitude (`Keys.largest`), a resumed state's sums' largest beside
    them."""

    exponent: jax.Array
    bound: jax.Array


class Partial(NamedTuple):
    """A query block's softmax state, unhooked: the weighted values `acc` `[..., bq, Dv]` (or, where a
    mainloop's accumulators hold them, the blocks of their columns it holds, `Keys.accumulate`), the running
    maximum `m` `[..., bq]` (in base 2 where the step's logits are) and the denominator `total`
    `[..., bq]`, fp32, both sums at each row's scale where its values can carry them past fp32's largest
    (`Forward.headroom`, `scale`, `SCALED_WALK`; None where the walk holds none); and within a compensated
    walk what its sums' additions dropped (`low`, `compensated`; None elsewhere: a walk's state is `settled`
    when it ends). `rescale`, `merge` and `finish` take any leading axes, so states a launch wrote per slice
    merge in XLA after it (`exported`, `imported`)."""

    acc: Any
    m: jax.Array
    total: jax.Array
    scale: Scale | None = None
    low: Low | None = None


SCALED_WALK = None
"""The scaled walk (`Forward.headroom`), the online softmax's guard against its sums passing fp32's largest:
FlashAttention's walk, each row with its own power-of-two scale where its own sums would pass it. A row
keeps the bound of its `Scale` beside its sums, each tile adding each of its weights times its key's
values' largest magnitude (`weighed_reach`, a key the row does not see weighing 0), and its
exponent the binades that bound passes `2^HEADROOM` by, 0 below (`exponent_of`). A tile's weights
`p = exp(s - m)` enter its product and its denominator at `2^-e` after it, and both sums before it take the
maximum's factor and `2^(e_old - e_new)`, a power of two, exact (`scaled_step`, `at_scale`); the output is
their ratio, which the scale leaves, and the residual denominator is taken back to natural units
(`finished_rows`).

Exactness: a row whose sums cannot pass fp32's largest keeps `e = 0`, its weights, sums and output
FlashAttention's, bit for bit the unscaled walk's, and its unnormalized weights are never below the
reference's normalized ones (`p >= p / l`), so it keeps every weight the reference keeps. A row whose sums
would holds every partial sum within `2^HEADROOM`; where its scale pushes a weight below fp32's least
normal, that weight enters a second product raised by `2^SMALL_SHIFT`, formed raised from its exponent,
and lowered after it (the two-tier product, `at_scale`), so no weight loses bits to the scale, one below
fp32's least normal at the row's maximum included. Its limit: a normal weight's product with a
value so small that the product at `2^-e` is below fp32's least normal keeps what its magnitude there
holds, at most two binades fewer than the reference's own product (`2^e < 4 l`, the bound being at most
`l` times fp32's largest), and only in a row whose sums would otherwise pass fp32's largest
(tests/test_attention_stability.py's families: a share in bf16's subnormal band, small values beside
unseen ones near fp32's largest, a row of small values beside one that overflows, and the subnormal share
of an overflowing row). A masked key weighs 0 and adds nothing to the bound; a row's scale is its own."""

HEADROOM = 127
"""The binade a scaled row's sums stay within (`exponent_of`): one below fp32's largest, so the bound's
own rounding never lets a partial sum pass it."""

BOUND_UNIT = 64
"""The power of two a row's bound is held below (`Scale.bound`): a bound of keys' weights times values
near fp32's largest, at most the keys' count times 2^128, stays finite at `2^-64` for any count below
2^64, and the values below 2^-62 it flushes are none a row's sums could pass fp32's largest with."""

SMALL_SHIFT = 63
"""The binades a scaled row's small weights are raised by for their product (`at_scale`): a weight a
scale of at most 64 pushed below fp32's least normal is raised back to a normal number, its product
lowered after it."""

WEIGHT_SHIFT = 64
"""The most of a row's shift the backward's first pass takes its weights by (`attention_gradient`'s
`off_anchor_weights`): the rest scales the tile's product, so no weight of 2^-62 or more is pushed below
fp32's least normal however large the shift."""

LEAST_NORMAL = float(jnp.finfo(jnp.float32).tiny)
"""fp32's (and bf16's) least normal number: a scaled weight below it takes the two-tier product."""


def exponent_of(bound: jax.Array) -> jax.Array:
    """The exponent a row of `bound` (`Scale.bound`, at `2^-BOUND_UNIT`) is held at: the binades its
    weighted values' bound passes `2^HEADROOM` by (rounded up), 0 below and where it is no positive number."""
    log = jnp.ceil(jnp.log2(jnp.where(bound > 0, bound, 1.0))) + (BOUND_UNIT - HEADROOM)
    return jnp.where((bound > 0) & finite(log) & (log > 0), log, 0.0)


def no_scale(rows: int) -> Scale:
    """`Scale` before any key: no exponent, no bound."""
    nothing = jnp.zeros((rows,), jnp.float32)
    return Scale(nothing, nothing)


ROWS = ("m", "total")
"""A state's row arrays a launch writes after its sums (`exported`), `Partial`'s fields in their order: its
maximum and denominator, and after them its scale's (`Scale`'s fields) where its walk is scaled."""


def exported_rows(scaled: bool) -> int:
    """The row arrays a launch writes of a state (`exported`): `ROWS`, and its scale's where `scaled`."""
    return len(ROWS) + (len(Scale._fields) if scaled else 0)


def exported(state: Partial) -> tuple[jax.Array, ...]:
    """A state's arrays as a launch writes them: its sums, then its rows (`ROWS`) and its scale's where it
    holds one (`Scale`), which a merge takes (`merge`)."""
    return state.acc, *(getattr(state, name) for name in ROWS), *(state.scale or ())


def imported(parts: tuple[jax.Array, ...]) -> Partial:
    """A state from the arrays a launch wrote (`exported`)."""
    acc, *rows = parts
    named = dict(zip(ROWS, rows[:len(ROWS)], strict=True))
    scale = rows[len(ROWS):]
    return Partial(acc, **named, scale=Scale(*scale) if scale else None)


LEAST = float(jnp.finfo(jnp.float32).min)
"""A walk's running maximum before any key: fp32's least finite value, the derived forms' floor of it
(`dew.algebra.forms`' init), which every state a step or a merge meets is at or above, so no factor between
two states is a NaN. A real logit may be it too (a hook's mask by fp32's least, as transformers' are), so a
row that saw no key is one of no denominator (`no_keys`), never one at `LEAST`: its factor onto a key's
maximum underflows to 0 as -inf's did, onto another such state's is 1 over zero sums."""


def no_keys(total: jax.Array) -> jax.Array:
    """Whether each row of a state's denominator `total` saw no key: none of its sums, as any key's weight
    at the row's maximum is 1 (a scaled walk's at its row's scale, never to 0)."""
    return total == 0


def empty(rows: int, width: int, *, scaled: bool = False) -> Partial:
    """The softmax before any key: `merge`'s identity, its maximum `LEAST`; with `scaled`, its scale before
    any key (`SCALED_WALK`)."""
    return Partial(jnp.zeros((rows, width), jnp.float32), jnp.full((rows,), LEAST, jnp.float32),
                   jnp.zeros((rows,), jnp.float32), no_scale(rows) if scaled else None)


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


def walked[C: (Partial, tuple), O](spec: Forward, walk: Callable[[Forward], C],
                                   finished: Callable[[Forward, C], O]) -> O:
    """A program's walk and its outputs under the fold's guard: `walk(form)` its walk in `form`, its carry a
    `Partial` or a tuple whose first element is one, and `finished(form, carry)` its outputs. Folded, a
    program one of whose rows' maximum passes `FOLD_REACH` walks again in the natural form (`unfolded`), its
    outputs that form's (its maximum natural, as the fold's are once finished). Ordinary inputs take the one
    walk, their bits and cost unchanged but for this check. A walk's sums are held at each row's scale in
    the one walk where they could pass fp32's largest (`SCALED_WALK`), no guard."""
    carry = walk(spec)
    if not spec.reaches:
        return finished(spec, carry)
    state = carry if isinstance(carry, Partial) else carry[0]
    reach = reach_of(state.m, unseen=spec.form == "spans")
    natural = unfolded(spec)
    return jax.lax.cond(reach >= FOLD_REACH, lambda: finished(natural, walk(natural)),
                        lambda: finished(spec, carry))


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


type Launch[O, F] = Callable[..., tuple[O, jax.Array, None] | tuple[O, None, F]]
"""A whole launch of a call in a form (`walked_launches`), `launch(form, live, into=None)`: its outputs and
each row's running maximum; or, a launch whose programs state their rows' rewalk flags themselves (`F`, its
mainloop's own), its outputs, no maxima and those flags, which it takes back as `live`; `into` refs holding
the plain launch's outputs, which it writes its rows `live` into in place (`walked_launches`' `in_place`)."""


def finished_form(spec: Forward) -> Forward:
    """The form a walk's finished states merge in (`slice_state`, `finished_merged`, `merged_slices`): its
    own where `finish` leaves its maxima in its units (natural, or a raw fold's products), else its fold's
    natural form (`unfolded`), whose units a folded walk's finished maxima are in (`finished_rows`)."""
    return spec if spec.natural or spec.raw else unfolded(spec)


def slice_state(spec: Forward, out: jax.Array, maximum: jax.Array, total: jax.Array) -> Partial:
    """A finished slice of a row's keys as a state of their merge (`finished_merged`): its output `out`
    normalized (`finish`'s), its maximum (in `spec`'s units, `finished_rows`', raw products where its fold
    keeps them) and its denominator its mass, kept apart: folded into the maximum (`m + log(total)`) it
    rounded away at a large one (row 4646: at 2^30, fp32's spacing 128, slices of denominators 128 and 64
    merged half and half). A slice of no key the walk's start (`LEAST`), its mass 0. A row or a launch's
    block of rows alike."""
    seen = total > 0
    return Partial(out, jnp.where(seen, maximum, LEAST), jnp.where(seen, total, 0.0))


def initial_state(spec: Forward, initial: Partial) -> Partial:
    """A call's initial state (its sums unnormalized, its maximum natural) as a finished slice's
    (`slice_state`): normalized where it holds mass, its maximum in `spec`'s units, saturated where a raw
    fold's passes fp32's largest (`raw_maximum`, as `resumed` takes it: a finite maximum of 1e38 at a scale
    of 1/8 was infinite, and its merge's mass NaN)."""
    held = initial.total > 0
    out = jnp.where(held[..., None], initial.acc / jnp.where(held, initial.total, 1.0)[..., None], 0.0)
    maximum = raw_maximum(initial.m, spec.scale) if spec.raw else initial.m
    return slice_state(spec, out, maximum, initial.total)


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
    # Each weight over the largest, so their sum is at most the states' count whatever the masses.
    largest = jnp.max(weights, axis=axis)
    weights = weights / jnp.expand_dims(jnp.where(largest > 0, largest, 1.0), axis)
    summed = jnp.sum(weights, axis=axis)
    shares = weights * jnp.expand_dims(1.0 / jnp.where(summed > 0, summed, 1.0), axis)
    out = jnp.sum(expand(shares, stacked.acc) * stacked.acc, axis=axis)
    return Partial(out, m, largest * summed)


def rewalks(spec: Forward, maxima: jax.Array) -> tuple[jax.Array, Forward]:
    """The rows a plain launch's fold guard walks again (`walked_launches`), from its rows' running `maxima`,
    and the form they walk in (`rewalked_form`): those whose maximum passes `FOLD_REACH` (`reached`: a padded
    query's -inf, a row that reads nothing, as a slice's of no keys). A launch's programs state them so of
    their own rows."""
    return reached(maxima, unseen=spec.form == "spans") >= FOLD_REACH, rewalked_form(spec)


def rewalked_form(spec: Forward) -> Forward:
    """The form a plain launch's fold guard walks its rows again in (`rewalks`): the natural form."""
    return unfolded(spec)


def walked_launches[O, F](spec: Forward, launch: Launch[O, F], *, in_place: bool = False) -> O:
    """`walked`'s fold guard over whole launches, for a mainloop whose program walks its keys once (a
    producer streams them, so no program walks again): `launch(form, live)` runs the call in `form` and
    returns its outputs and each row's running maximum in the walk's units, `[..., Q]`. Where the fold
    reaches (`Forward.reaches`) and some row's maximum passes `FOLD_REACH`, the call walks again in the
    natural form (`unfolded`), decided over the whole call; ordinary inputs take the one plain launch, their
    cost unchanged but for this check. Every launch's sums are held at each row's scale in its one walk
    where they could pass fp32's largest (`SCALED_WALK`), no guard.

    `in_place` (a masked walk's, whose launch walks only its live rows' keys, every other row's none:
    `launch(form, live, into=refs)`'s program over its rows' spans) takes one more launch every time, no
    branch: a branch's predicate is read on the host before its launch may run, a decode step's round trip
    (row 1978: 0.280 ms wall against 0.015 on the device), where a launch of empty walks costs its prologue.
    It walks in the natural form the rows the fold's reach passes `FOLD_REACH` a row (`reached`), its rewalk
    written into the plain launch's outputs held in refs (the launch writes its rows `live` and leaves every
    other row's bits, and gives no output of its own, which its programs that walk nothing would leave
    unwritten): the fold's plain launch costs its exponential's two instructions an element where the natural
    form's costs about sixty (ET-DECSASS-h100, ET-FOLDED-h100: Qwen3-8B's decode at 32 rows of 32768 keys,
    1.35 ms folded against 2.11 natural, the H100's bandwidth floor 1.28).

    Elsewhere the rare walk is a loop of at most one trip (`once`), which takes the plain launch's outputs
    as its carry in their own buffer: a branch's common side copied them into its result every call
    (ET-MEMCPY-h100: the output and its rows, 45 us of Qwen3-8B's causal forward on an H100)."""
    plain = launch(spec, None)
    outputs = plain[0]
    if not spec.reaches:
        return outputs
    if in_place:
        # A launch whose programs state its rows' rewalk flags themselves (`rewalks`) gives them in its
        # maxima's place: no array operation between the launches (ET-DPROF-h100: the two fusions forming
        # them took 6.3 us of a batch-1 decode step's 30.7).
        live, form = (plain[2], rewalked_form(spec)) if plain[1] is None else rewalks(spec, plain[1])
        held = jax.tree.map(jax.new_ref, outputs)
        launch(form, live, into=held)
        return jax.tree.map(jax.freeze, held)
    maxima = plain[1]
    assert maxima is not None, "a launch states its rows' rewalk flags in a masked walk alone"
    # The call's own rows alone: a padded query's maximum is -inf, a row that reads nothing (`seen`).
    reach = reach_of(maxima if spec.queries is None else maxima[..., :spec.queries],
                     unseen=spec.form == "spans")
    natural = unfolded(spec)
    return once(reach >= FOLD_REACH, lambda folded: launch(natural, None)[0], outputs)


def once[C](taken: jax.Array, body: Callable[[C], C], carried: C) -> C:
    """`body(carried)` where `taken`, else `carried`: a loop of at most one trip, whose carry XLA keeps in its
    operand's buffer, the rare side of a branch (`walked_launches`' rare walks) with no copy on the common
    one, whose identity a branch copies into its result (ET-MEMCPY-h100). Its trips counted from 0 below 1,
    beside `taken`, so a step's FLOP count bounds its body as it bounds a branch's (`instrumentation`'s
    counted trips)."""
    def trip(state):
        return state[0] + 1, body(state[1])

    return jax.lax.while_loop(lambda state: (state[0] < 1) & taken, trip, (jnp.int32(0), carried))[1]


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


def natural_exp(x: jax.Array, scale: float = 1.0, offset: jax.Array | None = None) -> jax.Array:
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
    # Outside these bounds every fp32 exponential is zero or infinite; infinities have no residual. A raise's
    # `offset` (`Forward.raised`) joins the exponent before its bounds, so a weight past them (a key no row
    # sees, -inf) stays 0: exact, a small weight's exponent at most -62 and its offset a whole number at most
    # 63, so the sum is nearer zero on the same ulp's grid. It joins past the clip, never the product: a GPU's
    # compile contracts a product's add into one fused multiply-add, whose sum holds the product's rounding,
    # which the residual `error` then counts again (the recurrence's prefix scan, M-MC-8/9).
    shift = 0.0 if offset is None else offset
    bounded = jnp.clip(hi, -150.0 - shift, 128.0 - shift)
    if offset is not None:
        bounded = bounded + shift
    exponent = jnp.floor(bounded)
    exponent += (bounded - exponent >= 0.5).astype(x.dtype)
    live = (jnp.abs(x) < jnp.inf) & (hi >= -150.0 - shift) & (hi <= 128.0 - shift)
    lo = jnp.where(live, lo, 0.0)
    # The exponent within [-150, 129]: two factors take it (`power2`'s `reach`).
    return power2(jnp.exp2((bounded - exponent) + lo), exponent, reach=NATURAL_EXPONENT_REACH)


def headroom(dtype) -> bool:
    """Whether a call's weighted values can pass fp32's largest, so its walk is scaled (`SCALED_WALK`):
    values of `dtype` two of which reach `2^126` (bf16's and fp32's). A walk adds to its sums at most its
    keys' count times the values' largest, which for fp16's values over any key count a call holds is far
    below the 2^103 that rounds fp32's largest past it; so an fp16 call's walk holds no scale, from a state
    or none."""
    return float(jnp.finfo(dtype).max) * 2 >= 2.0 ** 126


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


def carry_factor(x: jax.Array, alpha: jax.Array, drop: jax.Array | None = None) -> jax.Array:
    """A state's sum before a tile (`[bq, Dv]` or `[bq]`) taken by its factor `alpha` (`[bq]`, the tile's
    maximum's) and, in a scaled walk, by its scale's change `2^drop` (`Scaled.drop`, at most 0) on the
    product, a power of two: no factor below fp32's least normal is formed, so a scale that grows by many
    binades at once keeps a sum the factor alone would flush."""
    taken = rescale(x, alpha) if x.ndim > alpha.ndim else alpha * x
    if drop is None:
        return taken
    return power2(taken, drop if taken.ndim == drop.ndim else expand(drop, taken))


def denominator_after(total: jax.Array, factor: jax.Array, weighed: jax.Array, held: "Scaled | None" = None
                      ) -> jax.Array:
    """A row's denominator after a tile: the state's taken by `factor` (and a scaled tile's change of scale,
    `carry_factor`) with the tile's weights' sum `weighed` `[bq]` added, at the rows' scale on a scaled tile
    (`held`)."""
    if held is None:
        return carry_factor(total, factor) + weighed
    return carry_factor(total, factor, held.drop) + power2(weighed, -held.exponent, reach=2 * 127)


def stepped(state: Partial, m: jax.Array, factor: jax.Array, weighed: jax.Array,
            products: tuple[jax.Array, ...], scale: Scale | None = None, held: "Scaled | None" = None
            ) -> Partial:
    """The state after a tile: its sums taken by `factor` and on a scaled tile by its rows' change of scale
    (`carry_factor`, `held`), the tile's weights' sum `weighed` `[bq]` (at the rows' scale on a scaled tile)
    and the `products` of its weights with its values (a chunk of keys each in a compensated walk,
    `products_of`) added, each addition's rounding kept where the state carries them (`kept_in_turn`), and
    the rows' scale after the tile `scale` (`rescaled`)."""
    low = state.low
    drop = None if held is None else held.drop
    total, acc = carry_factor(state.total, factor, drop), carry_factor(state.acc, factor, drop)
    weights = (weighed if held is None else power2(weighed, -held.exponent, reach=2 * 127),)
    kept = None
    if low is None:
        total, acc = in_turn(total, weights), in_turn(acc, products)
    else:
        total, total_low = kept_in_turn(total, carry_factor(low.total, factor, drop), weights)
        acc, acc_low = kept_in_turn(acc, carry_factor(low.acc, factor, drop), products)
        kept = Low(acc_low, total_low)
    return Partial(acc, m, total, scale, kept)


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
    nothing dropped yet beside each (`Low`); else as it is."""
    if not spec.compensated:
        return state
    return state._replace(low=Low(jnp.zeros_like(state.acc), jnp.zeros_like(state.total)))


def settled(state: Partial) -> Partial:
    """A walk's end: each running sum with what its additions dropped added back (`compensated`), rounded
    once, and none carried; a sum that passed fp32's largest stays infinite, its residual no number."""
    low = state.low
    if low is None:
        return state

    def joined(total: jax.Array, dropped: jax.Array) -> jax.Array:
        return total + jnp.where(finite(dropped), dropped, 0.0)

    return Partial(joined(state.acc, low.acc), state.m, joined(state.total, low.total), state.scale)


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


def resumed(spec: Forward, m: jax.Array, total: jax.Array, acc: jax.Array) -> Partial:
    """The softmax resumed from a state `(m, total, acc)` (`Hooks.initial`: sinks, a resumed partial),
    its maximum taken to base 2 where the step's logits are, and where the walk is scaled both its sums at
    the scale its weighted values' largest magnitude takes, which its bound starts from (`SCALED_WALK`). A
    row of no key (a maximum of -inf: a padded row, a slice carrying no sink) resumes at the walk's start
    (`LEAST`), exact for every other."""
    m, total, acc = (array.astype(jnp.float32) for array in (m, total, acc))
    if spec.base2:
        m = raw_maximum(m, spec.scale) if spec.raw else base2_maximum(m)
    m = jnp.maximum(m, LEAST)
    if not spec.headroom:
        return Partial(acc, m, total)
    bound = scale_of(Partial(acc, m, total)).bound
    exponent = exponent_of(bound)
    return Partial(power2(acc, expand(-exponent, acc), reach=2 * 127), m,
                   power2(total, -exponent, reach=2 * 127), Scale(exponent, bound))


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
    return spec.power(_difference(spec, s, m, coords))


def _difference(spec: Forward, s: jax.Array, m: jax.Array, coords: Coords | None) -> jax.Array:
    """`weights`' exponent of each logit `s` against its row's maximum `m`."""
    peak = expand(m, s) if coords is None else coords.queries(m, s)
    if spec.natural and not spec.raw:
        # The accurate form's scaled logits: the largest's weight held to 1 against the contraction.
        return jnp.where(s < peak, s - peak, 0.0)
    return s - peak


def raised_weights(spec: Forward, s: jax.Array, m: Callable[[], jax.Array], coords: Coords | None = None
                   ) -> Callable[[jax.Array], jax.Array]:
    """A tile's weights under the running maximum `m()` (`weights`) times `2^k` for each row's `k` `[bq]`,
    formed without fp32's subnormals (`Forward.raised`): a scaled tile's small weights (`at_scale`), the
    maximum read only where a tile is scaled."""
    def raised(k: jax.Array) -> jax.Array:
        shift = expand(k, s) if coords is None else coords.queries(k, s)
        return spec.raised(_difference(spec, s, m(), coords), shift)

    return raised


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


def started(spec: Forward, initial: tuple[jax.Array, jax.Array, jax.Array] | None, rows: int,
            width: int) -> Partial:
    """A program's softmax before its first tile: resumed from `initial` (`resumed`), else `empty`, with
    its rows' scale where the walk is scaled (`SCALED_WALK`)."""
    if initial is not None:
        return resumed(spec, *initial)
    return empty(rows, width, scaled=spec.headroom)


Parts = tuple[jax.Array | product.Scaled, ...]
"""A product's parts as a mainloop issues them (`cotangent_parts`, `weights_parts`): each a narrowed
operand with the power of two its product takes back (None: none)."""


class Keys(NamedTuple):
    """A key tile as a mainloop gives it the algebra: its coordinates and (modified) logits `[bq, bk]`, its
    values `[bk, Dv]` as an operand of their type (`Operand`), which a step loads after the softmax (where
    jax's mha loads them; a derived step's evaluation where its form first reads them), the mainloop's
    product `dot(a, b)`, `a @ b` in fp32, where the mainloop holds the running sums of the weighted values
    itself (an MMA's accumulator), `accumulate(acc, scale, parts, value)`, which returns the running sums
    `acc` (the state's, in the mainloop's own form) taken by `scale(acc)` (None: no factor) with the weights'
    `parts`' products with the values added (`summed_products`), the products the logits are a factor
    of, where they are (`logits_of`), whose extreme the tile's maximum takes (`peak`), and on the scaled walk
    the values' largest magnitude a key, `largest()`, where the mainloop holds the values where it cannot
    read them (a Mosaic slot's shared memory; else `magnitudes` of the loaded tile, `weighed_reach`)."""

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



def _times(dot: Callable[[jax.Array, jax.Array], jax.Array], value: jax.Array, p: jax.Array, *,
           multiplier: jax.Array | None, chunk: int | None, bound: float, held: "Scaled | None" = None
           ) -> tuple[jax.Array, ...]:
    """Weights `p` times the tile's `value`s as the mainloop's `dot`, a `chunk` of keys at a time
    (`products_of`), as `tile_parts` gives them."""
    return products_of(tile_parts(p, held, value.dtype, multiplier, bound), value, dot, chunk)


def tile_parts(p: jax.Array, held: "Scaled | None", dtype, multiplier: jax.Array | None = None,
               bound: float = 1.0, *, axis: int = -1) -> Parts:
    """A tile's weights `p` (keys along `axis`) as the parts of their product with values of `dtype`:
    narrowed; with a dropout `multiplier` (at most `bound`), the dropped weights `p * multiplier` as
    `dropped_parts` narrows them; on a scaled tile (`held`) at their rows' scale in two tiers (`at_scale`,
    `scaled_parts`). A 16-bit dtype's whose dropped weights take a scale of their own (`drops_in_place`) is
    fp16's, whose walk holds no scale (`headroom`)."""
    if held is None:
        return (narrow(p, dtype),) if multiplier is None else dropped_parts(p * multiplier, dtype, bound)
    weights, small = at_scale(p, held, axis=axis)
    if multiplier is not None:
        weights, small = weights * multiplier, small * multiplier
    return scaled_parts(weights, small, dtype)


def dropped_parts(weights: jax.Array, dtype, bound: float) -> Parts:
    """A forward's dropped weights `p * multiplier` (fp32, unnormalized, `p` at most 1 and the draw's
    multiplier `bound`) narrowed for their product with values of `dtype`: as they are where `bound` is
    within the dtype's range (`drops_in_place`), each as exact as the reference's (`attention.reference`
    narrows the normalized `p / total * multiplier`, no larger), so a mainloop's sums take them in place;
    else (fp16 past a rate of 1 - 2^-16) at the power-of-two scale `weights_parts` gives them, which their
    product takes back."""
    if drops_in_place(dtype, bound):
        return (narrow(weights, dtype),)
    return weights_parts(weights, dtype)


def drops_in_place(dtype, bound: float) -> bool:
    """Whether a forward's dropped weights, at most the draw's multiplier `bound`, narrow to `dtype` as they
    are (`dropped_parts`): `bound` within its largest finite value."""
    return bound <= float(jnp.finfo(dtype).max)


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
    product and the tile's values."""
    m, alpha = peak(spec, state.m, keys.logits, keys.products, axis=keys.coords.key_axis)
    p = weights(spec, keys.logits, m, keys.coords)
    return m, alpha, p, keys.values.load()


def magnitudes(value: jax.Array) -> jax.Array:
    """Each key's values' largest magnitude, `[bk]` of a tile's `[bk, Dv]`, fp32 (`weighed_reach`)."""
    return jnp.max(jnp.abs(value.astype(jnp.float32)), axis=-1)


def weighed_reach(keys: Keys, value: jax.Array, p: jax.Array, *, axis: int = -1) -> jax.Array:
    """Each row's weighted values' bound over a tile, `[bq]` (`rescaled`), at `2^-BOUND_UNIT`: its weights
    `p` (keys along `axis`) times each key's largest value magnitude, summed, a reduction of its own (the
    denominator's sum keeps its one use, so the unscaled walk's arithmetic is compiled as ever); a key the
    row does not see weighs 0 and adds nothing. Each key's magnitude from the mainloop (`Keys.largest`, a
    vector of the keys in a logits tile's keys' layout), else the loaded tile's (`magnitudes`)."""
    largest = keys.largest() if keys.largest is not None else magnitudes(value)
    # At the bound's unit before the product: two weights of 1 over values near fp32's largest pass it.
    keyed = (across(largest, p) if axis % p.ndim == p.ndim - 1 else expand(largest, p)) * np.float32(
        2.0 ** -BOUND_UNIT)
    return (p * keyed).sum(axis=axis)


def rescaled(state: Partial, factor: jax.Array, weighed: Callable[[], jax.Array], drop: Draw | None = None
             ) -> Scale | None:
    """The rows' scale after a tile (`SCALED_WALK`), from the state's before it, the tile's maximum's
    `factor` `[bq]` and its rows' weighted values' bound `weighed()` (`weighed_reach`, at `2^-BOUND_UNIT`):
    the bound taken by the factor with the tile's added, times the largest multiplier a dropping call's draw
    takes a weight's product by (`drop`'s `bound`), and the exponent the binades it passes `2^HEADROOM` by,
    never less than the state's; None where the walk holds no scale."""
    before = state.scale
    if before is None:
        return None
    term = weighed()
    bound = factor * before.bound + (term if drop is None else term * np.float32(drop.bound))
    return Scale(jnp.maximum(before.exponent, exponent_of(bound)), bound)


def scales(scale: Scale | None) -> jax.Array | bool:
    """Whether any row of a tile's scale (`rescaled`) is scaled: a block's one predicate, so a tile of none
    takes the unscaled walk's arithmetic, its bits and its cost as ever (`scaled_step`)."""
    return False if scale is None else jnp.max(scale.exponent) > 0


class Scaled(NamedTuple):
    """A scaled tile's arithmetic (`scaled_step`): the change of each row's exponent `drop` `[bq]` (at most
    0) its state's sums take after their factor (`carry_factor`), its exponent after the tile, `exponent`, at
    which the tile's weights and their sums enter (`at_scale`), and `raised(k)`, the tile's weights times
    `2^k` formed without fp32's subnormals (`raised_weights`), its small weights' tier."""

    drop: jax.Array
    exponent: jax.Array
    raised: Callable[[jax.Array], jax.Array]


def scaled_of(state: Partial, scale: Scale, raised: Callable[[jax.Array], jax.Array]) -> Scaled:
    """The arithmetic of a tile that takes the rows from the state's scale to `scale`, its weights raised by
    `raised` (`Scaled`)."""
    assert state.scale is not None, "a scaled tile's state holds its rows' scale"
    return Scaled(state.scale.exponent - scale.exponent, scale.exponent, raised)


def at_scale(p: jax.Array, held: Scaled, *, axis: int = -1) -> tuple[jax.Array, jax.Array]:
    """A tile's weights `p` (keys along `axis`) at their rows' scale `2^-exponent` (`Scaled`): those it
    leaves at fp32's least normal or more, and the rest raised by `2^SMALL_SHIFT` (the two-tier product,
    `scaled_parts`), each zero where the other holds the weight. The small ones are formed raised from their
    exponents (`Scaled.raised`), so a weight already below fp32's least normal at the row's maximum, which
    the walk's `p` flushes, keeps its bits too. A row of no scale keeps every weight in the first, as the
    unscaled walk does."""
    def row(x):
        # A row's quantity broadcast over its tile as it lies: across its keys, along the tile's last axis
        # or, held keys first, its first.
        return x if axis % p.ndim == 0 else expand(x, p)

    shifted = power2(p, row(-held.exponent), reach=2 * 127)
    tiny = (shifted < LEAST_NORMAL) & row(held.exponent > 0)
    return jnp.where(tiny, 0.0, shifted), jnp.where(tiny, held.raised(SMALL_SHIFT - held.exponent), 0.0)


def scaled_parts(weights: jax.Array, small: jax.Array, dtype) -> Parts:
    """A scaled tile's weights (`at_scale`) as the parts of their product with values of `dtype`
    (`summed_products`): the weights narrowed, and the small ones' parts (`small_parts`)."""
    return narrow(weights, dtype), *small_parts(small, dtype)


def small_parts(small: jax.Array, dtype) -> tuple[product.Scaled, ...]:
    """A scaled tile's small weights (`at_scale`), raised, with their product's power of two,
    `2^-SMALL_SHIFT`, a static one (`taken_back`): fp32's as they are; a 16-bit dtype's as three of its
    parts, which sum to them exactly (`product.formed`), each's product with the values exact in fp32, so
    no weight the scale took below fp32's least normal loses its bits there, nor any to a 16-bit narrowing
    (base's normalized pass took their product in fp32)."""
    if jnp.dtype(dtype) == jnp.dtype(jnp.float32):
        return (product.Scaled(small, float(-SMALL_SHIFT)),)
    return tuple(product.Scaled(part, float(-SMALL_SHIFT)) for part in product.formed(small, 3))


def taken_back(value: jax.Array, exponent: jax.Array | float) -> jax.Array:
    """A scaled part's product, or a sum of its weights, `value` taken back by its power of two
    (`product.Scaled`): a static one's (`small_parts`) as one exact multiply by its constant, no power built
    in the kernel (Mosaic GPU's lowering holds no scalar a constant's arithmetic makes); a traced one's by
    `power2`."""
    if isinstance(exponent, int | float):
        return value * np.float32(2.0 ** exponent)
    return power2(value, exponent)


def scaled_step[C](state: Partial, scale: Scale | None, step: Callable[[Scaled | None], C], *,
                   raised: Callable[[jax.Array], jax.Array], branched: bool = True) -> C:
    """A tile's step on the scaled walk (`SCALED_WALK`): `step(held)` with the tile's scaled arithmetic
    (`scaled_of`, its weights raised by `raised`) where some row of the block is scaled after it (`scales`),
    else `step(None)`, the unscaled walk's tile, its arithmetic and its cost as ever; that alone where the
    walk holds no scale. A mainloop that holds its sums in an MMA's accumulator (`Keys.accumulate`), whose
    asynchronous products no branch may hold, takes the scaled arithmetic every tile (not `branched`): a row
    of no scale's bits are the unscaled walk's in it too (its weights and sums taken by `2^0`, its small part
    none), at a second product a tile (`may_scale` keeps a compute-bound launch from it)."""
    if scale is None:
        return step(None)
    if not branched:
        return step(scaled_of(state, scale, raised))
    return jax.lax.cond(scales(scale), lambda: step(scaled_of(state, scale, raised)), lambda: step(None))


def may_scale(value: jax.Array, *, keys_axis: int, initial: jax.Array | None = None, amplified: float = 1.0
              ) -> jax.Array:
    """Whether some row of a call over `value` (its keys along `keys_axis`, its columns the last axis), from
    the weighted values `initial` of its initial state where it has one, can be scaled (`SCALED_WALK`): a
    row's bound is at most the sum over its keys of each one's largest value magnitude (a weight is at most
    1), times a dropping call's largest multiplier (`amplified`), beside its initial sums' largest
    (`resumed`), so where the largest such sum of any key head is below `2^(HEADROOM - 1)` (a binade for the
    bound's own rounding) no row's is past `2^HEADROOM`, and the unscaled walk is the scaled one bit for bit:
    a call there launches its mainloop's unscaled kernels, one pass over its values before them."""
    # Each key's largest. A call of no keys or rows reads no value: its bound is 0.
    largest = jnp.max(jnp.abs(value.astype(jnp.float32)), axis=-1, initial=0)
    return bounded_past(largest, keys_axis=keys_axis, initial=initial, amplified=amplified)


def bounded_past(largest: jax.Array, *, keys_axis: int, initial: jax.Array | None = None,
                 amplified: float = 1.0) -> jax.Array:
    """`may_scale` of a call whose keys' largest value magnitudes `largest` (each key's over its columns,
    its keys along `keys_axis`) are given: a cache's, which states each written key's as it writes it
    (`attention_decode.attend`'s `bounds`), so a call over it reads no value to choose its kernel."""
    unit = np.float32(2.0 ** -BOUND_UNIT)
    # Each key's largest at the bound's unit, summed over the keys: a key head's bound.
    bound = jnp.max(jnp.sum(largest.astype(jnp.float32) * unit, axis=keys_axis), initial=0)
    bound = bound * np.float32(amplified)
    if initial is not None:
        bound = bound + jnp.max(jnp.abs(initial.astype(jnp.float32)), initial=0) * unit
    return bound >= np.float32(2.0 ** (HEADROOM - 1 - BOUND_UNIT))


def plain_step(spec: Forward, state: Partial, keys: Keys) -> Partial:
    """The state after a tile `keys`: its softmax and the product of its weights with its values, the
    mainloop's own running sums taking it where it holds them (`Keys.accumulate`, the state's `acc` in its
    form), at its rows' scale on the scaled walk (`scaled_step`). A tile held keys first
    (`Coords.keys_first`, a decode program's `K Qᵀ`) takes its sums so: the mainloop's, `[Dv, bq]`, its
    queries their columns. The derived forms' (`_derived_step`), but for a dropping call's: the derivation
    narrows a dropped weight unnormalized, past fp16's largest at a rate near 1 (a multiplier of 2^16), where
    this step's scaled parts (`weights_parts`) keep it, a representation the declaration states none of."""
    # A compensated or scaled walk's step takes its dropped roundings and its rows' scale through the derived
    # step's own hook (`_derived_step`).
    if keys.coords.drop is None:
        return _derived_step(spec, state, keys)
    m, factor, p, value = _softmax(spec, state, keys)
    axis = keys.coords.key_axis
    scale = rescaled(state, factor, lambda: weighed_reach(keys, value, p, axis=axis), keys.coords.drop)
    raised = raised_weights(spec, keys.logits, lambda: m, keys.coords)
    multiplier = drawn(keys.coords, keys.coords.q_pos, keys.coords.k_pos)
    bound = keys.coords.drop.bound
    accumulate = keys.accumulate
    if accumulate is not None:
        def accumulated(held: Scaled | None) -> Partial:
            # The weights the walk accumulates dropped where the call drops (`_times`' parts); the
            # denominator the undropped weights'.
            parts = tile_parts(p, held, value.dtype, multiplier, bound, axis=axis)
            # The denominator before the product, so the tile's fp32 weights die at their narrowing and hold
            # no registers through it (FS-4324-PEAK1: on an A100 they held 64 a thread past P·V, its peak 299
            # at 128 x 64, where FlashAttention 2's sums them before; its bits the same either way).
            total = denominator_after(state.total, factor, p.sum(axis=axis), held)
            taken = functools.partial(_queries_taken, keys.coords, factor, held)
            acc = accumulate(state.acc, taken, parts, value)
            return Partial(acc, m, total, scale)

        return scaled_step(state, scale, accumulated, raised=raised, branched=False)
    assert not keys.coords.keys_first, "a tile held keys first takes the mainloop's sums"

    def multiplied(held: Scaled | None) -> Partial:
        products = _times(keys.dot, value, p, multiplier=multiplier, chunk=chunk_of(spec), bound=bound,
                          held=held)
        # The weights' sum after their products, as the template's compensated step has taken it.
        return stepped(state, m, factor, p.sum(axis=axis), products, scale, held)

    return scaled_step(state, scale, multiplied, raised=raised)


def _queries_taken(coords: Coords, factor: jax.Array | None, held: Scaled | None, acc):
    """`carry_factor` on a mainloop's own sums `acc` (`Keys.accumulate`'s `scale`), each query's factor
    (none where None) and its change of scale broadcast over its sums as the mainloop holds them."""
    taken = acc if factor is None else coords.queries(factor, acc) * acc
    return taken if held is None else power2(taken, coords.queries(held.drop, acc))


Gathered = Callable[[jax.Array], jax.Array]
"""A training walk's value rows gathered by key: each row's `[bq, Dv]` at its anchor key `[bq]`."""


def _split_tile(keys: Keys, p: jax.Array, dtype, anchor: Argmax, multiplier: jax.Array | None,
                held: Scaled | None = None) -> tuple[Parts, Callable[[], jax.Array], jax.Array, Argmax]:
    """A training step's tile split at each row's `anchor` key (`anchor_split`): the other keys' weights as
    the parts of their product with the values (`dtype`), the other keys' narrowed weights' sum `[bq]` (a
    function, which the step takes after the products) and the anchor's own weight, on a scaled tile
    (`held`) at their rows' scale in two tiers (`at_scale`); under dropout (the tile's draw's `multiplier`)
    the other keys' dropped weights, the anchor's own weight undropped (it is `own`), and the anchor takes its
    draw's multiplier, which the forward's output and the backward read."""
    at_anchor = keys.coords.k_pos == expand(anchor.key, keys.logits)
    small = None
    if held is not None:
        p, small = at_scale(p, held)
        # The anchor's weight is its row's largest key's, normal at any scale a call reaches (a row is scaled
        # where its weights' sum times fp32's largest passes 2^127, so that weight is at least 2^-(1 + log2
        # n) of n keys, and the scale at most 2^-(2 + log2 n)): a small weight is another key's.
        small = jnp.where(at_anchor, 0.0, small)
    others, own_tile = anchor_split(narrow(p, dtype), at_anchor)
    lowered = () if small is None else small_parts(small, dtype)

    def rest() -> jax.Array:
        # The other keys' narrowed weights' sum, a scaled tile's small ones' taken back beside it (their
        # parts sum to them exactly): taken after the tile's products, which hold no row of it.
        summed = others.astype(jnp.float32).sum(axis=-1)
        return summed if small is None else summed + taken_back(small.sum(axis=-1), float(-SMALL_SHIFT))
    drop = keys.coords.drop
    if multiplier is None or drop is None:
        return (others, *lowered), rest, own_tile, anchor
    dropped = jnp.where(at_anchor, 0.0, p) * multiplier
    parts = (dropped_parts(dropped, dtype, drop.bound) if small is None
             else scaled_parts(dropped, small * multiplier, dtype))
    # The anchor's multiplier by its query's vector and its key's, no unit axis (a Mosaic program's lowers
    # none beside a tile).
    anchor_draw = drawn(keys.coords, keys.coords.query_places(), anchor.key)
    return parts, rest, own_tile, anchor._replace(multiplier=anchor_draw)


def training_step(spec: Forward, carry: Carry, keys: Keys, rows: Gathered) -> tuple[Carry, Gathered]:
    """A training walk's carry after a tile `keys`: its state, whose sums are the weighted values kept apart
    from the anchor key's (`B`, with the initial state's), the anchor key, its own weight, and the other
    keys' weights' sum (keys alone). Where the anchor moved to this tile, the old anchor's term (its weight
    times its value row, `rows(key)` each row's at its key `[bq]`) joins `B` and the sum before the rescale
    (`split_term`, `split_rows`), and the new anchor's weight is the tile's; the tile's other keys add their
    product and their narrowed weights. So neither is summed with a key's weight cancelled out of it. On the
    scaled walk every split sum is at its rows' scale (`scaled_step`). Where the mainloop holds the sums
    (`Keys.accumulate`), they take the tile there. Returns the carry and `rows`."""
    state, arg, own, others_total = carry
    # The derived forms' softmax (`_derived_softmax`), but for a dropping call's, as `plain_step` takes them.
    softmax = _derived_softmax if keys.coords.drop is None else _softmax
    m, factor, p, value = softmax(spec, state, keys)
    weighed = p.sum(axis=-1)
    scale = rescaled(state, factor, lambda: weighed_reach(keys, value, p), keys.coords.drop)
    anchor = argmax(arg, keys.logits, keys.coords.k_pos)
    multiplier = drawn(keys.coords, keys.coords.q_pos, keys.coords.k_pos)
    moved = anchor.peak > arg.peak
    # The old anchor's term as every split sum held it before the tile.
    term = split_term(own, moved, rows(arg.key), arg.multiplier)

    def split(held: Scaled | None) -> Carry:
        parts, rest, own_tile, anchored = _split_tile(keys, p, value.dtype, anchor, multiplier, held)
        drop = None if held is None else held.drop
        accumulate = keys.accumulate
        if accumulate is None:
            # `B` takes the old anchor's term before the factor (`split_term`), its rounding kept where the
            # walk compensates (`kept_in_turn`), then the tile's other keys' products as a step takes them.
            low = state.low
            if low is None:
                joined = state._replace(acc=in_turn(state.acc, (term,)))
            else:
                acc, acc_low = kept_in_turn(state.acc, low.acc, (term,))
                joined = state._replace(acc=acc, low=low._replace(acc=acc_low))
            after = stepped(joined, m, factor, weighed,
                            products_of(parts, value, keys.dot, chunk_of(spec)), scale, held)
        else:
            # The tile's fp32 weights done with before its products, so they hold no registers through them.
            total = denominator_after(state.total, factor, weighed, held)
            acc = accumulate(state.acc, lambda sums: carry_factor(sums + term, factor, drop), parts, value)
            after = Partial(acc, m, total, scale)
        own_after, others_after = split_rows(own, others_total, factor, moved, rest(), own_tile, drop)
        return after, anchored, own_after, others_after

    raised = raised_weights(spec, keys.logits, lambda: m, keys.coords)
    return scaled_step(state, scale, split, raised=raised, branched=keys.accumulate is None), rows


def anchored(spec: Forward, start: Partial, arg: Argmax) -> Carry:
    """A training walk's carry before its first tile at each row's anchor known (`arg`: the first key at the
    keys' largest logit, from a walk of `argmax` over the logits alone, `anchored_step`): the start taken to
    the walk's final maximum, the larger of its own and the keys', which no tile moves, its bound with it
    on the scaled walk, and no key's weight yet."""
    m = jnp.maximum(start.m, arg.peak)
    factor = carried(spec, start.m, m)
    nothing = jnp.zeros_like(start.total)
    scale = None if start.scale is None else start.scale._replace(bound=factor * start.scale.bound)
    return (Partial(expand(factor, start.acc) * start.acc, m, factor * start.total, scale), arg, nothing,
            nothing)


def anchored_step(spec: Forward, carry: Carry, keys: Keys) -> Carry:
    """`training_step` at the anchor and the final maximum known before the walk (`anchored`'s carry): the
    tile's weights at that maximum, so the sums take no maximum's factor and no old anchor's term ever joins
    them; the anchor's own weight from its own tile and the other keys' product and narrowed weights, each
    summed apart, as the gathering walk keeps them. No reduction finds a maximum or an anchor; the sums the
    mainloop holds take the product in place, unscaled (`Keys.accumulate`'s `scale` None), but on a scaled
    tile, whose sums take their rows' change of scale (`scaled_step`)."""
    state, arg, own, others_total = carry
    if keys.coords.drop is None:
        # The derived forms' weights at the known maximum, which no tile moves.
        _, _, p, value = _derived_softmax(spec, state, keys)
    else:
        p, value = weights(spec, keys.logits, state.m), keys.values.load()
    # A quantized model's weights in the value product its codes at the row's maximum, each at most
    # `rounds` and exact in the values' dtype; the denominator the weights themselves (`Forward.rounds`).
    taken = p if spec.rounds is None else coded(p, spec.rounds)
    scale = rescaled(state, jnp.ones_like(state.total), lambda: weighed_reach(keys, value, taken),
                     keys.coords.drop)
    multiplier = drawn(keys.coords, keys.coords.q_pos, keys.coords.k_pos)

    def split(held: Scaled | None) -> Carry:
        parts, rest, own_tile, anchor = _split_tile(keys, taken, value.dtype, arg, multiplier, held)
        if held is None:
            acc = (state.acc + summed_products(parts, value, keys.dot) if keys.accumulate is None
                   else keys.accumulate(state.acc, None, parts, value))
            return (Partial(acc, state.m, state.total + p.sum(axis=-1), scale), anchor, own + own_tile,
                    others_total + rest())
        drop = held.drop
        if keys.accumulate is None:
            acc = power2(state.acc, expand(drop, state.acc)) + summed_products(parts, value, keys.dot)
        else:
            rescale = functools.partial(_queries_taken, keys.coords, None, held)
            acc = keys.accumulate(state.acc, rescale, parts, value)
        total = power2(state.total, drop) + power2(p.sum(axis=-1), -held.exponent, reach=2 * 127)
        return (Partial(acc, state.m, total, scale), anchor, power2(own, drop) + own_tile,
                power2(others_total, drop) + rest())

    raised = raised_weights(spec, keys.logits, lambda: state.m, keys.coords)
    return scaled_step(state, scale, split, raised=raised, branched=keys.accumulate is None)


def tiles_walked[C](lo, hi, step: Callable[[jax.Array, C], C], carry: C, most: int | None = None) -> C:
    """`step(index, carry)` over a program's tiles `lo` to `hi`, from `carry`: `jax.lax.fori_loop`'s, a
    kernel's loop of traced bounds (a Pallas mainloop's, which Triton pipelines); given `most`, the tiles the
    call holds (static), a loop counted from 0 while under `most` and under `hi` past `lo`, so its program
    states at most `most` trips beside the spans' bounds (`dew.telemetry.instrumentation._counted_trips`): a
    program XLA runs whole (the plain-JAX mainloop's), whose loop of traced bounds alone stated no trip count,
    and a CPU training step at 3,072 tokens on that mainloop no FLOPs (`Trainer.step_flops` None)."""
    if most is None:
        return jax.lax.fori_loop(lo, hi, step, carry)

    def stepped(state: tuple[jax.Array, C]) -> tuple[jax.Array, C]:
        trip, carried = state
        return trip + 1, step(lo + trip, carried)

    return jax.lax.while_loop(lambda state: (state[0] < most) & (lo + state[0] < hi), stepped,
                              (jnp.int32(0), carry))[1]


def plain_walk(spec: Forward, lo, hi, keys_at: Callable[[jax.Array], Keys], start: Partial, *,
               most: int | None = None) -> Partial:
    """A program's walk of its key tiles `lo` to `hi` (`keys_at(index)`, of at most `most`, `tiles_walked`)
    from `start`, its sums compensated where its form takes them so (`compensated`, `settled`)."""
    return settled(tiles_walked(lo, hi, lambda index, state: plain_step(spec, state, keys_at(index)),
                                compensated(spec, start), most))


def training_walk(spec: Forward, lo, hi, keys_at: Callable[[jax.Array], Keys],
                  rows: Callable[[jax.Array], jax.Array], start: Partial, dropping: bool = False, *,
                  most: int | None = None) -> Carry:
    """A training forward's walk (`plain_walk`), its carry also holding the anchor key and the split
    sums, from none (`training_step`)."""
    nothing = jnp.zeros(start.total.shape, jnp.float32)
    state, arg, own, others_total = tiles_walked(
        lo, hi, lambda index, carry: training_step(spec, carry, keys_at(index), rows)[0],
        (compensated(spec, start), no_argmax(start.total.shape[0], dropping), nothing, nothing), most)
    return settled(state), arg, own, others_total


def anchored_walk(spec: Forward, lo, hi, keys_at: Callable[[jax.Array], Keys], start: Partial, *,
                  most: int | None = None) -> Carry:
    """A training forward's walk (`training_walk`'s carry) at each row's maximum and anchor found before it:
    a walk of `argmax` over the logits alone, then `anchored_step`s from `anchored`'s carry, as the Mosaic
    training launch walks a peak walk's anchors; a quantized call's (`Forward.rounds`), whose weights round
    at that maximum."""

    def peaked(index, arg):
        keys = keys_at(index)
        return argmax(arg, keys.logits, keys.coords.k_pos)

    arg = tiles_walked(lo, hi, peaked, no_argmax(start.total.shape[0]), most)
    state, arg, own, others_total = tiles_walked(
        lo, hi, lambda index, carry: anchored_step(spec, carry, keys_at(index)), anchored(spec, start, arg),
        most)
    return settled(state), arg, own, others_total


def anchor_multipliers(drop: Dropout, anchors: jax.Array) -> jax.Array:
    """Each row's anchor's dropout multiplier (`Draw`) at its query and its anchor key `anchors`
    `[B, H, Qp]`, every batch row and query head its draw's: the factor its own weight takes in the output
    (`joined`) and its value in D (a dropping backward's `gradient.split_delta`)."""
    batch, heads, queries = anchors.shape
    draw = Draw((drop.seed[0], drop.seed[1]), drop.rows.reshape(batch, 1, 1), drop.heads.reshape(1, heads, 1),
                drop.rate)
    return draw(jnp.arange(queries, dtype=jnp.int32), anchors)


def joined(carry: Carry, rows: Callable[[jax.Array], jax.Array]) -> Partial:
    """A training walk's state with its split sums joined, the anchor's term added back (`split_output`),
    the weighted values the output normalizes, at its rows' scale on the scaled walk."""
    split, arg, own, _ = carry
    anchored_own = own if arg.multiplier is None else own * arg.multiplier
    return split._replace(acc=split_output(split.acc, anchored_own, rows(arg.key)))


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


def split_term(own: jax.Array, moved: jax.Array, old_value: jax.Array,
               old_multiplier: jax.Array | None = None) -> jax.Array:
    """The old anchor's term a training step's split sums join where the anchor moved (`moved` `[bq]`): its
    weight `own` before the tile (under dropout times its draw's multiplier `old_multiplier`) times its value
    row `old_value` `[bq, Dv]`; zero elsewhere."""
    old = old_value.astype(jnp.float32)
    anchored = own if old_multiplier is None else own * old_multiplier
    return jnp.where(expand(moved, old), expand(anchored, old) * old, 0.0)


def split_rows(own: jax.Array, others_total: jax.Array, alpha: jax.Array, moved: jax.Array, rest: jax.Array,
               own_tile: jax.Array, drop: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """A training step's row sums after a tile (`training_step`): the anchor's own weight (the tile's where
    the anchor moved to it), and the other keys' weights' sum, the tile's `rest` `[bq]` added and the old
    anchor's weight joining it where it moved, each taken by `alpha` and a scaled tile's `drop`."""
    total = carry_factor(others_total + jnp.where(moved, own, 0.0), alpha, drop) + rest
    return jnp.where(moved, own_tile.astype(jnp.float32), carry_factor(own, alpha, drop)), total


NORMALIZED = -1.0
"""A split row's shift (`Split.shift`) where its sums are normalized over its natural denominator, an output
residual's (`attention.output_split`), not at `2^-shift`: every shift a walk takes (`Scale.exponent`) is at
least 0, so the sentinel needs no residual beside the shift (`normalized_rows`)."""


def normalized_rows(shift: jax.Array) -> jax.Array:
    """Which rows of a split's `shift` hold normalized sums (`NORMALIZED`)."""
    return shift < 0


def split_residuals(split: Partial, own: jax.Array, others_total: jax.Array) -> tuple[jax.Array, ...]:
    """A training walk's split sums for the backward as the walk left them, and their rows' shift
    (`Split.shift`): their scale's exponent on the scaled walk (`SCALED_WALK`), else 0."""
    shift = jnp.zeros_like(own) if split.scale is None else split.scale.exponent
    return split.acc, own, others_total, shift


def split_output(others: jax.Array, own: jax.Array, anchor_value: jax.Array) -> jax.Array:
    """The weighted values of every key, `B` and the anchor's own term (`anchor_value`, its value row)."""
    return others + expand(own, others) * anchor_value.astype(jnp.float32)


def merge(spec: Forward, first: Partial, second: Partial) -> Partial:
    """Two states over disjoint keys as one, `first`'s keys before `second`'s: each taken to the larger
    maximum and summed, in that order. Either may hold no key whatever the call (a slice past a row's
    keys), so `empty` is its identity under every spec. The derived forms' (`_derived_merge`), a scaled
    state's join of its sums the template's (`_scaled_merge`) over the factors they evaluate."""
    return _derived_merge(spec, first, second)


def _scaled_merge(first: Partial, second: Partial, m: jax.Array, factors: tuple[jax.Array, jax.Array]
                  ) -> Partial:
    """`merge`'s join of two states one of which holds a scale (`Scale`; an unscaled one's from its sums,
    `scale_of`), each taken to the joined maximum `m` by its factor of `factors` and to the joined scale, the
    larger of theirs and the one their joined bound takes, a power of two on the product (`carry_factor`)."""
    ones, others = scale_of(first), scale_of(second)
    a, b = factors
    bound = a * ones.bound + b * others.bound
    exponent = jnp.maximum(jnp.maximum(ones.exponent, others.exponent), exponent_of(bound))
    drops = ones.exponent - exponent, others.exponent - exponent
    acc = carry_factor(first.acc, a, drops[0]) + carry_factor(second.acc, b, drops[1])
    total = carry_factor(first.total, a, drops[0]) + carry_factor(second.total, b, drops[1])
    return Partial(acc, m, total, Scale(exponent, bound))


def scale_of(state: Partial) -> Scale:
    """`state`'s scale: its own, else an unscaled state's, no exponent and its sums' largest magnitude its
    bound (`resumed`'s)."""
    if state.scale is not None:
        return state.scale
    bound = jnp.max(jnp.abs(state.acc), axis=-1) * np.float32(2.0 ** -BOUND_UNIT)
    return Scale(jnp.zeros_like(state.total), bound)


def merged(spec: Forward, stacked: Partial, axis: int) -> Partial:
    """States over disjoint keys stacked on `axis` (a split launch's slices, `imported`) as one, at once:
    `merge`'s fold over them, each taken to the largest maximum and summed along `axis`, so a combine of
    many slices traces one sum, not a merge a slice (a decode step's 32 slices traced 44 thousand
    operations an attention, which at Dory 4B's depth compiled for over eight minutes; row 2491). Each
    state takes one factor to the largest maximum where the fold takes one a merge, so the sums round in
    another order, no less accurately. A slice of no key (a maximum of -inf) adds nothing. Scaled slices
    join at their joined scale, as `merge` joins them. Behind the switch, the derived forms'
    (`_derived_merged`)."""
    return _derived_merged(spec, stacked, axis)


def output_of(acc: jax.Array, denominator: jax.Array, *, guarded: bool) -> jax.Array:
    """The weighted values `acc` over their denominator, any leading axes: zero for a query that saw nothing
    where it may (`guarded`: a walk's whose row may see none, or one from a state)."""
    if not guarded:
        return acc / expand(denominator, acc)
    safe = expand(jnp.where(denominator > 0, denominator, 1.0), acc)
    return jnp.where(expand(denominator > 0, acc), acc / safe, 0.0)


def finished_rows(spec: Forward, state: Partial, *, initial: bool) -> tuple[jax.Array, jax.Array]:
    """The block's residual rows (`finish`'s): the maximum (0 for a query that saw nothing and started from no
    state, as the reference's; -inf for one that saw nothing elsewhere, the walk's start `LEAST` given back
    as the residual's convention) in the units the backward recomputes its logits in, natural, or raw
    products where the forward's are (`Forward.raw`), the products of the query its call's power of two took
    (`Forward.prescale`), and the denominator, natural (a scaled walk's taken back from its rows' scale,
    `SCALED_WALK`)."""
    m = state.m if spec.raw else spec.naturalized(state.m)
    m = jnp.where(no_keys(state.total), 0.0 if spec.guarded or initial else -jnp.inf, m)
    if state.scale is None:
        return m, state.total
    return m, power2(state.total, state.scale.exponent, reach=2 * 127)


def finish(spec: Forward, state: Partial, *, initial: bool, output_mod: Callable | None = None,
           output_params: tuple[jax.Array, ...] = (), normalized: bool = False
           ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The block's output `[bq, Dv]` fp32 (its weighted values over its denominator, zero for a query
    that saw nothing and started from no state, then the output hook) and its residual rows
    (`finished_rows`); the output is the derived forms' ratio (`_derived_output`), which a scaled walk's
    rows' scale leaves (both its sums are at it). A `normalized` state's sums are its output already (a
    merge of finished states, `finished_merged`): zero where it holds no mass."""
    m, denominator = finished_rows(spec, state, initial=initial)
    out = (jnp.where(expand(state.total > 0, state.acc), state.acc, 0.0) if normalized
           else _derived_output(spec, state))
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
    (`restored`). A tile's whole largest is its rows' largest reduced again, two reductions of one axis
    each, which Mosaic GPU lowers in every family's accumulator layout where one reduction of both axes
    does not ("Scalars must be represented by non-vector types", ET-EDGE16D's tile arm; FS-REDUCESCALAR5)."""
    largest = jnp.max(jnp.abs(x), axis=-1)
    top = largest if rows else jnp.max(largest)
    exponent = jnp.where(top > 0, jnp.ceil(jnp.log2(jnp.where(top > 0, top, 1.0))), 0.0)
    # A GPU's approximate log2 flushes a subnormal largest to zero, its exponent -inf: scaled by 2^381
    # (`power2`'s reach) the part passed fp32's largest and its remainder was inf - inf, NaN (row 4464, an
    # fp16 row whose keys weigh subnormal beside a state near fp32's largest). Such a part takes no scale.
    exponent = jnp.where(finite(exponent), exponent, 0.0)
    return power2(x, expand(14.0 - exponent, x) if rows else 14.0 - exponent), exponent - 14.0


def scaled_by(x: jax.Array, k: jax.Array) -> jax.Array:
    """`x` times `2^k` for a scalar integral `k` within twice fp32's exponent range, as two scalar
    powers of two, each within it (`exp2` of an integer, a GPU's exactly): no tile of exponent bits, which
    held a 128-wide fp16 fused backward's dK and dV beside two int tiles a step, 286 registers live at its
    peak and 2240 bytes spilled (et-c7512), and no bitcast of a scalar, which Mosaic GPU takes none of."""
    half = jnp.floor(k * 0.5)
    return x * jnp.exp2(half) * jnp.exp2(k - half)


def running_parts(x: jax.Array, exponent: jax.Array, dtype=jnp.float16
                  ) -> tuple[jax.Array, jax.Array, tuple[jax.Array, jax.Array]]:
    """An fp32 tile `x` as two `dtype` parts at a walk's running scale (a queries-first backward's dK's and
    dV's left operands, accumulated over its query tiles): the tile's largest exponent (two one-axis
    reductions, `_scaled`) joins the walk's running `exponent`, the largest so far (-inf before any), and
    both parts are `x` times `2^(14 - exponent')` (`scaled_by`), the high part rounded and the low part its
    remainder rounded at the same scale, so both products issue into one accumulator held at that scale.
    Returns the new exponent, the power of two that takes the accumulator from the old scale to the new
    (0 before any tile, else at most 0) and the parts; a product of them is taken back by
    `running_restored`. A part's entries far below the running largest lose their low bits to `dtype`'s
    subnormals, as an fp32 sum of them loses them to its rounding."""
    new = jnp.maximum(exponent, tile_exponent(x))
    return new, running_shift(exponent, new), parts_at(x, new, dtype)


def tile_exponent(x: jax.Array) -> jax.Array:
    """An fp32 tile's largest magnitude's exponent, rounded up (two one-axis reductions, `_scaled`), -inf
    where it holds none: `running_parts`' tile's."""
    largest = jnp.max(jnp.max(jnp.abs(x), axis=-1))
    tile = jnp.ceil(jnp.log2(jnp.where(largest > 0, largest, 1.0)))
    # A GPU's approximate log2 flushes a subnormal largest to zero (`_scaled`, row 4464): such a tile, as an
    # empty one, moves no exponent.
    return jnp.where((largest > 0) & finite(tile), tile, -jnp.inf)


def running_shift(old: jax.Array, new: jax.Array) -> jax.Array:
    """The power of two that takes an accumulator from the running scale of exponent `old` to `new`'s
    (`running_parts`): 0 before any tile, else at most 0."""
    return jnp.where(old == -jnp.inf, 0.0, old - jnp.where(new == -jnp.inf, 0.0, new))


def parts_at(x: jax.Array, exponent: jax.Array, dtype=jnp.float16) -> tuple[jax.Array, jax.Array]:
    """An fp32 tile `x` as two `dtype` parts at the running scale of `exponent`, its tile's own or larger
    (`running_parts`): `x` times `2^(14 - exponent)` rounded, then its remainder rounded."""
    scaled = scaled_by(x, jnp.where(exponent == -jnp.inf, 0.0, 14.0 - exponent))
    high = scaled.astype(dtype)
    return high, (scaled - high.astype(jnp.float32)).astype(dtype)


def running_restored(value: jax.Array, exponent: jax.Array) -> jax.Array:
    """A product of `running_parts`' parts, or an accumulator of them, at the walk's scale taken back:
    `value` times `2^(exponent - 14)` (`scaled_by`); itself where no tile moved the exponent, every part then
    zero. No select of a scalar's condition over the tile: Mosaic GPU broadcasts none implicitly
    (et-fp16lower)."""
    return scaled_by(value, jnp.where(exponent == -jnp.inf, 0.0, exponent - 14.0))


def restored(value: jax.Array, exponent: jax.Array | float) -> jax.Array:
    """A scaled part's product `value` taken back by its power of two (`product.Scaled`): a tile's one, or a
    row's where the part is scaled a row at a time (`_scaled`'s `rows`), or a static one (`taken_back`)."""
    if isinstance(exponent, int | float):
        return taken_back(value, exponent)
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
            return taken_back(dot(part.value, operand), part.exponent)
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
        return taken_back(part.value.astype(jnp.float32), part.exponent)
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


def _step_forms(spec: Forward, keys: Keys):
    """The derived forms of a walk's step over the tile `keys` (`derived_forms`), its tiles by name and its
    states' slots: over the raw products where the mainloop gives them (`logits_of`), their factor the
    declaration's own."""
    raw = keys.products is not None
    key = (spec.base2, bool(keys.coords.keys_first), spec.factor if raw else None, *_form_key(spec))
    tiles = {"logits": keys.products if raw else keys.logits, "values": keys.values}
    return derived_forms(*key), tiles, _slotted(key, {name: (value.shape, value.dtype)
                                                      for name, value in tiles.items()})


def _derived_softmax(spec: Forward, state: Partial, keys: Keys):
    """`_softmax` by the derived forms: the running maximum the forms' step reaches, the factor the state's
    sums take, the tile's weights (the denominator's terms, before their narrowing for the product) and
    values."""
    forms, tiles, fields = _step_forms(spec, keys)
    taken: list = []

    def own(states, parts):
        # The sums kept as they were: the walk takes the tile's weights and values itself, so the forms
        # evaluate no product here.
        taken.append(dict(zip(fields, parts, strict=True)))
        return tuple(held if field in ("acc", "total") else None
                     for field, held in zip(fields, states, strict=True))

    derived = forms.step(tuple(getattr(state, field) for field in fields), tiles, product=keys.dot,
                         stepped=own)
    m = dict(zip(fields, derived, strict=True))["m"]
    accumulated, added = taken[0]["acc"], taken[0]["total"]
    assert isinstance(accumulated, Accumulated) and isinstance(added, Added)
    return m, added.factor, added.terms, accumulated.values


def _derived_step(spec: Forward, state: Partial, keys: Keys) -> Partial:
    """`plain_step` by the derived forms: the tile's logits (-inf where a query does not see a key, as
    `modify` writes them) and values, its maximum, rescale and weights the forms'; the weighted values' sums
    the mainloop's own where it holds them (`Keys.accumulate`), every product the mainloop's (`Keys.dot`). A
    compensated walk's (the state's `Low`) takes its sums as the template's step takes them (`stepped`): the
    tile's products a chunk of keys each, each addition's rounding kept; a scaled walk's (`SCALED_WALK`) its
    rows' scale from the forms' weights and the tile's values (`rescaled`), each tile at it where some row is
    scaled (`scaled_step`)."""
    forms, tiles, fields = _step_forms(spec, keys)
    held_by, plain = keys.accumulate, state.low is None
    after: list[Partial] = []

    def own(states, parts):
        taken = dict(zip(fields, parts, strict=True))
        accumulated, added = taken["acc"], taken["total"]
        assert isinstance(accumulated, Accumulated) and isinstance(added, Added)
        value, axis = accumulated.values, keys.coords.key_axis
        scale = rescaled(state, added.factor, lambda: weighed_reach(keys, value, added.terms, axis=axis))

        def step(held: Scaled | None) -> Partial:
            if plain and held_by is not None:
                # The denominator from the forms' terms before the product, as `plain_step` takes it, so the
                # tile's fp32 weights die at their narrowing (FS-4324-PEAK1); the weighted values' sums the
                # mainloop's accumulators hold.
                weighed = added.terms.sum(axis=axis)
                total = (states[fields.index("total")] * added.factor + weighed if held is None
                         else denominator_after(states[fields.index("total")], added.factor, weighed, held))
                parts = ((accumulated.weights,) if held is None
                         else tile_parts(added.terms, held, value.dtype, axis=axis))
                # The weighted values' sums taken by their factor, a row's (none where the forms state none),
                # and a scaled tile's change of scale.
                taken = functools.partial(_queries_taken, keys.coords, accumulated.factor, held)
                acc = held_by(states[fields.index("acc")], taken, parts, value)
                return Partial(acc, state.m, total, scale)
            # The template's step over the forms' weights (the denominator's terms, before their narrowing
            # for the product) and rescale.
            products = _times(keys.dot, value, added.terms, multiplier=None, chunk=chunk_of(spec), bound=1.0,
                              held=held)
            return stepped(state, state.m, added.factor, added.terms.sum(axis=axis), products, scale, held)

        # The tile's maximum as the forms' step reaches it, read only where a tile is scaled.
        raised = raised_weights(spec, keys.logits, lambda: peak(spec, state.m, keys.logits, keys.products,
                                                                axis=axis)[0], keys.coords)
        after.append(scaled_step(state, scale, step, raised=raised, branched=held_by is None))
        return tuple(getattr(after[0], field) if field in ("acc", "total") else None for field in fields)

    derived = forms.step(tuple(getattr(state, field) for field in fields), tiles, product=keys.dot,
                         stepped=None if held_by is None and plain and state.scale is None else own)
    found = Partial(**dict(zip(fields, derived, strict=True)))
    return found if not after else found._replace(scale=after[0].scale, low=after[0].low)


def _rows(state: Partial) -> Partial:
    """`state` with its leading axes and its rows one axis of rows (`acc` `[rows, Dv]`, `m` and `total`
    `[rows]`, a scale's alike): each row's softmax is its own, so a merge or a finish of a split decode's
    states (`[..., splits, ..., bq]`, `decode_attention._combined`) is the declaration's over all their
    rows."""
    leading = state.m.ndim
    return jax.tree.map(lambda part: part.reshape(-1, *part.shape[leading:]), state)


def _derived_merge(spec: Forward, first: Partial, second: Partial) -> Partial:
    """`merge` by the derived forms (`_stated`'s), over the states' rows whatever their leading axes
    (`_rows`): a scaled state's sums joined as the template joins them (`_scaled_merge`) over the factors
    the forms evaluate, through the forms' merge hook."""
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
        joined.append(_scaled_merge(flat, other, m, (factor.first, factor.second)))
        return tuple(getattr(joined[0], field) if field in ("acc", "total") else None for field in fields)

    scaled = first.scale is not None or second.scale is not None
    merged = derived_forms(*key).merge(tuple(getattr(flat, field) for field in fields),
                                       tuple(getattr(other, field) for field in fields), shapes=shapes,
                                       merged=own if scaled else None)
    found = Partial(**dict(zip(fields, merged, strict=True)))
    if joined:
        found = found._replace(scale=joined[0].scale)
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
