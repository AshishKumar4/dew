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
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.kernels.templates.attention import LOG2E, NATURAL, Kind, natural_exp


@dataclasses.dataclass(frozen=True)
class Forward:
    """What the forward's functions specialize on, all static: the spans' `form` ('full', 'causal' or
    'spans', `KeySpans.form`), the call's key count `keys`, whether it packs documents (`ids`), the
    softmax `scale`, the score hook `score_mod(s, q_pos, k_pos, *params)` on natural logits, whether
    the call keeps its logits `natural` (`base2`), the accurate form, whether its weighted values can pass
    fp32's largest, so a program's walk is `guarded` (`headroom`, the accurate form's alone), and whether
    its softmax resumes from an `initial` state (`raw`)."""

    form: str
    keys: int
    ids: bool
    scale: float
    score_mod: Callable | None = None
    natural: bool = NATURAL
    headroom: bool = True
    initial: bool = False

    @property
    def raw(self) -> bool:
        """Whether the accurate form (`natural`) keeps its logits as the raw products, each exponent a
        difference of them scaled after it is taken, `(s - m) * scale * log2(e)`: no multiply precedes the
        subtraction for a compiler to contract into it, so the largest logit's weight is 1 exactly and
        every difference rounds once. Where a score hook reads the scaled logits, or an initial state's
        natural maximum would have to be divided into raw units, or the scale is not positive (whose raw
        maximum is not the scaled one's), the logits are scaled, the largest's weight held to 1 by a select
        (`weights`). The backward recomputes in the same units (`below`)."""
        return self.natural and self.score_mod is None and not self.initial and self.scale > 0

    @property
    def base2(self) -> bool:
        """Whether the logits fold log2(e) into the scale for exp2, the fast form and the default
        (FlashAttention 2's and jax's mha's arithmetic): unless the call keeps them `natural`, where no
        hook reads them and the scale times log2(e) is below 1, so the fold takes no logit the reference
        keeps finite past fp32's largest, and every folded logit is at least an ulp of fp32's largest
        below it (`base2_maximum`). Otherwise they stay natural, each exponential taken
        of their difference (`natural_exp`), as the reference's. The fold rounds each logit at log2(e)
        times its size, so logits the reference keeps apart can tie: keys of 2 * 1610613120 and 2 * 1610612992
        under a query of 1 at scale 0.5 weigh 1 and 0 in the reference and 0.5 each folded, and a
        cotangent of 2**125 then overflows its backward (docs/design/kernel-templates.md, §3, the
        exponent's forms). On drawn inputs both forms pass check 1."""
        # The factor as the kernel multiplies by it, rounded to fp32.
        return not self.natural and self.score_mod is None and float(np.float32(self.scale * LOG2E)) < 1.0

    @property
    def guarded(self) -> bool:
        """Whether a query can see no key, so its maxima and denominators need the empty row's guard: a
        call of no keys, one whose spans narrow, or one a score hook may mask."""
        return self.keys == 0 or self.form == "spans" or self.score_mod is not None

    def power(self, x: jax.Array) -> jax.Array:
        """The exponential the logits are in: exp2 folded, the raw form's scaled difference, `natural_exp`
        natural."""
        if self.base2:
            return jnp.exp2(x)
        return jnp.exp2(x * (self.scale * LOG2E)) if self.raw else natural_exp(x)

    def carry(self, x: jax.Array) -> jax.Array:
        """The factor a softmax state takes from one maximum to another `x` below it: exp2 folded, and
        natural, `jnp.exp`, the reference's own, whose state (a resumed partial, a sink) may be any size, so
        `natural_exp`'s extra rounding of `x * log2(e)`, about `|x|` ulps, would reach it."""
        if self.base2:
            return jnp.exp2(x)
        return jnp.exp(x * self.scale) if self.raw else jnp.exp(x)

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
    `[..., bq]`, fp32; on the guarded path (`guarded`) both sums at each row's `2^-shift` (`room`), a
    bound on the magnitude of the values the row has weighed, `reach` `[..., bq]`, and whether the row
    may shift, `live` (one whose plain walk overflowed), all None on the plain path. `rescale`, `merge`
    and `finish` take any leading axes, so states a launch wrote per slice merge in XLA after it."""

    acc: jax.Array
    m: jax.Array
    total: jax.Array
    shift: jax.Array | None = None
    reach: jax.Array | None = None
    live: jax.Array | None = None


def empty(rows: int, width: int) -> Partial:
    """The softmax before any key: `merge`'s identity."""
    return Partial(jnp.zeros((rows, width), jnp.float32), jnp.full((rows,), -jnp.inf, jnp.float32),
                   jnp.zeros((rows,), jnp.float32))


def headroomed(state: Partial, live: jax.Array | None = None) -> Partial:
    """`state` on the guarded path where the plain path carried no shift: none yet, its rows `live` (None,
    none), and a bound on its values' magnitudes, its weighted values' largest over its denominator (the
    weighted mean bounds the merged sums a slice's state joins, `merge`)."""
    if state.shift is not None:
        # A slice's state read back from its launch's outputs carries no `live`: none of its rows shifts
        # but by its own merge (`merge`).
        if state.live is not None:
            return state
        return state._replace(live=jnp.zeros(state.total.shape, jnp.bool_))
    rows = state.total
    largest = jnp.max(jnp.abs(state.acc), axis=-1)
    reach = jnp.where(rows > 0, largest / jnp.where(rows > 0, rows, 1.0), 0.0)
    live = jnp.zeros(rows.shape, jnp.bool_) if live is None else live
    return state._replace(shift=jnp.zeros_like(rows), reach=reach, live=live)


def overflowed(acc: jax.Array) -> jax.Array:
    """Which rows of a program's weighted values `acc` `[bq, Dv]` passed fp32's largest, `[bq]`: the one
    way a row of finite inputs turns non-finite (an infinite sum stays infinite under every later
    factor)."""
    # A maximum, the reduction every mainloop lowers (Pallas Triton takes no `jnp.all`).
    return jnp.max(jnp.where(jnp.isfinite(acc), 0.0, 1.0), axis=-1) > 0


def guarded(plain: Callable, shifted: Callable, weighted: Callable = lambda state: state.acc) -> object:
    """A program's softmax walk under headroom: `plain()`, its walk as ever, whose carry's first element is
    its `Partial`; where that walk's weighted values, `weighted(carry)` (a training walk's split sums
    joined, `split_output`), overflowed on some row (`overflowed`), `shifted(live)` instead, the walk
    again with those rows alone shifting (`room`), every other row's bits the plain walk's. Ordinary
    inputs take the plain walk alone, their bits and their cost unchanged but for this check; both
    return the guarded carry (`headroomed`)."""
    carry = plain()
    state, *rest = (carry,) if isinstance(carry, Partial) else carry
    live = overflowed(weighted(carry))

    def kept(live):
        held = headroomed(state)
        return (held, *rest) if rest else held

    return jax.lax.cond(jnp.max(live.astype(jnp.int32)) > 0, shifted, kept, live)


def power2(x: jax.Array, k: jax.Array) -> jax.Array:
    """`x * 2^k` for an integral `k` of any size within fp32's exponents, in two factors each within
    2^±64, so no factor itself overflows or flushes (`2^128` is infinite in fp32, `2^-128` subnormal)."""
    half = jnp.trunc(k / 2)
    return x * jnp.exp2(half) * jnp.exp2(k - half)


def headroom(dtype, keys: int, initial: bool) -> bool:
    """Whether a call's weighted values can pass fp32's largest: values of `dtype` whose largest times
    `keys + 1` reaches `2^HEADROOM` (bf16's and fp32's, not fp16's), or an initial state, whose sums may
    be any size. A call that cannot compiles no guard (`guarded`)."""
    return initial or float(jnp.finfo(dtype).max) * (keys + 1) >= 2.0 ** HEADROOM


HEADROOM = 126
"""The exponent a state's weighted values stay within (`needed`): one below fp32's largest binade, so a
tile's product adds at most as much again and the sum stays finite."""


def needed(total: jax.Array, reach: jax.Array) -> jax.Array:
    """The shift a row's sums need, `[...]` fp32 integers: none while its natural denominator `total` times
    the largest value it weighs, `reach`, is below `2^HEADROOM`, the weighted values' bound; past it, as
    many binades as that product passes it by (log2 rounded up, so the bound holds whatever their
    rounding)."""
    binades = jnp.ceil(jnp.log2(total) + jnp.log2(reach)) - HEADROOM
    return jnp.where(jnp.isfinite(binades) & (binades > 0), binades, 0.0)


WEIGHT_SHIFT = 64
"""The most of a row's shift its weights take (`room`): the rest scales the tile's product, so no weight
of 2^-62 or more is pushed below fp32's least normal however large the shift."""


class Room(NamedTuple):
    """A guarded tile's arithmetic (`room`): its weights at `2^-min(shift, WEIGHT_SHIFT)`, the factor the
    state's sums take, the row's new shift and value bound, and `beyond`, the shift past its weights', which
    the tile's product and weight sums take (`beyond_weights`)."""

    weights: jax.Array
    factor: jax.Array
    shift: jax.Array
    reach: jax.Array
    beyond: jax.Array


def beyond_weights(x: jax.Array, room: Room | None) -> jax.Array:
    """A tile's product `[bq, Dv]` or weight sum `[bq]` taken from its weights' shift to the row's; as it
    is on the plain walk (`room` None)."""
    if room is None:
        return x
    return power2(x, room.beyond if x.ndim == room.beyond.ndim else expand(room.beyond, x))


def tile(state: Partial, alpha: jax.Array, p: jax.Array, value: jax.Array
         ) -> tuple[jax.Array, jax.Array, Room | None]:
    """A tile's weights for its product and the factor the state's sums take, with the guarded walk's
    `Room` (`room`), None on the plain walk, whose weights and factor are `p` and `alpha` as ever."""
    if state.shift is None:
        return p, alpha, None
    held = room(state, alpha, p, value)
    return held.weights, held.factor, held


def stepped(state: Partial, m: jax.Array, factor: jax.Array, p: jax.Array, product: jax.Array,
            held: Room | None) -> Partial:
    """The state after a tile: its sums taken by `factor` and the tile's `product` of its weights `p` and
    their sum added at the row's shift (`beyond_weights`; `tile`), the plain walk's arithmetic where
    `held` is None."""
    denominator = factor * state.total + beyond_weights(p.sum(axis=-1), held)
    acc = rescale(state.acc, factor) + beyond_weights(product, held)
    if held is None:
        return Partial(acc, m, denominator)
    return Partial(acc, m, denominator, held.shift, held.reach, state.live)


def room(state: Partial, alpha: jax.Array, p: jax.Array, value: jax.Array) -> Room:
    """A tile's weights `p` (`weights`) and the factor its state's sums take (`alpha`, `peak`'s) with each
    live row's shift (`needed`; `Partial.live`). The weighted values of a row can pass fp32's largest only
    where its values are near it (a sum of up to `total` of them); a shift of `2^-k` on its sums keeps
    them within it, exactly, as a power of two scales, and leaves every ratio the softmax takes, so the
    output is the unshifted one. A row whose plain walk did not overflow never shifts, its bits the plain
    walk's. The weights take at most `WEIGHT_SHIFT` of a shift, the product and weight sums the rest
    (`beyond_weights`), so a weight lost below fp32's least normal is under `2^-62`: a live row's weighted
    values pass `2^128` in some column, so its output there is at least `2^128 / total`, and such a weight
    on a value at most fp32's largest moves its output by under `2^66 / total`, `2^-62` of that."""
    # The largest value magnitude of the keys each row weighs, as a key the row does not see (padding, a
    # stale cache slot, another document) would shift it for nothing.
    largest = jnp.max(jnp.abs(value.astype(jnp.float32)), axis=-1)
    reach = jnp.maximum(state.reach, jnp.where(p > 0, across(largest, p), 0.0).max(axis=-1))
    natural = power2(alpha * state.total, state.shift) + p.sum(axis=-1)
    shift = jnp.where(state.live, jnp.maximum(state.shift, needed(natural, reach)), state.shift)
    weight = jnp.minimum(shift, WEIGHT_SHIFT)
    return Room(power2(p, expand(-weight, p)), power2(alpha, state.shift - shift), shift, reach,
                weight - shift)


def base2_maximum(m: jax.Array) -> jax.Array:
    """A natural initial maximum (a sink's logit, a partial's) in base 2, `m * log2(e)`, saturated at fp32's
    largest where that product passes it. Exact as far as any key reads it: a folded logit is at most fp32's
    largest less an ulp (`Forward.base2`), so a maximum past fp32's largest in base 2 exceeds every key's
    by over an ulp there, 2^104, as its saturation does, and every key's weight is zero under either, the
    state's own one; one past its negative is below every key's alike."""
    folded = m * LOG2E
    largest = float(jnp.finfo(jnp.float32).max)
    return jnp.where(jnp.isinf(folded) & jnp.isfinite(m), jnp.sign(m) * largest, folded)


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
    state = headroomed(Partial(acc, m, total), live)
    shift = jnp.where(live, needed(total, state.reach), 0.0)
    return state._replace(acc=power2(acc, expand(-shift, acc)), total=power2(total, -shift), shift=shift)


def modify(spec: Forward, s: jax.Array, coords: Coords, params: tuple[jax.Array, ...] = (),
           masked: bool = True) -> jax.Array:
    """A tile's logits from its products `s` `[bq, bk]` (fp32): scaled (by log2(e) too in base 2; raw in
    the raw form, `Forward.raw`), then the score hook, then -inf at each key the query does not see.
    `masked` False is a tile every query of the block sees whole (a full tile of the block map), which
    takes no mask."""
    logits = s if spec.raw else s * (spec.scale * LOG2E if spec.base2 else spec.scale)
    if spec.score_mod is not None:
        logits = hooked(spec.score_mod, logits, coords.q_pos, coords.k_pos, params)
    if not masked:
        return logits
    seen = None
    if spec.form == "spans":
        start, end = _given(coords.start, coords.end, "its queries' intervals")
        seen = (coords.k_pos >= expand(start, s)) & (coords.k_pos < expand(end, s))
    elif spec.form == "causal":
        seen = coords.k_pos <= coords.q_pos
    if spec.form != "spans" and spec.keys % s.shape[1]:
        # A tile past the call's last key reads padding.
        seen = coords.k_pos < spec.keys if seen is None else seen & (coords.k_pos < spec.keys)
    if spec.ids:
        query_ids, key_ids = _given(coords.query_ids, coords.key_ids, "its queries' and keys' ids")
        same = expand(query_ids, s) == across(key_ids, s)
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
    return new, spec.carry(m - _safe(spec, new))


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
    key that holds it `[bq]` int32: the backward's anchor key (`off_anchor`), a residual of the forward."""

    peak: jax.Array
    key: jax.Array


def no_argmax(rows: int) -> Argmax:
    """`Argmax` before any key."""
    return Argmax(jnp.full((rows,), -jnp.inf, jnp.float32), jnp.zeros((rows,), jnp.int32))


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


def split_accumulate(others: jax.Array, own: jax.Array, others_total: jax.Array, alpha: jax.Array,
                     moved: jax.Array, old_value: jax.Array, product: jax.Array, rest: jax.Array,
                     own_tile: jax.Array, held: Room | None = None) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A training forward's weighted values kept apart from the anchor key's (`others`, `B`, with the
    initial state's), the anchor's own weight, and the other keys' weights' sum (`others_total`, keys
    alone), after a tile: where the anchor moved to this tile (`moved`), the old anchor's term (its
    weight times its value row, `old_value`, which the mainloop gathers, and its weight) joins `B` and
    the sum before the rescale, and the new anchor's weight is the tile's (`own_tile`); the tile's other
    keys add their `product` and their narrowed weights `rest`. So neither is summed with a key's weight
    cancelled out of it. On the guarded walk the tile's product, sum and own weight take the row's shift
    past its weights' (`beyond_weights`)."""
    old = old_value.astype(jnp.float32)
    joined = jnp.where(expand(moved, old), expand(own, old) * old, 0.0)
    total = (alpha * (others_total + jnp.where(moved, own, 0.0))
             + beyond_weights(rest.astype(jnp.float32).sum(axis=-1), held))
    own_tile = beyond_weights(own_tile.astype(jnp.float32), held)
    return (rescale(others + joined, alpha) + beyond_weights(product, held),
            jnp.where(moved, own_tile, alpha * own), total)


def split_output(others: jax.Array, own: jax.Array, anchor_value: jax.Array) -> jax.Array:
    """The weighted values of every key, `B` and the anchor's own term (`anchor_value`, its value row)."""
    return others + expand(own, others) * anchor_value.astype(jnp.float32)


def merge(spec: Forward, first: Partial, second: Partial) -> Partial:
    """Two states over disjoint keys as one, `first`'s keys before `second`'s: each taken to the larger
    maximum and summed, in that order. Either may hold no key whatever the call (a slice past a row's
    keys), so `empty` is its identity under every spec."""
    m = jnp.maximum(first.m, second.m)
    safe = jnp.where(m == -jnp.inf, 0.0, m)
    if first.shift is None and second.shift is None:
        a, b = spec.carry(first.m - safe), spec.carry(second.m - safe)
        acc = expand(a, first.acc) * first.acc + expand(b, second.acc) * second.acc
        return Partial(acc, m, a * first.total + b * second.total)
    first, second = headroomed(first), headroomed(second)
    reach = jnp.maximum(first.reach, second.reach)
    carried = spec.carry(first.m - safe), spec.carry(second.m - safe)
    natural = (power2(carried[0] * first.total, first.shift)
               + power2(carried[1] * second.total, second.shift))

    def joined(shift):
        a, b = power2(carried[0], first.shift - shift), power2(carried[1], second.shift - shift)
        acc = expand(a, first.acc) * first.acc + expand(b, second.acc) * second.acc
        return acc, a * first.total + b * second.total

    # Joined at the larger shift; a row whose join overflows (two finite slices can) shifts as it needs,
    # every other row's bits the plain join's.
    held = jnp.maximum(first.shift, second.shift)
    acc, _ = joined(held)
    live = overflowed(acc) | first.live | second.live
    shift = jnp.where(live, jnp.maximum(held, needed(natural, reach)), held)
    acc, total = joined(shift)
    return Partial(acc, m, total, shift, reach, live)


def finish(spec: Forward, state: Partial, *, initial: bool, output_mod: Callable | None = None,
           output_params: tuple[jax.Array, ...] = ()) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The block's output `[bq, Dv]` fp32 (its weighted values over its denominator, zero for a query
    that saw nothing and started from no state, then the output hook) and its residuals: the maximum (0
    for such a query, as the reference's) in the units the backward recomputes its logits in, natural, or
    raw products where the forward's are (`Forward.raw`), and the denominator, unshifted (`room`); the
    output is the shifted sums' ratio, which the shift leaves."""
    m = state.m if spec.raw else spec.naturalized(state.m)
    denominator = state.total
    if spec.guarded or initial:
        out = jnp.where(expand(denominator > 0, state.acc),
                        state.acc / expand(jnp.where(denominator > 0, denominator, 1.0), state.acc), 0.0)
        m = jnp.where(m == -jnp.inf, 0.0, m)
    else:
        out = state.acc / expand(denominator, state.acc)
    if output_mod is not None:
        out = output_mod(out, *output_params)
    return out, m, denominator if state.shift is None else power2(denominator, state.shift)


# The backward's algebra: what lies between its products (S = Q·Kᵀ, dP = dO·Vᵀ, then dV, dK and dQ),
# on natural logits.


def inverse(total: jax.Array) -> jax.Array:
    """Each query's `1 / total`, zero where it saw nothing (total 0)."""
    return jnp.where(total > 0, 1.0 / jnp.where(total > 0, total, 1.0), 0.0)


def floating_params(params) -> list[int]:
    """The places of a hook's floating parameters, the ones it differentiates along."""
    return [index for index, param in enumerate(params) if jnp.issubdtype(param.dtype, jnp.floating)]


def hooked(score_mod: Callable, logits: jax.Array, q_pos, k_pos, params) -> jax.Array:
    """`score_mod` on a tile; a custom-derivative call in it lowers as its primal function (Dew's JAX
    build, constraints.txt)."""
    return score_mod(logits, q_pos, k_pos, *params)


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
                     width: int, shift: jax.Array) -> tuple[OffAnchor, jax.Array, jax.Array]:
    """The first pass from the softmax's initial state (its maximum, denominator and share of D's
    numerator) weighed at the forward's `maximum`, its sums at each row's `2^-shift` (the forward's,
    `room`), and that state's denominator and share there, unshifted."""
    carried = jnp.where(peak0 == -jnp.inf, 0.0, jnp.exp(jnp.minimum(peak0 - maximum, 0.0)))
    state_total, state_share = total0 * carried, term0 * carried
    nothing = jnp.zeros_like(total0)
    first = OffAnchor(power2(state_total, -shift), jnp.zeros((total0.shape[0], width), jnp.float32),
                      nothing, nothing)
    return first, state_total, state_share


def off_anchor_weights(logits: jax.Array, maximum: jax.Array, at_anchor: jax.Array,
                       dtype, shift: jax.Array, how: Recompute = PLAIN
                       ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A tile's weights `exp(s - m)` at the forward's maximum, fp32, at the forward's weights' shift,
    `2^-min(shift, WEIGHT_SHIFT)` (`room`); them narrowed to the values' `dtype` as the forward rounds
    them, zero at the anchor key (`at_anchor`), the left operand of `B`'s product; and the anchor key's
    narrowed weight, fp32."""
    exponentials = below(logits, maximum, how)
    exponentials = power2(exponentials, expand(-jnp.minimum(shift, WEIGHT_SHIFT), exponentials))
    rounded = exponentials.astype(dtype)
    rest = jnp.where(at_anchor, jnp.zeros_like(rounded), rounded)
    own = jnp.where(at_anchor, rounded, jnp.zeros_like(rounded)).astype(jnp.float32).sum(axis=-1)
    return exponentials, rest, own


def off_anchor(state: OffAnchor, exponentials: jax.Array, rest: jax.Array, own: jax.Array,
               others: jax.Array, shift: jax.Array) -> OffAnchor:
    """The first pass after a tile (`off_anchor_weights`), `others` the tile's `rest · V`, its sums taken
    from the weights' shift to each row's `shift`, as the forward's (`beyond_weights`)."""
    beyond = jnp.minimum(shift, WEIGHT_SHIFT) - shift

    def taken(x):
        return power2(x, beyond if x.ndim == beyond.ndim else expand(beyond, x))

    return OffAnchor(state.total + taken(exponentials.sum(axis=-1)), state.others + taken(others),
                     state.others_total + taken(rest.astype(jnp.float32).sum(axis=-1)),
                     state.own + taken(own))


def off_anchor_delta(state: OffAnchor, anchor_value: jax.Array, dout: jax.Array, state_total: jax.Array,
                     state_share: jax.Array, key_index: jax.Array,
                     shift: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Each query's anchor c (`v_c · dO`, or 0), `D - c`, and the anchor key's index (-1 where c is 0),
    from the first pass. D's numerator two ways: about 0, `sum_j e_j v_j · dO`, as the output's own
    product with dO rounds it, which suits a query whose weights spread over many keys; and about
    `c = v_c · dO`, `sum_{j != c} e_j (v_j - v_c) · dO`, where c enters as exactly 0, which suits one
    whose weight sits on c. Each query takes the one nearer its D, the one whose terms cancel less. The
    pass's sums are at each row's `2^-shift` (`room`), and the state's are taken there."""
    cotangent = dout.astype(jnp.float32)
    anchor_value = anchor_value.astype(jnp.float32)
    anchor = jnp.sum(anchor_value * cotangent, axis=-1)
    state_total, state_share = power2(state_total, -shift), power2(state_share, -shift)
    own, others_total = expand(state.own, anchor_value), expand(state.others_total, anchor_value)
    about_zero = jnp.sum((state.others + own * anchor_value) * cotangent, axis=-1) + state_share
    about_anchor = (jnp.sum((state.others - others_total * anchor_value) * cotangent, axis=-1)
                    + state_share - state_total * anchor)
    anchored = jnp.abs(about_anchor) <= jnp.abs(about_zero)
    numerator = jnp.where(anchored, about_anchor, about_zero)
    return (jnp.where(anchored, anchor, 0.0), numerator * inverse(state.total),
            jnp.where(anchored, key_index, -1))


def split_delta(others: jax.Array, own: jax.Array, others_total: jax.Array, total: jax.Array,
                state_total: jax.Array, anchor_value: jax.Array, dout: jax.Array,
                key_index: jax.Array, shift: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`off_anchor_delta` from a training forward's split sums (`split_accumulate`), the backward's first
    pass taken by the forward: `B` with the initial state's share inside it, the other keys' weights'
    sum, the anchor's own weight, the denominator `total` and the initial state's own `state_total`, all
    at the forward's maximum. D's numerator about 0 is `(B + own v_c) · dO`, about `c = v_c · dO` it is
    `(B - others_total v_c) · dO - state_total c`, its terms kept apart so a state or keys that weigh
    little beside the anchor are not lost from a total; each query takes the one nearer its D. The split
    sums are at each row's `2^-shift` (`room`), and `total` and `state_total` are taken there."""
    total, state_total = power2(total, -shift), power2(state_total, -shift)
    cotangent = dout.astype(jnp.float32)
    anchor_value = anchor_value.astype(jnp.float32)
    anchor = jnp.sum(anchor_value * cotangent, axis=-1)
    about_zero = jnp.sum(split_output(others, own, anchor_value) * cotangent, axis=-1)
    about_anchor = (jnp.sum((others - expand(others_total, anchor_value) * anchor_value) * cotangent, axis=-1)
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
    return sums.dq * expand(ratio, sums.dq), delta + refined, total, mass, grads


def below(logits: jax.Array, peak: jax.Array, how: Recompute = PLAIN) -> jax.Array:
    """`exp(s - m)` for a tile's logits recomputed in the backward against the forward's maximum `m`,
    fp32, in the forward's units (`Recompute`): `m` is every key's maximum, so a logit past it by the
    rounding the two passes do not share (the forward's maximum taken in base 2 and back, or another tile
    shape's product) is the maximum itself, its weight 1, never past it; in the accurate form one at it
    weighs 1 however a compiler contracts its scaling into the subtraction (`weights`)."""
    peak = expand(peak, logits)
    if how.factor is not None:
        # Raw products: the difference is taken first, so the largest's is exactly 0.
        return jnp.exp(jnp.minimum(logits - peak, 0.0) * how.factor)
    if how.exact:
        return jnp.exp(jnp.where(logits < peak, logits - peak, 0.0))
    return jnp.exp(jnp.minimum(logits - peak, 0.0))


def normalized(logits: jax.Array, peak: jax.Array, total: jax.Array, how: Recompute = PLAIN
               ) -> tuple[jax.Array, jax.Array]:
    """A tile's unnormalized weights `exp(s - m)` (`below`) and its weights `p`, fp32."""
    exponentials = below(logits, peak, how)
    return exponentials, exponentials * expand(inverse(total), exponentials)


def grad_scores(exponentials: jax.Array, products: jax.Array, anchor: jax.Array, delta: jax.Array,
                seen: jax.Array, at_anchor: jax.Array | None, inverse_total: jax.Array) -> jax.Array:
    """The score cotangents `p (dP - D)` of a tile in the reference's own order, as JAX differentiates
    `exp(s - m) / T`: each query's `(dP - c) - (D - c)` about its anchor c (`off_anchor_delta`) over its
    denominator (`inverse_total`), then times the unnormalized weights `exp(s - m)`, so every
    intermediate lies where the reference's does (a weight below fp32's range keeps its share beside a
    large product, and none overflows that the reference keeps); zero where unseen. The anchor key's
    own `dP - c` is 0 by definition where `at_anchor` names it, whatever its product's rounding."""
    relative = products - expand(anchor, products)
    if at_anchor is not None:
        relative = jnp.where(at_anchor, 0.0, relative)
    centred = (relative - expand(delta, relative)) * expand(inverse_total, relative)
    return jnp.where(seen, exponentials * centred, 0.0)


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
    scaled = dout.astype(jnp.float32) * expand(inverse(total), dout)
    return exponentials.astype(dout.dtype).astype(jnp.float32).T, scaled
