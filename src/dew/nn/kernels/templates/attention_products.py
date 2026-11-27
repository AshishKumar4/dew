"""A step's score products declared as a list with their combine (kernel-templates.md §3, §A rows 16a and 25),
each combine lowered onto one call of the template, its one softmax over one value:

- `SUM` (absorbed MLA's content and rope products): `q1·k1ᵀ + q2·k2ᵀ` is the one product of the queries and
  the keys side by side along their width, `[q1 | q2]·[k1 | k2]ᵀ`, exactly (`summed`).
- `Banded` (piecewise attention, FlagAttention's `piecewise_attention`): product `p` takes the keys a band of
  distances from its query holds, the first those within `bands[0]`, the last every farther one. The keys
  stand once a product side by side along the keys' axis, product `p`'s copy its own columns of the keys'
  width (`[0 | k_p | 0]`) beside the queries side by side (`[q1 | q2 | ...]`), so each entry of copy `p` is
  `q_p·k_p` exactly, the value twice over, and each query's spans hold its band's keys of copy `p` alone
  (`selected`): a walk steps a copy's tiles where its band reaches them, both at a tile a band's edge cuts
  (the spec's tile classes), one softmax across them. Its every form, mainloop and backward is the
  template's own; the copies' gradients sum back (`jax.vjp` of the layout). Against one walk holding both
  products' operands, a visited tile's logits cost `len(products)` times the products' width and the keys'
  copies their bytes; the one walk needs a second query tile and key ring resident and, in the backward, a
  dQ and dK accumulator a product (three at width 128 past a 64 x 64 tile on two consumers' registers).
- `PARALLAX` (Parallax, arXiv 2605.29157): a second, unscaled query's product `t = r·k2ᵀ` weighs a second
  accumulator beside the softmax's, `p2 = p ∘ t`, the output `O1/d1·(1 + d2/d1) - O2/d1`, which is
  `Σ_j a_j (1 + τ - t_j) v_j` (`a` the softmax, `τ = Σ_j a_j t_j`): the softmax's output less its derivative
  along the queries `r / scale`, `Σ_j a_j (t_j - τ) v_j`. That derivative is the template's backward's dQ of
  one call (`parallax`): its queries `[q | 0]` over keys `[k | v]`, values `[k2 | v]` and the output's
  cotangent `[r | 0]`, whose score cotangents are `a (r·k2 - r·Σ a k2)`, so its dQ's value columns are
  `scale Σ_j a_j (t_j - τ) v_j` and its output's are the softmax's: a forward and a backward of the
  template, every mainloop's and form's. Its derivative is the plain formula's in XLA (`surrogate`); one
  walk holding the second product and accumulator, and its backward's quadratic term `a t (g - G)` (each
  entry's two products' product), are the open lever.
"""

import dataclasses
import itertools
import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp

from dew.nn.derivatives import contracted, surrogate
from dew.nn.kernels.templates import attention_route
from dew.nn.kernels.templates.attention import (
    REFERENCE_PRECISION,
    Hooks,
    KeySpans,
    Kind,
    dense,
    kept_value,
    kind_of,
    logit_scale,
)
from dew.nn.kernels.templates.attention_mosaic_program import score_places
from dew.nn.kernels.templates.hooks import Hook, Side
from dew.nn.precision import float32_einsum

SUM = "sum"
"""The combine summing a step's products (`summed`)."""

PARALLAX = "parallax"
"""The combine weighing a second accumulator by the second product (`parallax`)."""


@dataclasses.dataclass(frozen=True)
class Banded:
    """The combine selecting a step's products by the distance of each key from its query (`selected`):
    product 0 takes the keys within `bands[0]` of it, `|i - j| <= bands[0]`, product `p` those past
    `bands[p - 1]` and within `bands[p]`, the last every key past `bands[-1]`; so `len(bands) + 1` products,
    disjoint and covering every key by construction. A distance is a key's index less its query's place
    (`selected`'s `places`)."""

    bands: tuple[int, ...]

    def __post_init__(self):
        bands = tuple(self.bands)
        if not bands or any(not isinstance(band, int) or band < 0 for band in bands):
            raise ValueError(f"a banded combine's bands are distances, whole and at least 0: {bands}")
        if any(later <= earlier for earlier, later in itertools.pairwise(bands)):
            raise ValueError(f"a banded combine's bands increase, each product's keys its own: {bands}")


def summed(queries: Sequence[jax.Array], keys: Sequence[jax.Array]) -> tuple[jax.Array, jax.Array]:
    """The query and key of the products `queries[p]·keys[p]ᵀ` summed: each side by side along its width."""
    _paired(queries, keys)
    return jnp.concatenate(queries, axis=-1), jnp.concatenate(keys, axis=-1)


def selected(queries: Sequence[jax.Array], keys: Sequence[jax.Array], value: jax.Array, spans: KeySpans,
             combine: Banded, hooks: Hooks | None = None, places: jax.Array | None = None
             ) -> tuple[jax.Array, jax.Array, jax.Array, KeySpans, Hooks]:
    """The one call of the template whose softmax is the products `queries[p]` `[B, Q, H, Dp]` by `keys[p]`
    `[B, K, Hk, Dp]` over `value` `[B, K, Hk, Dv]` selected by `combine` (`Banded`) within `spans`, under
    `hooks`: `(query, key, value, spans, hooks)` as the module's docstring lays them, each query's distances
    counted from its place `places` `[B, Q]` (its index by default). A score hook's KEY and TILE parameters
    stand once a copy; one reading its keys' indices is refused (each copy's are its own), as are a band's
    and the prologues (each product's query and key take theirs before the call); a dropping call draws
    each entry at its key's own place."""
    hooks = Hooks() if hooks is None else hooks
    products = _paired(queries, keys)
    if products != len(combine.bands) + 1:
        raise ValueError(f"{len(combine.bands)} bands select among {len(combine.bands) + 1} products, "
                         f"not {products}")
    if hooks.query.fn is not None or hooks.key.fn is not None:
        raise ValueError("a selected call's products take their prologues before it: each its own query "
                         "and key")
    count = spans.keys
    if any(key.shape[1] != count for key in keys) or value.shape[1] != count:
        raise ValueError(f"each product's keys and the value are the spans' {count} keys")
    widths = [query.shape[-1] for query in queries]
    query = jnp.concatenate(queries, axis=-1)
    key = jnp.concatenate([jnp.pad(key, ((0, 0), (0, 0), (0, 0), (sum(widths[:p]), sum(widths[p + 1:]))))
                           for p, key in enumerate(keys)], axis=1)
    value = jnp.concatenate([value] * products, axis=1)
    return query, key, value, _banded_spans(spans, combine, places), _copied_hooks(hooks, products, count)


def attend(queries: Sequence[jax.Array], keys: Sequence[jax.Array], value: jax.Array, spans: KeySpans,
           combine: "str | Banded", hooks: Hooks | None = None, *, places: jax.Array | None = None,
           scale: float | None = None, natural: bool | None = None,
           accurate: bool | None = None) -> jax.Array:
    """Attention whose logits are the products `queries[p]·keys[p]ᵀ` combined by `combine` (`SUM` or a
    `Banded` selection), on the template's route (`attention_route.attend`): `[B, Q, H, Dv]`, differentiable
    in every product's query and key, the value and the hooks. `scale` is a Python float, `D ** -0.5` by
    default, `D` the summed products' width or a selected product's (the first's; give it where they
    differ)."""
    if combine == SUM:
        query, key = summed(queries, keys)
        return attention_route.attend(query, key, value, spans, hooks, scale=scale, natural=natural,
                                      accurate=accurate)
    if combine == PARALLAX:
        if hooks is not None:
            raise ValueError("a parallax call takes no hooks: its second accumulator is weighed by the "
                             "softmax's own weights")
        _paired(queries, keys)
        if len(queries) != 2:
            raise ValueError(f"a parallax call holds two products, the softmax's and the second's, not "
                             f"{len(queries)}")
        # Its value is a forward and a backward of the template, the backward's in its accurate form whatever
        # `accurate` names (`_parallax_kernels`).
        return parallax(queries[0], queries[1], keys[0], keys[1], value, spans, scale=scale, natural=natural)
    if not isinstance(combine, Banded):
        raise ValueError(f"a step's products combine by SUM, a Banded selection or PARALLAX, not {combine!r}")
    # The keys' and the value's copies laid out wide and their gradients handed back unrounded
    # (`attention_route.attend`'s `operand_dtype`), so each sums its copies' and rounds once.
    dtype = value.dtype
    query, key, value, spans, hooks = selected(queries, [key.astype(jnp.float32) for key in keys],
                                               value.astype(jnp.float32), spans, combine, hooks, places)
    scale = queries[0].shape[-1] ** -0.5 if scale is None else scale
    return attention_route.attend(query, key, value, spans, hooks, scale=scale, natural=natural,
                                  accurate=accurate, operand_dtype=dtype)


def parallax(query: jax.Array, second: jax.Array, key: jax.Array, second_key: jax.Array, value: jax.Array,
             spans: KeySpans, *, scale: float | None = None, natural: bool | None = None) -> jax.Array:
    """Parallax's attention `[B, Q, H, Dv]` (`PARALLAX`): the softmax of `query` `[B, Q, H, D]` by `key`
    `[B, K, Hk, D]` over `value` `[B, K, Hk, Dv]` within `spans`, its weights each taken by
    `1 + τ - t` where `t = second · second_keyᵀ`, unscaled, and `τ` its mean under the weights; on the
    template's kernels (the module's docstring), differentiable in every operand by the plain formula
    (`_parallax_formula`). `scale` is `D ** -0.5` by default."""
    scale = logit_scale(query.shape[-1], scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"a parallax call's logit scale is positive and finite, its second product's "
                         f"derivative taken along the queries over it: {scale}")

    def kernels(query, second, key, second_key, value, spans):
        return _parallax_kernels(query, second, key, second_key, value, spans, scale=scale, natural=natural)

    def derivative(primals, tangents, out):
        del out
        *operands, spans = primals

        def formula(query, second, key, second_key, value):
            return _parallax_formula(query, second, key, second_key, value, spans, scale=scale)

        return jax.jvp(formula, tuple(operands), tuple(tangents[:-1]))[1]

    rule = contracted("test_parallax_holds_every_order")(surrogate(kernels, derivative))
    return rule(query, second, key, second_key, value, spans)


def _parallax_kernels(query, second, key, second_key, value, spans: KeySpans, *, scale: float,
                      natural) -> jax.Array:
    """`parallax`'s value on the template's kernels: the softmax's output and its dQ along `second / scale`,
    one forward and one backward of the call the module's docstring lays out (`jax.vjp` of the route). The
    backward takes its accurate products and D about each query's anchor from its split sums whatever the
    call's form: D here is `τ`, the second product's mean, which a rounded output's (the fast
    form's `rowsum(dO ⊙ O)`) moved by the second keys' rounding, not their spread (keys of 8192 and 8200 in
    fp16 gave a mean of 8192). The cotangent takes the logit scale's inverse to its nearest power of two,
    exactly, and the dQ the rest of it, so neither under- nor overflows where the result does not."""
    width = query.shape[-1]
    laid_query = jnp.concatenate([query, jnp.zeros((*query.shape[:-1], value.shape[-1]), query.dtype)], -1)
    laid_key = jnp.concatenate([key, value], axis=-1)
    laid_value = jnp.concatenate([second_key, value], axis=-1)

    def attended(laid_query):
        return attention_route.attend(laid_query, laid_key, laid_value, spans, scale=scale, natural=natural,
                                      accurate=True)

    out, pull = jax.vjp(attended, laid_query)
    power = float(2.0 ** round(-math.log2(scale)))
    cotangent = jnp.concatenate([(second * jnp.asarray(power, second.dtype)).astype(out.dtype),
                                 jnp.zeros((*out.shape[:-1], value.shape[-1]), out.dtype)], axis=-1)
    (moved,) = pull(cotangent)
    softmax = out[..., second_key.shape[-1]:].astype(jnp.float32)
    along = moved[..., width:].astype(jnp.float32) * jnp.float32(1.0 / (scale * power))
    return (softmax - along).astype(query.dtype)


def _parallax_formula(query, second, key, second_key, value, spans: KeySpans, *, scale: float) -> jax.Array:
    """`parallax` in XLA ops, as the template's reference computes its softmax (`template.reference`): the
    logits' and the second product's entries fp32, the weights `a (1 + τ - t)` rounded to the value's dtype
    for their product, the output rounded to the query's dtype; a row that sees no key 0."""
    batch, queries, heads, _ = query.shape
    keys, kv_heads = key.shape[1], key.shape[2]
    wide = jnp.promote_types(query.dtype, jnp.float32)

    def products(left, right):
        grouped = left.reshape(batch, queries, kv_heads, heads // kv_heads, left.shape[-1])
        return float32_einsum("bqhgd,bkhd->bhgqk", grouped, right, precision=REFERENCE_PRECISION,
                              preferred_element_type=wide).reshape(batch, heads, queries, keys)

    seen = dense(spans)[:, None]
    logits = jnp.where(seen, products(query, key) * jnp.asarray(scale, wide), -jnp.inf)
    m = jnp.max(logits, axis=-1, initial=-jnp.inf)
    p = jnp.exp(logits - jnp.where(jnp.isfinite(m), m, 0)[..., None])
    total = jnp.sum(p, axis=-1, keepdims=True)
    weights = jnp.where(total > 0, p / jnp.where(total > 0, total, 1), 0)
    seconds = jnp.where(seen, products(second, second_key), 0)
    mean = jnp.sum(weights * seconds, axis=-1, keepdims=True)
    taken = (weights * (1 + mean - seconds)).reshape(batch, kv_heads, heads // kv_heads, queries, keys)
    out = float32_einsum("bhgqk,bkhd->bhgqd", taken.astype(value.dtype), value, precision=REFERENCE_PRECISION,
                         preferred_element_type=wide).reshape(batch, heads, queries, value.shape[-1])
    return jnp.swapaxes(out.astype(query.dtype), 1, 2)


def _paired(queries: Sequence[jax.Array], keys: Sequence[jax.Array]) -> int:
    """The count of products `queries[p]·keys[p]ᵀ`, at least two, each pair one width, every query of one
    batch, length and head count and every key alike."""
    if len(queries) != len(keys) or len(queries) < 2:
        raise ValueError(f"a step's products pair a query with a key, two or more: {len(queries)} queries, "
                         f"{len(keys)} keys")
    for query, key in zip(queries, keys, strict=True):
        if query.shape[-1] != key.shape[-1]:
            raise ValueError(f"a product's query and key are one width: {query.shape} and {key.shape}")
        if query.shape[:-1] != queries[0].shape[:-1] or key.shape[:-1] != keys[0].shape[:-1]:
            raise ValueError("a step's products' queries are one batch, length and head count, their keys "
                             "alike")
    return len(queries)


def _banded_spans(spans: KeySpans, combine: Banded, places: jax.Array | None) -> KeySpans:
    """`spans` over the keys' copies (`selected`): each query's runs of copy `p` its spans' keys within band
    `p` of its place, offset by `p` copies' keys, side by side in their copies' order."""
    spans = spans.per_query()
    count, bands = spans.keys, combine.bands
    start, end = ((spans.start, spans.end) if spans.start.ndim == 3
                  else (spans.start[..., None], spans.end[..., None]))
    place = (jnp.broadcast_to(jnp.arange(start.shape[1], dtype=jnp.int32)[None, :], start.shape[:2])
             if places is None else jnp.asarray(places, jnp.int32))[..., None]

    def within(lo, hi):
        # Each run's keys within `[lo, hi)` of the query's index, empty where none.
        return jnp.maximum(start, jnp.clip(lo, 0, count)), jnp.minimum(end, jnp.clip(hi, 0, count))

    # A band past every distance a query and a key of the call lie apart reads as that distance, so no edge
    # passes int32's range.
    reach = start.shape[1] + count
    bands = tuple(min(band, reach) for band in bands)
    copies = []
    for p in range(len(bands) + 1):
        if p == 0:
            sides = [within(place - bands[0], place + bands[0] + 1)]
        else:
            # Past band p - 1 and within band p: the keys before the query, then those after it; the last
            # band's outer edges the keys' own, whatever the query's place (one past the keys, a rectangular
            # call's).
            before = place - bands[p] if p < len(bands) else jnp.zeros_like(place)
            after = place + bands[p] + 1 if p < len(bands) else jnp.full_like(place, count)
            sides = [within(before, place - bands[p - 1]), within(place + bands[p - 1] + 1, after)]
        lo = jnp.concatenate([side[0] for side in sides], axis=-1)
        hi = jnp.concatenate([side[1] for side in sides], axis=-1)
        copies.append((lo + p * count, hi + p * count, p * count))
    lo = jnp.concatenate([copy[0] for copy in copies], axis=-1)
    hi = jnp.concatenate([copy[1] for copy in copies], axis=-1)
    # Runs disjoint and in order (`KeySpans`): an empty one held at the end of those before it.
    floor = jax.lax.cummax(jnp.concatenate([jnp.zeros_like(hi[..., :1]), hi[..., :-1]], axis=-1), axis=2)
    lo = jnp.maximum(lo, floor)
    hi = jnp.maximum(hi, lo)
    products = len(bands) + 1
    key_ids = None if spans.key_ids is None else jnp.concatenate([spans.key_ids] * products, axis=1)
    return KeySpans(lo, hi, spans.query_ids, key_ids, products * count, "spans")


def _copied_hooks(hooks: Hooks, products: int, count: int) -> Hooks:
    """`hooks` over the keys' `products` copies (`selected`): a KEY or TILE parameter once a copy, a dropping
    call's draw at each copy's keys' own places."""
    for side in hooks.score.sides:
        if kind_of(side) is Kind.BAND:
            raise ValueError("a band's parameters hold each query's keys about it, which a selected call's "
                             "copies of the keys do not")
    if score_places(hooks)[1]:
        raise ValueError("a selected call's score hook reads no key's index: each copy's keys take indices "
                         "of their own")
    # A side along the keys once a copy, at the dtype the kernels compute it in (`kept_value`), so its
    # copies' gradients sum before they round to its own (a 16-bit bias's near and far terms each past its
    # largest, summing to 0).
    sides = tuple(Side(jnp.concatenate([kept_value(side.value)] * products, axis=side.axes.index("keys")),
                       side.axes)
                  if "keys" in side.axes else side for side in hooks.score.sides)
    drop = hooks.drop
    if drop is not None:
        keys = jnp.arange(count, dtype=jnp.int32) if drop.keys is None else drop.keys
        drop = dataclasses.replace(drop, keys=jnp.concatenate([keys] * products))
    return dataclasses.replace(hooks, score=Hook(hooks.score.fn, sides), drop=drop)
