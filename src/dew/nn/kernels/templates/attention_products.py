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
"""

import dataclasses
import functools
import itertools
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp

from dew.nn.kernels.templates import attention_route
from dew.nn.kernels.templates.attention import (
    Hooks,
    KeySpans,
    Kind,
    kept_value,
    kind_of,
    moved_places,
    prologues_taken,
    reference,
    widened_places,
)
from dew.nn.kernels.templates.hooks import Hook, Side

SUM = "sum"
"""The combine summing a step's products (`summed`)."""



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
    stand once a copy and it reads each key's own index (`_copied_hooks`); a band's parameters and the
    prologues are refused (each product's query and key take theirs before the call); a dropping call draws
    each entry at its key's own place. A call wider than fp32 (`_wide`) keeps its sides' own dtypes, at
    least fp32, where the kernels' copies take fp32 (`_widely`'s).
    """
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
    wide = _wide(*queries, *keys, value, hooks=hooks)
    return query, key, value, _banded_spans(spans, combine, places), _copied_hooks(hooks, products, count,
                                                                                         wide=wide)


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
        wide = _wide(query, key, value, hooks=hooks)
        if wide is not None:
            return _widely(query, key, value, spans, hooks, scale, wide)
        return attention_route.attend(query, key, value, spans, hooks, scale=scale, natural=natural,
                                      accurate=accurate)
    if not isinstance(combine, Banded):
        raise ValueError(f"a step's products combine by SUM or a Banded selection, not {combine!r}")
    # The keys' and the value's copies laid out wide and their gradients handed back unrounded, each
    # computed in its own dtype (`attention_route.attend`'s `operand_dtypes`), so each sums its copies' and
    # rounds once; the queries as they come.
    dtypes = (jnp.result_type(*keys), value.dtype)
    # Widened, never narrowed, to the call's accumulation before their copies are made: fp32 at least, the
    # call's widest where it is wide (an fp64 query's call takes fp16 values to fp64), so two copies'
    # cotangents past fp32 (`±2^129`) sum where the call computes before any rounds (train 1461's review);
    # its hook's sides too where the call is wide (`selected`).
    widest = _wide(*queries, *keys, value, hooks=hooks) or jnp.dtype(jnp.float32)

    def widened(operand):
        return operand.astype(jnp.promote_types(operand.dtype, widest))

    query, key, value, spans, hooks = selected(queries, [widened(key) for key in keys], widened(value), spans,
                                               combine, hooks, places)
    scale = queries[0].shape[-1] ** -0.5 if scale is None else scale
    wide = _wide(query, key, value, hooks=hooks)
    if wide is not None:
        return _widely(query, key, value, spans, hooks, scale, wide, rounded_to=dtypes[1])
    return attention_route.attend(query, key, value, spans, hooks, scale=scale, natural=natural,
                                  accurate=accurate, operand_dtypes=dtypes)


def _wide(*operands, hooks: Hooks | None = None) -> jnp.dtype | None:
    """The dtype a call computes in where its operands or its score hook's sides are wider than fp32
    (x64's), their widest, which the template's arithmetic in XLA ops takes (`_widely`): the route's
    mainloops multiply and sum in fp32, which tied the logits `2^24` and `2^24 + 1` (train 1427's review);
    None for any other call."""
    sides = () if hooks is None else [side.value for side in hooks.score.sides]
    floating = [x for x in (*operands, *sides) if jnp.issubdtype(x.dtype, jnp.floating)]
    widest = jnp.dtype(jnp.result_type(*floating)) if floating else None
    return widest if widest is not None and widest.itemsize > 4 else None


def _widely(query, key, value, spans: KeySpans, hooks: Hooks | None, scale: float | None,
            wide: jnp.dtype, *, rounded_to=None) -> jax.Array:
    """A wide call on the template's arithmetic in XLA ops at its dtype `wide` (`_wide`), forward and
    backward (`reference`): its query and key under their prologues at their own dtypes first
    (`prologues_taken`: a prologue sees its operand's dtype), then the query taken to `wide`, so the
    softmax's dtype is the call's, not the query's alone (an fp32 query over fp64 keys tied the logits
    `2^24` and `2^24 + 1`); its output in the query's own dtype. Values a Banded call widened for its copies'
    gradients weigh at their own dtype, `rounded_to` (the route's `operand_dtypes`): fp16 values under fp64
    logits `[0, -18]` read 0, where weighed at fp32 they read 1.52e-8 (train 1444's review)."""
    taken, key, hooks = prologues_taken(query, key, hooks)
    return reference(taken.astype(wide), key, value, spans, hooks, scale=scale,
                     rounded_to=rounded_to).astype(query.dtype)


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
    # Each query's place in its own integer dtype (an index's int32 by default), its edges moved there and
    # clipped to the keys before they narrow to the spans' int32.
    place = (jnp.broadcast_to(jnp.arange(start.shape[1], dtype=jnp.int32)[None, :], start.shape[:2])
             if places is None else jnp.asarray(places))[..., None]
    if not jnp.issubdtype(place.dtype, jnp.integer):
        raise ValueError(f"a query's place is an integer, not {place.dtype}")
    # In a dtype that holds the keys' exclusive end too, the window's rule (`widened_places`), so an edge
    # moved past a narrow place's range still reaches the keys (an int8 place of 127 over 129 keys); the
    # end of the place's own signedness (an unsigned place's beside an unsigned end, no x64 needed).
    bound = jnp.asarray(count, jnp.uint32 if jnp.issubdtype(place.dtype, jnp.unsignedinteger) else jnp.int32)
    place, _ = widened_places((place, bound))

    def within(lo, hi):
        # Each run's keys within `[lo, hi)` of the query's place, empty where none.
        def held(edge):
            return jnp.clip(jnp.asarray(edge), 0, count).astype(start.dtype)
        return jnp.maximum(start, held(lo)), jnp.minimum(end, held(hi))

    # Each edge a place moved by a band, held at int32's ends rather than wrapping (`moved_places`), then
    # clipped to the keys: a band however wide, about a place however far, selects the keys it holds.
    copies = []
    for p in range(len(bands) + 1):
        if p == 0:
            sides = [within(moved_places(place, -bands[0]), moved_places(place, bands[0] + 1))]
        else:
            # Past band p - 1 and within band p: the keys before the query, then those after it; the last
            # band's outer edges the keys' own, whatever the query's place.
            before = moved_places(place, -bands[p]) if p < len(bands) else 0
            after = moved_places(place, bands[p] + 1) if p < len(bands) else count
            sides = [within(before, moved_places(place, -bands[p - 1])),
                     within(moved_places(place, bands[p - 1] + 1), after)]
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
    return KeySpans(lo, hi, spans.query_ids, key_ids, products * count, "spans", excluded=spans.excluded)


def _copied_hooks(hooks: Hooks, products: int, count: int, *, wide: jnp.dtype | None = None) -> Hooks:
    """`hooks` over the keys' `products` copies (`selected`): its score hook given each key's own index, a
    copy's taken back to its key's (`k % count`), in every mainloop and derivative alike; a KEY or TILE
    parameter once a copy, a dropping call's draw at each copy's keys' own places. Read off the hook's
    program at one tile, a shape-dependent hook reading its keys' indices at another passed unseen and was
    read at its copy's index (train 1362's F1)."""
    for side in hooks.score.sides:
        if kind_of(side) is Kind.BAND:
            raise ValueError("a band's parameters hold each query's keys about it, which a selected call's "
                             "copies of the keys do not")
    original = hooks.score.fn

    # A side along the keys once a copy, at the dtype the kernels compute it in (`kept_value`), so its
    # copies' gradients sum before they round to its own (a 16-bit bias's near and far terms each past its
    # largest, summing to 0); a wide call's at the call's accumulation, `wide` (an fp64 bias of
    # `-(2^24 + 1)` kept, an fp32 one's copies' cotangents past fp32 summed before they round, train
    # 1472's review).
    def kept(value):
        if wide is not None and jnp.issubdtype(value.dtype, jnp.floating):
            return value.astype(jnp.promote_types(value.dtype, wide))
        return kept_value(value)

    sides = tuple(Side(jnp.concatenate([kept(side.value)] * products, axis=side.axes.index("keys")),
                       side.axes)
                  if "keys" in side.axes else side for side in hooks.score.sides)
    drop = hooks.drop
    if drop is not None:
        keys = jnp.arange(count, dtype=jnp.int32) if drop.keys is None else drop.keys
        drop = dataclasses.replace(drop, keys=jnp.concatenate([keys] * products))
    score = None if original is None else functools.partial(_own_keys, original, count)
    return dataclasses.replace(hooks, score=Hook(score, sides), drop=drop)


def _own_keys(score: Callable, count: int, s, q_pos, k_pos, *params):
    """`score` given each key's own index of `count`, a copy's taken back (`_copied_hooks`): one function a
    hook and count, so a holdable hook's lowering stays held (`tuning.holdable`)."""
    return score(s, q_pos, k_pos % count, *params)
