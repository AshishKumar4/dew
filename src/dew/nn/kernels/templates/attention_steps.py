"""The attention template's algorithm on tiles, every backend's (`templates/attention_triton.py`'s
schedule, and every schedule after it): a query block's online softmax over the key tiles a schedule
feeds it, the span mask, the score and output hooks and the residuals. Pure jnp on one tile at a time;
a schedule supplies only which tiles, in what order, and the product `dot` its hardware takes, and
holds no masking, hook or softmax arithmetic of its own (tests/test_attention_triton.py runs these
steps under the Triton schedule and under a plain loop and asks for the same bits).

A step specializes at trace time to what the call holds (`Forward`): a full call's tiles take no
mask, a causal call's the diagonal's index mask, and only spans read intervals and ids; with no score
hook the logits stay in base 2 (log2(e) folded into the scale) for exp2; and only a call where a query
can see nothing (spans, a score hook) guards its maxima and denominators."""

import dataclasses
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

LOG2E = 1.4426950408889634
"""log2(e), which turns a natural exponent into a base-2 one."""

Dot = Callable[[jax.Array, jax.Array], jax.Array]
"""A schedule's product of two tiles, accumulated in fp32."""


@dataclasses.dataclass(frozen=True)
class Forward:
    """What a forward step specializes on, static: the spans' `form` ('full', 'causal', 'spans',
    `KeySpans.form`), the call's key count `keys`, whether it packs documents (`ids`), the softmax
    scale and the score hook (`score_mod(s, q_pos, k_pos, *params)` on natural logits)."""

    form: str
    keys: int
    ids: bool
    scale: float
    score_mod: Callable | None = None

    @property
    def base2(self) -> bool:
        """Whether the logits stay in base 2: no hook reads them."""
        return self.score_mod is None

    @property
    def guarded(self) -> bool:
        """Whether a query can see no key, so its maxima and denominators need the guard."""
        return self.form == "spans" or self.score_mod is not None


class Tile(NamedTuple):
    """One key tile as a schedule hands it to a query block: the block's queries `[bq, D]`, the
    tile's keys `[bk, D]` and values `[bk, Dv]`, their indices `q_pos` `[bq, 1]` and `k_pos` `[1, bk]`,
    the score hook's parameters laid out for the tile, and where the spans narrow, each query's
    interval `start`, `end` `[bq]` and, with documents, the queries' and keys' ids."""

    query: jax.Array
    key: jax.Array
    value: jax.Array
    q_pos: jax.Array
    k_pos: jax.Array
    params: tuple[jax.Array, ...] = ()
    start: jax.Array | None = None
    end: jax.Array | None = None
    query_ids: jax.Array | None = None
    key_ids: jax.Array | None = None


class State(NamedTuple):
    """A query block's online softmax: the weighted values `acc` `[bq, Dv]`, the running maximum
    `peak` `[bq]` (in base 2 where the step's logits are) and the denominator `total` `[bq]`, fp32."""

    acc: jax.Array
    peak: jax.Array
    total: jax.Array


def forward_start(spec: Forward, block_q: int, width: int, initial=None) -> State:
    """A query block's softmax before any key: empty, or the initial state `(m, total, acc)` of
    `Hooks.initial` for the block, its maximum taken to base 2 where the step's logits are."""
    if initial is None:
        return State(jnp.zeros((block_q, width), jnp.float32), jnp.full((block_q,), -jnp.inf, jnp.float32),
                     jnp.zeros((block_q,), jnp.float32))
    m, total, acc = (array.astype(jnp.float32) for array in initial)
    return State(acc, m * LOG2E if spec.base2 else m, total)


def forward_tiles(spec: Forward, block_q: int, block_k: int, first_query, bounds=None):
    """The key tiles `[lo, hi)` a query block from `first_query` visits: every tile of the call's keys,
    a causal block's up to its last query's, or the spans' block map's `(lo, hi)` (`bounds`)."""
    tiles = -(-spec.keys // block_k)
    if spec.form == "spans":
        return bounds[0], bounds[1]
    if spec.form == "causal":
        return 0, jnp.minimum(-(-(first_query + block_q) // block_k), tiles)
    return 0, tiles


def forward_step(spec: Forward, state: State, tile: Tile, dot: Dot) -> State:
    """The softmax over one more key tile: its logits (the scaled products, the score hook), each key
    the query does not see at -inf, then the online update of the maximum, the denominator and the
    weighted values."""
    logits = dot(tile.query, tile.key.T) * (spec.scale * LOG2E if spec.base2 else spec.scale)
    if spec.score_mod is not None:
        logits = spec.score_mod(logits, tile.q_pos, tile.k_pos, *tile.params)
    seen = None
    if spec.form == "spans":
        seen = (tile.k_pos >= tile.start[:, None]) & (tile.k_pos < tile.end[:, None])
    elif spec.form == "causal":
        seen = tile.k_pos <= tile.q_pos
    if spec.form != "spans" and spec.keys % tile.key.shape[0]:
        # A tile past the call's last key reads padding.
        seen = tile.k_pos < spec.keys if seen is None else seen & (tile.k_pos < spec.keys)
    if spec.ids:
        same = tile.query_ids[:, None] == tile.key_ids[None, :]
        seen = same if seen is None else seen & same
    if seen is not None:
        logits = jnp.where(seen, logits, -jnp.inf)
    peak = jnp.maximum(state.peak, logits.max(axis=-1))
    safe = jnp.where(peak == -jnp.inf, 0.0, peak) if spec.guarded else peak
    power = jnp.exp2 if spec.base2 else jnp.exp
    correction = power(state.peak - safe)
    weights = power(logits - safe[:, None])
    acc = correction[:, None] * state.acc + dot(weights.astype(tile.value.dtype), tile.value)
    return State(acc, peak, correction * state.total + weights.sum(axis=-1))


def forward_finish(spec: Forward, state: State, initial: bool, output_mod=None,
                   output_params: tuple[jax.Array, ...] = ()) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The block's output `[bq, Dv]` fp32 (its weighted values over its denominator, zero for a query
    that saw nothing and started from no state, then the output hook) and its residuals, the natural
    maximum (0 for such a query) and the denominator."""
    peak = state.peak / LOG2E if spec.base2 else state.peak
    total = state.total
    if spec.guarded or initial:
        out = jnp.where(total[:, None] > 0, state.acc / jnp.where(total > 0, total, 1.0)[:, None], 0.0)
        peak = jnp.where(peak == -jnp.inf, 0.0, peak)
    else:
        out = state.acc / total[:, None]
    if output_mod is not None:
        out = output_mod(out, *output_params)
    return out, peak, total
