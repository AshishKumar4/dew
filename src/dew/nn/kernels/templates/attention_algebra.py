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
        """Whether the logits stay in base 2: no hook reads them."""
        return self.score_mod is None

    @property
    def guarded(self) -> bool:
        """Whether a query can see no key, so its maxima and denominators need the empty row's guard."""
        return self.form == "spans" or self.score_mod is not None

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
    """A query block's softmax state, unhooked: the weighted values `acc` `[bq, Dv]`, the running
    maximum `m` `[bq]` (in base 2 where the step's logits are) and the denominator `total` `[bq]`, fp32."""

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
        seen = (coords.k_pos >= coords.start[:, None]) & (coords.k_pos < coords.end[:, None])
    elif spec.form == "causal":
        seen = coords.k_pos <= coords.q_pos
    if spec.form != "spans" and spec.keys % s.shape[1]:
        # A tile past the call's last key reads padding.
        seen = coords.k_pos < spec.keys if seen is None else seen & (coords.k_pos < spec.keys)
    if spec.ids:
        same = coords.query_ids[:, None] == coords.key_ids[None, :]
        seen = same if seen is None else seen & same
    return logits if seen is None else jnp.where(seen, logits, -jnp.inf)


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
    return spec.power(s - _safe(spec, m)[:, None])


def narrow(p: jax.Array, dtype) -> jax.Array:
    """The weights at the values' dtype, the operand of `P·V`: the reference rounds them so too."""
    return p.astype(dtype)


def total(denominator: jax.Array, alpha: jax.Array, p: jax.Array) -> jax.Array:
    """The denominator after a tile."""
    return alpha * denominator + p.sum(axis=-1)


def rescale(acc: jax.Array, alpha: jax.Array) -> jax.Array:
    """The weighted values before a tile, taken to the new maximum; `P·V` adds the tile's to them."""
    return alpha[:, None] * acc


def merge(spec: Forward, first: Partial, second: Partial) -> Partial:
    """Two states over disjoint keys as one, `first`'s keys before `second`'s: each taken to the larger
    maximum and summed, in that order."""
    m = jnp.maximum(first.m, second.m)
    safe = _safe(spec, m)
    a, b = spec.power(first.m - safe), spec.power(second.m - safe)
    return Partial(a[:, None] * first.acc + b[:, None] * second.acc, m, a * first.total + b * second.total)


def finish(spec: Forward, state: Partial, *, initial: bool, output_mod: Callable | None = None,
           output_params: tuple[jax.Array, ...] = ()) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The block's output `[bq, Dv]` fp32 (its weighted values over its denominator, zero for a query
    that saw nothing and started from no state, then the output hook) and its residuals: the natural
    maximum (0 for such a query, as the reference's) and the denominator."""
    m = state.m / LOG2E if spec.base2 else state.m
    denominator = state.total
    if spec.guarded or initial:
        out = jnp.where(denominator[:, None] > 0,
                        state.acc / jnp.where(denominator > 0, denominator, 1.0)[:, None], 0.0)
        m = jnp.where(m == -jnp.inf, 0.0, m)
    else:
        out = state.acc / denominator[:, None]
    if output_mod is not None:
        out = output_mod(out, *output_params)
    return out, m, denominator
