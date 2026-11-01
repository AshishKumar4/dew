"""Template forms derived from a declaration (`dew.algebra.derive`, docs/design/algebra.md §6.9): attention's
softmax over keys stated on whole operands, walked in key tiles through its derived step, the walk split in
two and merged, each held to the float64 truth (check 1) against the whole-operand function; and the
declarations no monoid along the walk holds, refused."""

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew.algebra import derive
from dew.nn import precision

QUERIES, KEYS, WIDTH, TILE = 16, 96, 32, 32


def attention(q, k, v, rows, keys):
    """Softmax over keys of the scaled logits, then the values: the max-shifted form a model writes."""
    del rows, keys
    s = (q @ k.T) * WIDTH ** -0.5
    m = jnp.max(s, axis=-1)
    p = jnp.exp(s - m[:, None])
    return (p @ v) / jnp.sum(p, axis=-1)[:, None]


def causal(q, k, v, rows, keys):
    """The same over each query's keys up to its own place (`rows`, `keys`: the queries' and keys'
    places, operands along their axes, so a tile's keys carry their own), the rest masked out by a
    selection."""
    s = (q @ k.T) * WIDTH ** -0.5
    seen = keys[None, :] <= rows[:, None]
    s = jnp.where(seen, s, -jnp.inf)
    m = jnp.max(s, axis=-1)
    p = jnp.where(seen, jnp.exp(s - m[:, None]), 0.0)
    return (p @ v) / jnp.sum(p, axis=-1)[:, None]


def base2(q, k, v, rows, keys):
    """The base-2 form, `log2(e)` folded into the logits' scale."""
    del rows, keys
    s = (q @ k.T) * (WIDTH ** -0.5 * float(np.log2(np.e)))
    m = jnp.max(s, axis=-1)
    p = jnp.exp2(s - m[:, None])
    return (p @ v) / jnp.sum(p, axis=-1)[:, None]


def shifted(q, k, v, rows, keys):
    """Causal attention whose first query sees no key, its maximum shifted safely (`where(m == -inf, 0, m)`)
    and its total guarded: a row of no key gives zeros."""
    s = (q @ k.T) * WIDTH ** -0.5
    seen = (keys[None, :] <= rows[:, None]) & (rows[:, None] > 0)
    s = jnp.where(seen, s, -jnp.inf)
    m = jnp.max(s, axis=-1)
    p = jnp.exp(s - jnp.where(jnp.isneginf(m), 0.0, m)[:, None])
    total = jnp.sum(p, axis=-1)
    return (p @ v) / jnp.where(total == 0, 1.0, total)[:, None]


def products(q, k, v, rows, keys):
    """Causal attention whose products are Dew's (`float32_einsum`, `float32_matmul`): under the suite's wide
    CPU arithmetic each a choice by the platform it lowers for, which the derivation reads as one product."""
    s = precision.float32_einsum("rw,kw->rk", q, k) * WIDTH ** -0.5
    seen = keys[None, :] <= rows[:, None]
    s = jnp.where(seen, s, -jnp.inf)
    m = jnp.max(s, axis=-1)
    p = jnp.where(seen, jnp.exp(s - m[:, None]), 0.0)
    return precision.float32_matmul(p, v) / jnp.sum(p, axis=-1)[:, None]


DECLARED: dict[str, tuple[Callable, Literal[2, "e"]]] = {
    "attention": (attention, "e"), "causal": (causal, "e"), "base2": (base2, 2), "shifted": (shifted, "e"),
    "products": (products, "e")}
AXES = {"q": ("rows", "width"), "k": ("keys", "width"), "v": ("keys", "values"), "rows": ("rows",),
        "keys": ("keys",)}
ROWS, PLACES = jnp.arange(QUERIES) * (KEYS // QUERIES), jnp.arange(KEYS)
"""The queries' places (each query a few keys on from the last) and the keys'."""


def whole(function):
    """`function` on (q, k, v), its places the whole sequence's."""
    return lambda q, k, v: function(q, k, v, ROWS, PLACES)


def draw(seed):
    """Queries, keys and values whose logits span six decades of magnitude across rows, so a row's maximum
    moves from tile to tile and the rescale matters."""
    rng = np.random.default_rng(seed)
    scale = 10.0 ** rng.uniform(-1, 2, (QUERIES, 1))
    return (jnp.asarray(rng.standard_normal((QUERIES, WIDTH)) * scale, jnp.float32),
            jnp.asarray(rng.standard_normal((KEYS, WIDTH)), jnp.float32),
            jnp.asarray(rng.standard_normal((KEYS, WIDTH)), jnp.float32))


def declared(function, exponent: Literal[2, "e"] = "e") -> derive.Reduction:
    return derive.Reduction(function, AXES, walked="keys", exponent=exponent)


def operands(q, k, v) -> dict:
    return {"q": q, "k": k, "v": v, "rows": ROWS, "keys": PLACES}


def tile_of(values: dict, tile: int) -> dict:
    """Each operand's keys of key tile `tile` (the whole of an operand with no keys axis)."""
    keys = slice(tile * TILE, (tile + 1) * TILE)
    return {name: value[keys] if AXES[name][0] == "keys" else value for name, value in values.items()}


def walked(forms: derive.Derived, values: dict, tiles: range):
    """The state after walking `tiles` of the keys in order, from the empty state."""
    first = tile_of(values, 0)
    state = forms.init({name: (value.shape, value.dtype) for name, value in first.items()})
    for tile in tiles:
        state = forms.step(state, tile_of(values, tile))
    return state


@pytest.mark.parametrize("name", list(DECLARED))
def test_the_derived_walk_computes_the_declared_function(name):
    """The declaration's derived step over every key tile in order, finished: on par with the whole-operand
    function against the float64 truth (check 1)."""
    forms = derive.derive(declared(*DECLARED[name]))

    def derived(q, k, v):
        values = operands(q, k, v)
        return forms.finish(walked(forms, values, range(KEYS // TILE)), values)

    assert_as_exact(jax.jit(derived), jax.jit(whole(DECLARED[name][0])), draw, f"{name} walked", seeds=16)


@pytest.mark.parametrize("name", list(DECLARED))
def test_two_walks_merged_compute_the_declared_function(name):
    """The keys walked in two disjoint halves, each from the empty state, their states merged (a split walk's
    combine, the work scheduler's fix-up), finished: on par with the whole-operand function."""
    forms = derive.derive(declared(*DECLARED[name]))
    tiles = KEYS // TILE

    def split(q, k, v):
        values = operands(q, k, v)
        first = walked(forms, values, range(tiles // 2))
        second = walked(forms, values, range(tiles // 2, tiles))
        return forms.finish(forms.merge(first, second), values)

    assert_as_exact(jax.jit(split), jax.jit(whole(DECLARED[name][0])), draw, f"{name} merged", seeds=16)


def unshifted(q, k, v, rows, keys):
    """Softmax with no maximum: no walked reduction scales the sums, each an independent sum."""
    del rows, keys
    p = jnp.exp((q @ k.T) * WIDTH ** -0.5)
    return (p @ v) / jnp.sum(p, axis=-1)[:, None]


def test_sums_no_maximum_scales_walk_as_plain_sums():
    """A declaration whose walked sums read no walked maximum derives plain sums: on par with the function."""
    forms = derive.derive(declared(unshifted))

    def derived(q, k, v):
        values = operands(q, k, v)
        return forms.finish(walked(forms, values, range(KEYS // TILE)), values)

    def small(seed):
        q, k, v = draw(seed)
        return q / jnp.max(jnp.abs(q)), k, v

    assert_as_exact(jax.jit(derived), jax.jit(whole(unshifted)), small, "unshifted walked", seeds=16)


def squared_shift(q, k, v, rows, keys):
    del rows, keys
    s = q @ k.T
    m = jnp.max(s, axis=-1)
    return jnp.sum((s - m[:, None]) ** 2, axis=-1)


def weights(q, k, v, rows, keys):
    del rows, keys
    s = q @ k.T
    return jnp.exp(s - jnp.max(s, axis=-1)[:, None])


@pytest.mark.parametrize("function", [squared_shift, weights])
def test_a_declaration_no_monoid_holds_is_refused(function):
    """A walked sum that reads the walked maximum other than through its exponent, and an output that varies
    along the walked axis, are refused before any form is built."""
    forms = derive.derive(declared(function))
    first = tile_of(operands(*draw(0)), 0)
    with pytest.raises(derive.Refused):
        forms.init({name: (value.shape, value.dtype) for name, value in first.items()})


def backward(q, k, v, do, lse, delta):
    """Attention's gradients from its residuals (each query's log-sum-exp and its `delta`, rowsum(dO * O)):
    the weights recomputed, dV and dK summed over the queries, dQ each query's own."""
    s = (q @ k.T) * WIDTH ** -0.5
    p = jnp.exp(s - lse[:, None])
    dv = p.T @ do
    ds = p * (do @ v.T - delta[:, None])
    return ds @ k * WIDTH ** -0.5, ds.T @ q * WIDTH ** -0.5, dv


def gradients(q, k, v):
    """The arguments `backward` takes for a cotangent of ones' scale: (q, k, v, dO, lse, delta), the
    residuals from the forward on whole operands."""
    s = (q @ k.T) * WIDTH ** -0.5
    lse = jax.nn.logsumexp(s, axis=-1)
    out = jnp.exp(s - lse[:, None]) @ v
    do = jnp.cos(out)
    return q, k, v, do, lse, jnp.sum(do * out, axis=-1)


QUERY_TILE = 4


def test_a_backward_walked_over_queries_sums_dk_dv_and_maps_dq():
    """Attention's backward walked over query tiles (keys first): dK and dV the walk's sums, finished; dQ
    each tile's own (`mapped`), the tiles' joined: on par with the whole-operand gradients."""
    axes = {"q": ("rows", "width"), "k": ("keys", "width"), "v": ("keys", "values"), "do": ("rows", "values"),
            "lse": ("rows",), "delta": ("rows",)}
    forms = derive.derive(derive.Reduction(backward, axes, walked="rows", exponent="e"))

    def tile(values: dict, index: int) -> dict:
        rows = slice(index * QUERY_TILE, (index + 1) * QUERY_TILE)
        return {name: value if name in ("k", "v") else value[rows] for name, value in values.items()}

    def derived(q, k, v):
        values = dict(zip(("q", "k", "v", "do", "lse", "delta"), gradients(q, k, v), strict=True))
        first = tile(values, 0)
        state = forms.init({name: (value.shape, value.dtype) for name, value in first.items()})
        dqs = []
        for index in range(QUERIES // QUERY_TILE):
            state = forms.step(state, tile(values, index))
            dqs.append(forms.mapped(tile(values, index))[0])
        _, dk, dv = forms.finish(state, values)
        return jnp.concatenate(dqs), dk, dv

    def small(seed):
        q, k, v = draw(seed)
        return q / jnp.max(jnp.abs(q), axis=-1, keepdims=True), k, v

    assert_as_exact(jax.jit(derived), jax.jit(lambda q, k, v: backward(*gradients(q, k, v))), small,
                    "backward over queries", seeds=16)
