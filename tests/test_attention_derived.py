"""Attention's forward derived from its declaration (theme A, `attention_algebra.declaration`,
`dew.algebra.derive`) beside the template's own forms (`attention_algebra.plain_step`, `merge`,
`finish`), the oracle: both walk the same template-formed tiles (scaled logits, a mask, values), a split
walk merged, and are held on par against the float64 truth (check 1). The derived forms replace the oracle
only on par with it and no slower in the kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew.algebra import derive
from dew.nn.kernels.templates import attention as template, attention_algebra as algebra, attention_portable
from dew.nn.precision import float32_einsum

ROWS, KEYS, WIDTH, TILE = 16, 96, 32, 32
QUERY_PLACES = jnp.arange(ROWS) * (KEYS // ROWS)
KEY_PLACES = jnp.arange(KEYS)
MASKS = {
    "full": lambda: jnp.ones((ROWS, KEYS), bool),
    "causal": lambda: KEY_PLACES[None, :] <= QUERY_PLACES[:, None],
    # The last tile half padding, and a query past every key it may see: a row of no key.
    "padded": lambda: (KEY_PLACES[None, :] < KEYS - TILE // 2) & (QUERY_PLACES[:, None] < KEYS - 8),
}


def draw(seed):
    """Logits spanning six decades across rows, so a row's maximum moves from tile to tile, and values."""
    rng = np.random.default_rng(seed)
    scale = 10.0 ** rng.uniform(-1, 2, (ROWS, 1))
    return (jnp.asarray(rng.standard_normal((ROWS, KEYS)) * scale, jnp.float32),
            jnp.asarray(rng.standard_normal((KEYS, WIDTH)), jnp.float32))


def tiles(values: dict, tile: int) -> dict:
    """The operands' keys of key tile `tile`."""
    keys = slice(tile * TILE, (tile + 1) * TILE)
    return {"logits": values["logits"][:, keys], "seen": values["seen"][:, keys],
            "values": values["values"][keys]}


def derived_walk(base2: bool, mask: str, split: bool):
    """The declaration's derived forms over the key tiles in order, or in two halves merged."""
    forms = derive.derive(algebra.declaration(base2=base2))
    count = KEYS // TILE

    def walk(values, steps):
        first = tiles(values, 0)
        state = forms.init({name: (value.shape, value.dtype) for name, value in first.items()})
        for tile in steps:
            state = forms.step(state, tiles(values, tile))
        return state

    def run(logits, value):
        values = {"logits": logits, "seen": MASKS[mask](), "values": value}
        if split:
            state = forms.merge(walk(values, range(count // 2)), walk(values, range(count // 2, count)))
        else:
            state = walk(values, range(count))
        return forms.finish(state, values)

    return run


def oracle_walk(base2: bool, mask: str, split: bool):
    """The template's own forms over the same tiles (`plain_step`, `merge`, `finish`), as a call whose spans
    narrow (`form="spans"`), so a query may see no key (`Forward.guarded`), its logits as the declaration's,
    unit-scaled in their base."""
    spec = algebra.Forward(form="spans", keys=KEYS, ids=False, scale=1.0, natural=not base2)
    count = KEYS // TILE

    def dot(a, b):
        return jnp.dot(a, b, preferred_element_type=jnp.float32)

    def run(logits, value):
        seen = MASKS[mask]()

        def keys_at(tile):
            keys = jax.lax.dynamic_slice_in_dim(KEY_PLACES, tile * TILE, TILE)
            coords = algebra.Coords(QUERY_PLACES[:, None], keys[None, :])
            masked = jnp.where(jax.lax.dynamic_slice_in_dim(seen, tile * TILE, TILE, axis=1),
                               jax.lax.dynamic_slice_in_dim(logits, tile * TILE, TILE, axis=1), -jnp.inf)
            tile_values = jax.lax.dynamic_slice_in_dim(value, tile * TILE, TILE)
            return algebra.Keys(coords, masked, lambda: tile_values, dot)

        def walk(lo, hi):
            return algebra.plain_walk(spec, lo, hi, keys_at, algebra.started(spec, None, ROWS, WIDTH, None))

        state = algebra.merge(spec, walk(0, count // 2), walk(count // 2, count)) if split else walk(0, count)
        return algebra.finish(spec, state, initial=False)[0]

    return run


def attended(logits: jax.Array, seen: jax.Array, values: jax.Array) -> jax.Array:
    """Each query's softmax over the keys it sees of its natural `logits` (XLA's, `jax.nn.softmax`), its
    weights' product with the `values`; a query that sees no key, zeros."""
    return float32_einsum("qk,kd->qd", jax.nn.softmax(logits, axis=-1, where=seen), values)


@pytest.mark.parametrize("split", [False, True], ids=["walked", "merged"])
@pytest.mark.parametrize("mask", list(MASKS))
@pytest.mark.parametrize("base2", [True, False], ids=["fold", "natural"])
def test_the_derived_forward_is_on_par_with_the_templates(base2, mask, split):
    """The derived walk (or a split walk merged) on par with the template's forms on the same tiles against
    the float64 truth (check 1), XLA's softmax of the logits in their base, a row of no key included."""
    natural = np.log(2.0) if base2 else 1.0
    assert_as_exact(jax.jit(derived_walk(base2, mask, split)), jax.jit(oracle_walk(base2, mask, split)), draw,
                    f"derived attention {mask} {'merged' if split else 'walked'}", seeds=16,
                    truth_fn=lambda logits, value: attended(logits * natural, MASKS[mask](), value))


LENGTH, DEPTH, BLOCK = 64, 32, 16
SCALE = DEPTH ** -0.5
SPANS = {"full": {}, "causal": {"causal": True},
         # Its first queries before the keys' start: rows of no key.
         "started": {"causal": True, "starts": jnp.array([LENGTH // 8])}}
TILES = {"block_q": BLOCK, "block_k": BLOCK, "pad": BLOCK, "scale": SCALE}


def backward_draw(seed):
    """Queries, keys and values of one head, unit-scaled rows, and the output's cotangents."""
    rng = np.random.default_rng(seed)
    q, k, v, d = (jnp.asarray(rng.standard_normal((LENGTH, DEPTH)), jnp.float32) for _ in range(4))
    return q / jnp.max(jnp.abs(q), axis=-1, keepdims=True), k, v, d


def seen_of(form: str) -> jax.Array:
    places = jnp.arange(LENGTH)
    causal = SPANS[form].get("causal")
    seen = places[None, :] <= places[:, None] if causal else jnp.ones((LENGTH, LENGTH), bool)
    return seen & (places[None, :] >= SPANS[form].get("starts", [0])[0])


def xla_backward(form: str):
    """XLA's gradients of the forward on whole operands (`attended`, by autodiff): the truth's arithmetic,
    and in float32 the reference a served template form is held to (check 1's)."""
    seen = seen_of(form)

    def run(q, k, v, d):
        return jax.vjp(lambda q, k, v: attended(float32_einsum("qd,kd->qk", q, k) * SCALE, seen, v),
                       q, k, v)[1](d)

    return run


def forward_rows(form: str):
    """The template's forward on this mainloop at the backward's tiles, its residual rows: each query's
    maximum and denominator, natural, and rowsum(dO O), which both backwards read, so they differ in their
    own forms alone."""
    spans = template.spans(LENGTH, LENGTH, 1, **SPANS[form])

    def run(q, k, v, d):
        q4, k4, v4 = (x[None, :, None] for x in (q, k, v))
        out, m, total, anchor, split = attention_portable.attend(q4, k4, v4, spans, residuals=True, **TILES)
        delta = jnp.sum(d * out[0, :, 0].astype(jnp.float32), axis=-1)
        return (m, total, anchor, split), m[0, 0], total[0, 0], delta

    return run


def derived_backward(form: str, walked: str):
    """The backward declaration's derived forms walked over key tiles (queries first) or query tiles (keys
    first), from the template forward's residual rows (`forward_rows`)."""
    forms = derive.derive(algebra.backward_declaration(scale=SCALE, walked=walked))
    seen = seen_of(form)
    rows_of = forward_rows(form)

    def tile(values: dict, index: int) -> dict:
        part = slice(index * BLOCK, (index + 1) * BLOCK)
        along = ("q", "dout", "peak", "total", "delta", "seen") if walked == "rows" else ("k", "values")

        def part_of(name, value):
            if walked == "keys" and name == "seen":
                return value[:, part]
            return value[part] if name in along else value

        return {name: part_of(name, value) for name, value in values.items()}

    def run(q, k, v, d):
        _, peak, total, delta = rows_of(q, k, v, d)
        values = {"q": q, "k": k, "values": v, "dout": d, "peak": peak, "total": total, "delta": delta,
                  "seen": seen}
        first = tile(values, 0)
        state = forms.init({name: (value.shape, value.dtype) for name, value in first.items()})
        mapped = []
        for index in range(LENGTH // BLOCK):
            state = forms.step(state, tile(values, index))
            mapped.append(forms.mapped(tile(values, index)))
        dq, dk, dv = forms.finish(state, values)
        if walked == "rows":
            return jnp.concatenate([parts[0] for parts in mapped]), dk, dv
        return (dq, jnp.concatenate([parts[1] for parts in mapped]),
                jnp.concatenate([parts[2] for parts in mapped]))

    return run


def oracle_backward(form: str):
    """The template's own fast backward on this mainloop (`attention_portable`) at the same tiles, from the
    same forward's residuals, D from its output as the declaration's (`split=False`, the route's output
    residuals): the accurate form's D from the split sums is a form of its own class, past this
    declaration's."""
    spans = template.spans(LENGTH, LENGTH, 1, **SPANS[form])
    rows_of = forward_rows(form)

    def run(q, k, v, d):
        residuals, *_ = rows_of(q, k, v, d)
        q4, k4, v4, d4 = (x[None, :, None] for x in (q, k, v, d))
        grads = attention_portable.backward(q4, k4, v4, spans, None, d4, residuals=residuals, split=False,
                                            accurate=False, **TILES)
        return grads.query[0, :, 0], grads.key[0, :, 0], grads.value[0, :, 0]

    return run


SERVED_BY_TEMPLATE = {("causal", "keys")}
"""The backward cells the template's own form serves, the derived form not on par there until the engine's
error-aware extraction prices the choice that passes (docs/design/algebra.md §1.5, the generator's first
acceptance case; a causal call walked queries first, whose derived dV erred 1.62 times the template's)."""


@pytest.mark.parametrize("walked", ["keys", "rows"], ids=["queries-first", "keys-first"])
@pytest.mark.parametrize("form", list(SPANS))
def test_the_backward_serving_each_cell_is_on_par(form, walked):
    """The backward form serving each cell on par against the float64 gradients (check 1): the backward
    declaration's derived forms, walked queries first or keys first, held to the template's own backward at
    the same tiles; the template's form where it serves (`SERVED_BY_TEMPLATE`), held to XLA's."""
    if (form, walked) in SERVED_BY_TEMPLATE:
        served, reference = oracle_backward(form), xla_backward(form)
    else:
        served, reference = derived_backward(form, walked), oracle_backward(form)
    assert_as_exact(jax.jit(served), jax.jit(reference), backward_draw,
                    f"attention backward {form} {walked}", seeds=16, truth_fn=xla_backward(form))
