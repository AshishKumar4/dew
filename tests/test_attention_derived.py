"""Attention's forward derived from its declaration (theme A, `attention_algebra.declaration`,
`dew.algebra.derive`) beside the template's own forms (`attention_algebra.plain_step`, `merge`,
`finish`), the oracle: both walk the same template-formed tiles (scaled logits, a mask, values), a split
walk merged, and are held on par against the float64 truth (check 1). The derived forms replace the oracle
only on par with it and no slower in the kernel."""

import contextlib
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import COMPILES_ON_CPU
from jax.sharding import Mesh
from reference_error import assert_as_exact
from test_algebra_derive import shaped, walked

from dew.algebra import SATURATION, derive
from dew.nn.kernels import decode_attention, generation
from dew.nn.kernels.templates import (
    attention as template,
    attention_algebra as algebra,
    attention_backward_algebra,
    attention_portable,
    attention_program,
)
from dew.nn.kernels.templates.hooks import Hook, Side
from dew.nn.precision import float32_einsum

SOURCE = Path(__file__).resolve().parents[1] / "src"
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
    return {"logits": values["logits"][:, keys], "values": values["values"][keys]}


def derived_walk(base2: bool, mask: str, split: bool):
    """The declaration's derived forms over the key tiles in order, or in two halves merged."""
    forms = derive.derive(algebra.declaration(base2=base2), accurate=False, saturation=SATURATION)
    count = KEYS // TILE

    def walk(values, steps):
        return walked(forms, [tiles(values, tile) for tile in steps])

    def run(logits, value):
        values = {"logits": jnp.where(MASKS[mask](), logits, -jnp.inf), "values": value}
        shapes = shaped(tiles(values, 0))
        if split:
            state = forms.merge(walk(values, range(count // 2)), walk(values, range(count // 2, count)),
                                values, shapes=shapes)
        else:
            state = walk(values, range(count))
        return forms.finish(state, values, shapes=shapes)

    return run


def attended(logits: jax.Array, seen: jax.Array, values: jax.Array) -> jax.Array:
    """Each query's softmax over the keys it sees of its natural `logits` (XLA's, `jax.nn.softmax`), its
    weights' product with the `values`; a query that sees no key, zeros."""
    return float32_einsum("qk,kd->qd", jax.nn.softmax(logits, axis=-1, where=seen), values)


@pytest.mark.parametrize("split", [False, True], ids=["walked", "merged"])
@pytest.mark.parametrize("mask", list(MASKS))
@pytest.mark.parametrize("base2", [True, False], ids=["fold", "natural"])
def test_the_derived_forward_is_on_par_with_xlas(base2, mask, split):
    """The derived walk (or a split walk merged) on par with XLA's softmax of the logits in their base
    against the float64 truth (check 1), a row of no key included."""
    natural = np.log(2.0) if base2 else 1.0

    def reference(logits, value):
        return attended(logits * natural, MASKS[mask](), value)

    assert_as_exact(jax.jit(derived_walk(base2, mask, split)), jax.jit(reference), draw,
                    f"derived attention {mask} {'merged' if split else 'walked'}", seeds=16)


def sliced_walk(mask: str, merged: bool):
    """The template's forms (a folding call's) over each key tile as a slice of its own, the
    slices' states stacked and merged at once (`algebra.merged`, a split decode's combine) or folded a merge
    at a time (`algebra.merge`), finished."""
    spec = algebra.Forward(form="spans", keys=KEYS, ids=False, scale=0.5, natural=False)

    def run(logits, value):
        seen = MASKS[mask]()

        def keys_at(tile):
            keys = jax.lax.dynamic_slice_in_dim(KEY_PLACES, tile * TILE, TILE)
            masked = jnp.where(jax.lax.dynamic_slice_in_dim(seen, tile * TILE, TILE, axis=1),
                               jax.lax.dynamic_slice_in_dim(logits, tile * TILE, TILE, axis=1), -jnp.inf)
            tile_values = jax.lax.dynamic_slice_in_dim(value, tile * TILE, TILE)
            return algebra.Keys(algebra.Coords(QUERY_PLACES[:, None], keys[None, :]), masked,
                                algebra.Operand(tile_values.shape, tile_values.dtype, lambda: tile_values),
                                lambda a, b: jnp.dot(a, b, preferred_element_type=jnp.float32))

        states = [attention_program.plain_walk(spec, tile, tile + 1, keys_at,
                                               algebra.started(spec, None, ROWS, WIDTH))
                  for tile in range(KEYS // TILE)]
        if merged:
            state = algebra.merged(spec, jax.tree.map(lambda *parts: jnp.stack(parts), *states), 0)
        else:
            state = states[0]
            for other in states[1:]:
                state = algebra.merge(spec, state, other)
        return algebra.finish(spec, state, initial=False)[0]

    return run


@pytest.mark.parametrize("mask", list(MASKS))
def test_slices_merged_at_once_are_on_par_with_their_fold(mask):
    """A split call's slices merged at once (`algebra.merged`, the derived forms' merges a level at a time),
    three of them so a level carries one, a slice of no key among them where the mask leaves one: on par with
    their fold a merge at a time against the float64 truth (check 1)."""
    natural = np.log(2.0)
    assert_as_exact(jax.jit(sliced_walk(mask, merged=True)), jax.jit(sliced_walk(mask, merged=False)),
                    draw, f"slices merged at once {mask}", seeds=16,
                    truth_fn=lambda logits, value: attended(logits * natural, MASKS[mask](), value))


LENGTH, DEPTH, BLOCK = 64, 32, 16
SCALE = DEPTH ** -0.5
SPANS = {"full": {}, "causal": {"causal": True},
         # Its first queries before the keys' start: rows of no key.
         "started": {"causal": True, "starts": jnp.array([LENGTH // 8])}}
TILES = attention_portable.Tiles(BLOCK, BLOCK, BLOCK)


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
    own forms alone: summed as the walk's dP is (`attention.interpreted_rows`), as the template's backward
    takes it (`attention_gradient.off_anchor_delta`), so a row of one key's dP and D cancel exactly."""
    spans = template.spans(LENGTH, LENGTH, 1, **SPANS[form])

    def run(q, k, v, d):
        q4, k4, v4 = (x[None, :, None] for x in (q, k, v))
        out, m, total, anchor, split = attention_portable.xla_mainloop(TILES, q4, k4, v4, spans,
                                                                       residuals=True, scale=SCALE)
        delta = template.interpreted_rows(d, out[0, :, 0].astype(jnp.float32))
        return (m, total, anchor, split), m[0, 0], total[0, 0], delta

    return run


def derived_backward(form: str, along: str, accurate: bool = False, how=algebra.PLAIN,
                     normalized_values: bool = False):
    """The backward declaration's derived forms walked `along` key tiles (queries first) or query tiles (keys
    first), from the template forward's residual rows (`forward_rows`): its logits the tile's, -inf where
    unseen, and D about no anchor (its whole `delta`), each step's products the template's CPU mainloop's
    issuer (`attention.interpreted_dot`), its outputs along the walk the step's own, scheduled as a mainloop
    tiles the walk: keys first, a key tile's dK and dV the walk's over the query tiles; queries first, each
    query tile walking the key tiles, its dQ the walk's and each key tile's dK and dV its share, summed over
    the query tiles in order (row 3310: one contraction over a key tile's every query at once, no
    mainloop's schedule, erred 1.63 times the template's in dV). The `accurate` form's under the accurate
    numerics, no key its query's anchor; its weights recomputed in `how`'s units (raw products scaled after
    their difference, their maximum the raw one, or an exact largest weight); dV's operands the normalized
    weights against dO where `normalized_values` (bf16's order, `softmax_gradients`)."""
    forms = derive.derive(attention_backward_algebra.backward_declaration(
        scale=SCALE, walked=along, how=how, accurate=accurate, normalized_values=normalized_values),
        accurate=accurate, saturation=SATURATION)
    seen = seen_of(form)
    rows_of = forward_rows(form)
    tiles = LENGTH // BLOCK
    by_row = ("q", "dout", "peak", "total", "anchor", "delta", "logits", "seen", "at_anchor")

    def tile(values: dict, index: int) -> dict:
        part = slice(index * BLOCK, (index + 1) * BLOCK)

        def part_of(name, value):
            if along == "rows":
                return value[part] if name in by_row else value
            if name in ("logits", "seen", "at_anchor"):
                return value[:, part]
            return value[part] if name in ("k", "values") else value

        return {name: part_of(name, value) for name, value in values.items()}

    # Each tile's outputs along the walk (dQ keys first, dK and dV queries first), the step's own, its
    # products the issuer's too, as a mainloop forms them (`derive.Derived.step`'s `outputs`).
    indices = (0,) if along == "rows" else (1, 2)

    def walk(values: dict):
        parts = [tile(values, index) for index in range(tiles)]
        state, mapped = forms.init(shaped(parts[0])), []
        for part in parts:
            state, outputs = forms.step(state, part, product=template.interpreted_dot, outputs=indices)
            mapped.append(dict(zip(indices, outputs, strict=True)))
        return forms.finish(state, values, shapes=shaped(parts[0])), mapped

    def run(q, k, v, d):
        _, peak, total, delta = rows_of(q, k, v, d)
        products = float32_einsum("qd,kd->qk", q, k)
        logits = jnp.where(seen, products if how.factor is not None else products * SCALE, -jnp.inf)
        if how.factor is not None:
            peak = jnp.max(logits, axis=-1)
        values = {"logits": logits, "q": q, "k": k, "values": v, "dout": d, "peak": peak, "total": total,
                  "anchor": jnp.zeros_like(delta), "delta": delta, "seen": seen,
                  **({"at_anchor": jnp.zeros_like(seen)} if accurate else {})}
        if along == "rows":
            (_, dk, dv, *_), mapped = walk(values)
            return jnp.concatenate([parts[0] for parts in mapped]), dk, dv
        dq, dk, dv = [], 0.0, 0.0
        for block in range(tiles):
            part = slice(block * BLOCK, (block + 1) * BLOCK)
            rows = {name: value[part] if name in by_row else value for name, value in values.items()}
            (block_dq, *_), mapped = walk(rows)
            dq.append(block_dq)
            dk = dk + jnp.concatenate([parts[1] for parts in mapped])
            dv = dv + jnp.concatenate([parts[2] for parts in mapped])
        return jnp.concatenate(dq), dk, dv

    return run


@pytest.mark.parametrize("normalized_values", [False, True], ids=["scaled-dout", "normalized-weights"])
@pytest.mark.parametrize("walked", ["keys", "rows"], ids=["queries-first", "keys-first"])
@pytest.mark.parametrize("form", list(SPANS))
def test_the_derived_backward_is_on_par_in_every_cell(form, walked, normalized_values):
    """The backward's derived forms on par against the float64 gradients (check 1) in every cell: the backward
    declaration's forms, walked queries first or keys first as a mainloop schedules them
    (`derived_backward`), dV's operands either order (`softmax_gradients`' `normalized_values`), held to
    XLA's gradients in float32 (`xla_backward`)."""
    assert_as_exact(jax.jit(derived_backward(form, walked, normalized_values=normalized_values)),
                    jax.jit(xla_backward(form)), backward_draw, f"attention backward {form} {walked}",
                    seeds=16)


@pytest.mark.parametrize("how", [algebra.PLAIN, algebra.Recompute(factor=SCALE),
                                 algebra.Recompute(exact=True)], ids=["scaled", "raw", "exact"])
@pytest.mark.parametrize("walked", ["keys", "rows"], ids=["queries-first", "keys-first"])
def test_the_accurate_backward_declaration_derives_on_par(walked, how):
    """The backward's accurate declaration (its maximum as given, the anchor's own term held to 0) derives
    under the accurate numerics, whose rewrites are exact alone, in each recomputation's units (scaled
    logits, raw products scaled after their difference, an exact largest weight), and walks on par with
    XLA's gradients in float32 against the float64 ones (check 1) where no key is its query's anchor, every
    query seeing a key."""
    served = derived_backward("causal", walked, accurate=True, how=how)
    assert_as_exact(jax.jit(served), jax.jit(xla_backward("causal")), backward_draw,
                    f"accurate attention backward {walked}", seeds=16)


def test_a_fast_tiles_cotangents_take_their_mainloops_hand_order():
    """The fast backward tile's derived cotangents (`derived_cotangents`) in each mainloop's own hand form's
    order, bit for bit (each hand form's arithmetic in JAX): Mosaic's from its weights, `(e / T) ((dP - c)
    - D)`; where a mainloop holds each query's anchor key (`anchored`, portable's and Triton's), its
    anchor's own dP - c exactly 0 and each cotangent scaled before the denominator, `e (((dP - c) - D) /
    T)`, so a weight below float32's least subnormal, its query resumed at a peak far above its logits,
    keeps its cotangent (the fp32 small-weight edge), where the weights' order rounds it to 0."""
    rng = np.random.default_rng(4228)
    logits = rng.standard_normal((16, 16)).astype(np.float32) * 3
    products = rng.standard_normal((16, 16)).astype(np.float32)
    peak = logits.max(axis=1)
    total = np.exp(logits - peak[:, None]).sum(axis=1).astype(np.float32)
    # Query 0 resumed at a peak of 87 over a denominator of 2^25, one key's dP near 2^102.
    logits[0], peak[0], total[0], products[0, 1] = 0.0, 87.0, 2.0 ** 25, 3.8e30
    at_anchor = np.arange(16)[None, :] == logits.argmax(axis=1)[:, None]
    anchor = np.where(at_anchor, products, 0).sum(axis=1).astype(np.float32)
    delta = rng.standard_normal(16).astype(np.float32)
    delta[0] = 0.0
    tiles = {"logits": logits, "products": products, "peak": peak, "total": total, "anchor": anchor,
             "delta": delta}

    def cotangents(anchored: bool, tiles: dict) -> np.ndarray:
        forms = attention_backward_algebra.derived_cotangents(keys_first=False, dropped=False, masked=False,
                                                             anchored=anchored)
        return np.asarray(forms.mapped({name: jnp.asarray(value) for name, value in tiles.items()})[2])

    @jax.jit
    def hand(logits, products, peak, total, anchor, delta, at_anchor):
        """Each hand form's cotangents, Mosaic's and an anchored mainloop's."""
        e = jnp.exp(logits - peak[:, None])
        over = (1 / total)[:, None]
        about = products - anchor[:, None]
        held = jnp.where(at_anchor, 0.0, about)
        return (e * over) * (about - delta[:, None]), e * ((held - delta[:, None]) * over)

    mosaic, anchored_hand = hand(*(jnp.asarray(value) for value in (*tiles.values(), at_anchor)))
    plain = cotangents(anchored=False, tiles=tiles)
    np.testing.assert_array_equal(plain, np.asarray(mosaic))
    anchored = cotangents(anchored=True, tiles={**tiles, "at_anchor": at_anchor})
    np.testing.assert_array_equal(anchored, np.asarray(anchored_hand))
    assert anchored[0, 1] != 0 and plain[0, 1] == 0


@contextlib.contextmanager
def fast(forms: str = "fast"):
    """Within the block, the run's `forms` (the fast ones' base-2 fold, or the accurate ones' natural logits,
    each the declaration states)."""
    with generation.run_precision(dataclasses.replace(generation.numerics(), forms=forms)):
        yield


CALLS = {"plain": {}, "residuals": {"residuals": True}, "dropped": {"drop": 0.25}}
"""The forward calls the derived walk serves: a call's output, a training call's with its residuals (the
training walk's step, `training_step`, taking the derived softmax), and a dropping call's."""


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("numerics", ["fast", "accurate"])
@pytest.mark.parametrize("call", list(CALLS))
@pytest.mark.parametrize("form", ["full", "causal"])
def test_the_portable_mainloop_computes_attention(form, call, numerics, dtype):
    """The plain-JAX mainloop's forward (`attention_portable.xla_mainloop`) over fp16 operands (a plain walk,
    of no compensation or guard) and fp32 ones (its running sums compensated, its plain walk under the
    headroom
    guard's), each of which the derived forms serve, under the run's fast forms or its accurate ones (natural
    logits, the largest weight held to 1 and the rescale compensated, which the declaration states), its
    walk's step, merge and finish the derived forms, the product of its weights with the values its own
    (`Keys.dot`): on par with the reference (check 1), a plain call's, a training call's (with its
    residuals) and a dropping call's."""
    spans = template.spans(LENGTH, LENGTH, 1, **SPANS[form])
    rate = CALLS[call].get("drop")
    residuals = CALLS[call].get("residuals", False)

    def hooks(words):
        return None if rate is None else template.Hooks(drop=template.Dropout(
            words, jnp.arange(1, dtype=jnp.int32), jnp.arange(1, dtype=jnp.int32), rate))

    def ours(q, k, v, words):
        with fast(numerics):
            out = attention_portable.xla_mainloop(TILES, q[None, :, None], k[None, :, None], v[None, :, None],
                                                  spans, hooks(words), residuals=residuals,
                                                  natural=numerics == "accurate", scale=SCALE)
        return (out[0] if residuals else out)[0, :, 0]

    def reference(q, k, v, words):
        return template.reference(q[None, :, None], k[None, :, None], v[None, :, None], spans,
                                  hooks(words))[0, :, 0]

    def operands(seed):
        rng = np.random.default_rng(seed)
        return (*(jnp.asarray(rng.standard_normal((LENGTH, DEPTH)), dtype) for _ in range(3)),
                jax.random.bits(jax.random.key(seed), (2,), jnp.uint32))

    assert_as_exact(jax.jit(ours), jax.jit(reference), operands,
                    f"portable forward {form} {call} {numerics} {dtype}", seeds=16)


@pytest.mark.parametrize("form", ["full", "causal"])
def test_the_portable_mainloop_differentiates_attention(form):
    """The plain-JAX mainloop's backward (`attention_portable.backward`) under the run's fast forms, its dQ
    walk over the keys and its dK and dV walks over the queries the derived forms
    (`attention_backward_algebra.derived_backward`), from the forward's residuals and its split D: the
    gradients on par with the reference's (check 1)."""
    spans = template.spans(LENGTH, LENGTH, 1, **SPANS[form])

    def ours(q, k, v, dout):
        with fast():
            grads = attention_portable.backward(TILES, q[None, :, None], k[None, :, None], v[None, :, None],
                                                spans, None, dout[None, :, None], scale=SCALE)
        return grads.query[0, :, 0], grads.key[0, :, 0], grads.value[0, :, 0]

    def reference(q, k, v, dout):
        _, pull = jax.vjp(lambda q, k, v: template.reference(q[None, :, None], k[None, :, None],
                                                             v[None, :, None], spans, None)[0, :, 0], q, k, v)
        return pull(dout)

    def operands(seed):
        rng = np.random.default_rng(seed)
        return tuple(jnp.asarray(rng.standard_normal((LENGTH, DEPTH)), jnp.float32) for _ in range(4))

    assert_as_exact(jax.jit(ours), jax.jit(reference), operands,
                    f"portable backward {form}", seeds=16)


ONE_KEY = attention_portable.Tiles(8, 8, 8)
"""The plain-JAX mainloop's tiles over a call of one query (`one_key`)."""


def one_key(dtype, query, key, value, dout):
    """A call's operands of one query and `len(key)` keys, each a row of width 8 (`[1, n, 1, 8]`)."""
    def laid(rows):
        rows = np.asarray(rows, np.float64).reshape(len(rows), -1)
        return jnp.asarray(np.pad(rows, ((0, 0), (0, 8 - rows.shape[1])))[None, :, None, :], dtype)

    return laid(query), laid(key), laid(value), laid(dout)


def test_a_derived_backward_of_one_key_gives_an_exactly_zero_query_gradient():
    """A bf16 query of one key, its value `(1, 2^-8)` and its output's cotangent `(1, 1)`: its score
    cotangent `P (dP - D)` is zero, `dP` the reference's `1 + 2^-8` and `D` the output's row sum, so its
    query's gradient is exactly zero, as the reference's is, under the derived backward too: its `dP`
    accumulates in fp32 (one rounded to bf16 is 1, and its cotangent -2^-8)."""
    query, key, value, dout = one_key(jnp.bfloat16, [[1.0]], [[1.0]], [[1.0, 2.0 ** -8]], [[1.0, 1.0]])
    spans = template.spans(1, 1, 1)
    with fast():
        grads = attention_portable.backward(ONE_KEY, query, key, value, spans, None, dout, scale=SCALE)
    np.testing.assert_array_equal(np.asarray(grads.query, np.float32), 0.0)


def test_a_derived_backward_masks_an_empty_querys_overflowing_product():
    """Two fp32 queries over one key, the second seeing none, the value's first component 2^127 and the second
    query's output cotangent 4 there: its product `dP` is 2^129, past fp32's largest, beside a weight of 0.
    Under the derived backward every gradient is finite, as the reference's are: the cotangents are masked
    where a query sees no key (`softmax_cotangents`), not multiplied by its zero weight."""
    query, key, value, dout = one_key(jnp.float32, [[0.0], [0.0]], [[0.0]], [[2.0 ** 127]],
                                             [[0.0], [4.0]])
    spans = template.KeySpans(jnp.asarray([[0, 1]], jnp.int32), jnp.asarray([[1, 1]], jnp.int32), None, None,
                              keys=1)
    _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans, None), query, key, value)
    assert all(np.isfinite(np.asarray(grad)).all() for grad in pull(dout))
    with fast():
        grads = attention_portable.backward(ONE_KEY, query, key, value, spans, None, dout, scale=SCALE)
    assert all(np.isfinite(np.asarray(grad)).all() for grad in (grads.query, grads.key, grads.value))


def test_a_query_a_hook_empties_takes_no_cotangent():
    """Two fp32 queries over one key, a score hook hiding it from the second (its logit -inf, its denominator
    0, its output the reference's zero), the value's first component 2^127 and the second query's output
    cotangent 4 there: its `dO Vᵀ` is 2^129, past fp32's largest, beside weights of 0. Every gradient is
    finite, as the reference's: the query takes no cotangent
    (`attention.launch_arguments`)."""
    query, key, value, dout = one_key(jnp.float32, [[0.0], [0.0]], [[0.0]], [[2.0 ** 127]],
                                             [[0.0], [4.0]])
    hooks = template.Hooks(score=Hook(first_query_only))
    spans = template.spans(2, 1, 1)
    _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans, hooks), query, key, value)
    assert all(np.isfinite(np.asarray(grad)).all() for grad in pull(dout))
    with fast():
        grads = attention_portable.backward(ONE_KEY, query, key, value, spans, hooks, dout, scale=SCALE)
    assert all(np.isfinite(np.asarray(grad)).all() for grad in (grads.query, grads.key, grads.value))


def first_query_only(s, q_pos, k_pos):
    """A score hook hiding every key from each query but the first."""
    del k_pos
    return jnp.where(q_pos == 0, s, -jnp.inf)


def test_a_derived_backward_scales_its_cotangents_before_their_products():
    """An fp32 query of zero over two keys of first components 2^127 and -2^127, values 4 and -4, the
    output's cotangent 1 and the scale 1/4: the score cotangents are 2 and -2, whose products with the keys
    at the scale give a query gradient of 2^127, finite as the reference's, where their unscaled sum 2^129
    overflows (bible §6)."""
    query, key, value, dout = one_key(jnp.float32, [[0.0]], [[2.0 ** 127], [-(2.0 ** 127)]],
                                             [[4.0], [-4.0]], [[1.0]])
    spans = template.spans(1, 2, 1)
    _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans, None, scale=0.25), query, key, value)
    assert all(np.isfinite(np.asarray(grad)).all() for grad in pull(dout))
    with fast():
        grads = attention_portable.backward(ONE_KEY, query, key, value, spans, None, dout, scale=0.25)
    assert all(np.isfinite(np.asarray(grad)).all() for grad in (grads.query, grads.key, grads.value))


def test_a_derived_backward_at_a_scale_of_zero_gives_zero_query_and_key_gradients():
    """A call at a scale of zero (`attention.prescale`'s own case) under the derived backward: its weights
    uniform, its query's and key's gradients zero, as the reference's, its value's finite."""
    rng = np.random.default_rng(0)
    query, key, value, dout = (jnp.asarray(rng.standard_normal((1, LENGTH, 1, DEPTH)), jnp.float32)
                               for _ in range(4))
    spans = template.spans(LENGTH, LENGTH, 1)
    with fast():
        grads = attention_portable.backward(TILES, query, key, value, spans, None, dout, scale=0.0)
    np.testing.assert_array_equal(np.asarray(grads.query), 0.0)
    np.testing.assert_array_equal(np.asarray(grads.key), 0.0)
    assert np.isfinite(np.asarray(grads.value)).all()


FRESH = """
import dataclasses
import jax, jax.numpy as jnp
from dew.nn.kernels import decode_attention, generation
from dew.nn.kernels.templates import attention as template, attention_portable

operands = [jnp.full((1, 16, 1, 8), 0.1 * (index + 1), jnp.float16) for index in range(3)]
fast = dataclasses.replace(generation.numerics(), forms="fast")
with generation.run_precision(fast):
    out = attention_portable.xla_mainloop(attention_portable.Tiles(8, 8, 8), *operands,
                                          template.spans(16, 16, 1), residuals=True)
print(bool(jnp.all(jnp.isfinite(out[0]))))
"""
"""A fresh process's first derived walk: a training call's forward, its step the training walk's, by the
derived forms' softmax."""


def test_a_fresh_processs_training_forward_finishes_by_the_derived_forms():
    """A training call's forward (`residuals=True`, its walk the training step's, by the derived forms) as a
    fresh process's first call under the derived forms: its finish takes the derived forms of the state it
    holds, whatever walk came before it in the process (none here), and its output is finite."""
    done = subprocess.run([sys.executable, "-c", FRESH], capture_output=True, text=True,
                          env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SOURCE)})
    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.strip().splitlines()[-1] == "True"


def tempered(s, q_pos, k_pos, temperature):
    """A score hook dividing each query's logits by its temperature, a query parameter."""
    del q_pos, k_pos
    return s / temperature


def test_a_derived_backward_masks_its_hook_cotangents_where_padding_reads_no_key():
    """One fp32 query and key under a query temperature of 1, padded to an 8-row tile: a padded query's
    temperature is zero (`attention._padded`), its hook derivative not finite, and its logits unseen. Under
    the derived backward the real gradients are finite, as the reference's are, each tile's cotangents masked
    before their products (`pulled`)."""
    rng = np.random.default_rng(0)
    query, key, value, dout = (jnp.asarray(rng.standard_normal((1, 1, 1, 8)), jnp.float32) for _ in range(4))
    spans = template.spans(1, 1, 1)
    temperature = Side(jnp.ones((1, 1, 1), jnp.float32), template.Kind.QUERY)
    hooks = template.Hooks(score=Hook(tempered, (temperature,)))
    with fast():
        grads = attention_portable.backward(attention_portable.Tiles(8, 8, 8), query, key, value, spans,
                                            hooks, dout)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(grads))


def test_a_split_decodes_states_merge_and_finish_by_the_derived_forms():
    """A split decode's slices' states (`[rows, splits, queries, heads]`, `decode_attention._combined`) of an
    fp16 call under the run's fast forms, merged and finished by the derived forms over every row whatever
    the states' leading axes: on par with XLA's merge in one sum against float64's (check 1)."""
    spec = decode_attention._spec(0.25, 1.0, None, natural=False, dtype=jnp.float16, sinks=False,
                                  stored=jnp.float16)

    def states(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.standard_normal((2, 3, 4, 2, 16)), jnp.float32),
                jnp.asarray(rng.standard_normal((2, 3, 4, 2)), jnp.float32),
                jnp.asarray(rng.uniform(0.5, 2.0, (2, 3, 4, 2)), jnp.float32))

    def combined(sums, peaks, totals):
        with fast():
            return decode_attention._combined(spec, sums, peaks, totals, axis=1)

    def exact(sums, peaks, totals):
        # XLA's merge of the states in one sum, in base 2 (`spec.base2`): raw products' peaks (`spec.raw`)
        # differ by the scale times log2(e).
        factor = spec.scale * np.log2(np.e) if spec.raw else 1.0
        weights = jnp.exp2((peaks - jnp.max(peaks, axis=1, keepdims=True)) * factor)
        return (jnp.sum(weights[..., None] * sums, axis=1)
                / jnp.sum(weights * totals, axis=1)[..., None])

    assert_as_exact(jax.jit(combined), jax.jit(exact), states, "a split decode's combine", seeds=16)


def test_forward_declined_checks_feasibility_without_pricing(tmp_path):
    """`MosaicForward.declined` agrees with `derived(...) is None` without tracing kernels through
    `price_config` / storing consumer choices, across both multi-consumer (`sm90`) and rewalked-apart
    (`sm100`) launches (rows 11724, 11883, 11898-11900, 11923-11932, 11952)."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "JAX_COMPILATION_CACHE_DIR": str(tmp_path)}
    script = "\n".join([
        "from pathlib import Path",
        "import jax, jax.numpy as jnp",
        "from dew.nn.kernels import generation",
        "from dew.nn.kernels.templates import attention as template, attention_mosaic",
        f"cache = Path({str(tmp_path)!r})",
        "q = jax.ShapeDtypeStruct((1, 1280, 6, 64), jnp.bfloat16)",
        "spans = template.spans(1280, 1280, 1, causal=True)",
        "for gen in ('sm90', 'sm100'):",
        "    with generation.assumed(gen):",
        "        target = generation.gpu_target()",
        "        op, args = attention_mosaic._forward_call(q, q, q, spans, None, scale=None, natural=False)",
        "        ba = op.bind(*args, residuals=template.Kept.NONE, interpret=False)",
        "        assert op.declined(ba, target) is None",
        "choices = list(cache.rglob('certified_*.json'))",
        "assert not choices, f'declined wrote choice cache: {choices}'",
        "with generation.assumed('sm90'):",
        "    target = generation.gpu_target()",
        "    op, args = attention_mosaic._forward_call(q, q, q, spans, None, scale=None, natural=False)",
        "    ba = op.bind(*args, residuals=template.Kept.NONE, interpret=False)",
        "    assert op.derived(ba, target) is not None",
        "assert list(cache.rglob('certified_*.json')), 'derived should have stored sm90 consumer choice'",
        "big = jax.ShapeDtypeStruct((1, 128, 4, 1024), jnp.bfloat16)",
        "big_spans = template.spans(128, 128, 1, causal=True)",
        "with generation.assumed('sm80'):",
        "    target = generation.gpu_target()",
        "    op, args = attention_mosaic._forward_call(",
        "        big, big, big, big_spans, None, scale=None, natural=False)",
        "    ba = op.bind(*args, residuals=template.Kept.NONE, interpret=False)",
        "    assert op.declined(ba, target) is not None and op.derived(ba, target) is None",
    ])
    subprocess.run([sys.executable, "-c", script], env=env, check=True)


def test_compute_bound_forward_skips_in_kernel_decided_and_uses_fast_may_scale():
    """Compute-bound forwards (`not streams`) skip in-kernel `decided` (avoiding per-tile `keys_read`
    SMEM reads and scaled register bloat) while streaming calls retain `decided`, and 1-kernel `may_scale`
    soundly bounds normal, non-finite, empty, and overflowing inputs (rows 5272, 5274)."""
    from dew.nn.kernels import generation
    from dew.nn.kernels.templates import attention as template, attention_algebra as algebra, attention_mosaic

    for gen, small_q in (("sm90", 64), ("sm80", 16)):
        with generation.assumed(gen):
            target = generation.gpu_target()
            q_big = jax.ShapeDtypeStruct((2, 4096, 8, 64), jnp.bfloat16)
            spans_big = template.spans(4096, 4096, 2, causal=True)
            op_big, args_big = attention_mosaic._forward_call(
                q_big, q_big, q_big, spans_big, None, scale=None, natural=False)
            ba_big = op_big.bind(*args_big, residuals=template.Kept.NONE, interpret=False)
            cfg_big = op_big.derived(ba_big, target)
            assert cfg_big is not None
            assert not op_big.deciding(ba_big, cfg_big, target=target), (gen, cfg_big)

            q_small = jax.ShapeDtypeStruct((1, small_q, 4, 64), jnp.bfloat16)
            spans_small = template.spans(small_q, 64, 1, causal=True)
            op_small, args_small = attention_mosaic._forward_call(
                q_small, q_small, q_small, spans_small, None, scale=None, natural=False)
            ba_small = op_small.bind(*args_small, residuals=template.Kept.NONE, interpret=False)
            cfg_small = op_small.derived(ba_small, target)
            assert cfg_small is not None
            assert op_small.deciding(ba_small, cfg_small, target=target), (gen, cfg_small)
    normal = jnp.ones((1, 4096, 4, 64), jnp.bfloat16)
    assert not bool(algebra.may_scale(normal, keys_axis=1))
    with_inf = normal.at[0, 0, 0, 0].set(jnp.inf).at[0, 1, 0, 0].set(jnp.nan)
    assert not bool(algebra.may_scale(with_inf, keys_axis=1))
    huge = normal.at[0, 0, 0, 0].set(jnp.bfloat16(2.0 ** 120))
    assert bool(algebra.may_scale(huge, keys_axis=1))
    empty = jnp.zeros((1, 0, 4, 64), jnp.bfloat16)
    assert not bool(algebra.may_scale(empty, keys_axis=1))
    assert bool(algebra.may_scale(normal, keys_axis=1, initial=jnp.full((1, 4, 1, 64), 2.0 ** 126)))


def test_explicit_target_consistency_for_forward_decides_and_fused_backward_staging():
    """Explicit `target` drives `deciding`, `registers`, `staged_bytes`, `resident`, `attention_decode.plan`,
    and `MosaicBackward._scopes` / `fused_staging` without leaking ambient `generation.gpu_target()`."""
    from dew.nn.kernels import generation
    from dew.nn.kernels.templates import (
        attention as template,
        attention_decode,
        attention_mosaic,
        attention_mosaic_fused as fused,
        attention_mosaic_program as program,
    )

    h100 = generation.gpu_target("NVIDIA H100 80GB HBM3")
    h200 = generation.gpu_target("NVIDIA H200")
    # Q=256 bf16 streams on H100 (ridge ~310) and is compute-bound on H200 (ridge ~214), both sm90 wgmma.
    q_mid = jax.ShapeDtypeStruct((1, 256, 4, 64), jnp.bfloat16)
    spans_mid = template.spans(256, 256, 1, causal=True)
    op_mid, args_mid = attention_mosaic._forward_call(
        q_mid, q_mid, q_mid, spans_mid, None, scale=None, natural=False)
    ba_mid = op_mid.bind(*args_mid, residuals=template.Kept.NONE, interpret=False)
    assert attention_mosaic.streams(q_mid, h100) and not attention_mosaic.streams(q_mid, h200)
    tiles_wgmma = attention_mosaic.Tiles(64, 64, 2, 1, family="wgmma")
    with generation.assumed(h200):
        h200_dec = op_mid.deciding(ba_mid, tiles_wgmma, target=h200)
        h200_regs = op_mid.registers(ba_mid, tiles_wgmma, target=h200)
        h200_bytes = op_mid.staged_bytes(ba_mid, tiles_wgmma, h200)
        h200_res = op_mid.resident(ba_mid, tiles_wgmma, h200)
        cross_h100_plan = attention_decode.plan(q_mid, q_mid, q_mid, spans_mid, None, scale=None, target=h100)
        assert cross_h100_plan is not None and cross_h100_plan.decides
    assert not h200_dec and h200_regs is not None
    with generation.assumed(h100):
        assert op_mid.deciding(ba_mid, tiles_wgmma, target=h200) == h200_dec
        assert op_mid.registers(ba_mid, tiles_wgmma, target=h200) == h200_regs
        assert op_mid.staged_bytes(ba_mid, tiles_wgmma, h200) == h200_bytes
        assert op_mid.resident(ba_mid, tiles_wgmma, h200) == h200_res
        assert op_mid.deciding(ba_mid, tiles_wgmma, target=h100)
        h100_regs = op_mid.registers(ba_mid, tiles_wgmma, target=h100)
        assert h100_regs is not None and h100_regs > h200_regs
        assert op_mid.staged_bytes(ba_mid, tiles_wgmma, h100) > h200_bytes
        h100_plan = attention_decode.plan(q_mid, q_mid, q_mid, spans_mid, None, scale=None, target=h100)
        assert h100_plan is not None and h100_plan.decides == cross_h100_plan.decides
    # Row 11987: cross-generation explicit `target` resolves `bundle_of` against `target`, not ambient gen.
    with generation.assumed("sm80"):
        assert op_mid.deciding(ba_mid, tiles_wgmma, target=h200) == h200_dec
        assert op_mid.registers(ba_mid, tiles_wgmma, target=h200) == h200_regs
        assert op_mid.staged_bytes(ba_mid, tiles_wgmma, h200) == h200_bytes
        assert not op_mid.legal(ba_mid, tiles_wgmma, generation.gpu_target("sm80"))
    with generation.assumed("sm100"):
        assert op_mid.takes(ba_mid, tiles_wgmma, h200)
        assert op_mid.resident(ba_mid, tiles_wgmma, h200) == h200_res
    sm90 = generation.gpu_target("sm90")
    # Explicit `target.shared_limit` controls `fused_staging` and `MosaicBackward.staged_bytes` even when
    # ambient `generation.gpu_target()` has a larger limit.
    q_bwd = jax.ShapeDtypeStruct((1, 512, 8, 128), jnp.bfloat16)
    k_bwd = jax.ShapeDtypeStruct((1, 512, 2, 128), jnp.bfloat16)
    spans_bwd = template.spans(512, 512, 1, causal=True)
    zero_row = jax.ShapeDtypeStruct((1, 8, 512), jnp.float32)
    zero_idx = jax.ShapeDtypeStruct((1, 8, 512), jnp.int32)
    split = template.Split(
        jax.ShapeDtypeStruct((1, 8, 512, 128), jnp.float32),
        zero_row, zero_row, zero_row)
    bwd_op, bwd_args = attention_mosaic._backward_call(
        q_bwd, k_bwd, k_bwd, spans_bwd, None, q_bwd,
        residuals=(zero_row, zero_row, zero_idx, split),
        scale=None, natural=attention_mosaic.NATURAL, accurate=False)
    bwd_ba = bwd_op.bind(*bwd_args, interpret=False)
    with generation.assumed("sm90"):
        family = program.bundle("sm90", jnp.bfloat16)
        assert family is not None
        bwd_cfg = attention_mosaic.BackwardTiles(64, 64, 2, 2, fused=True, family=family.name)
        full_bytes = bwd_op.staged_bytes(bwd_ba, bwd_cfg, sm90)
        full_plan = fused.fused_staging(family, bwd_cfg, 128, jnp.bfloat16, 2, reduced=0,
                                        grouped=True, rounded=True, anchored=False, target=sm90)
        assert full_plan.buffers == 2
        tight = dataclasses.replace(sm90, shared_limit=full_bytes - 1)
        tight_plan = fused.fused_staging(family, bwd_cfg, 128, jnp.bfloat16, 2, reduced=0,
                                         grouped=True, rounded=True, anchored=False, target=tight)
        assert tight_plan.buffers == 1
        tight_bytes = bwd_op.staged_bytes(bwd_ba, bwd_cfg, tight)
        assert tight.shared_limit is not None
        assert tight_bytes < full_bytes and tight_bytes <= tight.shared_limit
        # Row 11989: `_deepened` sizes statistics rings using padded `_width` (96 -> 128), matching `_scopes`.
        q_96 = jax.ShapeDtypeStruct((1, 512, 8, 96), jnp.bfloat16)
        k_96 = jax.ShapeDtypeStruct((1, 512, 2, 96), jnp.bfloat16)
        split_96 = template.Split(jax.ShapeDtypeStruct((1, 8, 512, 96), jnp.float32),
                                  zero_row, zero_row, zero_row)
        bwd96_op, bwd96_args = attention_mosaic._backward_call(
            q_96, k_96, k_96, spans_bwd, None, q_96,
            residuals=(zero_row, zero_row, zero_idx, split_96),
            scale=None, natural=attention_mosaic.NATURAL, accurate=False)
        bwd96_ba = bwd96_op.bind(*bwd96_args, interpret=False)
        deep96 = bwd96_op._deepened(bwd96_ba, bwd_cfg, sm90)
        assert bwd96_op.takes(bwd96_ba, deep96, sm90)
        assert bwd96_op.resident(bwd96_ba, deep96, sm90) == bwd96_op.resident(bwd96_ba, bwd_cfg, sm90)
        padded_plan = fused.fused_staging(family, deep96, 128, jnp.bfloat16, deep96.stages, reduced=0,
                                          grouped=True, rounded=True, anchored=False, target=sm90)
        assert padded_plan.streamed >= 3, (deep96, padded_plan)
    # Cross-generation explicit-target backward queries resolve `bundle_of` on `sm90` under ambient `sm80`.
    with generation.assumed("sm80"):
        assert bwd_op.staged_bytes(bwd_ba, bwd_cfg, sm90) == full_bytes
        assert bwd_op.registers(bwd_ba, bwd_cfg, target=sm90) is not None
        assert bwd_op.takes(bwd_ba, bwd_cfg, sm90)


def test_tpu_scopes_read_trailing_run_axis_and_hooks_key_step_live_tiles_by_tile_shape():
    """Rows 12072, 12073: TPU forward `_scope` and backward `_scopes` read the trailing run count `R` of
    3D `[B, Q, R]` span `start` arrays (and tolerate `start=None` on full calls), and `step_live_tiles`
    traces and caches hook-bearing calls at their requested `(block_q, block_k)`."""
    from dew.nn.kernels.templates import attention_mosaic, attention_mosaic_program as program
    from dew.nn.kernels.templates.attention_mosaic_scores import _STEP_TILE_CACHE, step_live_tiles

    q_len, r_runs = 256, 5
    q = jax.ShapeDtypeStruct((1, q_len, 4, 128), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((1, q_len, 2, 128), jnp.bfloat16)
    starts = jnp.broadcast_to(jnp.array([0, 32, 64, 128, 192], jnp.int32)[None, None, :],
                              (1, q_len, r_runs))
    ends = jnp.broadcast_to(jnp.array([16, 48, 96, 160, 224], jnp.int32)[None, None, :],
                            (1, q_len, r_runs))
    spans = template.KeySpans(starts, ends, None, None, q_len)
    tpu_target = generation.gpu_target("v5e")
    with generation.assumed("v5e"):
        family = program.bundle("v5e", jnp.bfloat16)
        assert family is not None
        # Full call with `start=None` must resolve `span_runs = 1` without AttributeError.
        full_op = attention_mosaic.MosaicForward(scale=128 ** -0.5, keys=q_len, form="full", natural=True)
        full_ba = full_op.bind(q, k, k, None, None, None, None, (), (), None,
                               residuals=template.Kept.NONE, interpret=False)
        fwd_cfg = attention_mosaic.Tiles(128, 128, 2, 1, family=family.name)
        assert full_op.staged_bytes(full_ba, fwd_cfg, tpu_target) > 0

        fwd_op, fwd_args = attention_mosaic._forward_call(
            q, k, k, spans, None, scale=None, natural=attention_mosaic.NATURAL)
        fwd_ba = fwd_op.bind(*fwd_args, residuals=template.Kept.NONE, interpret=False)
        assert fwd_op.staged_bytes(fwd_ba, fwd_cfg, tpu_target) > 0
        fwd_keys = [key for key in _STEP_TILE_CACHE
                    if isinstance(key, tuple) and len(key) > 14 and not key[2] and key[10] == "spans"]
        assert any(key[14] == r_runs for key in fwd_keys), fwd_keys
        assert not any(key[14] == q_len for key in fwd_keys), fwd_keys

        zero_row = jax.ShapeDtypeStruct((1, 4, q_len), jnp.float32)
        zero_idx = jax.ShapeDtypeStruct((1, 4, q_len), jnp.int32)
        split = template.Split(jax.ShapeDtypeStruct((1, 4, q_len, 128), jnp.float32),
                               zero_row, zero_row, zero_row)
        bwd_op, bwd_args = attention_mosaic._backward_call(
            q, k, k, spans, None, q, residuals=(zero_row, zero_row, zero_idx, split),
            scale=None, natural=attention_mosaic.NATURAL, accurate=False)
        bwd_ba = bwd_op.bind(*bwd_args, interpret=False)
        bwd_cfg = attention_mosaic.BackwardTiles(128, 128, 2, 1, fused=False, family=family.name)
        assert bwd_op.staged_bytes(bwd_ba, bwd_cfg, tpu_target) > 0
        bwd_keys = [key for key in _STEP_TILE_CACHE
                    if isinstance(key, tuple) and len(key) > 14 and key[2] and key[10] == "spans"]
        assert any(key[14] == r_runs for key in bwd_keys), bwd_keys
        assert not any(key[14] == q_len for key in bwd_keys), bwd_keys

    causal_spec = algebra.Forward("causal", 256, ids=False, scale=1.0 / 8.0)

    def shape_branching_hook(logits, _q, _k):
        if logits.shape[0] > 64:
            a = jnp.sin(logits)
            b = jnp.cos(logits)
            c = jnp.tanh(logits)
            return (a + b) * c + a * b + logits
        return logits

    small_tiles = step_live_tiles(64, 64, spec=causal_spec, score_mod=shape_branching_hook)
    large_tiles = step_live_tiles(128, 64, spec=causal_spec, score_mod=shape_branching_hook)
    assert len(large_tiles) > len(small_tiles), (small_tiles, large_tiles)


def test_staged_logits_is_skipped_when_unstaged_already_fits_shared_memory_at_same_residency():
    """Row 11975: `MosaicBackward._staged_helps` admits `staged_logits=True` on explicit staged/unstaged
    schedule pairs only when the un-staged config exceeds shared memory or has lower residency, so refusing
    an un-staged config for register spills skips its doomed `staged_logits=True` twin."""
    from dew.nn.kernels.templates import attention_mosaic, report

    spans = template.spans(1024, 1024, 1, causal=False)
    zero_row = jax.ShapeDtypeStruct((1, 8, 1024), jnp.float32)
    zero_idx = jax.ShapeDtypeStruct((1, 8, 1024), jnp.int32)

    def backward_ba(width: int, hooks: template.Hooks | None):
        q = jax.ShapeDtypeStruct((1, 1024, 8, width), jnp.float16)
        k = jax.ShapeDtypeStruct((1, 1024, 2, width), jnp.float16)
        split = template.Split(jax.ShapeDtypeStruct((1, 8, 1024, width), jnp.float32),
                               zero_row, zero_row, zero_row)
        op, args = attention_mosaic._backward_call(
            q, k, k, spans, hooks, q, residuals=(zero_row, zero_row, zero_idx, split),
            scale=None, natural=attention_mosaic.NATURAL, accurate=False)
        return op, op.bind(*args, interpret=False)

    bias = jax.ShapeDtypeStruct((1, 8, 1024), jnp.float16)
    hooks = template.Hooks(
        score=Hook(lambda s, _q, _k, a, b: s + a.astype(s.dtype) + 2.0 * b.astype(s.dtype),
                   (Side(bias, template.Kind.KEY), Side(bias, template.Kind.KEY))))
    with generation.assumed("sm80"):
        target = generation.gpu_target()
        op, ba = backward_ba(256, hooks)
        # Negative pair: unstaged 32x32x2 at width 256 fits SMEM at the same residency, so staged_logits is
        # rejected.
        unstaged = attention_mosaic.BackwardTiles(
            block_q=32, block_k=32, stages=2, consumers=1, fused=False, family="mma.sync",
            columns=1, staged_logits=False)
        staged = dataclasses.replace(unstaged, staged_logits=True)
        assert op.staged_bytes(ba, unstaged, target) <= op.program_shared(unstaged, target)
        assert op.resident(ba, staged, target) <= op.resident(ba, unstaged, target)
        assert op._staged_helps(ba, unstaged, target)
        assert not op._staged_helps(ba, staged, target)
        with report.refusals():
            for narrow in (False, True):
                report.refuse_launch(op, ba, dataclasses.replace(unstaged, narrow=narrow),
                                     frozenset({"attention_dkv"}))
            stepped = op.derived(ba, target)
        assert stepped is not None and op.takes(ba, stepped, target)
        assert stepped not in (staged, dataclasses.replace(staged, narrow=True)), stepped

        # Positive pair 1 (SMEM relief): unstaged 32x32x3 at width 256 exceeds SMEM while staged fits.
        op_256, ba_256 = backward_ba(256, None)
        over = attention_mosaic.BackwardTiles(
            block_q=32, block_k=32, stages=3, consumers=1, fused=False, family="mma.sync",
            columns=1, staged_logits=False)
        fits_staged = dataclasses.replace(over, staged_logits=True)
        assert op_256.staged_bytes(ba_256, over, target) > op_256.program_shared(over, target)
        assert op_256.staged_bytes(ba_256, fits_staged, target) <= op_256.program_shared(fits_staged, target)
        assert op_256._staged_helps(ba_256, fits_staged, target)

        # Positive pair 2 (residency gain): unstaged 32x32x3 at width 128 fits SMEM at 1 CTA/SM, staged at 2.
        op_128, ba_128 = backward_ba(128, None)
        low_res = attention_mosaic.BackwardTiles(
            block_q=32, block_k=32, stages=3, consumers=1, fused=False, family="mma.sync",
            columns=1, staged_logits=False)
        high_res = dataclasses.replace(low_res, staged_logits=True)
        assert op_128.staged_bytes(ba_128, low_res, target) <= op_128.program_shared(low_res, target)
        assert op_128.resident(ba_128, high_res, target) > op_128.resident(ba_128, low_res, target)
        assert op_128._staged_helps(ba_128, high_res, target)


def test_step_live_tiles_covers_keys_first_ragged_spans_and_array_field_hooks():
    """Rows 11963, 11973, 11988, 11994: `step_live_tiles` carries `how.factor` for raw/subtracted specs,
    orients keys-first `(block_k, block_q)` tiles, reflects ragged full visibility and multi-run spans,
    excludes HEAD/QUERY grads from keys-first `KEY_ROWS`, and caches array-holding `FieldHook`s via
    `tuning._held_key` while propagating genuine `TypeError`s."""
    from dew.nn.kernels import tuning
    from dew.nn.kernels.templates import attention_mosaic
    from dew.nn.kernels.templates.attention_mosaic_scores import _visibility_facts, step_live_tiles

    causal_spec = algebra.Forward("causal", 128, ids=False, scale=1.0 / 8.0)
    raw_spec = algebra.Forward("causal", 128, ids=False, scale=1.0 / 8.0, subtracted=True)
    assert algebra.recompute(raw_spec).factor is not None
    assert len(step_live_tiles(64, 64, spec=raw_spec, backward=True)) > 0

    kf_tiles = step_live_tiles(64, 32, spec=causal_spec, backward=True, keys_first=True,
                               score_mod=lambda s, q, k, qb, kb: s + qb + kb + (q - k).astype(jnp.float32),
                               score_types=((template.Kind.QUERY, "float32"),
                                            (template.Kind.KEY, "float32")))
    assert kf_tiles and all(shape == (32, 64) for shape, _ in kf_tiles), kf_tiles

    q_rag = jax.ShapeDtypeStruct((1, 100, 2, 64), jnp.bfloat16)
    spans_full = template.spans(100, 100, 1, causal=False)
    zero_row = jax.ShapeDtypeStruct((1, 2, 100), jnp.float32)
    zero_idx = jax.ShapeDtypeStruct((1, 2, 100), jnp.int32)
    res_rag = (zero_row, zero_row, zero_idx, template.Split(
        jax.ShapeDtypeStruct((1, 2, 100, 64), jnp.float32), zero_row, zero_row, zero_row))
    bwd_rag_op, _ = attention_mosaic._backward_call(
        q_rag, q_rag, q_rag, spans_full, None, q_rag, residuals=res_rag,
        scale=None, natural=attention_mosaic.NATURAL, accurate=False)
    rag_cfg = attention_mosaic.BackwardTiles(64, 64, 2, 1, family="wgmma")
    rag_spec = bwd_rag_op._spec(100, 100, ids=False, initial=False, config=rag_cfg)
    assert _visibility_facts(rag_spec, 64)[:5] == ("full", False, True, True, 0), rag_spec
    aligned_spec = algebra.Forward("full", 128, ids=False, scale=1.0 / 8.0)
    assert _visibility_facts(aligned_spec, 64)[:5] == ("full", False, False, False, 0)
    assert len(step_live_tiles(64, 64, backward=True, spec=rag_spec)) >= len(
        step_live_tiles(64, 64, backward=True, spec=aligned_spec))

    spans_spec = algebra.Forward("spans", 128, ids=True, scale=1.0 / 8.0)
    assert _visibility_facts(spans_spec, 64, runs=1)[:5] == ("spans", True, False, False, 1)
    assert _visibility_facts(spans_spec, 64, runs=3)[:5] == ("spans", True, False, False, 3)
    single_run = len(step_live_tiles(64, 64, spec=spans_spec, runs=1))
    multi_run = len(step_live_tiles(64, 64, spec=spans_spec, runs=3))
    assert multi_run >= single_run > 0, (single_run, multi_run)

    def multi_param_hook(s, _q, _k, p1, p2, p3):
        return s + p1 * jnp.sin(s) + p2 * jnp.cos(s) + p3 * jnp.tanh(s)

    dkv_head = len(step_live_tiles(64, 64, spec=causal_spec, backward=True, keys_first=True,
                                   score_mod=multi_param_hook,
                                   score_types=((template.Kind.HEAD, "float32"),) * 3))
    dkv_query = len(step_live_tiles(64, 64, spec=causal_spec, backward=True, keys_first=True,
                                    score_mod=multi_param_hook,
                                    score_types=((template.Kind.QUERY, "float32"),) * 3))
    dkv_tile = len(step_live_tiles(64, 64, spec=causal_spec, backward=True, keys_first=True,
                                   score_mod=multi_param_hook,
                                   score_types=((template.Kind.TILE, "float32"),) * 3))
    assert dkv_tile > dkv_head and dkv_tile > dkv_query, (dkv_tile, dkv_head, dkv_query)

    @dataclasses.dataclass(frozen=True)
    class ArrayHook(tuning.FieldHook):
        scale: jax.Array
        calls: list[int] = dataclasses.field(default_factory=list)

        def __call__(self, s, _q, _k):
            self.calls.append(1)
            return s * self.scale

    arr_hook = ArrayHook(jnp.asarray(0.5, jnp.float32))
    first_tiles = step_live_tiles(64, 64, spec=causal_spec, score_mod=arr_hook)
    second_tiles = step_live_tiles(64, 64, spec=causal_spec, score_mod=arr_hook)
    assert first_tiles == second_tiles and len(arr_hook.calls) == 1, arr_hook.calls
    with pytest.raises(TypeError):
        step_live_tiles(64, 64, spec=causal_spec, score_mod=lambda s, _q, _k: s + "bad")


@COMPILES_ON_CPU
@pytest.mark.kernel
@pytest.mark.usefixtures("without_deterministic_ops")
@pytest.mark.parametrize("width", [256, 64, 128])
def test_sm90_roofline_forward_and_fused_backward_compile_spill_free(width):
    """Row 5272: sm90 compute-bound causal forward (d64/d128, d256 single-launch) and fused dK/dV backward
    (overlapped scaled dP/dV WGMMAs via `product_with` + double-buffered TMA dQ adds at d128) derive and
    compile spill-free, a width a case (row 12410: the five in one took 131 s)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn.kernels.templates import attention_mosaic

    spans = template.spans(4096, 4096, 2, causal=True)
    heads, kv_heads = (16, 4) if width == 256 else (32, 8)
    q = jax.ShapeDtypeStruct((2, 4096, heads, width), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((2, 4096, kv_heads, width), jnp.bfloat16)
    cases = [KernelCase(f"sm90-fwd-d{width}", lambda: (
        lambda query, key, value: attention_mosaic.forward(
            query, key, value, spans, None, residuals=template.Kept.NONE), (q, k, k)))]
    with generation.assumed("sm90"):
        if width == 256:
            op, args = attention_mosaic._forward_call(q, k, k, spans, None, scale=None,
                                                      natural=attention_mosaic.NATURAL)
            bound = op.bind(*args, residuals=template.Kept.NONE, interpret=False)
            config = op.derived(bound, generation.gpu_target())
            assert config is not None and op.takes(bound, config, generation.gpu_target()), config
        else:
            zero_row = jax.ShapeDtypeStruct((2, heads, 4096), jnp.float32)
            zero_idx = jax.ShapeDtypeStruct((2, heads, 4096), jnp.int32)
            res = (zero_row, zero_row, zero_idx, template.Split(
                jax.ShapeDtypeStruct((2, heads, 4096, width), jnp.float32), zero_row, zero_row, zero_row))
            cases.append(KernelCase(f"sm90-bwd-d{width}", lambda: (
                lambda query, key, value, dout, r: attention_mosaic.backward(
                    query, key, value, spans, None, dout, residuals=r, accurate=False), (q, k, k, q, res))))
    found = compile_mosaic_kernels(cases, "sm90")
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


def _offered_at(offered, block_q: int, block_k: int, columns: int | None) -> int | None:
    """The place in a derivation's `offered` configs of the first `block_q x block_k` tile at `columns`."""
    return next((at for at, config in enumerate(offered)
                 if (config.block_q, config.block_k, config.columns) == (block_q, block_k, columns)), None)


@COMPILES_ON_CPU
def test_tcgen05_rows_prices_natural_exp_and_defers_same_residency_column_splits():
    """Rows 5535, 12210: `forward_registers` for `tcgen05.rows` (`places_on_role`) charges `natural_exp`
    when `natural=True`, and the natural 128x64 forward at d256 compiles spill-free on a compile-only B200.
    The derivation offers a column split that changes no residency (128x64 at `columns=128`, the whole tile
    fitting shared and tensor memory) after the whole tile, where a pin can still take it, and it lowers
    pinned; a whole tile past shared memory (128x128) is offered nowhere."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn.kernels import tuning
    from dew.nn.kernels.templates import attention_mosaic, attention_mosaic_program as program
    from dew.nn.kernels.templates.attention_mosaic_registers import forward_registers

    spans = template.spans(1024, 1024, 1, causal=True)
    q = jax.ShapeDtypeStruct((1, 1024, 16, 256), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((1, 1024, 4, 256), jnp.bfloat16)
    with generation.assumed("sm100"):
        target = generation.gpu_target()
        tiles_64 = attention_mosaic.Tiles(128, 64, 2, 1, family="tcgen05.rows")
        tiles_128 = attention_mosaic.Tiles(128, 128, 2, 1, family="tcgen05.rows")
        rows_family = program.bundle_of(tiles_64, jnp.bfloat16, target=target)
        folded_regs = forward_registers(rows_family, tiles_64, 256, jnp.bfloat16,
                                        residuals=False, natural=False)
        natural_regs = forward_registers(rows_family, tiles_64, 256, jnp.bfloat16,
                                         residuals=False, natural=True)
        assert natural_regs > folded_regs, (folded_regs, natural_regs)
        op_nat, args_nat = attention_mosaic._forward_call(q, k, k, spans, None, scale=None, natural=True)
        ba_nat = op_nat.bind(*args_nat, residuals=template.Kept.NONE, interpret=False)
        op_fold, args_fold = attention_mosaic._forward_call(q, k, k, spans, None, scale=None, natural=False)
        ba_fold = op_fold.bind(*args_fold, residuals=template.Kept.NONE, interpret=False)
        assert op_nat.registers(ba_nat, tiles_64, target=target) == natural_regs
        assert op_fold.registers(ba_fold, tiles_64, target=target) == folded_regs
        offered = op_nat.offered(ba_nat, rows_family, target)
        whole, split = _offered_at(offered, 128, 64, None), _offered_at(offered, 128, 64, 128)
        assert whole is not None and split is not None and whole < split, offered
        assert op_fold.staged_bytes(ba_fold, tiles_128, target) > op_fold.program_shared(tiles_128, target)
        folded_offered = op_fold.offered(ba_fold, rows_family, target)
        assert _offered_at(folded_offered, 128, 128, None) is None, folded_offered

        def natural_fwd(query, key, value):
            return attention_mosaic.forward(query, key, value, spans, None, natural=True)

        with tuning.pinned({attention_mosaic.MosaicForward: offered[split]}):
            jax.jit(natural_fwd).trace(q, k, k).lower(lowering_platforms=("cuda",))
        found = compile_mosaic_kernels(
            [KernelCase("natural-128x64", lambda: (natural_fwd, (q, k, k)))], "sm100")
    assert not (failed := mosaic_failures(found)), "\n".join(failed)


@COMPILES_ON_CPU
def test_row_indexed_hooks_take_and_lower_at_fewer_rows_on_mma_sync():
    """Rows 7360, 12210: a backward under a row-indexed hook (`q_pos[:, :1]`) or a dual-position hook over
    packed documents takes `mma.sync`'s 32-row and 64x32 tiles on a compile-only A100, and each pinned
    lowers both its launches (dQ and dK/dV) through Mosaic's layout solver, now that `_score_logits` holds
    query places in the accumulator's layout."""
    from dew.nn.kernels import tuning
    from dew.nn.kernels.templates import attention_mosaic

    q = jax.ShapeDtypeStruct((1, 512, 4, 64), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((1, 512, 2, 64), jnp.bfloat16)
    zero_row = jax.ShapeDtypeStruct((1, 4, 512), jnp.float32)
    zero_idx = jax.ShapeDtypeStruct((1, 4, 512), jnp.int32)
    split = template.Split(jax.ShapeDtypeStruct((1, 4, 512, 64), jnp.float32),
                           zero_row, zero_row, zero_row)
    residuals = (zero_row, zero_row, zero_idx, split)
    docs = jnp.where(jnp.arange(512, dtype=jnp.int32)[None, :] < 256, 1, 2)
    cases = [
        (template.spans(512, 512, 1, causal=True),
         template.Hooks(score=Hook(lambda s, q_pos, _k: s + 0.01 * q_pos[:, :1].astype(s.dtype)))),
        (template.spans(512, 512, 1, causal=True, segment_ids=docs),
         template.Hooks(score=Hook(lambda s, q_pos, k_pos: s + 0.01 * (q_pos - k_pos).astype(s.dtype)))),
    ]
    pins = [attention_mosaic.BackwardTiles(block_q=rows, block_k=32, stages=2, consumers=1, fused=False,
                                           family="mma.sync") for rows in (32, 64)]
    # Lowered for the A100 itself (its compile-only mesh), where its walks' copies are cp.async's.
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm80")
    with generation.assumed("sm80"), jax.sharding.use_abstract_mesh(mesh):
        target = generation.gpu_target()
        for spans, hooks in cases:
            op, args = attention_mosaic._backward_call(
                q, k, k, spans, hooks, q, residuals=residuals,
                scale=None, natural=attention_mosaic.NATURAL, accurate=True)
            ba = op.bind(*args, interpret=False)
            for pin in pins:
                assert op.takes(ba, pin, target), pin

                def bwd(query, key, value, dout, res, sp=spans, hk=hooks):
                    return attention_mosaic.backward(
                        query, key, value, sp, hk, dout, residuals=res, accurate=True)

                with tuning.pinned({attention_mosaic.MosaicBackward: pin}):
                    jax.jit(bwd).trace(q, k, k, q, residuals).lower(lowering_platforms=("cuda",))
