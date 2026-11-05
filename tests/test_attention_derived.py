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
from reference_error import assert_as_exact, equations
from test_algebra_derive import shaped, walked

from dew.algebra import SATURATION, derive, forms as algebra_forms
from dew.nn.kernels import decode_attention, generation
from dew.nn.kernels.templates import (
    attention as template,
    attention_algebra as algebra,
    attention_backward_algebra,
    attention_portable,
)
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


def oracle_walk(base2: bool, mask: str, split: bool):
    """The template's own forms over the same tiles (`plain_step`, `merge`, `finish`), as a call whose spans
    narrow (`form="spans"`), so a query may see no key (`Forward.guarded`), its logits as the declaration's,
    unit-scaled in their base: a folding call's scale times log2(e) below 1 (`algebra.folds`), so its forms
    exponentiate in base 2 as the declaration's do, the tiles' logits taken as they are."""
    spec = algebra.Forward(form="spans", keys=KEYS, ids=False, scale=0.5 if base2 else 1.0, natural=not base2)
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
            return algebra.Keys(coords, masked, algebra.Operand(tile_values.shape, tile_values.dtype,
                                                                lambda: tile_values), dot)

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


def sliced_walk(mask: str, merged: bool):
    """The template's forms (a folding call's, `oracle_walk`'s) over each key tile as a slice of its own, the
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

        states = [algebra.plain_walk(spec, tile, tile + 1, keys_at,
                                     algebra.started(spec, None, ROWS, WIDTH, None))
                  for tile in range(KEYS // TILE)]
        if merged:
            state = algebra.merged(spec, jax.tree.map(lambda *parts: jnp.stack(parts), *states), 0)
        else:
            state = states[0]
            for other in states[1:]:
                state = algebra.merge(spec, state, other)
        return algebra.finish(spec, state, initial=False)[0]

    return run


@pytest.mark.parametrize("derived", [False, True], ids=["template", "derived"])
@pytest.mark.parametrize("mask", list(MASKS))
def test_slices_merged_at_once_are_on_par_with_their_fold(mask, derived):
    """A split call's slices merged at once (`algebra.merged`: the template's one sum over them, or behind the
    switch the derived forms' merges a level at a time), three of them so a level carries one, a slice of no
    key among them where the mask leaves one: on par with their fold a merge at a time against the float64
    truth (check 1)."""
    natural = np.log(2.0)
    with algebra_forms.derived(derived):
        assert_as_exact(jax.jit(sliced_walk(mask, merged=True)), jax.jit(sliced_walk(mask, merged=False)),
                        draw, f"slices merged at once {mask}", seeds=16,
                        truth_fn=lambda logits, value: attended(logits * natural, MASKS[mask](), value))


@pytest.mark.parametrize("derived", [False, True], ids=["template", "derived"])
def test_a_split_decodes_combine_of_many_slices_traces_about_one_merge(derived):
    """A split decode's combine of 32 slices (`decode_attention._combined`) traces at most three times a
    combine of 4's equations, either forms: a fold traced a merge a slice, which at Dory 4B's depth compiled
    its decode step for minutes."""
    spec = algebra.Forward(form="spans", keys=KEYS, ids=False, scale=0.5, natural=False)

    def size(slices: int) -> int:
        parts = (jax.ShapeDtypeStruct((slices, ROWS, WIDTH), jnp.float32),
                 *[jax.ShapeDtypeStruct((slices, ROWS), jnp.float32)] * 2)
        combined = jax.make_jaxpr(lambda *parts: decode_attention._combined(spec, *parts, axis=0))(*parts)
        return sum(1 for _ in equations(combined.jaxpr))

    with algebra_forms.derived(derived):
        assert size(32) <= 3 * size(4)


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


def derived_backward(form: str, along: str):
    """The backward declaration's derived forms walked `along` key tiles (queries first) or query tiles (keys
    first), from the template forward's residual rows (`forward_rows`): its logits the tile's, -inf where
    unseen, and D about no anchor (its whole `delta`), each step's products the template's CPU mainloop's
    issuer (`attention.interpreted_dot`), scheduled as a mainloop tiles the walk: keys first, a key tile's dK
    and dV the walk's over the query tiles; queries first, each query tile walking the key tiles, its dQ the
    walk's and each key tile's dK and dV its share, summed over the query tiles in order (row 3310: one
    contraction over a key tile's every query at once, no mainloop's schedule, erred 1.63 times the
    template's in dV)."""
    forms = derive.derive(attention_backward_algebra.backward_declaration(scale=SCALE, walked=along),
                          accurate=False, saturation=SATURATION)
    seen = seen_of(form)
    rows_of = forward_rows(form)
    tiles = LENGTH // BLOCK
    by_row = ("q", "dout", "peak", "total", "anchor", "delta", "logits", "seen")

    def tile(values: dict, index: int) -> dict:
        part = slice(index * BLOCK, (index + 1) * BLOCK)

        def part_of(name, value):
            if along == "rows":
                return value[part] if name in by_row else value
            if name in ("logits", "seen"):
                return value[:, part]
            return value[part] if name in ("k", "values") else value

        return {name: part_of(name, value) for name, value in values.items()}

    def walk(values: dict):
        parts = [tile(values, index) for index in range(tiles)]
        mapped = [forms.mapped(part) for part in parts]
        return forms.finish(walked(forms, parts, template.interpreted_dot), values,
                            shapes=shaped(parts[0])), mapped

    def run(q, k, v, d):
        _, peak, total, delta = rows_of(q, k, v, d)
        logits = jnp.where(seen, float32_einsum("qd,kd->qk", q, k) * SCALE, -jnp.inf)
        values = {"logits": logits, "q": q, "k": k, "values": v, "dout": d, "peak": peak, "total": total,
                  "anchor": jnp.zeros_like(delta), "delta": delta, "seen": seen}
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


@pytest.mark.parametrize("walked", ["keys", "rows"], ids=["queries-first", "keys-first"])
@pytest.mark.parametrize("form", list(SPANS))
def test_the_derived_backward_is_on_par_in_every_cell(form, walked):
    """The backward's derived forms on par against the float64 gradients (check 1) in every cell: the backward
    declaration's forms, walked queries first or keys first as a mainloop schedules them
    (`derived_backward`), held to the template's own backward at the same tiles."""
    served, reference = derived_backward(form, walked), oracle_backward(form)
    assert_as_exact(jax.jit(served), jax.jit(reference), backward_draw,
                    f"attention backward {form} {walked}", seeds=16, truth_fn=xla_backward(form))


@contextlib.contextmanager
def fast(derived: bool, forms: str = "fast"):
    """Within the block, the run's `forms` (the fast ones' base-2 fold, or the accurate ones' natural logits,
    each the declaration states), the template's walks the hand-written forms or the derived ones
    (`dew.algebra.forms.derived`)."""
    with (generation.run_precision(dataclasses.replace(generation.numerics(), forms=forms)),
          algebra_forms.derived(derived)):
        yield


CALLS = {"plain": {}, "residuals": {"residuals": True}, "dropped": {"drop": 0.25}}
"""The forward calls the derived walk serves: a call's output, a training call's with its residuals (the
forward's step, merge and finish whatever walk the training step takes), and a dropping call's."""


@pytest.mark.parametrize("derived", [False, True], ids=["hand-written", "derived"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("numerics", ["fast", "accurate"])
@pytest.mark.parametrize("call", list(CALLS))
@pytest.mark.parametrize("form", ["full", "causal"])
def test_the_portable_mainloop_computes_attention_by_either_forms(form, call, numerics, dtype, derived):
    """The plain-JAX mainloop's forward (`attention_portable.attend`) over fp16 operands (a plain walk, of no
    compensation or guard) and fp32 ones (its running sums compensated, its plain walk under the headroom
    guard's), each of which the derived forms serve, under the run's fast forms or its accurate ones (natural
    logits, the largest weight held to 1 and the rescale compensated, which the declaration states), its
    walk's step, merge and finish the hand-written forms or the derived ones behind the switch, the product
    of its weights with the values its own (`Keys.dot`): on par with the reference (check 1) either way, a
    plain call's, a training call's (with its residuals) and a dropping call's."""
    spans = template.spans(LENGTH, LENGTH, 1, **SPANS[form])
    rate = CALLS[call].get("drop")
    residuals = CALLS[call].get("residuals", False)

    def hooks(words):
        return None if rate is None else template.Hooks(drop=template.Dropout(
            words, jnp.arange(1, dtype=jnp.int32), jnp.arange(1, dtype=jnp.int32), rate))

    def ours(q, k, v, words):
        with fast(derived, numerics):
            out = attention_portable.attend(q[None, :, None], k[None, :, None], v[None, :, None], spans,
                                            hooks(words), residuals=residuals, natural=numerics == "accurate",
                                            **TILES)
        return (out[0] if residuals else out)[0, :, 0]

    def reference(q, k, v, words):
        return template.reference(q[None, :, None], k[None, :, None], v[None, :, None], spans,
                                  hooks(words))[0, :, 0]

    def operands(seed):
        rng = np.random.default_rng(seed)
        return (*(jnp.asarray(rng.standard_normal((LENGTH, DEPTH)), dtype) for _ in range(3)),
                jax.random.bits(jax.random.key(seed), (2,), jnp.uint32))

    assert_as_exact(jax.jit(ours), jax.jit(reference), operands,
                    f"portable forward {form} {call} {numerics} {dtype} derived {derived}", seeds=16)


@pytest.mark.parametrize("derived", [False, True], ids=["hand-written", "derived"])
@pytest.mark.parametrize("form", ["full", "causal"])
def test_the_portable_mainloop_differentiates_attention_by_either_forms(form, derived):
    """The plain-JAX mainloop's backward (`attention_portable.backward`) under the run's fast forms, its dQ
    walk over the keys and its dK and dV walks over the queries the hand-written forms or the derived ones
    behind the switch (`dew.algebra.forms.derived`, `attention_backward_algebra.derived_backward`), from the
    forward's residuals and its split D: the gradients on par with the reference's (check 1) either way."""
    spans = template.spans(LENGTH, LENGTH, 1, **SPANS[form])

    def ours(q, k, v, dout):
        with fast(derived):
            grads = attention_portable.backward(q[None, :, None], k[None, :, None], v[None, :, None], spans,
                                                None, dout[None, :, None], **TILES)
        return grads.query[0, :, 0], grads.key[0, :, 0], grads.value[0, :, 0]

    def reference(q, k, v, dout):
        _, pull = jax.vjp(lambda q, k, v: template.reference(q[None, :, None], k[None, :, None],
                                                             v[None, :, None], spans, None)[0, :, 0], q, k, v)
        return pull(dout)

    def operands(seed):
        rng = np.random.default_rng(seed)
        return tuple(jnp.asarray(rng.standard_normal((LENGTH, DEPTH)), jnp.float32) for _ in range(4))

    assert_as_exact(jax.jit(ours), jax.jit(reference), operands,
                    f"portable backward {form} derived {derived}", seeds=16)


def one_key(dtype, query, key, value, dout, *, scale: float = SCALE):
    """A call's operands of one query and `len(key)` keys, each a row of width 8 (`[1, n, 1, 8]`)."""
    def laid(rows):
        rows = np.asarray(rows, np.float64).reshape(len(rows), -1)
        return jnp.asarray(np.pad(rows, ((0, 0), (0, 8 - rows.shape[1])))[None, :, None, :], dtype)

    return laid(query), laid(key), laid(value), laid(dout), {"block_q": 8, "block_k": 8, "pad": 8,
                                                             "scale": scale}


def test_a_derived_backward_of_one_key_gives_an_exactly_zero_query_gradient():
    """A bf16 query of one key, its value `(1, 2^-8)` and its output's cotangent `(1, 1)`: its score
    cotangent `P (dP - D)` is zero, `dP` the reference's `1 + 2^-8` and `D` the output's row sum, so its
    query's gradient is exactly zero, as the reference's is, under the derived backward too: its `dP`
    accumulates in fp32 (one rounded to bf16 is 1, and its cotangent -2^-8)."""
    query, key, value, dout, tiles = one_key(jnp.bfloat16, [[1.0]], [[1.0]], [[1.0, 2.0 ** -8]], [[1.0, 1.0]])
    spans = template.spans(1, 1, 1)
    with fast(derived=True):
        grads = attention_portable.backward(query, key, value, spans, None, dout, **tiles)
    np.testing.assert_array_equal(np.asarray(grads.query, np.float32), 0.0)


def test_a_derived_backward_masks_an_empty_querys_overflowing_product():
    """Two fp32 queries over one key, the second seeing none, the value's first component 2^127 and the second
    query's output cotangent 4 there: its product `dP` is 2^129, past fp32's largest, beside a weight of 0.
    Under the derived backward every gradient is finite, as the reference's are: the cotangents are masked
    where a query sees no key (`softmax_cotangents`, as `grad_scores`), not multiplied by its zero weight."""
    query, key, value, dout, tiles = one_key(jnp.float32, [[0.0], [0.0]], [[0.0]], [[2.0 ** 127]],
                                             [[0.0], [4.0]])
    spans = template.KeySpans(jnp.asarray([[0, 1]], jnp.int32), jnp.asarray([[1, 1]], jnp.int32), None, None,
                              keys=1)
    _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans, None), query, key, value)
    assert all(np.isfinite(np.asarray(grad)).all() for grad in pull(dout))
    with fast(derived=True):
        grads = attention_portable.backward(query, key, value, spans, None, dout, **tiles)
    assert all(np.isfinite(np.asarray(grad)).all() for grad in (grads.query, grads.key, grads.value))


@pytest.mark.parametrize("derived", [False, True], ids=["hand-written", "derived"])
def test_a_query_a_hook_empties_takes_no_cotangent(derived):
    """Two fp32 queries over one key, a score hook hiding it from the second (its logit -inf, its denominator
    0, its output the reference's zero), the value's first component 2^127 and the second query's output
    cotangent 4 there: its `dO Vᵀ` is 2^129, past fp32's largest, beside weights of 0. Every gradient is
    finite under either forms, as the reference's: the query takes no cotangent
    (`attention.launch_arguments`)."""
    query, key, value, dout, tiles = one_key(jnp.float32, [[0.0], [0.0]], [[0.0]], [[2.0 ** 127]],
                                             [[0.0], [4.0]])
    hooks = template.Hooks(score_mod=first_query_only)
    spans = template.spans(2, 1, 1)
    _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans, hooks), query, key, value)
    assert all(np.isfinite(np.asarray(grad)).all() for grad in pull(dout))
    with fast(derived):
        grads = attention_portable.backward(query, key, value, spans, hooks, dout, **tiles)
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
    query, key, value, dout, tiles = one_key(jnp.float32, [[0.0]], [[2.0 ** 127], [-(2.0 ** 127)]],
                                             [[4.0], [-4.0]], [[1.0]], scale=0.25)
    spans = template.spans(1, 2, 1)
    _, pull = jax.vjp(lambda q, k, v: template.reference(q, k, v, spans, None, scale=0.25), query, key, value)
    assert all(np.isfinite(np.asarray(grad)).all() for grad in pull(dout))
    with fast(derived=True):
        grads = attention_portable.backward(query, key, value, spans, None, dout, **tiles)
    assert all(np.isfinite(np.asarray(grad)).all() for grad in (grads.query, grads.key, grads.value))


def test_a_derived_forward_keeps_an_fp16_weight_its_dropout_multiplies_finite():
    """The fp16 witness of the stability suite (tests/test_attention_dropout.py) behind the switch: a rate of
    1 - 2^-16 (a kept weight's multiplier 2^16) with one of two equal-logit keys kept. The kept weight times
    its multiplier is past fp16's largest; its normalized weight times it is not, as the reference's is, and
    the output is finite (the dropping step's scaled parts, which the declaration states none of)."""
    rate = 1.0 - 2.0 ** -16
    seeds = jnp.arange(1 << 16, dtype=jnp.uint32)[:, None, None]
    kept = np.asarray(template.Draw((seeds, jnp.zeros_like(seeds)), jnp.int32(0), jnp.int32(0), rate).keep(
        jnp.zeros((1, 1), jnp.int32), jnp.arange(2, dtype=jnp.int32)[None, :]))
    words = jnp.asarray([next(index for index in range(kept.shape[0]) if int(kept[index, 0].sum()) == 1), 0],
                        jnp.uint32)
    hooks = template.Hooks(drop=template.Dropout(words, jnp.arange(1, dtype=jnp.int32),
                                                 jnp.arange(1, dtype=jnp.int32), rate))
    query, key, value, _, tiles = one_key(jnp.float16, [[0.0]], [[0.0], [0.0]], [[2.0 ** -10] * 8] * 2,
                                          [[0.0]])
    spans = template.spans(1, 2, 1)
    assert np.isfinite(np.asarray(template.reference(query, key, value, spans, hooks))).all()
    with fast(derived=True):
        out = attention_portable.attend(query, key, value, spans, hooks, **tiles)
    assert np.isfinite(np.asarray(out)).all()


def test_a_derived_backward_at_a_scale_of_zero_gives_zero_query_and_key_gradients():
    """A call at a scale of zero (`attention.prescale`'s own case) under the derived backward: its weights
    uniform, its query's and key's gradients zero, as the hand-written backward's are, its value's finite."""
    rng = np.random.default_rng(0)
    query, key, value, dout = (jnp.asarray(rng.standard_normal((1, LENGTH, 1, DEPTH)), jnp.float32)
                               for _ in range(4))
    spans = template.spans(LENGTH, LENGTH, 1)
    tiles = {**TILES, "scale": 0.0}
    with fast(derived=True):
        grads = attention_portable.backward(query, key, value, spans, None, dout, **tiles)
    np.testing.assert_array_equal(np.asarray(grads.query), 0.0)
    np.testing.assert_array_equal(np.asarray(grads.key), 0.0)
    assert np.isfinite(np.asarray(grads.value)).all()


FRESH = """
import dataclasses
import jax, jax.numpy as jnp
from dew.algebra import forms
from dew.nn.kernels import decode_attention, generation
from dew.nn.kernels.templates import attention as template, attention_portable

operands = [jnp.full((1, 16, 1, 8), 0.1 * (index + 1), jnp.float16) for index in range(3)]
fast = dataclasses.replace(generation.numerics(), forms="fast")
with generation.run_precision(fast), forms.derived(True):
    out = attention_portable.attend(*operands, template.spans(16, 16, 1), block_q=8, block_k=8, pad=8,
                                    residuals=True)
print(bool(jnp.all(jnp.isfinite(out[0]))))
"""
"""A fresh process's first derived walk: a training call's forward, its step the training walk's own."""


def test_a_fresh_processs_training_forward_finishes_by_the_derived_forms():
    """A training call's forward (`residuals=True`, its walk the hand-written training step's) as a fresh
    process's first call under the derived forms: its finish takes the derived forms of the state it holds,
    whatever walk came before it in the process (none here), and its output is finite."""
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
    the derived backward the real gradients are finite, as the reference's and the hand-written backward's
    are, each tile's cotangents masked before their products (`pulled`)."""
    rng = np.random.default_rng(0)
    query, key, value, dout = (jnp.asarray(rng.standard_normal((1, 1, 1, 8)), jnp.float32) for _ in range(4))
    spans = template.spans(1, 1, 1)
    temperature = template.Param(jnp.ones((1, 1, 1), jnp.float32), template.Kind.QUERY)
    hooks = template.Hooks(score_mod=tempered, score_params=(temperature,))
    with fast(derived=True):
        grads = attention_portable.backward(query, key, value, spans, hooks, dout, block_q=8, block_k=8,
                                            pad=8)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(grads))


def test_a_split_decodes_states_merge_and_finish_by_the_derived_forms():
    """A split decode's slices' states (`[rows, splits, queries, heads]`, `decode_attention._combined`) of an
    fp16 call under the run's fast forms, merged and finished by the derived forms over every row whatever
    the states' leading axes: on par with the hand-written forms' against float64's merge (check 1)."""
    from dew.nn.kernels import decode_attention

    spec = decode_attention._spec(0.25, 1.0, None, natural=False, dtype=jnp.float16, sinks=False)

    def states(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.standard_normal((2, 3, 4, 2, 16)), jnp.float32),
                jnp.asarray(rng.standard_normal((2, 3, 4, 2)), jnp.float32),
                jnp.asarray(rng.uniform(0.5, 2.0, (2, 3, 4, 2)), jnp.float32))

    def combined(*, derived: bool):
        def run(sums, peaks, totals):
            with fast(derived):
                return decode_attention._combined(spec, sums, peaks, totals, axis=1)
        return run

    def exact(sums, peaks, totals):
        # Base 2: the fold's logits (`spec.base2`).
        weights = jnp.exp2(peaks - jnp.max(peaks, axis=1, keepdims=True))
        return (jnp.sum(weights[..., None] * sums, axis=1)
                / jnp.sum(weights * totals, axis=1)[..., None])

    assert_as_exact(jax.jit(combined(derived=True)), jax.jit(combined(derived=False)), states,
                    "a split decode's combine", seeds=16, truth_fn=exact)
