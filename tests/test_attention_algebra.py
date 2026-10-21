"""The attention template's algebra/mainloop boundary (docs/design/kernel-templates.md §6): one algebra
(`attention_algebra`) under two mainloops, the Triton one interpreted and the plain-JAX one
(`attention_portable`), at the same tiles, gives the same bits: output and residuals, over every call
form and hook. A mainloop that masked, hooked or normalized on its own would part from the other here.
The algebra's own `merge` is held to one pass over the keys it joins."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, assert_as_exact_as_the_reference
from test_attention_template import CASES, ROWS, arguments
from test_attention_triton import (
    BACKWARD_FORMS,
    FORMS,
    HOOKS,
    SCORE_HOOKS,
    drawing,
    gate_of,
    geometry,
    hooks_for,
    score_hooks,
)

from dew.nn.kernels import tuning
from dew.nn.kernels.templates import (
    attention as template,
    attention_algebra as algebra,
    attention_portable,
    attention_triton,
)

TILES = (attention_triton.Tiles(32, 32, 4, 2), attention_triton.Tiles(64, 32, 4, 2),
         attention_triton.Tiles(32, 64, 4, 2))
"""The tiles both mainloops walk; a product's rounding depends on its shape, so the bits are compared
at equal tiles."""


RESIDUALS = ("output", "maximum", "denominator", "anchor key", "split sums")
"""The forward's outputs with `residuals`, in order."""


def both(case: dict, names: frozenset, dtype, tiles: attention_triton.Tiles):
    """The Triton mainloop's and the plain-JAX mainloop's output and residuals over `case` and the hooks
    `names`, seed 0."""
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    hooks = hooks_for(names, queries, 32, gate_of(queries, 32))
    query, key, value = drawing(queries, keys, dtype)(0)
    with tuning.pinned({attention_triton.TritonForward: tiles}):
        triton = attention_triton.forward(query, key, value, spans, hooks, residuals=True, interpret=True)
    portable = attention_portable.attend(query, key, value, spans, hooks, block_q=tiles.block_q,
                                         block_k=tiles.block_k, pad=attention_triton.BLOCK, residuals=True)
    return triton, portable


@pytest.mark.parametrize("tiles", TILES, ids=str)
@pytest.mark.parametrize("form", FORMS)
def test_both_mainloops_compute_the_same_bits_over_every_call_form(form, tiles):
    """Each call form's keys, fp32, no hooks."""
    triton, portable = both(CASES[form], HOOKS["none"], jnp.float32, tiles)
    for name, mine, theirs in zip(RESIDUALS, triton, portable, strict=True):
        assert_same_bits(mine, theirs, f"{form} {name}")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("hooks", HOOKS)
def test_both_mainloops_compute_the_same_bits_under_every_hook(hooks, dtype):
    """Each hook (softcap, ALiBi, sinks, the output gate, all at once) over a windowed causal call of
    packed documents, in both dtypes."""
    triton, portable = both(CASES["causal-documents-window"], HOOKS[hooks], dtype, TILES[1])
    for name, mine, theirs in zip(RESIDUALS, triton, portable, strict=True):
        assert_same_bits(mine, theirs, f"{hooks} {jnp.dtype(dtype).name} {name}")


def both_backward(case: dict, names: tuple, dtype, tiles: attention_triton.Tiles):
    """The Triton mainloop's and the plain-JAX mainloop's gradients over `case` and the score hooks
    `names` at `tiles` (both launches'), seed 0."""
    queries, keys = geometry(case)
    spans = template.spans(queries, keys, ROWS, **arguments(case))
    score_mod, kinds, draw_hooks = score_hooks(names, queries, keys, 32)
    query, key, value = drawing(queries, keys, dtype)(0)
    dout = jax.random.normal(jax.random.key(1000), (ROWS, queries, 4, 32), jnp.float32).astype(dtype)
    params, initial = draw_hooks(0)
    hooks = template.Hooks(score_mod, tuple(template.Param(value, kind)
                                            for value, kind in zip(params, kinds, strict=True)), initial)
    config = attention_triton.BackwardTiles(tiles.block_q, tiles.block_k, tiles.warps, tiles.stages,
                                            tiles.warps, tiles.stages)
    # Both mainloops' forwards at the same tiles, whose sums the backwards read (`Split`).
    with tuning.pinned({attention_triton.TritonBackward: config, attention_triton.TritonForward: tiles}):
        triton = attention_triton.backward(query, key, value, spans, hooks, dout, interpret=True)
    portable = attention_portable.backward(query, key, value, spans, hooks, dout, block_q=tiles.block_q,
                                           block_k=tiles.block_k, pad=attention_triton.BLOCK)
    return triton, portable


def assert_same_bits(mine, theirs, label: str):
    """Every leaf of two gradient trees equal to the bit."""
    for (path, a), b in zip(jax.tree_util.tree_leaves_with_path(mine), jax.tree.leaves(theirs), strict=True):
        np.testing.assert_array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32),
                                      err_msg=f"{label} {jax.tree_util.keystr(path)}")


@pytest.mark.parametrize("tiles", TILES[:2], ids=str)
@pytest.mark.parametrize("form", BACKWARD_FORMS)
def test_both_mainloops_compute_the_same_gradients_over_every_call_form(form, tiles):
    """Each call form's gradients of the queries, keys and values, fp32, no hooks."""
    triton, portable = both_backward(CASES[form], (), jnp.float32, tiles)
    assert_same_bits(triton, portable, form)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("hooks", SCORE_HOOKS)
def test_both_mainloops_compute_the_same_gradients_under_every_score_hook(hooks, dtype):
    """Each score hook's gradients (its parameters' and the initial state's too) over a windowed causal
    call of packed documents, in both dtypes."""
    triton, portable = both_backward(CASES["causal-documents-window"], SCORE_HOOKS[hooks], dtype, TILES[1])
    assert_same_bits(triton, portable, f"{hooks} {jnp.dtype(dtype).name}")


@pytest.mark.parametrize("hooks", ["none", "sinks", "softcap"])
def test_the_forwards_split_sums_give_the_first_passes_gradients(hooks):
    """The backward from the forward's split sums (`algebra.split_delta`) and from the dQ launch's own
    first pass over the keys (`algebra.off_anchor_delta`) on par with each other's reference (check 1),
    over a windowed causal call of packed documents: the training forward's sums are the first pass."""
    queries, keys = geometry(CASES["causal-documents-window"])
    spans = template.spans(queries, keys, ROWS, **arguments(CASES["causal-documents-window"]))
    score_mod, kinds, draw_hooks = score_hooks(SCORE_HOOKS[hooks], queries, keys, 32)
    draw_qkv = drawing(queries, keys, jnp.float32)

    def draw(seed):
        dout = jax.random.normal(jax.random.key(seed + 1000), (ROWS, queries, 4, 32), jnp.float32)
        return *draw_qkv(seed), dout, *draw_hooks(seed)

    def grads(*, split: bool):
        def run(query, key, value, dout, params, initial):
            given = tuple(template.Param(value, kind) for value, kind in zip(params, kinds, strict=True))
            hooked = template.Hooks(score_mod, given, initial)
            found = attention_portable.backward(query, key, value, spans, hooked, dout, block_q=32,
                                                block_k=32, pad=attention_triton.BLOCK, split=split)
            return found.query, found.key, found.value, found.score_params, found.initial

        return jax.jit(run)

    assert_as_exact(grads(split=True), grads(split=False), draw, f"split sums {hooks}", seeds=SPLIT_SEEDS)


SPLIT_SEEDS = 8
"""The split sums' site's declared draws (check 1)."""


MERGE_SEEDS = 8
"""The merge site's declared draws (check 1)."""


def test_two_states_merged_are_on_par_with_one_pass_over_their_keys():
    """`merge` of the states over the first and second halves of a row's keys, in order, finished, is on
    par with one pass over all of them (check 1, against float64's), and `merge` with `empty` changes
    no bit."""
    spec = algebra.Forward("full", 64, ids=False, scale=0.125)

    def state(s, v):
        m, alpha = algebra.peak(spec, jnp.full(s.shape[:1], -jnp.inf, s.dtype), s)
        p = algebra.weights(spec, s, m)
        return algebra.Partial(algebra.rescale(jnp.zeros((s.shape[0], v.shape[1]), s.dtype), alpha) + p @ v,
                               m, algebra.total(jnp.zeros(s.shape[:1], s.dtype), alpha, p))

    def draw(seed):
        streams = jax.random.split(jax.random.key(seed), 2)
        return (jax.random.normal(streams[0], (8, 64), jnp.float32) * 4,
                jax.random.normal(streams[1], (64, 16), jnp.float32))

    def merged(s, v):
        joined = algebra.merge(spec, state(s[:, :32], v[:32]), state(s[:, 32:], v[32:]))
        return algebra.finish(spec, joined, initial=False)[0]

    def one_pass(s, v):
        return algebra.finish(spec, state(s, v), initial=False)[0]

    assert_as_exact(jax.jit(merged), jax.jit(one_pass), draw, "merged halves", seeds=MERGE_SEEDS)
    whole = state(*draw(0))
    for mine, theirs in zip(algebra.merge(spec, algebra.empty(8, 16), whole), whole, strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


def forwards(query, key, value, spans, hooks=None, *, residuals=False):
    """The Triton mainloop's forward, interpreted, and the plain-JAX one's, at 32 by 32 tiles."""
    tiles = attention_triton.Tiles(32, 32, 4, 2)
    with tuning.pinned({attention_triton.TritonForward: tiles}):
        triton = attention_triton.forward(query, key, value, spans, hooks, residuals=residuals,
                                          interpret=True)
    portable = attention_portable.attend(query, key, value, spans, hooks, block_q=32, block_k=32,
                                         pad=attention_triton.BLOCK, residuals=residuals)
    return triton, portable


@pytest.mark.parametrize("form", FORMS)
def test_both_mainloops_compute_the_same_bits_where_no_residuals_are_kept(form):
    """An inference forward (no residuals), which keeps no split sums, rounds alike in both mainloops."""
    queries, keys = geometry(CASES[form])
    spans = template.spans(queries, keys, ROWS, **arguments(CASES[form]))
    triton, portable = forwards(*drawing(queries, keys, jnp.float32)(0), spans)
    np.testing.assert_array_equal(np.asarray(triton), np.asarray(portable))


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_a_plain_call_of_no_keys_reads_its_initial_state_or_zero_in_both_mainloops(causal):
    """A full or causal call of no keys walks no padding as keys: zero output from no state, and a resumed
    partial's own `acc / total` (1.5), in both mainloops."""
    spans = template.spans(3, 0, 1, causal=causal)
    query = jnp.ones((1, 3, 2, 32), jnp.float32)
    empty = jnp.zeros((1, 0, 1, 32), jnp.float32)
    for out in forwards(query, empty, empty, spans):
        np.testing.assert_array_equal(np.asarray(out), 0)
    state = template.Initial(jnp.zeros((1, 2, 3)), jnp.full((1, 2, 3), 2.0), jnp.full((1, 2, 3, 32), 3.0))
    for out in forwards(query, empty, empty, spans, template.Hooks(initial=state)):
        np.testing.assert_array_equal(np.asarray(out), 1.5)


@pytest.mark.parametrize("form", ["full", "causal"])
def test_merging_partials_of_no_keys_is_the_identity_whatever_the_form(form):
    """`empty` merged with `empty`, and ahead of a state with keys, under a full or causal spec (whose
    tile loop takes no guard), stays finite and leaves the state as it was."""
    spec = algebra.Forward(form, 64, ids=False, scale=0.125)
    nothing = algebra.merge(spec, algebra.empty(4, 8), algebra.empty(4, 8))
    assert np.isneginf(np.asarray(nothing.m)).all()
    np.testing.assert_array_equal(np.asarray(nothing.acc), 0)
    np.testing.assert_array_equal(np.asarray(nothing.total), 0)
    filled = algebra.Partial(jnp.ones((4, 8)), jnp.full((4,), 2.0), jnp.full((4,), 3.0))
    for mine, theirs in zip(algebra.merge(spec, nothing, filled), filled, strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


@pytest.mark.parametrize("keys", [8, 128])
def test_an_fp16_backward_keeps_a_gradient_whose_scores_are_large_but_cancel(keys):
    """fp16 values of ±1000 over 8 or 128 equally scored keys, under an output cotangent of 1000: each
    score cotangent, ±1000000 unnormalized, is ±125000 normalized over 8 keys (past fp16's range) and
    ±31250 scaled too, and D is 0, so dQ is 0 in both mainloops, finite."""
    spans = template.spans(keys, keys, 1)
    query = jnp.zeros((1, keys, 1, 16), jnp.float16)
    key = jnp.zeros((1, keys, 1, 16), jnp.float16)
    value = jnp.zeros((1, keys, 1, 16), jnp.float16).at[0, :, 0, 0].set(
        jnp.where(jnp.arange(keys) % 2 == 0, 1000.0, -1000.0).astype(jnp.float16))
    dout = jnp.zeros((1, keys, 1, 16), jnp.float16).at[..., 0].set(1000.0)
    with tuning.pinned({attention_triton.TritonBackward: attention_triton.BackwardTiles(32, 32, 4, 2, 4, 2),
                        attention_triton.TritonForward: attention_triton.Tiles(32, 32, 4, 2)}):
        triton = attention_triton.backward(query, key, value, spans, None, dout, interpret=True)
    portable = attention_portable.backward(query, key, value, spans, None, dout, block_q=32, block_k=32,
                                           pad=attention_triton.BLOCK)
    for found in (triton, portable):
        np.testing.assert_array_equal(np.asarray(found.query, np.float32), 0)


def test_a_timed_backward_launch_walks_every_key():
    """The backward's timing operands (`TritonBackward.operands`) give each query every key: its first
    key 0 and one past its last the key count, whichever integer operand (the anchor keys among them)
    comes first."""
    queries, keys = 64, 96
    arrays = {"query": jax.ShapeDtypeStruct((1, queries, 2, 32), jnp.bfloat16),
              "key": jax.ShapeDtypeStruct((1, keys, 1, 32), jnp.bfloat16)}
    specs = [arrays["query"], arrays["key"], arrays["key"], arrays["query"],
             *[jax.ShapeDtypeStruct((1, 2, queries), jnp.float32)] * 4,
             jax.ShapeDtypeStruct((1, 2, queries), jnp.int32),
             jax.ShapeDtypeStruct((1, queries, 2, 32), jnp.float32),
             jax.ShapeDtypeStruct((1, 2, queries, 32), jnp.float32),
             *[jax.ShapeDtypeStruct((1, 2, queries), jnp.float32)] * 3,
             *[jax.ShapeDtypeStruct((1, queries), jnp.int32)] * 2]
    filled = attention_triton.TritonBackward().operands(specs, 0)
    np.testing.assert_array_equal(np.asarray(filled[-2]), 0)
    np.testing.assert_array_equal(np.asarray(filled[-1]), keys)


@pytest.mark.parametrize("peak0, scale, key_size",
                         [(87.0, 1.0, 1.0), (0.0, 2.0 ** -126, 1.0), (0.0, 1.0, 2.0 ** 30),
                          (0.0, 0.25, 2.0 ** 30)],
                         ids=["small weight", "small scale", "large key", "large scaled key"])
def test_an_extreme_fp32_backward_is_as_exact_as_the_reference(peak0, scale, key_size):
    """fp32 at the range's edges, one query over one key resumed from a state holding a denominator of
    2**25, under an output cotangent of 2**100: a weight exp(-87) / 2**25 below fp32's least subnormal,
    a scale of 2**-126 over that denominator, or a key of 2**30 beside the cotangent. Both mainloops
    form each score cotangent as the reference does, its weight over the denominator first, so every
    intermediate stays within the reference's, and dQ is as exact as fp32's reference against
    float64's (check 1): a weight the reference loses below fp32 they lose alike, and no product
    overflows that the reference keeps."""
    spans = template.spans(1, 1, 1)
    query = jnp.zeros((1, 1, 1, 1), jnp.float32)
    key = jnp.full((1, 1, 1, 1), key_size, jnp.float32)
    value = jnp.ones((1, 1, 1, 1), jnp.float32)
    dout = jnp.full((1, 1, 1, 1), 2.0 ** 100, jnp.float32)
    state = template.Initial(jnp.full((1, 1, 1), peak0), jnp.full((1, 1, 1), 2.0 ** 25),
                             jnp.zeros((1, 1, 1, 1)))
    hooks = template.Hooks(initial=state)
    pins = {attention_triton.TritonBackward: attention_triton.BackwardTiles(32, 32, 4, 2, 4, 2),
            attention_triton.TritonForward: attention_triton.Tiles(32, 32, 4, 2)}
    with tuning.pinned(pins):
        triton = attention_triton.backward(query, key, value, spans, hooks, dout, scale=scale,
                                           interpret=True)
    portable = attention_portable.backward(query, key, value, spans, hooks, dout, block_q=32, block_k=32,
                                           pad=attention_triton.BLOCK, scale=scale)

    def reference_dq(dtype):
        def attend(q, k, v):
            given = template.Hooks(initial=jax.tree.map(lambda x: x.astype(dtype), state))
            return template.reference(q, k, v, spans, given, scale=scale)

        operands = (query.astype(dtype), key.astype(dtype), value.astype(dtype))
        return np.asarray(jax.vjp(attend, *operands)[1](dout.astype(dtype))[0])

    theirs = reference_dq(jnp.float32)
    with jax.enable_x64(new_val=True):
        want = reference_dq(jnp.float64)
    for found in (triton, portable):
        got = np.asarray(found.query)
        assert np.isfinite(got).all()
        assert_as_exact_as_the_reference(got, theirs, want, "dq at fp32's edges")


def test_a_logit_past_the_forwards_rounded_maximum_weighs_one_in_the_backward():
    """fp32, one query of 1610613120 over one key of 1, the value and output cotangent 1, scale 1, no
    hooks: the forward's maximum, taken in base 2 and back, rounds 128 below the backward's natural
    logit. Both mainloops weigh that key 1, as the maximum it is, and give the reference's finite
    gradients (dQ = dK = 0, dV = 1)."""
    spans = template.spans(1, 1, 1)
    query = jnp.full((1, 1, 1, 1), 1610613120.0, jnp.float32)
    key = value = dout = jnp.ones((1, 1, 1, 1), jnp.float32)
    pins = {attention_triton.TritonBackward: attention_triton.BackwardTiles(32, 32, 4, 2, 4, 2),
            attention_triton.TritonForward: attention_triton.Tiles(32, 32, 4, 2)}
    with tuning.pinned(pins):
        triton = attention_triton.backward(query, key, value, spans, None, dout, scale=1.0, interpret=True)
    portable = attention_portable.backward(query, key, value, spans, None, dout, block_q=32, block_k=32,
                                           pad=attention_triton.BLOCK, scale=1.0)
    for found in (triton, portable):
        np.testing.assert_array_equal(np.asarray(found.query), 0)
        np.testing.assert_array_equal(np.asarray(found.key), 0)
        np.testing.assert_array_equal(np.asarray(found.value), 1)
