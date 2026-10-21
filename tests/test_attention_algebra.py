"""The attention template's algebra/mainloop boundary (docs/design/kernel-templates.md §6): one algebra
(`attention_algebra`) under two mainloops, the Triton one interpreted and the plain-JAX one
(`attention_portable`), at the same tiles, gives the same bits: output and residuals, over every call
form and hook. A mainloop that masked, hooked or normalized on its own would part from the other here.
The algebra's own `merge` is held to one pass over the keys it joins."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact
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


RESIDUALS = ("output", "maximum", "denominator", "anchor key")
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
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs), err_msg=f"{form} {name}")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("hooks", HOOKS)
def test_both_mainloops_compute_the_same_bits_under_every_hook(hooks, dtype):
    """Each hook (softcap, ALiBi, sinks, the output gate, all at once) over a windowed causal call of
    packed documents, in both dtypes."""
    triton, portable = both(CASES["causal-documents-window"], HOOKS[hooks], dtype, TILES[1])
    for name, mine, theirs in zip(RESIDUALS, triton, portable, strict=True):
        np.testing.assert_array_equal(np.asarray(mine, np.float32), np.asarray(theirs, np.float32),
                                      err_msg=f"{hooks} {jnp.dtype(dtype).name} {name}")


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
    with tuning.pinned({attention_triton.TritonBackward: config}):
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
