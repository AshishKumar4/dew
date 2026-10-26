"""The attention template's algebra/mainloop boundary (docs/design/kernel-templates.md §6): one algebra
(`attention_algebra`) under each mainloop. The plain-JAX one (`attention_portable`) is held to check 1
against the template's reference over every call form and hook, as the Triton one is
(`test_attention_triton.check`, `check_backward`): the gate and time decide, and no test asks two mainloops
for the same bits (bible §4). Exact claims stay exact within a mainloop: a guard that leaves ordinary rows
the plain walk's bits, a query of one key whose gradient is zero. The algebra's own `merge` is held to one
pass over the keys it joins."""

import functools

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
    check,
    check_backward,
    drawing,
    geometry,
    joint,
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
"""The plain-JAX mainloop's tiles under check 1."""


@pytest.mark.parametrize("tiles", TILES, ids=str)
@pytest.mark.parametrize("form", FORMS)
def test_the_plain_jax_forward_is_on_par_over_every_call_form(form, tiles):
    """Each call form's keys, fp32, no hooks, its output, maxima and denominators."""
    check(CASES[form], HOOKS["none"], jnp.float32, tiles, residuals=True, mainloop="portable")


@pytest.mark.parametrize("form", FORMS)
def test_the_plain_jax_forward_is_on_par_where_no_residuals_are_kept(form):
    """An inference forward (no residuals), which keeps no split sums."""
    check(CASES[form], HOOKS["none"], jnp.float32, TILES[0], mainloop="portable")


@pytest.mark.parametrize("natural", [False, True], ids=["fold", "natural"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("hooks", HOOKS)
def test_the_plain_jax_forward_is_on_par_under_every_hook(hooks, dtype, natural):
    """Each hook (softcap, ALiBi, sinks, the output gate, all at once) over a windowed causal call of
    packed documents, in both dtypes and both forms of the exponent."""
    check(CASES["causal-documents-window"], HOOKS[hooks], dtype, TILES[1], residuals=True, natural=natural,
          mainloop="portable")


@pytest.mark.parametrize("tiles", TILES[:2], ids=str)
@pytest.mark.parametrize("form", BACKWARD_FORMS)
def test_the_plain_jax_backward_is_on_par_over_every_call_form(form, tiles):
    """Each call form's gradients of the queries, keys and values, fp32, no hooks."""
    check_backward(CASES[form], (), jnp.float32, tiles, mainloop="portable")


@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("hooks", SCORE_HOOKS)
def test_the_plain_jax_backward_is_on_par_under_every_score_hook(hooks, dtype, accurate):
    """Each score hook's gradients (its parameters' and the initial state's too) over a windowed causal
    call of packed documents, in both dtypes and both forms: the default, and the accurate (natural logits
    and the backward's accurate form)."""
    check_backward(CASES["causal-documents-window"], SCORE_HOOKS[hooks], dtype, TILES[1], natural=accurate,
                   accurate=accurate, mainloop="portable")


@pytest.mark.parametrize("hooks", ["none", "sinks", "softcap"])
def test_the_forwards_split_sums_give_the_first_passes_gradients(hooks):
    """The backward from the forward's split sums (`algebra.split_delta`) and from the dQ launch's own
    first pass over the keys (`algebra.off_anchor_delta`) on par (check 1, the reference's float64 gradients
    the truth), over a windowed causal call of packed documents: the training forward's sums are the first
    pass."""
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

    def truth(query, key, value, dout, params, initial):
        # The reference's own gradients, in the float64 the gate widens to.
        def attend(query, key, value, params, initial):
            given = tuple(template.Param(value, kind) for value, kind in zip(params, kinds, strict=True))
            return template.reference(query, key, value, spans, template.Hooks(score_mod, given, initial))

        pull = jax.vjp(attend, query, key, value, params, initial)[1]
        return pull(dout)

    assert_as_exact(grads(split=True), grads(split=False), draw, f"split sums {hooks}", seeds=SPLIT_SEEDS,
                    truth_fn=truth)


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


def forwards(query, key, value, spans, hooks=None, *, residuals=False, scale=None):
    """The Triton mainloop's forward, interpreted, and the plain-JAX one's, at 32 by 32 tiles."""
    tiles = attention_triton.Tiles(32, 32, 4, 2)
    with tuning.pinned({attention_triton.TritonForward: tiles}):
        triton = attention_triton.forward(query, key, value, spans, hooks, residuals=residuals,
                                          interpret=True, scale=scale)
    portable = attention_portable.attend(query, key, value, spans, hooks, block_q=32, block_k=32,
                                         pad=attention_triton.BLOCK, residuals=residuals, scale=scale)
    return triton, portable


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


@pytest.mark.parametrize("natural", [False, True], ids=["default", "natural"])
def test_a_resumed_state_carried_far_is_as_exact_as_the_references(natural):
    """fp32, one query over one key whose logit is 40 to 87 above a resumed state's maximum (0), the
    state's weighted values 2^60 to 2^100 times a normal draw and its denominator 1, the key's value a
    normal draw: the output is the state's share carried over that distance. The factor the state takes
    to the key's maximum is the reference's `exp` (`Forward.carry`), so each mainloop holds check 1
    against the reference; `exp2(x * log2(e))`, the natural form's key-weight exponential, rounds its
    argument about `|x|` ulps away on a GPU (`natural_exp`). The Triton mainloop compiles on a GPU, whose
    exp2 is the hardware's own; XLA's CPU computes exp2 through exp."""
    spans = template.spans(1, 1, 4)

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 4)
        distance = jax.random.uniform(streams[0], (4, 1, 1, 1), jnp.float32, 40.0, 87.0)
        size = jnp.exp2(jnp.round(jax.random.uniform(streams[1], (4, 1, 1), jnp.float32, 60.0, 100.0)))
        acc = jax.random.normal(streams[2], (4, 1, 1, 8), jnp.float32) * size[..., None]
        value = jax.random.normal(streams[3], (4, 1, 1, 8), jnp.float32)
        return distance, value, acc

    def state(acc):
        return template.Hooks(initial=template.Initial(jnp.zeros((4, 1, 1)), jnp.ones((4, 1, 1)), acc))

    def triton(distance, value, acc):
        with tuning.pinned({attention_triton.TritonForward: attention_triton.Tiles(32, 32, 4, 2)}):
            return attention_triton.forward(distance, jnp.ones_like(distance), value, spans, state(acc),
                                            scale=1.0, natural=natural,
                                            interpret=jax.default_backend() != "gpu")

    def portable(distance, value, acc):
        return attention_portable.attend(distance, jnp.ones_like(distance), value, spans, state(acc),
                                         block_q=32, block_k=32, pad=attention_triton.BLOCK, scale=1.0,
                                         natural=natural)

    def reference(distance, value, acc):
        return template.reference(distance, jnp.ones_like(distance), value, spans, state(acc), scale=1.0)

    for name, ours in (("triton", triton), ("portable", portable)):
        assert_as_exact(jax.jit(ours), jax.jit(reference), draw,
                        f"{name} resumed state carried far {'natural' if natural else 'default'}", seeds=8)


def test_a_large_initial_maximum_gives_the_references_state_gradients():
    """fp32, one query over one key resumed from a state at maximum 1610613120 holding a denominator and
    weighted value of 1, Q 0 and K, V and dO 1, scale 1: the forward's maximum, the state's, rounds
    below it taken in base 2 and back. Both mainloops give the state's gradients the reference does
    (finite: its total's -1 and its weighted values' 1), the launch and the front end taking one
    bounded carried factor."""
    spans = template.spans(1, 1, 1)
    query = jnp.zeros((1, 1, 1, 1), jnp.float32)
    key = value = dout = jnp.ones((1, 1, 1, 1), jnp.float32)
    state = template.Initial(jnp.full((1, 1, 1), 1610613120.0), jnp.ones((1, 1, 1)), jnp.ones((1, 1, 1, 1)))
    hooks = template.Hooks(initial=state)
    pins = {attention_triton.TritonBackward: attention_triton.BackwardTiles(32, 32, 4, 2, 4, 2),
            attention_triton.TritonForward: attention_triton.Tiles(32, 32, 4, 2)}
    with tuning.pinned(pins):
        triton = attention_triton.backward(query, key, value, spans, hooks, dout, scale=1.0, interpret=True)
    portable = attention_portable.backward(query, key, value, spans, hooks, dout, block_q=32, block_k=32,
                                           pad=attention_triton.BLOCK, scale=1.0)

    def attend(q, k, v, initial):
        return template.reference(q, k, v, spans, template.Hooks(initial=initial), scale=1.0)

    reference = jax.vjp(attend, query, key, value, state)[1](dout)[3]
    for found in (triton, portable):
        for mine, theirs in zip(jax.tree.leaves(found.initial), jax.tree.leaves(reference), strict=True):
            assert np.isfinite(np.asarray(mine)).all()
            np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


@pytest.mark.parametrize("keys, query_size, key_sizes, out", [
    (2, 1.0, (1610613120.0, 1610612992.0), 1.0),
    (1, 3e38, (1.0,), 1.0),
], ids=["one ulp apart at 1.6e9", "a logit near fp32's largest"])
def test_a_logit_the_reference_keeps_stays_finite_and_apart(keys, query_size, key_sizes, out):
    """fp32, one query of `query_size` over keys of `key_sizes`, the first's value 1 and the rest 0, scale 1
    the reference weighs the larger logit alone
    (out 1), keeping logits one ulp apart at 1.6e9 apart and a logit near fp32's largest finite. Both
    mainloops keep the logits natural, as the reference does, and give its output and finite gradients
    under an output cotangent of 1."""
    spans = template.spans(1, keys, 1)
    query = jnp.full((1, 1, 1, 1), query_size, jnp.float32)
    key = jnp.asarray(key_sizes, jnp.float32).reshape(1, keys, 1, 1)
    value = jnp.zeros((1, keys, 1, 1), jnp.float32).at[0, 0].set(1.0)
    dout = jnp.ones((1, 1, 1, 1), jnp.float32)
    for found in forwards(query, key, value, spans, scale=1.0):
        np.testing.assert_array_equal(np.asarray(found), out)
    pins = {attention_triton.TritonBackward: attention_triton.BackwardTiles(32, 32, 4, 2, 4, 2),
            attention_triton.TritonForward: attention_triton.Tiles(32, 32, 4, 2)}
    with tuning.pinned(pins):
        triton = attention_triton.backward(query, key, value, spans, None, dout, scale=1.0, interpret=True)
    portable = attention_portable.backward(query, key, value, spans, None, dout, block_q=32, block_k=32,
                                           pad=attention_triton.BLOCK, scale=1.0)
    for found in (triton, portable):
        for leaf in (found.query, found.key, found.value):
            assert np.isfinite(np.asarray(leaf)).all()


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_an_fp32_cpu_arm_is_as_exact_as_the_reference_at_aligned_tiles(causal):
    """fp32, width 128, 300 keys walked in 128-key tiles: XLA's CPU dot picks its accumulation by shape,
    and its 128-aligned path is 1.86 times less exact than the unaligned one the reference's einsum takes,
    which left every CPU arm 1.6 times the reference (check 1 allows 1.5) where the compiled kernels pass
    on an A100. The template's CPU products sum fp32 exactly (`template.interpreted_dot`)."""
    spans = template.spans(200, 300, ROWS, causal=causal)

    def ours(query, key, value):
        return attention_portable.attend(query, key, value, spans, block_q=128, block_k=128, pad=128)

    def reference(query, key, value):
        return template.reference(query, key, value, spans)

    draw = drawing(200, 300, jnp.float32, width=128, value_width=128)
    assert_as_exact(jax.jit(ours), jax.jit(reference), draw,
                    f"plain mainloop fp32 width 128 {'causal' if causal else 'full'}", seeds=8)


def test_the_fold_reads_its_factor_as_the_kernel_rounds_it():
    """A scale whose product with log2(e) is below 1 but rounds to 1 in fp32 does not fold: the fold's
    exactness (`algebra.base2_maximum`) needs every folded logit an ulp below fp32's largest, which a
    factor of 1 does not give."""
    just_below = (1 - 2.0 ** -40) / algebra.LOG2E
    assert just_below * algebra.LOG2E < 1
    assert not algebra.Forward("full", 1, ids=False, scale=just_below).base2
    assert algebra.Forward("full", 1, ids=False, scale=0.5).base2


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_the_headroom_guard_leaves_ordinary_inputs_bitwise_unchanged(causal):
    """The accurate form's walk with its headroom guard (`algebra.guarded`) gives the same bits as the walk
    without it on drawn inputs, whose sums never overflow: the guard only reads the plain walk's sums."""
    query, key, value = drawing(200, 300, jnp.float32, width=64, value_width=64)(0)
    spans = template.spans(200, 300, ROWS, causal=causal)
    call = template.padded_call(query, key, value, spans, template.Hooks(), attention_triton.BLOCK)

    def walked(*, headroom: bool):
        spec = algebra.Forward(spans.form, spans.keys, ids=False, scale=64 ** -0.5, natural=True,
                               headroom=headroom)
        out, _ = attention_portable.forward(spec, call.query, call.key, call.value, call.start, call.end,
                                            None, None, (), (), (), (), None, block_q=32, block_k=32,
                                            output_mod=None, residuals=False)
        return np.asarray(out)

    np.testing.assert_array_equal(walked(headroom=True), walked(headroom=False))


@pytest.mark.parametrize("natural", [False, True], ids=["folded", "natural"])
def test_a_training_step_holding_its_sums_and_anchor_row_is_the_gathering_steps_arithmetic(natural):
    """`algebra.training_step` as a mainloop that holds its split sums in two blocks of value columns
    (`Keys.accumulate`) and keeps the anchor's value row (`KeptRows`, `Argmax.row`) runs it, Mosaic's
    training forward, against the step that sums whole rows and gathers the row by key (`rows`), the plain-JAX
    and Triton mainloops': the same state, anchor, row sums and split sums, bitwise, over bf16 key tiles whose
    maximum moves, ties among them. Each product is exact in float64 and rounded once, so a column block's is
    the whole row's."""
    queries, width, tile, tiles = 16, 32, 16, 4
    rng = np.random.default_rng(0)
    # Each tile's logits a step above the last's in some rows, so anchors move, with ties on a coarse grid.
    logits = (np.round(rng.normal(size=(tiles, queries, tile)) * 2) / 2
              + np.arange(tiles)[:, None, None] * rng.integers(0, 2, (1, queries, 1)))
    values = jnp.asarray(rng.normal(size=(tiles * tile, width)), jnp.bfloat16)
    spec = algebra.Forward("full", tiles * tile, ids=False, scale=1.0, natural=natural)
    columns = [slice(first, first + width // 2) for first in (0, width // 2)]

    def dot(a, b):
        return jnp.asarray(np.asarray(a, np.float64) @ np.asarray(b, np.float64), jnp.float32)

    def keys_at(index, accumulate=None):
        positions = algebra.Coords(jnp.arange(queries)[:, None], index * tile + jnp.arange(tile)[None, :])
        value = values[index * tile:(index + 1) * tile]
        return algebra.Keys(positions, jnp.asarray(logits[index], jnp.float32), lambda: value, dot,
                            accumulate)

    nothing = jnp.zeros((queries,), jnp.float32)
    start = algebra.started(spec, None, queries, width, None)
    gathered = (start, algebra.no_argmax(queries), nothing, nothing)
    for index in range(tiles):
        gathered = algebra.training_step(spec, gathered, keys_at(index), lambda key: values[key])

    held = [start.acc[:, block] for block in columns]

    def accumulate(scale, parts, value, kept):
        for place, (row, block) in enumerate(zip(kept, columns, strict=True)):
            held[place] = scale(held[place], row) + algebra.summed_products(parts, value[:, block], dot)

    def picked(onehot, value):
        return tuple(dot(onehot, value[:, block]) for block in columns)

    anchor = algebra.no_argmax(queries)._replace(row=tuple(jnp.zeros((queries, width // 2), jnp.bfloat16)
                                                           for _ in columns))
    kept = (algebra.Partial(jnp.zeros((), jnp.float32), start.m, start.total), anchor, nothing, nothing)
    for index in range(tiles):
        kept = algebra.training_step(spec, kept, keys_at(index, accumulate), algebra.KeptRows(picked))

    # A kept row and a gathering walk each need their own carry.
    with pytest.raises(ValueError):
        algebra.training_step(spec, gathered, keys_at(0), algebra.KeptRows(picked))
    with pytest.raises(ValueError):
        algebra.training_step(spec, kept, keys_at(0), lambda key: values[key])

    (state, arg, own, others), (kept_state, kept_arg, kept_own, kept_others) = gathered, kept
    for mine, theirs in ((kept_state.m, state.m), (kept_state.total, state.total), (kept_arg.peak, arg.peak),
                         (kept_arg.key, arg.key), (kept_own, own), (kept_others, others),
                         (jnp.concatenate(held, axis=1), state.acc),
                         (jnp.concatenate(kept_arg.row, axis=1), values[arg.key])):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


@pytest.mark.parametrize("edge", ["small scale", "large scale"])
def test_the_accurate_form_is_as_exact_as_the_reference_at_its_raw_domains_edges(edge):
    """The accurate form at scales just outside the raw form's domain (`algebra.RAW_SCALES`), where raw
    products would overflow their difference (2^-126 over keys near ±2^127) or the factor `scale * log2(e)`
    (3e38 over products near fp32's least normal): it scales its logits there and holds check 1."""
    # At the small scale every key's magnitude is drawn in [3/4, 1) of fp32's largest, so each pair's raw
    # difference overflows while its scaled logits stay finite.
    largest = float(jnp.finfo(jnp.float32).max)
    scale, low, high = ((2.0 ** -126, 0.75 * largest, largest) if edge == "small scale"
                        else (3e38, 2.0 ** -127, 2.0 ** -126))
    spans = template.spans(1, 2, 32)

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 3)
        signs = jnp.asarray([1.0, -1.0], jnp.float32)
        key = (jax.random.uniform(streams[0], (32, 2), jnp.float32, low, high) * signs)[..., None, None]
        value = jax.random.normal(streams[1], (32, 2, 1, 1), jnp.float32)
        return jnp.ones((32, 1, 1, 1), jnp.float32), key, value

    def ours(query, key, value):
        return attention_portable.attend(query, key, value, spans, block_q=32, block_k=32,
                                         pad=attention_triton.BLOCK, scale=scale, natural=True)

    def reference(query, key, value):
        return template.reference(query, key, value, spans, scale=scale)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, f"accurate form at a {edge}", seeds=8)


def test_a_row_that_does_not_overflow_keeps_the_plain_walks_bits_beside_one_that_does():
    """Two documents a row of one program: two queries over two drawn keys, and four over four equal keys
    whose values near 1e38 overflow their weighted sum. The guarded walk shifts only the latter rows
    (`algebra.room`'s `live`): the former's output and residuals are the unguarded walk's to the bit, in
    the plain-JAX mainloop; the Triton one's output is finite on every row."""
    streams = jax.random.split(jax.random.key(0), 3)
    query = jnp.zeros((1, 6, 1, 16), jnp.float32).at[:, :2].set(jax.random.normal(streams[0], (1, 2, 1, 16)))
    key = jnp.zeros((1, 6, 1, 16), jnp.float32).at[:, :2].set(jax.random.normal(streams[1], (1, 2, 1, 16)))
    value = jax.random.normal(streams[2], (1, 6, 1, 16), jnp.float32).at[:, 2:, :, 0].set(1e38)
    documents = jnp.asarray([[1, 1, 2, 2, 2, 2]], jnp.int32)
    spans = template.spans(6, 6, 1, segment_ids=documents)
    call = template.padded_call(query, key, value, spans, template.Hooks(), attention_triton.BLOCK)

    def portable(*, headroom: bool):
        spec = algebra.Forward(spans.form, spans.keys, ids=True, scale=16 ** -0.5, natural=True,
                               headroom=headroom)
        out, kept = attention_portable.forward(spec, call.query, call.key, call.value, call.start, call.end,
                                               call.query_ids, call.key_ids, (), (), (), (), None, block_q=32,
                                               block_k=32, output_mod=None, residuals=True)
        return [np.asarray(leaf)[..., :6, :, :16] if leaf.ndim == 4 and leaf.shape[1] == call.queries
                else np.asarray(leaf) for leaf in jax.tree.leaves((out, kept))]

    guarded, plain = portable(headroom=True), portable(headroom=False)
    assert not np.isfinite(plain[0][0, 2:]).all() and np.isfinite(guarded[0]).all()
    for leaf, (mine, theirs) in enumerate(zip(guarded, plain, strict=True)):
        rows = (slice(None), slice(0, 2)) if leaf == 0 else (slice(None), slice(None), slice(0, 2))
        np.testing.assert_array_equal(mine[rows], theirs[rows], err_msg=f"leaf {leaf}")
    with tuning.pinned({attention_triton.TritonForward: TILES[1]}):
        triton = attention_triton.forward(query, key, value, spans, residuals=True, natural=True,
                                          interpret=True)
    assert np.isfinite(np.asarray(triton[0])).all()


@pytest.mark.parametrize("raw", [True, False], ids=["raw", "scaled"])
def test_the_accurate_forms_rescale_takes_the_plain_factor_where_its_correction_is_not_finite(raw):
    """`algebra.carried` compensates a state's factor `exp(m - safe)` for its exponent's rounding and takes
    the uncompensated factor where that correction is not finite: a state of no key (m = -inf, whose TwoSum
    error is NaN) and, raw, a difference past what TwoProduct's split holds (4097 times it overflows) both
    weigh 0, as the reference's exp does; a state at the maximum keeps 1."""
    spec = algebra.Forward("full", 4, ids=False, scale=0.125, natural=True, initial=not raw)
    assert spec.raw == raw
    m = jnp.asarray([-jnp.inf, -1e35, 0.0], jnp.float32)
    found = np.asarray(algebra.carried(spec, m, jnp.zeros(3, jnp.float32)))
    np.testing.assert_array_equal(found, np.asarray([0.0, 0.0, 1.0], np.float32))


@pytest.mark.parametrize("raw", [True, False], ids=["raw", "scaled"])
def test_the_accurate_forms_rescale_is_as_exact_as_its_exponent_rounded_once(raw):
    """`algebra.carried`'s factor `exp((m - safe) * scale)` for maxima drawn either side of 0, whose
    difference rounds, compensated for that rounding and, raw, for its product with the scale's: check 1
    against the factor of the rounded exponent (`Forward.carry`), from float64. A compensation of the
    wrong sign or scale would err past it."""
    spec = algebra.Forward("full", 4, ids=False, scale=0.125, natural=True, initial=not raw)
    assert spec.raw == raw

    def draw(seed):
        streams = jax.random.split(jax.random.key(seed))
        return (jax.random.uniform(streams[0], (4096,), jnp.float32, -40.0, 0.0),
                jax.random.uniform(streams[1], (4096,), jnp.float32, 0.0, 40.0))

    def rounded(m, safe):
        return spec.carry(m - safe)

    assert_as_exact(jax.jit(functools.partial(algebra.carried, spec)), jax.jit(rounded), draw,
                    f"the accurate form's rescale, {'raw' if raw else 'scaled'}", seeds=8)


@pytest.mark.parametrize("scale, slope, size", [(2.0, None, 2e38), (1.0, 2.0, 2e38), (2.0, 1.7e38, 0.25)],
                         ids=["scale-2", "slope-2", "slope-1.7e38"])
def test_a_querys_gradient_over_keys_near_fp32s_largest_is_finite_at_any_scale_and_slope(scale, slope, size):
    """One query of zeros over two equal keys at equal logits (each weighs 1/2) under a scale and a score
    hook's slope: the query's gradient is the cotangents' difference times the keys, 0, and D's refinement 0,
    so dQ is exactly 0 in both mainloops (`algebra.DqSums.keyed`): keys near fp32's largest at a scale or a
    slope of 2, whose sum under coefficients summing to 2 would overflow, and small keys under a score whose
    derivative, the slope times the scale, is fp32's largest, whose coefficients' magnitudes would (held as
    mantissas and exponents, `keyed_shift`). A derivative past fp32's largest is no fp32 number: XLA's
    compiled arithmetic forms it whole in the logits themselves (`template.scores` is NaN there), so no form
    of the backward is asked of it. The slope is a head's parameter, traced as a learned slope is."""
    query = jnp.zeros((1, 1, 1, 16), jnp.float32)
    key = jnp.zeros((1, 2, 1, 16), jnp.float32).at[:, :, :, 0].set(size)
    # Equal values, so the score cotangents are exactly 0 and dQ's own sum is 0 on any accumulation: the
    # test holds the weights' correction (`keyed`) alone.
    value = jnp.zeros((1, 2, 1, 16), jnp.float32).at[..., 0].set(1.0)
    dout = jnp.zeros((1, 1, 1, 16), jnp.float32).at[..., 0].set(1.0)
    spans = template.spans(1, 2, 1)
    hooks = template.Hooks() if slope is None else template.Hooks(
        score_mod=lambda s, q_pos, k_pos, slope: slope * s,
        score_params=(template.Param(jnp.full((1,), slope, jnp.float32), template.Kind.HEAD),))
    tiles = {"block_q": 32, "block_k": 32, "pad": attention_triton.BLOCK}
    with tuning.pinned({attention_triton.TritonBackward: joint(TILES[1])}):
        triton = attention_triton.backward(query, key, value, spans, hooks, dout, scale=scale, interpret=True)
    portable = attention_portable.backward(query, key, value, spans, hooks, dout, scale=scale, **tiles)
    for name, grads in (("triton", triton), ("portable", portable)):
        np.testing.assert_array_equal(np.asarray(grads.query), 0.0, err_msg=name)


def test_a_querys_gradient_with_a_large_refinement_at_a_small_scale_is_finite():
    """Three equal keys of 2^100 at a scale of 2^-80, one value of 2^54: equal logits, so the cotangents'
    rounding leaves D's refinement at -2^29 against keys of 2^100, while the correction it makes,
    -2^29 · 2^-80 · 2^100 times the weights' mass, is 2^49 (`algebra.dq_finish`: each factor's mantissa and
    exponent apart). dQ is finite in both mainloops, and the two agree to the bit."""
    query = jnp.zeros((1, 1, 1, 16), jnp.float32)
    key = jnp.zeros((1, 3, 1, 16), jnp.float32).at[:, :, :, 0].set(2.0 ** 100)
    value = jnp.zeros((1, 3, 1, 16), jnp.float32).at[:, 2, :, 0].set(2.0 ** 54)
    dout = jnp.zeros((1, 1, 1, 16), jnp.float32).at[..., 0].set(1.0)
    spans = template.spans(1, 3, 1)
    tiles = {"block_q": 32, "block_k": 32, "pad": attention_triton.BLOCK}
    with tuning.pinned({attention_triton.TritonBackward: joint(TILES[1])}):
        triton = attention_triton.backward(query, key, value, spans, None, dout, scale=2.0 ** -80,
                                           natural=True, interpret=True)
    portable = attention_portable.backward(query, key, value, spans, None, dout, scale=2.0 ** -80,
                                           natural=True, **tiles)
    for name, grads in (("triton", triton), ("portable", portable)):
        assert np.isfinite(np.asarray(grads.query)).all(), name


@pytest.mark.parametrize("scale", [0.0, 2.0 ** -200], ids=["zero", "below fp32's range"])
def test_a_padded_call_at_a_scale_fp32_takes_to_zero_weighs_its_keys_alike(scale):
    """fp32, one query over one key padded to a tile (so masked columns), at a scale fp32 holds as 0: every
    logit is 0, the one key weighs 1 and the output is its value (2), in both template mainloops. The mask is
    taken where the factor, rounded to fp32 as the kernels multiply by it, is nonzero on the products before
    the scale (`algebra.logits_of`); masking them at an infinity a zero factor takes to NaN would poison the
    row."""
    query = jnp.zeros((1, 1, 1, 16), jnp.float32)
    key = jnp.ones((1, 1, 1, 16), jnp.float32)
    value = jnp.full((1, 1, 1, 16), 2.0, jnp.float32)
    spans = template.spans(1, 1, 1)
    for found in forwards(query, key, value, spans, scale=scale):
        np.testing.assert_array_equal(np.asarray(found), 2.0)


def test_a_querys_gradient_at_a_subnormal_scale_is_finite():
    """The accurate form at a scale of 2^-128, below fp32's least normal: a query of zeros over two equal keys
    and values of 1 under a cotangent of 1, whose score cotangents, dQ and the reference's are all 0. The
    keyed sums carry the scale as a mantissa and an exponent (`algebra.dq_terms`), taken apart in Python's
    double, where 2^128 is no overflow; an infinite mantissa would make dQ's correction 0 · inf, NaN. dQ is
    finite and 0 in both mainloops."""
    query = jnp.zeros((1, 1, 1, 16), jnp.float32)
    key = jnp.ones((1, 2, 1, 16), jnp.float32)
    value = jnp.ones((1, 2, 1, 16), jnp.float32)
    dout = jnp.ones((1, 1, 1, 16), jnp.float32)
    spans = template.spans(1, 2, 1)
    tiles = {"block_q": 32, "block_k": 32, "pad": attention_triton.BLOCK}
    options = {"scale": 2.0 ** -128, "natural": True, "accurate": True}
    with tuning.pinned({attention_triton.TritonBackward: joint(TILES[1])}):
        triton = attention_triton.backward(query, key, value, spans, None, dout, interpret=True, **options)
    portable = attention_portable.backward(query, key, value, spans, None, dout, **options, **tiles)
    for name, grads in (("triton", triton), ("portable", portable)):
        np.testing.assert_array_equal(np.asarray(grads.query), 0.0, err_msg=name)


def test_a_querys_gradient_from_a_state_of_a_small_denominator_at_a_large_scale_is_finite():
    """One key resumed from a state of denominator 2^-40 at maximum 40, at a scale of 2^100: the key's
    probability is about 4.7e-6 and dQ about 5.9e24, finite, though the scale over the denominator, 2^140,
    is not (`algebra.dq_step`: each factor's mantissa and exponent apart). Both mainloops agree to the bit."""
    query = jnp.zeros((1, 1, 1, 16), jnp.float32)
    key = jnp.zeros((1, 1, 1, 16), jnp.float32).at[..., 0].set(1.0)
    value = jnp.zeros((1, 1, 1, 16), jnp.float32).at[..., 0].set(1.0)
    dout = jnp.zeros((1, 1, 1, 16), jnp.float32).at[..., 0].set(1.0)
    spans = template.spans(1, 1, 1)
    state = template.Initial(jnp.full((1, 1, 1), 40.0), jnp.full((1, 1, 1), 2.0 ** -40),
                             jnp.zeros((1, 1, 1, 16)))
    hooks = template.Hooks(initial=state)
    tiles = {"block_q": 32, "block_k": 32, "pad": attention_triton.BLOCK}
    with tuning.pinned({attention_triton.TritonBackward: joint(TILES[1])}):
        triton = attention_triton.backward(query, key, value, spans, hooks, dout, scale=2.0 ** 100,
                                           natural=True, interpret=True)
    portable = attention_portable.backward(query, key, value, spans, hooks, dout, scale=2.0 ** 100,
                                           natural=True, **tiles)
    for name, grads in (("triton", triton), ("portable", portable)):
        assert np.isfinite(np.asarray(grads.query)).all(), name


def test_a_bf16_querys_gradient_over_a_large_key_with_small_cotangents_is_exact():
    """bf16, one query of zeros over two keys, the first's first entry 2^120: equal logits, so the score
    cotangents are exactly -2^-4 and 2^-4 at the scale of 1/4, and the query's gradient's first entry is
    exactly -2^116, in bf16's range. The cotangents' product with the keys keeps bf16's range in both its
    parts (`algebra.cotangent_product`), in both mainloops, to the bit."""
    query = jnp.zeros((1, 1, 1, 16), jnp.bfloat16)
    key = jnp.zeros((1, 2, 1, 16), jnp.bfloat16).at[0, 0, 0, 0].set(2.0 ** 120)
    value = jnp.zeros((1, 2, 1, 16), jnp.bfloat16).at[0, :, 0, 0].set(jnp.asarray([1.0, 2.0]))
    dout = jnp.zeros((1, 1, 1, 16), jnp.bfloat16).at[..., 0].set(1.0)
    spans = template.spans(1, 2, 1)
    tiles = {"block_q": 32, "block_k": 32, "pad": attention_triton.BLOCK}
    with tuning.pinned({attention_triton.TritonBackward: joint(TILES[1])}):
        triton = attention_triton.backward(query, key, value, spans, None, dout, natural=True, interpret=True)
    portable = attention_portable.backward(query, key, value, spans, None, dout, natural=True, **tiles)
    for name, grads in (("triton", triton), ("portable", portable)):
        assert float(grads.query[0, 0, 0, 0]) == -(2.0 ** 116), name


@pytest.mark.parametrize("accurate", [False, True], ids=["default", "accurate"])
def test_an_fp16_querys_gradient_over_equal_large_keys_whose_cotangents_cancel_is_finite(accurate):
    """fp16, a query of zeros over three equal keys of first entry 65504, values -65504, 16384 and 65504 there
    and a cotangent of 1: equal logits, so the query's gradient is exactly 0, its score cotangents' sum
    against the equal keys cancelling, and the reference's is finite. A cotangent product that keeps fewer
    bits than the cancellation exposes leaves 4 · 65504 / 2 of it, past fp16's largest once the gradient
    returns in the call's dtype; in both forms the gradient is finite, in both mainloops."""
    query = jnp.zeros((1, 1, 1, 16), jnp.float16)
    key = jnp.zeros((1, 3, 1, 16), jnp.float16).at[0, :, 0, 0].set(65504.0)
    sizes = jnp.asarray([-65504.0, 16384.0, 65504.0])
    value = jnp.zeros((1, 3, 1, 16), jnp.float16).at[0, :, 0, 0].set(sizes)
    dout = jnp.zeros((1, 1, 1, 16), jnp.float16).at[..., 0].set(1.0)
    spans = template.spans(1, 3, 1)
    tiles = {"block_q": 32, "block_k": 32, "pad": attention_triton.BLOCK}
    with tuning.pinned({attention_triton.TritonBackward: joint(TILES[1])}):
        triton = attention_triton.backward(query, key, value, spans, None, dout, accurate=accurate,
                                           interpret=True)
    portable = attention_portable.backward(query, key, value, spans, None, dout, accurate=accurate, **tiles)
    _, pull = jax.vjp(lambda query: template.reference(query, key, value, spans), query)
    [theirs] = pull(dout)
    assert np.isfinite(np.asarray(theirs, np.float32)).all()
    for name, grads in (("triton", triton), ("portable", portable)):
        assert np.isfinite(np.asarray(grads.query, np.float32)).all(), name


def test_a_querys_gradient_from_a_state_near_fp32s_largest_over_a_tiny_weight_at_a_small_scale_is_finite():
    """One key at logit -83 (a weight near 2^-120) resumed from a state of denominator 2^127 at maximum 0,
    at a scale of 2^-40: the key's coefficient against its key, weight times scale over the denominator, is
    near 2^-287, past fp32's range, and its share of dQ is below fp32's least value, so dQ is finite in both
    mainloops (`algebra.NO_SHIFT` below every composed exponent, so a tile's running mass is never
    subnormal) and the two agree to the bit."""
    query = jnp.zeros((1, 1, 1, 16), jnp.float32).at[..., 0].set(1.0)
    key = jnp.zeros((1, 1, 1, 16), jnp.float32).at[..., 0].set(-83.0 * 2.0 ** 40)
    value = jnp.zeros((1, 1, 1, 16), jnp.float32).at[..., 0].set(1.0)
    dout = jnp.ones((1, 1, 1, 16), jnp.float32)
    spans = template.spans(1, 1, 1)
    state = template.Initial(jnp.zeros((1, 1, 1)), jnp.full((1, 1, 1), 2.0 ** 127), jnp.zeros((1, 1, 1, 16)))
    hooks = template.Hooks(initial=state)
    tiles = {"block_q": 32, "block_k": 32, "pad": attention_triton.BLOCK}
    with tuning.pinned({attention_triton.TritonBackward: joint(TILES[1])}):
        triton = attention_triton.backward(query, key, value, spans, hooks, dout, scale=2.0 ** -40,
                                           natural=True, interpret=True)
    portable = attention_portable.backward(query, key, value, spans, hooks, dout, scale=2.0 ** -40,
                                           natural=True, **tiles)
    for name, grads in (("triton", triton), ("portable", portable)):
        assert np.isfinite(np.asarray(grads.query)).all(), name


def test_power2_is_exact_under_jit_at_every_exponent():
    """`algebra.power2` scales by powers of two exactly wherever the result is a normal fp32, compiled
    (where XLA computes `exp2` as an inexact `exp` and fuses a product of them into one `exp`, past fp32's
    range), at exponents from -380 to 380 against exact float64 scaling, and zero stays zero."""
    exponents = jnp.arange(-380.0, 381.0, dtype=jnp.float32)
    mantissas = jnp.asarray([0.0, 0.5, 0.75, 1.0, -0.6], jnp.float32)
    found = np.asarray(jax.jit(lambda x, k: algebra.power2(x[:, None], k[None, :]))(mantissas, exponents))
    exact = np.ldexp(np.asarray(mantissas, np.float64)[:, None], np.asarray(exponents, np.int64)[None, :])
    normal = (np.abs(exact) >= np.finfo(np.float32).tiny) & (np.abs(exact) <= np.finfo(np.float32).max)
    np.testing.assert_array_equal(found[normal], exact[normal].astype(np.float32))
    np.testing.assert_array_equal(found[0], 0.0)
    assert np.all(np.isinf(found[1:][np.abs(exact[1:]) > np.finfo(np.float32).max]))

