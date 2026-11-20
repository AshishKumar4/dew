"""The attention template's algebra/mainloop boundary (docs/design/kernel-templates.md §6): one algebra
(`attention_algebra`) under each mainloop. The plain-JAX one (`attention_portable`) is held to check 1
against the template's reference over every call form and hook, as the Triton one is
(`test_attention_triton.check`, `check_backward`): the gate and time decide, and no test asks two mainloops
for the same bits (bible §4). Exact claims stay exact within a mainloop: a scaled walk that leaves ordinary
rows the unscaled walk's bits, a query of one key whose gradient is zero. The algebra's own
`merge` is held to one pass over the keys it joins."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arm_arithmetic import as_written, products_held
from low_precision_support import edge_values
from reference_error import FEWEST_SEEDS, as_given, assert_as_exact
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

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import (
    attention as template,
    attention_algebra as algebra,
    attention_portable,
    attention_triton,
)

TILES = (attention_triton.Tiles(32, 32, 4, 2), attention_triton.Tiles(64, 32, 4, 2),
         attention_triton.Tiles(32, 64, 4, 2))
"""The plain-JAX mainloop's tiles under check 1."""


@pytest.fixture(autouse=True)
def _released_as_each_test_ends():
    """Each test's compiled executables released as it ends (`conftest.release`), where the file's releases
    them at its end alone: a group of its merge cells holding every earlier cell's compiled programs, each a
    jit and an as-written program over its seeds, was killed at 10 GiB on armada."""
    import conftest

    yield
    conftest.release()


@pytest.fixture(autouse=True)
def _interpreted_as_an_a100():
    """An interpreted call in a CPU process states the GPU it emulates (`generation.assumed`): the frozen
    Triton mainloop's tiles and their costs are an A100's, its target's facts, where a CPU has none."""
    if jax.default_backend() != "cpu":
        yield
        return
    with generation.assumed("sm80"):
        yield


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
    """The backward from the forward's split sums (`attention_gradient.split_delta`) and from the dQ launch's
    own first pass over the keys (`attention_gradient.off_anchor_delta`) on par (check 1, the reference's
    float64 gradients the truth), over a windowed causal call of packed documents: the training forward's sums
    are the first pass."""
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
            found = attention_portable.backward(attention_portable.Tiles(32, 32, attention_triton.BLOCK),
                                                query, key, value, spans, hooked, dout, split=split)
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
    portable = attention_portable.xla_mainloop(attention_portable.Tiles(32, 32, attention_triton.BLOCK),
                                               query, key, value, spans, hooks, residuals=residuals,
                                               scale=scale)
    return triton, portable


@pytest.mark.parametrize("peak", [0.0, float(np.finfo(np.float32).min)], ids=["zero", "fp32-least"])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_a_plain_call_of_no_keys_reads_its_initial_state_or_zero_in_both_mainloops(causal, peak):
    """A full or causal call of no keys walks no padding as keys: zero output from no state, and a resumed
    partial's own `acc / total` (1.5), in both mainloops, its maximum 0 or fp32's least, a real state's (a
    hook's mask by fp32's least, as transformers' are) and not one of no keys (row 4486)."""
    spans = template.spans(3, 0, 1, causal=causal)
    query = jnp.ones((1, 3, 2, 32), jnp.float32)
    empty = jnp.zeros((1, 0, 1, 32), jnp.float32)
    for out in forwards(query, empty, empty, spans):
        np.testing.assert_array_equal(np.asarray(out), 0)
    state = template.Initial(jnp.full((1, 2, 3), peak), jnp.full((1, 2, 3), 2.0),
                             jnp.full((1, 2, 3, 32), 3.0))
    for out in forwards(query, empty, empty, spans, template.Hooks(initial=state)):
        np.testing.assert_array_equal(np.asarray(out), 1.5)


def test_a_state_at_fp32s_least_maximum_keeps_its_mass():
    """A state whose maximum is fp32's least, a real one (a hook's mask by fp32's least, as transformers'
    are), merged with a state of no keys keeps its own output and its finished rows its maximum: a row of no
    keys is one of no denominator, never one at fp32's least (row 4486)."""
    least = float(np.finfo(np.float32).min)
    spec = algebra.Forward("spans", 64, ids=False, scale=0.125)
    real = algebra.Partial(jnp.full((4, 8), 3.0), jnp.full((4,), least), jnp.full((4,), 2.0))
    merged = algebra.merge(spec, algebra.empty(4, 8), real)
    out, _, total = algebra.finish(spec, merged, initial=True)
    rows = algebra.finished_rows(spec, real, initial=False)
    np.testing.assert_array_equal(np.asarray(out), 1.5)
    np.testing.assert_array_equal(np.asarray(total), 2.0)
    np.testing.assert_array_equal(np.asarray(rows[0]), spec.naturalized(jnp.full((4,), least)))


@pytest.mark.parametrize("form", ["full", "causal"])
def test_merging_partials_of_no_keys_is_the_identity_whatever_the_form(form):
    """`empty` merged with `empty`, and ahead of a state with keys, under a full or causal spec (whose
    tile loop takes no guard), stays a state of no key, finite, and leaves the state as it was."""
    spec = algebra.Forward(form, 64, ids=False, scale=0.125)
    nothing = algebra.merge(spec, algebra.empty(4, 8), algebra.empty(4, 8))
    assert np.asarray(algebra.no_keys(nothing.total)).all()
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
    portable = attention_portable.backward(attention_portable.Tiles(32, 32, attention_triton.BLOCK), query,
                                           key, value, spans, None, dout)
    for found in (triton, portable):
        np.testing.assert_array_equal(np.asarray(found.query, np.float32), 0)


@pytest.mark.parametrize("peak0, scale, key_size",
                         [(87.0, 1.0, 1.0), (0.0, 2.0 ** -126, 1.0), (0.0, 1.0, 2.0 ** 30),
                          (0.0, 0.25, 2.0 ** 30)],
                         ids=["small weight", "small scale", "large key", "large scaled key"])
def test_an_extreme_fp32_backward_is_as_exact_as_the_reference(peak0, scale, key_size):
    """fp32 at the range's edges, one query over one key resumed from a state holding a denominator of 2**25,
    under an output cotangent of 2**100: a weight exp(-87) / 2**25 below fp32's least subnormal, a scale of
    2**-126 over that denominator, or a key of 2**30 beside the cotangent. Both mainloops form each score
    cotangent as the reference does, its cotangent term before the denominator (the weight exp(-87) / 2**25
    alone would round to 0 where e times the cotangent over 2**25 is 1.9e-15), so every intermediate stays
    within the reference's, and dQ is finite and on par with fp32's reference over the case's draws (check 1,
    `_scaled_draw`): a weight the reference loses below fp32 they lose alike, and no product overflows that
    the reference keeps."""
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
    draw, reference = _scaled_draw(query, key, value, dout), _reference_dq(spans, hooks, scale)

    def triton_dq(query, key, value, dout):
        with tuning.pinned(pins):
            return attention_triton.backward(query, key, value, spans, hooks, dout, scale=scale,
                                             interpret=True).query

    def portable_dq(query, key, value, dout):
        return attention_portable.backward(attention_portable.Tiles(32, 32, attention_triton.BLOCK), query,
                                           key, value, spans, hooks, dout, scale=scale).query

    for name, run in (("triton", triton_dq), ("portable", portable_dq)):
        assert np.isfinite(np.asarray(run(*draw(0)))).all(), name
        assert_as_exact(run, reference, draw, f"dq at fp32's edges, {name}", seeds=GPU_EDGE_SEEDS)


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
    portable = attention_portable.backward(attention_portable.Tiles(32, 32, attention_triton.BLOCK), query,
                                           key, value, spans, None, dout, scale=1.0)
    for found in (triton, portable):
        np.testing.assert_array_equal(np.asarray(found.query), 0)
        np.testing.assert_array_equal(np.asarray(found.key), 0)
        np.testing.assert_array_equal(np.asarray(found.value), 1)


@pytest.mark.parametrize("natural", [False, True], ids=["default", "natural"])
def test_a_resumed_state_carried_far_is_as_exact_as_the_references(natural):
    """fp32, one query over one key whose logit is 40 to 87 above a resumed state's maximum (0), the
    state's weighted values 2^60 to 2^100 times a normal draw and its denominator 1, the key's value a
    normal draw: the output is the state's share carried over that distance. The factor the state takes
    to the key's maximum keeps the residual of its exponent's conversion to base 2 (`Forward.power`,
    `natural_exp`), so each mainloop holds check 1 against the reference's `exp`, where the hardware's base-2
    exponential of `x * log2(e)` rounded is about `|x|` ulps off. The Triton mainloop compiles on a GPU,
    whose exp2 is the hardware's own; XLA's CPU computes exp2 through exp."""
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
        from generation_support import interprets

        with tuning.pinned({attention_triton.TritonForward: attention_triton.Tiles(32, 32, 4, 2)}):
            return attention_triton.forward(distance, jnp.ones_like(distance), value, spans, state(acc),
                                            scale=1.0, natural=natural, interpret=interprets())

    def portable(distance, value, acc):
        return attention_portable.xla_mainloop(attention_portable.Tiles(32, 32, attention_triton.BLOCK),
                                               distance, jnp.ones_like(distance), value, spans, state(acc),
                                               scale=1.0, natural=natural)

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
    portable = attention_portable.backward(attention_portable.Tiles(32, 32, attention_triton.BLOCK), query,
                                           key, value, spans, hooks, dout, scale=1.0)

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
    portable = attention_portable.backward(attention_portable.Tiles(32, 32, attention_triton.BLOCK), query,
                                           key, value, spans, None, dout, scale=1.0)
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
        return attention_portable.xla_mainloop(attention_portable.Tiles(128, 128, 128), query, key, value,
                                               spans)

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


@pytest.mark.parametrize("natural", [False, True], ids=["fold", "natural"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bfloat16"])
def test_the_scaled_walks_tiles_take_the_unscaled_arithmetic_where_no_row_is_scaled(dtype, natural):
    """The scaled walk (`algebra.SCALED_WALK`) over three drawn key tiles, each's maximum past the last's,
    none of whose rows' sums can pass fp32's largest, step for step against the walk with no scale: after
    every tile its sums, maximum and denominator (and an fp32 walk's dropped parts, `algebra.Low`) are the
    same bits, and so is the finished output. The operations run one at a time (`jax.disable_jit`): the
    claim is the arithmetic's, which a compiler's fusion of the walk around its branch does not decide
    (XLA's CPU rounded an fp32 call's sums an ulp apart in a third of its rows where a branch sat beside
    them)."""
    rows, keys, width = 16, 32, 16
    streams = jax.random.split(jax.random.key(0), 3)
    query = jax.random.normal(streams[0], (rows, width)).astype(dtype)
    key = jax.random.normal(streams[1], (3, keys, width)).astype(dtype)
    value = jax.random.normal(streams[2], (3, keys, width)).astype(dtype)
    specs = {headroom: algebra.Forward("full", 3 * keys, ids=False, scale=width ** -0.5, natural=natural,
                                       headroom=headroom, compensated=algebra.compensates(dtype))
             for headroom in (False, True)}

    def dot(a, b):
        return jnp.matmul(a, b, preferred_element_type=jnp.float32)

    with jax.disable_jit():
        states = {headroom: algebra.compensated(spec, algebra.started(spec, None, rows, width))
                  for headroom, spec in specs.items()}
        for tile in range(3):
            coords = algebra.Coords(jnp.arange(rows)[:, None], tile * keys + jnp.arange(keys)[None, :])
            # Each tile's logits past the last's, so every tile moves the maximum.
            scores = dot(query, key[tile].T) + 4.0 * tile
            for headroom, spec in specs.items():
                logits, products = algebra.logits_of(spec, scores, coords)
                tiled = value[tile]
                operand = algebra.Operand(tiled.shape, tiled.dtype, lambda tiled=tiled: tiled)
                states[headroom] = algebra.plain_step(
                    spec, states[headroom], algebra.Keys(coords, logits, operand, dot, products=products))
            for field in ("acc", "m", "total", "low"):
                for mine, theirs in zip(jax.tree.leaves(getattr(states[True], field)),
                                        jax.tree.leaves(getattr(states[False], field)), strict=True):
                    np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs),
                                                  err_msg=f"tile {tile} {field}")
        outputs = [np.asarray(algebra.finish(specs[headroom], algebra.settled(states[headroom]),
                                             initial=False)[0]) for headroom in (False, True)]
    np.testing.assert_array_equal(outputs[1], outputs[0])


@pytest.mark.parametrize("natural", [False, True], ids=["fold", "natural"])
def test_a_scaled_rows_unseen_keys_and_far_logits_weigh_nothing(natural):
    """A row the scaled walk scales (its first key's value near fp32's largest, `algebra.SCALED_WALK`)
    beside keys past its length whose values are near fp32's largest in another column, and keys it sees at
    logits far below its maximum (a weight below 2^-150, which fp32 holds as none): its small weights'
    product (`algebra.at_scale`, raised from each weight's exponent) gives every such key the weight 0,
    so that column of the output is exactly 0 and the first its one value."""
    keys, length = 16, 8
    value = np.zeros((1, keys, 1, 16), np.float32)
    value[0, 0, 0, 0] = 3e38
    value[0, length:, 0, 1] = 3e38
    key = np.zeros((1, keys, 1, 16), np.float32)
    # Every seen key but the first far below it, past fp32's smallest weight; the unseen keys at the first's
    # logit, which their length masks.
    key[0, 1:length, 0, 0] = -1e4
    query = np.zeros((1, 1, 1, 16), np.float32)
    query[0, 0, 0, 0] = 1.0
    spans = template.spans(1, keys, 1, lengths=jnp.asarray([length], jnp.int32))
    out = attention_portable.xla_mainloop(attention_portable.Tiles(16, 16, 16), jnp.asarray(query),
                                          jnp.asarray(key), jnp.asarray(value), spans, natural=natural)
    np.testing.assert_array_equal(np.asarray(out)[0, 0, 0, :2], np.float32([3e38, 0.0]))


def test_a_padded_calls_padding_rows_stay_finite_and_read_zero():
    """A full bf16 call walked past its queries (`Forward.queries`, a mainloop's padding to whole tiles),
    whose padding rows see no key (`seen`): its plain and anchored training walks, scaled
    (`algebra.SCALED_WALK`), leave every row's output finite and the padding rows' 0."""
    queries, rows, keys, width = 20, 32, 32, 16
    spec = algebra.Forward("full", keys, ids=False, scale=width ** -0.5,
                           headroom=algebra.headroom(jnp.bfloat16), queries=queries)
    streams = jax.random.split(jax.random.key(0), 3)
    query, key, value = (jax.random.normal(stream, (count, width), jnp.float32).astype(jnp.bfloat16)
                         for stream, count in zip(streams, (rows, keys, keys), strict=True))
    coords = algebra.Coords(jnp.arange(rows)[:, None], jnp.arange(keys)[None, :])
    logits, products = algebra.logits_of(spec, jnp.matmul(query, key.T, preferred_element_type=jnp.float32),
                                         coords)

    def dot(a, b):
        return jnp.matmul(a, b, preferred_element_type=jnp.float32)

    def accumulate(acc, scale, parts, value):
        return acc + algebra.summed_products(parts, value, dot)

    keys_of = algebra.Keys(coords, logits, algebra.Operand(value.shape, value.dtype, lambda: value), dot,
                           products=products)
    state = algebra.plain_step(spec, algebra.started(spec, None, rows, width), keys_of)
    arg = algebra.argmax(algebra.no_argmax(rows), logits, coords.k_pos)
    start = algebra.anchored(spec, algebra.started(spec, None, rows, width), arg)
    anchored = algebra.anchored_step(spec, start, keys_of._replace(accumulate=accumulate))[0]
    for walked in (state, anchored):
        out = np.asarray(algebra.finish(spec, walked, initial=False)[0])
        assert np.isfinite(out).all()
        np.testing.assert_array_equal(out[queries:], 0.0)


ANCHORED_SEEDS = 8
"""The anchored training walk's site's declared draws (check 1)."""


@pytest.mark.parametrize("natural", [False, True], ids=["folded", "natural"])
def test_a_training_walk_at_its_anchor_known_is_on_par_with_the_gathering_walk(natural):
    """The training walk at each row's anchor and maximum known before it (`algebra.anchored`, from a walk of
    `argmax` over the logits alone; `anchored_step`), as Mosaic's training forward runs it, its split sums in
    the mainloop's accumulator (`Keys.accumulate`), unscaled, against the step that finds the anchor as it
    walks and gathers the old anchor's value row where it moves (`training_step`, the plain-JAX and Triton
    mainloops'): the same maxima and anchor keys, and its split sums, own weight and row sums on par (check 1
    against their float64 values), over bf16 key tiles whose maximum moves, ties among them."""
    queries, width, tile, tiles = 16, 32, 16, 4
    spec = algebra.Forward("full", tiles * tile, ids=False, scale=1.0, natural=natural)

    def draw(seed):
        rng = np.random.default_rng(seed)
        # Each tile's logits a step above the last's in some rows, so anchors move, ties on a coarse grid.
        logits = (np.round(rng.normal(size=(tiles, queries, tile)) * 2) / 2
                  + np.arange(tiles)[:, None, None] * rng.integers(0, 2, (1, queries, 1)))
        values = rng.normal(size=(tiles * tile, width))
        return jnp.asarray(logits, jnp.float32), jnp.asarray(values, jnp.bfloat16)

    def dot(a, b):
        return jnp.matmul(a, b, preferred_element_type=jnp.promote_types(jnp.result_type(a, b), jnp.float32))

    def keys_at(logits, values, index, accumulate=None):
        positions = algebra.Coords(jnp.arange(queries)[:, None], index * tile + jnp.arange(tile)[None, :])
        value = values[index * tile:(index + 1) * tile]
        values = algebra.Operand(value.shape, value.dtype, lambda: value)
        return algebra.Keys(positions, logits[index], values, dot, accumulate)

    def walk(logits, values, *, anchored: bool):
        nothing = jnp.zeros((queries,), logits.dtype)
        start = algebra.started(spec, None, queries, width)
        start = start._replace(acc=start.acc.astype(logits.dtype), m=start.m.astype(logits.dtype),
                               total=start.total.astype(logits.dtype))
        arg = algebra.Argmax(nothing - jnp.inf, jnp.zeros((queries,), jnp.int32))

        def gathered_rows(key):
            return values[key]

        def accumulated(acc, scale, parts, value):
            assert scale is None
            return acc + algebra.summed_products(parts, value, dot)

        if anchored:
            for index in range(tiles):
                arg = algebra.argmax(arg, logits[index], index * tile + jnp.arange(tile)[None, :])
            carry = algebra.anchored(spec, start, arg)
            for index in range(tiles):
                carry = algebra.anchored_step(spec, carry, keys_at(logits, values, index, accumulated))
        else:
            carry = (start, arg, nothing, nothing)
            for index in range(tiles):
                carry, _ = algebra.training_step(spec, carry, keys_at(logits, values, index), gathered_rows)
        state, arg, own, others_total = carry
        return state.acc, state.total, own, others_total, state.m, arg.peak, arg.key

    def truth(logits, values):
        # The walk's sums in float64, whose arithmetic assumes fp32 (`natural_exp`'s split product): each
        # key's weight against the keys' largest logit, the anchor the first key holding it.
        flat = jnp.moveaxis(logits, 0, 1).reshape(queries, tiles * tile)
        e = (jnp.exp if natural else jnp.exp2)(flat - flat.max(axis=-1, keepdims=True))
        at = jnp.arange(tiles * tile)[None, :] == jnp.argmax(flat, axis=-1)[:, None]
        others = jnp.where(at, 0.0, e)
        return others @ values, e.sum(axis=-1), jnp.where(at, e, 0.0).sum(axis=-1), others.sum(axis=-1)

    anchored, gathered = (jax.jit(functools.partial(walk, anchored=known)) for known in (True, False))
    logits, values = draw(0)
    # The maxima and anchor keys are the logits' own, exact in either walk.
    for mine, theirs in zip(anchored(logits, values)[4:], gathered(logits, values)[4:], strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))
    assert_as_exact(lambda *drawn: anchored(*drawn)[:4], lambda *drawn: gathered(*drawn)[:4], draw,
                    f"anchored training walk {'natural' if natural else 'folded'}", seeds=ANCHORED_SEEDS,
                    truth_fn=truth)


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
        return attention_portable.xla_mainloop(attention_portable.Tiles(32, 32, attention_triton.BLOCK),
                                               query, key, value, spans, scale=scale, natural=True)

    def reference(query, key, value):
        return template.reference(query, key, value, spans, scale=scale)

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, f"accurate form at a {edge}", seeds=8)


def test_a_row_that_does_not_overflow_keeps_the_plain_walks_bits_beside_one_that_does():
    """Two documents a row of one program: two queries over two drawn keys, and four over four equal keys
    whose values near 1e38 overflow their weighted sum. The scaled walk scales only the latter rows
    (`algebra.SCALED_WALK`, each row's own bound): the former's output and residuals are the unscaled walk's
    to the bit, in the plain-JAX mainloop, though their program's tiles take the scaled arithmetic; the
    Triton one's output is finite on every row."""
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

    scaled, plain = portable(headroom=True), portable(headroom=False)
    assert not np.isfinite(plain[0][0, 2:]).all() and np.isfinite(scaled[0]).all()
    for leaf, (mine, theirs) in enumerate(zip(scaled, plain, strict=True)):
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
    against the factor of the rounded exponent (`Forward.power`), from float64. A compensation of the
    wrong sign or scale would err past it."""
    spec = algebra.Forward("full", 4, ids=False, scale=0.125, natural=True, initial=not raw)
    assert spec.raw == raw

    def draw(seed):
        streams = jax.random.split(jax.random.key(seed))
        return (jax.random.uniform(streams[0], (4096,), jnp.float32, -40.0, 0.0),
                jax.random.uniform(streams[1], (4096,), jnp.float32, 0.0, 40.0))

    def rounded(m, safe):
        return spec.power(m - safe)

    def exact(m, safe):
        # In float64 the exponent's difference and its product with the scale are exact.
        return jnp.exp((m - safe) * (spec.scale if raw else 1.0))

    assert_as_exact(jax.jit(functools.partial(algebra.carried, spec)), jax.jit(rounded), draw,
                    f"the accurate form's rescale, {'raw' if raw else 'scaled'}", seeds=8, truth_fn=exact)


EDGE_SEEDS = {"a small denominator": 256, "a tiny weight over a large state": 8}
"""An edge case's declared check-1 seeds (`_scaled_draw`), sized by `seeds_for` paired on armada; a case
the reference holds exactly (`_accurate_dq`) or a GPU runs takes `GPU_EDGE_SEEDS`."""
GPU_EDGE_SEEDS = 32


def _scaled_draw(query, key, value, dout):
    """Check 1's draws of an edge case: its operands with the values and the output cotangent each scaled by
    one factor drawn in [1, 2), which every quantity linear in them follows and no equality among a value's
    entries, logit or weight moves, so the case stays at its edge at every seed."""
    def draw(seed):
        factors = np.random.default_rng([0xED6E, seed]).uniform(1.0, 2.0, 2)
        return query, key, (value * factors[0]).astype(value.dtype), (dout * factors[1]).astype(dout.dtype)
    return draw


def _reference_dq(spans, hooks, scale):
    """The reference's dQ of a draw (`template.reference`), the state's hooks in the operands' dtype."""
    def dq(query, key, value, dout):
        def attend(q, k, v):
            given = jax.tree.map(lambda x: x.astype(q.dtype) if isinstance(x, jax.Array) else x, hooks)
            return template.reference(q, k, v, spans, given, scale=scale)
        return jax.vjp(attend, query, key, value)[1](dout)[0]
    return dq


def _accurate_dq(query, key, value, spans, hooks, dout, scale, name):
    """The accurate form's dQ (`natural=True, accurate=True`: the keyed sums, each factor's mantissa and
    exponent apart, `gradient.dq_finish`) on the Triton mainloop interpreted and the plain-JAX one: finite,
    and each on par with the reference's own fp32 dQ over the case's draws (`_scaled_draw`, check 1, the
    truth its float64 run); where the reference's is float64's exactly, which check 1 cannot weigh, each
    float64's exactly too. The Triton mainloop's first draw's dQ and float64's."""
    tiles = {"block_q": 32, "block_k": 32, "pad": attention_triton.BLOCK}
    form = {"scale": scale, "natural": True, "accurate": True}

    def triton(query, key, value, dout):
        with tuning.pinned({attention_triton.TritonBackward: joint(TILES[1])}):
            return attention_triton.backward(query, key, value, spans, hooks, dout, interpret=True,
                                             **form).query

    def portable(query, key, value, dout):
        return attention_portable.backward(attention_portable.Tiles(**tiles), query, key, value, spans, hooks,
                                           dout, **form).query

    draw, reference = _scaled_draw(query, key, value, dout), _reference_dq(spans, hooks, scale)
    found = np.asarray(triton(*draw(0)))
    assert np.isfinite(found).all(), name
    theirs = np.asarray(jax.jit(reference)(*draw(0)))
    with jax.enable_x64(new_val=True):
        want = np.asarray(reference(*(jnp.asarray(x, jnp.float64) for x in draw(0))))
    if np.array_equal(theirs, want):
        for arm, ours in (("triton", found), ("portable", np.asarray(portable(*draw(0))))):
            np.testing.assert_array_equal(ours, want, err_msg=f"{name}, {arm}")
    else:
        for arm, run in (("triton", triton), ("portable", portable)):
            assert_as_exact(run, reference, draw, f"{name}, {arm}",
                            seeds=EDGE_SEEDS.get(name, GPU_EDGE_SEEDS))
    return found, want


@pytest.mark.parametrize("scale, slope, size", [(2.0, None, 2e38), (1.0, 2.0, 2e38), (2.0, 1.7e38, 0.25)],
                         ids=["scale-2", "slope-2", "slope-1.7e38"])
def test_a_querys_gradient_over_keys_near_fp32s_largest_is_finite_at_any_scale_and_slope(scale, slope, size):
    """One query of zeros over two equal keys at equal logits (each weighs 1/2) under a scale and a score
    hook's slope: the query's gradient is the cotangents' difference times the keys, 0, and D's refinement 0,
    so the accurate form's dQ is exactly 0 in both mainloops, as float64's
    (`attention_gradient.DqSums.keyed`): keys near fp32's largest at a scale or a slope of 2, whose sum under
    coefficients summing to 2 would overflow, and small keys under a score whose derivative, the slope times
    the scale, is fp32's largest, whose coefficients' magnitudes would (held as mantissas and exponents,
    `keyed_shift`). A derivative past fp32's largest is no fp32 number: XLA's compiled arithmetic forms it
    whole in the logits themselves (`template.scores` is NaN there), so no form of the backward is asked of
    it. The slope is a head's parameter, traced as a learned slope is."""
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
    found, want = _accurate_dq(query, key, value, spans, hooks, dout, scale, f"keys of {size}")
    np.testing.assert_array_equal(found, 0.0)
    np.testing.assert_array_equal(want, 0.0)


def test_a_querys_gradient_with_a_large_refinement_at_a_small_scale_is_finite():
    """Three equal keys of 2^100 at a scale of 2^-80, one value of 2^54: equal logits, so the cotangents'
    rounding leaves D's refinement at -2^29 against keys of 2^100, while the correction it makes,
    -2^29 · 2^-80 · 2^100 times the weights' mass, is 2^49 (`attention_gradient.dq_finish`: each factor's
    mantissa and exponent apart). The accurate form's dQ is finite in both mainloops, the two agree to the
    bit, and it is on par with the reference's against float64's."""
    query = jnp.zeros((1, 1, 1, 16), jnp.float32)
    key = jnp.zeros((1, 3, 1, 16), jnp.float32).at[:, :, :, 0].set(2.0 ** 100)
    value = jnp.zeros((1, 3, 1, 16), jnp.float32).at[:, 2, :, 0].set(2.0 ** 54)
    dout = jnp.zeros((1, 1, 1, 16), jnp.float32).at[..., 0].set(1.0)
    spans = template.spans(1, 3, 1)
    _accurate_dq(query, key, value, spans, template.Hooks(), dout, 2.0 ** -80, "a large refinement")


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
    keyed sums carry the scale as a mantissa and an exponent (`attention_gradient.dq_terms`), taken apart in
    Python's double, where 2^128 is no overflow; an infinite mantissa would make dQ's correction 0 · inf, NaN.
    dQ is finite and 0 in both mainloops."""
    query = jnp.zeros((1, 1, 1, 16), jnp.float32)
    key = jnp.ones((1, 2, 1, 16), jnp.float32)
    value = jnp.ones((1, 2, 1, 16), jnp.float32)
    dout = jnp.ones((1, 1, 1, 16), jnp.float32)
    spans = template.spans(1, 2, 1)
    tiles = {"block_q": 32, "block_k": 32, "pad": attention_triton.BLOCK}
    options = {"scale": 2.0 ** -128, "natural": True, "accurate": True}
    with tuning.pinned({attention_triton.TritonBackward: joint(TILES[1])}):
        triton = attention_triton.backward(query, key, value, spans, None, dout, interpret=True, **options)
    portable = attention_portable.backward(attention_portable.Tiles(**tiles), query, key, value, spans, None,
                                           dout, **options)
    for name, grads in (("triton", triton), ("portable", portable)):
        np.testing.assert_array_equal(np.asarray(grads.query), 0.0, err_msg=name)


def test_a_querys_gradient_from_a_state_of_a_small_denominator_at_a_large_scale_is_finite():
    """One key resumed from a state of denominator 2^-40 at maximum 40, at a scale of 2^100: the key's
    probability is about 4.7e-6 and dQ about 5.9e24, finite, though the scale over the denominator, 2^140,
    is not (`attention_gradient.dq_issued`: each factor's mantissa and exponent apart). The accurate form's dQ
    agrees to the bit in both mainloops and is on par with the reference's against float64's."""
    query = jnp.zeros((1, 1, 1, 16), jnp.float32)
    key = jnp.zeros((1, 1, 1, 16), jnp.float32).at[..., 0].set(1.0)
    value = jnp.zeros((1, 1, 1, 16), jnp.float32).at[..., 0].set(1.0)
    dout = jnp.zeros((1, 1, 1, 16), jnp.float32).at[..., 0].set(1.0)
    spans = template.spans(1, 1, 1)
    state = template.Initial(jnp.full((1, 1, 1), 40.0), jnp.full((1, 1, 1), 2.0 ** -40),
                             jnp.zeros((1, 1, 1, 16)))
    hooks = template.Hooks(initial=state)
    _accurate_dq(query, key, value, spans, hooks, dout, 2.0 ** 100, "a small denominator")


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
    portable = attention_portable.backward(attention_portable.Tiles(**tiles), query, key, value, spans, None,
                                           dout, natural=True)
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
    portable = attention_portable.backward(attention_portable.Tiles(**tiles), query, key, value, spans, None,
                                           dout, accurate=accurate)
    _, pull = jax.vjp(lambda query: template.reference(query, key, value, spans), query)
    [theirs] = pull(dout)
    assert np.isfinite(np.asarray(theirs, np.float32)).all()
    for name, grads in (("triton", triton), ("portable", portable)):
        assert np.isfinite(np.asarray(grads.query, np.float32)).all(), name


def test_a_querys_gradient_from_a_state_near_fp32s_largest_over_a_tiny_weight_at_a_small_scale_is_finite():
    """One key at logit -83 (a weight near 2^-120) resumed from a state of denominator 2^127 at maximum 0, at
    a scale of 2^-40: the key's coefficient against its key, weight times scale over the denominator, is near
    2^-287, past fp32's range, and its share of dQ is below fp32's least value, so the accurate form's dQ is
    finite in both mainloops, on par with the reference's against float64's (`attention_gradient.NO_SHIFT`
    below every composed exponent, so a tile's running mass is never subnormal) and the two agree to the
    bit."""
    query = jnp.zeros((1, 1, 1, 16), jnp.float32).at[..., 0].set(1.0)
    key = jnp.zeros((1, 1, 1, 16), jnp.float32).at[..., 0].set(-83.0 * 2.0 ** 40)
    value = jnp.zeros((1, 1, 1, 16), jnp.float32).at[..., 0].set(1.0)
    dout = jnp.ones((1, 1, 1, 16), jnp.float32)
    spans = template.spans(1, 1, 1)
    state = template.Initial(jnp.zeros((1, 1, 1)), jnp.full((1, 1, 1), 2.0 ** 127), jnp.zeros((1, 1, 1, 16)))
    hooks = template.Hooks(initial=state)
    _accurate_dq(query, key, value, spans, hooks, dout, 2.0 ** -40, "a tiny weight over a large state")


def test_a_quotient_below_the_least_normal_is_correctly_rounded_on_a_flushing_cpu():
    """`algebra.quotient` on XLA's CPU, which flushes a subnormal result (row 6302's finish): weighted values
    across fp32's subnormals and least normals over totals from 1 to 3 * 2^121, each quotient it raises (its
    weighted value over 124 binades under its total) bitwise the correctly rounded fp32 of its exact ratio,
    the constructed tie `0x00c00002 / 3` among them (a second rounding's one-ulp miss), and every other one
    bitwise XLA's own division."""
    values = edge_values(np.float32)
    totals = np.asarray([1.0, 3.0, 7.0, 2.0 ** 20 + 1, 2.0 ** 60, 3.0 * 2.0 ** 121], np.float32)
    acc = np.concatenate([values, np.asarray([0x00C00002], np.uint32).view(np.float32)])
    rows = np.broadcast_to(acc, (len(totals), len(acc)))
    on_cpu = jax.devices("cpu")[0]
    arguments = (jax.device_put(rows, on_cpu), jax.device_put(totals, on_cpu))

    def divided(acc, total):
        return algebra.quotient(lambda held: held / total[:, None], acc, total)

    found = np.asarray(jax.jit(divided)(*arguments))
    plain = np.asarray(jax.jit(lambda acc, total: acc / total[:, None])(*arguments))
    # A ratio of 24-bit significands lies over 2^-49 off any fp32 midpoint, so its float64 rounding rounds
    # to fp32 as the exact ratio does.
    exact = (rows.astype(np.float64) / totals[:, None].astype(np.float64)).astype(np.float32)
    raised = (rows != 0) & (np.frexp(rows)[1] - np.frexp(totals)[1][:, None] < -124)
    np.testing.assert_array_equal(found[raised].view(np.uint32), exact[raised].view(np.uint32))
    np.testing.assert_array_equal(found[~raised].view(np.uint32), plain[~raised].view(np.uint32))


@pytest.mark.parametrize(("totals", "outputs", "share"), [((1.0, 1.0, 1.0), (0.0, 0.0, 1.0), 1.0 / 3.0),
                                                          ((128.0, 64.0), (1.0, 0.0), 2.0 / 3.0)],
                         ids=["one-key-each", "denominators-128-64"])
def test_finished_slices_merged_at_a_large_maximum_keep_each_ones_mass(totals, outputs, share):
    """Finished slices at the natural maximum 2^30, their denominators `totals` and outputs `outputs`,
    merged (`algebra.finished_merged`, the merge of a decode step's slices): each keeps its share of the
    mass, so the merged output is `share`. Folding a
    denominator into a maximum that large rounds the fold away (2^30 + log 2 is 2^30 in fp32): folded by a
    merge, three slices of one key each left the first two one slice's mass, the output 1/2 (row 4526's
    review); folded by each slice's state, slices of 128 and 64 merged half and half (row 4646)."""
    spec = algebra.finished_form(algebra.Forward("full", 3, ids=False, scale=1.0, natural=True))
    states = [algebra.slice_state(spec, jnp.full((1, 1), out, jnp.float32), jnp.full((1,), 2.0 ** 30),
                                  jnp.full((1,), total, jnp.float32))
              for out, total in zip(outputs, totals, strict=True)]
    stacked = algebra.Partial(*(jnp.stack(parts) for parts in zip(*(state[:3] for state in states),
                                                                    strict=True)))
    merged = algebra.finished_merged(spec, stacked, axis=0)
    out = algebra.finish(spec, merged, initial=True, normalized=True)[0]
    # The heaviest slice's share of the mass rounded once: the output fp32's `share` exactly.
    assert out.item() == np.float32(share), out.item()


def test_a_merged_state_landed_past_fp32s_largest_moves_its_maximum_and_keeps_its_mass():
    """Two slices of output 0 and denominator 2^127 at the natural maximum 2^30, merged
    (`algebra.finished_merged`): their mass, 2^128, passes fp32's largest, so their state lands at the next
    maximum fp32 holds past 2^30, 2^30 + 128 (fp32's spacing there), its mass taken by e^-128; merged again
    with a slice of output 1 and denominator 2^127 at 2^30, the output is 1/3. The landed mass and the
    output on par with the plain fp32 combination of each slice's mass at the moved maximum, exp(log(T) -
    128), against float64 (check 1), where a maximum moved by log 2 rounded back to 2^30 and the state lost
    half its mass: the output was 1/2 (the base review of 7d8b5e062, F2; of e97ed082d, F4: masses of 2^126
    landed at 2^30 unmoved)."""
    spec = algebra.finished_form(algebra.Forward("full", 3, ids=False, scale=1.0, natural=True))

    def state(out, total):
        return algebra.slice_state(spec, jnp.full((1, 1), out, jnp.float32), jnp.full((1,), 2.0 ** 30),
                                   jnp.reshape(total, (1,)).astype(jnp.float32))

    def merged(*states):
        stacked = algebra.Partial(*(jnp.stack(parts) for parts in zip(*(each[:3] for each in states),
                                                                        strict=True)))
        return algebra.finished_merged(spec, stacked, axis=0)

    totals = jnp.full((3,), 2.0 ** 127, jnp.float32)
    pair = merged(state(0.0, totals[0]), state(0.0, totals[1]))
    assert pair.m.item() == np.float32(2.0 ** 30 + 128), pair.m.item()

    def ours(totals):
        pair = merged(state(0.0, totals[0]), state(0.0, totals[1]))
        out = algebra.finish(spec, merged(pair, state(1.0, totals[2])), initial=True, normalized=True)[0]
        return jnp.reshape(pair.total, (1, 1)), jnp.reshape(out, (1, 1))

    def plain(totals):
        masses = jnp.exp(jnp.log(totals) - 128)
        return (jnp.reshape(masses[0] + masses[1], (1, 1)),
                jnp.reshape(masses[2] / jnp.sum(masses), (1, 1)))

    _on_par_jitted_and_op_by_op(ours, plain, lambda seed: (totals,),
                                "a merged state landed past fp32's largest", seeds=FEWEST_SEEDS)


LARGEST = float(np.finfo(np.float32).max)


def _finished_merged_output(outputs, maxima, totals):
    """Finished slices of one row each, their outputs, maxima (natural units) and denominators, merged as a
    decode step's slices merge (`algebra.finished_merged`) and finished normalized: the merged output."""
    spec = algebra.finished_form(algebra.Forward("full", 3, ids=False, scale=1.0, natural=True))
    count = len(outputs)
    states = [algebra.slice_state(spec, jnp.full((1, 1), outputs[i], jnp.float32),
                                  jnp.full((1,), maxima[i], jnp.float32),
                                  jnp.full((1,), totals[i], jnp.float32))
              for i in range(count)]
    stacked = algebra.Partial(*(jnp.stack(parts) for parts in zip(*(state[:3] for state in states),
                                                                    strict=True)))
    return algebra.finish(spec, algebra.finished_merged(spec, stacked, axis=0), initial=True,
                          normalized=True)[0]


@pytest.mark.parametrize(("outputs", "maxima", "totals"), [
    ((LARGEST, LARGEST), (0.0, 0.0), (29.0, 1.0)),
    ((LARGEST, -LARGEST), (0.0, 0.0), (3.0, 1.0)),
    ((2.0 ** 127, -(2.0 ** 127)), (0.0, 0.0), (1.0, 1.0)),
    ((1.0, 0.0), (-25.0 * np.log(2.0), 0.0), (1.0, 1.0)),
], ids=["largest-29-1", "largest-opposite-3-1", "opposite-2^127", "far-apart"])
def test_finished_slices_merge_within_their_outputs_at_fp32s_largest(outputs, maxima, totals):
    """Finished slices merged (`algebra.finished_merged`, a split decode step's merge): the merged output, a
    convex combination of theirs, lies between them, however its sum rounds, and is theirs where they are
    one value: outputs at fp32's largest whose shares 29/30 and 1/30 summed to inf, outputs of opposite
    signs there and at 2^127, and shares 2^-25 apart. Its derivative as to a denominator is finite where its
    value is (L/8 at outputs +-fp32's largest of shares 3:1, which a reverse pass accumulating each weight's
    cotangent by its weight overflowed, row 6827)."""
    found = np.asarray(_finished_merged_output(outputs, maxima, totals)).item()
    assert min(outputs) <= found <= max(outputs), found
    assert len(set(outputs)) > 1 or found == outputs[0], found
    gradient = jax.grad(lambda first: _finished_merged_output(outputs, maxima, (first, *totals[1:])).sum())(
        jnp.float32(totals[0]))
    assert np.isfinite(np.asarray(gradient)), gradient


CANCELLING_SEEDS = 16
"""The cancelling merge's declared draws (check 1)."""


def test_slices_that_cancel_near_fp32s_largest_merge_on_par_beside_a_least_normal_one():
    """Three finished slices of equal mass merged (`algebra.finished_merged`): two of outputs near +-2^127
    that cancel and one whose share of its output is near fp32's least normal, so the merged output is that
    third slice's share: on par with the plain fp32 convex combination against float64 (check 1). Summing
    the whole combination at half its size, as the largest outputs' terms are summed, would drop the third's
    last bits below the normal range (0 where the truth is 2^-126); summing the large terms' products
    unrounded, as a compiled sum did, left one product's rounding (3.6e30, row 6909)."""

    @_reaches_subnormals
    def draw(seed):
        streams = jax.random.split(jax.random.key(seed), 2)
        large = 2.0 ** 127 * (1 + jax.random.uniform(streams[0], (), jnp.float32))
        tiny = 3.0 * 2.0 ** -126 * (1 + jax.random.uniform(streams[1], (), jnp.float32))
        return jnp.stack([large, -large, tiny]).astype(jnp.float32), jnp.ones(3, jnp.float32)

    def merged(outputs, totals):
        return _finished_merged_output(outputs, jnp.zeros(3), totals)

    def plain(outputs, totals):
        return jnp.sum(totals / jnp.sum(totals) * outputs)[None, None]

    _on_par_jitted_and_op_by_op(merged, plain, draw, "slices cancelling beside a least normal one",
                                seeds=CANCELLING_SEEDS)


VALUE_SEEDS = 512
"""The merge of masses past fp32's largest's declared draws (check 1)."""


def _masses_past_fp32s_largest(seed):
    """Sixteen slices of masses near 2^124, whose total passes fp32's largest, under outputs near 1 and 2 by
    turns: a mean near 1.5, which shares formed over the total mass made 0 (their denominator inf). Each
    mass's reciprocal is normal, as XLA:CPU divides by a broadcast through its reciprocal (`_shares`)."""
    streams = jax.random.split(jax.random.key(seed), 2)
    totals = 2.0 ** 124 * (1 + 0.5 * jax.random.uniform(streams[0], (16,), jnp.float32))
    outputs = jnp.asarray([1.0, 2.0] * 8, jnp.float32) * (1 + 0.25 * jax.random.uniform(streams[1], (16,)))
    return outputs.astype(jnp.float32), totals


def _reaches_subnormals(draw):
    """`draw`, marked as one whose arithmetic reaches fp32's subnormals (`_on_par_jitted_and_op_by_op`)."""
    draw.reaches_subnormals = True
    return draw


@_reaches_subnormals
def _masses_near_fp32s_largest(seed):
    """Two slices of masses from 2^126 to 2^127 under outputs near 1: shares over the largest, which XLA:CPU
    divides by through its reciprocal, there below fp32's least normal (row 6904; the base review of
    7746edbd6, F1: masses of 2^127 merged to 0 where the mean is 1)."""
    streams = jax.random.split(jax.random.key(seed), 2)
    totals = 2.0 ** 126 * (1 + jax.random.uniform(streams[0], (2,), jnp.float32))
    outputs = 1 + 0.25 * jax.random.uniform(streams[1], (2,), jnp.float32)
    return outputs.astype(jnp.float32), totals.astype(jnp.float32)


def _scaled_masses(totals, *others):
    """`totals` and `others` scaled exactly (`jnp.ldexp`) by the power of two that takes the largest of
    `totals` into [1/2, 1): a reference's shares that divide by no mass, as XLA:CPU divides by a broadcast
    through its reciprocal (openxla/xla#51148), past 2^126 below fp32's least normal, even op by op (a
    jnp division is a program of its own)."""
    exponent = jnp.frexp(jnp.max(totals))[1]
    return tuple(jnp.ldexp(x, -exponent) for x in (totals, *others))


def _on_par_jitted_and_op_by_op(ours, reference, draw, label: str, *, seeds: int) -> None:
    """`ours` on par with `reference` against float64 (check 1), jitted and as written: its compiler
    rewrites a quotient divided again into one over the divisors' product, a division by a broadcast into a
    product with its reciprocal and a sum of products into one of unrounded products, so each order is held
    as written and as compiled. As written is op by op (`as_given`, which the gate runs uncompiled) where the
    draw reaches fp32's subnormals (`_reaches_subnormals`), which XLA's CPU keeps one primitive at a time and
    flushes in a compiled program (row 4023) — grouped 16 independent draws per seed through eager `jax.vmap`
    so every primitive still runs one at a time without compiling while cutting Python dispatch 16x (rows
    10136, 10146); compiled with no rewrite, its products held from any fused multiply-add
    (`arm_arithmetic.as_written`, `products_held`, its own program run as given) for every other draw, where
    op by op compiled each primitive of the merge's windows apart (the cotangent cell 6.81 GiB op by op, 1.28
    jitted). The reference runs op by op, so none of those rewrites reaches the bar it sets (row 6909: a
    compiled reference summing products shared Dew's error)."""
    reaches = getattr(draw, "reaches_subnormals", False)
    if reaches and seeds >= 32:
        group = 16
        base_draw = draw

        def draw(seed, _draw=base_draw, _group=group):
            return jax.tree.map(lambda *xs: jnp.stack(xs), *(_draw(seed * _group + i) for i in range(_group)))

        ours, reference, seeds = jax.vmap(ours), jax.vmap(reference), seeds // group
    written = as_given(ours if reaches else jax.jit(products_held(ours), compiler_options=as_written()))
    for program, run in ((jax.jit(ours), "jitted"), (written, "as written")):
        assert_as_exact(program, as_given(reference), draw, f"{label} {run}", seeds=seeds)


@pytest.mark.parametrize("draw", [_masses_past_fp32s_largest, _masses_near_fp32s_largest],
                         ids=["sixteen-near-2^124", "two-near-2^127"])
def test_a_merge_of_masses_near_fp32s_largest_is_on_par(draw):
    """Finished slices merged (`algebra.finished_merged`, at one maximum) of masses near fp32's largest: the
    merged output on par with the fp32 convex combination of their outputs weighed by their masses over the
    largest first, against float64 (check 1), where shares over the total were 0 past fp32's largest and
    shares over a largest past 2^126 flushed (row 6904)."""

    def merged(outputs, totals):
        return _finished_merged_output(outputs, jnp.zeros_like(totals), totals)

    def plain(outputs, totals):
        shares, = _scaled_masses(totals)
        return (jnp.sum(shares * outputs) / jnp.sum(shares))[None, None]

    _on_par_jitted_and_op_by_op(merged, plain, draw,
                                f"a merge of masses near fp32's largest, {draw.__name__}", seeds=VALUE_SEEDS)



def test_an_eager_merge_compiles_its_law_once():
    """Finished slices merged op by op (`algebra.finished_merged`, no jit) at a second draw of the same
    shapes compile nothing: the law's primitives hold the walk's units as a parameter, a value equal at an
    equal spec, where a factor made afresh each call keyed each eager merge anew, compiled it again and JAX's
    eager cache held every executable (8 MiB a call, a test's process past 12 GiB)."""
    from jax._src import dispatch

    first, second = (_masses_near_fp32s_largest(seed) for seed in (0, 1))
    jax.block_until_ready(_finished_merged_output(first[0], jnp.zeros_like(first[1]), first[1]))
    compiled = []

    def heard(event: str, duration: float, **values) -> None:
        del duration, values
        if event == dispatch.BACKEND_COMPILE_EVENT:
            compiled.append(event)

    jax.monitoring.register_event_duration_secs_listener(heard)
    try:
        jax.block_until_ready(_finished_merged_output(second[0], jnp.zeros_like(second[1]), second[1]))
    finally:
        jax.monitoring.unregister_event_duration_listener(heard)
    assert not compiled, len(compiled)

DERIVATIVE_SEEDS = 256
"""The merge derivative's sites' declared draws (check 1), each sized by `seeds_for` paired on armada."""


def _opposite_near_largest(seed):
    """Two slices of outputs of opposite signs within a factor 2 of fp32's largest, denominators 1 to 4."""
    streams = jax.random.split(jax.random.key(seed), 3)
    sizes = LARGEST * (1 - 0.5 * jax.random.uniform(streams[0], (2,), jnp.float32))
    totals = 1 + 3 * jax.random.uniform(streams[1], (2,), jnp.float32)
    return (sizes * jnp.asarray([1.0, -1.0], jnp.float32)).astype(jnp.float32), totals


@_reaches_subnormals
def _cancelling_beside_least_normal(seed):
    """Two slices of outputs near +-2^127 that cancel beside one whose output is near fp32's least normal."""
    streams = jax.random.split(jax.random.key(seed), 3)
    large = 2.0 ** 127 * (1 + jax.random.uniform(streams[0], (), jnp.float32))
    tiny = 3.0 * 2.0 ** -126 * (1 + jax.random.uniform(streams[1], (), jnp.float32))
    totals = 1 + jax.random.uniform(streams[2], (3,), jnp.float32)
    return jnp.stack([large, -large, tiny]).astype(jnp.float32), totals


def _subunit_masses_of_alternate_signs(seed):
    """Eight slices of outputs of alternate signs near 2^127, denominators near 1/8: a mean near zero over a
    total mass near 1, its derivatives near 2^127, where a gap over the largest mass alone reached 2^129
    (row 6886)."""
    streams = jax.random.split(jax.random.key(seed), 2)
    sizes = 2.0 ** 127 * (1 + 0.25 * jax.random.uniform(streams[0], (8,), jnp.float32))
    totals = 0.125 * (1 + 0.25 * jax.random.uniform(streams[1], (8,), jnp.float32))
    return (sizes * jnp.asarray([1.0, -1.0] * 4, jnp.float32)).astype(jnp.float32), totals


@_reaches_subnormals
def _subunit_masses_near_least_normal(seed):
    """Two slices of outputs of opposite signs near fp32's least normal, denominators near 1/8: a mean near
    zero over a total mass near 1/4, its derivatives near 6 x 2^-126, where a gap over the summed count
    first fell into the subnormals a CPU flushes (row 6894)."""
    streams = jax.random.split(jax.random.key(seed), 2)
    sizes = 1.5 * 2.0 ** -126 * (1 + 0.25 * jax.random.uniform(streams[0], (2,), jnp.float32))
    totals = 0.125 * (1 + 0.25 * jax.random.uniform(streams[1], (2,), jnp.float32))
    return (sizes * jnp.asarray([1.0, -1.0], jnp.float32)).astype(jnp.float32), totals


ALONG_SEEDS = 512
"""The merge derivative's forward sites' declared draws (check 1), each sized by `seeds_for` paired on
armada."""


def _subunit_masses_under_large_outputs(seed):
    """Two slices of masses near 2^-20 under outputs of opposite signs near 2^120: coefficients (o - out) / W
    near 2^139, past fp32's largest, where the derivative along masses near theirs is near 2^116."""
    streams = jax.random.split(jax.random.key(seed), 2)
    sizes = 2.0 ** 120 * (1 + 0.25 * jax.random.uniform(streams[0], (2,), jnp.float32))
    totals = 2.0 ** -20 * (1 + 0.25 * jax.random.uniform(streams[1], (2,), jnp.float32))
    return (sizes * jnp.asarray([1.0, -1.0], jnp.float32)).astype(jnp.float32), totals


@pytest.mark.parametrize("draw", [_opposite_near_largest, _cancelling_beside_least_normal,
                                  _subunit_masses_of_alternate_signs, _subunit_masses_near_least_normal],
                         ids=["opposite-near-largest", "cancelling-beside-least-normal",
                              "subunit-masses-of-alternate-signs", "subunit-masses-near-least-normal"])
def test_the_merges_derivative_as_to_its_denominators_is_on_par(draw):
    """The derivative of finished slices' merge (`algebra.finished_merged`, at one maximum) as to each
    slice's denominator, (o_j - out) / T, on par with the mean's own derivative formed in fp32 as
    o_j / T - out / T against float64 (check 1): slices of opposite signs near fp32's largest, whose sum's
    reverse pass overflowed (L/8 returned as inf, row 6827), slices cancelling near 2^127 beside a
    least-normal one, whose gaps halved would lose its bits, and eight of subunit masses under outputs of
    alternate signs near 2^127, whose gaps over the largest mass passed fp32 before their sum (row 6886),
    and two of subunit masses under outputs near fp32's least normal, whose gaps over the summed count first
    flushed to zero (row 6894)."""

    def ours(outputs, totals):
        return jax.grad(lambda each: _finished_merged_output(outputs, jnp.zeros_like(totals), each).sum())(
            totals)

    def reference(outputs, totals):
        total = jnp.sum(totals)
        return outputs / total - jnp.sum(totals / total * outputs) / total

    _on_par_jitted_and_op_by_op(ours, reference, draw, f"the merge's derivative, {draw.__name__}",
                                seeds=DERIVATIVE_SEEDS)


@pytest.mark.parametrize("draw", [_opposite_near_largest, _subunit_masses_of_alternate_signs,
                                  _subunit_masses_under_large_outputs, _masses_past_fp32s_largest,
                                  _masses_near_fp32s_largest],
                         ids=["opposite-near-largest", "subunit-masses-of-alternate-signs",
                              "subunit-masses-under-large-outputs", "masses-past-fp32s-largest",
                              "masses-near-fp32s-largest"])
def test_the_merges_derivative_along_its_denominators_is_on_par(draw):
    """The derivative of finished slices' merge (`algebra.finished_merged`, at one maximum) along its
    denominators, every other one's direction an eighth of it, on par with the mean's own formed in fp32 as
    (sum(d o / L) - out sum(d / L)) over the masses' sum over L, a power of two near their largest
    (`_scaled_masses`), against float64 (check 1): the reverse pass's sites whose outputs reach `HALVED`
    (row 6827, 6886), outputs near 2^120 over masses near 2^-20, whose coefficients passed fp32 where the
    derivative did not (NaN where it is near 2^116, row 6905), and masses near and past fp32's largest
    (row 6904)."""

    def along(totals):
        return totals * (0.125 * (jnp.arange(totals.shape[0]) % 2 == 0))

    def ours(outputs, totals):
        return jax.jvp(lambda each: _finished_merged_output(outputs, jnp.zeros_like(totals), each), (totals,),
                       (along(totals),))[1]

    def reference(outputs, totals):
        shares, directions = _scaled_masses(totals, along(totals))
        out = jnp.sum(shares * outputs) / jnp.sum(shares)
        return ((jnp.sum(directions * outputs) - out * jnp.sum(directions)) / jnp.sum(shares))[None, None]

    _on_par_jitted_and_op_by_op(ours, reference, draw, f"the merge's derivative along, {draw.__name__}",
                                seeds=ALONG_SEEDS)


def _opposite_near_largest_over_unequal_masses(seed):
    """Two slices of outputs of opposite signs within a quarter of fp32's largest, masses near 1 and 1/4:
    the second's mass coefficient (o - out) / W near -1.28 times fp32's largest, past it."""
    streams = jax.random.split(jax.random.key(seed), 2)
    sizes = LARGEST * (1 - 0.25 * jax.random.uniform(streams[0], (2,), jnp.float32))
    totals = jnp.asarray([1.0, 0.25], jnp.float32) * (1 + 0.25 * jax.random.uniform(streams[1], (2,)))
    return (sizes * jnp.asarray([1.0, -1.0], jnp.float32)).astype(jnp.float32), totals.astype(jnp.float32)


def _along_one_to_two(draw):
    """`draw`'s slices with directions from 1 to 2 along their outputs."""
    def drawn(seed):
        outputs, totals = draw(seed)
        return outputs, totals, jnp.linspace(1.0, 2.0, outputs.shape[0], dtype=jnp.float32)

    drawn.__name__ = draw.__name__
    return drawn


@_reaches_subnormals
def _tiny_and_huge_masses_along_the_tiny(seed):
    """Two slices of masses near 2^-100 and 2^100 under outputs near 0, a direction near 2^100 along the
    first's output: its share near 2^-200, past fp32's least subnormal, where its term is near 2^-100 (the
    base review of 6f09a9627, F1: 0 where it is 2^-100)."""
    streams = jax.random.split(jax.random.key(seed), 3)
    totals = jnp.asarray([2.0 ** -100, 2.0 ** 100], jnp.float32) * (1 + jax.random.uniform(streams[0], (2,)))
    outputs = jax.random.uniform(streams[1], (2,), jnp.float32, -1.0, 1.0)
    first = 2.0 ** 100 * (1 + jax.random.uniform(streams[2], ()))
    return outputs, totals.astype(jnp.float32), jnp.zeros(2, jnp.float32).at[0].set(first)


@pytest.mark.parametrize("draw", [_along_one_to_two(_opposite_near_largest_over_unequal_masses),
                                  _along_one_to_two(_subunit_masses_of_alternate_signs),
                                  _tiny_and_huge_masses_along_the_tiny],
                         ids=["opposite-near-largest-over-unequal-masses",
                              "subunit-masses-of-alternate-signs", "tiny-and-huge-masses-along-the-tiny"])
def test_the_merges_derivative_along_its_outputs_is_on_par(draw):
    """The derivative of finished slices' merge (`algebra.finished_merged`, at one maximum) along drawn
    directions of its outputs alone, on par with sum(w d) / sum(w) formed in fp32, against float64 (check
    1): its masses' tangents inactive form no term, where a zero times a mass coefficient past fp32's
    largest was NaN (the base review of 7746edbd6, F2: NaN where the derivative is 1), and a share below
    fp32's least subnormal is never formed where its term is normal (the base review of 6f09a9627, F1)."""

    def ours(outputs, totals, direction):
        return jax.jvp(lambda each: _finished_merged_output(each, jnp.zeros_like(totals), totals), (outputs,),
                       (direction,))[1]

    def reference(outputs, totals, direction):
        return (jnp.sum(totals * direction) / jnp.sum(totals))[None, None]

    _on_par_jitted_and_op_by_op(ours, reference, draw,
                                f"the merge's derivative along its outputs, {draw.__name__}",
                                seeds=ALONG_SEEDS)


MERGE_EDGE_SEEDS = 512
"""The merge derivative's and the decode merge's sites at fp32's edges' declared draws (check 1), each sized
by `seeds_for` paired on armada."""


SECOND_ORDER_SEEDS = 256
"""The merge's mixed second derivative's declared draws (check 1), sized by `seeds_for` paired on armada:
each order's two jvps over its draws pass the 120 s group target at the first-order sites' 512."""


def _equal_outputs_along_large_directions(seed):
    """Two slices of masses near 1 under equal outputs near 0 (a zero gap), masses along [2^60, 0] and outputs
    along [2^67, 0]: their mixed second derivative near 2^125, which a written chain's autodiff through
    `jnp.frexp` at the zero gap passed fp32's largest before its 2^-3 (the base review of 67b7edd06, F1:
    inf where it is 2^125)."""
    streams = jax.random.split(jax.random.key(seed), 4)
    totals = 1 + 0.25 * jax.random.uniform(streams[0], (2,), jnp.float32)
    outputs = jnp.full(2, jax.random.uniform(streams[1], (), jnp.float32, -1.0, 1.0))
    masses = jnp.zeros(2, jnp.float32).at[0].set(2.0 ** 60 * (1 + jax.random.uniform(streams[2], ())))
    along = jnp.zeros(2, jnp.float32).at[0].set(2.0 ** 67 * (1 + jax.random.uniform(streams[3], ())))
    return outputs, totals.astype(jnp.float32), masses, along


@_reaches_subnormals
def _a_subnormal_gap_along_units(seed):
    """Two slices of masses near 1 under outputs 0 and a subnormal near 2^-130, masses along [1, 0] and
    outputs along [0, 1]: a gap below fp32's least normal, where autodiff of `jnp.frexp` multiplied by
    2^-e = inf (train 934's review: NaN where the mixed second derivative is near -1/4)."""
    streams = jax.random.split(jax.random.key(seed), 4)
    totals = 1 + 0.25 * jax.random.uniform(streams[0], (2,), jnp.float32)
    outputs = jnp.zeros(2, jnp.float32).at[1].set(2.0 ** -130 * (1 + jax.random.uniform(streams[1], ())))
    masses = jnp.zeros(2, jnp.float32).at[0].set(1 + jax.random.uniform(streams[2], ()))
    along = jnp.zeros(2, jnp.float32).at[1].set(1 + jax.random.uniform(streams[3], ()))
    return outputs, totals.astype(jnp.float32), masses, along


@pytest.mark.parametrize("draw", [_equal_outputs_along_large_directions, _a_subnormal_gap_along_units],
                         ids=["equal-outputs-along-large-directions", "a-subnormal-gap-along-units"])
def test_the_merges_mixed_second_derivative_is_on_par(draw):
    """The mixed second derivative of finished slices' merge (`algebra.finished_merged`, at one maximum) along
    drawn directions of its masses and its outputs, in both orders, on par with sum(dw (do - d out)) / W
    formed in fp32 over the masses' power of two (`_scaled_masses`), d out = sum(w do) / W, against float64
    (check 1): every order of the mean's derivative holds as its first does (the base review of 67b7edd06,
    F1: inf where it is 2^125; train 934's review: NaN at a subnormal gap where it is near -1/4)."""

    def ours(outputs, totals, masses, along):
        def merged(outputs, totals):
            return _finished_merged_output(outputs, jnp.zeros_like(totals), totals)

        masses_then_outputs = jax.jvp(lambda o: jax.jvp(lambda w: merged(o, w), (totals,), (masses,))[1],
                                      (outputs,), (along,))[1]
        outputs_then_masses = jax.jvp(lambda w: jax.jvp(lambda o: merged(o, w), (outputs,), (along,))[1],
                                      (totals,), (masses,))[1]
        return jnp.stack([masses_then_outputs, outputs_then_masses]).reshape(1, 2)

    def reference(outputs, totals, masses, along):
        shares, moved = _scaled_masses(totals, masses)
        moving = jnp.sum(shares * along) / jnp.sum(shares)
        second = jnp.sum(moved * (along - moving)) / jnp.sum(shares)
        return jnp.stack([second, second]).reshape(1, 2)

    _on_par_jitted_and_op_by_op(ours, reference, draw,
                                f"the merge's mixed second derivative, {draw.__name__}",
                                seeds=SECOND_ORDER_SEEDS)


def _unit_masses_along_two(seed):
    """Eight slices of masses near 1 under outputs of alternate signs near 1.8 x 2^127, the first mass's
    tangent near 2: a term near 7.7e37, which a chain doubling the tangent before its quotient passed."""
    streams = jax.random.split(jax.random.key(seed), 3)
    totals = 1 + 0.25 * jax.random.uniform(streams[0], (8,), jnp.float32)
    sizes = 1.8 * 2.0 ** 127 * (1 - 0.1 * jax.random.uniform(streams[1], (8,), jnp.float32))
    direction = jnp.zeros(8, jnp.float32).at[0].set(2 * (1 + 0.25 * jax.random.uniform(streams[2], ())))
    return (sizes * jnp.asarray([1.0, -1.0] * 4, jnp.float32)).astype(jnp.float32), totals, direction


def _masses_past_2_127_along_their_own(seed):
    """Two slices of masses near 1.5 x 2^127 under outputs of opposite signs near 0.9 x 2^127, their tangents
    near their own masses: a doubled tangent passed fp32's largest (NaN where the derivative is near 0)."""
    streams = jax.random.split(jax.random.key(seed), 3)
    totals = 1.5 * 2.0 ** 127 * (1 - 0.1 * jax.random.uniform(streams[0], (2,), jnp.float32))
    sizes = 0.9 * 2.0 ** 127 * (1 - 0.1 * jax.random.uniform(streams[1], (2,), jnp.float32))
    direction = totals * (1 - 0.5 * jax.random.uniform(streams[2], (2,), jnp.float32))
    return (sizes * jnp.asarray([1.0, -1.0], jnp.float32)).astype(jnp.float32), totals, direction


@_reaches_subnormals
def _masses_near_2_126_along_least_normal(seed):
    """Two slices of masses near 1.5 x 2^126 under outputs of opposite signs near 1.8 x 2^127, the first
    mass's tangent near 2^-125: a term near 2.8e-38, which a quotient on the way flushed."""
    streams = jax.random.split(jax.random.key(seed), 3)
    totals = 1.5 * 2.0 ** 126 * (1 + 0.25 * jax.random.uniform(streams[0], (2,), jnp.float32))
    sizes = 1.8 * 2.0 ** 127 * (1 - 0.1 * jax.random.uniform(streams[1], (2,), jnp.float32))
    direction = jnp.zeros(2, jnp.float32).at[0].set(2.0 ** -125 * (1 + jax.random.uniform(streams[2], ())))
    return (sizes * jnp.asarray([1.0, -1.0], jnp.float32)).astype(jnp.float32), totals, direction


def _unit_masses_under_small_outputs_along_2_127(seed):
    """Two slices of masses near 1 under outputs near 2 and -2, the first mass's tangent near 2^127: a term
    near it, which a chain doubling the tangent before halving it passed fp32's largest with (the base
    review of 405c2e725)."""
    streams = jax.random.split(jax.random.key(seed), 3)
    totals = 1 + 0.25 * jax.random.uniform(streams[0], (2,), jnp.float32)
    sizes = 2 * (1 + 0.1 * jax.random.uniform(streams[1], (2,), jnp.float32))
    first = 2.0 ** 127 * (1 + 0.4 * jax.random.uniform(streams[2], ()))
    direction = jnp.zeros(2, jnp.float32).at[0].set(first)
    return (sizes * jnp.asarray([1.0, -1.0], jnp.float32)).astype(jnp.float32), totals, direction


def _halves_over_masses(outputs, totals):
    """`(o - out) / 2` over the masses' power of two (`_scaled_masses`), the scaled masses' sum, and the
    power's exponent: a reference's gaps, neither past fp32 nor flushed at these sites."""
    exponent = jnp.frexp(jnp.max(totals))[1]
    shares = jnp.ldexp(totals, -exponent)
    summed = jnp.sum(shares)
    out = jnp.sum(shares * outputs) / summed
    return jnp.ldexp(outputs * 0.5 - out * 0.5, -exponent), summed


@pytest.mark.parametrize("draw", [_unit_masses_along_two, _masses_past_2_127_along_their_own,
                                  _masses_near_2_126_along_least_normal,
                                  _unit_masses_under_small_outputs_along_2_127],
                         ids=["unit-masses-along-two", "masses-past-2^127-along-their-own",
                              "masses-near-2^126-along-least-normal",
                              "unit-masses-under-small-outputs-along-2^127"])
def test_the_merges_derivative_along_drawn_directions_is_on_par(draw):
    """The derivative of finished slices' merge (`algebra.finished_merged`, at one maximum) along drawn
    tangents of its denominators, on par with sum(d (o - out)) / W formed in fp32 over the masses' power
    of two and the tangents' own, exactly (`_halves_over_masses`), against float64 (check 1): a forward
    pass whose terms are finite where a chain's intermediate overflowed or flushed (train 931's review:
    inf where it is 7.7e37, NaN where it is near 0, 0 where it is 2.8e-38; the base review of 405c2e725:
    inf where it is 2^127)."""

    def ours(outputs, totals, direction):
        return jax.jvp(lambda each: _finished_merged_output(outputs, jnp.zeros_like(totals), each), (totals,),
                       (direction,))[1]

    def reference(outputs, totals, direction):
        halves, summed = _halves_over_masses(outputs, totals)
        exponent = jnp.frexp(jnp.max(jnp.abs(direction)))[1]
        return jnp.ldexp(jnp.sum(jnp.ldexp(direction, -exponent) * halves) / summed * 2, exponent)[None, None]

    _on_par_jitted_and_op_by_op(ours, reference, draw,
                                f"the merge's derivative along drawn directions, {draw.__name__}",
                                seeds=MERGE_EDGE_SEEDS)


def _masses_of_2_20_under_a_cotangent_of_16(seed):
    """Two slices of masses near 2^20 (a slice of 2^20 keys) under outputs of opposite signs near 0.9 x
    2^127, a cotangent near 16: gradients near 1.2e33, which a reverse pass multiplying the cotangent by a
    gap before its quotient passed."""
    streams = jax.random.split(jax.random.key(seed), 3)
    totals = 2.0 ** 20 * (1 + 0.25 * jax.random.uniform(streams[0], (2,), jnp.float32))
    sizes = 0.9 * 2.0 ** 127 * (1 - 0.1 * jax.random.uniform(streams[1], (2,), jnp.float32))
    cotangent = 16 * (1 + 0.25 * jax.random.uniform(streams[2], (), jnp.float32))
    return (sizes * jnp.asarray([1.0, -1.0], jnp.float32)).astype(jnp.float32), totals, cotangent


def test_the_merges_derivative_under_a_large_cotangent_is_on_par():
    """The reverse pass of finished slices' merge (`algebra.finished_merged`, at one maximum) as to its
    denominators under a drawn cotangent, on par with c (o - out) / W formed in fp32 over the masses' power
    of two (`_halves_over_masses`), against float64 (check 1): gradients finite where a transposed chain's
    intermediate overflowed (train 931's review: inf where they are 1.2e33)."""

    def ours(outputs, totals, cotangent):
        out, pull = jax.vjp(lambda each: _finished_merged_output(outputs, jnp.zeros_like(totals), each),
                            totals)
        return pull(jnp.full_like(out, cotangent))[0]

    def reference(outputs, totals, cotangent):
        halves, summed = _halves_over_masses(outputs, totals)
        return cotangent * halves / summed * 2

    _on_par_jitted_and_op_by_op(ours, reference, _masses_of_2_20_under_a_cotangent_of_16,
                                "the merge's derivative under a cotangent of 16", seeds=MERGE_EDGE_SEEDS)


def _equal_outputs_over_small_masses_under_a_large_cotangent(seed):
    """Two slices of masses near 2^-40 under equal outputs (a zero gap) and a cotangent near 2^100: the
    masses' gradients 0, the outputs' near 2^99, where a zero gap's growing exponent met the cotangent
    before its zero in the transposed chain (the base review of 826882bbf: NaN where the gradient is 0)."""
    streams = jax.random.split(jax.random.key(seed), 3)
    totals = 2.0 ** -40 * (1 + jax.random.uniform(streams[0], (2,), jnp.float32))
    outputs = jnp.full(2, jax.random.uniform(streams[1], (), jnp.float32, -1.0, 1.0))
    cotangent = 2.0 ** 100 * (1 + jax.random.uniform(streams[2], (), jnp.float32))
    return outputs, totals.astype(jnp.float32), cotangent


def test_the_merges_gradients_over_equal_outputs_are_on_par():
    """The reverse pass of finished slices' merge (`algebra.finished_merged`, at one maximum) as to its
    denominators and its outputs under a drawn cotangent, on par with c (o - out) / W and c w / W formed in
    fp32 over the masses' power of two, against float64 (check 1): a zero gap's term 0 however large the
    cotangent (the base review of 826882bbf)."""

    def ours(outputs, totals, cotangent):
        out, pull = jax.vjp(lambda o, w: _finished_merged_output(o, jnp.zeros_like(w), w), outputs, totals)
        d_outputs, d_masses = pull(jnp.full_like(out, cotangent))
        return jnp.concatenate([d_masses, d_outputs.reshape(-1)]).reshape(1, -1)

    def reference(outputs, totals, cotangent):
        exponent = jnp.frexp(jnp.max(totals))[1]
        shares = jnp.ldexp(totals, -exponent)
        summed = jnp.sum(shares)
        out = jnp.sum(shares * outputs) / summed
        d_masses = jnp.ldexp(cotangent * (outputs - out) / summed, -exponent)
        return jnp.concatenate([d_masses, cotangent * shares / summed]).reshape(1, -1)

    _on_par_jitted_and_op_by_op(ours, reference, _equal_outputs_over_small_masses_under_a_large_cotangent,
                                "the merge's gradients over equal outputs", seeds=MERGE_EDGE_SEEDS)


def _opposite_outputs_along_a_common_scaling(seed):
    """Two slices of equal masses under outputs a drawn 2^100 apart either side of a drawn 2^90 (each sum
    exact), the masses along a common scaling of them near 2^100, and the outputs along ones: a derivative
    0 along the scaling, whose terms are near 2^199 apart (the base review of 85c6cee2f: NaN where it is 0),
    and 1 along the outputs."""
    streams = jax.random.split(jax.random.key(seed), 4)
    mass = 1 + jax.random.randint(streams[0], (), 0, 8).astype(jnp.float32) / 8
    spread = 2.0 ** 100 * (1 + jax.random.randint(streams[1], (), 0, 8).astype(jnp.float32) / 8)
    centre = 2.0 ** 90 * (1 + jax.random.randint(streams[2], (), 0, 8).astype(jnp.float32) / 8)
    scaling = 2.0 ** 100 * (1 + jax.random.uniform(streams[3], (), jnp.float32))
    return (jnp.stack([centre + spread, centre - spread]), jnp.full(2, mass, jnp.float32),
            jnp.full(2, scaling, jnp.float32))


def test_the_merges_derivative_along_a_common_scaling_of_its_masses_is_on_par():
    """The derivative of finished slices' merge (`algebra.finished_merged`, at one maximum) along a common
    scaling of its masses and along ones of its outputs, on par with c sum(w (o - out)) / W and 1 formed in
    fp32 over the masses' power of two, against float64 (check 1): the mean does not move under a common
    scaling of its masses, however large its terms (the base review of 85c6cee2f)."""

    def ours(outputs, totals, scaling):
        def merged(o, w):
            return _finished_merged_output(o, jnp.zeros_like(w), w)

        along_masses = jax.jvp(lambda w: merged(outputs, w), (totals,), (scaling * totals,))[1]
        along_outputs = jax.jvp(lambda o: merged(o, totals), (outputs,), (jnp.ones_like(outputs),))[1]
        return jnp.stack([along_masses, along_outputs]).reshape(1, 2)

    def reference(outputs, totals, scaling):
        exponent = jnp.frexp(jnp.max(totals))[1]
        shares = jnp.ldexp(totals, -exponent)
        summed = jnp.sum(shares)
        out = jnp.sum(shares * outputs) / summed
        gaps, gap_exponent = jnp.frexp(outputs - out)
        top = jnp.max(gap_exponent)
        moved = jnp.sum(shares * jnp.ldexp(gaps, gap_exponent - top)) / summed
        return jnp.stack([jnp.ldexp(moved * scaling[0], top), jnp.sum(shares) / summed]).reshape(1, 2)

    _on_par_jitted_and_op_by_op(ours, reference, _opposite_outputs_along_a_common_scaling,
                                "the merge's derivative along a common scaling", seeds=MERGE_EDGE_SEEDS)


def _zero_gap(seed):
    """Two slices of drawn masses under equal outputs (a zero gap), drawn directions and a cotangent: train
    943's review's first input, where a jvp of the gradient was 2 where it is 1/4."""
    streams = jax.random.split(jax.random.key(seed), 5)
    totals = 1 + 0.25 * jax.random.uniform(streams[0], (2,), jnp.float32)
    outputs = jnp.full(2, jax.random.uniform(streams[1], (), jnp.float32, -1.0, 1.0))
    return (outputs, totals, jax.random.normal(streams[2], (2,)), jax.random.normal(streams[3], (2,)),
            jax.random.normal(streams[4], ()))


def _zero_mass(seed):
    """A slice of a drawn mass beside one of none (its weight 0), drawn outputs, directions and a cotangent:
    the review's second input, -1, 4 where it is -1, 1 (row 7124: a slice of zero total moves the mean along
    its own total by its gap)."""
    streams = jax.random.split(jax.random.key(seed), 5)
    totals = jnp.zeros(2, jnp.float32).at[0].set(1 + 0.25 * jax.random.uniform(streams[0], ()))
    outputs = jax.random.uniform(streams[1], (2,), jnp.float32, -1.0, 1.0)
    return (outputs, totals, jax.random.normal(streams[2], (2,)), jax.random.normal(streams[3], (2,)),
            jax.random.normal(streams[4], ()))


@_reaches_subnormals
def _subnormal_gap(seed):
    """Two slices of drawn masses under outputs 0 and a subnormal near 2^-130, drawn directions and a
    cotangent: the review's third input, inf where it is finite."""
    streams = jax.random.split(jax.random.key(seed), 5)
    totals = 1 + 0.25 * jax.random.uniform(streams[0], (2,), jnp.float32)
    outputs = jnp.zeros(2, jnp.float32).at[1].set(2.0 ** -130 * (1 + jax.random.uniform(streams[1], ())))
    return (outputs, totals, jax.random.normal(streams[2], (2,)), jax.random.normal(streams[3], (2,)),
            jax.random.normal(streams[4], ()))


@pytest.mark.parametrize("order", ["mixed", "forward over reverse", "reverse over reverse"])
@pytest.mark.parametrize("draw", [_zero_gap, _zero_mass, _subnormal_gap],
                         ids=["zero-gap", "zero-mass", "subnormal-gap"])
def test_the_merges_derivatives_of_its_gradient_are_on_par(draw, order):
    """The derivatives of the gradient as to the masses of finished slices' merge (`algebra.finished_merged`,
    at one maximum) under a drawn cotangent c, each `order` its own program (one program of the three held
    an op-by-op run past 10 GiB): its jvp along drawn output directions d, c (d - sum(w d) / W) / W
    (`mixed`), and its jvp and gradient along drawn mass directions b, the Hessian's product -c (sum(b (o -
    out)) + (o - out) sum(b)) / W^2, forward over reverse and reverse over reverse, on par with those formed
    in fp32 over the masses' power of two, against float64 (check 1): a reverse pass's derivatives never
    read the frexp its terms are formed by (train 943's review: 2 where it is 1/4 at a zero gap, inf at a
    subnormal gap)."""

    def ours(outputs, totals, along_outputs, along_masses, cotangent):
        def gradient(o, w):
            return jax.grad(lambda each: cotangent * _finished_merged_output(o, jnp.zeros_like(each),
                                                                            each)[0, 0])(w)

        if order == "mixed":
            return jax.jvp(lambda o: gradient(o, totals), (outputs,), (along_outputs,))[1].reshape(1, -1)
        if order == "forward over reverse":
            return jax.jvp(lambda w: gradient(outputs, w), (totals,), (along_masses,))[1].reshape(1, -1)
        return jax.grad(lambda w: jnp.vdot(gradient(outputs, w), along_masses))(totals).reshape(1, -1)

    def reference(outputs, totals, along_outputs, along_masses, cotangent):
        shares, moved = _scaled_masses(totals, along_masses)
        exponent = jnp.frexp(jnp.max(totals))[1]
        summed = jnp.sum(shares)
        out = jnp.sum(shares * outputs) / summed
        if order == "mixed":
            mixed = cotangent * (along_outputs - jnp.sum(shares * along_outputs) / summed) / summed
            return jnp.ldexp(mixed, -exponent).reshape(1, -1)
        gaps = outputs - out
        hessian = -cotangent * (jnp.sum(moved * gaps) + gaps * jnp.sum(moved)) / (summed * summed)
        return jnp.ldexp(hessian, -exponent).reshape(1, -1)

    _on_par_jitted_and_op_by_op(ours, reference, draw,
                                f"the derivatives of the merge's gradient, {draw.__name__}, {order}",
                                seeds=SECOND_ORDER_SEEDS)


@_reaches_subnormals
def _slice_and_initial_near_fp32s_largest(seed):
    """A slice and an initial state of masses from 2^126 to 2^127, their outputs near 1: the initial state's
    sums over its total, which XLA:CPU divided by through its reciprocal, there below fp32's least normal
    (train 931's review: its output 0, the merge near half where it is near 1)."""
    streams = jax.random.split(jax.random.key(seed), 2)
    totals = 2.0 ** 126 * (1 + jax.random.uniform(streams[0], (2,), jnp.float32))
    outputs = 1 + 0.25 * jax.random.uniform(streams[1], (2,), jnp.float32)
    return outputs.astype(jnp.float32), totals.astype(jnp.float32), (outputs * totals)[1]


def test_a_decode_merge_from_an_initial_state_near_fp32s_largest_is_on_par():
    """A split decode step's slice merged with the call's initial state (`algebra.merged_slices`, as
    attention_decode merges them), both of masses from 2^126 to 2^127: on par with their outputs' convex
    combination over the masses' power of two, the initial state's output its sums over its total, against
    float64 (check 1)."""

    def merged(outputs, totals, sums):
        spec = algebra.Forward("spans", 1, ids=False, scale=1.0, natural=True, initial=True)
        initial = template.Initial(jnp.zeros((1, 1, 1)), totals[1].reshape(1, 1, 1),
                                   sums.reshape(1, 1, 1, 1))
        return algebra.merged_slices(spec, outputs[0].reshape(1, 1, 1, 1, 1), jnp.zeros((1, 1, 1, 1)),
                                     totals[0].reshape(1, 1, 1, 1), initial)

    def plain(outputs, totals, sums):
        shares, = _scaled_masses(totals)
        each = jnp.stack([outputs[0], sums / totals[1]])
        return (jnp.sum(shares * each) / jnp.sum(shares)).reshape(1, 1, 1, 1)

    _on_par_jitted_and_op_by_op(merged, plain, _slice_and_initial_near_fp32s_largest,
                                "a decode merge from an initial state near fp32's largest",
                                seeds=MERGE_EDGE_SEEDS)


@pytest.mark.parametrize("form", ["causal", "full"])
def test_a_padded_query_walks_nothing_again_where_a_real_row_past_fp32_does(form):
    """The fold guard's rewalk (`algebra.rewalks`) over a call of 6 queries padded to 8: a real row whose
    maximum is -inf (its products all passed fp32 below zero, as every row of a causal or full call sees a
    key) walks again, and so does one whose maximum is past every reach, at the call's rows as a whole call
    states them, at a block's rows from its first query and at each slice's of a split launch; the padded
    rows' -inf maxima, which see no key, never."""
    spec = algebra.Forward(form, keys=8, ids=False, scale=1.0, queries=6)
    maxima = jnp.asarray([0.0, -jnp.inf, 1e6, 2.0, 0.5, 1.0, -jnp.inf, -jnp.inf], jnp.float32)
    expected = [False, True, True, False, False, False, False, False]
    walked, _ = algebra.rewalks(spec, maxima)
    np.testing.assert_array_equal(np.asarray(walked), expected)
    block, _ = algebra.rewalks(spec, maxima[4:], jnp.arange(4, 8))
    np.testing.assert_array_equal(np.asarray(block), expected[4:])
    # A split launch's maxima a slice at a time (`[..., S, Qp]`, as the Mosaic forward hands them over): the
    # second slice's real rows walk again as the first's, its padding never.
    later = jnp.asarray([0.0, 3.0, 1.0, 0.5, 2.0, 1e6, -jnp.inf, -jnp.inf], jnp.float32)
    sliced, _ = algebra.rewalks(spec, jnp.stack([maxima, later]))
    np.testing.assert_array_equal(np.asarray(sliced), [expected, [False] * 5 + [True, False, False]])
