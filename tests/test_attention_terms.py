"""An attention layer's terms computed from its own input (`dew.nn.attention_terms`, theme M71): a head's
max-L2 clip, a query's log-ratio temperature and an input-conditioned relative-position bias, each against
its formula written here."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from model_support import attending
from reference_error import assert_as_exact

from dew.nn.attention_terms import QKClip, RelativeBias
from dew.nn.mixer_base import MixerContext
from dew.nn.mixers.attention import CausalSelfAttention

FEATURES, HEADS, KV_HEADS, WIDTH, ROWS, LENGTH = 16, 4, 2, 8, 2, 9
SEEDS = {"terms": 8}
"""Check 1's declared seeds for each site of the file."""

TERMS = {"qk_clip": QKClip(1.5), "temperature_tuning": (4.0, 0.5), "temperature_form": "log_ratio",
         "relative_bias": RelativeBias(rank=3, extent=5, clip=2.0, temperature=True), "projected_sinks": True}
"""A layer clipping its heads (at a threshold its drawn heads exceed), tuning each query's temperature past
its fourth place, biasing each query by its distance from a key within five places, and weighing each
query's own sink, projected from its input (rows 12933, 12934)."""


def _layer(dtype=jnp.float32) -> CausalSelfAttention:
    return CausalSelfAttention(MixerContext(
        emb_features=FEATURES, num_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=WIDTH, max_seq_len=16,
        qk_norm=False, dtype=dtype), nope=True, **TERMS)


def _clip(x, threshold, scale=None):
    x = x if scale is None else x * scale
    return x * jnp.minimum(threshold / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6), 1)


def _by_hand(dtype):
    """The layer's formula in `dtype`: queries scaled and clipped, keys clipped, each query and its
    relative states tuned by `1 + 0.5 log(max((p + 1) / 4, 1))`, the relative states against the bank at
    each query's distance from each key (zero outside [0, 5)), causal softmax of `q k / sqrt(d) + bias`
    beside each query's sink logit `x @ sink_proj`, the sink weighing no value."""
    def run(variables, x):
        p = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype), variables["params"])
        x = jnp.asarray(x, dtype)
        q = (x @ p["q_proj"]["kernel"]).reshape(ROWS, LENGTH, HEADS, WIDTH)
        k = (x @ p["k_proj"]["kernel"]).reshape(ROWS, LENGTH, KV_HEADS, WIDTH)
        v = (x @ p["v_proj"]["kernel"]).reshape(ROWS, LENGTH, KV_HEADS, WIDTH)
        q, k = _clip(q, 1.5, p["q_clip_scale"]), _clip(k, 1.5)
        tuned = 1 + 0.5 * jnp.log(jnp.maximum((jnp.arange(LENGTH, dtype=dtype) + 1) / 4, 1))
        q = q * tuned[None, :, None, None]
        states = (x @ p["relative"]["kernel"]).reshape(ROWS, LENGTH, HEADS, 3)
        states = _clip(states, 2.0, p["relative_scale"]) * tuned[None, :, None, None]
        profile = jnp.einsum("bqhr,re->bhqe", states, p["relative_bank"])
        distance = jnp.arange(LENGTH)[:, None] - jnp.arange(LENGTH)[None, :]
        bias = jnp.where((distance >= 0) & (distance < 5),
                         jnp.take_along_axis(profile, jnp.broadcast_to(jnp.clip(distance, 0, 4),
                                                                       (ROWS, HEADS, LENGTH, LENGTH)), -1), 0)
        k, v = (jnp.repeat(each, HEADS // KV_HEADS, axis=2) for each in (k, v))
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(jnp.asarray(WIDTH, dtype)) + bias
        logits = jnp.where(distance >= 0, logits, -jnp.inf)
        sinks = jnp.swapaxes(x @ p["sink_proj"]["kernel"], 1, 2)[..., None]
        weights = jax.nn.softmax(jnp.concatenate([logits, sinks], axis=-1), -1)[..., :-1]
        out = jnp.einsum("bhqk,bkhd->bqhd", weights, v).reshape(ROWS, LENGTH, -1)
        return out @ p["o_proj"]["kernel"]
    return jax.jit(run)


@pytest.mark.parametrize("kernel", ["reference", "template"])
def test_an_attention_layers_own_terms_are_their_formulas(kernel):
    """A layer clipping its heads, tuning its queries' temperature by the log of their places' ratio,
    biasing its logits by its relative states read at each key's distance and weighing each query's own
    sink: its output, on the portable softmax and on the template (its sinks the initial state), on par
    with the formula written here (`_by_hand`, check 1 against it in float64), every parameter drawn off
    its init, the relative bank among them (rows 12933, 12934)."""
    layer = _layer()

    def draw(seed):
        rng = np.random.default_rng([0xA771, seed])
        x = rng.normal(size=(ROWS, LENGTH, FEATURES)).astype(np.float32)
        params = jax.tree.map(lambda leaf: jnp.asarray(rng.normal(scale=0.5, size=leaf.shape), leaf.dtype),
                              nn.unbox(layer.init(jax.random.key(seed), x)))
        return params, x

    run = attending(kernel)(jax.jit(layer.apply))
    assert_as_exact(run, _by_hand(jnp.float32), draw, f"an attention layer's own terms on the {kernel}",
                    seeds=SEEDS["terms"], truth_fn=_by_hand(jnp.float64))


def test_a_clip_stands_in_for_a_head_norm():
    """A layer bounds its heads by a clip or by a norm, not both."""
    context = MixerContext(emb_features=FEATURES, num_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=WIDTH,
                           max_seq_len=16, qk_norm=True)
    layer = CausalSelfAttention(context, nope=True, qk_clip=QKClip(1.5))
    with pytest.raises(ValueError):
        layer.init(jax.random.key(0), jnp.zeros((1, 4, FEATURES)))
    with pytest.raises(ValueError):
        dataclasses.replace(QKClip(1.5), threshold=0.0)


def test_a_layers_sinks_are_a_heads_or_each_querys_own():
    """A layer takes a head's static sinks or each query's projected own, not both."""
    context = MixerContext(emb_features=FEATURES, num_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=WIDTH,
                           max_seq_len=16, qk_norm=False, attention_sinks=True)
    layer = CausalSelfAttention(context, nope=True, projected_sinks=True)
    with pytest.raises(ValueError):
        layer.init(jax.random.key(0), jnp.zeros((1, 4, FEATURES)))


def test_a_zero_head_passes_its_gradient_through_the_clip():
    """A head of zeros lies in the clip's constant region: its gradient is the cotangent itself, where the
    square root of its zero length would make it NaN."""
    from dew.nn.attention_terms import clipped

    x = jnp.zeros((2, 3, 8), jnp.float32).at[1].set(5.0)
    cotangent = jnp.arange(48, dtype=jnp.float32).reshape(2, 3, 8)
    _, pull = jax.vjp(lambda x: clipped(x, 4.0, 1e-6), x)
    (gradient,) = pull(cotangent)
    np.testing.assert_array_equal(np.asarray(gradient[0]), np.asarray(cotangent[0]))
    assert bool(jnp.all(jnp.isfinite(gradient)))


@pytest.mark.parametrize(("head", "threshold", "clipped_head"), [
    ({0: 2.0 ** 66}, 8.0, {0: 8.0}),
    ({3: -(2.0 ** 127)}, 8.0, {3: -8.0}),
    ({3: -(2.0 ** 127)}, 1.0, {3: -1.0}),
    ({0: 2.0 ** 63, 1: 2.0 ** 63, 2: 2.0 ** 63, 3: 2.0 ** 63}, 8.0, {0: 4.0, 1: 4.0, 2: 4.0, 3: 4.0}),
    ({4: 2.0 ** 61, 5: 2.0 ** 61, 6: 2.0 ** 61, 7: 2.0 ** 61}, 8.0, {4: 4.0, 5: 4.0, 6: 4.0, 7: 4.0}),
    ({0: 2.0 ** 62, 1: 2.0 ** -100}, 2.0 ** 61, {0: 2.0 ** 61, 1: 2.0 ** -101}),
    ({0: 2.0 ** 127}, 2.0 ** -100, {0: 2.0 ** -100}),
], ids=["one-past-the-square", "one-near-the-largest", "near-the-largest-gain-subnormal", "four-past",
        "four-at-the-bound", "a-small-component-kept", "a-subnormal-gain-to-a-small-threshold"])
def test_a_finite_head_whose_squares_overflow_clips_to_the_threshold(head, threshold, clipped_head):
    """fp32 heads of eight whose squares' sum overflows (one component 2^66 or -2^127, whose reciprocal is
    subnormal too, under a threshold whose gain stays normal or does not; four of 2^63), four just inside
    the bound past which the clip takes its length scaled, and a head keeping a small component beside a
    large one, clip to the threshold along themselves, bit for bit (powers of two, which the clip's
    arithmetic keeps exact), with a finite gradient, where their summed squares' infinity clipped the
    overflowing ones to zero (row 13137)."""
    from dew.nn.attention_terms import clipped

    x, expected = np.zeros(8, np.float32), np.zeros(8, np.float32)
    for at, value in head.items():
        x[at] = value
    for at, value in clipped_head.items():
        expected[at] = value
    out, pull = jax.vjp(lambda x: clipped(x, threshold, 1e-6), jnp.asarray(x))
    np.testing.assert_array_equal(np.asarray(out), expected)
    (gradient,) = pull(jnp.arange(8, dtype=jnp.float32))
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_a_clipped_heads_small_component_keeps_its_derivative_both_ways():
    """A head `[2^62, 2^-100, 0, ...]` clipped to `2^61` halves: its derivative along its small component
    is the gain, a tangent `2^-100` there going to `2^-101` and a cotangent `2^64` coming back as `2^63`,
    bit for bit, where a derivative taken in the head's scaled space lost the first and overflowed the
    second (row 13137's review)."""
    from dew.nn.attention_terms import clipped

    x = np.zeros(8, np.float32)
    x[0], x[1] = 2.0 ** 62, 2.0 ** -100
    tangent = np.zeros(8, np.float32)
    tangent[1] = 2.0 ** -100
    _, moved = jax.jvp(lambda x: clipped(x, 2.0 ** 61, 1e-6), (jnp.asarray(x),), (jnp.asarray(tangent),))
    expected = np.zeros(8, np.float32)
    expected[1] = 2.0 ** -101
    np.testing.assert_array_equal(np.asarray(moved), expected)
    _, pull = jax.vjp(lambda x: clipped(x, 2.0 ** 61, 1e-6), jnp.asarray(x))
    cotangent = np.zeros(8, np.float32)
    cotangent[1] = 2.0 ** 64
    (gradient,) = pull(jnp.asarray(cotangent))
    expected[1] = 2.0 ** 63
    np.testing.assert_array_equal(np.asarray(gradient), expected)


def test_a_giant_heads_tiny_gain_keeps_its_representable_derivative():
    """`[2^127, 0, ...]` clipped to `2^-50`, its gain `2^-177` below fp32's range: a tangent or cotangent
    `2^100` across the head moves its clip by `2^-77`, bit for bit, where the gain rounded to zero first
    lost it (row 13137's review)."""
    from dew.nn.attention_terms import clipped

    x, seed, expected = (np.zeros(8, np.float32) for _ in range(3))
    x[0], seed[1], expected[1] = 2.0 ** 127, 2.0 ** 100, 2.0 ** -77
    _, moved = jax.jvp(lambda x: clipped(x, 2.0 ** -50, 1e-6), (jnp.asarray(x),), (jnp.asarray(seed),))
    np.testing.assert_array_equal(np.asarray(moved), expected)
    _, pull = jax.vjp(lambda x: clipped(x, 2.0 ** -50, 1e-6), jnp.asarray(x))
    np.testing.assert_array_equal(np.asarray(pull(jnp.asarray(seed))[0]), expected)


@pytest.mark.parametrize(("width", "head", "threshold", "seed", "expected"), [
    (8, {0: 2.0 ** 127}, 2.0 ** 126, {1: 2.0 ** -100}, {1: 2.0 ** -101}),
    (32, dict.fromkeys(range(16), 2.0 ** 62), 2.0 ** 63, {16: 2.0 ** 65}, {16: 2.0 ** 64}),
], ids=["half-gain-small-seed", "half-gain-large-seed"])
def test_a_giant_heads_normal_gain_scales_its_seeds_whole(width, head, threshold, seed, expected):
    """A giant head clipped to half its length moves a seed across it (a tangent forward, a cotangent back)
    by half, bit for bit, at either end of the range: `2^-100` to `2^-101`, `2^65` to `2^64`, where the
    gain applied in factors lost the first to a subnormal and overflowed the second (row 13137's review)."""
    from dew.nn.attention_terms import clipped

    x, seeded, wanted = (np.zeros(width, np.float32) for _ in range(3))
    for held, values in ((x, head), (seeded, seed), (wanted, expected)):
        for at, value in values.items():
            held[at] = value
    _, moved = jax.jvp(lambda x: clipped(x, threshold, 1e-6), (jnp.asarray(x),), (jnp.asarray(seeded),))
    np.testing.assert_array_equal(np.asarray(moved), wanted)
    _, pull = jax.vjp(lambda x: clipped(x, threshold, 1e-6), jnp.asarray(x))
    np.testing.assert_array_equal(np.asarray(pull(jnp.asarray(seeded))[0]), wanted)


def test_a_giant_heads_radial_derivative_keeps_epsilons_share():
    """`[2^70, 0, ...]` clipped to `2^69` with epsilon 1: along the head, where its length's share of
    `|x| + eps` rounds to 1, its derivative is epsilon's alone, `g eps / (|x| + eps)` = `2^-71`, both ways,
    where `1 - r` cancelled it to zero (row 13137's review)."""
    from dew.nn.attention_terms import clipped

    x, seed, expected = (np.zeros(8, np.float32) for _ in range(3))
    x[0], seed[0], expected[0] = 2.0 ** 70, 1.0, 2.0 ** -71
    _, moved = jax.jvp(lambda x: clipped(x, 2.0 ** 69, 1.0), (jnp.asarray(x),), (jnp.asarray(seed),))
    np.testing.assert_array_equal(np.asarray(moved), expected)
    _, pull = jax.vjp(lambda x: clipped(x, 2.0 ** 69, 1.0), jnp.asarray(x))
    np.testing.assert_array_equal(np.asarray(pull(jnp.asarray(seed))[0]), expected)


def test_a_subnormal_gain_keeps_a_normal_small_output():
    """A 32-wide head of sixteen `2^126` and a `2` clipped to 2, its gain subnormal: the sixteen go to
    `1/2` and the `2` to fp32's smallest normal `2^-126`, bit for bit, the quotient `2 / |y|` normal and
    taken first, where the head over its length first formed a subnormal (row 13137's review)."""
    from dew.nn.attention_terms import clipped

    x, expected = np.zeros(32, np.float32), np.zeros(32, np.float32)
    x[:16], x[16], expected[:16], expected[16] = 2.0 ** 126, 2.0, 0.5, 2.0 ** -126
    np.testing.assert_array_equal(np.asarray(clipped(jnp.asarray(x), 2.0, 1e-6)), expected)


def test_an_unclipped_giant_heads_derivative_is_the_identity():
    """`[2^62, 2^62, 0, ...]` under `2^63`, a giant head the clip leaves as it is: a seed `[1, 2^-30, 0,
    ...]` passes through it unchanged both ways, where subtracting and adding back its part along the head
    lost the small component (row 13137's review)."""
    from dew.nn.attention_terms import clipped

    x, seed = np.zeros(8, np.float32), np.zeros(8, np.float32)
    x[:2], seed[0], seed[1] = 2.0 ** 62, 1.0, 2.0 ** -30
    _, moved = jax.jvp(lambda x: clipped(x, 2.0 ** 63, 1e-6), (jnp.asarray(x),), (jnp.asarray(seed),))
    np.testing.assert_array_equal(np.asarray(moved), seed)
    _, pull = jax.vjp(lambda x: clipped(x, 2.0 ** 63, 1e-6), jnp.asarray(x))
    np.testing.assert_array_equal(np.asarray(pull(jnp.asarray(seed))[0]), seed)


def test_a_subnormal_gains_tangent_near_one_stays_finite_both_ways():
    """A head of four channels `2^127` (the rest zero) clipped to 2, its gain subnormal: a tangent or a
    cotangent `2^127` across it, on the sixth channel, moves its clip by 1 there, bit for bit, both ways,
    where the product `2^127 * 2` overflowed before the head's length scaled it back (row 13169)."""
    from dew.nn.attention_terms import clipped

    x, seed, expected = (np.zeros(8, np.float32) for _ in range(3))
    x[:4], seed[5], expected[5] = 2.0 ** 127, 2.0 ** 127, 1.0
    _, moved = jax.jvp(lambda x: clipped(x, 2.0, 1e-6), (jnp.asarray(x),), (jnp.asarray(seed),))
    np.testing.assert_array_equal(np.asarray(moved), expected)
    _, pull = jax.vjp(lambda x: clipped(x, 2.0, 1e-6), jnp.asarray(x))
    np.testing.assert_array_equal(np.asarray(pull(jnp.asarray(seed))[0]), expected)


def test_a_float64_giant_heads_subnormal_gain_tangent_stays_in_float64():
    """A float64 head `[2^1023, 0, ...]` clipped to 1, its gain subnormal in float64: a tangent or a
    cotangent `2^1023` across it moves its clip by 1 there, both ways, bit for bit, where the balanced
    product taken in float32 read the head's scale as zero (row 13169's review)."""
    from dew.nn.attention_terms import clipped

    with jax.enable_x64(new_val=True):
        x, seed, expected = (np.zeros(8, np.float64) for _ in range(3))
        x[0], seed[5], expected[5] = 2.0 ** 1023, 2.0 ** 1023, 1.0
        _, moved = jax.jvp(lambda x: clipped(x, 1.0, 1e-6), (jnp.asarray(x),), (jnp.asarray(seed),))
        np.testing.assert_array_equal(np.asarray(moved), expected)
        _, pull = jax.vjp(lambda x: clipped(x, 1.0, 1e-6), jnp.asarray(x))
        np.testing.assert_array_equal(np.asarray(pull(jnp.asarray(seed))[0]), expected)


def test_a_tiny_epsilons_radial_share_survives_a_giant_head():
    """`[2^62, 0, ...]` clipped to `2^61` with epsilon `2^-62`: a radial seed `2^120` moves the clip along
    the head by `2^-5`, both ways, jitted, where `eps * s` (`2^-129`) flushed before its division by the
    scaled length and the share was zero (row 13169's review)."""
    from dew.nn.attention_terms import clipped

    x, seed = np.zeros(8, np.float32), np.zeros(8, np.float32)
    x[0], seed[0] = 2.0 ** 62, 2.0 ** 120

    def clip(x):
        return clipped(x, 2.0 ** 61, 2.0 ** -62)

    moved = jax.jit(lambda x, seed: jax.jvp(clip, (x,), (seed,))[1])(jnp.asarray(x), jnp.asarray(seed))
    pulled = jax.jit(lambda x, seed: jax.vjp(clip, x)[1](seed)[0])(jnp.asarray(x), jnp.asarray(seed))
    for derivative in (np.asarray(moved), np.asarray(pulled)):
        assert derivative[0] == 2.0 ** -5, derivative


def test_a_giant_heads_epsilon_share_keeps_its_derivative_in_the_head():
    """A one-channel head `2^64` with epsilon `2^64` clipped to 1: its tangent under a unit seed, differenti-
    ated again along `2^64`, is `-2^-66`, bit for bit, jitted, as the share's plain quotient differentiates
    through its normal intermediates, where the balanced share held the length constant and gave `-2^-67`
    (row 13260)."""
    from dew.nn.attention_terms import clipped

    def tangent(x):
        return jax.jvp(lambda x: clipped(x, 1.0, 2.0 ** 64), (x,), (jnp.ones_like(x),))[1]

    head = jnp.full((1,), 2.0 ** 64, jnp.float32)
    second = jax.jit(lambda x: jax.jvp(tangent, (x,), (jnp.full_like(x, 2.0 ** 64),))[1])(head)
    np.testing.assert_array_equal(np.asarray(second), np.float32([-(2.0 ** -66)]))


def test_a_subnormal_scaled_epsilons_share_rounds_once():
    """`[2^62, 0, ...]` clipped to `2^61` with epsilon `(1 + 2^-19) 2^-64`, its scaled epsilon `2^-131 +
    2^-150` subnormal and its share `2^-126` normal: a radial seed `2^120` moves the clip by `2^-7 +
    2^-26`, both ways, jitted, epsilon's low bit kept, where the scaled epsilon rounded before its division
    lost it on a backend keeping subnormals (row 13260's review)."""
    from dew.nn.attention_terms import clipped

    x, seed = np.zeros(8, np.float32), np.zeros(8, np.float32)
    x[0], seed[0] = 2.0 ** 62, 2.0 ** 120
    epsilon = float(np.float32((1 + 2.0 ** -19) * 2.0 ** -64))

    def clip(x):
        return clipped(x, 2.0 ** 61, epsilon)

    moved = jax.jit(lambda x, seed: jax.jvp(clip, (x,), (seed,))[1])(jnp.asarray(x), jnp.asarray(seed))
    pulled = jax.jit(lambda x, seed: jax.vjp(clip, x)[1](seed)[0])(jnp.asarray(x), jnp.asarray(seed))
    for derivative in (np.asarray(moved), np.asarray(pulled)):
        assert derivative[0] == np.float32(2.0 ** -7 + 2.0 ** -26), derivative


def test_an_ordinary_heads_clip_differentiates_as_its_formula():
    """A head whose squares sum in range clips by its formula as written, so JAX differentiates it: at the
    tie `|x| + eps = threshold` (`[1]`, eps 1, threshold 2) its derivative is `min`'s half, `3/4`; clipped
    (`[1]`, eps 1, threshold 1: `x / (x + 1)`), its second derivative `-1/4` (row 13137's review)."""
    from dew.nn.attention_terms import clipped

    def clip(threshold):
        return lambda value: clipped(value[None], threshold, 1.0)[0]

    assert float(jax.grad(clip(2.0))(jnp.float32(1.0))) == 0.75
    assert float(jax.grad(jax.grad(clip(1.0)))(jnp.float32(1.0))) == -0.25


@attending("reference")
def test_a_relative_bias_reads_its_tokens_slots_whatever_places_its_visibility_counts():
    """A layer scored over a causal pairwise mask, its keys' visibility places stated from 100 on
    (`attention_key_positions`, which a windowless layer's visibility does not read), computes what the same
    call with no places stated does: its relative bias reads its tokens' slots on both sides, as
    transformers' does (`get_relative_distance`)."""
    from dew.nn.inputs import AttentionMetadata, MixerCall

    layer = _layer()
    rng = np.random.default_rng(0xA772)
    x = jnp.asarray(rng.normal(size=(ROWS, LENGTH, FEATURES)), jnp.float32)
    params = jax.tree.map(lambda leaf: jnp.asarray(rng.normal(scale=0.5, size=leaf.shape), leaf.dtype),
                          nn.unbox(layer.init(jax.random.key(0), x)))
    causal = jnp.broadcast_to(jnp.tril(jnp.ones((LENGTH, LENGTH), bool)), (ROWS, LENGTH, LENGTH))
    places = jnp.broadcast_to(jnp.arange(LENGTH) + 100, (ROWS, LENGTH))

    def run(key_positions):
        call = MixerCall(AttentionMetadata(pairwise_mask=causal, key_positions=key_positions))
        return np.asarray(attending("reference")(jax.jit(lambda params, x: layer.apply(params, x, call)))(
            params, x))
    np.testing.assert_array_equal(run(places), run(None))


def test_a_bidirectional_layer_holds_no_relative_bias():
    """A relative bias reads each query's backward distance in the stream's slots, which a bidirectional
    canvas over a frozen prefix places otherwise: a layer that is not causal refuses it as it is built,
    where the causal one builds."""
    x = jnp.zeros((ROWS, LENGTH, FEATURES), jnp.float32)
    causal = _layer()
    causal.init(jax.random.key(0), x)
    canvas = causal.clone(ctx=dataclasses.replace(causal.ctx, causal=False))
    with pytest.raises(ValueError):
        canvas.init(jax.random.key(0), x)


BAND_QUERIES = 1600
"""A band's queries in its tests: past two of its groups (`template.BAND_GROUP`), so three origins."""


@pytest.mark.parametrize("extent, offset", [(1, 0), (5, 37), (300, [0, 1000]), (1024, 1)],
                         ids=["one", "five-chunked", "rows-apart", "inkling"])
def test_a_relative_band_holds_the_relative_bias_it_writes_out(extent, offset):
    """Row 13138: an input-conditioned relative bias as a band (`relative_band`), written out over the call's
    keys (`BandBias.dense`), is the bias `relative_logits` forms over every query and key, at a prefill from
    the first key, at a chunk's (`offset` its first place) and at rows a chunk apart, over three of the
    band's query groups, integer states and bank so either's sums are exact; and its every key within
    `extent - 1` before a query's place through its place lies in the query's band."""
    from dew.nn.attention import BandBias
    from dew.nn.attention_terms import relative_band, relative_logits
    from dew.nn.kernels.templates import band as template

    rng = np.random.default_rng([0x13138, extent])
    rows, heads, rank = 2, 2, 3
    starts = np.broadcast_to(np.asarray(offset, np.int32), (rows,))
    keys = BAND_QUERIES + int(starts.max())
    states = jnp.asarray(rng.integers(-3, 4, (rows, BAND_QUERIES, heads, rank)), jnp.float32)
    bank = jnp.asarray(rng.integers(-3, 4, (rank, extent)), jnp.float32)
    band = relative_band(states, bank, jnp.asarray(offset, jnp.int32))
    places = starts[:, None] + np.arange(BAND_QUERIES)[None, :]
    np.testing.assert_array_equal(np.asarray(band.dense(keys)),
                                  np.asarray(relative_logits(states, bank, places, np.arange(keys))))
    ones = BandBias(jnp.ones((1, 1, BAND_QUERIES, template.band_width(extent)), jnp.float32),
                    jnp.asarray(offset, jnp.int32), extent)
    distance = places[:, :, None] - np.arange(keys)[None, None, :]
    held = np.broadcast_to(np.asarray(ones.dense(keys))[:, 0], distance.shape)
    assert (held[(distance >= 0) & (distance < extent)] == 1).all()


def test_a_relative_band_holds_a_groups_products_at_a_time():
    """Row 13138: a relative band (`relative_band`) forms each query group's products against the bank and
    reads them into the band, never every query's at once beside it: no array of its program holds the
    call's `[Q, extent]` products (an Inkling 2 prefill's 32768 queries', 3.2 GiB a layer)."""
    from reference_error import equations

    from dew.nn import programs
    from dew.nn.attention_terms import relative_band
    from dew.nn.kernels.templates import band as template

    extent = 64
    states = jax.ShapeDtypeStruct((1, BAND_QUERIES, 2, 3), jnp.float32)
    bank = jax.ShapeDtypeStruct((3, extent), jnp.float32)
    traced = jax.make_jaxpr(lambda s, b: relative_band(s, b, jnp.int32(0)).values)(states, bank).jaxpr
    found = [getattr(var.aval, "shape", ()) for equation in equations(traced, prune=programs.opaque)
             for var in equation.outvars]
    assert any(shape[-2:] == (template.BAND_GROUP, extent) for shape in found), found
    assert not any(shape[-2:] == (BAND_QUERIES, extent) for shape in found)


def test_a_relative_band_of_no_queries_or_rows_is_empty():
    """Row 13138: a relative band (`relative_band`) over no queries, and a band of no rows written out
    (`BandBias.dense`), are empty as `relative_logits`' bias over them is, its other axes theirs."""
    from dew.nn.attention import BandBias
    from dew.nn.attention_terms import relative_band, relative_logits
    from dew.nn.kernels.templates import band as template

    extent, keys = 5, 16
    states = jnp.zeros((2, 0, 3, 4), jnp.float32)
    bank = jnp.zeros((4, extent), jnp.float32)
    band = relative_band(states, bank, jnp.int32(0))
    assert band.dense(keys).shape == relative_logits(states, bank, np.arange(0), np.arange(keys)).shape
    rowless = BandBias(jnp.zeros((0, 3, 7, template.band_width(extent)), jnp.float32), jnp.int32(0), extent)
    assert rowless.dense(keys).shape == (0, 3, 7, keys)
