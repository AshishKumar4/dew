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
