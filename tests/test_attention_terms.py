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
