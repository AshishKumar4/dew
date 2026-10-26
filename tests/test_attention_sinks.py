"""GPT OSS eager-attention parity from tools/gpt_oss_reference.py."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_computes_the_oracle, chain_roundings

from dew.nn.attention import scaled_dot_product_attention

FIXTURE = Path(__file__).parent / "fixtures" / "gpt_oss" / "attention.npz"


def test_xla_sink_attention_keeps_causal_window_and_gradients():
    with np.load(FIXTURE) as fixture:
        arrays = {name: jnp.asarray(value) for name, value in fixture.items()}

    def attend(sinks):
        return scaled_dot_product_attention(
            arrays["query"], arrays["key"], arrays["value"], sinks=sinks,
            implementation="xla", causal=True, sliding_window=3)

    np.testing.assert_allclose(attend(arrays["sinks"]), arrays["output"], atol=5e-7, rtol=0)
    gradient = jax.grad(lambda sinks: jnp.square(attend(sinks)).sum())(arrays["sinks"])
    assert bool(jnp.all(gradient < 0))
    without = attend(jnp.full_like(arrays["sinks"], -jnp.inf))
    assert float(jnp.max(jnp.abs(without - arrays["output"]))) > 0.5


@pytest.mark.parametrize("dropout_rate", [0.0, 0.25])
def test_sinks_a_softcap_and_dropout_compose_on_one_reference_path(dropout_rate):
    """Sinks, Gemma 2's softcap and probability dropout ride one score
    pipeline (`reference_attention`): run in float64, a grouped causal call
    with all three, training or not, computes the float64 oracle (the scaled
    logits capped, the causal mask, a softmax a sink joins, the weights the
    draw keeps over the keep rate, the value product) within float64
    rounding."""
    with jax.enable_x64(new_val=True):
        keys = jax.random.split(jax.random.key(0), 4)
        query = jax.random.normal(keys[0], (2, 6, 4, 8), jnp.float64)
        key, value = (jax.random.normal(drawn, (2, 6, 2, 8), jnp.float64) for drawn in keys[1:3])
        sinks = jax.random.normal(keys[3], (4,), jnp.float64)
        rng = jax.random.key(1)

        def attend(query, key, value, sinks):
            return scaled_dot_product_attention(
                query, key, value, sinks=sinks, softcap=2.0, causal=True, implementation="xla",
                dropout_rate=dropout_rate, dropout_rng=rng, deterministic=False)

        group = jnp.arange(4) // 2
        raw = jnp.einsum("bqhd,bkhd->bhqk", query, key[:, :, group]) / np.sqrt(8)
        logits = jnp.where(jnp.tril(jnp.ones((6, 6), bool)), 2.0 * jnp.tanh(raw / 2.0), -jnp.inf)
        sink = jnp.broadcast_to(sinks[None, :, None, None], (2, 4, 6, 1))
        weights = jax.nn.softmax(jnp.concatenate([logits, sink], -1), axis=-1)[..., :-1]
        if dropout_rate:
            # The draw keeps the units a float32 run keeps, in every precision (`dew.nn.dropout`).
            kept = jax.random.bernoulli(rng, jnp.float32(1 - dropout_rate), weights.shape)
            weights = weights * kept / (1 - dropout_rate)
        truth = jnp.einsum("bhqk,bkhd->bqhd", weights, value[:, :, group])
        assert_computes_the_oracle(
            np.asarray(attend(query, key, value, sinks)), np.asarray(truth), "sinks, softcap and dropout",
            roundings=chain_roundings(jax.make_jaxpr(attend)(query, key, value, sinks)))


@pytest.mark.parametrize("implementation", ["cudnn", "tpu"])
def test_fused_attention_refuses_sinks_by_name(implementation):
    heads = jnp.ones((1, 2, 2, 8))
    with pytest.raises(ValueError):
        scaled_dot_product_attention(heads, heads, heads,
                                     implementation=implementation, sinks=jnp.zeros(2))


@pytest.mark.parametrize("heads, kv_heads", [(4, 1), (4, 2)], ids=["shared", "grouped"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
def test_a_masked_key_weighs_nothing_beside_a_sink_however_low_its_logit(heads, kv_heads, dtype):
    """Beside a sink a masked key is out of the softmax: a row whose every key is masked attends the
    sink alone and reads nothing, even where the sink's logit sits at or below the dtype's most
    negative finite value, where a finite floor let every masked key share the row."""
    from dew.nn.attention import reference_attention

    query = jax.random.normal(jax.random.key(0), (1, 3, heads, 8), dtype)
    key, value = (jax.random.normal(jax.random.key(seed), (1, 5, kv_heads, 8), dtype) for seed in (1, 2))
    mask = jnp.zeros((1, 1, 3, 5), bool).at[..., :2, :].set(True)
    # At float32's most negative finite logit, or below float16's, where a finite floor ties the sink.
    sinks = jnp.full((heads,), float(jnp.finfo(dtype).min) * (2 if dtype == jnp.float16 else 1), jnp.float32)
    out = reference_attention(query, key, value, mask=mask, sinks=sinks)
    assert out.dtype == dtype
    np.testing.assert_array_equal(np.asarray(out[:, 2]), 0)
    assert bool(jnp.all(jnp.isfinite(out)))

