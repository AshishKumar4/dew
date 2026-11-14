"""GPT OSS eager-attention parity from tools/gpt_oss_reference.py."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import attention_on_par, kept
from contract_support import normal_drawn
from reference_error import FEWEST_SEEDS, assert_computes_the_oracle, chain_roundings

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.kernels.templates.attention import dropout_keep, dropout_seed

FIXTURE = Path(__file__).parent / "fixtures" / "gpt_oss" / "attention.npz"


def test_xla_sink_attention_keeps_causal_window_and_gradients():
    """GPT-OSS's sinks under a causal window of three keys on the xla kernel, at the shapes of GPT-OSS's
    recorded eager attention (tools/gpt_oss_reference.py's fixture): over fresh draws its output and every
    gradient, the sinks' among them, on par with the reference path's from float64 (check 1,
    `attention_on_par`); the reference path's formula is GPT-OSS's to float64's own rounding
    (`test_sinks_a_softcap_and_dropout_compose_on_one_reference_path`)."""
    with np.load(FIXTURE) as fixture:
        shapes = [fixture[name].shape for name in ("query", "key", "value", "sinks", "output")]

    def attend(implementation):
        def run(query, key, value, sinks):
            return scaled_dot_product_attention(query, key, value, sinks=sinks,
                                                implementation=implementation, causal=True, sliding_window=3)
        return run

    queries, keys = shapes[0][1], shapes[1][1]
    attention_on_par(attend("xla"), attend("reference"), normal_drawn(0x515, *shapes),
                     "xla sinks in a window", keep=kept(queries, keys, causal=True, sliding_window=3),
                     seeds=FEWEST_SEEDS)


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
            keep = dropout_keep(dropout_seed(rng), dropout_rate, weights.shape)
            scale = jnp.asarray(1.0 / (1.0 - dropout_rate), weights.dtype)
            weights = weights * jnp.where(keep, scale, 0)
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

