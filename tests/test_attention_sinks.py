"""GPT OSS eager-attention parity from tools/gpt_oss_reference.py."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


@pytest.mark.parametrize("implementation", ["cudnn", "tpu"])
def test_fused_attention_refuses_sinks_by_name(implementation):
    heads = jnp.ones((1, 2, 2, 8))
    with pytest.raises(ValueError):
        scaled_dot_product_attention(heads, heads, heads,
                                     implementation=implementation, sinks=jnp.zeros(2))
