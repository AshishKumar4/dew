"""GPT OSS eager-attention parity from tools/gpt_oss_reference.py."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn.attention import scaled_dot_product_attention
from dew.nn.attention_sinks import attention_with_sinks

FIXTURE = Path(__file__).parent / "fixtures" / "gpt_oss" / "attention.npz"


def test_sink_attention_matches_gpt_oss_eager():
    """fp32 atol 5e-7; maximum observed absolute difference 1.1920929e-7."""
    with np.load(FIXTURE) as fixture:
        arrays = {name: jnp.asarray(value) for name, value in fixture.items()}
    actual = attention_with_sinks(arrays["query"], arrays["key"], arrays["value"],
                                  arrays["sinks"], mask=arrays["mask"])
    np.testing.assert_allclose(actual, arrays["output"], atol=5e-7, rtol=0)
    # Dropping the sink term must break the reference comparison.
    without = attention_with_sinks(arrays["query"], arrays["key"], arrays["value"],
                                   jnp.full_like(arrays["sinks"], -jnp.inf),
                                   mask=arrays["mask"])
    assert float(jnp.max(jnp.abs(without - arrays["output"]))) > 0.5


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


@pytest.mark.parametrize("implementation", ["cudnn", "tpu"])
def test_fused_attention_refuses_sinks_by_name(implementation):
    heads = jnp.ones((1, 2, 2, 8))
    with pytest.raises(ValueError, match=f"{implementation}.*sinks"):
        scaled_dot_product_attention(heads, heads, heads,
                                     implementation=implementation, sinks=jnp.zeros(2))
