"""GPT OSS parity with transformers 5.16.1, from tools/gpt_oss_reference.py."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.gpt_oss import GptOssMLP, dequantize_mxfp4

FIXTURES = Path(__file__).parent / "fixtures" / "gpt_oss"


def test_biased_interleaved_experts_match_reference():
    """fp32 atol 5e-5, maximum observed absolute difference 3.4e-05, with gates
    above the clamp and both expert biases in play."""
    with np.load(FIXTURES / "moe.npz") as fixture:
        arrays = {name: jnp.asarray(value) for name, value in fixture.items()}
    params = {
        "router": {"kernel": arrays["router.weight"].T, "bias": arrays["router.bias"]},
        "experts": {name.removeprefix("experts."): value for name, value in arrays.items()
                    if name.startswith("experts.")},
    }
    module = GptOssMLP(16, 24, 4, 2)

    def run(tree: dict) -> jax.Array:
        return jnp.asarray(module.apply({"params": tree}, arrays["hidden"]))

    np.testing.assert_allclose(run(params), arrays["output"], atol=5e-5, rtol=0)
    params["experts"]["down_proj_bias"] = jnp.zeros_like(params["experts"]["down_proj_bias"])
    assert float(jnp.max(jnp.abs(run(params) - arrays["output"]))) > 0.5


def test_mxfp4_matches_reference_dequantization_exactly():
    """bf16 output, atol 0; maximum observed absolute difference 0."""
    with np.load(FIXTURES / "mxfp4.npz") as fixture:
        actual = dequantize_mxfp4(jnp.asarray(fixture["blocks"]), jnp.asarray(fixture["scales"]))
        assert actual.dtype == jnp.bfloat16
        np.testing.assert_array_equal(actual.astype(jnp.float32), fixture["output"])
