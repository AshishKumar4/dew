"""Checkpoint NVFP4 inputs through the existing Qwix Dense precision path."""


import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import NVFP4_QDQ_CASES, nvfp4_qdq_case

from dew.training.quantization import InputQuantizer, nvfp4_input_qdq


@pytest.fixture(scope="module", params=NVFP4_QDQ_CASES)
def case(request):
    return request.param, *nvfp4_qdq_case(request.param)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
def test_the_checkpoint_input_quantizer_matches_the_librarys_forward_bit_for_bit(case, dtype):
    """Qwix codes and CT's effective scales, including each scope's non-power-of-two global."""
    kind, _, reference = case
    for key, value in reference.items():
        if not key.endswith("/inputs"):
            continue
        stem = key.removesuffix("/inputs")
        spec = InputQuantizer.nvfp4(float(reference[stem + "/global"].reshape(())), kind == "e4m3")
        actual = jax.jit(lambda x, spec=spec: nvfp4_input_qdq(x, spec))(jnp.asarray(value, dtype))
        expected = reference[stem + ("/qdq" if dtype == jnp.float32 else "/qdq_bf16")]
        np.testing.assert_array_equal(np.asarray(actual, np.float32).view(np.uint32),
                                      expected.view(np.uint32), err_msg=stem)


def test_input_qdq_uses_the_training_paths_straight_through_gradient(case):
    kind, _, reference = case
    key = next(key for key in reference if key.endswith("/inputs"))
    stem = key.removesuffix("/inputs")
    spec = InputQuantizer.nvfp4(float(reference[stem + "/global"].reshape(())), kind == "e4m3")
    x = jnp.asarray(reference[key])
    cotangent = jnp.arange(x.size, dtype=x.dtype).reshape(x.shape) / x.size
    gradient = jax.jit(jax.grad(lambda values: jnp.sum(nvfp4_input_qdq(values, spec) * cotangent)))(x)
    np.testing.assert_array_equal(gradient, cotangent)
