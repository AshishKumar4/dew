"""Checkpoint NVFP4 inputs through the existing Qwix Dense precision path."""


import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from interop_support import NVFP4_QDQ_CASES, nvfp4_qdq_case

from dew.quantization import InputQuantizer, nvfp4_input_qdq


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


TIES = np.asarray([-0.25004873, 0.74987483, -1.2500610], np.float32)
"""Quotients off an E2M1 tie by less than half a float16 step: rounded to float16 first, each lands on the
tie and rounds to even, away from its nearest E2M1 value (-0.25004873 to -0.25, then to -0, where its
nearest is -0.5)."""


def test_a_scanned_input_quantizer_rounds_each_quotient_once():
    """The input QDQ inside a scan, which XLA:CPU compiles as a hoisted small loop, rounds each quotient
    to its nearest E2M1 value, ties to even, as ml_dtypes does: a block holding 6 under a global scale of
    one divides by exactly one, so each value is its own quotient, the values just off a tie (TIES) among
    drawn ones. XLA:CPU converts such a loop's float32 to E2M1 through float16, rounding twice."""
    values = np.random.default_rng(0).uniform(-6, 6, (2, 8, 64)).astype(np.float32)
    values[..., ::16] = 6
    values[..., 1:1 + TIES.size] = TIES
    spec = InputQuantizer.nvfp4(1.0, e4m3_scale=False)
    scanned = jax.jit(lambda xs: jax.lax.scan(lambda carry, x: (carry, nvfp4_input_qdq(x, spec)), 0, xs)[1])
    np.testing.assert_array_equal(np.asarray(scanned(jnp.asarray(values))),
                                  values.astype(ml_dtypes.float4_e2m1fn).astype(np.float32))
