"""Checkpoint NVFP4 inputs through the existing Qwix Dense precision path."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import assert_same_stored_tensors
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained
from dew.interop.safetensors_io import read_weights
from dew.training.quantization import NVFP4Input, nvfp4_input_qdq

FIXTURE = Path(__file__).parent / "fixtures" / "codecs" / "nvfp4_qdq"


@pytest.fixture(scope="module", params=["unrounded", "e4m3"])
def case(request):
    directory = FIXTURE / request.param
    with np.load(directory / "reference.npz") as reference:
        return request.param, directory, dict(reference)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
def test_the_checkpoint_input_quantizer_matches_the_librarys_forward_bit_for_bit(case, dtype):
    """Qwix codes and CT's effective scales, including each scope's non-power-of-two global."""
    kind, _, reference = case
    for key, value in reference.items():
        if not key.endswith("/inputs"):
            continue
        stem = key.removesuffix("/inputs")
        spec = NVFP4Input(float(reference[stem + "/global"].reshape(())), kind == "e4m3")
        actual = jax.jit(lambda x, spec=spec: nvfp4_input_qdq(x, spec))(jnp.asarray(value, dtype))
        expected = reference[stem + ("/qdq" if dtype == jnp.float32 else "/qdq_bf16")]
        np.testing.assert_array_equal(np.asarray(actual, np.float32).view(np.uint32),
                                      expected.view(np.uint32), err_msg=stem)


def test_loading_nvfp4_executes_input_qdq_and_matches_the_same_file_reference(case):
    """The quantized function, held to the reference's own error against float64."""
    _, directory, reference = case
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    actual = loaded.model.apply(loaded.variables, reference["ids"].astype(np.int32))
    assert_as_exact_as_the_reference(actual, reference["logits"], reference["logits_f64"], "NVFP4 QDQ logits")
    np.testing.assert_array_equal(np.argmax(actual, -1), np.argmax(reference["logits"], -1))


def test_export_keeps_the_calibrated_input_scale_and_quantized_forward(case, tmp_path):
    _, directory, reference = case
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    loaded.save(tmp_path)
    written, original = read_weights(tmp_path), read_weights(directory)
    assert_same_stored_tensors(written, original)
    reloaded = Pretrained.load(tmp_path, dtype="float32", attention_impl="reference")
    actual = reloaded.model.apply(reloaded.variables, reference["ids"].astype(np.int32))
    assert_as_exact_as_the_reference(actual, reference["logits"], reference["logits_f64"], "NVFP4 export")


def test_input_qdq_uses_the_training_paths_straight_through_gradient(case):
    kind, _, reference = case
    key = next(key for key in reference if key.endswith("/inputs"))
    stem = key.removesuffix("/inputs")
    spec = NVFP4Input(float(reference[stem + "/global"].reshape(())), kind == "e4m3")
    x = jnp.asarray(reference[key])
    cotangent = jnp.arange(x.size, dtype=x.dtype).reshape(x.shape) / x.size
    gradient = jax.jit(jax.grad(lambda values: jnp.sum(nvfp4_input_qdq(values, spec) * cotangent)))(x)
    np.testing.assert_array_equal(gradient, cotangent)
