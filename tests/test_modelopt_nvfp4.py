"""ModelOpt's weight reader and activation arithmetic on its own files, bit for bit; the
checkpoint's public load and save are tests/test_codec_checkpoints.py's."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop import codecs
from dew.interop.safetensors_io import read_weights
from dew.quantization import InputQuantizer, nvfp4_input_qdq

ROOT = Path(__file__).parent / "fixtures" / "codecs" / "modelopt"


@pytest.fixture(scope="module")
def checkpoint():
    directory = ROOT / "tiny"
    with np.load(directory / "reference.npz") as reference:
        return directory, dict(reference)


def test_modelopt_weights_match_its_own_reader_of_the_same_files(checkpoint):
    directory, reference = checkpoint
    stored = read_weights(directory)
    codec = codecs.source_quantization(json.loads((directory / "config.json").read_text()))
    assert codec is not None
    assert len(codec.names(stored)) == 7
    for name in codec.names(stored):
        expected = reference[name.removesuffix(".weight") + "/weight"]
        np.testing.assert_array_equal(codec.decode(stored, name).view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
def test_modelopt_input_arithmetic_matches_its_own_fake_quantizer_in_every_bit(checkpoint, dtype):
    _, reference = checkpoint
    for key in reference:
        if not key.endswith("/inputs"):
            continue
        stem = key.removesuffix("/inputs")
        spec = InputQuantizer.nvfp4(float(reference[stem + "/global"].reshape(())), e4m3_scale=True,
                                    format="modelopt")
        actual = jax.jit(lambda x, spec=spec: nvfp4_input_qdq(x, spec))(jnp.asarray(reference[key], dtype))
        suffix = "/qdq" if dtype == jnp.float32 else "/qdq_bf16"
        np.testing.assert_array_equal(np.asarray(actual, np.float32).view(np.uint32),
                                      reference[stem + suffix].view(np.uint32), err_msg=stem)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
def test_the_pinned_real_modelopt_layer_uses_its_stored_multipliers_bit_for_bit(dtype):
    """32 real q_proj rows, including the division-order .75 tie and tiny-scale floor."""
    stored = read_weights(ROOT / "real")
    config = json.loads((ROOT / "tiny" / "config.json").read_text())
    codec = codecs.source_quantization(config)
    assert codec is not None
    with np.load(ROOT / "real" / "reference.npz") as reference:
        weight = codec.decode(stored, "m.weight")
        np.testing.assert_array_equal(weight.view(np.uint32), reference["weight"].view(np.uint32))
        spec = InputQuantizer.nvfp4(float(reference["global_scale"].reshape(())), e4m3_scale=True,
                                    format="modelopt")
        actual = jax.jit(lambda x: nvfp4_input_qdq(x, spec))(jnp.asarray(reference["inputs"], dtype))
        expected = reference["qdq_f32" if dtype == jnp.float32 else "qdq"]
        np.testing.assert_array_equal(np.asarray(actual, np.float32).view(np.uint32),
                                      expected.view(np.uint32))
