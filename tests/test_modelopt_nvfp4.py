"""ModelOpt's own files, weight reader and activation arithmetic, through Dew's public load/save."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import assert_same_stored_tensors
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained, codecs
from dew.interop.safetensors_io import read_weights
from dew.training.quantization import NVFP4Input, nvfp4_input_qdq

ROOT = Path(__file__).parent / "fixtures" / "codecs" / "modelopt"


@pytest.fixture(scope="module")
def checkpoint():
    directory = ROOT / "tiny"
    with np.load(directory / "reference.npz") as reference:
        loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
        return directory, dict(reference), loaded


def test_modelopt_weights_match_its_own_reader_of_the_same_files(checkpoint):
    directory, reference, _ = checkpoint
    stored = read_weights(directory)
    codec = codecs.source_quantization(json.loads((directory / "config.json").read_text()))
    assert codec is not None
    assert len(codec.names(stored)) == 7
    for name in codec.names(stored):
        expected = reference[name.removesuffix(".weight") + "/weight"]
        np.testing.assert_array_equal(codec.decode(stored, name).view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["fp32", "bf16"])
def test_modelopt_input_arithmetic_matches_its_own_fake_quantizer_in_every_bit(checkpoint, dtype):
    _, reference, _ = checkpoint
    for key in reference:
        if not key.endswith("/inputs"):
            continue
        stem = key.removesuffix("/inputs")
        spec = NVFP4Input(float(reference[stem + "/global"].reshape(())), e4m3_scale=True, format="modelopt")
        actual = jax.jit(lambda x, spec=spec: nvfp4_input_qdq(x, spec))(jnp.asarray(reference[key], dtype))
        suffix = "/qdq" if dtype == jnp.float32 else "/qdq_bf16"
        np.testing.assert_array_equal(np.asarray(actual, np.float32).view(np.uint32),
                                      reference[stem + suffix].view(np.uint32), err_msg=stem)


def test_modelopt_logits_match_the_authors_read_and_input_quantizers(checkpoint):
    _, reference, loaded = checkpoint
    actual = loaded.model.apply(loaded.variables, reference["ids"].astype(np.int32))
    assert_as_exact_as_the_reference(actual, reference["logits"], reference["logits_f64"], "ModelOpt logits")
    np.testing.assert_array_equal(np.argmax(actual, -1), np.argmax(reference["logits"], -1))


def test_modelopt_save_keeps_the_original_packed_bytes_and_input_multipliers(checkpoint, tmp_path):
    directory, reference, loaded = checkpoint
    loaded.save(tmp_path)
    original, written = read_weights(directory), read_weights(tmp_path)
    assert_same_stored_tensors(written, original)
    reloaded = Pretrained.load(tmp_path, dtype="float32", attention_impl="reference")
    actual = reloaded.model.apply(reloaded.variables, reference["ids"].astype(np.int32))
    assert_as_exact_as_the_reference(actual, reference["logits"], reference["logits_f64"], "ModelOpt export")


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
        spec = NVFP4Input(float(reference["global_scale"].reshape(())), e4m3_scale=True, format="modelopt")
        actual = jax.jit(lambda x: nvfp4_input_qdq(x, spec))(jnp.asarray(reference["inputs"], dtype))
        expected = reference["qdq_f32" if dtype == jnp.float32 else "qdq"]
        np.testing.assert_array_equal(np.asarray(actual, np.float32).view(np.uint32),
                                      expected.view(np.uint32))
