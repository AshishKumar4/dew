"""Checkpoint NVFP4 inputs through the existing Qwix Dense precision path."""

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import assert_same_stored_tensors
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained, codecs
from dew.interop.safetensors_io import read_weights
from dew.training.quantization import NVFP4Input, _nvfp4_divide, nvfp4_input_qdq

FIXTURE = Path(__file__).parent / "fixtures" / "codecs" / "nvfp4_qdq"


def test_the_quantizer_divider_is_ieee_fp32_across_its_scale_ranges():
    """2.5M normal ratios, half at or beside E2M1 boundaries, with x64 disabled.

    Denominators span 2**-24 to 2**24 over normed activations, MLP products
    and inverse global scales; quotients reach 128 before E2M1 saturation.
    NumPy's IEEE fp32 divide supplies every expected bit. The correction
    runs in fp32 even when the surrounding application enables x64.
    """
    rng = np.random.default_rng(2046)
    count = 1_000_000
    denominators = np.exp2(rng.uniform(-24, 24, count)).astype(np.float32)
    quotients = (rng.uniform(-8, 8, count) * np.exp2(rng.uniform(-24, 4, count))).astype(np.float32)
    numerators = (denominators.astype(np.float64) * quotients.astype(np.float64)).astype(np.float32)
    boundaries = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], np.float32)
    q, b = rng.choice(boundaries, count // 2), denominators[:count // 2]
    a = (q.astype(np.float64) * b.astype(np.float64)).astype(np.float32)
    numerators = np.concatenate([numerators, a, np.nextafter(a, np.inf), np.nextafter(a, -np.inf)])
    denominators = np.concatenate([denominators, b, b, b])
    with jax.enable_x64(new_val=False):
        actual = jax.jit(_nvfp4_divide)(jnp.asarray(numerators), jnp.asarray(denominators))
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint32),
                                  (numerators / denominators).view(np.uint32))


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


@pytest.mark.parametrize("scale", [0, -1, np.inf, np.nan])
def test_an_invalid_global_scale_is_refused_before_compilation(scale):
    with pytest.raises(ValueError, match="positive and finite"):
        NVFP4Input(scale, e4m3_scale=False)


def test_an_input_scale_missing_from_the_file_cannot_silently_drop_qdq(case, tmp_path):
    _, directory, _ = case
    config = json.loads((directory / "config.json").read_text())
    stored = read_weights(directory)
    name = next(name for name in stored if name.endswith(".input_global_scale"))
    del stored[name]
    codec = codecs.source_quantization(config)
    assert codec is not None
    with pytest.raises(ValueError, match="input_global_scale"):
        codec.names(stored)


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("DEW_NETWORK_TESTS") != "1",
                    reason="set DEW_NETWORK_TESTS=1 for Hub configs")
def test_the_pinned_redhat_config_selects_unrounded_local_scales():
    """Config and index only, no weights from the 32B checkpoint."""
    from huggingface_hub import hf_hub_download

    repo, revision = "RedHatAI/Qwen3-32B-NVFP4", "10a4cab9378ab938be865492b96ff42faff11c3f"
    config = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())
    index_path = Path(hf_hub_download(repo, "model.safetensors.index.json", revision=revision))
    index = json.loads(index_path.read_text())
    codec = codecs.source_quantization(config)
    assert codec is not None and codec.input_scale_dtype == "unrounded"
    modules = [name.removesuffix(".weight_packed") for name in index["weight_map"]
               if name.endswith(".weight_packed")]
    assert len(modules) == 64 * 7
    for module in modules:
        assert module + ".input_global_scale" in index["weight_map"]
