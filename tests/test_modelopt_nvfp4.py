"""ModelOpt's own files, weight reader and activation arithmetic, through Dew's public load/save."""

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


def test_direct_rn_scale_conversion_accounts_for_the_sm89_backend_difference(checkpoint):
    """Only the cloned author's conversion helper changes; the forward inputs stay fixed."""
    _, reference, _ = checkpoint
    counts = {}
    for key in reference:
        if not key.endswith('/forward_inputs'):
            continue
        stem = key.removesuffix('/forward_inputs')
        spec = NVFP4Input(float(reference[stem + '/global'].reshape(())), e4m3_scale=True, format='modelopt')
        actual = jax.jit(lambda x, spec=spec: nvfp4_input_qdq(x, spec))(jnp.asarray(reference[key]))
        corrected = reference[stem + '/forward_qdq']
        original = reference[stem + '/forward_sm89_qdq']
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint32), corrected.view(np.uint32))
        counts[stem] = int(np.count_nonzero(corrected.view(np.uint32) != original.view(np.uint32)))
    assert counts == {f'model.layers.0.{side}': 13 if side in
                      ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj') else 0
                      for side in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                                   'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')}


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


@pytest.mark.parametrize("mutation, message", [
    ({"quant_algo": "MIXED_PRECISION"}, "MIXED_PRECISION quantized_layers"),
    ({"kv_cache_quant_algo": "FP8"}, "KV cache quantization"),
])
def test_modelopt_variants_without_a_reference_forward_stay_refused_by_name(checkpoint, mutation, message):
    directory, _, _ = checkpoint
    config = json.loads((directory / "config.json").read_text())
    config["quantization_config"] |= mutation
    with pytest.raises(ValueError, match=message):
        codecs.source_quantization(config)


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("DEW_NETWORK_TESTS") != "1",
                    reason="set DEW_NETWORK_TESTS=1 for the pinned Hub config/index")
def test_the_pinned_nvidia_config_and_index_describe_the_supported_nvfp4_parts():
    from huggingface_hub import hf_hub_download

    repo, revision = "nvidia/Qwen3-14B-NVFP4", "bc39319a4dc265d9bbb9a9731bc52c4988d9ece7"
    config = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())
    index_path = Path(hf_hub_download(repo, "model.safetensors.index.json", revision=revision))
    index = json.loads(index_path.read_text())["weight_map"]
    codec = codecs.source_quantization(config)
    assert codec is not None and codec.input_format == "modelopt"
    stem = "model.layers.0.self_attn.q_proj"
    suffixes = (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")
    assert {stem + suffix for suffix in suffixes} <= index.keys()
