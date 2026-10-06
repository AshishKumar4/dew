"""Weight-only compressed storage and the activation formats that it cannot compute."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from interop_support import assert_same_stored_tensors
from reference_error import assert_as_exact_as_the_reference

from dew.interop import Pretrained, codecs
from dew.interop.safetensors_io import read_weights

FIXTURE = Path(__file__).parent / "fixtures" / "codecs" / "storage"


@pytest.fixture(scope="module", params=["fp8", "nvfp4"])
def checkpoint(request):
    directory = FIXTURE / request.param
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    with np.load(directory / "reference.npz") as reference:
        return directory, loaded, dict(reference)


def test_weight_only_storage_logits_match_the_references_read_of_the_same_files(checkpoint):
    """The reference's own from_pretrained CPU forward, held to float64 rounding.

    NVFP4's reader rounds decoded weights to bf16, and FP8's uses the stored
    scale dtype. Float64 widens those decoded weights exactly, so format
    quantization is shared and the bound measures only forward rounding.
    """
    _, loaded, reference = checkpoint
    actual = loaded.model.apply(loaded.variables, reference["ids"].astype(np.int32))
    assert_as_exact_as_the_reference(actual, reference["logits"], reference["logits_f64"], "storage logits")
    np.testing.assert_array_equal(np.argmax(actual, -1), np.argmax(reference["logits"], -1))


def test_weight_only_storage_decodes_as_the_same_file_reference_reader(checkpoint):
    directory, _, reference = checkpoint
    stored = read_weights(directory)
    codec = codecs.source_quantization(json.loads((directory / "config.json").read_text()))
    assert codec is not None and len(codec.names(stored)) == 7
    for name in codec.names(stored):
        np.testing.assert_array_equal(codec.read(stored, name), reference[name], err_msg=name)


def test_an_untrained_weight_only_storage_export_keeps_the_sources_bytes(checkpoint, tmp_path):
    directory, loaded, _ = checkpoint
    loaded.save(tmp_path)
    original, written = read_weights(directory), read_weights(tmp_path)
    assert_same_stored_tensors(written, original)


@pytest.mark.parametrize("kind, dynamic", [("fp8", True), ("fp8", False), ("nvfp4", False)],
                         ids=["dynamic-fp8", "static-fp8", "static-nvfp4"])
def test_a_storage_load_refuses_input_quantization_even_without_stored_scales(kind, dynamic, tmp_path):
    """Dynamic=True has no scale tensor to lose, but its CPU reference still QDQs inputs."""
    config = json.loads((FIXTURE / kind / "config.json").read_text())
    group = config["quantization_config"]["config_groups"]["group_0"]
    group["input_activations"] = {**group["weights"], "dynamic": dynamic}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=r"input_activations have dynamic=.*input quantizers"):
        codecs.source_quantization(config)
    with pytest.raises(ValueError, match=r"input_activations have dynamic=.*input quantizers"):
        Pretrained.load(tmp_path, dtype="float32")


def test_modelopt_mixed_precision_is_refused_with_its_missing_input_rules(tmp_path):
    config = {"model_type": "qwen3",
              "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=r"ModelOpt MIXED_PRECISION quantized_layers"):
        Pretrained.load(tmp_path, dtype="float32")


RELEASES = [
    ("RedHatAI/Llama-3.2-3B-Instruct-FP8", "377571d314b30f1d58448499e4100e2deafe7d7d", "dynamic=False"),
]


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("DEW_NETWORK_TESTS") != "1",
                    reason="set DEW_NETWORK_TESTS=1 for Hub configs")
@pytest.mark.parametrize("repo, revision, message", RELEASES)
def test_pinned_real_activation_formats_are_refused_before_any_weight_download(repo, revision, message):
    """Only config.json is read; the multi-billion-parameter weight files stay on the Hub."""
    from huggingface_hub import hf_hub_download

    config = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())
    with pytest.raises(ValueError, match=message):
        codecs.source_quantization(config)
