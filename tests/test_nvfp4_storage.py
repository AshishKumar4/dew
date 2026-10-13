"""Weight-only compressed storage and the activation formats that it cannot compute."""

import json
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
