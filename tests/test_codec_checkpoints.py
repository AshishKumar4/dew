"""Every quantized checkpoint fixture through Dew's public load and save.

Each one computes its format's own reference forward over the same files,
saves back to the bytes it was read from, and reloads to the same logits:
ModelOpt's NVFP4 and its per-layer FP8 and NVFP4 table, compressed-tensors'
NVFP4 with input QDQ under either local-scale rounding, and weight-only FP8
and NVFP4 storage. The weight readers, activation quantizers and gradients
the forwards compose are each format's own file's: tests/test_modelopt_nvfp4.py,
tests/test_modelopt_mixed.py, tests/test_nvfp4_qdq.py and
tests/test_nvfp4_storage.py.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import NVFP4_QDQ_CASES, assert_same_stored_tensors
from reference_error import assert_as_exact_as_the_reference, float64_forward

from dew.interop import Pretrained
from dew.interop.safetensors_io import read_weights

CODECS = Path(__file__).parent / "fixtures" / "codecs"
CHECKPOINTS = ("modelopt/tiny", "modelopt/mixed", *(f"nvfp4_qdq/{kind}" for kind in NVFP4_QDQ_CASES),
               "storage/fp8", "storage/nvfp4")


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_a_quantized_checkpoint_computes_its_reference_and_saves_its_own_bytes(name, tmp_path):
    """The reference is the format author's own CPU forward of the same
    files, held to float64 rounding. Float64 widens the weights the author's
    reader decodes exactly (weight-only NVFP4 rounds them to bf16, FP8 to
    its stored scale's dtype), so the format's quantization is common to
    both forwards and the bound measures only their rounding. The checkpoint
    held in its own storage (`param_dtype='auto'`) saves its own bytes; held
    in float32 masters it saves each dense weight the source stored narrower
    at the masters' width, the same values, and every other tensor's bytes.
    Both reload to the same logits."""
    directory = CODECS / name
    with np.load(directory / "reference.npz") as reference:
        ids, expected, truth = reference["ids"].astype(np.int32), reference["logits"], reference["logits_f64"]
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    actual = np.asarray(loaded.model.apply(loaded.variables, ids))
    _, terms = float64_forward(loaded.model, loaded.variables, ids)
    assert_as_exact_as_the_reference(actual, expected, truth, f"{name} logits", terms=terms)
    np.testing.assert_array_equal(np.argmax(actual, -1), np.argmax(expected, -1))
    stored = read_weights(directory)
    own = tmp_path / "own"
    Pretrained.load(directory, dtype="float32", param_dtype="auto", attention_impl="reference").save(own)
    assert_same_stored_tensors(read_weights(own), stored)
    masters = tmp_path / "masters"
    loaded.save(masters)
    written = read_weights(masters)
    assert set(written) == set(stored)
    for tensor, value in stored.items():
        if written[tensor].dtype == value.dtype:
            assert_same_stored_tensors({tensor: written[tensor]}, {tensor: value})
        else:
            assert jnp.issubdtype(value.dtype, jnp.floating) and written[tensor].dtype == np.float32, tensor
            np.testing.assert_array_equal(written[tensor], value.astype(np.float32), err_msg=tensor)
    for saved in (own, masters):
        reloaded = Pretrained.load(saved, dtype="float32", attention_impl="reference")
        np.testing.assert_array_equal(reloaded.model.apply(reloaded.variables, ids), actual)
