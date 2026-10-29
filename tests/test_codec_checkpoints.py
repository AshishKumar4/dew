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
from interop_support import NVFP4_QDQ_CASES, assert_same_stored_tensors, compressed_tensors_arms, fixture_ids
from reference_error import assert_as_exact, declared_seeds, indexed_recording, recording

from dew.interop import Pretrained
from dew.interop.safetensors_io import read_weights

CODECS = Path(__file__).parent / "fixtures" / "codecs"
CHECKPOINTS = ("modelopt/tiny", "modelopt/mixed", *(f"nvfp4_qdq/{kind}" for kind in NVFP4_QDQ_CASES),
               "storage/fp8", "storage/nvfp4")

SEEDS = 0
"""Each compressed-tensors checkpoint's declared draws of ids against the author's reader run in the
test (check 1), sized by `seeds_for` paired with Dew's logits on the CPU."""


def checked_logits(name: str, loaded, draw) -> np.ndarray:
    """Dew's logits of the checkpoint `name` on par with its format author's reference from float64 over
    drawn ids (check 1): compressed-tensors' CPU reader run in the test (`compressed_tensors_arms`), or
    ModelOpt's, which runs on CUDA alone, recorded over the seeds (tools/modelopt_nvfp4_reference.py
    `record`). Returns the reference's logits on the first draw."""
    def ours(ids):
        return loaded.model.apply(loaded.variables, ids)

    directory = CODECS / name
    if name.startswith("modelopt/"):
        path = directory / "recorded_logits.npz"
        seeds = declared_seeds(path)["logits"]
        recorded, reference, truth = indexed_recording(recording(path), "logits", seeds,
                                                       inputs_of=lambda seed: {"input_ids": draw(seed)[0]})
        assert_as_exact(lambda index, inputs: {"logits": ours(inputs["input_ids"])}, reference, recorded,
                        f"{name} logits", seeds=seeds, truth_fn=truth)
        return np.asarray(reference(*recorded(0))["logits"])
    reference, truth = compressed_tensors_arms(directory, input_qdq=name.startswith("nvfp4_qdq/"))
    assert_as_exact(ours, reference, draw, f"{name} logits", seeds=SEEDS, truth_fn=truth)
    return reference(*draw(0))


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_a_quantized_checkpoint_computes_its_reference_and_saves_its_own_bytes(name, tmp_path):
    """The reference is the format author's own forward of the same files,
    Dew's logits on par with it from float64 over drawn ids (`checked_logits`).
    Float64 widens the weights the author's reader decodes exactly
    (weight-only NVFP4 rounds them to bf16, FP8 to its stored scale's dtype),
    so the format's quantization is common to both forwards and the check
    measures only their rounding. The checkpoint
    held in its own storage (`param_dtype='auto'`) saves its own bytes; held
    in float32 masters it saves each dense weight the source stored narrower
    at the masters' width, the same values, and every other tensor's bytes.
    Both reload to the same logits."""
    directory = CODECS / name
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    with np.load(directory / "reference.npz") as reference:
        draw = fixture_ids(directory, shape=reference["ids"].shape)
    expected = checked_logits(name, loaded, draw)
    (ids,) = draw(0)
    actual = np.asarray(loaded.model.apply(loaded.variables, ids))
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
