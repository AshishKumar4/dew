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

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from float64_terms import float64_forward
from interop_support import NVFP4_QDQ_CASES, assert_same_stored_tensors
from one_draw import assert_as_exact_as_the_reference
from reference_runs import assert_arms_as_recorded

from dew.interop import Pretrained
from dew.interop.safetensors_io import read_weights
from dew.registry import ModelConfig

CODECS = Path(__file__).parent / "fixtures" / "codecs"
CHECKPOINTS = ("modelopt/tiny", "modelopt/mixed", *(f"nvfp4_qdq/{kind}" for kind in NVFP4_QDQ_CASES),
               "storage/fp8", "storage/nvfp4")
RECORDED = ("storage/fp8", "storage/nvfp4")
"""The weight-only checkpoints, recorded over drawn ids (tools/codec_worker.py). Those with input QDQ stay
on one draw (row 4741): ModelOpt's reference runs its author's CUDA kernels, and compressed-tensors' float64
truth is not Dew's float64 run's (1.6e-07 RMS apart)."""
ONE_DRAW = ("modelopt/tiny", "modelopt/mixed", *(f"nvfp4_qdq/{kind}" for kind in NVFP4_QDQ_CASES))
CODEC_SEEDS = dict.fromkeys(RECORDED, 8)
"""Each checkpoint's declared seeds, sized by `seeds_for` paired as its recording confirmed them."""


def site(name: str) -> str:
    """tools/venv_reference.py's site of checkpoint `name`: `codec-<family>-<kind>`."""
    return "codec-" + name.replace("/", "-")


def drawing(name: str):
    """`draw(seed)` of checkpoint `name`: two rows of ids over its vocabulary, as long as its stored ids."""
    directory = CODECS / name
    vocab = json.loads((directory / "config.json").read_text())["vocab_size"]
    with np.load(directory / "reference.npz") as reference:
        shape = reference["ids"].shape

    def draw(seed: int) -> dict[str, np.ndarray]:
        return {"ids": np.random.default_rng([47, seed]).integers(0, vocab, shape).astype(np.int32)}
    return draw


def computing(name: str):
    """`dew(dtype, inputs)` of checkpoint `name`: its logits at `dtype` through Dew's public load, the model
    rebuilt in float64 for a float64 run (`ModelConfig.with_dtype`) over the same decoded weights."""
    loaded = Pretrained.load(CODECS / name, dtype="float32", attention_impl="reference")
    wide = ModelConfig.from_model(loaded.model).with_dtype("float64").build()

    def dew(dtype, inputs) -> dict:
        model = loaded.model if dtype == jnp.float32 else wide
        variables = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype) if jnp.issubdtype(
            jnp.asarray(leaf).dtype, jnp.floating) else leaf, loaded.variables)
        return {"logits": model.apply(variables, inputs["ids"])}
    return dew


def pipeline_arms(name: str):
    """tools/venv_reference.py's site of checkpoint `name`: its recording, draw, Dew run and leaf."""
    return CODECS / name / "recording.npz", drawing(name), computing(name), ("logits",)


@pytest.mark.parametrize("name", RECORDED)
def test_a_quantized_checkpoint_computes_its_reference(name):
    """The reference is the format author's own CPU forward of the same files, and Dew's logits are on par
    with it over drawn ids (check 1). Float64 widens the weights the author's reader decodes exactly
    (weight-only NVFP4 rounds them to bf16, FP8 to its stored scale's dtype), so the format's quantization is
    common to both forwards and the check measures only their rounding."""
    assert_arms_as_recorded(pipeline_arms(name), site(name), CODEC_SEEDS[name], f"{name} logits")


@pytest.mark.parametrize("name", ONE_DRAW)
def test_an_input_qdq_checkpoint_computes_its_reference(name):
    """ModelOpt's NVFP4 and its per-layer FP8 and NVFP4 table, and compressed-tensors' NVFP4 with input QDQ
    under either local-scale rounding: the author's forward of the same files on its recorded ids, held by
    the one-draw rule (row 4741)."""
    directory = CODECS / name
    with np.load(directory / "reference.npz") as reference:
        ids, expected, truth = reference["ids"].astype(np.int32), reference["logits"], reference["logits_f64"]
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    actual = np.asarray(loaded.model.apply(loaded.variables, ids))
    _, terms = float64_forward(loaded.model, loaded.variables, ids)
    assert_as_exact_as_the_reference(actual, expected, truth, f"{name} logits", terms=terms)


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_a_quantized_checkpoint_saves_its_own_bytes(name, tmp_path):
    """The checkpoint held in its own storage (`param_dtype='auto'`) saves its own bytes; held in float32
    masters it saves each dense weight the source stored narrower at the masters' width, the same values,
    and every other tensor's bytes. Both reload to the same logits."""
    directory = CODECS / name
    with np.load(directory / "reference.npz") as reference:
        ids = reference["ids"].astype(np.int32)
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    actual = np.asarray(loaded.model.apply(loaded.variables, ids))
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
