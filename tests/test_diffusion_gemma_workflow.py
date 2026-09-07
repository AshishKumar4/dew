"""The general pretrained interface loads, generates, decodes and saves DiffusionGemma."""

from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

from dew.diffusion.block import BlockProcess, CanvasGeneration
from dew.interop import load_pretrained

FIXTURE = Path(__file__).resolve().parent / "fixtures/hf/diffusion-gemma-workflow"
PROMPTS = ["<bos> t5 t7 t9 t11", "<bos> t6 t8 t10 t12"]


def test_public_pretrained_text_workflow_and_checkpoint_readback(tmp_path):
    bundle = load_pretrained(str(FIXTURE), dtype="float32", attention_impl="xla", max_seq_len=32)
    inputs = bundle.processor(PROMPTS)
    generated = bundle.generate(inputs, 7, key=jax.random.key(11))
    assert isinstance(generated, CanvasGeneration)
    with np.load(FIXTURE / "reference.npz") as reference:
        np.testing.assert_array_equal(generated.tokens, reference["tokens"][:, :12])
        np.testing.assert_array_equal(generated.decoder_steps, reference["steps"])
    assert bundle.processor.decode(generated.tokens[:, inputs.tokens.shape[1]:]) == [
        "t37 t49 t62 t14 t23 t49 t34", "t39 t55 t53 t39 t31 t29 t58"]

    trained = jax.tree.map(lambda leaf: leaf + np.float32(0.001), bundle.variables)
    bundle.save(str(tmp_path), variables=trained)
    restored = load_pretrained(str(tmp_path), dtype="float32", attention_impl="xla", max_seq_len=32)
    for expected, actual in zip(jax.tree.leaves(trained), jax.tree.leaves(restored.variables)):
        np.testing.assert_array_equal(actual, expected)
    restored_inputs = restored.processor(PROMPTS)
    np.testing.assert_array_equal(restored_inputs.tokens, inputs.tokens)
    result = restored.generate(restored_inputs, 7, key=jax.random.key(11))
    assert isinstance(result, CanvasGeneration)
    assert result.tokens.shape == (2, 12)
    np.testing.assert_array_equal(result.decoder_steps, [8, 8])


def test_public_generation_override_keeps_canvas_semantics():
    bundle = load_pretrained(str(FIXTURE), dtype="float32", attention_impl="xla", max_seq_len=32)
    process = BlockProcess(canvas_length=4, vocab_size=64, max_steps=4)
    process = replace(process, stability_threshold=0, confidence_threshold=10.0)
    result = bundle.generate(bundle.processor(PROMPTS), 7, key=jax.random.key(11), generation=process)
    assert isinstance(result, CanvasGeneration)
    with np.load(FIXTURE / "reference.npz") as reference:
        np.testing.assert_array_equal(result.tokens, reference["stopped"][:, :12])
        np.testing.assert_array_equal(result.decoder_steps, reference["stopped_steps"])
