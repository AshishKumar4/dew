"""The general pretrained interface loads, generates, decodes and saves DiffusionGemma."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest

from dew.diffusion.block import BlockProcess
from dew.interop import load_pretrained

FIXTURE = Path(__file__).resolve().parent / "fixtures/hf/diffusion-gemma-workflow"
PROMPTS = ["<bos> t5 t7 t9 t11", "<bos> t6 t8 t10 t12"]


def test_public_pretrained_text_workflow_and_checkpoint_readback(tmp_path):
    bundle = load_pretrained(str(FIXTURE), dtype="float32", attention_impl="xla", max_seq_len=32)
    inputs = bundle.processor(PROMPTS)
    task = bundle.block_generation()
    generated = task(PROMPTS, 7, key=jax.random.key(11))
    with np.load(FIXTURE / "reference.npz") as reference:
        np.testing.assert_array_equal(generated.tokens, reference["tokens"][:, :12])
        np.testing.assert_array_equal(generated.decoder_steps, reference["steps"])
    assert task.decode(generated) == (
        "t37 t49 t62 t14 t23 t49 t34", "t39 t55 t53 t39 t31 t29 t58")
    with pytest.raises(TypeError):
        bundle.text_generation()

    trained = jax.tree.map(lambda leaf: leaf + np.float32(0.001), bundle.variables)
    bundle.save(str(tmp_path), variables=trained)
    restored = load_pretrained(str(tmp_path), dtype="float32", attention_impl="xla", max_seq_len=32)
    for expected, actual in zip(jax.tree.leaves(trained), jax.tree.leaves(restored.variables),
                                strict=True):
        np.testing.assert_array_equal(actual, expected)
    restored_inputs = restored.processor(PROMPTS)
    np.testing.assert_array_equal(restored_inputs.tokens, inputs.tokens)
    result = restored.block_generation()(restored_inputs, 7, key=jax.random.key(11))
    assert result.tokens.shape == (2, 12)
    np.testing.assert_array_equal(result.decoder_steps, [8, 8])


def test_public_generation_override_keeps_canvas_semantics():
    bundle = load_pretrained(str(FIXTURE), dtype="float32", attention_impl="xla", max_seq_len=32)
    process = BlockProcess(canvas_length=4, vocab_size=64, max_steps=4)
    process = replace(process, stability_threshold=0, confidence_threshold=10.0)
    result = bundle.block_generation()(PROMPTS, 7, key=jax.random.key(11), process=process)
    with np.load(FIXTURE / "reference.npz") as reference:
        np.testing.assert_array_equal(result.tokens, reference["stopped"][:, :12])
        np.testing.assert_array_equal(result.decoder_steps, reference["stopped_steps"])


def test_public_pipeline_source_storage_and_saved_block_compute_are_independent(tmp_path):
    import json

    import jax.numpy as jnp
    import optax

    import dew
    from dew.inference import BlockGeneration
    from dew.inference.pipeline import place
    from dew.nn.diffusion_gemma import DiffusionGemma
    from dew.objectives.base import FROZEN
    from dew.objectives.diffusion.block import BlockDiffusionObjective
    from dew.training import Checkpoints, Trainer

    bundle = load_pretrained(str(FIXTURE), dtype="float32", param_dtype="bfloat16")
    assert isinstance(bundle.model, DiffusionGemma)
    source = dew.pipeline(str(FIXTURE), dtype="float32", param_dtype="bfloat16")
    assert isinstance(source, BlockGeneration)
    for expected, actual in zip(jax.tree.leaves(bundle.variables), jax.tree.leaves(source.variables), strict=True):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    wanted = bundle.block_generation()([[1, 5, 7]], 3, seed=4).host()
    actual = source([[1, 5, 7]], 3, seed=4).host()
    np.testing.assert_array_equal(actual.tokens, wanted.tokens)
    np.testing.assert_array_equal(actual.decoder_steps, wanted.decoder_steps)

    objective = BlockDiffusionObjective(bundle.model, prompt_length=3, pretrained=bundle.variables)
    state = Trainer(objective, optax.sgd(0.01), key=jax.random.PRNGKey(2)).initial_state()
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(0, state, None)
    checkpoints.wait()
    record = {"objective": "block_diffusion",
              "model": {"architecture": "diffusion_gemma",
                        "config": {**bundle.config, "max_seq_len": bundle.model.max_seq_len},
                        "dtype": "float32", "param_dtype": None, "matmul_precision": None,
                        "attention_impl": "auto"},
              "tokenizer": "byte", "pad_token_id": 0}
    (tmp_path / "run.json").write_text(json.dumps(record))
    restored = dew.pipeline(str(tmp_path), ema=False, dtype="bfloat16", param_dtype="float32")
    assert isinstance(restored, BlockGeneration)
    expected_vars = {name: jax.tree.map(lambda leaf: leaf.astype(jnp.float32), value)
                     if name in ("params", FROZEN) else value for name, value in state.params.items()}
    expected_model = objective.model.clone(text=objective.model.text.clone(dtype=jnp.bfloat16))
    expected_task = BlockGeneration(expected_model, place(expected_vars, None, None),
                                    BlockProcess(expected_model.canvas_length, expected_model.vocab_size))
    assert jnp.dtype(restored.model.text.dtype) == jnp.dtype(jnp.bfloat16)
    for expected, actual in zip(jax.tree.leaves(expected_vars), jax.tree.leaves(restored.variables), strict=True):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    result = restored([[1, 5, 7]], 3, seed=8).host()
    expected = expected_task([[1, 5, 7]], 3, seed=8).host()
    np.testing.assert_array_equal(result.tokens, expected.tokens)
    np.testing.assert_array_equal(result.decoder_steps, expected.decoder_steps)



def test_a_published_checkpoint_with_a_sampler_index_loads_as_the_decoder(tmp_path):
    """google/diffusiongemma-26B-A4B-it ships a diffusers `model_index.json`
    naming only its scheduler beside the decoder's `config.json`; the loader
    reads the decoder the config names instead of a latent pipeline."""
    shutil.copytree(FIXTURE, tmp_path / "published")
    (tmp_path / "published" / "model_index.json").write_text(json.dumps(
        {"_class_name": "DiffusionGemmaPipeline", "_diffusers_version": "0.39.0.dev0",
         "scheduler": ["diffusers", "BlockRefinementScheduler"]}))
    reference = load_pretrained(str(FIXTURE), dtype="float32", attention_impl="xla", max_seq_len=32)
    bundle = load_pretrained(str(tmp_path / "published"), dtype="float32", attention_impl="xla", max_seq_len=32)
    assert type(bundle.model) is type(reference.model)
    assert jax.tree.structure(bundle.variables) == jax.tree.structure(reference.variables)


def test_a_text_decoder_takes_its_tokenizer_whatever_processor_files_ship(tmp_path):
    """The published 26B repo carries a Gemma 4 processor config (image and
    audio feature extractors) beside a text-only model; the loader reads the
    tokenizer, not a processor that needs towers the model does not have."""
    shutil.copytree(FIXTURE, tmp_path / "published")
    (tmp_path / "published" / "processor_config.json").write_text(json.dumps(
        {"processor_class": "Gemma4Processor", "image_processor": {"image_processor_type": "Gemma4ImageProcessor"}}))
    bundle = load_pretrained(str(tmp_path / "published"), dtype="float32", attention_impl="xla", max_seq_len=32)
    reference = load_pretrained(str(FIXTURE), dtype="float32", attention_impl="xla", max_seq_len=32)
    assert type(bundle.processor.reference) is type(reference.processor.reference)
