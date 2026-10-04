"""The general pretrained interface loads, generates, decodes and saves DiffusionGemma."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

from dew.diffusion.block import BlockProcess
from dew.interop import Pretrained

FIXTURE = Path(__file__).resolve().parent / "fixtures/hf/diffusion-gemma-workflow"
PROMPTS = ["<bos> t5 t7 t9 t11", "<bos> t6 t8 t10 t12"]


def test_public_pretrained_text_workflow_and_checkpoint_readback(tmp_path):
    bundle = Pretrained.load(str(FIXTURE), dtype="float32", attention_impl="xla", max_seq_len=32)
    inputs = bundle.processor(PROMPTS)
    task = bundle.block_generation()
    generated = task(PROMPTS, 7, key=jax.random.key(11))
    with np.load(FIXTURE / "reference.npz") as reference:
        np.testing.assert_array_equal(generated.tokens, reference["tokens"][:, :12])
        np.testing.assert_array_equal(generated.decoder_steps, reference["steps"])
    assert task.decode(generated) == (
        "t37 t49 t62 t14 t23 t49 t34", "t39 t55 t53 t39 t31 t29 t58")
    assert not hasattr(bundle, "text_generation")

    trained = jax.tree.map(lambda leaf: leaf + np.float32(0.001), bundle.variables)
    bundle.save(str(tmp_path), variables=trained)
    restored = Pretrained.load(str(tmp_path), dtype="float32", attention_impl="xla", max_seq_len=32)
    for expected, actual in zip(jax.tree.leaves(trained), jax.tree.leaves(restored.variables),
                                strict=True):
        np.testing.assert_array_equal(actual, expected)
    restored_inputs = restored.processor(PROMPTS)
    np.testing.assert_array_equal(restored_inputs.tokens, inputs.tokens)
    result = restored.block_generation()(restored_inputs, 7, key=jax.random.key(11))
    assert result.tokens.shape == (2, 12)
    np.testing.assert_array_equal(result.decoder_steps, [8, 8])


def test_public_generation_override_keeps_canvas_semantics():
    bundle = Pretrained.load(str(FIXTURE), dtype="float32", attention_impl="xla", max_seq_len=32)
    process = BlockProcess(canvas_length=4, vocab_size=64, max_steps=4)
    process = replace(process, stability_threshold=0, confidence_threshold=10.0)
    result = bundle.block_generation()(PROMPTS, 7, key=jax.random.key(11), process=process)
    with np.load(FIXTURE / "reference.npz") as reference:
        np.testing.assert_array_equal(result.tokens, reference["stopped"][:, :12])
        np.testing.assert_array_equal(result.decoder_steps, reference["stopped_steps"])


def test_public_pipeline_preserves_the_block_model_layout_and_generation():
    import dew

    bundle = Pretrained.load(str(FIXTURE), dtype="float32", param_dtype="bfloat16")
    source = dew.pipeline(str(FIXTURE), dtype="float32", param_dtype="bfloat16")
    for expected, actual in zip(jax.tree.leaves(bundle.variables), jax.tree.leaves(source.variables),
                                strict=True):
        np.testing.assert_array_equal(actual, expected)
    expected = bundle.block_generation()([[1, 5, 7]], 3, key=4).host()
    actual = source([[1, 5, 7]], 3, key=4).host()
    np.testing.assert_array_equal(actual.tokens, expected.tokens)
    np.testing.assert_array_equal(actual.decoder_steps, expected.decoder_steps)


def test_public_pipeline_source_storage_and_saved_block_compute_are_independent(tmp_path):
    import jax.numpy as jnp
    import optax

    import dew
    from dew.inference import BlockGeneration
    from dew.inference.pipeline import place
    from dew.nn.diffusion_gemma import DiffusionGemma
    from dew.objectives.base import FROZEN
    from dew.objectives.diffusion.block import BlockDiffusionObjective
    from dew.training import Checkpoints, Trainer

    bundle = Pretrained.load(str(FIXTURE), dtype="float32", param_dtype="bfloat16")
    assert isinstance(bundle.model, DiffusionGemma)
    source = dew.pipeline(str(FIXTURE), dtype="float32", param_dtype="bfloat16")
    assert isinstance(source, BlockGeneration)
    for expected, actual in zip(
        jax.tree.leaves(bundle.variables), jax.tree.leaves(source.variables), strict=True
    ):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    wanted = bundle.block_generation()([[1, 5, 7]], 3, key=4).host()
    actual = source([[1, 5, 7]], 3, key=4).host()
    np.testing.assert_array_equal(actual.tokens, wanted.tokens)
    np.testing.assert_array_equal(actual.decoder_steps, wanted.decoder_steps)

    objective = BlockDiffusionObjective(bundle.model, prompt_length=3, pretrained=bundle.variables)
    state = Trainer(objective, optax.sgd(0.01), key=jax.random.PRNGKey(2)).initial_state()
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(0, state, None, artifact=objective.inference_record())
    checkpoints.wait()
    restored = dew.pipeline(str(tmp_path), ema=False, dtype="bfloat16", param_dtype="float32")
    assert isinstance(restored, BlockGeneration)
    expected_vars = {name: jax.tree.map(lambda leaf: leaf.astype(jnp.float32), value)
                     if name in ("params", FROZEN) else value for name, value in state.variables.items()}
    expected_model = objective.model.clone(text=objective.model.text.clone(dtype=jnp.bfloat16))
    expected_task = BlockGeneration(expected_model, place(expected_vars, None, None),
                                    BlockProcess(expected_model.canvas_length, expected_model.vocab_size))
    assert jnp.dtype(restored.model.text.dtype) == jnp.dtype(jnp.bfloat16)
    for expected, actual in zip(
        jax.tree.leaves(expected_vars), jax.tree.leaves(restored.variables), strict=True
    ):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    result = restored([[1, 5, 7]], 3, key=8).host()
    expected = expected_task([[1, 5, 7]], 3, key=8).host()
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
    reference = Pretrained.load(str(FIXTURE), dtype="float32", attention_impl="xla", max_seq_len=32)
    bundle = Pretrained.load(
        str(tmp_path / "published"), dtype="float32", attention_impl="xla", max_seq_len=32
    )
    assert type(bundle.model) is type(reference.model)
    assert jax.tree.structure(bundle.variables) == jax.tree.structure(reference.variables)


def test_a_text_decoder_takes_its_tokenizer_whatever_processor_files_ship(tmp_path):
    """The published 26B repo carries a Gemma 4 processor config (image and
    audio feature extractors) beside a text-only model; the loader reads the
    tokenizer, not a processor that needs towers the model does not have."""
    shutil.copytree(FIXTURE, tmp_path / "published")
    (tmp_path / "published" / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Gemma4Processor",
                "image_processor": {"image_processor_type": "Gemma4ImageProcessor"},
            }
        )
    )
    bundle = Pretrained.load(
        str(tmp_path / "published"), dtype="float32", attention_impl="xla", max_seq_len=32
    )
    reference = Pretrained.load(str(FIXTURE), dtype="float32", attention_impl="xla", max_seq_len=32)
    assert type(bundle.processor.reference) is type(reference.processor.reference)


def test_transformers_reads_a_trained_export_at_dews_logits_and_tokens(tmp_path):
    """The export of a changed DiffusionGemma is a checkpoint transformers'
    DiffusionGemmaForBlockDiffusion loads with a clean report. On the
    fixture's prompt and canvas its bare and self-conditioned logits, in fp32
    and float64, hold Dew's on the changed weights to tests/reference_error.py's
    rule, and its generation from matched draws
    (tools/diffusion_gemma_reference.py) writes Dew's tokens."""
    import torch
    from reference_error import assert_as_exact_as_the_reference
    from test_block_diffusion import prefill
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion

    from dew.nn.inputs import ModelInputs
    from tools.diffusers_wan_reference import float64
    from tools.diffusion_gemma_reference import reference_generation

    bundle = Pretrained.load(str(FIXTURE), dtype="float32", attention_impl="xla", max_seq_len=32)
    leaves, tree = jax.tree.flatten(bundle.variables)
    keys = jax.random.split(jax.random.key(5), len(leaves))
    changed = jax.tree.unflatten(tree, [leaf + 0.02 * jax.random.normal(key, leaf.shape, leaf.dtype)
                                        for leaf, key in zip(leaves, keys, strict=True)])
    bundle.save(str(tmp_path), variables=changed)
    with np.load(FIXTURE / "reference.npz") as stored:
        reference = {name: stored[name] for name in stored.files}

    def theirs(dtype):
        model, report = DiffusionGemmaForBlockDiffusion.from_pretrained(
            str(tmp_path), dtype=dtype, local_files_only=True, output_loading_info=True,
            experts_implementation="eager")
        assert not any(report.values()), report
        model = model.eval()
        model.set_attn_implementation("eager")
        prompt, canvas = (torch.from_numpy(reference[key]) for key in ("prompt", "canvas"))
        previous = torch.from_numpy(reference["previous"]).to(dtype)
        with torch.no_grad():
            bare = model(input_ids=prompt, decoder_input_ids=canvas).logits
            conditioned = model(input_ids=prompt, decoder_input_ids=canvas,
                                self_conditioning_logits=previous).logits
        return model, bare.numpy(), conditioned.numpy()

    model, bare, conditioned = theirs(torch.float32)
    with float64():
        _, bare_f64, conditioned_f64 = theirs(torch.float64)
    cache = prefill(bundle.model, changed, reference["prompt"])
    ours_bare = bundle.model.apply({**changed, "cache": cache}, reference["canvas"])
    ours_conditioned = bundle.model.apply({**changed, "cache": cache}, reference["canvas"],
                                          self_conditioning_logits=reference["previous"])
    assert_as_exact_as_the_reference(np.asarray(ours_bare), bare, bare_f64, "bare")
    assert_as_exact_as_the_reference(np.asarray(ours_conditioned), conditioned, conditioned_f64,
                                     "conditioned")

    generated, _, _ = reference_generation(model, torch.from_numpy(reference["prompt"]))
    process = bundle.block_generation().process
    prompt = ModelInputs(jax.numpy.asarray(reference["prompt"]))
    ours = process.generate(bundle.model, changed, prompt, 7, key=jax.random.key(11))
    np.testing.assert_array_equal(ours.tokens, generated.sequences.numpy()[:, :12])
