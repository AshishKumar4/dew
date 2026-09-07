"""Published DiffusionGemma inference with tiny released-code reference weights.

The full checkpoint contains sliding/full attention, routed experts, shared
encoder/decoder weights, self-conditioning, and a vision tower. The committed
Transformers generate trajectory uses matched random inputs, not an assumed
identity between Torch and JAX seeds (tools/diffusion_gemma_reference.py).
"""

from dataclasses import replace
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import load_file

from dew.diffusion.block import BlockProcess
from dew.interop import diffusion_gemma as adapter
from dew.nn.diffusion_gemma import SelfConditioning, soft_embeddings, translate_weights
from dew.nn.inputs import ModelInputs

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"
WORKFLOW = FIXTURES / "diffusion-gemma-workflow"


@pytest.fixture(scope="module")
def system():
    config = json.loads((WORKFLOW / "config.json").read_text())
    model = adapter.build(config, dtype="float32", attention_impl="xla")
    variables = adapter.translate_weights(load_file(str(WORKFLOW / "model.safetensors")), config)
    process = adapter.generation_process(config, json.loads((WORKFLOW / "generation_config.json").read_text()))
    with np.load(WORKFLOW / "reference.npz") as stored:
        reference = {name: stored[name] for name in stored.files}
    return model, variables, process, reference, config


def prefill(model, variables, prompt):
    cache = model.apply(variables, prompt.shape[0], method=model.init_cache, mutable=["cache"])[1]["cache"]
    return model.apply({**variables, "cache": cache}, prompt, method=model.encode,
                       mutable=["cache"])[1]["cache"]


def test_entropy_bound_accepts_by_entropy_not_position():
    process = BlockProcess(canvas_length=3, vocab_size=2, entropy_bound=0.05)
    accepted, mask = process.accept(
        np.array([[0, 0, 0]]), np.array([[1, 1, 1]]),
        np.array([[[0., 0.], [2., -2.], [1., -1.]]], np.float32))
    np.testing.assert_array_equal(mask, [[False, True, False]])
    np.testing.assert_array_equal(accepted, [[0, 1, 0]])


@pytest.mark.parametrize("fields", [
    {"canvas_length": 0}, {"vocab_size": 0}, {"max_steps": 0},
    {"entropy_bound": float("nan")}, {"t_min": 0.9},
    {"confidence_threshold": float("inf")}, {"stability_threshold": -1},
])
def test_invalid_sampling_geometry_is_rejected(fields):
    with pytest.raises(ValueError):
        BlockProcess(**({"canvas_length": 4, "vocab_size": 64} | fields))


def test_native_shared_model_forward_matches_reference(system):
    """fp32 maximum error 2.2e-6, identical argmax, tolerance 1e-4.

    Five prompt tokens exceed the four-token local window. Every canvas query
    reads the same last three prefix keys and all four canvas keys; the old
    query-relative mask changes these logits.
    """
    model, variables, _, reference, _ = system
    cache = prefill(model, variables, reference["prompt"])
    before = jax.tree.map(np.asarray, cache)
    bare = model.apply({**variables, "cache": cache}, reference["canvas"])
    conditioned = model.apply({**variables, "cache": cache}, reference["canvas"],
                              self_conditioning_logits=reference["previous"])
    np.testing.assert_allclose(bare, reference["bare"], atol=1e-4, rtol=0)
    np.testing.assert_allclose(conditioned, reference["conditioned"], atol=1e-4, rtol=0)
    np.testing.assert_array_equal(jnp.argmax(conditioned, -1), reference["conditioned"].argmax(-1))
    for expected, actual in zip(jax.tree.leaves(before), jax.tree.leaves(cache)):
        np.testing.assert_array_equal(actual, expected)


def test_multi_canvas_generation_matches_full_reference_loop(system):
    model, variables, process, reference, _ = system
    inputs = ModelInputs(jnp.asarray(reference["prompt"]))
    result = process.generate(model, variables, inputs, 7, key=jax.random.key(11))
    # Transformers returns the whole final canvas. Dew honors max_new_tokens
    # at its public result while refining the same four-token canvas internally.
    np.testing.assert_array_equal(result.tokens, reference["tokens"][:, :12])
    np.testing.assert_array_equal(result.lengths, [7, 7])
    np.testing.assert_array_equal(result.decoder_steps, reference["steps"])
    assert not bool(result.terminated.any())


def test_refinements_return_the_last_prediction_without_an_extra_call(system):
    """Raw final-step error below 1e-4 on the complete reference trajectory."""
    model, variables, process, reference, _ = system
    cache = prefill(model, variables, reference["prompt"])
    state = process.refine(model, variables, cache, jax.random.fold_in(jax.random.key(11), 0),
                           2, jnp.zeros((2,), bool))
    np.testing.assert_allclose(state.logits * process.temperature(1),
                               reference["trajectory"][3], atol=1e-4, rtol=0)
    np.testing.assert_array_equal(state.decoder_steps, [4, 4])
    np.testing.assert_array_equal(state.argmax, reference["tokens"][:, 5:9])


def test_adaptive_stopping_resets_for_each_canvas(system):
    model, variables, process, reference, _ = system
    quick = replace(process, stability_threshold=0, confidence_threshold=10.0)
    result = quick.generate(model, variables, ModelInputs(jnp.asarray(reference["prompt"])),
                            7, key=jax.random.key(11))
    np.testing.assert_array_equal(result.tokens, reference["stopped"][:, :12])
    np.testing.assert_array_equal(result.decoder_steps, reference["stopped_steps"])
    np.testing.assert_array_equal(result.lengths, [7, 7])


def test_eos_finishes_rows_independently_and_padding_is_not_a_token(system):
    model, variables, process, reference, _ = system
    eos = int(reference["eos_id"])
    result = process.generate(model, variables, ModelInputs(jnp.asarray(reference["prompt"])),
                              7, key=jax.random.key(11), eos_token_ids=(eos,), pad_token_id=0)
    expected = reference["eos_tokens"][:, :12]
    np.testing.assert_array_equal(result.tokens, expected)
    np.testing.assert_array_equal(result.decoder_steps, reference["eos_steps"])
    np.testing.assert_array_equal(result.lengths, [1, 7])
    np.testing.assert_array_equal(result.terminated, [True, False])


def test_checkpoint_export_keeps_updated_weights_and_generation(system):
    model, variables, process, reference, config = system
    changed = jax.tree.map(lambda value: value + jnp.asarray(0.001, value.dtype), variables)
    restored = adapter.translate_weights(adapter.export_weights(model, changed, config), config)
    for wanted, actual in zip(jax.tree.leaves(changed), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(actual, wanted)
    inputs = ModelInputs(jnp.asarray(reference["prompt"]))
    wanted = process.generate(model, changed, inputs, 7, key=jax.random.key(11))
    actual = process.generate(model, restored, inputs, 7, key=jax.random.key(11))
    np.testing.assert_array_equal(actual.tokens, wanted.tokens)


def test_zero_tokens_does_not_prefill_and_capacity_uses_whole_canvases(system):
    model, variables, process, reference, _ = system
    inputs = ModelInputs(jnp.asarray(reference["prompt"]))
    empty = process.generate(model, variables, inputs, 0, key=jax.random.key(11))
    np.testing.assert_array_equal(empty.tokens, inputs.tokens)
    np.testing.assert_array_equal(empty.decoder_steps, [0, 0])
    small = model.clone(text=model.text.clone(max_seq_len=11))
    with pytest.raises(ValueError, match="rounded-up canvases"):
        process.generate(small, variables, inputs, 6, key=jax.random.key(11))


def test_self_conditioning_matches_the_reference_implementation():
    """Independent self-conditioning fixture: observed fp32 error 1.1e-6."""
    directory = FIXTURES / "diffusion-gemma-sc-tiny"
    module = SelfConditioning(hidden_size=32, intermediate_size=64)
    variables = {"params": translate_weights(load_file(str(directory / "model.safetensors")))}
    embeds, signal = np.load(directory / "inputs.npy"), np.load(directory / "signal.npy")
    np.testing.assert_allclose(np.asarray(module.apply(variables, embeds, signal)),
                               np.load(directory / "ref.npy"), atol=1e-4, rtol=0)


def test_soft_embeddings_averages_the_table_under_the_distribution():
    table = np.array([[1., 0.], [0., 1.]], np.float32)
    got = soft_embeddings(np.array([[[10., 0.], [0., 0.]]], np.float32), table, 2.)
    np.testing.assert_allclose(got, [[[2., 0.], [1., 1.]]], atol=1e-3)


def test_media_prefill_and_canvas_generation_match_reference(system):
    """Complete image-conditioned forward: observed fp32 maximum error 1.4e-6."""
    model, variables, process, reference, _ = system
    inputs = ModelInputs(
        jnp.asarray(reference["image_prompt"]),
        {"image_indices": jnp.where(reference["image_prompt"] == 60, 0, -1)},
        {"pixel_values": jnp.asarray(reference["pixels"])})
    cache = model.apply(variables, 2, method=model.init_cache, mutable=["cache"])[1]["cache"]
    cache = model.apply(
        {**variables, "cache": cache}, inputs,
        method=lambda module, batch: module.encode(batch.tokens, **batch.kwargs()),
        mutable=["cache"])[1]["cache"]
    logits = model.apply({**variables, "cache": cache}, reference["canvas"])
    np.testing.assert_allclose(np.asarray(logits), reference["image_logits"], atol=1e-4, rtol=0)
    generated = process.generate(model, variables, inputs, 7, key=jax.random.key(11))
    np.testing.assert_array_equal(generated.tokens, reference["image_tokens"][:, :12])
    np.testing.assert_array_equal(generated.decoder_steps, reference["image_steps"])


def test_padded_prefill_keeps_each_rows_logical_cache_position(system):
    model, variables, _, reference, _ = system
    tokens = jnp.array([[0, 0, 2, 3, 4], [2, 5, 7, 9, 11]], jnp.int32)
    mask = jnp.array([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], bool)
    positions = jnp.maximum(jnp.cumsum(mask, axis=1) - 1, 0)
    cache = model.apply(variables, 2, method=model.init_cache, mutable=["cache"])[1]["cache"]
    cache = model.apply({**variables, "cache": cache}, tokens, attention_mask=mask,
                        positions=positions, method=model.encode, mutable=["cache"])[1]["cache"]
    batched = np.asarray(model.apply({**variables, "cache": cache}, reference["canvas"]))
    for row, start in ((0, 2), (1, 0)):
        individual = prefill(model, variables, tokens[row:row + 1, start:])
        expected = model.apply({**variables, "cache": individual}, reference["canvas"][row:row + 1])
        np.testing.assert_allclose(batched[row:row + 1], np.asarray(expected), atol=1e-5, rtol=0)
