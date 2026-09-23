"""Published DiffusionGemma inference with tiny released-code reference weights.

The full checkpoint contains sliding/full attention, routed experts, shared
encoder/decoder weights, self-conditioning, and a vision tower. The committed
Transformers generate trajectory uses matched random inputs, not an assumed
identity between Torch and JAX seeds (tools/diffusion_gemma_reference.py).
"""

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference
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
    with np.load(WORKFLOW / "numerics.npz") as stored:
        reference.update({name: stored[name] for name in stored.files})
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
    """Bare and self-conditioned canvas logits, held to twice the fp32
    reference's own distance from its float64 run (tests/reference_error.py).
    Observed RMS: bare 3.5e-07 against the reference's 3.3e-07, conditioned
    3.5e-07 against 3.7e-07.

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
    assert_as_exact_as_the_reference(bare, reference["bare"], reference["bare_f64"], "bare")
    assert_as_exact_as_the_reference(conditioned, reference["conditioned"],
                                     reference["conditioned_f64"], "conditioned")
    np.testing.assert_array_equal(jnp.argmax(conditioned, -1), reference["conditioned"].argmax(-1))
    for expected, actual in zip(jax.tree.leaves(before), jax.tree.leaves(cache), strict=True):
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


def test_canvas_continuations_refine_the_shared_prefill_without_moving_the_first_one(system):
    """Several continuations per prompt refine the encoded prompt again, each
    over the original prompt rows: continuation zero is the reference loop's
    canvas row for row, the others are different canvases, rows stay prompt
    major and growing the count leaves the earlier continuations alone."""
    model, variables, process, reference, _ = system
    inputs = ModelInputs(jnp.asarray(reference["prompt"]))
    eos = (int(reference["eos_id"]),)

    def draw(count):
        return process.generate(model, variables, inputs, 7, key=jax.random.key(11),
                                n=count, eos_token_ids=eos, pad_token_id=0)

    one, two, three = draw(1), draw(2), draw(3)

    assert two.tokens.shape == (4, 12) and two.rows == 4 and two.prompt_width == 5
    np.testing.assert_array_equal(np.asarray(two.tokens)[:, :5],
                                  np.repeat(reference["prompt"], 2, axis=0))
    for field in ("tokens", "lengths", "terminated", "decoder_steps"):
        single = np.asarray(getattr(one, field))
        np.testing.assert_array_equal(np.asarray(getattr(two, field))[::2], single)
        np.testing.assert_array_equal(np.asarray(getattr(three, field))[::3], single)
        np.testing.assert_array_equal(
            np.asarray(getattr(three, field)).reshape((2, 3, *single.shape[1:]))[:, 1],
            np.asarray(getattr(two, field)).reshape((2, 2, *single.shape[1:]))[:, 1])
    assert not np.array_equal(np.asarray(two.tokens)[0, 5:], np.asarray(two.tokens)[1, 5:])
    # An unrequested canvas is still no canvas: the prompts repeat and nothing
    # is refined for any continuation.
    empty = process.generate(model, variables, inputs, 0, key=jax.random.key(11), n=2)
    np.testing.assert_array_equal(empty.tokens, np.repeat(reference["prompt"], 2, axis=0))
    np.testing.assert_array_equal(empty.decoder_steps, [0, 0, 0, 0])
    with pytest.raises(ValueError, match="positive integer"):
        draw(0)


@pytest.mark.parametrize("budget", [0, 3, 7])
def test_canvas_generation_can_resume_at_committed_boundaries(system, budget):
    from dew.diffusion.block import CanvasPlan, _advance, _begin, _materialize

    model, variables, process, reference, _ = system
    inputs = ModelInputs(jnp.asarray(reference["prompt"]))
    eos_ids = (int(reference["eos_id"]),)
    expected = process.generate(model, variables, inputs, budget, key=jax.random.key(11),
                                eos_token_ids=eos_ids, pad_token_id=0)
    plan = CanvasPlan(process, eos_ids, 0, budget)
    begin = jax.jit(lambda weights, data: _begin(model, weights, data, plan))
    advance = jax.jit(lambda weights, state: _advance(model, weights, state, jax.random.key(11), plan))
    state = begin(variables, inputs)
    for _ in range(plan.blocks):
        state = advance(variables, jax.device_get(state))
    result = _materialize(state, inputs.tokens.shape[1], budget)
    np.testing.assert_array_equal(result.tokens, expected.tokens)
    np.testing.assert_array_equal(result.lengths, expected.lengths)
    np.testing.assert_array_equal(result.terminated, expected.terminated)
    np.testing.assert_array_equal(result.decoder_steps, expected.decoder_steps)


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
    for wanted, actual in zip(jax.tree.leaves(changed), jax.tree.leaves(restored), strict=True):
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


def test_a_one_term_change_to_self_conditioning_fails_the_reference(system, monkeypatch):
    """The bound above is tight enough for the self-conditioning branch: its
    GELU in the erf form instead of the reference's tanh form puts the
    conditioned logits 124 times the reference's rounding away, while the
    bare logits, whose signal is zero, stay inside it."""
    model, variables, _, reference, _ = system

    def erf_gelu(self, inputs_embeds, signal):
        normed = self.pre_norm(signal)
        gated = self.down_proj(jax.nn.gelu(self.gate_proj(normed), approximate=False)
                               * self.up_proj(normed))
        return self.post_norm(inputs_embeds + gated)

    monkeypatch.setattr(SelfConditioning, "__call__", erf_gelu)
    cache = prefill(model, variables, reference["prompt"])
    bare = model.apply({**variables, "cache": cache}, reference["canvas"])
    conditioned = model.apply({**variables, "cache": cache}, reference["canvas"],
                              self_conditioning_logits=reference["previous"])
    assert_as_exact_as_the_reference(bare, reference["bare"], reference["bare_f64"], "bare")
    with pytest.raises(AssertionError, match="ratio"):
        assert_as_exact_as_the_reference(conditioned, reference["conditioned"],
                                         reference["conditioned_f64"], "conditioned")


def test_the_released_layout_denoiser_matches_the_reference():
    """tools/hf_reference.py's diffusion-gemma-denoise-tiny: the released
    checkpoint layout (text weights under both the encoder and the decoder
    prefix, a separate head, two full layers with routed experts), read
    through the public wrapper translation. One prompt of four, one canvas of
    four, bare and self-conditioned, each held to twice the fp32 reference's
    distance from float64: observed RMS bare 4.4e-08 against 4.3e-08,
    conditioned 5.2e-08 against 6.0e-08."""
    directory = FIXTURES / "diffusion-gemma-denoise-tiny"
    config = {"model_type": "diffusion_gemma", "canvas_length": 4,
              "text_config": json.loads((directory / "config.json").read_text())}
    model = adapter.build(config, dtype="float32", attention_impl="xla")
    variables = adapter.translate_weights(load_file(str(directory / "model.safetensors")), config)
    cache = prefill(model, variables, np.load(directory / "prompt.npy"))
    canvas = np.load(directory / "canvas.npy")
    with np.load(directory / "numerics.npz") as exact:
        for name, previous in (("ref_bare", None), ("ref_conditioned", np.load(directory / "prev_logits.npy"))):
            logits = model.apply({**variables, "cache": cache}, canvas, self_conditioning_logits=previous)
            reference = np.load(directory / f"{name}.npy")
            assert_as_exact_as_the_reference(logits, reference, exact[f"{name}_f64"], name)
            np.testing.assert_array_equal(np.argmax(logits, -1), reference.argmax(-1))


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
    """Complete image-conditioned forward, held to twice the fp32 reference's
    distance from float64: observed RMS 3.3e-07 against the reference's 2.9e-07."""
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
    assert_as_exact_as_the_reference(logits, reference["image_logits"],
                                     reference["image_logits_f64"], "image")
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


def test_canvas_generation_rejects_lossy_token_coercion_and_batched_keys(system):
    model, variables, process, reference, _ = system
    with pytest.raises(ValueError, match="integer"):
        process.generate(model, variables, reference["prompt"].astype(np.float32),
                         3, key=jax.random.key(1))
    with pytest.raises(ValueError, match="PRNG key"):
        process.generate(model, variables, reference["prompt"], 3,
                         key=jax.random.split(jax.random.key(1), 2))
