"""Native processor, model, objective and checkpoint parity on tiny Gemma3.

The reference is the actual Transformers 5.16.1 processor and conditional
model, written by tools/multimodal_reference.py without downloads. Four soft
tokens per image exercise image bidirectionality; unequal image counts and
left padding exercise per-row conditioning and cache addressing.
"""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from safetensors.numpy import load_file

from dew.data.dataset import Dataset
from dew.interop.pretrained import load_pretrained
from dew.nn.inputs import ModelInputs
from dew.nn.mixers.attention import AttentionMixer
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.sampling.text import Sampling
from dew.training import Layout, MeshSpec, Trainer


FIXTURE = Path(__file__).parent / "fixtures" / "hf" / "gemma3-native-tiny"


@pytest.fixture(scope="module")
def source():
    loaded = load_pretrained(FIXTURE, dtype="float32", attention_impl="reference")
    images = np.load(FIXTURE / "raw_images.npy")
    prompts = json.loads((FIXTURE / "prompts.json").read_text())
    assert loaded.processor is not None
    inputs = loaded.processor(prompts, images=[[images[0]], [images[1], images[2]]])
    return loaded, inputs


def test_public_processor_and_native_forward_match_conditional_model(source):
    """Valid-slot fp32 logit error 9.30e-6; tolerance 1e-4, identical argmax."""
    loaded, inputs = source
    np.testing.assert_array_equal(inputs.tokens, np.load(FIXTURE / "input_ids.npy"))
    model = loaded.model
    output = jax.jit(lambda variables: model.apply(variables, inputs.tokens, **inputs.kwargs()))(loaded.variables)
    valid = np.asarray(inputs.token_fields["attention_mask"])
    reference = np.load(FIXTURE / "logits.npy")
    np.testing.assert_allclose(np.asarray(output)[valid], reference[valid], atol=1e-4, rtol=0)
    np.testing.assert_array_equal(np.asarray(output)[valid].argmax(-1), reference[valid].argmax(-1))
    causal_images = model.clone(language_model=model.language_model.clone(mixer=AttentionMixer()))
    wrong = causal_images.apply(loaded.variables, inputs.tokens, **inputs.kwargs())
    assert np.max(np.abs(np.asarray(wrong)[valid] - reference[valid])) > 1e-2


def test_objective_pixel_gradient_matches_reference(source):
    """Pixel-gradient max error 2.67e-7 and CE error 1.44e-6 against Torch.

    The gradient crosses image encoder, projection, decoder and shifted loss.
    """
    loaded, inputs = source
    objective = LMObjective(loaded.model, inputs.tokens.shape[1] - 1,
                            pretrained=loaded.variables, ema_decay=None, pad_id=0)
    step = Step(step=jnp.int32(0), key=jax.random.key(4), ema=None)

    def loss(pixels):
        prepared = dataclasses.replace(inputs, conditioning={**inputs.conditioning, "pixel_values": pixels})
        statistics, _ = objective.loss(loaded.variables, {"text": prepared}, step)
        return objective.reduce_loss(statistics)[0]

    value, gradient = jax.jit(jax.value_and_grad(loss))(inputs.conditioning["pixel_values"])
    reference = json.loads((FIXTURE / "training.json").read_text())
    np.testing.assert_allclose(value, reference["loss"], atol=1e-5, rtol=0)
    real_images = jnp.stack([gradient[0, 0], gradient[1, 0], gradient[1, 1]])
    np.testing.assert_allclose(real_images, np.load(FIXTURE / "pixel_gradient.npy"), atol=1e-5, rtol=1e-4)
    np.testing.assert_array_equal(gradient[0, 1], jnp.zeros_like(gradient[0, 1]))


def test_trainer_update_exports_and_reloads_the_complete_model(source, tmp_path):
    """One real Trainer update agrees with the reference's all-parameter SGD."""
    loaded, inputs = source
    reference = json.loads((FIXTURE / "training.json").read_text())
    objective = LMObjective(loaded.model, inputs.tokens.shape[1] - 1,
                            pretrained=loaded.variables, ema_decay=None, pad_id=0)
    rows = 2 * jax.device_count()
    training_inputs = inputs.take_rows(jnp.arange(rows) % 2)
    data = Dataset(train=lambda: iter([{"text": training_inputs}]), val=None, records=rows, batch=rows)
    trainer = Trainer(objective, optax.sgd(reference["learning_rate"]), key=jax.random.key(3),
                      mesh=MeshSpec(), layout=Layout(min_shard=2**30))
    state = trainer.fit(data, steps=1, log_every=1)
    output = loaded.model.apply(state.params, inputs.tokens, **inputs.kwargs())
    valid = np.asarray(inputs.token_fields["attention_mask"])
    expected = np.load(FIXTURE / "updated_logits.npy")
    np.testing.assert_allclose(np.asarray(output)[valid], expected[valid], atol=1e-4, rtol=0)
    before = np.load(FIXTURE / "logits.npy")
    assert np.max(np.abs(np.asarray(output)[valid] - before[valid])) > 1e-3
    loaded.save(tmp_path, variables=state.params)
    reloaded = load_pretrained(tmp_path, dtype="float32", attention_impl="reference")
    for actual, wanted in zip(jax.tree.leaves(reloaded.variables), jax.tree.leaves(state.params)):
        np.testing.assert_array_equal(actual, wanted)
    restored = reloaded.model.apply(reloaded.variables, inputs.tokens, **inputs.kwargs())
    np.testing.assert_allclose(np.asarray(restored)[valid], expected[valid], atol=1e-4, rtol=0)
    assert reloaded.processor is not None
    assert loaded.processor is not None
    assert reloaded.processor.decode(inputs.tokens) == loaded.processor.decode(inputs.tokens)


def test_model_inputs_alignment_preserves_media_identity(source):
    loaded, inputs = source
    padding = (~inputs.token_fields["attention_mask"]).sum(axis=1)
    aligned = jax.jit(lambda value: value.align_left(padding).take_rows(jnp.array([1, 0])))(inputs)
    np.testing.assert_array_equal(aligned.conditioning["pixel_values"], inputs.conditioning["pixel_values"][jnp.array([1, 0])])
    for row, old_row in enumerate((1, 0)):
        start = int(padding[old_row])
        np.testing.assert_array_equal(aligned.token_fields["image_indices"][row, :inputs.tokens.shape[1] - start],
                                      inputs.token_fields["image_indices"][old_row, start:])
    assert isinstance(aligned, ModelInputs)


def test_source_processor_rejects_unknown_fields_and_incorrect_image_counts(source):
    loaded, _ = source
    assert loaded.processor is not None
    values = {name: np.load(FIXTURE / f"{name}.npy") for name in
              ("input_ids", "attention_mask", "pixel_values", "token_type_ids")}
    with pytest.raises(ValueError, match="processor fields"):
        loaded.processor.from_hf({**values, "mystery_mask": np.ones((2, 19))})
    with pytest.raises(ValueError, match="placeholder counts disagree"):
        loaded.processor.from_hf({**values, "pixel_values": values["pixel_values"][:1]})


def test_public_cached_generation_preserves_image_conditioning(source):
    """Greedy continuation matches HF; changing pixels changes raw likelihoods."""
    loaded, inputs = source
    generated = loaded.generate(inputs, 3, key=jax.random.key(1), generation=Sampling(temperature=0))
    np.testing.assert_array_equal(generated.tokens[:, -3:], np.load(FIXTURE / "continuation.npy"))
    np.testing.assert_array_equal(generated.lengths, [3, 3])
    blank = dataclasses.replace(inputs, conditioning={
        **inputs.conditioning, "pixel_values": jnp.zeros_like(inputs.conditioning["pixel_values"])})
    altered = loaded.generate(blank, 3, key=jax.random.key(1), generation=Sampling(temperature=0))
    # The observed maximum change is 0.0840, even though greedy tokens agree.
    assert np.max(np.abs(generated.raw_log_probs - altered.raw_log_probs)) > 1e-2



def test_gemma4_standardization_buffers_are_frozen_by_real_adamw_training(tmp_path):
    """The exported HF buffers stay bitwise unchanged while AdamW updates weights."""
    directory = FIXTURE.parent / "gemma4-tiny-mm"
    loaded = load_pretrained(directory, dtype="float32", attention_impl="reference")
    patches = np.load(directory / "pixels.npy")
    batch, count, _ = patches.shape
    side = int(count ** 0.5)
    patch = loaded.model.vision.patch_size
    images = patches.reshape(batch, side, side, patch, patch, 3).transpose(
        0, 5, 1, 3, 2, 4).reshape(batch, 1, 3, side * patch, side * patch)
    tokens = jnp.asarray(np.load(directory / "input_ids.npy"))
    marks = tokens == loaded.model.image_token_id
    inputs = ModelInputs(tokens, {"image_indices": jnp.where(marks, jnp.cumsum(marks, axis=1) - 1, -1)},
                         {"pixel_values": jnp.asarray(images)})
    rows = 2 * jax.device_count()
    repeated = inputs.take_rows(jnp.arange(rows) % 2)
    objective = LMObjective(loaded.model, tokens.shape[1] - 1, pretrained=loaded.variables,
                            ema_decay=None, pad_id=0)
    data = Dataset(train=lambda: iter([{"text": repeated}]), val=None, records=rows, batch=rows)
    trainer = Trainer(objective, optax.adamw(1e-3, weight_decay=0.1), key=jax.random.key(12),
                      mesh=MeshSpec(), layout=Layout(min_shard=2**30))
    state = trainer.fit(data, steps=1, log_every=1)
    loaded.save(tmp_path, variables=state.params)
    before = load_file(str(directory / "model.safetensors"))
    after = load_file(str(tmp_path / "model.safetensors"))
    for name in ("std_bias", "std_scale"):
        key = "model.vision_tower." + name
        np.testing.assert_array_equal(after[key], before[key])
    for name in before:
        if name.endswith("layer_scalar"):
            np.testing.assert_array_equal(after[name], before[name])
    trained = "model.vision_tower.patch_embedder.input_proj.weight"
    assert np.max(np.abs(after[trained] - before[trained])) > 1e-4



@pytest.fixture(scope="module")
def gemma4_source():
    pytest.importorskip("torchvision", reason="the optional vision extra supplies the actual Gemma4 processor")
    directory = FIXTURE.parent / "gemma4-native-tiny"
    loaded = load_pretrained(directory, dtype="float32", attention_impl="reference")
    images = np.load(directory / "raw_images.npy")
    assert loaded.processor is not None
    inputs = loaded.processor(json.loads((directory / "prompts.json").read_text()),
                              images=[[images[0]], [images[1], images[2]]])
    return loaded, inputs, directory


def test_gemma4_processor_clipping_and_cached_generation_match_reference(gemma4_source):
    """Actual padded processor patches, active clipping, image masks and decode.

    FP32 valid-logit max error 1.35e-5, tolerance 1e-4. The reference samples
    video placeholder ID56; its next decode input must use the pad embedding.
    """
    loaded, inputs, directory = gemma4_source
    np.testing.assert_array_equal(inputs.tokens, np.load(directory / "input_ids.npy"))
    logits = jax.jit(lambda variables: loaded.model.apply(
        variables, inputs.tokens, **inputs.kwargs()))(loaded.variables)
    valid = np.asarray(inputs.token_fields["attention_mask"])
    expected = np.load(directory / "logits.npy")
    np.testing.assert_allclose(np.asarray(logits)[valid], expected[valid], atol=1e-4, rtol=0)
    np.testing.assert_array_equal(np.asarray(logits)[valid].argmax(-1), expected[valid].argmax(-1))
    generated = loaded.generate(inputs, 3, key=jax.random.key(1), generation=Sampling(temperature=0))
    np.testing.assert_array_equal(generated.tokens[:, -3:], np.load(directory / "continuation.npy"))


def test_gemma4_source_backward_and_trained_export_match_reference(gemma4_source, tmp_path):
    """Pixel-gradient max error 1.88e-6 against actual Transformer backward."""
    loaded, inputs, directory = gemma4_source
    objective = LMObjective(loaded.model, inputs.tokens.shape[1] - 1,
                            pretrained=loaded.variables, ema_decay=None, pad_id=0)
    step = Step(step=jnp.int32(0), key=jax.random.key(4), ema=None)

    def loss(params, pixels):
        values = {**loaded.variables, "params": params}
        data = dataclasses.replace(inputs, conditioning={**inputs.conditioning, "pixel_values": pixels})
        statistics, _ = objective.loss(values, {"text": data}, step)
        return objective.reduce_loss(statistics)[0]

    value, (gradient, pixels) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(
        loaded.variables["params"], inputs.conditioning["pixel_values"])
    expected = json.loads((directory / "training.json").read_text())
    np.testing.assert_allclose(value, expected["loss"], atol=1e-5, rtol=0)
    real = jnp.stack([pixels[0, 0], pixels[1, 0], pixels[1, 1]])
    np.testing.assert_allclose(real, np.load(directory / "pixel_gradient.npy"), atol=1e-5, rtol=1e-4)
    optimizer = optax.sgd(expected["learning_rate"])
    params = loaded.variables["params"]
    updates, _ = optimizer.update(gradient, optimizer.init(params), params)
    updated = {**loaded.variables, "params": optax.apply_updates(params, updates)}
    loaded.save(tmp_path, variables=updated)
    restored = load_pretrained(tmp_path, dtype="float32", attention_impl="reference")
    output = restored.model.apply(restored.variables, inputs.tokens, **inputs.kwargs())
    valid = np.asarray(inputs.token_fields["attention_mask"])
    np.testing.assert_allclose(np.asarray(output)[valid], np.load(directory / "updated_logits.npy")[valid], atol=1e-4, rtol=0)
    original = load_file(str(directory / "model.safetensors"))
    saved = load_file(str(tmp_path / "model.safetensors"))
    for name, tensor in original.items():
        if name.endswith(("std_bias", "std_scale", "input_min", "input_max", "output_min", "output_max")):
            np.testing.assert_array_equal(saved[name], tensor)



@pytest.fixture(scope="module")
def qwen_source():
    pytest.importorskip("torchvision", reason="the vision extra supplies the actual Qwen processor")
    directory = FIXTURE.parent / "qwen35-native-tiny"
    loaded = load_pretrained(directory, dtype="float32", attention_impl="reference")
    images = np.load(directory / "raw_images.npy")
    assert loaded.processor is not None
    inputs = loaded.processor(json.loads((directory / "prompts.json").read_text()),
                              images=[[images[0]], [images[1], images[2]]])
    return loaded, inputs, directory


def test_qwen_processor_spatial_rotary_and_cached_generation_match_reference(qwen_source):
    """Actual Qwen3VLProcessor to hybrid decoder; fp32 max error 4.37e-5.

    The interleaved spatial positions survive cached continuation. Removing
    them must fail the same 1e-4 reference assertion.
    """
    loaded, inputs, directory = qwen_source
    valid = np.asarray(inputs.token_fields["attention_mask"])
    expected = np.load(directory / "logits.npy")
    output = jax.jit(lambda variables: loaded.model.apply(
        variables, inputs.tokens, **inputs.kwargs()))(loaded.variables)
    np.testing.assert_allclose(np.asarray(output)[valid], expected[valid], atol=1e-4, rtol=0)
    np.testing.assert_array_equal(np.asarray(output)[valid].argmax(-1), expected[valid].argmax(-1))
    without_spatial = dataclasses.replace(inputs, token_fields={
        name: value for name, value in inputs.token_fields.items() if name != "rotary_positions"})
    wrong = loaded.model.apply(loaded.variables, without_spatial.tokens, **without_spatial.kwargs())
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.asarray(wrong)[valid], expected[valid], atol=1e-4, rtol=0)
    generated = loaded.generate(inputs, 3, key=jax.random.key(1), generation=Sampling(temperature=0))
    np.testing.assert_array_equal(generated.tokens[:, -3:], np.load(directory / "continuation.npy"))


def test_qwen_source_head_ownership_backward_and_export_match_reference(qwen_source, tmp_path):
    """Root tie_word_embeddings=False overrides the nested text flag.

    Pixel-gradient error 1.60e-6 and post-SGD logit error 4.14e-5. Incorrectly
    tying the head preserves the initial logits but misses this update by
    0.00167, so forward-only evidence cannot cover this source contract.
    """
    loaded, inputs, directory = qwen_source
    objective = LMObjective(loaded.model, inputs.tokens.shape[1] - 1,
                            pretrained=loaded.variables, ema_decay=None, pad_id=0)
    step = Step(step=jnp.int32(0), key=jax.random.key(0), ema=None)

    def loss(params, pixels):
        data = dataclasses.replace(inputs, conditioning={**inputs.conditioning, "pixel_values": pixels})
        statistics, _ = objective.loss({"params": params}, {"text": data}, step)
        return objective.reduce_loss(statistics)[0]

    value, (gradient, pixels) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(
        loaded.variables["params"], inputs.conditioning["pixel_values"])
    expected = json.loads((directory / "training.json").read_text())
    np.testing.assert_allclose(value, expected["loss"], atol=1e-5, rtol=0)
    real = jnp.concatenate([pixels[0, 0], pixels[1, 0], pixels[1, 1]])
    np.testing.assert_allclose(real, np.load(directory / "pixel_gradient.npy"), atol=1e-5, rtol=1e-4)
    optimizer = optax.sgd(expected["learning_rate"])
    params = loaded.variables["params"]
    updates, _ = optimizer.update(gradient, optimizer.init(params), params)
    updated = {"params": optax.apply_updates(params, updates)}
    loaded.save(tmp_path, variables=updated)
    restored = load_pretrained(tmp_path, dtype="float32", attention_impl="reference")
    output = restored.model.apply(restored.variables, inputs.tokens, **inputs.kwargs())
    valid = np.asarray(inputs.token_fields["attention_mask"])
    np.testing.assert_allclose(np.asarray(output)[valid], np.load(directory / "updated_logits.npy")[valid], atol=1e-4, rtol=0)

