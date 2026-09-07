"""Native processor, model, objective and checkpoint parity on tiny sources.

The reference is the actual Transformers 5.16.1 processor and conditional
model per family, written by tools/multimodal_reference.py without downloads
one row of the batch encoding at a time. Gemma3 carries the Trainer and
mutation coverage; Gemma4, Qwen3.5, Llama4 and Gemma3n (images and audio)
run the same processor, forward, backward, export and cached generation
path, and the two audio-only fixtures pin waveform conditioning. Unequal
image counts and left padding exercise per-row conditioning and cache
addressing.
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
    """Valid-slot fp32 logit error 5.96e-6; tolerance 1e-4, identical argmax."""
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
    """Pixel-gradient max error 2.05e-7 and CE error 4.77e-7 against Torch.

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



FAMILIES = {
    # family: (references, checkpoint, forward error, pixel-gradient error, post-SGD error)
    "gemma4": ("gemma4-native-tiny", "gemma4-native-tiny", 1.34e-5, 2.04e-6, 1.43e-5),
    "qwen35": ("qwen35-native-tiny", "qwen35-native-tiny", 3.46e-5, 2.69e-6, 5.28e-5),
    "llama4": ("llama4-native-tiny", "llama4-native-tiny", 6.76e-6, 5.4e-8, 1.22e-5),
    # Images and audio through the same prompt; the checkpoint and its
    # processor are the audio fixture's, the references sit beside them.
    "gemma3n": ("gemma3n-native-tiny", "gemma-3n-audio-tiny", 1.95e-6, 2.39e-7, 6.41e-5),
}


def _wrong_inputs(family: str, inputs: ModelInputs) -> ModelInputs:
    """One plausible integration bug per family that the parity assertion must catch."""
    if family == "gemma4":
        # The 2D position tables read (x, y); swapping them is the natural slip.
        positions = inputs.conditioning["image_position_ids"]
        return dataclasses.replace(inputs, conditioning={**inputs.conditioning, "image_position_ids": positions[..., ::-1]})
    if family == "qwen35":
        # Text-only rotary coordinates instead of the interleaved spatial ones.
        return dataclasses.replace(inputs, token_fields={
            name: value for name, value in inputs.token_fields.items() if name != "rotary_positions"})
    if family == "gemma3n":
        # The reference hands its audio encoder the inverted mask; getting
        # the polarity wrong keeps every shape.
        mask = inputs.conditioning["input_features_mask"]
        return dataclasses.replace(inputs, conditioning={**inputs.conditioning, "input_features_mask": ~mask})
    # Llama 4 orders local tiles before the global tile; reversing them keeps
    # every shape and count.
    pixels = inputs.conditioning["pixel_values"]
    return dataclasses.replace(inputs, conditioning={**inputs.conditioning, "pixel_values": pixels[:, ::-1]})



def _real_pixels(inputs: ModelInputs, gradient: jax.Array) -> jax.Array:
    """The per-image gradient in the reference's flat image order, without padding."""
    lengths = np.asarray(inputs.conditioning["image_lengths"])
    grid = inputs.conditioning.get("image_grid_thw")
    if grid is None:
        return jnp.concatenate([gradient[row, :int(count)] for row, count in enumerate(lengths)])
    return jnp.concatenate([gradient[row, image, :int(np.prod(np.asarray(grid[row, image])))]
                            for row, count in enumerate(lengths) for image in range(int(count))])


@pytest.fixture(scope="module", params=sorted(FAMILIES))
def family_source(request):
    pytest.importorskip("torchvision", reason="the vision extra supplies the actual processors")
    references, checkpoint, *_ = FAMILIES[request.param]
    directory = FIXTURE.parent / references
    source = FIXTURE.parent / checkpoint
    loaded = load_pretrained(source, dtype="float32", attention_impl="reference")
    images = np.load(directory / "raw_images.npy")
    waveforms = sorted(source.glob("waveform_*.npy"))
    assert loaded.processor is not None
    inputs = loaded.processor(json.loads((directory / "prompts.json").read_text()),
                              images=[[images[0]], [images[1], images[2]]],
                              audio=[np.load(path) for path in waveforms] if waveforms else None)
    return request.param, loaded, inputs, directory


@pytest.fixture(scope="module", params=["gemma-3n-audio-tiny", "gemma-4-audio-tiny"])
def audio_source(request):
    directory = FIXTURE.parent / request.param
    loaded = load_pretrained(directory, dtype="float32", attention_impl="reference")
    meta = json.loads((directory / "meta.json").read_text())
    assert loaded.processor is not None
    inputs = loaded.processor(meta["prompts"], audio=[np.load(directory / f"waveform_{index}.npy") for index in range(2)])
    return loaded, inputs, np.load(directory / "reference.npz")


def test_audio_only_processor_forward_and_greedy_continuation_match_reference(audio_source):
    """Waveforms through the actual Gemma processors and native audio towers.

    Valid-logit errors 3.2e-7 (Gemma 3n, fixed slots and padding embeddings)
    and 2.3e-7 (Gemma 4, one slot per encoded frame) at tolerance 1e-4;
    greedy continuation matches, including Gemma 3n's hard-range samples.
    """
    loaded, inputs, reference = audio_source
    np.testing.assert_array_equal(inputs.tokens, reference["input_ids"])
    valid = np.asarray(inputs.token_fields["attention_mask"])
    logits = jax.jit(lambda variables: loaded.model.apply(variables, inputs.tokens, **inputs.kwargs()))(loaded.variables)
    np.testing.assert_allclose(np.asarray(logits)[valid], reference["logits"][valid], atol=1e-4, rtol=0)
    generated = loaded.generate(inputs, 3, key=jax.random.key(1), generation=Sampling(temperature=0))
    np.testing.assert_array_equal(generated.tokens[:, -3:], reference["generated"][:, -3:])


def test_gemma3n_left_padded_batch_has_finite_gradients():
    """A zero pad embedding at a padded slot must not poison the gradient.

    The AltUp magnitude is a square root of that slot's zero RMS; without a
    finite derivative there the masked slot's zero upstream gradient became
    NaN in the embedding table (observed on the pad row alone).
    """
    loaded = load_pretrained(FIXTURE.parent / "gemma-3n-audio-tiny", dtype="float32", attention_impl="reference")
    model = loaded.model.language_model
    params = {"params": loaded.variables["params"]["language_model"]}
    tokens = jnp.asarray([[0, 0, 0, 5, 4, 7, 9, 3], [8, 4, 9, 7, 3, 2, 5, 6]], jnp.int32)
    valid = tokens != 0
    inputs = ModelInputs(tokens, {"attention_mask": valid,
                                  "positions": jnp.maximum(jnp.cumsum(valid, axis=1) - 1, 0).astype(jnp.int32)}, {})

    def loss(variables):
        logits = model.apply(variables, inputs.tokens, **inputs.kwargs())
        log_probs = jax.nn.log_softmax(logits[:, :-1])
        picked = jnp.take_along_axis(log_probs, tokens[:, 1:, None], -1)[..., 0]
        return -(picked * valid[:, 1:]).sum() / valid[:, 1:].sum()

    gradient = jax.grad(loss)(params)
    assert all(bool(jnp.isfinite(leaf).all()) for leaf in jax.tree.leaves(gradient))



def test_source_processor_forward_and_cached_generation_match_reference(family_source):
    """Actual processors to native models: Gemma4's padded patches with active
    clipping and its video placeholder decode rule, Qwen3.5's packed
    channel-time patches with spatial M-RoPE, Llama4's local and global tiles,
    Gemma3n's MobileNet features, hard vocabulary ranges, fixed audio slots
    and per-layer input masking beside its conformer audio.

    FP32 valid-logit errors are recorded in FAMILIES at tolerance 1e-4; each
    family's characteristic slip must fail the same assertion.
    """
    family, loaded, inputs, directory = family_source
    np.testing.assert_array_equal(inputs.tokens, np.load(directory / "input_ids.npy"))
    logits = jax.jit(lambda variables: loaded.model.apply(
        variables, inputs.tokens, **inputs.kwargs()))(loaded.variables)
    valid = np.asarray(inputs.token_fields["attention_mask"])
    expected = np.load(directory / "logits.npy")
    np.testing.assert_allclose(np.asarray(logits)[valid], expected[valid], atol=1e-4, rtol=0)
    np.testing.assert_array_equal(np.asarray(logits)[valid].argmax(-1), expected[valid].argmax(-1))
    wrong_inputs = _wrong_inputs(family, inputs)
    wrong = loaded.model.apply(loaded.variables, wrong_inputs.tokens, **wrong_inputs.kwargs())
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.asarray(wrong)[valid], expected[valid], atol=1e-4, rtol=0)
    generated = loaded.generate(inputs, 3, key=jax.random.key(1), generation=Sampling(temperature=0))
    np.testing.assert_array_equal(generated.tokens[:, -3:], np.load(directory / "continuation.npy"))


def test_source_backward_trained_export_and_frozen_buffers_match_reference(family_source, tmp_path):
    """Loss, pixel gradient and all-parameter SGD against the reference backward.

    Recorded errors are in FAMILIES. The trained model exports under the
    source's tensor names and reloads to the reference's updated logits; every
    non-parameter collection (Gemma4's standardization and clipping buffers)
    survives training bitwise. Qwen3.5's root tie_word_embeddings=False
    overrides its nested text flag: tying the head keeps the initial logits
    but misses this update by 0.00167.
    """
    _, loaded, inputs, directory = family_source
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
    np.testing.assert_allclose(_real_pixels(inputs, pixels), np.load(directory / "pixel_gradient.npy"),
                               atol=1e-5, rtol=1e-4)
    optimizer = optax.sgd(expected["learning_rate"])
    params = loaded.variables["params"]
    updates, _ = optimizer.update(gradient, optimizer.init(params), params)
    updated = {**loaded.variables, "params": optax.apply_updates(params, updates)}
    loaded.save(tmp_path, variables=updated)
    restored = load_pretrained(tmp_path, dtype="float32", attention_impl="reference")
    output = restored.model.apply(restored.variables, inputs.tokens, **inputs.kwargs())
    valid = np.asarray(inputs.token_fields["attention_mask"])
    np.testing.assert_allclose(np.asarray(output)[valid], np.load(directory / "updated_logits.npy")[valid], atol=1e-4, rtol=0)
    for collection, tree in restored.variables.items():
        if collection != "params":
            for actual, original in zip(jax.tree.leaves(tree), jax.tree.leaves(loaded.variables[collection])):
                np.testing.assert_array_equal(actual, original)
