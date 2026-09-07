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

