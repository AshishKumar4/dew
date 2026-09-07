"""Source-matched Qwen3.8 checkpoints using their actual Qwen3.5 model types.

The source revisions, configs, templates and weight indexes live in
qwen38-source. The tiny fixtures come from Transformers 5.16.1 through
qwen_qualification_reference.py; no model weights are downloaded at test time.
"""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop import load_pretrained
from dew.interop.hf_decoders import translate_config, translate_wrapper_config
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.sampling import Sampling, generate

ROOT = Path(__file__).parent / "fixtures" / "hf"


@pytest.fixture(scope="module", params=["dense", "moe"])
def source(request):
    if request.param == "dense":
        pytest.importorskip("torchvision")
    directory = ROOT / f"qwen38-{request.param}-tiny"
    loaded = load_pretrained(directory, dtype="float32", attention_impl="reference")
    assert loaded.processor is not None
    prompts = json.loads((directory / "prompts.json").read_text())
    if request.param == "dense":
        inputs = loaded.processor(prompts, images=[[image] for image in np.load(directory / "images.npy")])
    else:
        inputs = loaded.processor(prompts)
    return loaded, inputs, np.load(directory / "reference.npz")


def test_source_geometry_is_configurable_without_allocating_released_weights():
    dense = translate_wrapper_config(json.loads((ROOT / "qwen38-source/dense/config.json").read_text()))
    moe = translate_config(json.loads((ROOT / "qwen38-source/moe/config.json").read_text()))
    dense_model = models.build("causal_transformer", **dense["text"])
    moe_model = models.build("causal_transformer", **moe)
    assert dense_model.num_layers == 64 and dense_model.mixture is None
    assert moe_model.num_layers == 92
    assert moe_model.mixture.experts == 512 and moe_model.mixture.top_k == 10
    assert moe_model.mixture.shared_gate


def test_source_processor_and_forward_match_reference(source):
    """fp32 max error 7.04e-6 dense / 2.19e-6 MoE; tolerance 1e-4."""
    loaded, inputs, reference = source
    np.testing.assert_array_equal(inputs.tokens, reference["input_ids"])
    valid = np.asarray(inputs.token_fields["attention_mask"])
    result = jax.jit(lambda variables: loaded.model.apply(variables, inputs.tokens, **inputs.kwargs()))(loaded.variables)
    np.testing.assert_allclose(np.asarray(result)[valid], reference["logits"][valid], atol=1e-4, rtol=0)
    np.testing.assert_array_equal(np.asarray(result)[valid].argmax(-1), reference["logits"][valid].argmax(-1))


def test_source_update_exports_and_decodes_as_reference(source, tmp_path):
    """All-parameter SGD error 7.90e-6 dense / 3.46e-6 MoE; greedy ids exact.

    This catches reversed fused-expert transposes, incorrect shared gating
    gradients and source layouts that drop trained parameters on export.
    """
    loaded, inputs, reference = source
    objective = LMObjective(loaded.model, inputs.tokens.shape[1] - 1,
                            pretrained=loaded.variables, ema_decay=None,
                            pad_id=int(loaded.generation_config["pad_token_id"]))
    step = Step(step=jnp.int32(0), key=jax.random.key(0), ema=None)

    def loss(params):
        statistics, _ = objective.loss({**loaded.variables, "params": params}, {"text": inputs}, step)
        return objective.reduce_loss(statistics)[0]

    value, gradient = jax.jit(jax.value_and_grad(loss))(loaded.variables["params"])
    np.testing.assert_allclose(value, reference["loss"], atol=1e-5, rtol=0)
    variables = {**loaded.variables, "params": jax.tree.map(
        lambda weight, grad: weight - reference["learning_rate"] * grad,
        loaded.variables["params"], gradient)}
    loaded.save(tmp_path, variables=variables)
    restored = load_pretrained(tmp_path, dtype="float32", attention_impl="reference")
    for before, after in zip(jax.tree.leaves(variables), jax.tree.leaves(restored.variables), strict=True):
        np.testing.assert_array_equal(before, after)
    output = restored.model.apply(restored.variables, inputs.tokens, **inputs.kwargs())
    valid = np.asarray(inputs.token_fields["attention_mask"])
    np.testing.assert_allclose(np.asarray(output)[valid], reference["updated_logits"][valid], atol=1e-4, rtol=0)
    generated = generate(loaded.model, loaded.variables, inputs, 3, key=jax.random.key(1),
                         sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(generated.tokens[:, -3:], reference["generated"][:, -3:])


def test_moe_shared_gate_cannot_be_replaced_by_an_ungated_branch():
    loaded = load_pretrained(ROOT / "qwen38-moe-tiny", dtype="float32", attention_impl="reference")
    reference = np.load(ROOT / "qwen38-moe-tiny/reference.npz")
    wrong = loaded.model.clone(mixture=dataclasses.replace(loaded.model.mixture, shared_gate=False))
    output = wrong.apply(loaded.variables, reference["input_ids"], attention_mask=reference["attention_mask"].astype(bool))
    valid = reference["attention_mask"].astype(bool)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.asarray(output)[valid], reference["logits"][valid], atol=1e-4, rtol=0)



def test_shipped_prediction_weights_match_published_composition(source):
    """MTP error 4.06e-6 dense / 1.32e-6 MoE with the real HF decoder layer.

    Source glue is pinned in qwen_mtp_reference.py to vLLM's forward. The
    embedding/hidden norms cannot be swapped even when their shapes match.
    """
    loaded, inputs, _ = source
    reference = np.load(loaded.source / "mtp_reference.npz")
    hidden = loaded.model.apply(loaded.variables, inputs.tokens,
                                 method=loaded.model.hidden_states, **inputs.kwargs())
    predictions = loaded.model.apply(loaded.variables, hidden, inputs.tokens,
                                      method=loaded.model.mtp_logits, **inputs.kwargs())[0]
    valid = reference["valid"]
    np.testing.assert_allclose(np.asarray(predictions)[valid], reference["logits"][valid], atol=1e-4, rtol=0)
    np.testing.assert_array_equal(np.asarray(predictions)[valid].argmax(-1), reference["logits"][valid].argmax(-1))
    changed = jax.tree.map(lambda value: value, loaded.variables)
    text = changed["params"].get("language_model", changed["params"])
    depth = text["mtp_0"]
    depth["enorm"], depth["hnorm"] = depth["hnorm"], depth["enorm"]
    wrong = loaded.model.apply(changed, hidden, inputs.tokens,
                                method=loaded.model.mtp_logits, **inputs.kwargs())[0]
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.asarray(wrong)[valid], reference["logits"][valid], atol=1e-4, rtol=0)


def test_prediction_cache_matches_reference_candidate_steps(source):
    """Independent cached MTP max errors 4.77e-7 dense / 4.18e-7 MoE."""
    loaded, inputs, _ = source
    reference = np.load(loaded.source / "mtp_reference.npz")
    positions = reference["positions"][..., 1:]
    rotary = np.moveaxis(positions, 0, -1) if positions.ndim == 3 else None
    state = loaded.model.apply(loaded.variables, inputs.tokens.shape[0],
                                method=loaded.model.init_mtp_cache, mutable=["cache"])[1]
    chunks = []
    for index in range(inputs.tokens.shape[1] - 1):
        options = {"attention_mask": jnp.asarray(reference["valid"][:, index:index + 1]),
                   "input_embeddings": jnp.asarray(reference["embeddings"][:, index + 1:index + 2])}
        if rotary is None:
            options["positions"] = jnp.asarray(positions[:, index:index + 1])
        else:
            options["rotary_positions"] = jnp.asarray(rotary[:, index:index + 1])
        logits, state = loaded.model.apply(
            {**loaded.variables, **state}, jnp.asarray(reference["hidden"][:, index:index + 1]),
            inputs.tokens[:, index + 1:index + 2], decode=True,
            method=loaded.model.mtp_step, mutable=["cache"], **options)
        chunks.append(np.asarray(logits))
    result = np.concatenate(chunks, axis=1)
    valid = reference["valid"]
    np.testing.assert_allclose(result[valid], reference["logits"][valid], atol=1e-4, rtol=0)


def test_prediction_loss_respects_padding_and_exports_trained_depth(source, tmp_path):
    """Main+0.2*MTP CE loss error below 1e-6; post-update logits below 4.3e-6.

    The reference masks every consumed token of a depth's shifted target.
    Padded hidden states must not contribute auxiliary supervision.
    """
    loaded, inputs, _ = source
    reference = np.load(loaded.source / "mtp_reference.npz")
    objective = LMObjective(loaded.model, inputs.tokens.shape[1] - 1,
                            pretrained=loaded.variables, ema_decay=None, mtp_weight=0.2,
                            pad_id=int(loaded.generation_config["pad_token_id"]))
    step = Step(step=jnp.int32(0), key=jax.random.key(0), ema=None)

    def loss(params):
        statistics, _ = objective.loss({**loaded.variables, "params": params}, {"text": inputs}, step)
        return objective.reduce_loss(statistics)[0]

    value, gradient = jax.jit(jax.value_and_grad(loss))(loaded.variables["params"])
    np.testing.assert_allclose(value, reference["loss"], atol=1e-5, rtol=0)
    variables = {**loaded.variables, "params": jax.tree.map(
        lambda weight, grad: weight - 1e-4 * grad, loaded.variables["params"], gradient)}
    loaded.save(tmp_path, variables=variables)
    restored = load_pretrained(tmp_path, dtype="float32", attention_impl="reference")
    hidden = restored.model.apply(restored.variables, inputs.tokens,
                                   method=restored.model.hidden_states, **inputs.kwargs())
    logits = restored.model.apply(restored.variables, inputs.tokens, **inputs.kwargs())
    predictions = restored.model.apply(restored.variables, hidden, inputs.tokens,
                                        method=restored.model.mtp_logits, **inputs.kwargs())[0]
    valid = np.asarray(inputs.token_fields["attention_mask"])
    np.testing.assert_allclose(np.asarray(logits)[valid], reference["updated_logits"][valid], atol=1e-4, rtol=0)
    np.testing.assert_allclose(np.asarray(predictions)[reference["valid"]],
                               reference["updated_mtp_logits"][reference["valid"]], atol=1e-4, rtol=0)



@pytest.fixture(scope="module")
def video_source():
    pytest.importorskip("torchvision")
    path = ROOT / "qwen38-dense-tiny"
    loaded = load_pretrained(path, dtype="float32", attention_impl="reference")
    assert loaded.processor is not None
    raw = json.loads((path / "video_inputs.json").read_text())
    inputs = loaded.processor(raw["prompts"], images=[[image] for image in np.load(path / "images.npy")],
                              videos=[np.load(path / f"video_{index}.npy") for index in range(2)],
                              video_metadata=raw["metadata"])
    return loaded, inputs, np.load(path / "video_reference.npz")


def test_timestamped_videos_and_images_share_the_reference_frame_order(video_source):
    """Five- and three-frame clips pad their final temporal patch; grids differ.

    Interleaving image and video occurrences matches HF: logits 5.98e-6,
    MTP logits 3.97e-6, greedy tokens exact. A frame-order reversal fails.
    """
    loaded, inputs, reference = video_source
    np.testing.assert_array_equal(inputs.tokens, reference["input_ids"])
    valid = np.asarray(inputs.token_fields["attention_mask"])
    logits = loaded.model.apply(loaded.variables, inputs.tokens, **inputs.kwargs())
    hidden = loaded.model.apply(loaded.variables, inputs.tokens,
                                 method=loaded.model.hidden_states, **inputs.kwargs())
    mtp = loaded.model.apply(loaded.variables, hidden, inputs.tokens,
                              method=loaded.model.mtp_logits, **inputs.kwargs())[0]
    np.testing.assert_allclose(np.asarray(logits)[valid], reference["logits"][valid], atol=1e-4, rtol=0)
    admitted = valid[:, :-1] & valid[:, 1:]
    np.testing.assert_allclose(np.asarray(mtp)[admitted], reference["mtp_logits"][admitted], atol=1e-4, rtol=0)
    wrong = dataclasses.replace(inputs, conditioning={
        **inputs.conditioning, "pixel_values": inputs.conditioning["pixel_values"][:, ::-1]})
    moved = loaded.model.apply(loaded.variables, wrong.tokens, **wrong.kwargs())
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.asarray(moved)[valid], reference["logits"][valid], atol=1e-4, rtol=0)
    result = generate(loaded.model, loaded.variables, inputs, 3, key=jax.random.key(1),
                       sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(result.tokens[:, -3:], reference["generated"][:, -3:])


def test_video_prediction_training_exports_the_reference_update(video_source, tmp_path):
    """Main+MTP video update error 8.80e-6 with unchanged 1e-4 tolerance.

    These 78 tokens cross a DeltaNet chunk. The former post-exp-only mask
    gave finite forward values but non-finite gradients and exported weights.
    """
    loaded, inputs, reference = video_source
    objective = LMObjective(loaded.model, inputs.tokens.shape[1] - 1,
                            pretrained=loaded.variables, ema_decay=None, mtp_weight=.2,
                            pad_id=int(loaded.generation_config["pad_token_id"]))
    step = Step(step=jnp.int32(0), key=jax.random.key(0), ema=None)

    def loss(params):
        statistics, _ = objective.loss({**loaded.variables, "params": params}, {"text": inputs}, step)
        return objective.reduce_loss(statistics)[0]

    value, gradient = jax.jit(jax.value_and_grad(loss))(loaded.variables["params"])
    np.testing.assert_allclose(value, reference["loss"], atol=1e-5, rtol=0)
    variables = {**loaded.variables, "params": jax.tree.map(
        lambda weight, grad: weight - 1e-4 * grad, loaded.variables["params"], gradient)}
    loaded.save(tmp_path, variables=variables)
    restored = load_pretrained(tmp_path, dtype="float32", attention_impl="reference")
    logits = restored.model.apply(restored.variables, inputs.tokens, **inputs.kwargs())
    valid = np.asarray(inputs.token_fields["attention_mask"])
    np.testing.assert_allclose(np.asarray(logits)[valid], reference["updated_logits"][valid], atol=1e-4, rtol=0)


def test_video_grid_and_modality_labels_are_checked_before_execution(video_source):
    loaded, _, reference = video_source
    values = {key: reference[key] for key in ("input_ids", "attention_mask", "mm_token_type_ids",
                                              "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")}
    with pytest.raises(ValueError, match="mm_token_type_ids"):
        loaded.processor.from_hf({**values, "mm_token_type_ids": np.zeros_like(values["input_ids"])})
    with pytest.raises(ValueError, match="video_grid_thw"):
        loaded.processor.from_hf({**values, "video_grid_thw": values["video_grid_thw"] * 0})


def test_source_templates_preserve_reasoning_controls_and_history(source):
    """Both source templates retain or remove historical reasoning as requested."""
    loaded, _, _ = source
    processor = loaded.processor
    messages = [{"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer", "reasoning_content": "historical analysis marker"},
                {"role": "user", "content": "next"}]
    retained = processor.chat(messages, reasoning_effort="medium", preserve_thinking=True)
    removed = processor.chat(messages, reasoning_effort="medium", preserve_thinking=False)
    assert "historical analysis marker" in processor.decode(retained.tokens)[0]
    assert "historical analysis marker" not in processor.decode(removed.tokens)[0]
    for effort in ("low", "medium", "xhigh"):
        expected = processor.reference.apply_chat_template(messages, tokenize=True, return_dict=True,
                                                            return_tensors="pt", add_generation_prompt=True,
                                                            reasoning_effort=effort, preserve_thinking=True)
        actual = processor.chat(messages, reasoning_effort=effort, preserve_thinking=True)
        np.testing.assert_array_equal(actual.tokens, expected["input_ids"].numpy())
    from jinja2 import TemplateError
    with pytest.raises(TemplateError, match="reasoning effort"):
        processor.chat(messages, reasoning_effort="invalid")

