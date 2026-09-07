"""Multimodal wrappers: config translation and weight maps with parity.

translate_wrapper_config turns a gemma3, llama4, gemma4 or qwen3_5 wrapper
into its decoder, tower and projector records; translate_wrapper_weights
routes the released model.* nesting into the three trees, whose prefixes the
translation tests write down per family. Tower and projector parity rides the
tiny wrapper fixtures, whose vision halves come from the shared systems in
tools/hf_reference.py at a fresh seed with these observed differences, fp32
on CPU:

- gemma3-tiny-mm tower: max |difference| 9.6e-07, projector 4.8e-07,
  tolerance 1e-4. Prefixes model.vision_tower.* and
  model.multi_modal_projector.*.
- llama4-tiny-mm tower: max |difference| 7.0e-06, projector 1.5e-06,
  tolerance 1e-4. Bare vision_model.* and multi_modal_projector.* with no
  model. prefix.
- gemma4-tiny-mm tower: max |difference| 2.5e-05, projector 9.6e-07,
  tolerance 1e-4. Prefixes model.vision_tower.* and model.embed_vision.*.
- qwen35-tiny-mm tower: max |difference| 7.4e-06, projector 3.9e-06,
  tolerance 1e-4. Prefixes model.visual.* with the merger under
  model.visual.merger.*.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import load_file

from dew.interop.hf_decoders import (
    translate_config,
    translate_wrapper_config,
    translate_wrapper_weights,
)
from dew.nn import vision as V
from dew.registry import models, with_precision

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"


def load_wrapper(name):
    directory = FIXTURES / name
    record = translate_wrapper_config(
        json.loads((directory / "config.json").read_text()))
    variables = translate_wrapper_weights(
        load_file(str(directory / "model.safetensors")), record)
    return directory, record, variables


def test_gemma3_wrapper_translates_to_three_records():
    directory = FIXTURES / "gemma3-tiny-mm"
    record = translate_wrapper_config(
        json.loads((directory / "config.json").read_text()))
    assert record["text_model_type"] == "gemma3_text"
    assert record["text"]["emb_features"] == 64
    assert record["tower"]["kind"] == "siglip"
    assert record["tower"]["hidden_size"] == 32
    assert record["projector"]["kind"] == "gemma"
    assert record["projector"]["text_width"] == 64
    assert record["image_token_id"] == 202
    assert record["tokens_per_image"] == 1


def test_llama4_wrapper_translates_to_three_records():
    directory = FIXTURES / "llama4-tiny-mm"
    record = translate_wrapper_config(
        json.loads((directory / "config.json").read_text()))
    assert record["text_model_type"] == "llama4_text"
    assert record["tower"]["kind"] == "llama4"
    assert record["projector"]["kind"] == "llama4"
    assert record["projector"]["vision_width"] == 64
    assert record["image_token_id"] == 92
    assert record["tokens_per_image"] == 1


def test_gemma4_wrapper_translates_to_three_records():
    directory = FIXTURES / "gemma4-tiny-mm"
    record = translate_wrapper_config(
        json.loads((directory / "config.json").read_text()))
    assert record["text_model_type"] == "gemma4_text"
    assert record["text"]["emb_features"] == 32
    assert record["tower"]["kind"] == "gemma4"
    assert record["tower"]["hidden_size"] == 32
    assert record["tower"]["pooling_kernel_size"] == 2
    assert record["projector"]["kind"] == "gemma4"
    assert record["projector"] == {
        "kind": "gemma4", "vision_width": 32, "text_width": 32,
        "norm_eps": 1e-06}
    assert record["image_token_id"] == 60
    # The pooled count follows the image resolution, so the record leaves it
    # open; the fixture's 4x4 patch grid pools to four soft tokens.
    assert record["tokens_per_image"] is None


def test_qwen35_wrapper_translates_to_three_records():
    directory = FIXTURES / "qwen35-tiny-mm"
    record = translate_wrapper_config(
        json.loads((directory / "config.json").read_text()))
    assert record["text_model_type"] == "qwen3_5_text"
    assert record["tower"]["kind"] == "qwen3_5"
    assert record["tower"]["spatial_merge_size"] == 2
    assert record["tower"]["temporal_patch_size"] == 2
    assert record["projector"]["kind"] == "qwen3_5"
    assert record["projector"] == {
        "kind": "qwen3_5", "vision_width": 32, "merge_size": 2, "out_width": 64}
    assert record["image_token_id"] == 200
    assert record["tokens_per_image"] is None


def test_the_released_gemma4_wrapper_translates():
    """google/gemma-4-26B-A4B's wrapper, config only: the tower reads 1152
    wide with a pooling kernel of 3 and standardization on, the text half
    translates as gemma4_text, and the count stays open."""
    config = json.loads((FIXTURES / "gemma4-26b-a4b" / "config.json").read_text())
    record = translate_wrapper_config(config)
    assert translate_config(config["text_config"])["emb_features"] == 2816
    assert record["tower"]["hidden_size"] == 1152
    assert record["tower"]["pooling_kernel_size"] == 3
    assert record["tower"]["rope_theta"] == 100.0
    assert record["projector"]["text_width"] == 2816
    assert record["tokens_per_image"] is None


def test_the_released_qwen35_wrapper_translates():
    """Qwen/Qwen3.5-0.8B's wrapper, config only: the tower keeps the 0.8B
    checkpoint's qwen3_5 model_type spelling, and the merger width meets the
    decoder width."""
    config = json.loads((FIXTURES / "qwen35-0.8b" / "config.json").read_text())
    record = translate_wrapper_config(config)
    assert record["tower"]["hidden_size"] == 768
    assert record["tower"]["out_hidden_size"] == 1024
    assert record["projector"] == {
        "kind": "qwen3_5", "vision_width": 768, "merge_size": 2,
        "out_width": 1024}
    assert record["image_token_id"] == 248056


def test_the_released_scout_wrapper_translates():
    """meta-llama/Llama-4-Scout-17B-16E's wrapper, config only: the flat rope
    theta its vision config carries reads, and the soft-token count derives
    from the 24x24 patch grid shuffled by a half."""
    config = json.loads((FIXTURES / "llama-4-scout" / "config.json").read_text())
    record = translate_wrapper_config(config)
    assert record["tower"]["rope_theta"] == 10000.0
    assert record["tower"]["image_size"] == 336
    assert record["tokens_per_image"] == 144
    assert record["projector"] == {
        "kind": "llama4", "vision_width": 4096, "text_width": 5120}


def wrapper_pixels(directory, record):
    """Fixture pixels as the tower reads them: the Gemma 4 fixture stores
    processor patches, which fold back into the image row-major."""
    pixels = np.load(directory / "pixels.npy")
    if record["tower"]["kind"] != "gemma4":
        return pixels
    batch, count, _ = pixels.shape
    grid = int(count ** 0.5)
    patch = int(record["tower"]["patch_size"])
    side = grid * patch
    return pixels.reshape(batch, grid, grid, patch, patch, 3).transpose(
        0, 5, 1, 3, 2, 4).reshape(batch, 3, side, side)


def test_wrapper_tower_and_projector_match_the_reference():
    """The vision halves of all four wrapper fixtures against their committed
    reference outputs."""
    for name, tolerance in (("gemma3-tiny-mm", 1e-4), ("llama4-tiny-mm", 1e-4),
                            ("gemma4-tiny-mm", 1e-4), ("qwen35-tiny-mm", 1e-4)):
        directory, record, variables = load_wrapper(name)
        pixels = wrapper_pixels(directory, record)
        tower_ref = np.load(directory / "tower_ref.npy")
        projector_ref = np.load(directory / "projector_ref.npy")
        tower = V.tower_from_record(record["tower"]).build()
        assert np.max(np.abs(np.asarray(
            tower.apply(variables["tower"], pixels)) - tower_ref)) < tolerance
        projector = V.projector_from_record(record["projector"]).build()
        assert np.max(np.abs(np.asarray(
            projector.apply(variables["projector"], tower_ref))
            - projector_ref)) < tolerance


def test_wrapper_language_halves_build_and_score():
    """The language halves are complete decoder trees: they build and score a
    row of ids. The tied Gemma halves keep one leaf for head and embedding."""
    for name, tied in (("gemma3-tiny-mm", True), ("llama4-tiny-mm", False),
                        ("gemma4-tiny-mm", True), ("qwen35-tiny-mm", False)):
        _, record, variables = load_wrapper(name)
        model = models.build("causal_transformer", **with_precision(
            "causal_transformer", record["text"], dtype="float32",
            attention_impl="reference"))
        leaves, _ = jax.tree_util.tree_flatten_with_path(variables["language_model"])
        names = {".".join(str(entry.key) for entry in path) for path, _ in leaves}
        assert ("params.lm_head.kernel" in names) is not tied
        ids = np.zeros((1, 4), np.int32)
        assert np.asarray(model.apply(
            variables["language_model"], ids)).shape == (1, 4, model.vocab_size)


def test_a_wrapper_tensor_outside_the_three_prefixes_is_refused():
    """Audio and tile metadata ride no prefix this map reads, so they name
    themselves."""
    directory, record, _ = load_wrapper("gemma3-tiny-mm")
    tensors = dict(load_file(str(directory / "model.safetensors")))
    tensors["model.audio_tower.layers.0.weight"] = np.zeros((2, 2), np.float32)
    with pytest.raises(ValueError, match="unknown tensor name"):
        translate_wrapper_weights(tensors, record)


def test_a_wrapper_without_an_image_token_is_refused():
    config = json.loads((FIXTURES / "gemma3-tiny-mm" / "config.json").read_text())
    del config["image_token_index"]
    with pytest.raises(ValueError, match="image_token_id"):
        translate_wrapper_config(config)


def _multimodal_logits(name, image_id, shift=0):
    """Pixels through the translated tower and projector, merged at the image
    marks, into the translated decoder through the input-embeddings hook."""
    directory = FIXTURES / name
    record = translate_wrapper_config(
        json.loads((directory / "config.json").read_text()))
    variables = translate_wrapper_weights(
        load_file(str(directory / "model.safetensors")), record)
    pixels = wrapper_pixels(directory, record)
    ids = np.load(directory / "input_ids.npy")
    tower = V.tower_from_record(record["tower"]).build()
    features = tower.apply(variables["tower"], pixels)
    projector = V.projector_from_record(record["projector"]).build()
    soft = projector.apply(variables["projector"], features)
    length = ids.shape[1]
    positions = (np.stack([np.where(row == image_id)[0] for row in ids]) + shift) % length
    positions = positions.astype(np.int32)
    fields = dict(record["text"])
    if name.startswith("gemma3"):
        # The reference conditional applies no logit cap (only its causal-LM
        # head path does), so the decoder builds without the text record's.
        fields["final_logit_softcap"] = None
    model = models.build("causal_transformer", **with_precision(
        "causal_transformer", fields, dtype="float32", attention_impl="reference"))
    return (np.asarray(model.apply(variables["language_model"], ids,
                                   input_embeddings=np.asarray(soft),
                                   embedding_positions=positions)),
            np.load(directory / "wrapper_ref.npy"), positions)


def test_gemma3_multimodal_forward_matches_the_wrapper():
    """fp32 end-to-end parity on the tiny Gemma 3 wrapper: tolerance 1e-4,
    observed max |logit difference| 2.4e-06 with identical argmax. Soft tokens
    at shifted positions miss by more than 1.0, so the placement is live."""
    logits, reference, _ = _multimodal_logits("gemma3-tiny-mm", 202)
    assert np.max(np.abs(logits - reference)) < 1e-4
    assert (logits.argmax(-1) == reference.argmax(-1)).all()
    misplaced, _, _ = _multimodal_logits("gemma3-tiny-mm", 202, shift=1)
    assert np.max(np.abs(misplaced - reference)) > 1.0


def test_llama4_multimodal_forward_matches_the_wrapper():
    """fp32 end-to-end parity on the tiny Llama 4 wrapper: tolerance 1e-4,
    observed max |logit difference| 4.1e-06 with identical argmax."""
    logits, reference, _ = _multimodal_logits("llama4-tiny-mm", 92)
    assert np.max(np.abs(logits - reference)) < 1e-4
    assert (logits.argmax(-1) == reference.argmax(-1)).all()


def test_gemma4_multimodal_forward_matches_the_wrapper():
    """fp32 end-to-end parity on the tiny Gemma 4 wrapper: tolerance 1e-4,
    observed max |logit difference| 1.4e-05 with identical argmax. Soft tokens
    at shifted positions miss by more than 1.0, so the placement is live."""
    logits, reference, _ = _multimodal_logits("gemma4-tiny-mm", 60)
    assert np.max(np.abs(logits - reference)) < 1e-4
    assert (logits.argmax(-1) == reference.argmax(-1)).all()
    misplaced, _, _ = _multimodal_logits("gemma4-tiny-mm", 60, shift=1)
    assert np.max(np.abs(misplaced - reference)) > 1.0


def test_qwen35_multimodal_forward_matches_the_wrapper():
    """fp32 end-to-end parity on the tiny Qwen 3.5 wrapper: tolerance 1e-4,
    observed max |logit difference| 2.4e-05 with identical argmax. The wrapper
    reference runs on pre-merged embeddings, since the image-grid positions it
    would otherwise use are outside what the decoder models."""
    logits, reference, _ = _multimodal_logits("qwen35-tiny-mm", 200)
    assert np.max(np.abs(logits - reference)) < 1e-4
    assert (logits.argmax(-1) == reference.argmax(-1)).all()
    misplaced, _, _ = _multimodal_logits("qwen35-tiny-mm", 200, shift=1)
    assert np.max(np.abs(misplaced - reference)) > 1.0
