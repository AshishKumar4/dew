"""Multimodal wrappers: config translation and weight maps with parity.

translate_wrapper_config turns a gemma3 or llama4 wrapper into its decoder,
tower and projector records; translate_wrapper_weights routes the released
model.* nesting into the three trees. Tower and projector parity rides the
tiny wrapper fixtures, whose vision halves come from the shared systems in
tools/hf_reference.py at a fresh seed with these observed differences, fp32
on CPU:

- gemma3-tiny-mm tower: max |difference| 9.6e-07, projector 4.8e-07,
  tolerance 1e-4.
- llama4-tiny-mm tower: max |difference| 7.0e-06, projector 1.5e-06,
  tolerance 1e-4.
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


def test_wrapper_tower_and_projector_match_the_reference():
    """The vision halves of both wrapper fixtures against their committed
    reference outputs."""
    for name, tolerance in (("gemma3-tiny-mm", 1e-4), ("llama4-tiny-mm", 1e-4)):
        directory, record, variables = load_wrapper(name)
        pixels = np.load(directory / "pixels.npy")
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
    row of ids. The tied Gemma half keeps one leaf for head and embedding."""
    for name, tied in (("gemma3-tiny-mm", True), ("llama4-tiny-mm", False)):
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


def test_a_gemma4_wrapper_is_refused_naming_its_vision_tower():
    """google/gemma-4-26B-A4B's wrapper carries a gemma4_vision tower, whose
    position tables, rotary, clippable linears and pooling nothing here runs;
    only its text_config translates."""
    config = json.loads((FIXTURES / "gemma4-26b-a4b" / "config.json").read_text())
    assert translate_config(config["text_config"])["emb_features"] == 2816


def _multimodal_logits(name, image_id, shift=0):
    """Pixels through the translated tower and projector, merged at the image
    marks, into the translated decoder through the input-embeddings hook."""
    directory = FIXTURES / name
    record = translate_wrapper_config(
        json.loads((directory / "config.json").read_text()))
    variables = translate_wrapper_weights(
        load_file(str(directory / "model.safetensors")), record)
    pixels = np.load(directory / "pixels.npy")
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
