"""SigLIP and Llama 4 vision towers with their projectors, against transformers.

The claim is parity: the same weights and the same pixels through the
reference implementation and through dew produce the same features.
tools/hf_reference.py writes the fixtures under torch and transformers (the
reference), so the comparison runs in CI without a download.

Tolerances and the differences actually observed, fp32 on CPU:

- siglip-tiny tower: max |difference| 1.1e-06, tolerance 1e-4. Two layers of
  width 32 over a 2x2 patch grid, sharing CLIPAttention; the trunk sequence
  alone (the attention pooling head some checkpoints carry maps to nothing).
- siglip projector : max |difference| 3.3e-07, tolerance 1e-4. Block average,
  (1 + w) RMSNorm and the plain matrix on the reference trunk output.
- llama4 tower     : max |difference| 5.5e-06, tolerance 1e-4. Unfold patches,
  trailing class token, grid rotary, pixel shuffle and the adapter MLP with
  its GELU after both maps.
- llama4 projector : max |difference| 9.6e-07, tolerance 1e-4. One bias-free
  map on the reference trunk output.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import load_file

from dew.nn import vision as V

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"


def load_fixture(name):
    directory = FIXTURES / name
    config = json.loads((directory / "config.json").read_text())
    tensors = load_file(str(directory / "model.safetensors"))
    projector_tensors = load_file(str(directory / "projector.safetensors"))
    projector = json.loads((directory / "projector.json").read_text())
    return {
        "config": config, "tensors": tensors, "projector_tensors": projector_tensors,
        "projector": projector, "pixels": np.load(directory / "pixels.npy"),
        "tower_ref": np.load(directory / "tower_ref.npy"),
        "projector_ref": np.load(directory / "projector_ref.npy"),
    }


def test_siglip_tower_matches_the_reference_implementation():
    """fp32 parity on the tiny SigLIP trunk, and the positions are live: with
    them zeroed the trunk leaves the reference by more than 0.01."""
    fixture = load_fixture("siglip-tiny")
    record = V.translate_siglip_vision_config(fixture["config"])
    tower = V.tower_from_record(record).build()
    variables = {"params": V.translate_siglip_vision_weights(fixture["tensors"])}
    assert np.max(np.abs(
        np.asarray(tower.apply(variables, fixture["pixels"])) - fixture["tower_ref"])) < 1e-4
    unpositioned = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf)
        if path[-1].key == "embedding" else leaf, variables)
    assert np.max(np.abs(
        np.asarray(tower.apply(unpositioned, fixture["pixels"]))
        - fixture["tower_ref"])) > 1e-2


def test_gemma_projector_matches_the_reference_implementation():
    """fp32 parity on the tiny Gemma projector over the reference trunk
    output, and the norm is live: with its scale zeroed the soft tokens leave
    the reference by more than 0.1."""
    fixture = load_fixture("siglip-tiny")
    trunk = V.translate_siglip_vision_config(fixture["config"])
    record = V.translate_gemma_projector_config(
        trunk, fixture["projector"]["text_width"],
        fixture["projector"]["mm_tokens_per_image"])
    projector = V.projector_from_record(record).build()
    variables = {"params": V.translate_gemma_projector_weights(
        fixture["projector_tensors"])}
    assert np.max(np.abs(np.asarray(
        projector.apply(variables, fixture["tower_ref"]))
        - fixture["projector_ref"])) < 1e-4
    unnormed = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf)
        if path[-1].key == "scale" else leaf, variables)
    assert np.max(np.abs(np.asarray(
        projector.apply(unnormed, fixture["tower_ref"]))
        - fixture["projector_ref"])) > 0.1


def test_llama4_tower_matches_the_reference_implementation():
    """fp32 parity on the tiny Llama 4 trunk, and the grid positions are live:
    with them zeroed the trunk leaves the reference by more than 0.01."""
    fixture = load_fixture("llama4-vision-tiny")
    record = V.translate_llama4_vision_config(fixture["config"])
    tower = V.tower_from_record(record).build()
    variables = {"params": V.translate_llama4_vision_weights(fixture["tensors"])}
    assert np.max(np.abs(
        np.asarray(tower.apply(variables, fixture["pixels"])) - fixture["tower_ref"])) < 1e-4
    unpositioned = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf)
        if path[-1].key == "positional_embedding" else leaf, variables)
    assert np.max(np.abs(
        np.asarray(tower.apply(unpositioned, fixture["pixels"]))
        - fixture["tower_ref"])) > 1e-2


def test_llama4_projector_matches_the_reference_implementation():
    """fp32 parity on the tiny outer projector over the reference trunk
    output."""
    fixture = load_fixture("llama4-vision-tiny")
    trunk = V.translate_llama4_vision_config(fixture["config"])
    record = V.translate_llama4_projector_config(
        trunk, fixture["projector"]["text_width"])
    projector = V.projector_from_record(record).build()
    variables = {"params": V.translate_llama4_projector_weights(
        fixture["projector_tensors"])}
    assert np.max(np.abs(np.asarray(
        projector.apply(variables, fixture["tower_ref"]))
        - fixture["projector_ref"])) < 1e-4


def test_merge_soft_tokens_places_one_image():
    """Two soft tokens land on the two marked positions in order, the rest of
    the row passes through, and a row marking any other count names it."""
    token_embeds = np.zeros((1, 5, 2), np.float32)
    soft_tokens = np.array([[[1.0, 2.0], [3.0, 4.0]]], np.float32)
    mask = np.array([[False, True, False, True, False]])
    assert np.asarray(V.merge_soft_tokens(token_embeds, soft_tokens, mask)).tolist() == [
        [[0.0, 0.0], [1.0, 2.0], [0.0, 0.0], [3.0, 4.0], [0.0, 0.0]]]
    with pytest.raises(ValueError, match="image positions"):
        V.merge_soft_tokens(token_embeds, soft_tokens, np.ones((1, 5), bool))


@pytest.mark.parametrize("record", [{"kind": "clip"}, {"kind": "mlp"}, {}])
def test_an_unknown_tower_or_projector_kind_is_refused(record):
    """A kind neither registry holds names itself with the known names."""
    with pytest.raises(ValueError, match="kind"):
        V.tower_from_record(record)
    with pytest.raises(ValueError, match="kind"):
        V.projector_from_record(record)


@pytest.mark.parametrize("field,value,message", [
    ("image_size", [28, 14], "square"),
    ("hidden_act", "swiglu", "hidden_act"),
    ("attention_dropout", 0.1, "training-time"),
])
def test_a_siglip_field_with_no_counterpart_is_refused(field, value, message):
    config = dict(json.loads(
        (FIXTURES / "siglip-tiny" / "config.json").read_text()))
    config[field] = value
    with pytest.raises(ValueError, match=message):
        V.translate_siglip_vision_config(config)


@pytest.mark.parametrize("field,value,message", [
    ("vision_feature_select_strategy", "full", "vision_feature_select_strategy"),
    ("multi_modal_projector_bias", True, "multi_modal_projector_bias"),
    ("attention_dropout", 0.1, "training-time"),
])
def test_a_llama4_field_with_no_counterpart_is_refused(field, value, message):
    config = dict(json.loads(
        (FIXTURES / "llama4-vision-tiny" / "config.json").read_text()))
    config[field] = value
    with pytest.raises(ValueError, match=message):
        V.translate_llama4_vision_config(config)
