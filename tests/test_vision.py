"""SigLIP, Llama 4, Gemma 4 and Qwen 3.5 vision towers against transformers.

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
- gemma4 tower     : max |difference| 1.96e-05, tolerance 1e-4. Scaled patch
  pixels, summed 2D tables, RMS blocks with the 2D rotary and gated
  feed-forwards, position pooling and standardization over a 4x4 grid.
- gemma4 projector : max |difference| 9.6e-07, tolerance 1e-4. Scale-free RMS
  norm and the map on the reference trunk output.
- qwen3_5 tower    : max |difference| 7.4e-06, tolerance 1e-4. Block-order
  patches with the frame repeated along time, resampled positions, the 2D
  rotary and full-attention blocks over a 4x4 grid on an 8x8 table.
- qwen3_5 projector: max |difference| 3.9e-06, tolerance 1e-4. The merger
  (norm, block grouping, exact GELU) on the reference trunk output.
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


def gemma4_image(pixels, patch):
    """Fixture patches folded back into the image the trunk reads: the
    processor unfolds row-major, so the fold inverts it."""
    batch, count, _ = pixels.shape
    grid = int(count ** 0.5)
    side = grid * patch
    return pixels.reshape(batch, grid, grid, 3, patch, patch).transpose(
        0, 3, 1, 4, 2, 5).reshape(batch, 3, side, side)


def test_gemma4_tower_matches_the_reference_implementation():
    """fp32 parity on the tiny Gemma 4 trunk, and the positions and the
    standardization are live: with the tables zeroed the trunk leaves the
    reference by more than 12.0, with the scale zeroed by more than 23.0."""
    fixture = load_fixture("gemma4-vision-tiny")
    record = V.translate_gemma4_vision_config(fixture["config"])
    tower = V.tower_from_record(record).build()
    variables = {"params": V.translate_gemma4_vision_weights(fixture["tensors"])}
    patch = record["patch_size"]
    assert isinstance(patch, int)
    image = gemma4_image(fixture["pixels"], patch)
    assert np.max(np.abs(
        np.asarray(tower.apply(variables, image)) - fixture["tower_ref"])) < 1e-4
    unpositioned = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf)
        if path[-1].key == "position_table" else leaf, variables)
    assert np.max(np.abs(
        np.asarray(tower.apply(unpositioned, image))
        - fixture["tower_ref"])) > 12.0
    unstandardized = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf)
        if path[-1].key == "std_scale" else leaf, variables)
    assert np.max(np.abs(
        np.asarray(tower.apply(unstandardized, image))
        - fixture["tower_ref"])) > 23.0


def test_gemma4_projector_matches_the_reference_implementation():
    """fp32 parity on the tiny Gemma 4 embedder over the reference trunk
    output, and the map is live: with it zeroed the soft tokens leave the
    reference by more than 3.0."""
    fixture = load_fixture("gemma4-vision-tiny")
    trunk = V.translate_gemma4_vision_config(fixture["config"])
    record = V.translate_gemma4_projector_config(
        trunk, fixture["projector"]["text_width"])
    projector = V.projector_from_record(record).build()
    variables = {"params": V.translate_gemma4_projector_weights(
        fixture["projector_tensors"])}
    assert np.max(np.abs(np.asarray(
        projector.apply(variables, fixture["tower_ref"]))
        - fixture["projector_ref"])) < 1e-4
    unmapped = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf)
        if path[-1].key == "kernel" else leaf, variables)
    assert np.max(np.abs(np.asarray(
        projector.apply(unmapped, fixture["tower_ref"]))
        - fixture["projector_ref"])) > 3.0


def test_qwen35_tower_matches_the_reference_implementation():
    """fp32 parity on the tiny Qwen 3.5 trunk, and the resampled positions
    are live: with the table zeroed the trunk leaves the reference by more
    than 0.7."""
    fixture = load_fixture("qwen35-vision-tiny")
    record = V.translate_qwen35_vision_config(fixture["config"])
    tower = V.tower_from_record(record).build()
    variables = {"params": V.translate_qwen35_vision_weights(fixture["tensors"])}
    assert np.max(np.abs(
        np.asarray(tower.apply(variables, fixture["pixels"])) - fixture["tower_ref"])) < 1e-4
    unpositioned = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf)
        if path[-1].key == "embedding" else leaf, variables)
    assert np.max(np.abs(
        np.asarray(tower.apply(unpositioned, fixture["pixels"]))
        - fixture["tower_ref"])) > 0.7


def test_qwen35_projector_matches_the_reference_implementation():
    """fp32 parity on the tiny Qwen 3.5 merger over the reference trunk
    output, and the merger is live: with its first map zeroed the soft
    tokens leave the reference by more than 10.0."""
    fixture = load_fixture("qwen35-vision-tiny")
    trunk = V.translate_qwen35_vision_config(fixture["config"])
    record = V.translate_qwen35_projector_config(
        trunk, fixture["projector"]["text_width"])
    projector = V.projector_from_record(record).build()
    variables = {"params": V.translate_qwen35_projector_weights(
        fixture["projector_tensors"])}
    assert np.max(np.abs(np.asarray(
        projector.apply(variables, fixture["tower_ref"]))
        - fixture["projector_ref"])) < 1e-4
    unmerged = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf)
        if path[-2].key == "fc1" else leaf, variables)
    assert np.max(np.abs(np.asarray(
        projector.apply(unmerged, fixture["tower_ref"]))
        - fixture["projector_ref"])) > 10.0


@pytest.mark.parametrize("field,value,message", [
    ("head_dim", 16, "head_dim"),
    ("hidden_activation", "swiglu", "hidden_activation"),
    ("attention_bias", True, "attention_bias"),
    ("attention_dropout", 0.1, "training-time"),
    ("use_clipped_linears", True, "use_clipped_linears"),
    ("standardize", False, "standardize"),
    ("output_proj_dims", 64, "output_proj_dims"),
])
def test_a_gemma4_field_with_no_counterpart_is_refused(field, value, message):
    config = dict(json.loads(
        (FIXTURES / "gemma4-vision-tiny" / "config.json").read_text()))
    config[field] = value
    with pytest.raises(ValueError, match=message):
        V.translate_gemma4_vision_config(config)


@pytest.mark.parametrize("field,value,message", [
    ("num_position_embeddings", 60, "square"),
    ("hidden_act", "swiglu", "hidden_act"),
    ("patch_size", [16, 8], "square patches"),
    ("model_type", "qwen3_vl", "vision model_type"),
])
def test_a_qwen35_field_with_no_counterpart_is_refused(field, value, message):
    config = dict(json.loads(
        (FIXTURES / "qwen35-vision-tiny" / "config.json").read_text()))
    config[field] = value
    with pytest.raises(ValueError, match=message):
        V.translate_qwen35_vision_config(config)


def test_a_qwen35_merger_beside_the_decoder_width_is_refused():
    """The merger output enters the text embeddings directly, so a width
    beside the decoder's refuses with both."""
    trunk = V.translate_qwen35_vision_config(json.loads(
        (FIXTURES / "qwen35-vision-tiny" / "config.json").read_text()))
    with pytest.raises(ValueError, match="out_hidden_size"):
        V.translate_qwen35_projector_config(trunk, 32)
