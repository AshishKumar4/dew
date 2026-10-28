"""SigLIP, Llama 4, Gemma 4, Qwen 3.5, MiMo-V2.6 and Muse Glimmer vision towers against their references.

The claim is parity: the same weights and the same pixels through the
reference implementation and through dew produce the same features.
tools/hf_reference.py writes the fixtures under torch and transformers (the
reference), so the comparison runs in CI without a download.

Tolerances and the differences actually observed, fp32 on CPU:

- siglip-tiny tower: max |difference| 1.1e-06, tolerance 1e-4. Two layers of
  width 32 over a 2x2 patch grid, CLIP's layer (`clip_layer`); the trunk sequence
  alone (the attention pooling head some checkpoints carry maps to nothing).
- siglip projector : max |difference| 3.3e-07, tolerance 1e-4. Block average,
  (1 + w) RMSNorm and the plain matrix on the reference trunk output.
- llama4 tower     : max |difference| 5.5e-06, tolerance 1e-4. Unfold patches,
  trailing class token, grid rotary, pixel shuffle and the adapter MLP with
  its GELU after both maps.
- llama4 projector : max |difference| 9.6e-07, tolerance 1e-4. One bias-free
  map on the reference trunk output.
- gemma4 tower     : max |difference| 2.01e-05, tolerance 1e-4. Scaled patch
  pixels, summed 2D tables, RMS blocks with the 2D rotary and gated
  feed-forwards, position pooling and standardization over a 4x4 grid.
- gemma4 projector : max |difference| 9.6e-07, tolerance 1e-4. Scale-free RMS
  norm and the map on the reference trunk output.
- qwen3_5 tower    : max |difference| 6.44e-06, tolerance 1e-4. Block-order
  patches with the frame repeated along time, resampled positions, the 2D
  rotary and full-attention blocks over a 4x4 grid on an 8x8 table.
- qwen3_5 projector: max |difference| 3.9e-06, tolerance 1e-4. The merger
  (norm, block grouping, exact GELU) on the reference trunk output.
"""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import (
    as_given,
    assert_as_exact,
    assert_as_exact_as_the_reference,
    float64_forward,
    ieee_fixture,
)
from safetensors.numpy import load_file

from dew.nn.vision import (
    gemma4,
    llama4,
    mimo,
    muse_glimmer,
    projector_layout,
    projector_variables,
    qwen35,
    siglip,
    tower_variables,
)
from dew.registry import projectors, towers

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


def assert_tower_with_live_positions(tower, variables, fixture, positions: str) -> None:
    """`tower`'s features of the fixture's pixels within 1e-4 of its reference's everywhere, and,
    with the parameters named `positions` zeroed, more than 0.01 away somewhere: the positions are
    live."""
    assert np.max(np.abs(
        np.asarray(tower.apply(variables, fixture["pixels"])) - fixture["tower_ref"])) < 1e-4
    unpositioned = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf) if path[-1].key == positions else leaf, variables)
    assert np.max(np.abs(
        np.asarray(tower.apply(unpositioned, fixture["pixels"])) - fixture["tower_ref"])) > 1e-2


@pytest.mark.parametrize("name, family, translate, positions", [
    ("siglip-tiny", "siglip", siglip.translate_siglip_vision_config, "embedding"),
    ("llama4-vision-tiny", "llama4", llama4.translate_llama4_vision_config, "embedding"),
], ids=["siglip", "llama4"])
def test_a_tower_matches_the_reference_implementation(name, family, translate, positions):
    """fp32 parity on the tiny SigLIP and Llama 4 trunks, and their positions (Llama 4's grid
    positions) are live: with them zeroed the trunk leaves the reference by more than 0.01."""
    fixture = load_fixture(name)
    tower = towers.from_record(translate(fixture["config"])).build()
    variables, _ = tower_variables(family, fixture["tensors"], "float32")
    assert_tower_with_live_positions(tower, variables, fixture, positions)


SIGLIP_SEEDS = 8
"""The quick_gelu SigLIP tower's declared draws against transformers' (check 1)."""


def test_a_siglip_tower_runs_any_layer_activation_and_an_inactive_dropout_as_the_reference_does():
    """A SigLIP config may name any activation the shared tower layer runs,
    quick_gelu here, and an attention dropout, which only a training forward
    draws: the tower loads and its features are on par with the reference's
    from the reference in float64 over drawn pixels (check 1)."""
    import torch
    from transformers import SiglipVisionConfig, SiglipVisionModel

    from tools.torch_precision import float64, float64_twin

    config = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                num_attention_heads=4, image_size=28, patch_size=14, num_channels=3,
                                hidden_act="quick_gelu", attention_dropout=0.1, layer_norm_eps=1e-6)
    torch.manual_seed(5)
    reference = SiglipVisionModel(config).eval()
    tensors = {name: tensor.numpy().copy() for name, tensor in reference.state_dict().items()}
    wide = float64_twin(reference)

    def draw(seed):
        return (np.random.default_rng(seed).random((2, 3, 28, 28)).astype(np.float32),)

    @as_given
    def theirs(pixels):
        with torch.no_grad():
            given = torch.from_numpy(np.asarray(pixels, np.float32))
            return reference(pixel_values=given).last_hidden_state.numpy()

    @as_given
    def truth(pixels):
        with torch.no_grad(), float64():
            given = torch.from_numpy(np.asarray(pixels, np.float64))
            return wide(pixel_values=given).last_hidden_state.numpy()

    tower = towers.from_record(siglip.translate_siglip_vision_config(config.to_dict())).build()
    variables = tower_variables("siglip", tensors, "float32")[0]
    assert_as_exact(lambda pixels: tower.apply(variables, pixels), theirs, draw,
                    "siglip quick_gelu features", seeds=SIGLIP_SEEDS, truth_fn=truth)


def test_gemma_projector_matches_the_reference_implementation():
    """fp32 parity on the tiny Gemma projector over the reference trunk
    output, and the norm is live: with its scale zeroed the soft tokens leave
    the reference by more than 0.1."""
    fixture = load_fixture("siglip-tiny")
    trunk = siglip.translate_siglip_vision_config(fixture["config"])
    record = siglip.translate_gemma_projector_config(
        trunk["fields"], fixture["projector"]["text_width"],
        fixture["projector"]["mm_tokens_per_image"])
    projector = projectors.from_record(record)
    variables = {"params": siglip.translate_gemma_projector_weights(
        fixture["projector_tensors"])[0]}
    assert np.max(np.abs(np.asarray(
        projector.apply(variables, fixture["tower_ref"]))
        - fixture["projector_ref"])) < 1e-4
    unnormed = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf)
        if path[-1].key == "scale" else leaf, variables)
    assert np.max(np.abs(np.asarray(
        projector.apply(unnormed, fixture["tower_ref"]))
        - fixture["projector_ref"])) > 0.1


def test_llama4_projector_matches_the_reference_implementation():
    """fp32 parity on the tiny outer projector over the reference trunk
    output."""
    fixture = load_fixture("llama4-vision-tiny")
    record = llama4.translate_llama4_projector_config(fixture["projector"]["text_width"])
    projector = projectors.from_record(record)
    variables = {"params": projector_variables("llama4", fixture["projector_tensors"], "float32")[0]}
    assert np.max(np.abs(np.asarray(
        projector.apply(variables, fixture["tower_ref"]))
        - fixture["projector_ref"])) < 1e-4


def gemma4_image(pixels, patch):
    """Fixture patches folded back into the image the trunk reads: the
    processor unfolds row-major, so the fold inverts it."""
    batch, count, _ = pixels.shape
    grid = int(count ** 0.5)
    side = grid * patch
    return pixels.reshape(batch, grid, grid, patch, patch, 3).transpose(
        0, 5, 1, 3, 2, 4).reshape(batch, 3, side, side)


def test_gemma4_tower_matches_the_reference_implementation():
    """fp32 parity on the tiny Gemma 4 trunk, and the positions and the
    standardization are live: with the tables zeroed the trunk leaves the
    reference by more than 12.0, with the scale zeroed by more than 23.0."""
    fixture = load_fixture("gemma4-vision-tiny")
    record = gemma4.translate_gemma4_vision_config(fixture["config"])
    tower = towers.from_record(record).build()
    variables, _ = tower_variables("gemma4", fixture["tensors"], "float32")
    patch = record["fields"]["patch_size"]
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
    trunk = gemma4.translate_gemma4_vision_config(fixture["config"])
    record = gemma4.translate_gemma4_projector_config(
        trunk["fields"], fixture["projector"]["text_width"])
    projector = projectors.from_record(record)
    variables = {"params": gemma4.translate_gemma4_projector_weights(
        fixture["projector_tensors"])[0]}
    assert np.max(np.abs(np.asarray(
        projector.apply(variables, fixture["tower_ref"]))
        - fixture["projector_ref"])) < 1e-4
    unmapped = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf)
        if path[-1].key == "kernel" else leaf, variables)
    assert np.max(np.abs(np.asarray(
        projector.apply(unmapped, fixture["tower_ref"]))
        - fixture["projector_ref"])) > 3.0


def test_gemma4_widened_head_computes_the_reference_features():
    from flax.errors import ScopeParamShapeError

    fixture = load_fixture('gemma4-vision-wide-tiny')
    record = gemma4.translate_gemma4_vision_config(fixture['config'])
    vision = towers.from_record(record)
    tower = vision.build()
    variables, _ = tower_variables("gemma4", fixture["tensors"], "float32")
    positions = np.load(FIXTURES / 'gemma4-vision-wide-tiny' / 'positions.npy')
    actual = np.asarray(tower.apply(variables, fixture['pixels'], positions))
    np.testing.assert_allclose(actual, fixture['tower_ref'], atol=1e-4, rtol=0)
    wrong = dataclasses.replace(vision, head_dim=None).build()
    with pytest.raises(ScopeParamShapeError):
        wrong.apply(variables, fixture['pixels'], positions)


def test_qwen35_tower_matches_the_reference_implementation():
    """fp32 parity on the tiny Qwen 3.5 trunk, and the resampled positions
    are live: with the table zeroed the trunk leaves the reference by more
    than 0.69."""
    fixture = load_fixture("qwen35-vision-tiny")
    record = qwen35.translate_qwen35_vision_config(fixture["config"])
    tower = towers.from_record(record).build()
    variables, _ = tower_variables("qwen3_5", fixture["tensors"], "float32")
    assert np.max(np.abs(
        np.asarray(tower.apply(variables, fixture["pixels"])) - fixture["tower_ref"])) < 1e-4
    unpositioned = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf)
        if path[-1].key == "embedding" else leaf, variables)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            np.asarray(tower.apply(unpositioned, fixture["pixels"])),
            fixture["tower_ref"], atol=1e-4, rtol=0)


def test_qwen35_projector_matches_the_reference_implementation():
    """fp32 parity on the tiny Qwen 3.5 merger over the reference trunk
    output, and the merger is live: with its first map zeroed the soft
    tokens leave the reference by more than 10.0."""
    fixture = load_fixture("qwen35-vision-tiny")
    record = qwen35.translate_qwen35_projector_config(
        fixture["config"], fixture["projector"]["text_width"])
    projector = projectors.from_record(record)
    variables = {"params": projector_variables("qwen3_5", fixture["projector_tensors"], "float32")[0]}
    assert np.max(np.abs(np.asarray(
        projector.apply(variables, fixture["tower_ref"]))
        - fixture["projector_ref"])) < 1e-4
    unmerged = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf)
        if path[-2].key == "fc1" else leaf, variables)
    assert np.max(np.abs(np.asarray(
        projector.apply(unmerged, fixture["tower_ref"]))
        - fixture["projector_ref"])) > 10.0


@pytest.mark.parametrize("name", ["gemma4-vision-tiny", "gemma4-vision-wide-tiny",
                                  "diffusion-gemma-workflow"])
def test_a_gemma4_tower_writes_back_the_config_it_was_read_from(name):
    """Every field the writer emits is the published value, and reading what
    it writes gives the same tower; an unset head_dim is written as the width
    it stands for, so it reads back as that width."""
    published = json.loads((FIXTURES / name / "config.json").read_text())
    vision = published.get("vision_config", published)
    tower = towers.from_record(gemma4.translate_gemma4_vision_config(published))
    written = gemma4.export_gemma4_vision_config(tower)
    assert {key: (value, vision[key]) for key, value in written.items() if vision[key] != value} == {}
    assert towers.from_record(gemma4.translate_gemma4_vision_config(written)) == tower
    unset = dataclasses.replace(tower, head_dim=None, rope_theta=50.0, standardize=False,
                                use_clipped_linears=True, hidden_act="gelu")
    assert towers.from_record(gemma4.translate_gemma4_vision_config(
        gemma4.export_gemma4_vision_config(unset))) == (
        dataclasses.replace(unset, head_dim=unset.hidden_size // unset.num_heads))
    # The base reads alike from rope_parameters under either spelling of its type and from the flat
    # field; the decoders' 'none' alias is no type of the towers' reference.
    theta = written["rope_parameters"]["rope_theta"]
    for spelled in ({"rope_parameters": {"type": "default", "rope_theta": theta}},
                    {"rope_parameters": None, "rope_theta": theta}):
        assert towers.from_record(gemma4.translate_gemma4_vision_config({**written, **spelled})) == tower
    with pytest.raises(ValueError):
        gemma4.translate_gemma4_vision_config(
            {**written, "rope_parameters": {"rope_type": "none", "rope_theta": theta}})


def shared_fixture(name):
    """A tools/mimo_reference.py or tools/muse_glimmer_reference.py fixture: the reference config, the
    tower's and projector's tensors,
    and the IEEE references (`refs.npz`)."""
    directory = FIXTURES / name
    return (json.loads((directory / "config.json").read_text()),
            load_file(str(directory / "model.safetensors")),
            load_file(str(directory / "projector.safetensors")), ieee_fixture(directory / "refs.npz"))


def real(values, grid, merge: int = 1):
    """Each row's own entries of padded `[B, L, ...]` values, a row's patches `prod(grid) / merge**2`."""
    counts = np.prod(grid, axis=1) // merge ** 2
    return np.concatenate([np.asarray(values)[row, :count] for row, count in enumerate(counts)])


def test_mimo_tower_is_as_exact_as_the_reference():
    """MiMo ViT on two rows, a 4x6 image and a two-frame 4x4 clip: full layers, a band of three keys
    either side in the processor's order and in merge-block column order, and the sinks, within
    the float64 gate; with the sinks zeroed the trunk leaves the reference by far more."""
    config, tensors, _, refs = shared_fixture("mimo-vision-tiny")
    tower = towers.from_record(mimo.translate_mimo_vision_config(config)).build()
    variables, _ = tower_variables("mimo", tensors, "float32")
    out = tower.apply(variables, refs["pixels"], refs["grid"])
    grid = refs["grid"]
    _, terms = float64_forward(tower, variables, refs["pixels"], grid)
    assert_as_exact_as_the_reference(real(out, grid), real(refs["tower"], grid),
                                     real(refs["tower_f64"], grid), "mimo tower",
                                     terms=real(np.asarray(terms), grid))
    sinkless = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf) if path[-1].key.startswith("sinks_") else leaf, variables)
    with pytest.raises(AssertionError):
        assert_as_exact_as_the_reference(real(tower.apply(sinkless, refs["pixels"], refs["grid"]), grid),
                                         real(refs["tower"], grid), real(refs["tower_f64"], grid), "sinkless")


@pytest.mark.parametrize("native", [False, True])
def test_mimo_merger_is_as_exact_as_the_reference(native):
    config, _, projector_tensors, refs = shared_fixture("mimo-vision-tiny")
    record = mimo.translate_mimo_projector_config(config, config["text_config"]["hidden_size"])
    projector = projectors.from_record(record)
    if native:
        vision = config["vision_config"]
        projector = mimo.MiMoProjector(
            hidden_size=vision["hidden_size"], spatial_merge_size=vision["spatial_merge_size"],
            width=vision["hidden_size"] * vision["spatial_merge_size"] ** 2,
            out_width=config["text_config"]["hidden_size"])
    variables = {"params": projector_variables("mimo", projector_tensors, "float32")[0]}
    out = projector.apply(variables, refs["tower"])
    grid = refs["grid"]
    assert_as_exact_as_the_reference(real(out, grid, 2), real(refs["projector"], grid, 2),
                                     real(refs["projector_f64"], grid, 2), "mimo merger")


def test_muse_glimmer_tower_is_as_exact_as_the_reference():
    """Muse Glimmer's trunk on a 6x10 grid of 4-patch windows, some partial, and an 8x4 one: the
    resampled table, the windows and full layers, the column-first rotary, within the float64 gate,
    and the first row again from its NCHW image packed in the processor's time-then-channel order."""
    config, tensors, _, refs = shared_fixture("muse-glimmer-vision-tiny")
    tower = towers.from_record(muse_glimmer.translate_muse_glimmer_vision_config(config)).build()
    variables, _ = tower_variables("muse_glimmer", tensors, "float32")
    grid = refs["grid"]
    out = tower.apply(variables, refs["pixels"], grid)
    assert_as_exact_as_the_reference(real(out, grid), real(refs["tower"], grid),
                                     real(refs["tower_f64"], grid), "muse glimmer tower")
    image = tower.apply(variables, refs["image"])
    count = int(np.prod(grid[0]))
    assert_as_exact_as_the_reference(np.asarray(image)[0], refs["tower"][0, :count],
                                     refs["tower_f64"][0, :count], "muse glimmer image")


def test_muse_glimmer_projector_is_as_exact_as_the_reference():
    """The pixel shuffle channel by channel, the adapter's two activated maps, the projection and the
    scale-free norm, each tensor named as the checkpoint router leaves it (`projector_layout`)."""
    config, _, projector_tensors, refs = shared_fixture("muse-glimmer-vision-tiny")
    projector = projectors.from_record(muse_glimmer.translate_muse_glimmer_projector_config(config))
    routed = {name[len(prefix):]: tensor for name, tensor in projector_tensors.items()
              for prefix in projector_layout("muse_glimmer").prefix if name.startswith(prefix)}
    variables = {"params": projector_variables("muse_glimmer", routed, "float32")[0]}
    out = projector.apply(variables, refs["tower"])
    grid = refs["grid"]
    assert_as_exact_as_the_reference(real(out, grid, 2), real(refs["projector"], grid, 2),
                                     real(refs["projector_f64"], grid, 2), "muse glimmer projector")


def test_gemma4_clipping_matches_reference_forward_and_backward():
    """Transformers 5.16.1 Gemma4ClippableLinear on a 3-to-2 map.

    Reference output [0.2, 0.06], input gradient [0, -0.1, 0], and the second
    weight row gradient [-1, 0.4, 1] distinguish both clipping locations.
    """
    linear = gemma4.Gemma4ClippableLinear(features=2, use_bias=False, use_clipped_linears=True)
    kernel = jnp.array([[0.1, 0.1], [0.2, -0.1], [0.3, 0.2]], jnp.float32)
    constants = {"input_min": jnp.float32(-1), "input_max": jnp.float32(1),
                 "output_min": jnp.float32(-0.2), "output_max": jnp.float32(0.2)}
    inputs = jnp.array([[-2, 0.4, 2]], jnp.float32)

    def value(weight, pixels):
        return jnp.asarray(linear.apply({"params": {"kernel": weight}, "constants": constants}, pixels))

    np.testing.assert_allclose(value(kernel, inputs), [[0.2, 0.06]], atol=1e-7, rtol=0)
    weights, pixels = jax.grad(lambda weight, x: value(weight, x).sum(), argnums=(0, 1))(kernel, inputs)
    np.testing.assert_allclose(pixels, [[0, -0.1, 0]], atol=1e-7, rtol=0)
    np.testing.assert_allclose(weights, [[0, -1], [0, 0.4], [0, 1]], atol=1e-7, rtol=0)
    unclipped = jnp.asarray(
        linear.clone(use_clipped_linears=False).apply({"params": {"kernel": kernel}}, inputs)
    )
    assert np.max(np.abs(unclipped - value(kernel, inputs))) > 0.25
