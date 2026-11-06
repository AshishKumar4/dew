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
import fractions
import functools
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from float64_terms import float64_forward
from one_draw import assert_as_exact_as_the_reference
from reference_error import as_given, assert_as_exact, ieee_fixture
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
"""The quick_gelu SigLIP tower's declared draws against transformers' (check 1), sized by `seeds_for`
paired with Dew's tower on the CPU."""


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


def _gemma_projector(fixture):
    trunk = siglip.translate_siglip_vision_config(fixture["config"])
    record = siglip.translate_gemma_projector_config(
        trunk["fields"], fixture["projector"]["text_width"], fixture["projector"]["mm_tokens_per_image"])
    return projectors.from_record(record), {"params": siglip.translate_gemma_projector_weights(
        fixture["projector_tensors"])[0]}


def _gemma4_projector(fixture):
    trunk = gemma4.translate_gemma4_vision_config(fixture["config"])
    record = gemma4.translate_gemma4_projector_config(trunk["fields"], fixture["projector"]["text_width"])
    return projectors.from_record(record), {"params": gemma4.translate_gemma4_projector_weights(
        fixture["projector_tensors"])[0]}


def _qwen35_projector(fixture):
    record = qwen35.translate_qwen35_projector_config(fixture["config"], fixture["projector"]["text_width"])
    return projectors.from_record(record), {
        "params": projector_variables("qwen3_5", fixture["projector_tensors"], "float32")[0]}


PROJECTORS = {
    "gemma": ("siglip-tiny", _gemma_projector, lambda path: path[-1].key == "scale", 0.1),
    "gemma4": ("gemma4-vision-tiny", _gemma4_projector, lambda path: path[-1].key == "kernel", 3.0),
    "qwen35": ("qwen35-vision-tiny", _qwen35_projector, lambda path: path[-2].key == "fc1", 10.0),
}
"""Each family's tiny projector: its fixture, its build from the fixture's tensors, the leaf whose zeroing
shows the projector live (Gemma's norm scale, Gemma 4's map, Qwen 3.5's merger's first map) and how far
the zeroed projector's soft tokens leave the reference."""


@pytest.mark.parametrize("name", sorted(PROJECTORS))
def test_a_projector_matches_the_reference_implementation(name):
    """fp32 parity on each family's tiny projector over the reference trunk output, and the projector is
    live: with its leaf zeroed the soft tokens leave the reference by the family's margin (`PROJECTORS`)."""
    fixture_name, built, zeroed, margin = PROJECTORS[name]
    fixture = load_fixture(fixture_name)
    projector, variables = built(fixture)

    def gap(variables):
        output = np.asarray(projector.apply(variables, fixture["tower_ref"]))
        return np.max(np.abs(output - fixture["projector_ref"]))

    assert gap(variables) < 1e-4
    assert gap(jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf) if zeroed(path) else leaf, variables)) > margin


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


def test_a_grid_convolution_convolves_each_frames_patch_grid_whatever_the_token_order():
    """LFM3-VL's gated grid convolution (`GatedGridConv`) over a row of two 3 x 4 frames and padding after
    them: each token's output is its frame's depthwise 3 x 3 convolution of B * x over the frame's grid,
    zero past its edges (`jax.lax.conv_general_dilated` over the frame's raster grid), gated by C and
    mapped back; the tokens shuffled give the same outputs shuffled. Every value is a small integer, so
    each sum is exact in float32 whatever its order."""
    from dew.batch import coordinates
    from dew.nn.inputs import AttentionMetadata, MixerCall
    from dew.nn.mixers.short_conv import GatedGridConv

    height, width, frames, padding, features = 3, 4, 2, 3, 4
    rng = np.random.default_rng(0x2067)
    length = frames * height * width + padding
    x = rng.integers(-2, 3, (1, length, features)).astype(np.float32)
    rows = np.concatenate([np.repeat(np.arange(height), width)] * frames + [np.zeros(padding, int)])
    columns = np.concatenate([np.tile(np.arange(width), height)] * frames + [np.zeros(padding, int)])
    segments = np.concatenate([np.full(height * width, frame + 1) for frame in range(frames)]
                              + [np.zeros(padding, int)])
    mixer = GatedGridConv(features)
    params = jax.tree.map(lambda leaf: jnp.asarray(rng.integers(-2, 3, leaf.shape), jnp.float32),
                          mixer.init(jax.random.key(0), jnp.asarray(x), MixerCall(AttentionMetadata(
                              rotary_positions=jnp.zeros((1, length, 2), jnp.int32)))))

    def mixed(order):
        metadata = AttentionMetadata(
            coordinates=coordinates(segments=jnp.asarray(segments[order][None], jnp.int32),
                                    positions=jnp.arange(length)[None]),
            rotary_positions=jnp.asarray(np.stack([rows[order], columns[order]], -1)[None], jnp.int32))
        return np.asarray(mixer.apply(params, jnp.asarray(x[:, order]), MixerCall(metadata)))

    weights = params["params"]
    gate_in, gate_out, value = np.split(x @ np.asarray(weights["in_proj"]["kernel"]), 3, axis=-1)
    grids = (gate_in * value)[0, :frames * height * width].reshape(frames, height, width, features)
    convolved = np.asarray(jax.lax.conv_general_dilated(
        jnp.asarray(grids), weights["spatial_conv"]["kernel"], (1, 1), ((1, 1), (1, 1)),
        dimension_numbers=("NHWC", "HWIO", "NHWC"), feature_group_count=features))
    expected = (gate_out[0, :frames * height * width] * convolved.reshape(-1, features)) @ np.asarray(
        weights["out_proj"]["kernel"])
    in_order = mixed(np.arange(length))
    np.testing.assert_array_equal(in_order[0, :frames * height * width], expected)
    shuffled = rng.permutation(length)
    np.testing.assert_array_equal(mixed(shuffled), in_order[:, shuffled])


def test_a_moonvit_pools_frames_finitely_beside_a_padded_item():
    """MoonViT's packed rows beside the padding a processor gives a row of fewer items (a grid of no frame,
    `Processor._images`): a clip's pooled tokens its frames' mean at its first frame's places, the padded
    row zero, and the gradient of the clip's tokens finite everywhere, the padded row's division included
    (row 855's review)."""
    from dew.nn.vision.moonvit import MoonViT

    tower = MoonViT(vt_hidden_size=16, vt_intermediate_size=24, vt_num_attention_heads=2,
                    vt_num_hidden_layers=1, patch_size=2, init_pos_emb_height=4, init_pos_emb_time=2)
    module = tower.build()
    grid = jnp.asarray([[2, 4, 4], [0, 2, 2]], jnp.int32)
    pixels = jax.random.normal(jax.random.key(0), (2, 32, 12))
    variables = module.init(jax.random.key(1), pixels, grid)
    out = np.asarray(module.apply(variables, pixels, grid))
    assert np.isfinite(out).all()
    # The clip's 16 places of its first frame hold its tokens; the rest of its row and the padded row none.
    assert np.abs(out[0, :16]).sum(axis=-1).min() > 0
    assert not out[0, 16:].any() and not out[1].any()
    gradient = jax.grad(lambda held: module.apply(held, pixels, grid)[0, :16].sum())(variables)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(gradient))


def exact_bicubic_weights(size_in: int, size_out: int, antialias: bool) -> list[list[fractions.Fraction]]:
    """The `[size_out][size_in]` weights of torch's bicubic resize along one axis (`align_corners=False`),
    exactly: without antialiasing the cubic of a = -3/4 on the four taps about each half-pixel source
    position, clamped at the border; with it the cubic of a = -1/2 over the support widened by the
    shrink, normalized by its total (`dew.nn.blocks.torch_bicubic_resize`, torch's upsample_bicubic2d)."""
    half = fractions.Fraction(1, 2)
    scale = fractions.Fraction(size_in, size_out)
    weights = [[fractions.Fraction(0)] * size_in for _ in range(size_out)]
    for row in range(size_out):
        if antialias:
            stretch, center = max(scale, 1), scale * (row + half)
            low = max(math.floor(center - 2 * stretch + half), 0)
            high = min(math.floor(center + 2 * stretch + half), size_in)
            taps = {}
            for tap in range(low, high):
                x = abs((tap - center + half) / stretch)
                taps[tap] = ((3 * half * x - 5 * half) * x * x + 1 if x < 1 else
                             ((-x * half + 5 * half) * x - 4) * x + 2 if x < 2 else 0)
            total = sum(taps.values())
            for tap, weight in taps.items():
                weights[row][tap] = weight / total
            continue
        a = fractions.Fraction(-3, 4)
        source = scale * (row + half) - half
        index = math.floor(source)
        t = source - index
        for tap, x in zip(range(index - 1, index + 3), (t + 1, t, 1 - t, 2 - t), strict=True):
            near = ((a + 2) * x - (a + 3)) * x * x + 1
            far = ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
            weights[row][min(max(tap, 0), size_in - 1)] += near if x <= 1 else far
    return weights


@pytest.mark.parametrize("antialias", [False, True], ids=["plain", "antialiased"])
@pytest.mark.parametrize("size_out", [5, 11], ids=["shrinking", "growing"])
def test_a_float64_bicubic_resize_weighs_each_pixel_in_float64(antialias, size_out):
    """A float64 bicubic resize (`torch_bicubic_resize`, REPA's and RAE's preprocessing, a vision tower's
    position table) weighs each source pixel in float64, as torch's resize of a float64 input does (row
    4445): each unit image of seven rows (three columns wide, which a resize to three columns weighs by
    one, exactly) resized to five or eleven rows reads out every weight of its row, within float64's
    rounding of its own computation of their exact values (`exact_bicubic_weights`,
    `assert_computes_the_oracle`): its host chain (the source coordinate, the cubic's Horner steps, and the
    border's tap sum or the antialiased total and its division) and the resize's contraction
    (`chain_roundings`). Weights built in float32 sit near 1e-8 from them, far past the bound."""
    from reference_error import assert_computes_the_oracle, chain_roundings

    from dew.nn.blocks import torch_bicubic_resize

    size_in = 7
    units = np.eye(size_in)[:, :, None, None].repeat(3, axis=2)
    with jax.enable_x64(new_val=True):
        resize = functools.partial(torch_bicubic_resize, height=size_out, width=3, antialias=antialias)
        ours = np.asarray(resize(jnp.asarray(units)))
        contraction = chain_roundings(jax.make_jaxpr(resize)(jnp.asarray(units)))
    assert ours.dtype == np.float64
    taps = math.ceil(4 * max(size_in / size_out, 1)) + 2
    host = 3 + 3 + 6 + taps + 1 if antialias else 4 + 1 + 6 + 3
    exact = np.asarray([[float(weight) for weight in row] for row in exact_bicubic_weights(
        size_in, size_out, antialias)])
    # Pixel by row, each column the same weight.
    assert_computes_the_oracle(ours[..., 0].transpose(1, 0, 2), exact[..., None].repeat(3, axis=2),
                               f"bicubic weights, {size_in} to {size_out} rows", roundings=host + contraction)
