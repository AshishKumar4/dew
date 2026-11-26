"""Native RAE against the actual `AutoencoderRAE`.

`tools/diffusers_rae_reference.py` builds three tiny instances, one per
frozen encoder (DINOv2 with registers, SigLIP, ViT-MAE), saves each with
`save_pretrained`, and records in float32 the latent of a batch with the
gradient of a probe against the pixels, and the decode of a fixed latent
with the gradients of a probe against the latent and every decoder
parameter, and over drawn seeds each walk's recording for check 1. Each
resizes its image and its position table, as the published checkpoints do;
DINOv2's shrinks both, SigLIP's and MAE's enlarge the image.

It also records what the published checkpoints compute in float64 on
inputs numpy builds (`tests/fixtures/rae_published.npz`), which the
`network` test holds the port to after downloading the weights.

Arrays are recorded `[B, C, H, W]` with pixels in [0, 1]; the tests move
channels last and hand the autoencoder pixels in [-1, 1]. The tiny walks are
held to the source's by check 1 over drawn operands and probes, recorded with
their float64 truth (`rae_seeds.npz`, the tool's `seeds`); the published
checkpoints' one draw, to twice the source's own float32 distance from its
float64 result (`scaled_gap`).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from autoencoder_walks import RAE, key_biases, rae_call, wide_rae
from interop_support import (
    assert_layout_tensors,
    assert_walk_as_recorded,
    channels_last,
    dtype_arms,
    extract_fixture,
    fixture_arrays,
    scaled_gap,
)
from model_support import attending
from reference_error import as_given, assert_as_exact, torch_with_gradients, with_gradients
from tool_support import load

from dew.interop.autoencoders.rae import UNREAD, load_dinov2, load_rae
from dew.interop.diffusion import component_tensors

ROOT = Path(__file__).resolve().parents[1]
FORWARD = 1e-5
VARIANTS = ("dinov2", "siglip2", "mae")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return extract_fixture(ROOT / "tests/fixtures/rae.tar.xz", tmp_path_factory.mktemp("rae"))


@pytest.fixture(scope="module", params=VARIANTS)
def variant(request, source):
    directory = source / request.param
    return directory, load_rae(directory), fixture_arrays(directory / "reference.npz")


def test_every_computed_tensor_exports_and_reloads_after_an_encoder_update(variant, tmp_path):
    import shutil

    from dew.interop.decoder_parts import layout_tensors
    from dew.interop.safetensors_io import save_sharded

    directory, (autoencoder, params, layouts, _), reference = variant
    tensors = component_tensors(directory, "")
    read = {name: tensor for name, tensor in tensors.items() if not name.startswith(UNREAD)}
    assert_layout_tensors(layouts, {"autoencoder": params}, read, "vae/")
    image = 2 * channels_last(reference["image"]) - 1
    probe = channels_last(reference["probe_latent"])
    gradient = jax.grad(lambda values: jnp.sum(autoencoder.encode(values, image) * probe))(params)
    trained = jax.tree.map(lambda value, grad: value - 0.01 * grad, params, gradient)
    expected = autoencoder.encode(trained, image)
    assert not np.array_equal(expected, autoencoder.encode(params, image))
    exported = layout_tensors({layout.name.removeprefix("vae/"): layout for layout in layouts},
                              {"autoencoder": trained},
                              retained={name: tensor for name, tensor in tensors.items()
                                        if name.startswith(UNREAD)})
    shutil.copyfile(directory / "config.json", tmp_path / "config.json")
    save_sharded(exported, tmp_path, max_shard_size=20_000, stem="diffusion_pytorch_model")
    restored, variables, _, _ = load_rae(tmp_path)
    np.testing.assert_array_equal(restored.encode(variables, image), expected)


@attending("reference")
@pytest.mark.parametrize("walk", ["encode", "decode"])
def test_every_walk_and_gradient_matches_the_source(variant, walk):
    """The normalized latent of an image in [0, 1], against a drawn probe its pixels' gradient through the
    frozen encoder, and the decode of a latent, against a drawn probe its gradients in the latent and in
    every decoder parameter (the encoder is frozen), each written back into its source tensor's layout,
    on par with `AutoencoderRAE`'s from float64 over drawn operands and probes (check 1, the recording of
    tools/diffusers_rae_reference.py `seeds`). The latent is the encoder's patch tokens as a grid, less
    the checkpoint's mean and over its deviation, per position where it gives one per position; the
    DINOv2 and MAE checkpoints omit encoder_norm_mean and encoder_norm_std, so their latents hold the
    encoder input's default ImageNet statistics (`dew.nn.blocks.IMAGENET_MEAN`) to the source's."""
    directory, (autoencoder, params, layouts, _), _ = variant
    name = directory.name
    decoder = [layout for layout in layouts if layout.name.startswith("vae/decoder.")]
    assert_walk_as_recorded(ROOT / "tests/fixtures/rae_seeds.npz", f"{name} {walk}", RAE[name][walk],
                            rae_call(autoencoder, walk), params, decoder if walk == "decode" else [],
                            wide=rae_call(wide_rae(autoencoder), walk),
                            cancelled=key_biases([layout.name.removeprefix("vae/") for layout in decoder]))


def test_the_latent_takes_the_grid_and_an_empty_batch(variant):
    _, (autoencoder, params, _, _), reference = variant
    image = channels_last(reference["image"])
    latent = autoencoder.encode(params, 2 * image - 1)
    assert latent.shape == (image.shape[0], *autoencoder.latent_shape(image.shape[1:]))
    empty = autoencoder.encode(params, 2 * image[:0] - 1)
    assert empty.shape == latent[:0].shape


@pytest.mark.network
@pytest.mark.parametrize("name", VARIANTS)
def test_the_published_rae_matches_the_source(name, monkeypatch):
    """Each published RAE, downloaded, encodes a smooth 256x256 image and
    decodes a standard normal latent as diffusers' `AutoencoderRAE` does in
    float64, on the channels and pixels `rae_published.npz` keeps, to within
    twice the source's own float32 distance from that result: two float32
    walks rounding in different orders, and SigLIP's residual stream reaches
    several hundred."""
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    tool = load("diffusers_rae_reference")
    repo, revision = tool.PUBLISHED[name]
    reference = np.load(ROOT / "tests/fixtures/rae_published.npz")
    autoencoder, params, _, config = load_rae(repo, revision=revision)
    grid = config["encoder_input_size"] // config["encoder_patch_size"]

    latent = autoencoder.encode(params, 2 * channels_last(tool.smooth_image()) - 1)[
        ..., : tool.LATENT_CHANNELS
    ]
    decoded = autoencoder.decode(
        params, channels_last(tool.normal_latent(config["encoder_hidden_size"], grid))
    )
    gaps = {"latent": scaled_gap(latent, channels_last(reference[f"{name}.latent"])),
            "pixels": scaled_gap((decoded[:, :tool.CROP, :tool.CROP] + 1) / 2,
                                 channels_last(reference[f"{name}.pixels"]))}
    bounds = {key: max(FORWARD, 2 * float(reference[f"{name}.{key}.float32_gap"])) for key in gaps}
    print(f"published {name} gaps {gaps}, bounds {bounds}")
    assert all(gaps[key] < bounds[key] for key in gaps), (gaps, bounds)


PLAIN_SEEDS = 8
"""The plain DINOv2's declared check-1 seeds, sized paired by `seeds_for` on armada (11-14)."""


def test_the_plain_dinov2_matches_transformers(source, tmp_path):
    """transformers' `Dinov2Model` (no registers, the final norm kept), loaded from the fixture's files:
    patch tokens of drawn 112-pixel batches, the 37x37 table resized to the grid without antialiasing,
    and the gradient of a drawn probe against the pixels, on par (check 1) with Dew's, its float64 twin
    the truth (ledger 705)."""
    import torch
    from transformers import Dinov2Model

    from _dew_torch.precision import float64_twin

    module, params, layouts = load_dinov2(source / "dinov2_plain")
    assert {layout.name for layout in layouts} == {
        f"dinov2/{name}"
        for name in component_tensors(source / "dinov2_plain", "")
        if name != "embeddings.mask_token"
    }
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "config.json").write_bytes((source / "dinov2_plain" / "config.json").read_bytes())
    rebound, supplied, supplied_layouts = load_dinov2(metadata, params=params)
    assert supplied is params and supplied_layouts == ()
    module = module.clone(input_size=112)
    narrow = Dinov2Model.from_pretrained(source / "dinov2_plain", local_files_only=True).eval()
    models = {torch.float32: narrow, torch.float64: float64_twin(narrow)}

    def draw(seed):
        generator = np.random.default_rng([710, seed])
        return (generator.standard_normal((1, 3, 112, 112)).astype(np.float32),
                generator.standard_normal((1, 64, 64)).astype(np.float32))

    def tokens(params, pixels):
        return module.apply({"params": params}, pixels).reshape(1, 64, 64)

    compiled = jax.jit(with_gradients(lambda pixels: tokens(params, pixels)))

    @as_given
    def native(pixels, probe):
        out, grad = compiled(jnp.moveaxis(jnp.asarray(pixels), 1, -1), probe)
        return np.asarray(out), np.moveaxis(np.asarray(grad), -1, 1)

    def walk(dtype, pixels, probe):
        # Its own graph: the arms run under `torch.no_grad` (`torch_arms`).
        with torch.enable_grad():
            return torch_with_gradients(lambda pixels: models[dtype](pixels).last_hidden_state[:, 1:])(
                np.asarray(pixels, dtype=np.float64 if dtype == torch.float64 else np.float32), probe)

    reference, truth = dtype_arms(walk)
    assert_as_exact(native, reference, draw, "plain DINOv2", seeds=PLAIN_SEEDS, truth_fn=truth)
    first = jnp.moveaxis(jnp.asarray(draw(0)[0]), 1, -1)
    np.testing.assert_array_equal(rebound.clone(input_size=112).apply({"params": supplied}, first),
                                  module.apply({"params": params}, first))


@pytest.mark.network
def test_the_published_dinov2_matches_transformers(monkeypatch):
    """`facebook/dinov2-base`, downloaded, on the smooth image at 224 pixels,
    against `Dinov2Model` in float64, as the published RAEs are held."""
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    tool = load("diffusers_rae_reference")
    reference = np.load(ROOT / "tests/fixtures/rae_published.npz")
    module, params, _ = load_dinov2(tool.DINOV2[0], revision=tool.DINOV2[1])
    mean, std = (np.asarray(value, np.float32) for value in tool.IMAGENET)
    pixels = (channels_last(tool.smooth_image(224)) - mean) / std
    tokens = module.clone(input_size=224).apply({"params": params}, pixels)[..., :tool.LATENT_CHANNELS]
    gap = scaled_gap(tokens.reshape(1, 256, -1), reference["dinov2_plain.tokens"])
    bound = max(FORWARD, 2 * float(reference["dinov2_plain.tokens.float32_gap"]))
    print(f"published DINOv2 gap {gap:.3g}, bound {bound:.3g}")
    assert gap < bound
