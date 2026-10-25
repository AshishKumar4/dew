"""Native RAE against the actual `AutoencoderRAE`.

`tools/diffusers_rae_reference.py` builds three tiny instances, one per
frozen encoder (DINOv2 with registers, SigLIP, ViT-MAE), saves each with
`save_pretrained`, and records in float32 the latent of a batch with the
gradient of a probe against the pixels, and the decode of a fixed latent
with the gradients of a probe against the latent and every decoder
parameter. Each resizes its image and its position table, as the published
checkpoints do; DINOv2's shrinks both, SigLIP's and MAE's enlarge the image.

It also records what the published checkpoints compute in float64 on
inputs numpy builds (`tests/fixtures/rae_published.npz`), which the
`network` test holds the port to after downloading the weights.

Arrays are recorded `[B, C, H, W]` with pixels in [0, 1]; the tests move
channels last and hand the autoencoder pixels in [-1, 1]. Every gap is
scaled by max(1, |reference|).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import (
    assert_autoencoder_vjp,
    assert_layout_tensors,
    channels_last,
    extract_fixture,
    fixture_arrays,
    scaled_gap,
)
from tool_support import load

from dew.interop.diffusion import component_tensors
from dew.nn.autoencoders.rae import UNREAD, load_dinov2, load_rae

ROOT = Path(__file__).resolve().parents[1]
FORWARD = 1e-5
GRADIENT = 1e-4
VARIANTS = ("dinov2", "siglip2", "mae")


def test_a_ninety_six_wide_encoder_imports_its_full_source_projections(tmp_path):
    import torch
    from diffusers import AutoencoderRAE
    from reference_error import assert_as_exact, seeds_for

    from tools.torch_precision import float64, float64_twin

    torch.manual_seed(53)
    source = AutoencoderRAE(
        encoder_type="mae", encoder_hidden_size=96, encoder_num_hidden_layers=1,
        encoder_patch_size=16, encoder_input_size=32, patch_size=16,
        decoder_hidden_size=32, decoder_num_hidden_layers=1,
        decoder_num_attention_heads=4, decoder_intermediate_size=64).eval()
    source.save_pretrained(tmp_path)
    native, variables, _, _ = load_rae(tmp_path)
    exact = float64_twin(source)

    def draw(seed):
        return (np.random.default_rng(seed).random((4, 32, 32, 3), dtype=np.float32),)

    def observed(model, images):
        pixels = torch.from_numpy(np.asarray(images).copy()).permute(0, 3, 1, 2)
        pixels.requires_grad_(requires_grad=True)
        latent = model.encode(pixels).latent
        gradient, = torch.autograd.grad(latent.sum(), pixels)
        return {"latent": latent.detach().numpy().transpose(0, 2, 3, 1),
                "gradient": gradient.numpy().transpose(0, 2, 3, 1)}

    def truth(images):
        with float64():
            return observed(exact, images)

    def reference(images):
        return observed(source, images)

    @jax.jit
    def ours(images):
        def encode(pixels):
            return native.encode(variables, 2 * pixels - 1)
        latent, pull = jax.vjp(encode, images)
        gradient, = pull(jnp.ones_like(latent))
        return {"latent": latent, "gradient": gradient}

    count = seeds_for(reference, draw, "96-wide RAE", truth_fn=truth)
    assert_as_exact(ours, reference, draw, "96-wide RAE", seeds=count, truth_fn=truth)


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


def test_the_latent_matches_the_source(variant):
    """The normalized latent: the encoder's patch tokens as a grid, less the
    checkpoint's mean and over its deviation, per position where it gives
    one per position. The DINOv2 and MAE checkpoints omit encoder_norm_mean
    and encoder_norm_std, so their latents hold the encoder input's default
    ImageNet statistics (`dew.nn.blocks.IMAGENET_MEAN`) to the source's."""
    _, (autoencoder, params, _, config), reference = variant
    image = channels_last(reference["image"])
    latent = autoencoder.encode(params, 2 * image - 1)
    expected = channels_last(reference["latent"])
    assert latent.shape == expected.shape == (image.shape[0], *autoencoder.latent_shape(image.shape[1:]))
    gap = scaled_gap(latent, expected)
    print(f"{config['encoder_type']} latent gap {gap:.3g}")
    assert gap < FORWARD
    empty = autoencoder.encode(params, 2 * image[:0] - 1)
    assert empty.shape == expected[:0].shape


def test_the_decode_matches_the_source(variant):
    _, (autoencoder, params, _, config), reference = variant
    decoded = autoencoder.decode(params, channels_last(reference["code"]))
    gap = scaled_gap((decoded + 1) / 2, channels_last(reference["pixels"]))
    print(f"{config['encoder_type']} decode gap {gap:.3g}")
    assert gap < FORWARD


def test_the_pixel_gradient_through_the_frozen_encoder_matches_the_source(variant):
    _, (autoencoder, params, _, config), reference = variant
    probe = channels_last(reference["probe_latent"])
    grad = jax.grad(lambda image: jnp.sum(autoencoder.encode(params, 2 * image - 1) * probe))(
        channels_last(reference["image"]))
    gap = scaled_gap(grad, channels_last(reference["encode.grad_image"]))
    print(f"{config['encoder_type']} pixel gradient gap {gap:.3g}")
    assert gap < GRADIENT


def test_decoder_gradients_match_the_source(variant):
    """The decoder's parameters; the encoder is frozen."""
    _, (autoencoder, params, layouts, _), reference = variant
    assert_autoencoder_vjp(
        "decode", lambda params, code: (autoencoder.decode(params, code) + 1) / 2, params,
        channels_last(reference["code"]), [channels_last(reference["probe"])],
        [layout for layout in layouts if layout.name.startswith("vae/decoder.")], reference,
        channels_last(reference["decode.grad_code"]), GRADIENT)


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


def test_the_plain_dinov2_matches_transformers(source, tmp_path):
    """transformers' `Dinov2Model` (no registers, the final norm kept):
    patch tokens of a normalized 112-pixel batch, the 37x37 table resized to
    the grid without antialiasing, and the gradient of a probe against the
    pixels."""
    module, params, layouts = load_dinov2(source / "dinov2_plain")
    reference = dict(np.load(source / "dinov2_plain" / "reference.npz"))
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
    pixels = channels_last(reference["pixels"])
    tokens = module.apply({"params": params}, pixels)
    np.testing.assert_array_equal(
        rebound.clone(input_size=112).apply({"params": supplied}, pixels), tokens)
    assert tokens.shape == (1, 8, 8, 64)
    probe = reference["probe"].reshape(1, 8, 8, 64)
    grad = jax.grad(lambda pixels: jnp.sum(module.apply({"params": params}, pixels) * probe))(pixels)
    gaps = {"tokens": scaled_gap(tokens.reshape(1, 64, 64), reference["tokens"]),
            "gradient": scaled_gap(grad, channels_last(reference["grad_pixels"]))}
    print(f"plain DINOv2 gaps {gaps}")
    assert gaps["tokens"] < FORWARD and gaps["gradient"] < GRADIENT, gaps


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
