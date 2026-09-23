"""Native Qwen-Image 2.1 VAE against the actual `AutoencoderKLQwenImage21`.

`tools/diffusers_qwen_image_vae_reference.py` builds a tiny instance with the
published architecture's controls (residual stack, RGBA, four spatial
halvings, the published temporal flags), saves it with `save_pretrained`, and
walks it the way `QwenImage21Pipeline` walks one frame, recording in float32
the posterior mean and standard deviation for a rectangular RGBA batch, the
decode of a fixed latent, and the vector-Jacobian products of both against
fixed probes for the input and every parameter.

Arrays are recorded channel-first with the frame axis dropped; the tests move
channels last. Parameter gradients are written back into the source layout
through the same `WeightLayout`s an export uses, so each is compared against
the source tensor it names. Every gap is scaled by max(1, |reference|).
"""

import json
import shutil
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import save_file

from dew.interop.diffusion import component_tensors
from dew.nn.autoencoders.qwen_image import (
    QwenImageAutoencoder,
    load_qwen_image_vae,
    qwen_image_vae_fields,
    qwen_image_vae_path,
)

ROOT = Path(__file__).resolve().parents[1]
FORWARD = 1e-5
GRADIENT = 1e-4


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("qwen-image-vae")
    with tarfile.open(ROOT / "tests/fixtures/qwen_image_vae.tar.xz") as archive:
        archive.extractall(directory, filter="data")
    return directory


@pytest.fixture(scope="module")
def reference(source):
    return dict(np.load(source / "qwen_image_vae.npz"))


@pytest.fixture(scope="module")
def loaded(source):
    return load_qwen_image_vae(source, jnp.float32)


def channels_last(array):
    return np.asarray(array).transpose(0, 2, 3, 1)


def scaled_gap(actual, expected) -> float:
    expected = np.asarray(expected, np.float64)
    difference = np.abs(np.asarray(actual, np.float64) - expected).max()
    return float(difference / max(1.0, float(np.abs(expected).max())))


def parameter_gaps(layouts, gradients, reference, walk: str) -> dict[str, float]:
    """Each native parameter gradient, exported to its source tensor's layout,
    against the source's gradient of that tensor."""
    return {layout.name: scaled_gap(layout.export({"autoencoder": gradients}),
                                    reference[f"{walk}.grad.{layout.name.removeprefix('vae/')}"])
            for layout in layouts}


def test_every_published_tensor_is_mapped_and_exports_bit_identical(source, loaded):
    _, params, layouts, _ = loaded
    tensors = component_tensors(source, "vae")
    assert {layout.name for layout in layouts} == {f"vae/{name}" for name in tensors}
    time_convs = [name for name in tensors if "time_conv" in name]
    assert time_convs, "the fixture must carry the time_conv weights only a second frame reads"
    for layout in layouts:
        written = layout.export({"autoencoder": params})
        published = tensors[layout.name.removeprefix("vae/")]
        assert written.dtype == published.dtype and written.shape == published.shape, layout.name
        assert written.tobytes() == published.tobytes(), layout.name


def posterior(model, params, image):
    """The posterior's mean and standard deviation, split from the encoder's
    moments as `posterior_latent` splits them."""
    moments = model.apply({"params": params}, image, method=lambda vae, x: vae.quant_conv(vae.encoder(x)))
    mean, log_variance = jnp.split(moments, 2, axis=-1)
    return mean, jnp.exp(0.5 * jnp.clip(log_variance, -30.0, 20.0))


def test_the_posterior_matches_the_source(loaded, reference):
    autoencoder, params, _, _ = loaded
    model = autoencoder.model
    image = channels_last(reference["image"])
    mean, std = posterior(model, params, image)
    assert mean.shape == (2, 2, 3, 8)
    gaps = {"mean": scaled_gap(mean, channels_last(reference["mean"])),
            "std": scaled_gap(std, channels_last(reference["std"]))}
    print(f"posterior gaps {gaps}")
    assert max(gaps.values()) < FORWARD, gaps
    latent = model.apply({"params": params}, image, method=model.encode)
    assert np.array_equal(np.asarray(latent), np.asarray(mean))


def test_a_sampled_latent_is_the_mean_plus_std_times_the_draw(loaded, reference):
    autoencoder, params, _, _ = loaded
    model = autoencoder.model
    image = channels_last(reference["image"])
    key = jax.random.key(7)
    mean, std = posterior(model, params, image)
    sampled = model.apply({"params": params}, image, key, method=model.encode)
    expected = mean + std * jax.random.normal(key, mean.shape, mean.dtype)
    assert scaled_gap(sampled, expected) < 1e-6
    assert scaled_gap(sampled, mean) > 1e-2


def test_the_decode_matches_the_source(loaded, reference):
    autoencoder, params, _, _ = loaded
    model = autoencoder.model
    pixels = model.apply({"params": params}, channels_last(reference["latent"]), method=model.decode)
    assert pixels.shape == (2, 32, 48, 4)
    gap = scaled_gap(pixels, channels_last(reference["pixels"]))
    print(f"decode gap {gap:.3g}")
    assert gap < FORWARD
    assert float(jnp.abs(pixels).max()) <= 1.0


def test_encoder_gradients_match_the_source(loaded, reference):
    autoencoder, params, layouts, _ = loaded
    model = autoencoder.model
    probe_mean = channels_last(reference["probe_mean"])
    probe_std = channels_last(reference["probe_std"])

    def objective(params, image):
        mean, std = posterior(model, params, image)
        return jnp.sum(mean * probe_mean) + jnp.sum(std * probe_std)

    grad_params, grad_image = jax.grad(objective, argnums=(0, 1))(params, channels_last(reference["image"]))
    image_gap = scaled_gap(grad_image, channels_last(reference["encode.grad_image"]))
    gaps = parameter_gaps(layouts, grad_params, reference, "encode")
    worst = max(gaps, key=gaps.get)
    print(f"encode: image gradient gap {image_gap:.3g}; worst parameter {worst} {gaps[worst]:.3g}")
    assert image_gap < GRADIENT
    assert gaps[worst] < GRADIENT, worst


def test_decoder_gradients_match_the_source(loaded, reference):
    autoencoder, params, layouts, _ = loaded
    model = autoencoder.model
    probe = channels_last(reference["probe"])

    def objective(params, latent):
        return jnp.sum(model.apply({"params": params}, latent, method=model.decode) * probe)

    grad_params, grad_latent = jax.grad(objective, argnums=(0, 1))(params, channels_last(reference["latent"]))
    latent_gap = scaled_gap(grad_latent, channels_last(reference["decode.grad_latent"]))
    gaps = parameter_gaps(layouts, grad_params, reference, "decode")
    worst = max(gaps, key=gaps.get)
    print(f"decode: latent gradient gap {latent_gap:.3g}; worst parameter {worst} {gaps[worst]:.3g}")
    assert latent_gap < GRADIENT
    assert gaps[worst] < GRADIENT, worst


def test_the_autoencoder_normalizes_as_the_pipeline_does(loaded, reference):
    """`_encode_vae_image` stores `(z - latents_mean) / latents_std` and the
    decode step undoes it as `z * latents_std + latents_mean`, per channel.
    XLA may divide by a broadcast operand through its reciprocal, so the
    formula is held to one float32 rounding rather than to the bit."""
    autoencoder, params, _, config = loaded
    assert autoencoder.downscale_factor == 16 and autoencoder.latent_channels == 8
    mean = np.asarray(config["latents_mean"], np.float32)
    std = np.asarray(config["latents_std"], np.float32)
    image = channels_last(reference["image"])
    raw = np.asarray(autoencoder.encode_batch(params, image))
    normalized = np.asarray(autoencoder.encode(params, image))
    np.testing.assert_allclose(normalized, (raw - mean) / std, rtol=1e-6, atol=1e-7)
    video = np.asarray(autoencoder.encode(params, image[None]))
    np.testing.assert_array_equal(video[0], normalized)

    latent = channels_last(reference["latent"])
    decoded = np.asarray(autoencoder.decode(params, latent))
    np.testing.assert_allclose(decoded, np.asarray(autoencoder.decode_batch(params, latent * std + mean)),
                               rtol=1e-6, atol=1e-6)
    assert np.asarray(autoencoder.decode(params, latent[None])).shape == (1, 2, 32, 48, 4)


def test_a_missing_tensor_is_refused(source, tmp_path):
    shutil.copytree(source / "vae", tmp_path / "vae")
    tensors = component_tensors(source, "vae")
    del tensors["decoder.up_blocks.0.upsampler.time_conv.weight"]
    save_file(tensors, tmp_path / "vae" / "diffusion_pytorch_model.safetensors")
    with pytest.raises(ValueError, match="time_conv"):
        load_qwen_image_vae(tmp_path, jnp.float32)


@pytest.mark.parametrize("name, ndim", [
    ("encoder.down_blocks.0.resnets.0.norm3.gamma", 4),
    ("decoder.mid_block.attentions.0.to_q.weight", 4),
    ("encoder.down_blocks.1.downsampler.resample.0.weight", 4),
    ("vae.encoder.conv_in.weight", 4),
    ("encoder.conv_in.weight", 2),
])
def test_an_unknown_tensor_is_refused(name, ndim):
    with pytest.raises(ValueError):
        qwen_image_vae_path(name, ndim)


@pytest.mark.parametrize("change", [
    {"is_residual": False}, {"dropout": 0.1}, {"patch_size": 2}, {"out_channels": 3},
    {"scale_factor_spatial": 8}, {"temperal_downsample": [False, True, True]},
    {"dim_mult": [1, 3, 2, 4, 4]},
])
def test_an_unsupported_config_is_refused(source, change):
    config = json.loads((source / "vae" / "config.json").read_text())
    qwen_image_vae_fields(config)
    with pytest.raises(ValueError):
        qwen_image_vae_fields({**config, **change})


def test_mismatched_latent_statistics_are_refused(loaded):
    autoencoder, params, _, config = loaded
    with pytest.raises(ValueError, match="latent channel"):
        QwenImageAutoencoder(model=autoencoder.model, params=params,
                             latents_mean=config["latents_mean"][:-1], latents_std=config["latents_std"])
