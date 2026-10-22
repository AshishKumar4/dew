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
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from interop_support import (
    assert_coder_vjps,
    assert_layout_tensors,
    channels_last,
    extract_fixture,
    fixture_arrays,
    posterior_of,
    scaled_gap,
)

from dew.interop.diffusion import component_tensors
from dew.nn.autoencoders.qwen_image import load_qwen_image_vae

ROOT = Path(__file__).resolve().parents[1]
FORWARD = 1e-5
GRADIENT = 1e-4


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return extract_fixture(ROOT / "tests/fixtures/qwen_image_vae.tar.xz",
                           tmp_path_factory.mktemp("qwen-image-vae"))


@pytest.fixture(scope="module")
def reference(source):
    return fixture_arrays(source / "qwen_image_vae.npz")


@pytest.fixture(scope="module")
def loaded(source):
    return load_qwen_image_vae(source, jnp.float32)





def test_a_configs_training_dropout_loads_the_same_autoencoder(source, loaded, tmp_path):
    """`dropout` acts in a training forward alone, which a frozen autoencoder
    never runs: a config that states one loads the same model and weights."""
    copy = shutil.copytree(source, tmp_path / "source")
    config = json.loads((copy / "vae" / "config.json").read_text())
    (copy / "vae" / "config.json").write_text(json.dumps({**config, "dropout": 0.1}))
    autoencoder, params, _, _ = load_qwen_image_vae(copy, jnp.float32)
    assert autoencoder.model == loaded[0].model
    for got, want in zip(jax.tree.leaves(params), jax.tree.leaves(loaded[1]), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_every_published_tensor_is_mapped_and_exports_bit_identical(source, loaded):
    _, params, layouts, _ = loaded
    tensors = component_tensors(source, "vae")
    time_convs = [name for name in tensors if "time_conv" in name]
    assert time_convs, "the fixture must carry the time_conv weights only a second frame reads"
    assert_layout_tensors(layouts, {"autoencoder": params}, tensors, "vae/")


def posterior(model, params, image):
    """The posterior's mean and standard deviation, split from the encoder's
    moments as `posterior_latent` splits them."""
    return posterior_of(model.apply({"params": params}, image,
                                    method=lambda vae, x: vae.quant_conv(vae.encoder(x))))


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


def test_the_coder_gradients_match_the_source(loaded, reference):
    assert_coder_vjps(loaded, reference, "image", posterior, GRADIENT)


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
