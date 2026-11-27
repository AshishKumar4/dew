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
the source tensor it names. Each walk is held to the source's by check 1 over
drawn operands and probes, recorded with its float64 truth
(`qwen_image_vae_seeds.npz`, the tool's `seeds`).
"""

import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from autoencoder_walks import QWEN_IMAGE_VAE, posterior, qwen_image_vae_call
from interop_support import (
    assert_layout_tensors,
    assert_walk_as_recorded,
    channels_last,
    extract_fixture,
    fixture_arrays,
)
from model_support import attending

from dew.interop.autoencoders.qwen_image import load_qwen_image_vae
from dew.interop.diffusion import component_tensors

ROOT = Path(__file__).resolve().parents[1]


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


@attending("reference")
@pytest.mark.parametrize("walk", [
    "encode",
    # Its recording's 512 seeds, which check 1's stages draw five times over, past one group's 120 s
    # (143 s in train 1342): drawn in armada's map tasks.
    pytest.param("decode", id="decode", marks=pytest.mark.check1_shards(seeds=512, per_task=128))])
def test_the_posterior_and_the_decode_and_every_gradient_match_the_source(loaded, walk):
    """The posterior's mean and standard deviation of an RGBA image, and the decode of a latent, and
    against drawn probes the gradients of their probed sum in the operand and in every parameter (zero
    where a walk never reads one: the other half of the autoencoder, the `time_conv` weights only a second
    frame reaches), each written back into its source tensor's layout, on par with
    `AutoencoderKLQwenImage21`'s from float64 over drawn operands and probes (check 1)."""
    autoencoder, params, layouts, _ = loaded
    model = autoencoder.model
    assert_walk_as_recorded(ROOT / "tests/fixtures/qwen_image_vae_seeds.npz", walk, QWEN_IMAGE_VAE[walk],
                            qwen_image_vae_call(model, walk), params, layouts,
                            wide=qwen_image_vae_call(model.clone(dtype=jnp.float64), walk))


def test_the_encode_is_the_posteriors_mean(loaded, reference):
    autoencoder, params, _, _ = loaded
    model = autoencoder.model
    image = channels_last(reference["image"])
    mean, _ = posterior(model, params, image)
    assert mean.shape == (2, 2, 3, 8)
    latent = model.apply({"params": params}, image, method=model.encode)
    assert np.array_equal(np.asarray(latent), np.asarray(mean))


def test_a_sampled_latent_is_the_mean_plus_std_times_the_draw(loaded, reference):
    """An encode given a key draws Diffusers' `DiagonalGaussianDistribution.sample`: the posterior's mean
    plus its deviation times the key's standard normal draw, in float32, each step rounded as that
    formula rounds it, so bit for bit; and not its mean."""
    autoencoder, params, _, _ = loaded
    model = autoencoder.model
    image = channels_last(reference["image"])
    key = jax.random.key(7)
    mean, std = posterior(model, params, image)
    sampled = model.apply({"params": params}, image, key, method=model.encode)
    expected = mean + std * jax.random.normal(key, mean.shape, jnp.float32)
    np.testing.assert_array_equal(np.asarray(sampled), np.asarray(expected))
    assert not np.array_equal(np.asarray(sampled), np.asarray(mean))


def test_the_decode_clamps_its_pixels(loaded, reference):
    autoencoder, params, _, _ = loaded
    model = autoencoder.model
    pixels = model.apply({"params": params}, channels_last(reference["latent"]), method=model.decode)
    assert pixels.shape == (2, 32, 48, 4)
    assert float(jnp.abs(pixels).max()) <= 1.0


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
