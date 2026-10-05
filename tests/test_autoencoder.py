"""The AutoEncoder contract, over a tiny randomly initialized AutoencoderKL.

Nothing here asserts reconstruction quality: what matters is the contract the
samplers and input config depend on (the advertised latent geometry, video
flattening, and the latent normalization seam). The last tests hold that
contract to Diffusers' own VAE, posterior and SD3 pipeline normalization
(tests/fixtures/vae/contract.npz, tools/autoencoder_contract_reference.py).
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact_as_the_reference

from dew.nn.autoencoders import AutoencoderKL, StableDiffusionVAE
from dew.nn.autoencoders.kl import posterior_latent

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "vae"

IMAGE_SIZE = 8


def tiny_vae(channels=(8, 16, 16), latent_shift=0.0, latent_scale=1.0, params=None):
    model = AutoencoderKL(channels=channels, blocks_per_level=1, norm_groups=4, dtype=jnp.float32)
    if params is None:
        params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3)))["params"]
    return StableDiffusionVAE(model=model, params=params, dtype=jnp.float32,
                              latent_shift=latent_shift, latent_scale=latent_scale)


@pytest.fixture(scope="module")
def autoencoder():
    return tiny_vae()


@pytest.fixture(scope="module")
def image():
    return jax.random.uniform(
        jax.random.PRNGKey(1), (2, IMAGE_SIZE, IMAGE_SIZE, 3), minval=-1.0, maxval=1.0
    )


@pytest.mark.parametrize("channels", [(8,), (8, 16), (8, 16, 16)])
def test_the_advertised_geometry_is_the_encoders(channels):
    """`downscale_factor` and `latent_channels` are what the samplers and the
    input config size latents by, so they must be what the encoder produces
    and the decoder takes back to the image."""
    autoencoder = tiny_vae(channels)
    size = 2 * autoencoder.downscale_factor
    image = jnp.zeros((1, size, size, 3))
    latent = autoencoder.encode(autoencoder.params, image)
    assert latent.shape == (1, 2, 2, autoencoder.latent_channels)
    assert jnp.all(jnp.isfinite(latent))
    other = autoencoder.encode(autoencoder.params, jnp.ones_like(image))
    assert not jnp.allclose(latent, other, atol=1e-6)
    decoded = autoencoder.decode(autoencoder.params, latent)
    assert decoded.shape == image.shape
    assert jnp.all(jnp.isfinite(decoded))
    # A decoder that dropped its latent, for a bias or a lost residual, would
    # keep every shape in this file and reconstruct the same image regardless.
    assert not jnp.allclose(
        decoded, autoencoder.decode(autoencoder.params, jnp.ones_like(latent)), atol=1e-6)


def test_video_frames_match_the_same_frames_encoded_as_images(autoencoder):
    """The B*T flattening must not mix frames together."""
    frames = jax.random.uniform(
        jax.random.PRNGKey(3), (6, IMAGE_SIZE, IMAGE_SIZE, 3), minval=-1.0, maxval=1.0
    )
    video = frames.reshape(2, 3, IMAGE_SIZE, IMAGE_SIZE, 3)
    per_frame = autoencoder.encode(autoencoder.params, frames)
    video_latent = autoencoder.encode(autoencoder.params, video)
    assert jnp.allclose(video_latent, per_frame.reshape(2, 3, *per_frame.shape[1:]), atol=1e-6)


def test_latent_normalization_is_inverted_by_decode(autoencoder, image):
    """encode applies (z - shift) * scale and decode must undo exactly it, so
    decode(encode(x)) is the raw decoder applied to the raw latent."""
    normalized = tiny_vae(latent_shift=0.3, latent_scale=2.5, params=autoencoder.params)
    raw_latent = autoencoder.encode(autoencoder.params, image)  # identity normalization
    latent = normalized.encode(normalized.params, image)
    assert jnp.allclose(latent, (raw_latent - 0.3) * 2.5, atol=1e-5)
    assert jnp.allclose(normalized.decode(normalized.params, latent),
                        autoencoder.decode(autoencoder.params, raw_latent), atol=1e-5)


@pytest.fixture(scope="module")
def contract():
    with np.load(FIXTURES / "contract.npz") as loaded:
        return dict(loaded)


@pytest.fixture(scope="module")
def sd3_tiny():
    return StableDiffusionVAE(str(FIXTURES / "sd3-tiny"), dtype=jnp.float32)


def test_a_clip_samples_and_normalizes_as_diffusers_does_frame_by_frame(contract, sd3_tiny):
    """A clip's frames encoded as one `[B, T]` batch draw the latent SD3's
    img2img pipeline draws for each frame as an image (`retrieve_latents`,
    then `(z - shift) * scale`), from one standard normal draw, and their
    normalized means are its: by the float64 rule, against Diffusers'
    own VAE and posterior."""
    batch, frames, height, width = json.loads(contract["meta"].tobytes())["clip"]
    clip = jnp.asarray(contract["frames"]).reshape(batch, frames, height, width, 3)
    sampled = sd3_tiny.encode(sd3_tiny.params, clip, key=jax.random.key(7))
    mean = sd3_tiny.encode(sd3_tiny.params, clip)
    assert sampled.shape[:2] == (batch, frames)
    for name, value in (("latents", sampled), ("normalized_mean", mean)):
        assert_as_exact_as_the_reference(np.asarray(value).reshape(-1, *value.shape[2:]),
                                         contract[f"fp32.{name}"], contract[f"fp64.{name}"], name)


def test_a_clip_decodes_as_diffusers_decodes_its_frames(contract, sd3_tiny):
    """Normalized latents of a clip decode to what Diffusers' VAE decodes of
    each frame's latent taken back SD3's way (`z / scale + shift`)."""
    latents = jnp.asarray(contract["fp32.normalized_mean"])
    clip = latents.reshape(2, 3, *latents.shape[1:])
    decoded = sd3_tiny.decode(sd3_tiny.params, clip)
    assert_as_exact_as_the_reference(np.asarray(decoded).reshape(-1, *decoded.shape[2:]),
                                     contract["fp32.decoded"], contract["fp64.decoded"], "decoded")


def test_a_posterior_draw_clamps_its_log_variance_as_diffusers_does(contract):
    """`posterior_latent` against `DiagonalGaussianDistribution.sample` from
    one draw, on log-variances past the [-30, 20] clamp on both sides."""
    drawn = posterior_latent(jnp.asarray(contract["moments"]), jax.random.key(7))
    assert_as_exact_as_the_reference(np.asarray(drawn), contract["fp32.posterior"],
                                     contract["fp64.posterior"], "posterior")
