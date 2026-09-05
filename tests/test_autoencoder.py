"""SimpleAutoEncoder tests.

The tutorial-grade AE has no pretrained weights, so nothing here asserts
reconstruction quality: what matters is that it satisfies the AutoEncoder
contract the samplers and input config depend on (the advertised latent
geometry, video flattening, and the latent normalization seam).
"""

import jax
import jax.numpy as jnp
import pytest

from dew.nn.autoencoders import SimpleAutoEncoder

# Small enough to stay quick on CPU: 3 stages -> downscale factor 8
DEPTHS = (8, 16, 32)
IMAGE_SIZE = 8


@pytest.fixture(scope="module")
def autoencoder():
    return SimpleAutoEncoder(latent_channels=4, feature_depths=DEPTHS)


@pytest.fixture(scope="module")
def image(autoencoder):
    return jax.random.uniform(
        jax.random.PRNGKey(1), (2, IMAGE_SIZE, IMAGE_SIZE, 3), minval=-1.0, maxval=1.0
    )


@pytest.mark.parametrize("depths", [(8,), (8, 16), (8, 16, 32)])
def test_the_advertised_geometry_is_the_encoders(depths):
    """`downscale_factor` and `latent_channels` are what the samplers and the
    input config size latents by, so they must be what the encoder produces
    and the decoder takes back to the image."""
    autoencoder = SimpleAutoEncoder(latent_channels=2, feature_depths=depths)
    size = 2 * autoencoder.downscale_factor
    image = jnp.zeros((1, size, size, 3))
    latent = autoencoder.encode(autoencoder.params, image)
    assert latent.shape == (1, 2, 2, autoencoder.latent_channels)
    assert autoencoder.decode(autoencoder.params, latent).shape == image.shape


def test_video_frames_match_the_same_frames_encoded_as_images(autoencoder):
    """The B*T flattening must not mix frames together."""
    frames = jax.random.uniform(
        jax.random.PRNGKey(3), (6, IMAGE_SIZE, IMAGE_SIZE, 3), minval=-1.0, maxval=1.0
    )
    video = frames.reshape(2, 3, IMAGE_SIZE, IMAGE_SIZE, 3)
    per_frame = autoencoder.encode(autoencoder.params, frames)
    assert jnp.allclose(
        autoencoder.encode(autoencoder.params, video), per_frame.reshape(2, 3, *per_frame.shape[1:]), atol=1e-6
    )


def test_latent_normalization_is_inverted_by_decode(autoencoder, image):
    """encode applies (z - shift) * scale and decode must undo exactly it, so
    decode(encode(x)) is the raw decoder applied to the raw latent."""
    normalized = SimpleAutoEncoder(
        latent_channels=4,
        feature_depths=DEPTHS,
        latent_shift=0.3,
        latent_scale=2.5,
        params=autoencoder.params,
    )
    raw_latent = autoencoder.encode(autoencoder.params, image)  # identity normalization by default
    latent = normalized.encode(normalized.params, image)
    assert jnp.allclose(latent, (raw_latent - 0.3) * 2.5, atol=1e-5)
    assert jnp.allclose(normalized.decode(normalized.params, latent), autoencoder.decode(autoencoder.params, raw_latent), atol=1e-5)


def test_normalized_latents_can_be_whitened(image):
    """Point of the seam: pass dataset statistics and the latents come out
    centred and unit-variance for the diffusion model."""
    plain = SimpleAutoEncoder(latent_channels=4, feature_depths=DEPTHS)
    latents = plain.encode(plain.params, image)
    whitened = SimpleAutoEncoder(
        latent_channels=4,
        feature_depths=DEPTHS,
        latent_shift=float(jnp.mean(latents)),
        latent_scale=1.0 / float(jnp.std(latents)),
        params=plain.params,
    ).encode(plain.params, image)
    assert abs(float(jnp.mean(whitened))) < 1e-4
    assert float(jnp.std(whitened)) == pytest.approx(1.0, abs=1e-4)


def test_group_norm_survives_depths_not_divisible_by_norm_groups():
    """GroupNorm needs a divisor of the channel count; the AE picks one."""
    autoencoder = SimpleAutoEncoder(latent_channels=2, feature_depths=(12,), norm_groups=8)
    image = jnp.zeros((1, 2, 2, 3))
    assert autoencoder(autoencoder.params, image).shape == image.shape


def test_params_can_be_reused_across_instances(autoencoder, image):
    """Checkpointed weights load by construction, no re-init."""
    reloaded = SimpleAutoEncoder(
        latent_channels=4, feature_depths=DEPTHS, params=autoencoder.params
    )
    assert jnp.allclose(reloaded.encode(reloaded.params, image), autoencoder.encode(autoencoder.params, image), atol=1e-6)


def test_fresh_instances_get_different_random_weights(image):
    """No pretrained weights: two seeds must not agree."""
    a = SimpleAutoEncoder(latent_channels=4, feature_depths=DEPTHS, key=jax.random.PRNGKey(0))
    b = SimpleAutoEncoder(latent_channels=4, feature_depths=DEPTHS, key=jax.random.PRNGKey(1))
    assert not jnp.allclose(a.encode(a.params, image), b.encode(b.params, image), atol=1e-4)

