"""The 16-channel latent space of SD3.5 and Flux, through the vendored VAE.

The claim is parity: the same weights and the same pixels through diffusers'
AutoencoderKL and through dew produce the same latent and the same
reconstruction. tools/vae_reference.py writes the fixture under torch and
diffusers, a random-weight 16-channel VAE in the diffusers layout with the
two flags the SD3 config sets (`use_quant_conv` and `use_post_quant_conv`
false), so the comparison runs in CI without a download.

Tolerances and the differences actually observed, fp32 on CPU:

- sd3-tiny: max |latent difference| 9.5e-07 on values reaching 1.2, max
  |reconstruction difference| 1.8e-06 on values reaching 1.5, tolerance 1e-5.
  The normalized round trip through `encode`/`decode` reproduces the
  reference's own reconstruction to 1e-4, at the reference's 9.8 dB (random
  weights reconstruct nothing; the number pins the pair, not the quality).
- stable-diffusion-3-medium-vae, downloaded: 16 channels at an eighth of the
  resolution, and a smooth 64px image round trips at 29.1 dB.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from dew.nn.autoencoders.sd_vae import StableDiffusionVAE

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "vae"
TINY = FIXTURES / "sd3-tiny"
TOLERANCE = 1e-5


def reference():
    return np.load(TINY / "reference.npz")


def synthetic_image():
    recipe = json.loads((TINY / "inputs.json").read_text())["image"]
    return np.random.RandomState(recipe["seed"]).randint(
        0, 256, tuple(recipe["shape"]), dtype=np.uint8)


def largest_difference(actual, expected) -> float:
    return float(np.max(np.abs(np.asarray(actual, np.float32) - expected)))


@pytest.fixture(scope="module")
def autoencoder():
    """The fixture VAE with its own normalization, not the SD1 defaults."""
    return StableDiffusionVAE(str(TINY), dtype=np.float32)


def test_the_latent_space_is_sixteen_channels(autoencoder):
    """What the seam reports is what the config says, so a diffusion model
    sized off `latent_channels` denoises 16 of them."""
    assert autoencoder.latent_channels == 16
    assert autoencoder.downscale_factor == 2
    assert autoencoder.latent_shift == pytest.approx(0.0609)
    assert autoencoder.latent_scale == pytest.approx(1.5305)


def test_the_encoder_matches_the_reference_latent(autoencoder):
    """The posterior mean of the reference, from the same pixels, before the
    latent normalization the seam applies."""
    expected = reference()
    raw = autoencoder.encode_batch(autoencoder.params, np.asarray(expected["sample"], np.float32))

    difference = largest_difference(raw, expected["latent"])
    assert difference < TOLERANCE, f"max |latent difference| {difference:.3e}"


def test_the_decoder_matches_the_reference_reconstruction(autoencoder):
    expected = reference()
    decoded = autoencoder.decode_batch(autoencoder.params, np.asarray(expected["latent"], np.float32))

    difference = largest_difference(decoded, expected["decoded"])
    assert difference < TOLERANCE, f"max |reconstruction difference| {difference:.3e}"


def test_the_normalized_latent_round_trips(autoencoder):
    """encode applies (z - shift) * scale and decode inverts exactly it, so
    the pair is the identity around the reference's own reconstruction."""
    expected = reference()
    sample = np.asarray(expected["sample"], np.float32)

    latent = autoencoder.encode(autoencoder.params, sample)
    scaled = (np.asarray(expected["latent"]) - 0.0609) * 1.5305
    assert largest_difference(latent, scaled) < 1e-4

    decoded = autoencoder.decode(autoencoder.params, latent)
    difference = largest_difference(decoded, expected["decoded"])
    assert difference < 1e-4, f"max |round trip difference| {difference:.3e}"


def test_a_zeroed_input_convolution_fails_parity(autoencoder):
    """The translated weights are load-bearing: zeroing the encoder's first
    convolution must move the latent past the tolerance, or the parity tests
    above prove nothing."""
    import copy

    expected = reference()
    broken = copy.deepcopy(autoencoder.params)
    kernel = broken["encoder"]["conv_in"]["kernel"]
    broken["encoder"]["conv_in"]["kernel"] = np.zeros_like(kernel)
    mutated = StableDiffusionVAE(str(TINY), dtype=np.float32, params=broken)

    difference = largest_difference(
        mutated.encode_batch(mutated.params, np.asarray(expected["sample"], np.float32)), expected["latent"])
    assert difference > TOLERANCE, f"zeroed conv_in still matches: {difference:.3e}"


@pytest.mark.network
def test_the_real_sd3_vae_round_trips():
    """The released 16-channel VAE, downloaded, on a smooth image: the latent
    is 16 channels at an eighth of the resolution and the reconstruction is
    close in PSNR."""
    import jax.numpy as jnp

    vae = StableDiffusionVAE("chendelong/stable-diffusion-3-medium-vae", dtype=np.float32)
    assert vae.latent_channels == 16 and vae.downscale_factor == 8

    ramp = jnp.linspace(-0.8, 0.8, 64)
    image = jnp.broadcast_to(ramp[None, :, None, None], (1, 64, 64, 3)).transpose(0, 2, 1, 3)
    latent = vae.encode(vae.params, image)
    assert latent.shape == (1, 8, 8, 16)

    reconstruction = vae.decode(vae.params, latent)
    mse = float(jnp.mean((reconstruction - image) ** 2))
    psnr = 10 * np.log10(4.0 / mse)
    assert psnr > 20, f"reconstruction too poor: {psnr:.1f}dB"
