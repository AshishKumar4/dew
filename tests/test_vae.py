"""Vendored Stable Diffusion VAE tests, plus the latent normalization seam.

Everything touching the pretrained weights is network-marked: they download
from the HuggingFace Hub on first run. Excluded in CI (-m "not network").
"""

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope="module")
def vae():
    from dew.nn.autoencoders.sd_vae import StableDiffusionVAE
    return StableDiffusionVAE(dtype=jnp.float32)


@pytest.mark.network
def test_vae_roundtrip_reconstructs(vae, rng):
    # A smooth image should survive the encode/decode roundtrip well
    ramp = jnp.linspace(-0.8, 0.8, 64)
    x = jnp.broadcast_to(ramp[None, :, None, None], (1, 64, 64, 3)).transpose(0, 2, 1, 3)
    rec = vae.decode(vae.params, vae.encode(vae.params, x))
    mse = float(jnp.mean((rec - x) ** 2))
    psnr = 10 * np.log10(4.0 / mse)
    assert psnr > 20, f"reconstruction too poor: {psnr:.1f}dB"
