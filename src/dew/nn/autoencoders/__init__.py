"""Autoencoders a latent diffusion model reads and writes its samples through: the interface (`AutoEncoder`)
and the KL and Stable Diffusion VAEs. Each name is imported from its own module when first read, so a reader
of the interface alone reads neither VAE."""

from typing import TYPE_CHECKING

from dew._exports import lookup as _lookup

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew.nn.autoencoders.api import AutoEncoder, LatentAffine, LatentNormalization
    from dew.nn.autoencoders.kl import AutoencoderKL
    from dew.nn.autoencoders.sd_vae import StableDiffusionVAE

_EXPORTS = {
    "AutoEncoder": "dew.nn.autoencoders.api",
    "LatentAffine": "dew.nn.autoencoders.api",
    "LatentNormalization": "dew.nn.autoencoders.api",
    "AutoencoderKL": "dew.nn.autoencoders.kl",
    "StableDiffusionVAE": "dew.nn.autoencoders.sd_vae",
}

__getattr__ = _lookup(__name__, _EXPORTS)

__all__ = [
    "AutoEncoder",
    "AutoencoderKL",
    "LatentAffine",
    "LatentNormalization",
    "StableDiffusionVAE",
]
