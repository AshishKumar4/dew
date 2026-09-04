from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp


class AutoEncoder(ABC):
    """An encoder and decoder pair a latent diffusion model trains behind.

    A subclass encodes and decodes one batch of frames, `[B, H, W, C]` to
    `[B, h, w, c]` and back; `encode` and `decode` here flatten video
    `[B, T, H, W, C]` to frames around that and apply the latent
    normalization. Latents are normalized as (z - latent_shift) * latent_scale
    on the way out and inverted on the way in, the SD3 convention. The
    defaults are the identity; set them to the dataset's own latent mean and
    1/std so the diffusion model sees roughly unit-variance, zero-mean inputs.
    """

    latent_shift: float = 0.0
    latent_scale: float = 1.0

    @abstractmethod
    def encode_batch(self, x: jnp.ndarray, key: Optional[jax.Array] = None) -> jnp.ndarray:
        """Frames `[B, H, W, C]` to raw latents `[B, h, w, c]`; `key` draws a
        stochastic encoder's sample, and None takes its mean."""

    @abstractmethod
    def decode_batch(self, z: jnp.ndarray) -> jnp.ndarray:
        """Raw latents `[B, h, w, c]` to frames `[B, H, W, C]`."""

    @property
    @abstractmethod
    def downscale_factor(self) -> int:
        """H / h, the spatial factor between a frame and its latent."""

    @property
    @abstractmethod
    def latent_channels(self) -> int:
        """c, the channels of a latent."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name a config refers to this autoencoder by."""

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """The configuration that rebuilds this autoencoder."""

    def encode(self, x: jnp.ndarray, key: Optional[jax.Array] = None) -> jnp.ndarray:
        """Images `[B, H, W, C]` or video `[B, T, H, W, C]` to normalized
        latents with the same leading axes."""
        if x.ndim == 5:
            batch_size, seq_len, height, width, channels = x.shape
            latent = self.encode_batch(x.reshape(-1, height, width, channels), key=key)
            latent = latent.reshape(batch_size, seq_len, *latent.shape[1:])
        else:
            latent = self.encode_batch(x, key=key)
        return (latent - self.latent_shift) * self.latent_scale

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """Normalized latents `[B, h, w, c]` or `[B, T, h, w, c]` back to
        images or video."""
        z = z / self.latent_scale + self.latent_shift
        if z.ndim == 5:
            batch_size, seq_len, height, width, channels = z.shape
            decoded = self.decode_batch(z.reshape(-1, height, width, channels))
            return decoded.reshape(batch_size, seq_len, *decoded.shape[1:])
        return self.decode_batch(z)

    def __call__(self, x: jnp.ndarray, key: Optional[jax.Array] = None) -> jnp.ndarray:
        """Encode then decode."""
        return self.decode(self.encode(x, key=key))
