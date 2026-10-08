from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.objectives.base import Variables


class AutoEncoder(ABC):
    """Encodes images or video to the latents a latent diffusion model trains on, and decodes them back.

    A subclass encodes and decodes one batch of frames, `[B, H, W, C]` to
    `[B, h, w, c]` and back. `encode` and `decode` here take video
    `[B, T, H, W, C]` frame by frame, unless the subclass encodes time itself
    (`encode_video`, `decode_video`, `latent_shape`), and they apply the
    latent normalization. On the way out, latents are normalized as
    (z - latent_shift) * latent_scale, and on the way in the normalization is
    inverted, the SD3 convention. `latent_shift` and `latent_scale` are each
    one number or one per latent channel. The defaults are the identity; set
    them to the dataset's own latent mean and 1/std so the diffusion model
    sees roughly unit-variance, zero-mean inputs.

    The weights are an argument, as a `ConditionEncoder`'s are. `params` holds
    what a run loaded, and every call takes the tree to use, so the layout
    places the weights and the checkpoint stores them.
    """

    latent_shift: float | np.ndarray | jax.Array = 0.0
    latent_scale: float | np.ndarray | jax.Array = 1.0
    params: Variables

    def to_json(self) -> dict:
        """Return the record of this built-in autoencoder's constructor, without its parameters.

        The base class raises `TypeError`; a built-in subclass returns its own
        record.
        """
        raise TypeError(f"{type(self).__name__} needs an explicit autoencoder record declaration")

    @classmethod
    def from_json(cls, record, *, params: Variables) -> AutoEncoder:
        """Rebuild a built-in autoencoder from its `record`, around the checkpoint's own `params`.

        The built-in record is 'sd_vae'; any other name raises `ValueError`.
        """
        from .sd_vae import StableDiffusionVAE
        if record['name'] == 'sd_vae':
            from dew.registry import from_record

            from .kl import AutoencoderKL
            fields = dict(record['fields'])
            return StableDiffusionVAE(model=from_record(AutoencoderKL, fields.pop('model')), params=params,
                                      **fields)
        raise ValueError(f"{record['name']!r} is not a built-in autoencoder declaration")

    @abstractmethod
    def encode_batch(self, params, x: jnp.ndarray,
                     key: jax.Array | None = None) -> jnp.ndarray:
        """Encode frames `[B, H, W, C]` to raw latents `[B, h, w, c]`.

        `key` draws a stochastic encoder's sample, and None takes its mean.
        """

    @abstractmethod
    def decode_batch(self, params, z: jnp.ndarray) -> jnp.ndarray:
        """Decode raw latents `[B, h, w, c]` to frames `[B, H, W, C]`."""

    @property
    @abstractmethod
    def downscale_factor(self) -> int:
        """H / h, the spatial factor between a frame and its latent."""

    @property
    @abstractmethod
    def latent_channels(self) -> int:
        """c, the number of channels in a latent."""

    def moments(self, params, images: jnp.ndarray) -> jnp.ndarray:
        """Return the KL posterior of frames `[B, H, W, C]`, its mean and log-variance stacked on channels.

        A KL autoencoder's latent is one draw from it normalized by
        `latent_shift` and `latent_scale` alone, `encode` being
        `(posterior_latent(moments, key) - latent_shift) * latent_scale`, so a
        loss may train it through the posterior and renormalize the draw
        (REPA-E). Any other autoencoder raises TypeError, naming itself.
        """
        raise TypeError(f"{type(self).__name__} has no KL posterior its latent is a draw of")

    def decode_raw(self, params, latents: jnp.ndarray) -> jnp.ndarray:
        """Decode a draw from `moments`' posterior to frames; raises as `moments` does."""
        raise TypeError(f"{type(self).__name__} has no KL posterior whose draws it decodes")

    def latent_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """Return the latent shape for one example of `shape`.

        `shape` is `(H, W, C)` or, for video, `(T, H, W, C)`.
        """
        *lead, height, width, _ = shape
        factor = self.downscale_factor
        return (*lead, height // factor, width // factor, self.latent_channels)

    def encode_video(self, params, x: jnp.ndarray, key: jax.Array | None = None) -> jnp.ndarray:
        """Encode video `[B, T, H, W, C]` to raw latents, frame by frame."""
        batch_size, seq_len, height, width, channels = x.shape
        latent = self.encode_batch(params, x.reshape(-1, height, width, channels), key=key)
        return latent.reshape(batch_size, seq_len, *latent.shape[1:])

    def decode_video(self, params, z: jnp.ndarray) -> jnp.ndarray:
        """Decode raw latents `[B, t, h, w, c]` to video, frame by frame."""
        batch_size, seq_len, height, width, channels = z.shape
        decoded = self.decode_batch(params, z.reshape(-1, height, width, channels))
        return decoded.reshape(batch_size, seq_len, *decoded.shape[1:])

    def encode(self, params, x: jnp.ndarray,
               key: jax.Array | None = None) -> jnp.ndarray:
        """Encode images `[B, H, W, C]` or video `[B, T, H, W, C]` to normalized latents."""
        latent = self.encode_video(params, x, key) if x.ndim == 5 else self.encode_batch(params, x, key=key)
        shift, scale = (jnp.asarray(value, latent.dtype) for value in (self.latent_shift, self.latent_scale))
        return (latent - shift) * scale

    def decode(self, params, z: jnp.ndarray) -> jnp.ndarray:
        """Decode normalized latents `[B, h, w, c]` or `[B, t, h, w, c]` back to images or video."""
        shift, scale = (jnp.asarray(value, z.dtype) for value in (self.latent_shift, self.latent_scale))
        z = z / scale + shift
        return self.decode_video(params, z) if z.ndim == 5 else self.decode_batch(params, z)

    def __call__(self, params, x: jnp.ndarray,
                 key: jax.Array | None = None) -> jnp.ndarray:
        """Encode `x`, then decode it."""
        return self.decode(params, self.encode(params, x, key=key))


class ModuleAutoEncoder[M: nn.Module](AutoEncoder):
    """A frozen flax autoencoder module and the params it loaded. The
    module's `encode(x, key)` and `decode(z)` each take one batch of frames,
    and each runs as one jitted `apply`."""

    def __init__(self, model: M, params: Variables):
        self.model = model
        self.params = params
        self.encode_single_frame = jax.jit(lambda params, image, key=None: model.apply(
            {"params": params}, image, key, method=model.encode))
        self.decode_single_frame = jax.jit(lambda params, latent: model.apply(
            {"params": params}, latent, method=model.decode))

    def encode_batch(self, params, x, key=None):
        return self.encode_single_frame(params, x, key)

    def decode_batch(self, params, z):
        return self.decode_single_frame(params, z)
