from __future__ import annotations

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.objectives.base import Variables


class AutoEncoder(ABC):
    """An encoder and decoder pair a latent diffusion model trains behind.

    A subclass encodes and decodes one batch of frames, `[B, H, W, C]` to
    `[B, h, w, c]` and back; `encode` and `decode` here take video
    `[B, T, H, W, C]` frame by frame unless the subclass encodes time itself
    (`encode_video`, `decode_video`, `latent_shape`), and apply the latent
    normalization. Latents are normalized as (z - latent_shift) * latent_scale
    on the way out and inverted on the way in, the SD3 convention; each is
    one number or one per latent channel. The defaults are the identity; set
    them to the dataset's own latent mean and 1/std so the diffusion model
    sees roughly unit-variance, zero-mean inputs.

    The weights are an argument, as a `ConditionEncoder`'s are: `params` holds
    what a run loaded, and every call takes the tree to use, so the layout
    places the weights and the checkpoint carries them.
    """

    latent_shift: float | np.ndarray | jax.Array = 0.0
    latent_scale: float | np.ndarray | jax.Array = 1.0
    params: Variables

    def to_json(self) -> dict:
        """Declare this built-in autoencoder's constructor, without its parameters."""
        raise TypeError(f"{type(self).__name__} needs an explicit autoencoder record declaration")

    @classmethod
    def from_json(cls, record, *, params: Variables) -> AutoEncoder:
        """Rebuild maintained autoencoders around the checkpoint's own parameters."""
        from .sd_vae import StableDiffusionVAE
        if record['name'] == 'sd_vae':
            from dew.registry import resolve_dtype

            from .kl import AutoencoderKL
            fields = dict(record['fields'])
            model = dict(fields.pop('model'))
            model['dtype'] = resolve_dtype(model['dtype'])
            for name in ('channels', 'decoder_channels'):
                if model.get(name) is not None:
                    model[name] = tuple(model[name])
            return StableDiffusionVAE(model=AutoencoderKL(**model), params=params, **fields)
        raise ValueError(f"{record['name']!r} is not a built-in autoencoder declaration")

    @abstractmethod
    def encode_batch(self, params, x: jnp.ndarray,
                     key: jax.Array | None = None) -> jnp.ndarray:
        """Frames `[B, H, W, C]` to raw latents `[B, h, w, c]`; `key` draws a
        stochastic encoder's sample, and None takes its mean."""

    @abstractmethod
    def decode_batch(self, params, z: jnp.ndarray) -> jnp.ndarray:
        """Raw latents `[B, h, w, c]` to frames `[B, H, W, C]`."""

    @property
    @abstractmethod
    def downscale_factor(self) -> int:
        """H / h, the spatial factor between a frame and its latent."""

    @property
    @abstractmethod
    def latent_channels(self) -> int:
        """c, the channels of a latent."""

    def latent_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        """The latent shape of one example of `shape`: `(H, W, C)` or, for
        video, `(T, H, W, C)`."""
        *lead, height, width, _ = shape
        factor = self.downscale_factor
        return (*lead, height // factor, width // factor, self.latent_channels)

    def encode_video(self, params, x: jnp.ndarray, key: jax.Array | None = None) -> jnp.ndarray:
        """Video `[B, T, H, W, C]` to raw latents, frame by frame."""
        batch_size, seq_len, height, width, channels = x.shape
        latent = self.encode_batch(params, x.reshape(-1, height, width, channels), key=key)
        return latent.reshape(batch_size, seq_len, *latent.shape[1:])

    def decode_video(self, params, z: jnp.ndarray) -> jnp.ndarray:
        """Raw latents `[B, t, h, w, c]` to video, frame by frame."""
        batch_size, seq_len, height, width, channels = z.shape
        decoded = self.decode_batch(params, z.reshape(-1, height, width, channels))
        return decoded.reshape(batch_size, seq_len, *decoded.shape[1:])

    def encode(self, params, x: jnp.ndarray,
               key: jax.Array | None = None) -> jnp.ndarray:
        """Images `[B, H, W, C]` or video `[B, T, H, W, C]` to normalized
        latents."""
        latent = self.encode_video(params, x, key) if x.ndim == 5 else self.encode_batch(params, x, key=key)
        shift, scale = (jnp.asarray(value, latent.dtype) for value in (self.latent_shift, self.latent_scale))
        return (latent - shift) * scale

    def decode(self, params, z: jnp.ndarray) -> jnp.ndarray:
        """Normalized latents `[B, h, w, c]` or `[B, t, h, w, c]` back to
        images or video."""
        shift, scale = (jnp.asarray(value, z.dtype) for value in (self.latent_shift, self.latent_scale))
        z = z / scale + shift
        return self.decode_video(params, z) if z.ndim == 5 else self.decode_batch(params, z)

    def __call__(self, params, x: jnp.ndarray,
                 key: jax.Array | None = None) -> jnp.ndarray:
        """Encode then decode."""
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
