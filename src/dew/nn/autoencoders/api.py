from __future__ import annotations

import copy
import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.objectives.base import Variables


@dataclasses.dataclass(frozen=True, eq=False)
class LatentNormalization:
    """A latent normalization, one description: the source values an autoencoder's record keeps
    (`record`, as given) and the shift and factor derived from them once. Latents are normalized as
    `(z - shift) * factor` and restored as `z / factor + shift`; one that `divides` normalizes as
    `(z - shift) / factor` and restores as `z * factor + shift`, as its source rounds."""
    record: Mapping[str, Any]
    shift: Any
    factor: Any
    divides: bool = False

    @property
    def scale(self):
        """The factor a normalized latent is the shifted one times."""
        return np.float32(1.0) / self.factor if self.divides else self.factor

    def normalized(self, latent: jnp.ndarray) -> jnp.ndarray:
        shift, factor = (jnp.asarray(value, latent.dtype) for value in (self.shift, self.factor))
        return (latent - shift) / factor if self.divides else (latent - shift) * factor

    def restored(self, z: jnp.ndarray) -> jnp.ndarray:
        shift, factor = (jnp.asarray(value, z.dtype) for value in (self.shift, self.factor))
        return z * factor + shift if self.divides else z / factor + shift


IDENTITY = LatentNormalization({}, 0.0, 1.0)
"""The normalization that leaves latents as they are."""


class AutoEncoder(ABC):
    """Encodes images or video to the latents a latent diffusion model trains on, and decodes them back.

    A subclass encodes and decodes one batch of frames, `[B, H, W, C]` to
    `[B, h, w, c]` and back. `encode` and `decode` here take video
    `[B, T, H, W, C]` frame by frame, unless the subclass encodes time itself
    (`encode_video`, `decode_video`, `latent_shape`), and they apply the
    latent normalization (`normalization`, `LatentNormalization`). On the way
    out, latents are normalized as (z - latent_shift) * latent_scale, and on
    the way in the normalization is inverted, the SD3 convention.
    `latent_shift` and `latent_scale` are each one number or one per latent
    channel, read from `normalization`. The default is the identity; an
    autoencoder normalized by the dataset's own latent mean and 1/std, so the
    diffusion model sees roughly unit-variance, zero-mean inputs, is
    `with_normalization`'s.

    The weights are an argument, as a `ConditionEncoder`'s are. `params` holds
    what a run loaded, and every call takes the tree to use, so the layout
    places the weights and the checkpoint stores them.
    """

    normalization: LatentNormalization = IDENTITY
    params: Variables
    model: nn.Module

    record_name: ClassVar[str | None] = None
    """The name a built-in subclass's record goes by (`to_json`); None for
    an autoencoder no record declares."""
    record_module: ClassVar[type[Any]]
    """The native module class a built-in subclass's `model` is, which its
    record's 'model' rebuilds (each subclass's own `model` type)."""

    def record_arguments(self) -> dict:
        """The built-in subclass's constructor arguments besides `model` and
        `params`, as it was given them: its compute dtype and latent
        normalization."""
        raise TypeError(f"{type(self).__name__} needs an explicit autoencoder record declaration")

    @property
    def latent_shift(self):
        return self.normalization.shift

    @property
    def latent_scale(self):
        return self.normalization.scale

    def with_normalization(self, normalization: LatentNormalization) -> AutoEncoder:
        """This autoencoder normalizing its latents by `normalization` instead: a copy, this one
        unchanged."""
        replaced = copy.copy(self)
        replaced.normalization = normalization
        return replaced

    def to_json(self) -> dict:
        """Return the record of this built-in autoencoder's constructor, without its parameters:
        `record_name`, its module's fields and `record_arguments`.

        One no record declares (`record_name` None) raises `TypeError`.
        """
        if self.record_name is None:
            raise TypeError(f"{type(self).__name__} needs an explicit autoencoder record declaration")
        from dew.registry import record_fields
        arguments = {name: np.asarray(value).tolist() if isinstance(value, np.ndarray | jax.Array) else value
                     for name, value in self.record_arguments().items()}
        return {'name': self.record_name,
                'fields': {'model': record_fields(self.model, self.record_module), **arguments}}

    @classmethod
    def from_json(cls, record, *, params: Variables) -> AutoEncoder:
        """Rebuild a built-in autoencoder from its `record`, around the checkpoint's own `params`,
        reading no source. A name no built-in declares raises `ValueError`.
        """
        from dew.registry import from_record

        from .dc_ae import DCAutoencoder
        from .flux2 import Flux2Autoencoder
        from .qwen_image import QwenImageAutoencoder
        from .rae import RAEAutoencoder
        from .sd_vae import StableDiffusionVAE
        from .wan import WanAutoencoder

        built = {wrapper.record_name: wrapper for wrapper in (
            StableDiffusionVAE, DCAutoencoder, WanAutoencoder, RAEAutoencoder, Flux2Autoencoder,
            QwenImageAutoencoder)}
        if record['name'] not in built:
            raise ValueError(f"{record['name']!r} is not a built-in autoencoder declaration")
        wrapper = built[record['name']]
        fields = dict(record['fields'])
        model = from_record(wrapper.record_module, fields.pop('model'))
        return wrapper(model=model, params=params, **fields)

    @abstractmethod
    def encode_batch(self, params, x: jnp.ndarray, key: jax.Array | None = None,
                     noise: jnp.ndarray | None = None) -> jnp.ndarray:
        """Encode frames `[B, H, W, C]` to raw latents `[B, h, w, c]`.

        `key` draws a stochastic encoder's sample, and None takes its mean.
        `noise`, a standard normal draw of the latents' shape, is the sample's
        in place of one from `key` (`posterior_latent`); a deterministic
        encoder reads neither.
        """

    @abstractmethod
    def decode_batch(self, params, z: jnp.ndarray) -> jnp.ndarray:
        """Decode raw latents `[B, h, w, c]` to frames `[B, H, W, C]`."""

    @property
    @abstractmethod
    def downscale_factor(self) -> int:
        """H / h, the spatial factor between a frame and its latent."""

    @property
    def stochastic(self) -> bool:
        """Whether encoding samples its latent, so a training step draws each example's `noise`
        (`encode_batch`). A deterministic encoder (DC-AE, RAE) is False and refuses noise, raising
        TypeError naming itself."""
        return True

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

    def encode_video(self, params, x: jnp.ndarray, key: jax.Array | None = None,
                     noise: jnp.ndarray | None = None) -> jnp.ndarray:
        """Encode video `[B, T, H, W, C]` to raw latents, frame by frame."""
        batch_size, seq_len, height, width, channels = x.shape
        frames = None if noise is None else noise.reshape(-1, *noise.shape[2:])
        latent = self.encode_batch(params, x.reshape(-1, height, width, channels), key=key, noise=frames)
        return latent.reshape(batch_size, seq_len, *latent.shape[1:])

    def decode_video(self, params, z: jnp.ndarray) -> jnp.ndarray:
        """Decode raw latents `[B, t, h, w, c]` to video, frame by frame."""
        batch_size, seq_len, height, width, channels = z.shape
        decoded = self.decode_batch(params, z.reshape(-1, height, width, channels))
        return decoded.reshape(batch_size, seq_len, *decoded.shape[1:])

    def encode(self, params, x: jnp.ndarray, key: jax.Array | None = None,
               noise: jnp.ndarray | None = None) -> jnp.ndarray:
        """Encode images `[B, H, W, C]` or video `[B, T, H, W, C]` to normalized latents, a stochastic
        encoder's sample drawn from `key` or read off `noise` (`encode_batch`)."""
        latent = (self.encode_video(params, x, key, noise) if x.ndim == 5
                  else self.encode_batch(params, x, key=key, noise=noise))
        return self.normalization.normalized(latent)

    def decode(self, params, z: jnp.ndarray) -> jnp.ndarray:
        """Decode normalized latents `[B, h, w, c]` or `[B, t, h, w, c]` back to images or video."""
        z = self.normalization.restored(z)
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
        self.encode_single_frame = jax.jit(lambda params, image, key=None, noise=None: model.apply(
            {"params": params}, image, key, noise, method=model.encode))
        self.decode_single_frame = jax.jit(lambda params, latent: model.apply(
            {"params": params}, latent, method=model.decode))

    def encode_batch(self, params, x, key=None, noise=None):
        return self.encode_single_frame(params, x, key, noise)

    def decode_batch(self, params, z):
        return self.decode_single_frame(params, z)

    @property
    def downscale_factor(self) -> int:
        """The module's own, where encoding adds nothing to it."""
        return self.model.downscale_factor

    @property
    def latent_channels(self) -> int:
        """The module's own, where encoding adds nothing to it."""
        return self.model.latent_channels
