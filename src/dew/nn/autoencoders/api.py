from __future__ import annotations

import copy
import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct

from dew.batch import Variables

type RecordValue = float | np.ndarray | None
"""A value a latent normalization's record keeps: a number, one per channel or position, or absent."""


def check_class(config: Mapping[str, object], names: tuple[str, ...], where, *,
                nameless: bool = False) -> None:
    """Refuse a published autoencoder's `config` whose `_class_name` is none of `names`, naming the class it
    holds (a Wan VAE's given to the KL loader); one that names none is taken as `names`' own where
    `nameless` (an older AutoencoderKL's config)."""
    named = config.get("_class_name")
    if named not in names and not (named is None and nameless):
        raise ValueError(f"{where} holds a {named}, not an {names[0]}")


def published_source(name_or_dir, revision, subfolder: str, names: tuple[str, ...], *, weights: bool,
                     nameless: bool = False):
    """A published autoencoder's directory and config (`dew.interop.components.component_source`), its
    config checked first (`check_class`) and its weights fetched only after, so a foreign checkpoint is
    refused by name before any of its weights is read."""
    from dew.interop.components import component_source

    directory, config = component_source(name_or_dir, revision, subfolder, weights=False)
    check_class(config, names, directory, nameless=nameless)
    if weights:
        directory, config = component_source(name_or_dir, revision, subfolder, weights=True)
    return directory, config


@dataclasses.dataclass(frozen=True, eq=False)
class LatentNormalization:
    """A latent normalization, one description: the source values an autoencoder's record keeps
    (`record`, as given) beside the shift and factor its autoencoder derived from them, written
    together once and read-only after. Latents are normalized as `(z - shift) * factor` and restored
    as `z / factor + shift`; one that `divides` normalizes as `(z - shift) / factor` and restores as
    `z * factor + shift`, as its source rounds."""
    record: Mapping[str, RecordValue]
    shift: float | np.ndarray
    factor: float | np.ndarray
    divides: bool = False

    def __post_init__(self):
        # Read-only copies, so no reader writes the normalization through what it reads.
        record = {name: _frozen(value) for name, value in self.record.items()}
        object.__setattr__(self, 'record', _Record(record))
        object.__setattr__(self, 'shift', _frozen(self.shift))
        object.__setattr__(self, 'factor', _frozen(self.factor))

    @property
    def scale(self) -> float | np.ndarray:
        """The factor a normalized latent is the shifted one times."""
        return _frozen(np.asarray(np.float32(1.0) / self.factor)) if self.divides else self.factor

    def affine(self, dtype) -> LatentAffine:
        """The map this normalization applies, its shift and factor in `dtype`."""
        return LatentAffine(jnp.asarray(self.shift, dtype), jnp.asarray(self.factor, dtype), self.divides)


@struct.dataclass
class LatentAffine:
    """The affine map an autoencoder's latents are normalized by, as the arrays a trace holds
    (`AutoEncoder.latent_affine`): `(z - shift) * factor` and back as `z / factor + shift`, or, one that
    `divides`, `(z - shift) / factor` and back as `z * factor + shift`. A configured normalization gives
    its host values (`LatentNormalization.affine`); a tuned run, the running statistics each call's
    variables hold. `divides` is static, so the map crosses a jit boundary as its two arrays."""
    shift: jax.Array
    factor: jax.Array
    divides: bool = struct.field(pytree_node=False, default=False)

    def normalized(self, latent: jnp.ndarray) -> jnp.ndarray:
        shifted = latent - self.shift
        return shifted / self.factor if self.divides else shifted * self.factor

    def restored(self, z: jnp.ndarray) -> jnp.ndarray:
        return z * self.factor + self.shift if self.divides else z / self.factor + self.shift


class _Record(Mapping[str, RecordValue]):
    """A record's values, read-only (a mapping with no way to set one), and picklable as any object is."""

    def __init__(self, values: Mapping[str, RecordValue]):
        self._values = dict(values)

    def __getitem__(self, name: str) -> RecordValue:
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def _frozen[T](value: T) -> T:
    """`value`, or a read-only copy where it is an array."""
    if not isinstance(value, np.ndarray):
        return value
    frozen = np.array(value, copy=True)
    frozen.setflags(write=False)
    return frozen


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
    channel, read from `normalization`; one that divides by its factor (Qwen-
    Image's) rounds as a division does. The default is the identity; an
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

    def record_arguments(self) -> dict:
        """The subclass's constructor arguments besides `model` and `params`, as it was given them (a
        built-in's compute dtype and latent normalization), which its record rebuilds it with
        (`dew.registry.to_record`). A subclass a run records states them."""
        raise TypeError(f"{type(self).__name__} states no constructor arguments for its record "
                        f"(`AutoEncoder.record_arguments`)")

    @property
    def latent_shift(self) -> float | np.ndarray:
        return self.normalization.shift

    @property
    def latent_scale(self) -> float | np.ndarray:
        return self.normalization.scale

    def with_normalization(self, normalization: LatentNormalization) -> Self:
        """This autoencoder normalizing its latents by `normalization` instead: a copy, this one
        unchanged."""
        replaced = copy.copy(self)
        replaced.normalization = normalization
        return replaced

    def recorded_fields(self) -> dict:
        """The record a run keeps of this autoencoder, without its parameters (`dew.registry.Recorded`):
        its module, whose record names its class, and `record_arguments`, an array as a list; `bound`
        passes them back to its constructor with `params`."""
        arguments = {name: np.asarray(value).tolist() if isinstance(value, np.ndarray | jax.Array) else value
                     for name, value in self.record_arguments().items()}
        return {'model': self.model, **arguments}

    @staticmethod
    def bound(record, *, params: Variables) -> AutoEncoder:
        """Rebuild the autoencoder a run recorded (`dew.registry.to_record`), around the checkpoint's own
        `params`, reading no source: its class found as every component's is (`dew.registry.component`),
        of Dew's or of a package the run trusts, and its fields read against its constructor."""
        from dew.registry import component, read_arguments

        autoencoder, fields = component(AutoEncoder, record)
        # The module's record names its class, which an inherited constructor's `model: M` cannot.
        arguments = read_arguments(autoencoder, fields, declared={'model': nn.Module})
        return _built(autoencoder, arguments, params)

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

    def latent_affine(self, params, dtype) -> LatentAffine:
        """The map `encode` normalizes latents by and `decode` restores them by, in `dtype`, for the
        weights `params`: this autoencoder's configured normalization (`normalization`), whatever the
        weights. A subclass whose normalization is data the variables hold reads it from `params`."""
        del params
        return self.normalization.affine(dtype)

    def encode(self, params, x: jnp.ndarray, key: jax.Array | None = None,
               noise: jnp.ndarray | None = None) -> jnp.ndarray:
        """Encode images `[B, H, W, C]` or video `[B, T, H, W, C]` to normalized latents, a stochastic
        encoder's sample drawn from `key` or read off `noise` (`encode_batch`), by `latent_affine`."""
        latent = (self.encode_video(params, x, key, noise) if x.ndim == 5
                  else self.encode_batch(params, x, key=key, noise=noise))
        return self.latent_affine(params, latent.dtype).normalized(latent)

    def decode(self, params, z: jnp.ndarray) -> jnp.ndarray:
        """Decode normalized latents `[B, h, w, c]` or `[B, t, h, w, c]` back to images or video, by
        `latent_affine`."""
        z = self.latent_affine(params, z.dtype).restored(z)
        return self.decode_video(params, z) if z.ndim == 5 else self.decode_batch(params, z)

    def __call__(self, params, x: jnp.ndarray,
                 key: jax.Array | None = None) -> jnp.ndarray:
        """Encode `x`, then decode it."""
        return self.decode(params, self.encode(params, x, key=key))



def _built(autoencoder: Callable[..., AutoEncoder], arguments: Mapping[str, object],
           params: Variables) -> AutoEncoder:
    """The subclass a record names (`AutoEncoder.bound`), constructed from its read `arguments` around
    `params`: called as the callable it is, since its abstract base declares no constructor."""
    return autoencoder(**arguments, params=params)


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
