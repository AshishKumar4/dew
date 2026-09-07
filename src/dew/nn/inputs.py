"""Numeric model inputs shared by preprocessing, training and generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


@struct.dataclass
class ModelInputs:
    """Token rows with sequence-aligned fields and row-aligned conditioning.

    Every leaf has batch axis zero. ``token_fields`` also has sequence axis
    one, with the same length as ``tokens``. ``conditioning`` holds media
    payloads and their valid lengths; it contains no text-slot coordinates.
    A token field such as ``image_indices`` identifies the media feature read
    at that text slot, with -1 for text. Slicing a prompt therefore preserves
    feature identity without rewriting indices hidden in media payloads.

    The processor validates field names and values for its model on the host.
    These shape-preserving operations also work inside JIT.
    """

    tokens: jax.Array
    token_fields: Mapping[str, jax.Array] = struct.field(default_factory=dict)
    conditioning: Mapping[str, jax.Array] = struct.field(default_factory=dict)

    @classmethod
    def from_value(cls, value: ModelInputs | jax.typing.ArrayLike | Sequence[Sequence[int]]) -> ModelInputs:
        """Normalize host input without lossy ID coercion or moving resident arrays.

        Distributed algorithms call this inside their agreed validation phase;
        this method itself performs no collectives or model execution.
        """
        if isinstance(value, cls):
            result = value
        elif isinstance(value, jax.Array):
            result = cls(value)
        else:
            array = np.asarray(value)
            if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
                raise ValueError("tokens must be an integer [B, S] array")
            bounds = np.iinfo(np.int32)
            if np.any(array < bounds.min) or np.any(array > bounds.max):
                raise ValueError("token IDs must be representable as int32")
            result = cls(jnp.asarray(array, jnp.int32))
        result.validate()
        return result


    def validate(self) -> None:
        """Check the numeric layout on the host before dispatching a model."""
        if self.tokens.ndim != 2 or not jnp.issubdtype(self.tokens.dtype, jnp.integer):
            raise ValueError("tokens must be an integer [B, S] array")
        reserved = {"tokens", "conditioning", "train", "decode", "rngs", "method",
                    "mutable", "capture_intermediates", "attention_pairwise_mask",
                    "attention_key_positions"}
        if reserved.intersection(self.token_fields):
            raise ValueError(f"token_fields cannot contain {sorted(reserved.intersection(self.token_fields))}")
        for name, value in self.token_fields.items():
            if value.ndim < 2 or value.shape[:2] != self.tokens.shape:
                raise ValueError(f"token field {name!r} must start with {self.tokens.shape}, got {value.shape}")
        for name, value in self.conditioning.items():
            if value.ndim < 1 or value.shape[0] != self.tokens.shape[0]:
                raise ValueError(f"conditioning {name!r} must have batch size {self.tokens.shape[0]}")

    def take_rows(self, indices: jax.Array) -> ModelInputs:
        """Select the same rows from tokens, sequence fields and media."""
        return replace(self,
            tokens=self.tokens[indices],
            token_fields={name: value[indices] for name, value in self.token_fields.items()},
            conditioning={name: value[indices] for name, value in self.conditioning.items()})

    def slice_tokens(self, start: int | None = None, stop: int | None = None) -> ModelInputs:
        """Slice token slots; media features retain their original indices."""
        selection = slice(start, stop)
        return replace(self,
            tokens=self.tokens[:, selection],
            token_fields={name: value[:, selection] for name, value in self.token_fields.items()})

    def align_left(self, left_padding: jax.Array) -> ModelInputs:
        """Move each row's real token prefix left and its padding to the end.

        Logical positions are sequence fields, so their values travel with the
        tokens. Their values are not recomputed from the padded slot number.
        """
        if left_padding.shape != (self.tokens.shape[0],):
            raise ValueError("left_padding must have one count per token row")
        slots = (jnp.arange(self.tokens.shape[1])[None, :] + left_padding[:, None]) % self.tokens.shape[1]

        def shift(value: jax.Array) -> jax.Array:
            indices = slots.reshape(slots.shape + (1,) * (value.ndim - 2))
            return jnp.take_along_axis(value, indices, axis=1)

        return replace(self, tokens=shift(self.tokens),
                            token_fields={name: shift(value) for name, value in self.token_fields.items()})

    def kwargs(self) -> dict[str, object]:
        """Keywords for a native model call, excluding the token argument."""
        result: dict[str, object] = dict(self.token_fields)
        if self.conditioning:
            result["conditioning"] = self.conditioning
        return result


@struct.dataclass
class AttentionMetadata:
    """Per-token attention data independent of physical cache-slot addresses."""

    valid: jax.Array | None = None
    image_groups: jax.Array | None = None
    rotary_positions: jax.Array | None = None
    pairwise_mask: jax.Array | None = None
    key_positions: jax.Array | None = None
