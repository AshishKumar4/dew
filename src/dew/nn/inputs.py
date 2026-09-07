"""Numeric model inputs shared by preprocessing, training and generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from dew.nn.sharding import DATA_AXIS, EXPERT_AXIS, FSDP_AXIS, TENSOR_AXIS

BATCH_AXES = (DATA_AXIS, EXPERT_AXIS, FSDP_AXIS, TENSOR_AXIS)
"""The mesh axes a request's rows split over: every axis but sequence and stage."""


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

def generation_signature(inputs: ModelInputs, controls: object) -> np.ndarray:
    """Digest execution shapes and stable host controls without reading payloads.

    Algorithms agree this fixed-width signature after local validation and
    before distributed execution. Controls must have a deterministic repr,
    such as frozen configuration values, tuples and dictionaries of scalars.
    Tensor contents are excluded: this is not a prefix-cache identity.
    """
    schema = (str(jax.tree.structure(inputs)),
              [(leaf.shape, str(leaf.dtype)) for leaf in jax.tree.leaves(inputs)], controls)
    return np.frombuffer(hashlib.sha256(repr(schema).encode()).digest(), np.uint8)


def local_rows(leaf) -> np.ndarray:
    """This process's rows of a batch leaf, in global order.

    A global array hands back the rows this process's devices hold, with a
    sequence-split second dimension reassembled. Anything a process can read
    whole is read whole.
    """
    if not isinstance(leaf, jax.Array) or leaf.is_fully_addressable:
        return np.asarray(leaf)
    if leaf.ndim == 0:
        return np.asarray(leaf.addressable_shards[0].data)
    pieces: dict[tuple[int, ...], np.ndarray] = {}
    for shard in leaf.addressable_shards:
        pieces.setdefault(tuple(part.start or 0 for part in shard.index), np.asarray(shard.data))
    blocks = []
    for start in sorted({key[0] for key in pieces}):
        columns = sorted(key for key in pieces if key[0] == start)
        blocks.append(np.concatenate([pieces[key] for key in columns], axis=1)
                      if len(columns) > 1 else pieces[columns[0]])
    return np.concatenate(blocks, axis=0)


def request_key(key: jax.Array | None, seed: int | None) -> jax.Array:
    """One typed PRNG key from either a key or an integer seed, never both."""
    if (key is None) == (seed is None):
        raise ValueError("pass exactly one of key and seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        return jax.random.key(int(seed))
    assert key is not None
    typed = jax.random.wrap_key_data(jax.random.key_data(key), impl=jax.random.key_impl(key))
    if typed.shape != ():
        raise ValueError("key must be a single JAX PRNG key")
    return typed


def mesh_of(tree) -> jax.sharding.Mesh | None:
    """The mesh the tree's leaves sit on, or None for single-device arrays."""
    for leaf in jax.tree.leaves(tree):
        mesh = getattr(getattr(leaf, "sharding", None), "mesh", None)
        if mesh is not None and not mesh.empty:
            return mesh
    return None


@dataclass(frozen=True)
class RowPlan:
    """Where one request's rows sit while a task runs.

    Without a mesh every array stays on the default device. On a mesh, rows
    split over its batch axes; each process contributes ``count`` rows, its
    ``rows`` real ones followed by repeats that pad to the device count, so
    every device holds the same shape and every collective lines up. Results
    keep that sharding; ``host`` reads a process's real rows back.
    """

    mesh: jax.sharding.Mesh | None
    rows: int
    count: int
    process: int
    processes: int

    @classmethod
    def over(cls, mesh: jax.sharding.Mesh | None, rows: int) -> RowPlan:
        if mesh is None:
            return cls(None, rows, rows, 0, 1)
        processes, process = jax.process_count(), jax.process_index()
        batch_devices = math.prod(mesh.shape[axis] for axis in BATCH_AXES)
        if batch_devices % processes:
            raise ValueError("the batch axes of the mesh do not split evenly over the processes")
        local_devices = batch_devices // processes
        return cls(mesh, rows, -(-rows // local_devices) * local_devices, process, processes)

    @property
    def sharding(self) -> jax.sharding.NamedSharding | None:
        if self.mesh is None:
            return None
        return jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(BATCH_AXES))

    @property
    def global_rows(self) -> int:
        """Rows of a placed array across every process."""
        return self.count * self.processes

    @property
    def padding(self) -> np.ndarray:
        """Which of the ``count`` placed rows are repeats, not real rows."""
        return np.arange(self.count) >= self.rows

    def pad(self, tree):
        """Rows repeated up to ``count`` on the host; unchanged when none are needed."""
        if self.count == self.rows:
            return tree
        indices = np.arange(self.count) % self.rows
        return jax.tree.map(lambda leaf: np.asarray(leaf)[indices], tree)

    def place(self, tree):
        """Padded host rows as device arrays, row-sharded on a mesh."""
        sharding = self.sharding
        if sharding is None:
            return jax.tree.map(jnp.asarray, tree)

        def put(leaf):
            rows = np.asarray(leaf)
            if rows.ndim == 0 or rows.shape[0] != self.count:
                raise ValueError(f"a placed leaf needs {self.count} rows on axis zero, got {rows.shape}")
            return jax.make_array_from_process_local_data(sharding, rows)

        return jax.tree.map(put, tree)

    def keys(self, key: jax.Array) -> jax.Array:
        """One key per placed row, folded by global row index, so a pool draws
        what a single process draws for the same rows."""
        row_keys = jax.vmap(lambda row: jax.random.fold_in(key, row))(
            jnp.arange(self.count) + self.process * self.rows)
        sharding = self.sharding
        if sharding is None:
            return row_keys
        data = jax.make_array_from_process_local_data(sharding, np.asarray(jax.random.key_data(row_keys)))
        return jax.random.wrap_key_data(data, impl=jax.random.key_impl(key))

    def host(self, leaf) -> np.ndarray:
        """This process's real rows of a result leaf as a host array."""
        return local_rows(leaf)[:self.rows]
