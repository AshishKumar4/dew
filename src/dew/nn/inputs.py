"""Numeric model inputs shared by preprocessing, training and generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import itertools
import math
from dataclasses import dataclass, replace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from typing_extensions import TypeVar

from dew.nn.sharding import DATA_AXIS, EXPERT_AXIS, FSDP_AXIS, TENSOR_AXIS

ArrayT = TypeVar("ArrayT", bound=jax.Array | np.ndarray, default=jax.Array, covariant=True)
TreeT = TypeVar("TreeT")

BATCH_AXES = (DATA_AXIS, EXPERT_AXIS, FSDP_AXIS, TENSOR_AXIS)
"""The mesh axes a request's rows split over: every axis but sequence and stage."""


def pad_token_rows(rows: Sequence[Sequence[int]] | np.ndarray, *, pad_id: int = 0,
                   padding_side: Literal["left", "right"] = "left",
                   fields: Mapping[str, Sequence[Sequence[int]] | np.ndarray] | None = None
                   ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Pad ragged token rows and their scalar token fields on the host.

    Filler IDs carry no content; attention_mask alone marks real slots. Rows
    that all fill the width take no filler, so no attention_mask comes back.
    Absent validity is how a host says every slot is real. An all-true mask
    would say the same thing in a form the model cannot read the contents
    of, and it would cost it the fused attention kernel.
    Tokenizer state is not involved in this numeric layout operation.
    """
    if padding_side not in ("left", "right"):
        raise ValueError("padding_side must be left or right")
    limits = np.iinfo(np.int32)
    if type(pad_id) is not int or not 0 <= pad_id <= limits.max:
        raise ValueError("pad_id must be a nonnegative int32 token id")
    arrays = [np.asarray(row) for row in rows]
    if not arrays or any(row.ndim != 1 or row.size == 0 or not np.issubdtype(row.dtype, np.integer) for row in arrays):
        raise ValueError("each prompt must contain a nonempty integer token row")
    if any(np.any((row < 0) | (row > limits.max)) for row in arrays):
        raise ValueError("token IDs must be nonnegative int32 values")
    width = max(row.size for row in arrays)
    tokens = np.full((len(arrays), width), pad_id, np.int32)
    valid = np.zeros(tokens.shape, bool)
    slots = [slice(width - row.size, width) if padding_side == "left" else slice(0, row.size) for row in arrays]
    for index, (row, slot) in enumerate(zip(arrays, slots)):
        tokens[index, slot] = row
        valid[index, slot] = True
    padded = {"attention_mask": valid}
    for name, values in (fields or {}).items():
        aligned = [np.asarray(row) for row in values]
        if len(aligned) != len(arrays) or any(value.shape != row.shape for value, row in zip(aligned, arrays)):
            raise ValueError(f"token field {name!r} must align with the token rows")
        if name == "attention_mask" and any(np.any((row != 0) & (row != 1)) for row in aligned):
            raise ValueError("attention_mask must contain only zero or one")
        dtype = bool if name == "attention_mask" else np.result_type(*(row.dtype for row in aligned))
        value = np.zeros(tokens.shape, dtype=dtype)
        for index, (row, slot) in enumerate(zip(aligned, slots)):
            value[index, slot] = row
        padded[name] = value
    if padded["attention_mask"].all():
        del padded["attention_mask"]
    return tokens, padded


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

def generation_signature(inputs: object, controls: object) -> np.ndarray:
    """Digest execution shapes and stable host controls without reading payloads.

    Algorithms agree this fixed-width signature after local validation and
    before distributed execution. Controls must have a deterministic repr,
    such as frozen configuration values, tuples and dictionaries of scalars.
    Tensor contents are excluded: this is not a prefix-cache identity.
    """
    schema = (str(jax.tree.structure(inputs)),
              [(leaf.shape, str(leaf.dtype)) for leaf in jax.tree.leaves(inputs)], controls)
    return np.frombuffer(hashlib.sha256(repr(schema).encode()).digest(), np.uint8)


VALIDITY_FIELD = "attention_mask"
"""The token field that marks real slots. Absent means every slot is real."""


def validity_sites(tree: object) -> list[ModelInputs]:
    """The tree's `ModelInputs` nodes, in flatten order.

    A node is a site whether or not it carries validity, so every process
    finds the same sites in the same order and a per-site vector has the same
    length everywhere.
    """
    leaves = jax.tree.leaves(tree, is_leaf=lambda node: isinstance(node, ModelInputs))
    return [leaf for leaf in leaves if isinstance(leaf, ModelInputs)]


def _validity_agnostic(tree: TreeT) -> TreeT:
    """`tree` with every validity field dropped."""
    return jax.tree.map(
        lambda node: (replace(node, token_fields={
            name: value for name, value in node.token_fields.items()
            if name != VALIDITY_FIELD}) if isinstance(node, ModelInputs) else node),
        tree, is_leaf=lambda node: isinstance(node, ModelInputs))


def assembly_signature(tree: object, controls: object = ()) -> np.ndarray:
    """`generation_signature` of the tree with validity left out.

    Whether a process's own rows needed padding is rank-local, so a digest
    that counted validity would refuse a pool that agrees on everything else.
    This one still carries every other field, shape, dtype and control, so a
    real schema disagreement is still refused, and it is fixed-width and
    computed from local structure alone, which is what lets it be the first
    thing a pool agrees on.
    """
    return generation_signature(_validity_agnostic(tree), controls)


def filled_validity(tree: TreeT, wanted: bool | Sequence[bool] = True) -> TreeT:
    """All-true validity at the sites that want it and do not carry it.

    `wanted` is one flag per site of `validity_sites`, or one flag for all of
    them. Nothing is read from a resident array and no site that carries
    validity is touched, so this is a schema operation and not a mask.
    """
    sites = itertools.count()

    def fill(node):
        if not isinstance(node, ModelInputs):
            return node
        index = next(sites)
        needed = wanted if isinstance(wanted, bool) else bool(wanted[index])
        if not needed or VALIDITY_FIELD in node.token_fields:
            return node
        shape = np.shape(node.tokens)
        ones = (np.ones(shape, bool) if isinstance(node.tokens, np.ndarray)
                else jnp.ones(shape, bool))
        return replace(node, token_fields={**node.token_fields, VALIDITY_FIELD: ones})

    return jax.tree.map(fill, tree, is_leaf=lambda node: isinstance(node, ModelInputs))


def agreed_validity(tree: TreeT, processes: int, *, controls: object = (),
                    phase: str = "input") -> TreeT:
    """One validity schema for the whole pool, agreed before arrays are built.

    A host that padded nothing carries no validity, which is what keeps
    attention on its fused kernel. Padding is a property of a process's own
    rows, so one process can omit the field while another carries it, and the
    same step would then receive two different pytrees. Where any process
    carries validity for a site, every process materializes it; where none
    does, the omission stays.

    The collectives are fixed-shape and their number does not depend on what
    this process holds: the validity-agnostic signature is agreed first, and
    the site count that sizes the presence vector comes from that agreed
    schema. Both run on the calling thread, so call this where the caller's
    other collectives are issued, in the same order on every process.
    """
    if processes <= 1:
        return tree
    from jax.experimental import multihost_utils

    multihost_utils.assert_equal(
        assembly_signature(tree, controls),
        f"{phase} schemas, shapes and controls must agree across processes")
    count = len(validity_sites(tree))
    if not count:
        return tree
    present = np.asarray([VALIDITY_FIELD in site.token_fields
                          for site in validity_sites(tree)], np.int32)
    gathered = np.asarray(multihost_utils.process_allgather(present)).reshape(-1, count)
    return filled_validity(tree, [bool(flag) for flag in gathered.max(axis=0)])


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
