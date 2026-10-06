"""Numeric model inputs shared by preprocessing, training and generation."""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, overload

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.core import Tracer

from dew.coordination import agreed
from dew.nn.sharding import DATA_AXIS, EXPERT_AXIS, FSDP_AXIS, TENSOR_AXIS

if TYPE_CHECKING:
    from PIL.Image import Image

# What a caller hands a host processor as one media argument: pixels or audio
# samples in an array, a PIL image, or one entry per row of either. The source's
# own processor is the only code that reads it; dew moves it.
type Media = np.ndarray | Image | Sequence[Media]

# A request as the pytree utilities walk it: the numeric inputs themselves,
# the arrays a caller assembled them from, and the tuples and records those
# travel in. A digest reads the structure and each leaf's shape and dtype, so
# a record's values are leaves to it and are not narrowed here.
type InputTree = (ModelInputs | jax.Array | np.ndarray | None
                  | Sequence[InputTree] | Mapping[str, object])

# A keyword a model call takes past its tokens: a token field or position
# array, the conditioning mapping (`ModelInputs.kwargs`), a flag, or None for
# an absent input.
type ModelKwarg = jax.Array | Mapping[str, jax.Array] | bool | None

# A token field's `[B, W, ...]` response values from its `[B, S, ...]` prompt
# values, the prompt's `[B, S]` real slots and the width (`ModelInputs.extended`).
type FieldExtension = Callable[[jax.Array, jax.Array, int], jax.Array]

PredictionPhase = Literal["ordinary", "extend", "draft"]
BATCH_AXES = (DATA_AXIS, EXPERT_AXIS, FSDP_AXIS, TENSOR_AXIS)
"""The mesh axes a request's rows split over: every axis but sequence and stage."""


def pad_token_rows(rows: Sequence[Sequence[int]] | np.ndarray, *, pad_id: int = 0,
                   padding_side: Literal["left", "right"] = "left",
                   fields: Mapping[str, Sequence[Sequence[int]] | np.ndarray] | None = None
                   ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Pad ragged token rows and their scalar token fields on the host.

    Filler IDs carry no content; attention_mask alone marks real slots. Rows
    that all fill the width come back without one: absent validity says every
    slot is real, which an all-true mask would hide from the model and cost it
    the fused attention kernel.
    """
    if padding_side not in ("left", "right"):
        raise ValueError("padding_side must be left or right")
    limits = np.iinfo(np.int32)
    if type(pad_id) is not int or not 0 <= pad_id <= limits.max:
        raise ValueError("pad_id must be a nonnegative int32 token id")
    arrays = [np.asarray(row) for row in rows]
    if not arrays or any(
        row.ndim != 1 or row.size == 0 or not np.issubdtype(row.dtype, np.integer) for row in arrays
    ):
        raise ValueError("each prompt must contain a nonempty integer token row")
    if any(np.any((row < 0) | (row > limits.max)) for row in arrays):
        raise ValueError("token IDs must be nonnegative int32 values")
    width = max(row.size for row in arrays)
    tokens = np.full((len(arrays), width), pad_id, np.int32)
    valid = np.zeros(tokens.shape, bool)
    slots = [
        slice(width - row.size, width) if padding_side == "left" else slice(0, row.size) for row in arrays
    ]
    for index, (row, slot) in enumerate(zip(arrays, slots, strict=True)):
        tokens[index, slot] = row
        valid[index, slot] = True
    padded = {"attention_mask": valid}
    for name, values in (fields or {}).items():
        aligned = [np.asarray(row) for row in values]
        if len(aligned) != len(arrays) or any(
            value.shape != row.shape for value, row in zip(aligned, arrays, strict=True)
        ):
            raise ValueError(f"token field {name!r} must align with the token rows")
        if name == "attention_mask" and any(np.any((row != 0) & (row != 1)) for row in aligned):
            raise ValueError("attention_mask must contain only zero or one")
        dtype = bool if name == "attention_mask" else np.result_type(*(row.dtype for row in aligned))
        value = np.zeros(tokens.shape, dtype=dtype)
        for index, (row, slot) in enumerate(zip(aligned, slots, strict=True)):
            value[index, slot] = row
        padded[name] = value
    if padded["attention_mask"].all():
        del padded["attention_mask"]
    return tokens, padded


def host_token_rows(array: np.ndarray) -> np.ndarray:
    """Integer `[B, S]` token ids normalized on the host without a device round-trip."""
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("tokens must be an integer [B, S] array")
    bounds = np.iinfo(np.int32)
    if np.any(array < bounds.min) or np.any(array > bounds.max):
        raise ValueError("token IDs must be representable as int32")
    return array.astype(np.int32, copy=False)


@struct.dataclass
class ModelInputs:
    """Token rows with sequence-aligned fields and row-aligned conditioning.

    Every leaf has the batch on axis 0, and each of ``token_fields`` also has
    the sequence on axis 1, at the length of ``tokens``. ``conditioning`` holds
    media payloads and their valid lengths. A token field such as
    ``image_indices`` gives the index of the media feature read at each text
    slot (-1 for a text token), so a sliced prompt still points at the same
    features. The processor validates the fields on the host. `take_rows`,
    `slice_tokens` and `align_left` also work inside JIT.
    """

    tokens: jax.Array
    token_fields: Mapping[str, jax.Array] = struct.field(default_factory=dict)
    conditioning: Mapping[str, jax.Array] = struct.field(default_factory=dict)

    @classmethod
    def from_value(cls, value: ModelInputs | jax.typing.ArrayLike | Sequence[Sequence[int]]) -> ModelInputs:
        """Return `value` as validated `ModelInputs`, without lossy id conversion or moving resident arrays.

        A `ModelInputs` is validated as it is, and a JAX array is wrapped where
        it sits. Anything else is read on the host as an integer `[B, S]`
        array, and ids that int32 cannot represent raise ValueError.

        Distributed algorithms call this inside their agreed validation phase.
        It runs no collectives and no model.
        """
        if isinstance(value, cls):
            prepared = value
        elif isinstance(value, jax.Array):
            prepared = cls(value)
        else:
            prepared = cls(jnp.asarray(host_token_rows(np.asarray(value))))
        prepared.validate()
        return prepared


    def validate(self) -> None:
        """Check the numeric layout on the host before dispatching a model.

        Raises ValueError when `tokens` is not an integer `[B, S]` array, when
        a token field has a reserved name or does not start with the tokens'
        shape, or when a conditioning entry has another batch size.
        """
        if self.tokens.ndim != 2 or not jnp.issubdtype(self.tokens.dtype, jnp.integer):
            raise ValueError("tokens must be an integer [B, S] array")
        reserved = {"tokens", "conditioning", "train", "decode", "rngs", "method",
                    "mutable", "capture_intermediates", "attention_pairwise_mask",
                    "attention_key_positions"}
        if reserved.intersection(self.token_fields):
            raise ValueError(
                f"token_fields cannot contain {sorted(reserved.intersection(self.token_fields))}"
            )
        for name, value in self.token_fields.items():
            if value.ndim < 2 or value.shape[:2] != self.tokens.shape:
                raise ValueError(
                    f"token field {name!r} must start with {self.tokens.shape}, got {value.shape}"
                )
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
        """Slice the tokens and token fields to slots `start:stop`; media features keep their indices."""
        selection = slice(start, stop)
        return replace(self,
            tokens=self.tokens[:, selection],
            token_fields={name: value[:, selection] for name, value in self.token_fields.items()})

    def align_left(self, left_padding: jax.Array) -> ModelInputs:
        """Rotate each row so its real tokens start at slot 0 and its `left_padding` slots move to the end.

        Positions are token fields, so they move with the tokens and keep their
        values; they are not recomputed from the new slot numbers. Raises
        ValueError unless `left_padding` has one count per row.
        """
        if left_padding.shape != (self.tokens.shape[0],):
            raise ValueError("left_padding must have one count per token row")
        slots = (jnp.arange(self.tokens.shape[1])[None, :] + left_padding[:, None]) % self.tokens.shape[1]

        def shift(value: jax.Array) -> jax.Array:
            indices = slots.reshape(slots.shape + (1,) * (value.ndim - 2))
            return jnp.take_along_axis(value, indices, axis=1)

        return replace(self, tokens=shift(self.tokens),
                            token_fields={name: shift(value) for name, value in self.token_fields.items()})

    def kwargs(self) -> dict[str, jax.Array | Mapping[str, jax.Array]]:
        """Return the keyword arguments for a native model call, without the tokens.

        Each token field is its own keyword, and ``conditioning`` is included
        only when it is not empty.
        """
        prepared: dict[str, jax.Array | Mapping[str, jax.Array]] = dict(self.token_fields)
        if self.conditioning:
            prepared["conditioning"] = self.conditioning
        return prepared

    def extended(self, tokens: jax.Array, *, rules: Mapping[str, FieldExtension] | None = None
                 ) -> ModelInputs:
        """These inputs with a response's `[B, W]` `tokens` after the prompt's slots,
        each token field continued by its rule in `rules` or `RESPONSE_FIELDS`.

        A field with no rule raises ValueError rather than take a guessed
        value; the conditioning stays as it is. Works inside JIT.
        """
        if tokens.ndim != 2 or tokens.shape[0] != self.tokens.shape[0]:
            raise ValueError(f"a response is [{self.tokens.shape[0]}, W] token ids, got {tokens.shape}")
        known = {**RESPONSE_FIELDS, **(rules or {})}
        unruled = sorted(set(self.token_fields) - set(known))
        if unruled:
            raise ValueError(f"token fields {unruled} have no rule for a response's slots; "
                             "name one in `rules`")
        valid = jnp.asarray(self.token_fields.get(VALIDITY_FIELD, jnp.ones(self.tokens.shape, bool)), bool)
        width = tokens.shape[1]
        return replace(
            self, tokens=jnp.concatenate([self.tokens, tokens.astype(self.tokens.dtype)], axis=1),
            token_fields={name: jnp.concatenate([value, known[name](value, valid, width).astype(value.dtype)],
                                                axis=1)
                          for name, value in self.token_fields.items()})


@struct.dataclass
class AttentionMetadata:
    """Per-token data the layers read alongside their inputs, independent of physical cache-slot addresses.

    `token_ids` are the rows' vocabulary ids, for a layer that routes by them
    (DeepSeek V4's hash router); the model sets them when it has such a layer.
    `engram_ids` are every engram layer's n-gram bucket ids,
    `[B, S, layers, columns]`. The model hashes them from the token ids once
    for the whole stack (`dew.nn.engram`), the same way it gives a hash router
    the ids. `media` `[B, S]` marks the positions a media encoder fills;
    DeepSeek-V4.1 routes those by its image bias and keeps them out of every
    n-gram. `admitted` turns a cached call into a serving step's mixed call
    (`Admitted`).
    """

    valid: jax.Array | None = None
    image_groups: jax.Array | None = None
    rotary_positions: jax.Array | None = None
    pairwise_mask: jax.Array | None = None
    key_positions: jax.Array | None = None
    token_ids: jax.Array | None = None
    engram_ids: jax.Array | None = None
    media: jax.Array | None = None
    admitted: Admitted | None = None


@struct.dataclass
class Admitted:
    """A serving step's cached call over every token it runs, in one row.

    The call's `[1, rows + A * W]` tokens are one per cache row, in row
    order, then the prompt pieces of the `A` rows admitted this step, `W`
    tokens each, row `slots[a]` continuing from slot `cursors[a]`. A slot
    past the cache's rows is padding. Attention writes every valid token's
    keys into its row and reads that row's cache (`CausalSelfAttention`);
    every other layer works token by token, so the step's projections read
    their weights once for the decoding rows and the prompts together.
    Unless `continuing`, every piece starts its row (the cursors are 0), and
    a piece's queries read its own keys rather than the row's whole cache.
    Over a paged cache `tables` are the admitted rows' page tables.
    """

    slots: jax.Array
    cursors: jax.Array
    tables: jax.Array | None = None
    continuing: bool = struct.field(pytree_node=False, default=True)


@struct.dataclass
class LayerInputs:
    """Inputs each decoder layer reads for itself, as `[B, S, layers, ...]` leaves.

    The stack slices every leaf on axis 2 the same way. So each layer gets its
    own `[B, S, ...]` slice whether the stack runs as a scan, reads its weights
    from a bank one layer at a time, or runs as a pipeline stage. `embeddings`
    is Gemma 3n/4's per-layer input signal. `experts` and `routed` are a
    routing replay's `[..., top_k]` expert ids and coverage
    (`dew.nn.moe.Routes`), with `routed` broadcast over the layer axis.
    """

    embeddings: jax.Array | None = None
    experts: jax.Array | None = None
    routed: jax.Array | None = None

    def span(self, first: int, count: int) -> LayerInputs:
        """Return layers `first` through `first + count - 1`, keeping the layer axis."""
        return jax.tree.map(lambda leaf: leaf[:, :, first:first + count], self)

    def layer(self, index) -> LayerInputs:
        """Return one layer's slice without the layer axis; `index` may be traced."""
        return jax.tree.map(
            lambda leaf: jax.lax.dynamic_index_in_dim(leaf, index, 2, keepdims=False), self)


def generation_signature(inputs: InputTree, controls: tuple) -> np.ndarray:
    """Digest execution shapes and stable host controls without reading payloads.

    Algorithms agree this fixed-width signature after local validation and
    before distributed execution. The controls are one tuple, and each entry
    must have a deterministic repr, such as a frozen configuration value, a
    scalar, or a tuple or dictionary of them. Tensor contents are excluded:
    this is not a prefix-cache identity.
    """
    schema = (str(jax.tree.structure(inputs)),
              [(leaf.shape, str(leaf.dtype)) for leaf in jax.tree.leaves(inputs)], controls)
    return np.frombuffer(hashlib.sha256(repr(schema).encode()).digest(), np.uint8)


VALIDITY_FIELD = "attention_mask"
"""The token field that marks real slots. Absent means every slot is real."""


def _last_real(value: jax.Array, valid: jax.Array) -> jax.Array:
    """Each row's value at its last real slot, or at slot 0 in a row with none."""
    last = jnp.max(jnp.where(valid, jnp.arange(valid.shape[1])[None, :], 0), axis=1)
    return value[jnp.arange(value.shape[0]), last]


def _response_validity(value: jax.Array, valid: jax.Array, width: int) -> jax.Array:
    """Real in each row with a real prompt token."""
    return jnp.broadcast_to(valid.any(axis=1)[:, None], (valid.shape[0], width))


def _response_positions(value: jax.Array, valid: jax.Array, width: int) -> jax.Array:
    """On from the last real token's position, which a document reset may have lowered."""
    return _last_real(value, valid)[:, None] + jnp.arange(1, width + 1)[None]


def _response_coordinates(value: jax.Array, valid: jax.Array, width: int) -> jax.Array:
    """One past the largest real coordinate, on every axis: text moves along
    all of them together (Qwen's M-RoPE)."""
    keep = valid.reshape(valid.shape + (1,) * (value.ndim - 2))
    last = jnp.max(jnp.where(keep, value, -1), axis=tuple(range(1, value.ndim)))
    coordinates = last[:, None] + jnp.arange(1, width + 1)[None]
    return jnp.broadcast_to(coordinates.reshape((value.shape[0], width) + (1,) * (value.ndim - 2)),
                            (value.shape[0], width, *value.shape[2:]))


def _response_document(value: jax.Array, valid: jax.Array, width: int) -> jax.Array:
    """The last real token's document."""
    return jnp.broadcast_to(_last_real(value, valid)[:, None], (value.shape[0], width))


def _response_text(value: jax.Array, valid: jax.Array, width: int) -> jax.Array:
    """Text: no media feature, no image group (-1)."""
    return jnp.full((value.shape[0], width, *value.shape[2:]), -1, value.dtype)


RESPONSE_FIELDS: Mapping[str, FieldExtension] = {
    VALIDITY_FIELD: _response_validity,
    "positions": _response_positions,
    "rotary_positions": _response_coordinates,
    "segment_ids": _response_document,
    "image_indices": _response_text,
    "audio_indices": _response_text,
    "image_groups": _response_text,
}
"""How each token field Dew's processors write continues into a response."""


def validity_sites(tree: InputTree) -> list[ModelInputs]:
    """The tree's `ModelInputs` nodes, in flatten order.

    A node is a site whether or not it carries validity, so every process
    finds the same sites in the same order and a per-site vector has the same
    length everywhere.
    """
    leaves = jax.tree.leaves(tree, is_leaf=lambda node: isinstance(node, ModelInputs))
    return [leaf for leaf in leaves if isinstance(leaf, ModelInputs)]


def _validity_agnostic[TreeT](tree: TreeT) -> TreeT:
    """`tree` with every validity field dropped."""
    return jax.tree.map(
        lambda node: (replace(node, token_fields={
            name: value for name, value in node.token_fields.items()
            if name != VALIDITY_FIELD}) if isinstance(node, ModelInputs) else node),
        tree, is_leaf=lambda node: isinstance(node, ModelInputs))


def assembly_signature(tree: InputTree, controls: tuple = ()) -> np.ndarray:
    """`generation_signature` of the tree with validity left out.

    Whether a process's own rows needed padding is rank-local, so this digest,
    which still carries every other field, shape, dtype and control, is the
    fixed-width first thing a pool agrees on.
    """
    return generation_signature(_validity_agnostic(tree), controls)


def filled_validity[TreeT](tree: TreeT, wanted: bool | Sequence[bool] = True) -> TreeT:
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


def agreed_validity[TreeT: InputTree](tree: TreeT, processes: int, *, controls: tuple = (),
                                      phase: str = "input") -> TreeT:
    """One validity schema for the whole pool, agreed before arrays are built.

    A host that padded nothing carries no validity, which keeps attention on its
    fused kernel, so processes may differ; where any process carries validity
    for a site, every process materializes it. The collectives are fixed-shape
    and fixed in number (the validity-agnostic signature first, then the
    presence vector its schema sizes) and run on the calling thread, so call
    this in the same order as the caller's other collectives on every process.
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
    return filled_validity(tree, [bool(present) for present in gathered.max(axis=0)])


@overload
def local_rows(leaf: jax.typing.ArrayLike, *, host: Literal[True] = True) -> np.ndarray: ...


@overload
def local_rows(leaf: jax.typing.ArrayLike, *, host: bool) -> jax.Array | np.ndarray: ...


def local_rows(leaf: jax.typing.ArrayLike, *, host: bool = True) -> jax.Array | np.ndarray:
    """This process's rows in global order, optionally kept on device.

    With ``host=False``, fully addressable JAX arrays retain their placement.
    Partial global arrays still gather local shards on the host, reassembling
    a sequence-split second dimension.
    """
    if isinstance(leaf, jax.Array) and leaf.is_fully_addressable and not host:
        return leaf
    if not isinstance(leaf, jax.Array) or leaf.is_fully_addressable:
        return np.asarray(jax.device_get(leaf))
    if leaf.ndim == 0:
        return np.asarray(jax.device_get(leaf.addressable_shards[0].data))
    pieces: dict[tuple[int, ...], np.ndarray] = {}
    for shard in leaf.addressable_shards:
        pieces.setdefault(tuple(part.start or 0 for part in shard.index),
                          np.asarray(jax.device_get(shard.data)))
    blocks = []
    for start in sorted({key[0] for key in pieces}):
        columns = sorted(key for key in pieces if key[0] == start)
        blocks.append(np.concatenate([pieces[key] for key in columns], axis=1)
                      if len(columns) > 1 else pieces[columns[0]])
    return np.concatenate(blocks, axis=0)


def request_key(key: int | jax.Array | None) -> jax.Array:
    """Normalize an integer seed or a single JAX key to one typed PRNG key."""
    if key is None or isinstance(key, bool):
        raise ValueError("key must be an integer seed or a single JAX PRNG key")
    if isinstance(key, (int, np.integer)):
        return jax.random.key(int(key))
    typed = jax.random.wrap_key_data(jax.random.key_data(key), impl=jax.random.key_impl(key))
    if typed.shape != ():
        raise ValueError("key must be a single JAX PRNG key")
    return typed


def key_seed(key: int | jax.Array | None) -> int | None:
    """The normalized key as an integer for an external sampler's wire seed.

    Threefry's key(n) stores [0, n], so summing the words preserves an
    integer seed at the external server. Split keys can collide in 32 bits.
    Native samplers keep the full key instead of this wire representation.
    """
    if key is None:
        return None
    words = np.asarray(jax.random.key_data(request_key(key))).reshape(-1)
    return sum(int(word) for word in words) % (1 << 32)


def continuation_keys(key: jax.Array, n: int) -> jax.Array:
    """``n`` keys on a new leading axis, the first one being ``key`` itself.

    Independent continuations of one request draw from these. Continuation
    zero keeps the request's own key, so a single continuation draws what an
    unrepeated request draws, and no fold depends on ``n``, so asking for more
    continuations leaves the ones already drawn alone. ``key`` is one key or a
    batch of them; a batch folds row by row.
    """
    fold = jax.random.fold_in if key.shape == () else jax.vmap(jax.random.fold_in, in_axes=(0, None))
    indices = jnp.arange(1, n, dtype=jnp.uint32)
    folded = jax.vmap(lambda index: fold(key, index))(indices)
    return jnp.concatenate((key[None], folded), axis=0)


def prompt_major(tree):
    """Mapped continuations as rows: ``[n, B, ...]`` leaves become ``[B * n, ...]``.

    Each prompt's continuations stay together and the prompts keep their
    request order, which is also the order a row plan pads and reads back.
    """
    return jax.tree.map(
        lambda leaf: jnp.swapaxes(leaf, 0, 1).reshape((-1, *leaf.shape[2:])), tree)


def mesh_of(tree) -> jax.sharding.Mesh | None:
    """The mesh the tree's leaves sit on, or None for single-device arrays."""
    for leaf in jax.tree.leaves(tree):
        # A tracer has no sharding to read; an abstract leaf may carry one.
        if isinstance(leaf, Tracer):
            continue
        sharding = leaf.sharding if isinstance(leaf, (jax.Array, jax.ShapeDtypeStruct)) else None
        if (isinstance(sharding, jax.sharding.NamedSharding) and isinstance(sharding.mesh, jax.sharding.Mesh)
                and not sharding.mesh.empty):
            return sharding.mesh
    return None


@dataclass(frozen=True)
class RowPlan:
    """Where one request's rows sit while a task runs.

    Without a mesh every array stays on the default device. On a mesh, rows
    split over its batch axes. Each process contributes ``count`` rows, its
    ``rows`` real ones followed by repeats that pad them to a multiple of its
    devices on those axes, so every device holds the same shape and every
    collective lines up. Results keep that sharding, and ``host`` reads a
    process's real rows back.
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
        """The number of rows of a placed array, over all processes."""
        return self.count * self.processes

    @property
    def padding(self) -> np.ndarray:
        """A mask over the ``count`` placed rows, True for the repeats and False for real rows."""
        return np.arange(self.count) >= self.rows

    def pad(self, tree):
        """Repeat rows up to ``count``, cycling through the real ones; device arrays stay on device."""
        if self.count == self.rows:
            return tree
        indices = np.arange(self.count) % self.rows
        return jax.tree.map(lambda leaf: leaf[indices] if isinstance(leaf, jax.Array)
                            else np.asarray(leaf)[indices], tree)

    def place(self, tree):
        """Place padded host or device rows, sharded by row on a mesh.

        Without a mesh this is `jax.device_put`. Raises ValueError for a leaf
        without ``count`` rows on axis 0.
        """
        sharding = self.sharding
        if sharding is None:
            return jax.device_put(tree)

        def put(leaf):
            rows = leaf if isinstance(leaf, jax.Array) else np.asarray(leaf)
            if rows.ndim == 0 or rows.shape[0] != self.count:
                raise ValueError(f"a placed leaf needs {self.count} rows on axis zero, got {rows.shape}")
            return jax.make_array_from_process_local_data(sharding, rows)

        return jax.tree.map(put, tree)

    def keys(self, key: jax.Array) -> jax.Array:
        """Return one key per placed row, folded from `key` by global row index.

        A pool of processes therefore draws the same values as one process does
        for the same rows. On a mesh, `key` may be replicated over several
        processes; the keys are folded from this process's copy on one device
        and then sharded to their devices.
        """
        sharding = self.sharding
        # A request key replicated over a multi-process mesh is not addressable,
        # and folding it would hand `make_array_from_process_local_data` rows this
        # process cannot place. Every replica holds the same key data, so the rows
        # fold from this process's first replica, on its one device, and are sliced
        # from there to their devices. Folded on every device of a replicated key,
        # they went through the host to be split, because jax reshards a
        # multi-device array whose shards hold no target slice by reading it.
        local = key if sharding is None else key.addressable_data(0)
        start = self.process * self.rows
        indices = jax.device_put(np.arange(start, start + self.count, dtype=np.uint32), local.sharding)
        row_keys = jax.vmap(lambda row: jax.random.fold_in(local, row))(indices)
        if sharding is None:
            return row_keys
        sharded = jax.make_array_from_process_local_data(sharding, jax.random.key_data(row_keys))
        return jax.random.wrap_key_data(sharded, impl=jax.random.key_impl(key))

    def host(self, leaf) -> np.ndarray:
        """Return this process's real rows of a result leaf as a host array."""
        return local_rows(leaf)[:self.rows]


@dataclass(frozen=True)
class Request:
    """One sampler request as this process holds it, ready to place.

    `inputs` are this process's rows as the sampler checked them, `plan`
    places them and `key` is the request's one PRNG key. When the weights sit
    on a mesh across processes the pool has agreed, before anything is
    placed, on the check's outcome, one validity schema and the signature of
    `inputs` with the sampler's controls, so every process enters the program
    with the same shapes; otherwise each process serves its own request.
    """

    inputs: ModelInputs
    plan: RowPlan
    key: jax.Array

    @classmethod
    def prepare[CheckedT](
            cls, inputs: ModelInputs | jax.typing.ArrayLike | Sequence[Sequence[int]],
            key: int | jax.Array | None, mesh: jax.sharding.Mesh | None,
            check: Callable[[ModelInputs, bool], tuple[ModelInputs, tuple, CheckedT]],
            *, phase: str) -> tuple[Request, CheckedT]:
        """The request, with whatever else `check` resolved.

        `check` takes the request as `ModelInputs` and whether a pool shares
        it, and returns the inputs the sampler runs, the controls a pool
        compares (each with a deterministic repr) and anything else it
        resolved. What it raises, a pool raises together, before any rank
        enters a collective.
        """
        pooled = mesh is not None and jax.process_count() > 1

        def setup() -> tuple[jax.Array, tuple[ModelInputs, tuple, CheckedT]]:
            return request_key(key), check(ModelInputs.from_value(inputs), pooled)

        random_key, (prepared, controls, checked) = (agreed(f"{phase} setup", setup) if pooled
                                                     else setup())
        if pooled:
            from jax.experimental import multihost_utils

            prepared = agreed_validity(prepared, jax.process_count(), controls=controls,
                                       phase=f"{phase} input")
            multihost_utils.assert_equal(generation_signature(prepared, controls),
                                         f"{phase} input shapes and controls must agree across processes")
        return cls(prepared, RowPlan.over(mesh, prepared.tokens.shape[0]), random_key), checked

    def padded(self) -> ModelInputs:
        """`inputs` filled out to the rows the devices take, the repeats
        marked invalid, so they hold no real token and finish at once."""
        plan, padded = self.plan, self.plan.pad(self.inputs)
        if plan.count == plan.rows:
            return padded
        valid = padded.token_fields.get(VALIDITY_FIELD, jnp.ones(padded.tokens.shape, bool))
        valid = jnp.asarray(valid, bool) & ~plan.padding[:, None]
        return replace(padded, token_fields={**padded.token_fields, VALIDITY_FIELD: valid})


__all__ = ["RESPONSE_FIELDS", "AttentionMetadata", "FieldExtension", "LayerInputs", "ModelInputs",
           "ModelKwarg", "RowPlan"]
