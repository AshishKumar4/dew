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

from dew._coordination import agreed
from dew.batch import (
    COORDINATES_FIELD,
    VALIDITY_FIELD,
    Coordinates,
    coordinates,
    document_firsts,
    document_ranks,
    document_starts,
    padded_rows,
    real_slots,
    token_ids,
    token_positions,
    token_validity,
    with_validity,
)
from dew.mesh import SEQUENCE_AXIS, STAGE_AXIS
from dew.nn.sharding import explicit_spec, rows_like

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
type InputTree = (ModelInputs | jax.Array | np.ndarray
                  | Sequence[InputTree] | Mapping[str, object] | None)

# A keyword a model call takes past its tokens: a token field or position
# array, the conditioning mapping (`ModelInputs.kwargs`), a flag, or None for
# an absent input.
type ModelKwarg = jax.Array | Mapping[str, jax.Array] | bool | None

# A token field's `[B, W, ...]` response values from its `[B, S, ...]` prompt
# values, the prompt's `[B, S]` real slots and the width (`ModelInputs.extended`).
type FieldExtension = Callable[[jax.Array, jax.Array, int], jax.Array]

PredictionPhase = Literal["ordinary", "extend", "draft"]


def sequence_blocks(x: jax.Array, block: int, blocks: int | None = None, *,
                    leading: bool = False, fill=0) -> jax.Array:
    """`x` `[B, S, ...]` padded with `fill` at the end to `blocks` blocks of `block` positions (as
    many as `S` needs when None) and cut into them, `[B, blocks, block, ...]`, or with the blocks
    leading, `[blocks, B, block, ...]`, for a scan or map over them (`leading`)."""
    blocks = -(-x.shape[1] // block) if blocks is None else blocks
    x = jnp.pad(x, [(0, 0), (0, blocks * block - x.shape[1])] + [(0, 0)] * (x.ndim - 2),
                constant_values=fill)
    x = x.reshape(x.shape[0], blocks, block, *x.shape[2:])
    return jnp.moveaxis(x, 1, 0) if leading else x


def pad_token_rows(rows: Sequence[Sequence[int]] | np.ndarray, *, pad_id: int = 0,
                   padding_side: Literal["left", "right"] = "left",
                   fields: Mapping[str, Sequence[Sequence[int]] | np.ndarray] | None = None,
                   width: int | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Pad ragged token rows and their scalar token fields on the host, to `width` when given (a
    fixed shape, which no row may exceed), else to the longest row.

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
    for row in arrays:
        _check_token_ids(row)
    longest = max(row.size for row in arrays)
    if width is None:
        width = longest
    elif longest > width:
        raise ValueError(f"a token row holds {longest} ids, more than the width {width}")
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


def _check_token_ids(array: np.ndarray) -> None:
    """Refuse an integer array holding an id outside `token_id`'s law: 0 through int32's largest."""
    if np.any((array < 0) | (array > np.iinfo(np.int32).max)):
        raise ValueError("token IDs must be nonnegative int32 values")


def _token_row(row: object) -> list[int] | np.ndarray:
    """One row of token ids by `token_id`'s law, before rows join into one array, where NumPy would
    promote a bool into an id: a sequence token by token (`token_ids`), an array whole by its dtype,
    an integer one, whose range `host_token_rows` checks with the rest."""
    if isinstance(row, Sequence) and not isinstance(row, str | bytes):
        return token_ids(list(row), where="a token row")
    array = np.asarray(row)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"a token row holds integer token ids, got an array of {array.dtype}")
    return array


def host_token_rows(array: np.ndarray) -> np.ndarray:
    """Integer `[B, S]` token ids normalized on the host without a device round-trip,
    read by `token_id`'s law whatever container holds them."""
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("tokens must be an integer [B, S] array")
    _check_token_ids(array)
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
        array, and ids that int32 cannot represent raise ValueError. Rows given
        as a sequence are each read by `token_id`'s law first (`_token_row`), so
        a bool among them is refused rather than promoted into an id.

        Distributed algorithms call this inside their agreed validation phase.
        It runs no collectives and no model.
        """
        if isinstance(value, cls):
            prepared = value
        elif isinstance(value, jax.Array):
            prepared = cls(value)
        else:
            rows = ([_token_row(row) for row in value]
                    if isinstance(value, Sequence) and not isinstance(value, str | bytes) else value)
            prepared = cls(jnp.asarray(host_token_rows(np.asarray(rows))))
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
                    "attention_key_positions", "attention_key_runs"}
        if reserved.intersection(self.token_fields):
            raise ValueError(
                f"token_fields cannot contain {sorted(reserved.intersection(self.token_fields))}"
            )
        for name, value in self.token_fields.items():
            # A field may be a tree of arrays; `coordinates` states which of its arrays run along the tokens
            # (`Coordinates.TOKEN_ALIGNED`), its documents' firsts one entry a document.
            aligned = value.tokens() if isinstance(value, Coordinates) else value
            for leaf in jax.tree.leaves(aligned):
                if leaf.ndim < 2 or leaf.shape[:2] != self.tokens.shape:
                    raise ValueError(
                        f"token field {name!r} must start with {self.tokens.shape}, got {leaf.shape}"
                    )
            if isinstance(value, Coordinates) and value.firsts is not None and (
                    value.firsts.shape[0] != self.tokens.shape[0]):
                raise ValueError(f"token field {name!r}'s documents' firsts must have batch size "
                                 f"{self.tokens.shape[0]}, got {value.firsts.shape}")
        for name, value in self.conditioning.items():
            if value.ndim < 1 or value.shape[0] != self.tokens.shape[0]:
                raise ValueError(f"conditioning {name!r} must have batch size {self.tokens.shape[0]}")

    def take_rows(self, indices: jax.Array) -> ModelInputs:
        """Select the same rows from tokens, sequence fields and media."""
        return replace(self,
            tokens=self.tokens[indices],
            token_fields={name: jax.tree.map(lambda leaf: leaf[indices], value)
                          for name, value in self.token_fields.items()},
            conditioning={name: value[indices] for name, value in self.conditioning.items()})

    def slice_tokens(self, start: int | None = None, stop: int | None = None) -> ModelInputs:
        """Slice the tokens and token fields to slots `start:stop`; media features keep their indices."""
        selection = slice(start, stop)
        # A field may be a tree of arrays, each sliced alike; `coordinates` slices its token-aligned ones
        # (`Coordinates.along_tokens`).
        def sliced(value):
            if isinstance(value, Coordinates):
                return value.along_tokens(lambda leaf: leaf[:, selection])
            return jax.tree.map(lambda leaf: leaf[:, selection], value)

        return replace(self, tokens=self.tokens[:, selection],
                       token_fields={name: sliced(value) for name, value in self.token_fields.items()})

    def align_left(self, left_padding: jax.Array) -> ModelInputs:
        """Rotate each row so its real tokens start at slot 0 and its `left_padding` slots move to the end.

        Positions are token fields, so they move with the tokens and keep their
        values; they are not recomputed from the new slot numbers. Raises
        ValueError unless `left_padding` has one count per row.
        """
        if left_padding.shape != (self.tokens.shape[0],):
            raise ValueError("left_padding must have one count per token row")

        def shift(value: jax.Array) -> jax.Array:
            return rotate_rows(value, left_padding)

        def shifted(value):
            # `coordinates` rotates its token-aligned arrays (`Coordinates.along_tokens`).
            if isinstance(value, Coordinates):
                return value.along_tokens(shift)
            return jax.tree.map(shift, value)

        return replace(self, tokens=shift(self.tokens),
                            token_fields={name: shifted(value) for name, value in self.token_fields.items()})

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
        value; the conditioning stays as it is. The rules read the prompt's
        real slots (`real_slots`), so a packed row's padding continues
        nothing. Works inside JIT.
        """
        if tokens.ndim != 2 or tokens.shape[0] != self.tokens.shape[0]:
            raise ValueError(f"a response is [{self.tokens.shape[0]}, W] token ids, got {tokens.shape}")
        known = {**RESPONSE_FIELDS, **(rules or {})}
        unruled = unruled_fields(self.token_fields, rules)
        if unruled:
            raise ValueError(f"token fields {unruled} have no rule for a response's slots; "
                             "name one in `rules`")
        valid = real_slots(self.token_fields, self.tokens.shape)
        width = tokens.shape[1]

        def continued(name: str, value: jax.Array | Coordinates) -> jax.Array | Coordinates:
            if isinstance(value, Coordinates):
                return _continued_layout(value, valid, width)
            return jnp.concatenate([value, known[name](value, valid, width).astype(value.dtype)], axis=1)

        return replace(
            self, tokens=jnp.concatenate([self.tokens, tokens.astype(self.tokens.dtype)], axis=1),
            token_fields={name: continued(name, value) for name, value in self.token_fields.items()})


def rotate_rows(values: jax.Array, shifts: jax.Array) -> jax.Array:
    """Each row of `values` `[B, S, ...]` rotated left by its own count in `shifts` `[B]`: slot `i`
    reads slot `(i + shift) % S`, so a row's first `shift` slots move to its end, and a negative
    count rotates it back."""
    slots = (jnp.arange(values.shape[1])[None, :] + shifts[:, None]) % values.shape[1]
    return jnp.take_along_axis(values, slots.reshape(slots.shape + (1,) * (values.ndim - 2)), axis=1)


def cu_seqlens(begins: jax.Array, sequences: int, firsts: jax.Array | None = None) -> jax.Array:
    """The cumulative offsets of the varlen sequences over the `[B * S]`
    tokens of `begins` `[B, S]`, which marks each sequence's first token: a
    kernel's `cu_seqlens` `[sequences + 1]`, the sequences past the marked
    ones empty at the end. Every row's first token begins one. They come from
    each row's documents' first slots, the batch's `firsts`
    (`Coordinates.firsts`) where given, offset by their rows and sorted: no
    scan over the flattened tokens, whose split over a mesh XLA runs a token
    a loop iteration (row 1897)."""
    rows, length = begins.shape
    given = document_firsts(begins, -(-sequences // rows)) if firsts is None else firsts
    split = explicit_spec(given)
    if split is not None:
        # A batch's few documents, whole on each device, as the kernels read the offsets.
        whole = jax.sharding.NamedSharding(split.mesh, jax.sharding.PartitionSpec())
        given = jax.sharding.reshard(given, whole)
    placed = jnp.where(given < length, given + jnp.arange(rows, dtype=jnp.int32)[:, None] * length,
                       rows * length).reshape(-1)
    starts = jnp.sort(placed)[:sequences]
    starts = jnp.pad(starts, (0, sequences - starts.shape[0]), constant_values=rows * length)
    return jnp.concatenate([starts, jnp.full((1,), rows * length, starts.dtype)]).astype(jnp.int32)


def coordinates_of(metadata: AttentionMetadata | None, batch: int, length: int) -> Coordinates:
    """The call's `Coordinates`: its metadata's, else those of whole rows of real tokens."""
    if metadata is not None and metadata.coordinates is not None:
        return metadata.coordinates
    return coordinates((batch, length))


def counted_places(segments=None, valid=None, positions=None) -> jax.Array | None:
    """Each token's place counted over its document's real tokens `[B, S]`, which windows, chunks and
    ALiBi read (`Coordinates`): the place its document's first real token is given (`positions`
    `[S]` or `[B, S]`, 0 where none are) plus the real tokens before it in its document, a padding
    slot the place before it. Places that count a document's real tokens, as `Coordinates.positions`
    does, come back as given, so a continuation or canvas origin cuts its chunks as its whole pass
    does; any other places move the rotation alone. None where no field is given: each token's
    place is its index. The counts run along the sequence, which JAX refuses over an Explicit axis
    that splits it; these fields are small, so each device takes its rows of each whole
    (`rows_like`)."""
    if segments is None and valid is None and positions is None:
        return None
    # A shared `[S]` vector of places is a row, its sequence the second dimension as the others'.
    given = None if positions is None else jnp.atleast_2d(positions)
    segments, valid, given = (None if x is None else rows_like(jnp.asarray(x), jnp.asarray(x))
                              for x in (segments, valid, given))
    shape = jnp.shape(next(x for x in (segments, valid, given) if x is not None))
    real = jnp.ones(shape, bool) if valid is None else jnp.asarray(valid, bool)
    if segments is not None:
        real = real & (segments != 0)
    starts = jnp.zeros(shape, bool) if segments is None else document_starts(segments, real)
    count = document_ranks(real, starts)
    if given is None:
        return count
    firsts = starts | (real & (jnp.cumsum(real, axis=1) == 1))
    slots = _last_marked(firsts)
    return jnp.take_along_axis(jnp.broadcast_to(given, shape), slots, axis=1) + count


def numbered_blocks(tokens: Coordinates, shape: tuple[int, int], block: int) -> jax.Array:
    """Each token's block of `block` places, counted over its document's real tokens
    (`counted_places`), numbered along its row from 0, a padding slot -1 `[B, S]`: the tokens a block
    router chooses its experts for together (`dew.nn.moe.Router.block_mask`), each row's and each
    document's own, so a request routes as it does alone and padding votes for none."""
    counted = counted_places(tokens.segments, tokens.valid, tokens.positions)
    cut = jnp.broadcast_to(jnp.arange(shape[1]) if counted is None else counted, shape) // block
    opens = jnp.concatenate([jnp.ones((shape[0], 1), bool), cut[:, 1:] != cut[:, :-1]], axis=1)
    if tokens.segments is not None:
        real = jnp.ones(shape, bool) if tokens.valid is None else jnp.broadcast_to(tokens.valid, shape)
        opens = opens | document_starts(jnp.broadcast_to(jnp.asarray(tokens.segments), shape), real)
    numbered = jnp.cumsum(opens, axis=1, dtype=jnp.int32) - 1
    if tokens.valid is None:
        return numbered
    return jnp.where(jnp.broadcast_to(tokens.valid, shape), numbered, -1)


def document_reach(segments, valid, places) -> jax.Array:
    """Each token's document's furthest place `[B, S]`: the largest of `places` over the document's real
    tokens, which LongRoPE picks its factors by, as the reference picks them by a sequence's largest
    position, so a packed document reads the factors it reads alone, its places in any order. A padding
    slot reads the next document's, or its row's last. Documents are where `document_starts` puts them."""
    segments = rows_like(jnp.asarray(segments), jnp.asarray(segments))
    shape = jnp.shape(segments)
    places = rows_like(jnp.broadcast_to(places, shape), segments)
    real = segments != 0
    if valid is not None:
        real = real & rows_like(jnp.asarray(valid, bool), segments)
    index = jnp.broadcast_to(jnp.arange(shape[1]), shape)
    starts = document_starts(segments, real)
    # Each real token's document, numbered per row from its first real token, and that document's
    # largest place; a padding slot's number reaches no document.
    numbered = jnp.cumsum(starts | (real & (jnp.cumsum(real, axis=1) == 1)), axis=1, dtype=jnp.int32)
    documents = jnp.where(real, numbered + jnp.arange(shape[0])[:, None] * (shape[1] + 1), -1)
    furthest = jax.ops.segment_max(jnp.where(real, places, jnp.iinfo(jnp.int32).min).ravel(),
                                   documents.ravel(), num_segments=shape[0] * (shape[1] + 1))
    held = jnp.take(furthest, jnp.maximum(documents, 0)).reshape(shape)
    # The slot of the next document's first real token after each slot, the row's length past the last.
    following = jnp.where(starts, index, shape[1])
    following = jnp.flip(jax.lax.cummin(jnp.flip(following, axis=1), axis=1), axis=1)
    following = jnp.pad(following[:, 1:], ((0, 0), (0, 1)), constant_values=shape[1])
    # The last real token before it, which ends the slot's document.
    ends = jnp.take_along_axis(_last_marked(real), following - 1, axis=1)
    return jnp.take_along_axis(held, ends, axis=1).astype(places.dtype)


def _last_marked(marked: jax.Array) -> jax.Array:
    """Each slot's last `marked` slot at or before it `[B, S]`, 0 before any: a document's first
    slot where `marked` holds its starts (`counted_places`), its last real one where it holds the
    real tokens (`document_reach`)."""
    return jax.lax.cummax(jnp.where(marked, jnp.arange(marked.shape[1]), 0), axis=1)


CONTEXT = "context"
"""The `ModelInputs.conditioning` entry a decoder's cross-attention layers read,
`[B, C, features]` (`dew.nn.cross_attention`)."""
CONTEXT_LENGTHS = "context_lengths"
"""The `ModelInputs.conditioning` entry counting each row's real context tokens, `[B]`
integers; a row's padding comes after them. Without it every context token is real."""
CONTEXT_SEGMENTS = "context_segments"
"""The `ModelInputs.conditioning` entry naming each context token's document `[B, C]` integers, the
packed row of the call it belongs to (0 for padding): a call's token reads the context of its own
document alone, as several packed rows each read their own context. Without it every token of a row
reads the row's whole context."""


@struct.dataclass
class CrossContext:
    """The sequence a call's cross-attention layers read: `embeddings` `[B, C, features]`
    and each row's count of real tokens, `lengths` `[B]` int32, its padding after them, and where the
    row packs several documents, each context token's document, `segments` `[B, C]` (0 for padding,
    `CONTEXT_SEGMENTS`), whose tokens the call's tokens of that document alone read; and the call's own
    tokens' documents, `queries` `[B, S]`, which the decoder sets from its coordinates."""

    embeddings: jax.Array
    lengths: jax.Array
    segments: jax.Array | None = None
    queries: jax.Array | None = None

    @staticmethod
    def parted(conditioning: Mapping[str, jax.Array] | None
               ) -> tuple[dict[str, jax.Array] | None, dict[str, jax.Array] | None]:
        """`conditioning`'s context entries (`CONTEXT`, `CONTEXT_LENGTHS`), which a decoder reads, and
        the rest, which a module around the decoder reads itself (a multimodal model's media); each
        None where it holds nothing."""
        context = {name: value for name, value in (conditioning or {}).items()
                   if name in (CONTEXT, CONTEXT_LENGTHS, CONTEXT_SEGMENTS)}
        rest = {name: value for name, value in (conditioning or {}).items() if name not in context}
        return context or None, rest or None

    @classmethod
    def from_conditioning(cls, conditioning: Mapping[str, jax.Array]) -> CrossContext:
        """The context a call's `conditioning` carries under `CONTEXT`, `CONTEXT_LENGTHS` and
        `CONTEXT_SEGMENTS`, as JAX arrays (`checked`)."""
        embeddings, lengths, segments = cls.checked(conditioning, jnp)
        return cls(jnp.asarray(embeddings), jnp.asarray(lengths, jnp.int32),
                   None if segments is None else jnp.asarray(segments, jnp.int32))

    @staticmethod
    def checked(conditioning: Mapping[str, jax.typing.ArrayLike], xp=np
                ) -> tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike | None]:
        """The context, its lengths and its tokens' documents a call's `conditioning` carries under
        `CONTEXT`, `CONTEXT_LENGTHS` and `CONTEXT_SEGMENTS`, as arrays of `xp` (NumPy's on the host,
        where a server admits a request; JAX's), each row's length its context's width where none is
        given, the documents None where none are given, each in the dtype given.

        Raises ValueError for any other entry, for a context that is not
        `[B, C, features]` floats, for lengths that are not one integer
        per row, and for documents that are not one integer per context token.
        """
        unknown = sorted(set(conditioning) - {CONTEXT, CONTEXT_LENGTHS, CONTEXT_SEGMENTS})
        if unknown or CONTEXT not in conditioning:
            raise ValueError(f"a decoder's conditioning is {CONTEXT!r} and optionally {CONTEXT_LENGTHS!r} "
                             f"and {CONTEXT_SEGMENTS!r}, got {sorted(conditioning)}")
        embeddings = xp.asarray(conditioning[CONTEXT])
        if embeddings.ndim != 3 or not jnp.issubdtype(embeddings.dtype, jnp.floating):
            raise ValueError(f"a context is [B, C, features] floats, got "
                             f"{embeddings.dtype}{embeddings.shape}")
        lengths = conditioning.get(CONTEXT_LENGTHS)
        if lengths is None:
            lengths = xp.full(embeddings.shape[:1], embeddings.shape[1], np.int32)
        lengths = xp.asarray(lengths)
        if lengths.shape != embeddings.shape[:1] or not jnp.issubdtype(lengths.dtype, jnp.integer):
            raise ValueError(f"context lengths are one integer per row, [{embeddings.shape[0]}], "
                             f"got {lengths.dtype}{lengths.shape}")
        segments = conditioning.get(CONTEXT_SEGMENTS)
        if segments is not None:
            segments = xp.asarray(segments)
            if segments.shape != embeddings.shape[:2] or not jnp.issubdtype(segments.dtype, jnp.integer):
                raise ValueError(f"context segments are one integer per context token, "
                                 f"{list(embeddings.shape[:2])}, got {segments.dtype}{segments.shape}")
        return embeddings, lengths, segments


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
    (`Admitted`). `position_bias` is added to every attention logit,
    `[B or 1, heads, S, keys]`: T5's relative position table, which its
    encoder computes once for its layers. `blocks` `[B, S]` number each row's
    routing blocks of `routing_block` places for a block router
    (`numbered_blocks`), which the model computes once for its layers.
    `context` is the sequence its cross-attention layers read (`CrossContext`).
    `pairwise_mask` `[B, S, keys]`, or `key_runs` `[B, S, R, 2]`, each query's
    key intervals whose union it reads, replaces a call's visibility, a
    windowed layer windowing it over `key_positions` `[B, keys]`.

    `image_groups` and `block_groups` `[B, S]` int32 number the tokens' blocks,
    each a run of contiguous tokens, an image's or a prompt's, and -1 for a
    token outside any, padding among them; the model calls take them as
    `image_groups=` and `block_groups=`, and so does
    `dew.nn.attention.scaled_dot_product_attention`. A causal attention layer that reads
    them (`AttentionMixer.bidirectional_images`, `bidirectional_blocks`) lets
    each token of an image see the image's later tokens within its window,
    and each token of a block its whole block, within its document; a layer
    reading both sees the union, and every other layer stays causal.

    `coordinates` are the tokens' documents, validity, positions and cache
    slots (`Coordinates`). Their `right_padded` says each row's real tokens
    come first, its padding after them, as a training batch pads its rows. A
    real token then reads no padding through a causal layer, so those layers
    skip what would keep it out: causal attention takes no key mask and stays
    on its fused kernel, and a conv reads the row as it stands
    (`dew.nn.linear.held_conv1d`). A padding slot's output is then
    unspecified.
    """

    coordinates: Coordinates | None = None
    image_groups: jax.Array | None = None
    block_groups: jax.Array | None = None
    rotary_positions: jax.Array | None = None
    pairwise_mask: jax.Array | None = None
    key_positions: jax.Array | None = None
    key_runs: jax.Array | None = None
    token_ids: jax.Array | None = None
    engram_ids: jax.Array | None = None
    media: jax.Array | None = None
    admitted: Admitted | None = None
    position_bias: jax.Array | None = None
    blocks: jax.Array | None = None
    context: CrossContext | None = None
    routing_block: int | None = struct.field(pytree_node=False, default=None)

    @property
    def valid(self) -> jax.Array | None:
        """The real tokens (`Coordinates.valid`), None where every token is."""
        return None if self.coordinates is None else self.coordinates.valid

    @property
    def right_padded(self) -> bool:
        return self.coordinates is not None and self.coordinates.right_padded


MIXING = ("pairwise_mask", "key_positions", "key_runs", "position_bias", "admitted")
"""The `AttentionMetadata` fields that say which tokens a mixer lets each token
read, or how much: a call's own mask, or its keys' runs, and the key positions
it is read by, an additive bias, a server's mixed layout. A mixer runs a call that sets one only
where it reads it (`MixerCall`). The rest describe the tokens themselves, which
a mixer reads as far as its form uses them: their coordinates, rotary
positions, and image and block groups, which only attention that sees an image
or a block whole reads, and the ids the block reads for its feed-forward and
n-gram lookup."""

TOKENWISE = frozenset(MIXING)
"""The `MIXING` fields a mixer that reads no other token holds to: every one,
by construction, since none changes what a token alone computes."""


@dataclass(frozen=True)
class MixerCall:
    """Everything a token mixer is called with beside its input: the stack's
    per-token `metadata`, whether the call `decode`s against the layer's cache,
    and the store a run of layers shares keys and values through (`kv_store`),
    which a provider writes and a sharing layer reads. A mixer declares the
    `MIXING` fields it reads (`reads`), and the block refuses to call it with
    any other one set (`unread`)."""

    metadata: AttentionMetadata | None = None
    decode: bool = False
    kv_store: dict | None = None
    """Each name a layer publishes under to the run's later layers: attention's
    keys, values and positions, an index selection, a draft's context."""

    def unread(self, reads: frozenset[str]) -> tuple[str, ...]:
        """The `MIXING` fields this call sets that a mixer reading `reads` does not read."""
        if self.metadata is None:
            return ()
        return tuple(name for name in MIXING
                     if name not in reads and getattr(self.metadata, name) is not None)


PLAIN_CALL = MixerCall()
"""A mixer call over whole rows of real tokens: no metadata, no cache, no store."""


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
    (`dew.nn.protocols.Routes`), with `routed` broadcast over the layer axis.
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
    """Text: no media feature, no image or block group (-1)."""
    return jnp.full((value.shape[0], width, *value.shape[2:]), -1, value.dtype)


def _continued_layout(value: Coordinates, valid: jax.Array, width: int) -> Coordinates:
    """A request's `Coordinates` continued over its response's `width` slots: each field by its
    own field's rule (`RESPONSE_FIELDS`), no document starting in the response, and its real
    tokens no longer all first, since the response follows the prompt's padding."""
    def continued(field, rule):
        return None if field is None else jnp.concatenate(
            [field, rule(field, valid, width).astype(field.dtype)], axis=1)

    starts = None if value.starts is None else jnp.concatenate(
        [value.starts, jnp.zeros((len(value.starts), width), value.starts.dtype)], axis=1)
    # No document starts in the response: each row's firsts stand, past its last now `width` further.
    firsts = value.firsts
    if firsts is not None and value.starts is not None:
        length = value.starts.shape[1]
        firsts = jnp.where(firsts < length, firsts, length + width)
    return replace(value, valid=continued(value.valid, _response_validity),
                   positions=continued(value.positions, _response_positions),
                   segments=continued(value.segments, _response_document), starts=starts, firsts=firsts,
                   right_padded=False)


RESPONSE_FIELDS: Mapping[str, FieldExtension] = {
    VALIDITY_FIELD: _response_validity,
    "positions": _response_positions,
    "rotary_positions": _response_coordinates,
    "segment_ids": _response_document,
    "image_indices": _response_text,
    "audio_indices": _response_text,
    "image_groups": _response_text,
    "block_groups": _response_text,
}
"""How each token field Dew's processors write continues into a response."""


def unruled_fields(fields: Mapping[str, object],
                   rules: Mapping[str, FieldExtension] | None = None) -> list[str]:
    """The token fields with no rule for a response's slots, by name: neither in `RESPONSE_FIELDS` nor
    in `rules`. A request's `Coordinates` continue by their own fields' rules (`_continued_layout`).
    What every extension of a request (`ModelInputs.extended`, masked generation) admits."""
    return sorted(set(fields) - set(RESPONSE_FIELDS) - set(rules or ()) - {COORDINATES_FIELD})


def next_position(fields: Mapping[str, jax.Array], valid: jax.Array) -> jax.Array | None:
    """Each row's logical position after its last real token `[B]`, its
    response's first by `RESPONSE_FIELDS`: one past the largest coordinate a
    real token holds on any axis of its `rotary_positions`, else one past its
    last real token's position (`token_positions`), which a document reset may
    have lowered below an earlier one. None where the request gives neither,
    and a cache's cursor counts each row on. `valid` is its real slots (`real_slots`)."""
    rotary = fields.get("rotary_positions")
    if rotary is not None:
        return _response_coordinates(rotary, valid, 1).reshape(rotary.shape[0], -1)[:, 0]
    positions = token_positions(fields)
    return None if positions is None else _response_positions(positions, valid, 1)[:, 0]


def validity_sites(tree: InputTree) -> list[ModelInputs]:
    """The tree's `ModelInputs` nodes, in flatten order.

    A node is a site whether or not it carries validity, so every process
    finds the same sites in the same order and a per-site vector has the same
    length everywhere.
    """
    leaves = jax.tree.leaves(tree, is_leaf=lambda node: isinstance(node, ModelInputs))
    return [leaf for leaf in leaves if isinstance(leaf, ModelInputs)]


def _validity_agnostic[TreeT](tree: TreeT) -> TreeT:
    """`tree` with every validity dropped (`with_validity`)."""
    return jax.tree.map(
        lambda node: (replace(node, token_fields=with_validity(node.token_fields, None))
                      if isinstance(node, ModelInputs) else node),
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
        if not needed or token_validity(node.token_fields) is not None:
            return node
        shape = np.shape(node.tokens)
        ones = (np.ones(shape, bool) if isinstance(node.tokens, np.ndarray)
                else jnp.ones(shape, bool))
        return replace(node, token_fields=with_validity(node.token_fields, ones))

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
    present = np.asarray([token_validity(site.token_fields) is not None
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


def host_rows(leaf: jax.typing.ArrayLike, rows: int | None) -> np.ndarray:
    """This process's first `rows` real rows of a result leaf, as a host array (`local_rows`): the
    result without the rows its batch was padded with to fill the devices. None keeps every row."""
    return local_rows(leaf)[:rows]


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


def seed_of(key: int | jax.Array) -> int | None:
    """`key` as an integer seed, None for a JAX PRNG key, refused as `request_key` refuses it, without
    making a key: a seed's key opens JAX's backend."""
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
        return int(key)
    if not isinstance(key, jax.Array):
        raise ValueError("key must be an integer seed or a single JAX PRNG key")
    return None


def request_rows(keys: jax.Array) -> jax.Array:
    """Normalize a vector of JAX keys, one a row (`dew.sampling.solvers.RowKeys`), to typed keys."""
    typed = jax.random.wrap_key_data(jax.random.key_data(keys), impl=jax.random.key_impl(keys))
    if typed.ndim != 1:
        raise ValueError(f"row keys are a vector of JAX PRNG keys, got shape {typed.shape}")
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


def default_placement() -> jax.sharding.SingleDeviceSharding:
    """Where a `jnp` array is created: `jax.default_device`'s device, or this process's first of the
    default backend's (`jax_default_device` may name a backend)."""
    default = jax.config.jax_default_device
    if default is None or isinstance(default, str):
        return jax.sharding.SingleDeviceSharding(jax.local_devices(backend=default)[0])
    return jax.sharding.SingleDeviceSharding(default)


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


def request_axes(mesh: jax.sharding.Mesh) -> tuple[str, ...]:
    """The mesh axes a request's rows split over: every axis but sequence and
    stage, so each process places rows of its own requests on its own
    devices. A training batch's rows split over the data axes alone
    (`dew.mesh.batch_axes`): every process of a share reads the same
    rows there, where each process here brings its own."""
    return tuple(axis for axis in mesh.axis_names if axis not in (SEQUENCE_AXIS, STAGE_AXIS))


@dataclass(frozen=True)
class RowPlan:
    """Where one request's rows sit while a task runs.

    Without a mesh every array stays on the default device. On a mesh, rows
    split over its request axes (`request_axes`). Each process contributes ``count`` rows, its
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
        batch_devices = math.prod(mesh.shape[axis] for axis in request_axes(mesh))
        if batch_devices % processes:
            raise ValueError("the batch axes of the mesh do not split evenly over the processes")
        local_devices = batch_devices // processes
        return cls(mesh, rows, -(-rows // local_devices) * local_devices, process, processes)

    @property
    def sharding(self) -> jax.sharding.NamedSharding | None:
        if self.mesh is None:
            return None
        return jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(request_axes(self.mesh)))

    @property
    def global_rows(self) -> int:
        """The number of rows of a placed array, over all processes."""
        return self.count * self.processes

    @property
    def padding(self) -> np.ndarray:
        """A mask over the ``count`` placed rows, True for the repeats and False for real rows
        (`padded_rows`)."""
        return padded_rows(self.rows, self.count)[1]

    def pad(self, tree):
        """Repeat rows up to ``count``, cycling through the real ones (`padded_rows`); device arrays
        stay on device."""
        if self.count == self.rows:
            return tree
        indices, _ = padded_rows(self.rows, self.count)
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
        return host_rows(leaf, self.rows)


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
        valid = token_validity(padded.token_fields)
        valid = (jnp.ones(padded.tokens.shape, bool) if valid is None else jnp.asarray(valid, bool)
                 ) & ~plan.padding[:, None]
        return replace(padded, token_fields=with_validity(padded.token_fields, valid))


__all__ = ["CONTEXT", "CONTEXT_LENGTHS", "RESPONSE_FIELDS", "AttentionMetadata", "CrossContext",
           "FieldExtension", "LayerInputs", "ModelInputs", "ModelKwarg", "RowPlan"]


def routing_blocks(feedforward, metadata: AttentionMetadata | None, shape: tuple[int, ...]) -> dict:
    """The blocks a block-routed feed-forward (`dew.nn.moe.SparseMLP`'s `routing_block`) routes its
    tokens in, over a call of `shape[:2]`: its metadata's where they number blocks of its size,
    which the model numbers once for its layers, else numbered from the call's coordinates
    (`numbered_blocks`); nothing for another feed-forward."""
    from dew.nn.protocols import RoutesBlocks

    block = feedforward.routing_block if isinstance(feedforward, RoutesBlocks) else None
    if block is None:
        return {}
    if metadata is not None and metadata.blocks is not None and metadata.routing_block == block:
        return {"blocks": metadata.blocks}
    batch, length = shape[:2]
    return {"blocks": numbered_blocks(coordinates_of(metadata, batch, length), (batch, length), block)}
