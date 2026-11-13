"""The batch contract every layer shares, the one module below them all that names it: a batch
(`Batch`) and a variables tree (`Variables`), the columns a batch's rows carry beside their fields
(`VALID_ROWS`, `EXAMPLE_ROWS`) and the collection frozen weights stay in (`FROZEN`), how a batch is
filled out with repeats of its own rows (`padded_rows`), how a batch's token fields say which slots
are real and where each token sits (`Coordinates`, `real_slots`), the token ids a field holds
(`token_ids`) and which targets a language-model loss scores (`scored_targets`). Data builds batches
by it, models read them by it, and objectives and the trainer score and place them by it, so none of
them imports another for these."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct, traverse_util

if TYPE_CHECKING:
    from dew._records import JSON
from flax.core import FrozenDict, meta

type Variables = Mapping[str, Any]
"""A Flax variables dict: the `params` collection plus any other collection
the modules keep (`moe`, `batch_stats`, an objective's frozen encoders)."""


type Batch = Mapping[str, Any]
"""One batch of training examples, as the trainer and every objective read it.

It is declared here and imported by `dew.data.dataset`, so a field added on
one side has the same type on the other. The values are not all arrays.
Besides the arrays a step consumes, a batch can hold the prepared text inputs
of a language run, the file paths a video corpus loads lazily, the integer
row counts a packer keeps and the record lists a mixture reads. A narrower
value type would be wrong about what `dew.data` already builds."""


def merge(tree: Variables, overlay: Variables) -> Variables:
    """Return a copy of `tree` with every leaf in `overlay` written over it."""
    merged = dict(tree)
    for name, child in overlay.items():
        held = tree.get(name)
        merged[name] = (merge(held, child)
                        if isinstance(held, Mapping) and isinstance(child, Mapping)
                        else child)
    return merged


FROZEN = "frozen"
"""The name of the collection where a partially trained run keeps its frozen weights.

The optimizer updates the `params` collection and nothing else, so the
leaves that `freeze` keeps in `params` are the ones that train. The frozen
leaves stay in this collection as part of the state, and `thaw` merges the
two before the model reads them."""

PACKED_FROZEN = "packed_frozen"
"""The collection that flags, beside a projection group packed from a tree split for training whose
members are some trained and some frozen (`dew.inference.projections`), which members were frozen: at
each of its packed leaves' paths, one flag a member, in the group's order. With `FROZEN` it is the
training split, which a split of the group restores and `thaw` drops: the model reads neither."""


VALID_ROWS = "valid_rows"
"""The field every evaluation batch holds: one bool per row, True for a real record and False for a repeat.

A split's last batch is padded to full size with repeats of its own rows, and
a process whose share runs out before the others' scores a copy of its last
batch in which every row is a repeat. That way a pass is whole batches on
every process and counts each record once. A loss weights its rows by this
field, and its batch-wide terms are taken over the real rows only. Training
batches never have it.
"""


EXAMPLE_ROWS = "example_rows"
"""A training batch's column of each row's place among its step's records, int32: the order the
records' stream holds them in, whatever process read a row (`dew.training.distributed.marked`). The
trainer's transaction reads it for the examples' draws (`Step.example_index`) and takes it out of the
batch the objective sees."""


EXAMPLES_CONSUMED = "examples_consumed"
"""A training batch's column of the source examples each row is the last of, int32, in a run bounded in
examples (`dew.data.pipeline.CONSUMED`): the batch's sum over its real rows is the examples its dispatch
consumes, which the trainer's step adds to the run's clock (`TrainState.consumed`) and takes out of the
batch the objective sees. A repeat filling out a short batch holds none."""


CLOSES_WINDOW = "closes_window"
"""A training batch's step-wide flag, in a run bounded in examples that counts no steps ahead: whether it
is the last its stream holds, so the update window it ends commits over the microbatches it holds however
few (`dew.training.transaction.Transaction.prepare_attempt`), where a window of a whole update's would be
left open, and the examples it consumed with it, as the stream ends."""

CONSUMED_COUNT = "consumed_count"
"""A training batch's host field of the examples it consumes (`ConsumedCount`), which the trainer counts
as it prepares the batch, so its loop reads the run's progress without waiting on its devices."""


@dataclasses.dataclass(frozen=True)
class ConsumedCount:
    """A training batch's consumed examples as its host counted them before placing it, the sum of its
    `EXAMPLES_CONSUMED` column: an object no device holds (`dew.distributed.batches.split_fields`)."""

    examples: int


PHASE = "data_phase"
"""A training batch's host field of the data phase it was read in (`Phase`), which a phased stream sets on
every batch it delivers (`dew.data.dataset.PhasedStream`), so a step knows its phase on the host whoever
read the batch ahead of it; a batch of a run in no phases holds none, its phase the first."""


DRAWS = "data_draws"
"""A training batch's host field of the draws of corpora phases share that it trains (`Draws`), which a
phase on its own stream reads off each batch and drops before the run sees it
(`dew.data.dataset.StepPhases`), so the next phase continues each shared corpus from them."""


@dataclasses.dataclass(frozen=True)
class Draws:
    """The draws of a phase's shared corpora a batch trains (`DRAWS`), each `(node, member, pull,
    position)` (`dew.data.pipeline.Draw`): an object no device holds."""

    draws: tuple[tuple[str, str, int, JSON], ...]


@dataclasses.dataclass(frozen=True)
class Phase:
    """The data phase a training batch was read in, by its place among the run's phases (`PHASE`): an
    object no device holds (`dew.distributed.batches.split_fields`)."""

    index: int


def padded_rows(rows: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """The rows `count` places hold of `rows` real ones, cycling through them, and which places hold
    a repeat: the rule a batch filled out with its own rows (`dew.data.dataset.FilledBatch`) and a
    request's rows padded to its devices (`dew.nn.inputs.RowPlan`) both follow. Places to fill from
    no rows are refused: there is nothing to repeat."""
    if count and not rows:
        raise ValueError(f"{count} places can't repeat rows of none")
    places = np.arange(count)
    return places % rows, places >= rows


FLAT = "flat"
"""The metadata key of a `Coordinates` field a call's flat layout gives (`LAID_FIELDS`): the flat field's
name."""

VALIDITY_FIELD = "attention_mask"
"""The token field that marks real slots. Absent means every slot is real."""

SEGMENTS_FIELD = "segment_ids"
"""The token field that numbers each slot's document. Absent means one document a row."""

POSITIONS_FIELD = "positions"
"""The token field that holds each slot's place. Absent means each slot's own, counted from its row's
start."""


COORDINATES_FIELD = "coordinates"
"""The token field that holds the tokens' `Coordinates`, which a decoder reads in place of the
validity field, `segment_ids` and `positions`."""

TEXT_KEY = "text"
"""Batch key the token pipeline packs `[B, seq_len + 1]` int32 ids under."""

BLOCKS_FIELD = "block_groups"
"""The token field that numbers the tokens' blocks, which the attention layers that read blocks see whole
(`AttentionMetadata.block_groups`, `prefix_blocks`)."""


def token_id(value: object, *, where: str) -> int:
    """`value` read as one token id: a Python or NumPy integer, not a bool,
    from 0 through int32's largest. Every parser of ids, a dataset's or a
    request's, reads them by this law; `where` names what held the value."""
    if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            or not 0 <= int(value) <= np.iinfo(np.int32).max):
        raise ValueError(f"{where} holds token ids, nonnegative int32 integers, got {value!r}")
    return int(value)


def token_ids(values: object, *, where: str) -> list[int]:
    """`values` read as a nonempty row of token ids by `token_id`'s law: a list, a tuple or a
    one-dimensional NumPy array, as a tokenizer, a dataset's column or a step of Dew's own
    (`dew.data.text.Tokenize`) holds them."""
    if isinstance(values, np.ndarray) and values.ndim == 1:
        values = values.tolist()
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{where} holds a nonempty row of token ids, got {values!r}")
    return [token_id(token, where=where) for token in values]


def document_spans(segments, count: int, *, xp=jnp):
    """Each of a row's `count` documents' first slot and the slot past its last `[B, D]`, `S` and `S`
    for one the row holds none of: a document is a contiguous run of its row's slots (`segments`,
    numbered from 1, 0 on padding)."""
    batch, length = segments.shape
    slots = xp.broadcast_to(xp.arange(length, dtype=np.int32), (batch, length))
    number = xp.maximum(xp.asarray(segments, np.int32) - 1, 0)
    held = segments > 0
    if xp is np:
        first = np.full((batch, count), length, np.int32)
        end = np.zeros((batch, count), np.int32)
        rows = np.broadcast_to(np.arange(batch)[:, None], (batch, length))
        np.minimum.at(first, (rows[held], number[held]), slots[held])
        np.maximum.at(end, (rows[held], number[held]), slots[held] + 1)
    else:
        rows = jnp.arange(batch)[:, None]
        first = jnp.full((batch, count), length, jnp.int32).at[rows, number].min(
            jnp.where(held, slots, length))
        end = jnp.zeros((batch, count), jnp.int32).at[rows, number].max(
            jnp.where(held, slots + 1, 0))
    return first, xp.where(end > 0, end, length)


def real_bounds(segments, real, count: int, *, xp=jnp):
    """Each of a row's `count` documents' first and last real slot `[B, D]` (`document_spans` over
    `segments` kept where `real` `[B, S]` marks a token), `S` and -1 for one holding no real token."""
    held = xp.asarray(real, bool) & (xp.asarray(segments) > 0)
    first, end = document_spans(xp.where(held, xp.asarray(segments, np.int32), 0), count, xp=xp)
    return first, xp.where(first < held.shape[1], end - 1, -1)


@struct.dataclass
class Forks:
    """A row's documents of which some continue another: a forest of shared prefixes. `parents` `[B, D]`
    is the document each of a row's documents continues, by its number in the row's `segments`
    (`Coordinates.segments`, numbered from 1 in the row's order, 0 on padding), 0 for one that continues
    the row's start: nothing in an uncached call, the row's cache in a cached one. `depth`, static, is the
    longest chain of parents in the batch, 0 where no document continues another.

    A document that continues another reads its whole chain of ancestors as its own beginning, and no other
    document: attention their keys (`runs`), a recurrent layer the state its parent's tokens leave and a
    conv its parent's last tokens (`dew.nn.linear.forked_scan`), each token at the place its chain gives it
    (`positions`). So a document computes what its own row of its ancestors and itself computes, and a
    prefix that several continue is computed once: a request's questions over its state, a prompt's
    completions, a speculative tree's candidates (each a document of one token). `dew.batch.ForkedRows`
    packs rows that share prefixes into such documents. A document's parent is laid before it in its row,
    so one walk over the row in order computes every parent before its children.

    `held`, static, is the most of a row's documents that another continues, which a recurrent layer
    keeps the final memories of (`dew.nn.linear.forked_scan`): its layout's own bound, one value a run (a
    request's shared part a window's row, a pair's prompt, a tree's inner nodes), never the documents bound;
    None where it states none, which a recurrent layer refuses."""

    parents: jax.Array | np.ndarray
    depth: int = struct.field(pytree_node=False, default=1)
    joint: bool = struct.field(pytree_node=False, default=False)
    """Whether each tree's documents are one row's, read together, its dropout drawn once (a decision
    row's questions with siblings, `QuestionLayout.siblings`), rather than rows a shared prefix stands in
    for, each of which would draw its own (`dew.nn.backbones.decoder_refusals.forked`)."""
    held: int | None = struct.field(pytree_node=False, default=None)

    @property
    def count(self) -> int:
        """The documents a row may number, `D`."""
        return self.parents.shape[1]

    def spans(self, segments, *, xp=jnp):
        """Each document's first slot and the slot past its last `[B, D]` (`document_spans`)."""
        return document_spans(segments, self.count, xp=xp)

    def ancestors(self, *, xp=jnp):
        """Each document's ancestors `[B, D, depth]`: its parent, its parent's parent and so on, by number,
        0 past its chain."""
        parents = xp.asarray(self.parents, np.int32)
        chain, above = [], parents
        for _ in range(self.depth):
            chain.append(above)
            above = xp.where(above > 0, xp.take_along_axis(parents, xp.maximum(above - 1, 0), axis=1), 0)
        if not chain:
            return xp.zeros((*parents.shape, 0), np.int32)
        return xp.stack(chain, axis=-1)

    def trees(self, segments, *, xp=jnp):
        """Each token's tree `[B, S]`, from 1 in the order the trees' roots stand in the row, 0 on padding:
        the root its document's chain of ancestors ends in. A call that reads a context a tree reads each
        token's by it (`CrossContext.queries`): a row over one state, its questions continuing its state,
        is one tree."""
        segments = xp.asarray(segments, np.int32)
        if self.depth == 0:
            return segments
        ancestors = self.ancestors(xp=xp)
        levels = xp.sum(ancestors > 0, axis=-1, dtype=np.int32)
        deepest = xp.take_along_axis(ancestors, xp.maximum(levels - 1, 0)[..., None], axis=-1)[..., 0]
        roots = xp.where(levels > 0, deepest, xp.arange(1, self.count + 1, dtype=np.int32)[None])
        # Each root's rank among the row's roots, read at each document's root, then at each token.
        ranked = xp.take_along_axis(xp.cumsum(levels == 0, axis=1, dtype=np.int32), roots - 1, axis=1)
        return xp.where(segments > 0, xp.take_along_axis(ranked, xp.maximum(segments - 1, 0), axis=1), 0)

    def levels(self, *, xp=jnp):
        """Each document's level `[B, D]`: how many ancestors it continues, 0 for a root."""
        return xp.sum(self.ancestors(xp=xp) > 0, axis=-1, dtype=np.int32)

    def offsets(self, segments, valid=None, *, xp=jnp):
        """Each document's place before its first token `[B, D]`: its ancestors' lengths summed, each its
        real tokens (`valid` `[B, S]`, every one where None)."""
        segments = xp.asarray(segments, np.int32)
        batch = segments.shape[0]
        real = (segments > 0) if valid is None else (segments > 0) & xp.asarray(valid, bool)
        number = xp.maximum(segments - 1, 0)
        if xp is np:
            lengths = np.zeros((batch, self.count), np.int32)
            rows = np.broadcast_to(np.arange(batch)[:, None], segments.shape)
            np.add.at(lengths, (rows[real], number[real]), 1)
        else:
            lengths = jnp.zeros((batch, self.count), jnp.int32).at[jnp.arange(batch)[:, None], number].add(
                real.astype(jnp.int32))
        chain = self.ancestors(xp=xp)
        taken = xp.take_along_axis(lengths[:, :, None], xp.maximum(chain - 1, 0).reshape(
            chain.shape[0], -1, 1), axis=1).reshape(chain.shape)
        return xp.sum(xp.where(chain > 0, taken, 0), axis=-1, dtype=np.int32)

    def positions(self, segments, base=None, valid=None, *, xp=jnp):
        """Each token's place `[B, S]`: `base` `[B]` (0 where None: an uncached call; a cached call's the
        place of the row's next token), its document's ancestors' real tokens (`valid` `[B, S]`, every one
        where None), and the real tokens before it in its document, as its own row of its ancestors and
        itself counts it, a token a row does not hold taking no place; 0 on padding."""
        segments = xp.asarray(segments, np.int32)
        batch, length = segments.shape
        real = (segments > 0) if valid is None else (segments > 0) & xp.asarray(valid, bool)
        first, _ = self.spans(segments, xp=xp)
        number = xp.maximum(segments - 1, 0)
        before = xp.take_along_axis(self.offsets(segments, valid, xp=xp), number, axis=1)
        counted = xp.cumsum(real.astype(np.int32), axis=1) - real.astype(np.int32)
        start = xp.minimum(xp.take_along_axis(first, number, axis=1), length - 1)
        opened = xp.take_along_axis(counted, start, axis=1)
        shift = 0 if base is None else xp.asarray(base, np.int32).reshape(batch, 1)
        places = before + counted - opened + shift
        return xp.where(segments > 0, places, 0).astype(np.int32)

    def predecessors(self, segments, *, xp=jnp):
        """The slot whose output predicts each token `[B, S]`: the one before it in its document, its
        parent's last for a document's first token, -1 for a root's first token and for padding, which no
        token predicts. A next-token loss scores each token from its predecessor's state."""
        segments = xp.asarray(segments, np.int32)
        batch, length = segments.shape
        first, end = self.spans(segments, xp=xp)
        number = xp.maximum(segments - 1, 0)
        slots = xp.broadcast_to(xp.arange(length, dtype=np.int32), (batch, length))
        opening = slots == xp.take_along_axis(first, number, axis=1)
        parent = xp.take_along_axis(xp.asarray(self.parents, np.int32), number, axis=1)
        last = xp.take_along_axis(end, xp.maximum(parent - 1, 0), axis=1) - 1
        before = xp.where(opening, xp.where(parent > 0, last, -1), slots - 1)
        return xp.where(segments > 0, before, -1).astype(np.int32)

    def runs(self, segments, reach=None):
        """Each token's key runs `[B, S, depth + 1, 2]` over the call's own slots
        (`dew.nn.inputs.AttentionMetadata.key_runs`): its ancestors whole, its root's first, then its own
        document up to itself, or up to `reach` `[B, S]` (the slot past the last it reads, a bidirectional
        group's end) within its document, empty runs `(0, 0)` past a shallower chain; a
        padding slot reads itself alone. A cached call's attention places them after the row's cache
        (`fork_runs`)."""
        segments = jnp.asarray(segments, jnp.int32)
        batch, length = segments.shape
        first, end = self.spans(segments)
        number = jnp.maximum(segments - 1, 0)
        slots = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
        # Each token's document's ancestors `[B, S, depth]`, nearest first.
        chain = jnp.take_along_axis(self.ancestors(), number[:, :, None], axis=1)
        level = jnp.sum(chain > 0, axis=-1)
        reached = slots + 1 if reach is None else jnp.clip(
            jnp.asarray(reach, jnp.int32), slots + 1, jnp.take_along_axis(end, number, axis=1))
        runs = []
        for index in range(self.depth + 1):
            # Run `index` is the ancestor `level - index` above the token's document, its own at `level`.
            distance = level - index
            ancestor = jnp.take_along_axis(jnp.pad(chain, ((0, 0), (0, 0), (1, 0))),
                                           jnp.clip(distance, 0, self.depth)[..., None], axis=-1)[..., 0]
            lo = jnp.take_along_axis(first, jnp.maximum(ancestor - 1, 0), axis=1)
            hi = jnp.take_along_axis(end, jnp.maximum(ancestor - 1, 0), axis=1)
            own = distance == 0
            lo = jnp.where(own, jnp.take_along_axis(first, number, axis=1), lo)
            hi = jnp.where(own, reached, hi)
            kept = (distance >= 0) & (segments > 0)
            pad = segments == 0
            lo = jnp.where(kept, lo, jnp.where(pad & (index == 0), slots, 0))
            hi = jnp.where(kept, hi, jnp.where(pad & (index == 0), slots + 1, 0))
            runs.append(jnp.stack([lo, hi], axis=-1))
        return jnp.stack(runs, axis=2).astype(jnp.int32)

    @staticmethod
    def held_of(parents) -> int:
        """The most of a row's documents another continues, of `parents` `[B, D]` (host arrays): the
        `held` a layout packed ahead states, its plan's whole."""
        parents = np.asarray(parents, np.int64)
        return max((len(set(row[row > 0].tolist())) for row in parents), default=0)

    @staticmethod
    def depth_of(parents) -> int:
        """The longest chain of `parents` `[B, D]` (host arrays), refusing a document that continues
        itself through its chain: the static `depth` a `Forks` states."""
        parents = np.asarray(parents, np.int64)
        if parents.size and (parents.min() < 0 or parents.max() > parents.shape[1]):
            raise ValueError(f"a document continues another of its row's {parents.shape[1]} by number, or 0 "
                             f"for none, got {parents.min()} to {parents.max()}")
        depth, above = 0, parents
        while np.any(above > 0):
            depth += 1
            if depth > parents.shape[1]:
                raise ValueError("a document's chain of parents returns to itself")
            above = np.where(above > 0, np.take_along_axis(parents, np.maximum(above - 1, 0), axis=1), 0)
        return depth


@struct.dataclass
class Coordinates:
    """Where each token of a batch sits, built once (`coordinates`) and read by every token mixer.

    `positions` `[B, S]` is each real token's logical place in its document,
    counted over real tokens: a packed batch's or the caller's own, None
    where none is given, each token's place then its rank among its row's
    real tokens (`places`) or, in a cached call, the slot its cache writes it
    at. Rotation and NoPE's temperature read the logical places; windows,
    chunks and ALiBi count each document's real tokens from its first place
    (`counted_places`), the logical places wherever
    those count real tokens; a cache addresses keys by slot. `segments` numbers each
    row's packed documents from 1, 0 on the padding slots after them, which
    run as a document of their own, so no layer carries a real document's
    state or history into them or back; `valid` is the caller's mask of real
    tokens; `starts` marks each document's first real token but a row's
    first, which continues whatever state the row holds. Each is None where
    it holds nothing (one document a row, every token real, no start), which
    a layer's fast path reads as a static fact. `firsts` `[B, D]` are each
    row's documents' first slots in order, the row's first slot among them,
    `S` past its last (`document_firsts`): the packed documents' coordinates
    every varlen kernel reads (`dew.nn.inputs.cu_seqlens`), built with the
    coordinates, never by a layer: by one builder wherever coordinates are
    made, a host's batch and a traced step's alike, or their slots sliced or
    rotated (`along_tokens`; row 1923); each layer's own scan over the
    batch's flattened tokens, which a mesh splits, ran one token a loop
    iteration on a 2xH100 run (row 1897). `max_segments` bounds the documents a row holds
    (the packer's static plan, else the row length) and `right_padded` says
    every row's real tokens come first. `finals` are a row's trailing slots that hold targets alone,
    each a document's final id, which no input reads (`predecessors`).
    """

    positions: jax.Array | None = struct.field(default=None, metadata={FLAT: POSITIONS_FIELD})
    segments: jax.Array | None = struct.field(default=None, metadata={FLAT: SEGMENTS_FIELD})
    valid: jax.Array | None = struct.field(default=None, metadata={FLAT: VALIDITY_FIELD})
    starts: jax.Array | None = None
    firsts: jax.Array | None = None
    forks: Forks | None = None
    """The documents within its rows that continue another (`Forks`), None where none does."""
    max_segments: int = struct.field(pytree_node=False, default=1)
    right_padded: bool = struct.field(pytree_node=False, default=False)
    finals: int = struct.field(pytree_node=False, default=1)
    """The trailing slots of a row that hold targets alone: 1, a row's last id, where its ids shift by
    one; past one, a tail in which each document's final id stands, its segment and place its
    document's, so a document's final id is a target and never an input (a preference pair's halves,
    `dew.data.preferences`)."""

    TOKEN_ALIGNED: ClassVar[tuple[str, ...]] = ("positions", "segments", "valid", "starts")
    """The fields one entry a token, `[B, S]`; `firsts` is one a document, `[B, D]`, its entries slots."""

    def predecessors(self, *, xp=jnp):
        """The slot whose output predicts each token `[B, S]`, -1 where none does (a root document's first
        token, padding): within the inputs (all but the `finals`), the one before it in its document, a
        document's first token its parent's last (`Forks.predecessors`); in the tail, its document's last
        input, or its parent's last where the document has none."""
        assert self.segments is not None, "predecessors read the rows' documents"
        segments = xp.asarray(self.segments, np.int32)
        batch, length = segments.shape
        inputs = length - self.finals
        body, tail = segments[:, :inputs], segments[:, inputs:]
        count = self.max_segments if self.forks is None else self.forks.count
        first, end = document_spans(body, count, xp=xp)
        if self.forks is not None:
            before = self.forks.predecessors(body, xp=xp)
            parents = xp.asarray(self.forks.parents, np.int32)
        else:
            slots = xp.broadcast_to(xp.arange(inputs, dtype=np.int32), (batch, inputs))
            opening = slots == xp.take_along_axis(first, xp.maximum(body - 1, 0), axis=1)
            before = xp.where((body > 0) & ~opening, slots - 1, -1).astype(np.int32)
            parents = xp.zeros((batch, count), np.int32)
        number = xp.maximum(tail - 1, 0)
        last = xp.take_along_axis(end, number, axis=1) - 1
        held = xp.take_along_axis(end, number, axis=1) > xp.take_along_axis(first, number, axis=1)
        parent = xp.take_along_axis(parents, number, axis=1)
        above = xp.where(parent > 0, xp.take_along_axis(end, xp.maximum(parent - 1, 0), axis=1) - 1, -1)
        after = xp.where(tail > 0, xp.where(held, last, above), -1).astype(np.int32)
        return xp.concatenate([before, after], axis=1)

    def tokens(self) -> dict[str, jax.Array]:
        """The token-aligned fields it holds (`TOKEN_ALIGNED`), by name: what a check of a batch's token
        layout reads."""
        return {name: getattr(self, name) for name in self.TOKEN_ALIGNED if getattr(self, name) is not None}

    def along_tokens(self, transform: Callable[[jax.Array], jax.Array], *,
                     prefix: bool = False) -> Coordinates:
        """`transform` applied along the tokens to each token-aligned field (`TOKEN_ALIGNED`): a slice or a
        rotation of the slots. The documents' firsts, which index the slots, are built again from the
        moved starts (`document_firsts`), as every `Coordinates` builds them. `prefix` says the transform
        keeps each row's first slots where they stand (a slice from slot 0), which keeps its forks'
        documents in place, each at its places."""
        if self.forks is not None and not prefix:
            raise ValueError("documents that continue another are read whole in their rows; a call that "
                             "slices or rotates the rows' slots runs none")
        moved = replace(self, **{name: transform(value) for name, value in self.tokens().items()})
        if moved.starts is None:
            return moved
        return replace(moved, firsts=document_firsts(moved.starts, moved.max_segments + 1))

    def places(self) -> jax.Array | None:
        """Each token's logical place in an uncached call: the given `positions`, else its rank among
        its row's real tokens `[B, S]` (a padding slot's the rank before it, 0 before any); None
        where every token is real and none is given, each token's place its index. A cached call's
        places are the slots its cache writes them at where none are given."""
        if self.positions is not None or self.valid is None:
            return self.positions
        return real_places(self.valid)


SCAN_BLOCKS = 64
"""How many blocks a running sum or maximum along a token axis cuts the axis into (`prefix_sum`): its scans
run inside each block, so a mesh that splits the axis into up to this many shards keeps every scan on one
shard, and the blocks' carries are one reduction over the blocks before each."""


def prefix_sum(x, axis: int = 1, *, dtype=None, xp=jnp):
    """`x`'s running sum along `axis`, `cumsum`'s, in `dtype` (as `cumsum` takes it where None: a bool's
    or a narrow integer's in the default integer), as a program a mesh that splits the axis keeps
    parallel: XLA's SPMD partitioner turns one scan along a split axis into a scan with a live carry,
    which the GPU's associative-scan rewrite refuses, and runs it as a loop of one iteration a token (a
    2xH100 training step ran ten times slower, ledger 1905). Here each of `SCAN_BLOCKS` blocks scans its
    own slots along an axis no mesh splits, and each block adds the sum of the blocks before it, in the
    result's dtype throughout, so an integer sum wraps as `cumsum`'s does. Exact for integers; a float's
    sum is reassociated by blocks. `xp` numpy computes it on the host."""
    if xp is np:
        return np.cumsum(x, axis=axis, dtype=dtype)
    x = jnp.asarray(x)
    # The dtype `cumsum` sums `x` in, given `dtype` or not.
    one = jax.ShapeDtypeStruct((1,), x.dtype)
    summed = jax.eval_shape(lambda first: jnp.cumsum(first, dtype=dtype), one).dtype
    return _blocked_scan(x.astype(summed), axis, lambda part, at: jnp.cumsum(part, axis=at, dtype=summed),
                         lambda part, axis: jnp.sum(part, axis=axis, dtype=summed), jnp.add, 0)


def prefix_max(x, axis: int = 1, *, xp=jnp):
    """`x`'s running maximum along `axis`, `cummax`'s, as a program a mesh that splits the axis keeps
    parallel (`prefix_sum`)."""
    if xp is np:
        return np.maximum.accumulate(x, axis=axis)
    x = jnp.asarray(x)
    return _blocked_scan(x, axis, lambda part, at: jax.lax.cummax(part, axis=at), jnp.max, jnp.maximum,
                         _extreme(x.dtype, low=True))


def prefix_min(x, axis: int = 1, *, reverse: bool = False, xp=jnp):
    """`x`'s running minimum along `axis`, from its end where `reverse`, as a program a mesh that splits
    the axis keeps parallel (`prefix_sum`)."""
    flipped = xp.flip(xp.asarray(x), axis) if reverse else xp.asarray(x)
    if xp is np:
        running = np.minimum.accumulate(flipped, axis=axis)
    else:
        running = _blocked_scan(flipped, axis, lambda part, at: jax.lax.cummin(part, axis=at), jnp.min,
                                jnp.minimum, _extreme(flipped.dtype, low=False))
    return xp.flip(running, axis) if reverse else running


def _extreme(dtype, *, low: bool):
    """The identity of a running maximum (`low`) or minimum over `dtype`: a float's infinity, an integer's
    bound, a bool's False or True."""
    if jnp.issubdtype(dtype, jnp.floating):
        return -jnp.inf if low else jnp.inf
    if dtype == jnp.bool_:
        return not low
    bounds = jnp.iinfo(dtype)
    return bounds.min if low else bounds.max


def _blocked_scan(x, axis: int, scan, reduce, combine, identity):
    """`scan` along `axis` of `x` cut into `SCAN_BLOCKS` blocks, each block's scan combined with the
    `reduce` of the blocks' totals before it, the axis padded with `identity` to whole blocks. An empty
    axis scans to itself."""
    moved = jnp.moveaxis(x, axis, -1)
    *lead, length = moved.shape
    if length == 0:
        return x
    size = -(-length // SCAN_BLOCKS)
    widths = [(0, 0)] * len(lead) + [(0, size * SCAN_BLOCKS - length)]
    padded = jnp.pad(moved, widths, constant_values=jnp.asarray(identity, x.dtype))
    inner = scan(padded.reshape(*lead, SCAN_BLOCKS, size), len(lead) + 1)
    earlier = jnp.tril(jnp.ones((SCAN_BLOCKS, SCAN_BLOCKS), bool), -1)
    carried = reduce(jnp.where(earlier, inner[..., None, :, -1], jnp.asarray(identity, x.dtype)), axis=-1)
    joined = combine(inner, carried[..., None]).astype(x.dtype)
    joined = joined.reshape(*lead, size * SCAN_BLOCKS)[..., :length]
    return jnp.moveaxis(joined, -1, axis)


def real_places(valid: jax.Array) -> jax.Array:
    """Each slot's rank among its row's real tokens under `valid` `[B, S]`: a real token's place, a
    padding slot's the rank before it, 0 before any (`Coordinates.places` where no positions are given)."""
    return jnp.maximum(prefix_sum(valid, dtype=jnp.int32) - 1, 0)


def coordinates(shape: tuple[int, ...] | None = None, *, segments=None, valid=None, positions=None,
                max_segments: int | None = None, right_padded: bool = False, xp=jnp) -> Coordinates:
    """The `Coordinates` of `[batch, length]` tokens from what the batch carries: packed `segments`
    (one document a row where None), a `valid` mask (every token real where None), explicit
    `positions` (with `segments`, each document's real tokens counted from 0 where None; without,
    None, each token's place its row index or cache slot) and the packer's `max_segments` (the row
    length where None). `shape` is the first given field's where None. `xp` is the array module the
    fields are built with: jax.numpy in a traced call, numpy where a data reader builds a batch's on
    the host."""
    if shape is None:
        given = [x for x in (segments, valid, positions) if x is not None]
        if not given:
            raise ValueError("token coordinates take their shape from a field, or `shape`")
        batch, length = jnp.shape(given[0])[:2]
        shape = (batch, length)
    length = shape[1]
    segments = None if segments is None else xp.asarray(segments, np.int32)
    valid = None if valid is None else xp.asarray(valid, bool)
    for name, field in (("segments", segments), ("valid", valid)):
        if field is not None and field.shape != shape:
            raise ValueError(f"token coordinates are {shape}, got {name} {field.shape}")
    starts = None if segments is None else document_starts(
        segments, xp.ones(shape, bool) if valid is None else valid, xp=xp)
    if positions is None and segments is not None:
        real = xp.ones(shape, bool) if valid is None else valid
        real = real & (segments != 0)
        positions = xp.where(real, document_ranks(real, starts, xp=xp), 0)
    positions = None if positions is None else xp.asarray(positions, np.int32)
    max_segments = (1 if segments is None else length) if max_segments is None else max_segments
    # The documents' firsts, by the one builder a host's batch and a traced step's alike take.
    firsts = None if starts is None else document_firsts(starts, max_segments + 1, xp=xp)
    return Coordinates(positions, segments, valid, starts, firsts, max_segments=max_segments,
                       right_padded=right_padded)


LAID_FIELDS: Mapping[str, str] = MappingProxyType(
    {field.name: field.metadata[FLAT] for field in dataclasses.fields(Coordinates) if FLAT in field.metadata})
"""Each `Coordinates` field a call's flat layout gives, by the flat field that gives it (`FLAT`)."""

TOKEN_LAYOUT_FIELDS = tuple(LAID_FIELDS.values())
"""The flat token fields a call's layout is given in where it gives no `Coordinates`
(`token_coordinates`): its places, its documents and its validity."""


def token_coordinates(shape: tuple[int, ...], given: Coordinates | None = None, *, segments=None,
                      valid=None, positions=None, right_padded: bool = False) -> Coordinates | None:
    """A call's token layout as one `Coordinates`, the one conversion of its fields: `given` as it is,
    else built (`coordinates`) from the flat fields `segment_ids` (`segments`), `attention_mask`
    (`valid`) and `positions`, None where the call gives none. A call giving both is refused: its
    `Coordinates` would silently drop its flat fields (ledger 1122)."""
    given_fields = {SEGMENTS_FIELD: segments, VALIDITY_FIELD: valid, POSITIONS_FIELD: positions}
    flat = ([name for name in TOKEN_LAYOUT_FIELDS if given_fields[name] is not None]
            + (["right_padded"] if right_padded else []))
    if given is not None:
        if flat:
            raise ValueError(f"a call's tokens are laid out by their coordinates or by their fields, not "
                             f"both: coordinates with {', '.join(flat)}")
        return given
    if not flat:
        return None
    return coordinates(shape, segments=segments, valid=valid, positions=positions, right_padded=right_padded)


def field_coordinates(fields: Mapping[str, jax.Array | np.ndarray | Coordinates],
                      shape: tuple[int, ...]) -> Coordinates | None:
    """A row's token fields' layout as one `Coordinates`, the one conversion of a field dict's flat
    layout (`token_coordinates`): their `Coordinates` as they are, else built from their flat
    `segment_ids`, `attention_mask` and `positions`, None where they carry neither; both refused."""
    return token_coordinates(shape, laid_coordinates(fields), segments=flat_field(fields, SEGMENTS_FIELD),
                             valid=flat_field(fields, VALIDITY_FIELD),
                             positions=flat_field(fields, POSITIONS_FIELD))


def document_firsts(starts, documents: int, *, xp=jnp):
    """Each row's first `documents` documents' first slots in order `[B, min(documents, S)]`, the row's
    first slot and each of `starts` `[B, S]`, `S` past its last: a sort of each row's slots, which a mesh
    splitting the rows sorts on each device (`Coordinates.firsts`)."""
    rows, length = starts.shape
    if xp is np:
        begun = np.asarray(starts, bool).copy()
        begun[:, 0] = True
        slots = np.broadcast_to(np.arange(length, dtype=np.int32), (rows, length))
    else:
        # Placed as the starts are (an Explicit mesh's rows split), each row sorted on its own device.
        begun = jnp.asarray(starts, bool).at[:, 0].set(True)
        placed = jax.typeof(begun).sharding
        slots = jax.lax.broadcasted_iota(jnp.int32, (rows, length), 1, out_sharding=placed)
    return xp.sort(xp.where(begun, slots, length), axis=1)[:, :min(documents, length)].astype(np.int32)


def document_ranks(real, starts, *, xp=jnp):
    """Each slot's count of the real tokens before it in its document `[B, S]`, `starts` marking each
    document's first real token but a row's first, a padding slot taking the count before it (0
    before any)."""
    rank = prefix_sum(real, dtype=np.int32, xp=xp) - 1
    first = prefix_max(xp.where(starts, rank, 0), xp=xp)
    return xp.maximum(rank - first, 0)


def valid_order(valid: jax.Array, *, xp=jnp) -> tuple[jax.Array, jax.Array]:
    """Each row's real slots first, in order, then its padding slots: the
    physical slot behind each compact column `[rows, slots]`, and how many
    slots of each row are real `[rows]`. A stable sort of the validity, so a
    row's tokens keep their order; a caller says what its columns past the
    count hold (decoding's -1 history tokens, a convolution's zero column). `xp` is the array module."""
    valid = xp.asarray(valid, bool)
    return xp.argsort(~valid, axis=-1, stable=True), xp.sum(valid, axis=-1, dtype=np.int32)


def document_starts(segments, valid, *, xp=jnp):
    """`[B, S]`: each real token whose segment differs from the previous real token's in its row, a
    row's first real token continuing what precedes the row and a padding slot starting nothing."""
    order, _ = valid_order(valid, xp=xp)
    rank = prefix_sum(valid, dtype=np.int32, xp=xp) - 1
    compact = xp.take_along_axis(segments, order, axis=1)
    changed = xp.concatenate([xp.zeros_like(compact[:, :1], bool), compact[:, 1:] != compact[:, :-1]],
                             axis=1)
    return valid & xp.take_along_axis(changed, xp.maximum(rank, 0), axis=1)


def prefix_blocks(segments, places, prefix: int, *, xp=jnp):
    """The block each token of a row is under a prefix run of `prefix` tokens: the first `prefix` places of
    each document, numbered by the document's segment id, -1 for the rest and for padding (segment 0). The
    `[B, S]` `segments` number a row's documents from 1, and `places` is each token's place in its document.
    The data stage `Blocks` and a served prompt (`prompt_blocks`) read this one rule."""
    return xp.where((segments > 0) & (places < prefix), segments, -1).astype(xp.int32)


def role_blocks(segments, roles, assistant: int, *, starts=None, xp=jnp):
    """Each document's tokens before its first `assistant` token as one block, numbered by the document's
    segment id, -1 for the rest and for padding (segment 0): a chat row's prompt, which a prefix run of
    "prompt" reads whole. `segments` and `roles` are `[B, S]`. A document opens at each of `starts`, where a
    caller's documents run on across holes (`dew.batch.document_starts`), else wherever `segments`
    changes. The data stage `Blocks` and the loss of a batch without blocks (`LMObjective`) read this rule."""
    places = xp.broadcast_to(xp.arange(segments.shape[-1], dtype=xp.int32), segments.shape)
    opens = starts if starts is not None else xp.concatenate(
        [xp.ones_like(segments[..., :1], bool), segments[..., 1:] != segments[..., :-1]], axis=-1)
    last = segments.ndim - 1
    start = prefix_max(xp.where(opens, places, 0), last, xp=xp)
    spoken = prefix_max(xp.where((segments > 0) & (roles == assistant), places, -1), last, xp=xp) >= start
    return xp.where((segments > 0) & ~spoken, segments, -1).astype(xp.int32)


def prefix_groups(documents, real, prefix: int | Literal["prompt"], *, positions=None, roles=None,
                  assistant: int = -1, xp=jnp):
    """The blocks a prefix run of `prefix` reads in rows `[B, S]`: each document's first `prefix` places
    (`prefix_blocks`), or under "prompt" its tokens before its first `assistant` token by the rows'
    `roles` (`role_blocks`). `documents` number each row's documents and `real` marks its real tokens: a
    document runs on across a hole in its validity, its starts and its places, where no `positions` give
    them, counted over its real tokens as the decoder's coordinates count them (`document_starts`,
    `coordinates`). The data stage `Blocks` and the loss of a batch without blocks (`LMObjective`) read
    this one rule."""
    documents = xp.asarray(documents).astype(xp.int32)
    segments = xp.where(real, documents, 0).astype(xp.int32)
    if prefix == "prompt":
        starts = document_starts(documents, real, xp=xp)
        return role_blocks(segments, xp.asarray(roles), assistant, starts=starts, xp=xp)
    if positions is None:
        places = coordinates(segments.shape, segments=documents, valid=real, xp=xp).positions
    else:
        places = xp.broadcast_to(xp.asarray(positions), segments.shape)
    return prefix_blocks(segments, places, prefix, xp=xp)


def flat_field(fields: Mapping[str, jax.Array | np.ndarray | Coordinates], name: str
               ) -> jax.Array | np.ndarray | None:
    """The token field `name`, an array, or None where the fields hold none; a `Coordinates` under a
    flat field's name is refused."""
    value = fields.get(name)
    if isinstance(value, Coordinates):
        raise TypeError(f"token field {name!r} holds an array, not Coordinates")
    return value


def laid_coordinates(fields: Mapping[str, jax.Array | np.ndarray | Coordinates]) -> Coordinates | None:
    """The `Coordinates` a row's token fields carry, or None where they carry flat layout fields or
    none; fields carrying both are refused, as a call giving both is (`token_coordinates`), since one
    would silently win over the other, and so is anything but `Coordinates` under their key (ledger
    1122)."""
    laid = fields.get(COORDINATES_FIELD)
    if laid is None:
        return None
    if not isinstance(laid, Coordinates):
        raise TypeError(f"token field {COORDINATES_FIELD!r} holds Coordinates, not {type(laid).__name__}")
    flat = [name for name in TOKEN_LAYOUT_FIELDS if fields.get(name) is not None]
    if flat:
        raise ValueError(f"token fields are laid out by their coordinates or by their fields, not both: "
                         f"coordinates with {', '.join(flat)}")
    return laid


def token_validity(fields: Mapping[str, jax.Array | np.ndarray | Coordinates]):
    """The validity `[B, S]` a row's token fields hold, as given: their `Coordinates`' where the fields
    carry them (`laid_coordinates`), else the validity field; None where neither holds one."""
    laid = laid_coordinates(fields)
    return laid.valid if laid is not None else flat_field(fields, VALIDITY_FIELD)


def token_layout(fields: Mapping[str, jax.Array | np.ndarray | Coordinates], *, xp=jnp):
    """The validity `[B, S]` and documents `[B, S]` of a row's token fields as the decoder reads them:
    their `Coordinates`' where the fields carry them (`laid_coordinates`), else the validity field and
    `segment_ids`. Each is None where nothing gives it (every slot real, one document a row). `xp` is
    the array module."""
    laid = laid_coordinates(fields)
    valid = laid.valid if laid is not None else flat_field(fields, VALIDITY_FIELD)
    segments = laid.segments if laid is not None else flat_field(fields, SEGMENTS_FIELD)
    return (None if valid is None else xp.asarray(valid, bool),
            None if segments is None else xp.asarray(segments))


def with_validity(fields: Mapping[str, jax.Array | np.ndarray | Coordinates],
                  valid: jax.Array | np.ndarray | None) -> dict:
    """`fields` holding `valid` as their validity where `token_validity` reads it, None removing it, their
    layout one `Coordinates` built again from it (`coordinates`, its documents' starts and firsts
    following it), whether the fields carried coordinates or flat layout fields (ledger 1122); none where
    nothing is left to lay out (one document a row, every token real, at its slot), as fields that
    carry no layout. A field dict's layout so takes one shape for one set of given fields, which a
    pool's processes agree on (`dew.nn.inputs.agreed_validity`). Arrays stay on the host where `valid`
    and the fields are numpy."""
    laid = laid_coordinates(fields)
    segments = laid.segments if laid is not None else flat_field(fields, SEGMENTS_FIELD)
    positions = laid.positions if laid is not None else flat_field(fields, POSITIONS_FIELD)
    rest = {name: value for name, value in fields.items()
            if name not in (*TOKEN_LAYOUT_FIELDS, COORDINATES_FIELD)}
    given = [field for field in (valid, segments, positions) if field is not None]
    if not given:
        return rest
    host = all(isinstance(field, np.ndarray) for field in given)
    rebuilt = coordinates(tuple(np.shape(given[0])[:2]), segments=segments, valid=valid, positions=positions,
                          max_segments=None if laid is None else laid.max_segments,
                          right_padded=laid is not None and laid.right_padded, xp=np if host else jnp)
    if laid is not None and laid.forks is not None:
        rebuilt = replace(rebuilt, forks=laid.forks)
    return {**rest, COORDINATES_FIELD: rebuilt}


def token_positions(fields: Mapping[str, jax.Array | np.ndarray | Coordinates], *, xp=jnp):
    """The logical positions `[B, S]` of a row's token fields as the decoder reads them: their
    `Coordinates`' where the fields carry them (`laid_coordinates`), else the `positions` field; None
    where neither gives any. `xp` is the array module."""
    laid = laid_coordinates(fields)
    positions = laid.positions if laid is not None else flat_field(fields, POSITIONS_FIELD)
    return None if positions is None else xp.asarray(positions)


def real_slots(fields: Mapping[str, jax.Array | np.ndarray | Coordinates], shape: tuple[int, ...], *, xp=jnp):
    """Which of a request's `shape` `[B, S]` slots hold a real token: those its
    validity marks, every one where it has none, less the padding a packed
    row numbers 0 in its documents (`token_layout`). Samplers select, place
    and continue a row by these, and losses score and count them, so a row
    padded either way reads as the row alone; a caller's mask is narrowed,
    never replaced. `xp` is the array module."""
    valid, segments = token_layout(fields, xp=xp)
    real = xp.ones(shape, bool) if valid is None else valid
    return real if segments is None else real & (segments != 0)


def shifted_span(shape: tuple[int, ...], shift: int, *, valid=None, segments=None, xp=jnp):
    """Which `[B, S - shift]` source positions can reach the token `shift` places later: every
    intervening token is valid, including both endpoints, and their document IDs match. Padding
    IDs and target roles belong to the caller; `segments` describes identity, not validity.
    `xp` is NumPy for host geometry and JAX NumPy for model/loss execution."""
    batch, length = shape
    width = length - shift
    admitted = xp.ones((batch, width), bool)
    if valid is not None:
        for offset in range(shift + 1):
            admitted = admitted & xp.asarray(valid[:, offset:offset + width], bool)
    if segments is not None:
        admitted = admitted & (segments[:, :width] == segments[:, shift:shift + width])
    return admitted


def scored_targets(xp, tokens, pad_id: int | None, attention_mask, segment_ids, depth: int = 0,
                   roles=None, role: int | None = None, blocks=None, predecessors=None):
    """Whether each target the loss scores at prediction `depth` counts, as
    `[B, S - 1 - depth]` booleans, in `xp` (numpy for the host's FLOP
    geometry and the data's target counts, jax.numpy for the loss): the
    state at p scores the token `depth + 1` later, which counts when it is
    not padding, when the attention mask keeps every position from the state
    to it, in a packed batch when the state and it share a document (segment
    0 is padding), and with a `role` (`loss_role`) when `roles`, one per
    token, give the target that role. With `blocks`, each token's block (-1 none), a target in its state's
    block does not count. Blocks are contiguous runs (`AttentionMetadata.block_groups`), so this is exact
    for the trunk's targets (`depth` 0): a prediction depth reads blocks beyond its own source, so
    `token_scores` refuses depths with blocks. With `predecessors` (`Coordinates.predecessors`, at
    `depth` 0, where its rows carry forks or a tail of finals), each target is its own token's, scored from
    its predecessor's state: it counts where it has one and both are real, a document's first token from
    its parent's last, and a block's from the predecessor's block."""
    span = depth + 1
    before = None
    if predecessors is not None:
        if depth or segment_ids is None:
            raise ValueError("a target scored from its predecessor is its own token's, at depth 0 over the "
                             "segments its rows number")
        before = xp.asarray(predecessors)[:, 1:]
        real = xp.ones(tokens.shape, bool) if attention_mask is None else xp.asarray(attention_mask, bool)
        source = xp.take_along_axis(real, xp.maximum(before, 0), axis=1)
        counted = (before >= 0) & real[:, 1:] & source
    else:
        counted = shifted_span(tokens.shape, span, valid=attention_mask, segments=segment_ids, xp=xp)
    if pad_id is not None:
        counted = counted & (tokens[:, span:] != pad_id)
    if segment_ids is not None:
        counted = counted & (segment_ids[:, span:] != 0)
    if role is not None and roles is not None:
        counted = counted & (roles[:, span:] == role)
    if blocks is not None:
        # The state's block holds the target (`dew.nn.visibility.same_group`, which this module sits below):
        # the state at the target's predecessor where the rows carry forks.
        source = (blocks[:, :-span] if before is None
                  else xp.take_along_axis(xp.asarray(blocks), xp.maximum(before, 0), axis=1))
        counted = counted & ~((source == blocks[:, span:]) & (source >= 0))
    return counted


type Path = tuple[str, ...]


type PathFilter = Callable[[Path], bool]
"""A function that selects leaves of a variables tree by the tuple of dict keys above them.

One filter type is used for the EMA selection, for `optax.multi_transform`
labels and for frozen subtrees."""


def everything(path: Path) -> bool:
    return True


def under(*prefix: str) -> PathFilter:
    """Return a filter that accepts the leaves below `prefix`.

    For example, `under("params", "context_encoder")`.
    """
    return lambda path: path[:len(prefix)] == prefix


def select(tree: Variables, keep: PathFilter) -> Variables:
    """Return the subtree of `tree` whose leaves `keep` accepts, nested the same way.

    A branch with no accepted leaf is dropped, so the result is what the EMA
    stores and what `merge` puts back. Raises `ValueError` when `keep`
    accepts no leaf at all.
    """
    def prune(node, path):
        if isinstance(node, Mapping):
            kept = {name: prune(child, (*path, name)) for name, child in node.items()}
            return {name: child for name, child in kept.items() if child is not None} or None
        return node if keep(path) else None

    selected = prune(tree, ())
    if not selected:
        raise ValueError("the filter selected no leaf of the variables tree")
    return selected


def without(tree: Variables, paths: Sequence[Sequence[str]]) -> Variables:
    """`tree` without the subtree at each of `paths`, a path the tree does not hold left as it is."""
    kept = dict(tree)
    for path in paths:
        name, *rest = path
        if name in kept:
            if rest:
                held = kept[name]
                kept[name] = without(held, [rest]) if isinstance(held, Mapping) else held
            else:
                del kept[name]
    return kept


def at_path(tree, path: Sequence[str]):
    """The subtree of `tree` at `path`, or None where `tree` holds none."""
    for key in path:
        tree = tree.get(key) if isinstance(tree, Mapping) else None
    return tree


def placed_at(tree, path: Sequence[str], value):
    """`tree` with `value` at `path`, its spine copied and missing nodes made."""
    if not path:
        return value
    return {**tree, path[0]: placed_at(tree.get(path[0], {}), path[1:], value)}


PARAMETER_ROOTS = (("params",), (FROZEN,))
"""The collections of a variables tree that hold a model's parameters, which a run's loaders read
and a storage dtype casts: the trained ones and the frozen."""


def under_roots(path: tuple, roots: tuple[tuple[str, ...], ...]) -> bool:
    """Whether the leaf at `path` (a tree path of `DictKey`s) lies under one of `roots`, each a path of
    names: the parameters a load stores in its `param_dtype` (`PARAMETER_ROOTS`), the rest as written."""
    return any(path[:len(root)] == tuple(jax.tree_util.DictKey(name) for name in root) for root in roots)


def freeze(variables: Variables, trainable: PathFilter) -> Variables:
    """Move the `params` leaves that `trainable` rejects into the `FROZEN` collection.

    `trainable` sees full leaf paths, `("params", ...)`. A tree already split
    keeps training the leaves its `params` hold, beside those `trainable`
    takes from either collection, and keeps the rest frozen: an adapter
    attached over another keeps both training. Raises `ValueError` when the
    split keeps every leaf or none, since then there is nothing to split.
    """
    # A leaf is what a dict holds, a parameter's box as its array (as `select` walks it), so a draw
    # splits in the boxes its modules made it in.
    training = (frozenset() if FROZEN not in variables else frozenset(
        ("params", *path) for path in traverse_util.flatten_dict(variables["params"])))
    variables = thaw(variables)

    def trains(path: Path) -> bool:
        return path in training or trainable(path)

    paths = [("params", *path) for path in traverse_util.flatten_dict(variables["params"])]
    kept = sum(map(trains, paths))
    if kept in (0, len(paths)):
        raise ValueError(
            f"trainable keeps {kept} of {len(paths)} parameter leaves, which "
            f"{'freezes' if kept else 'trains'} nothing")
    moving = select(variables, lambda path: path[0] == "params" and trains(path))
    frozen = select(variables, lambda path: path[0] == "params" and not trains(path))
    return {**variables, "params": moving["params"], FROZEN: frozen["params"]}


def thaw(variables: Variables) -> Variables:
    """Undo `freeze`, merging the frozen leaves back into one `params` collection.

    The flags of a packed group's frozen members (`PACKED_FROZEN`) go with the split they describe.
    Variables without either collection are returned unchanged, and a part that trains nothing (`part`
    of a frozen module) holds its frozen leaves alone.
    """
    if FROZEN not in variables and PACKED_FROZEN not in variables:
        return variables
    rest = {name: value for name, value in variables.items() if name not in (FROZEN, PACKED_FROZEN)}
    return {**rest, "params": merge(variables.get(FROZEN, {}), variables.get("params", {}))}


def unboxed[Tree](tree: Tree) -> Tree:
    """`tree` with each box of axis metadata (`nn.with_logical_partitioning`'s) replaced by the value it
    holds, as stored: how a tree enters Dew. No constraint its names would apply is applied, which
    `LogicallyPartitioned.unbox` (and `nn.unbox`) applies under an active mesh and bound Flax rules: Dew
    places the arrays from its run's own table (`Objective.parameter_axes`), in one place. A tree that
    holds no box is the caller's own, as `dicts` keeps it, so a placement writing each placed leaf into
    the dict that held it writes into the caller's."""
    def boxed(leaf) -> bool:
        return isinstance(leaf, meta.Partitioned)

    if not any(boxed(leaf) for leaf in jax.tree.leaves(tree, is_leaf=boxed)):
        return tree
    return jax.tree.map(lambda leaf: leaf.unbox(apply_constraint=False) if boxed(leaf) else leaf, tree,
                        is_leaf=boxed)


def dicts[Tree](tree: Tree) -> Tree:
    """`tree` (a variables tree, or a state holding several) with every container mapping a dict: a tree
    that holds a mapping of the kinds Flax's `unfreeze` makes dicts of (a `FrozenDict`, a dict of
    another type, an `OrderedDict`) anywhere, under a mapping, a sequence or any other node, rebuilt
    over the same leaves with each one a dict, any other the caller's own tree. `unfreeze` itself stops
    at a list. A record that is a mapping of its own kind (an optimizer's `nnx.State`) stays as it is,
    its contents read the same way. The trees an objective holds, the state's and a step's
    written-back collections are dicts, so that a placement writing each placed leaf into the dict
    that held it (`dew.distributed.host.stream`) writes into the objective's tree, and every tree of a
    step has the state's structure."""
    nodes = jax.tree.leaves(
        tree, is_leaf=lambda node: isinstance(node, (dict, FrozenDict)) and type(node) is not dict)
    if not any(isinstance(node, (dict, FrozenDict)) and type(node) is not dict for node in nodes):
        return tree
    return jax.tree.map(
        lambda node: ({key: dicts(value) for key, value in node.items()}
                      if isinstance(node, (dict, FrozenDict)) and type(node) is not dict else node),
        tree, is_leaf=lambda node: isinstance(node, (dict, FrozenDict)) and type(node) is not dict)


def owned_dicts[Tree](tree: Tree) -> Tree:
    """`tree` as a state holds its trees: every mapping a dict (`dicts`), the containers the
    state's own, so nothing the state does to them reaches the caller's tree."""
    return jax.tree.map(lambda leaf: leaf, dicts(tree))


PROBES = "cotangent_probes"
"""The collection a step's cotangent probes ride in, laid out by the model calls that read them
(`dew.nn.inspection.tapped`)."""

PARTS = "__parts__"
"""The key under which a step's observations and probes of a model applied in parts (a decision run's backbone
and head, a PPO run's policy and critic) hold each part's tree by its name in the model's variables
(`dew.nn.inspection.in_parts`): a key of those trees' own, as their summaries' (`dew.nn.inspection.SOWN`)
and probes' (`PROBE`) are, which names no module."""


def part(variables: Variables, name: str) -> Variables:
    """Cut the `name` subtree out of every collection that holds one: one
    module's own tree out of an objective's that nests several under each
    collection, `params/policy` and `frozen/policy`, so an adapted module's
    base comes with its factors. A step's cotangent probes, laid out as the
    objective nests what its model calls sow, give `name`'s where it nests
    them under `name`, else pass whole (`PROBES`)."""
    cut = {collection: subtree[name] for collection, subtree in variables.items()
           if collection != PROBES and name in subtree}
    return cut if PROBES not in variables else {**cut, PROBES: probed(variables, name)[PROBES]}


def probed(variables: Variables, name: str | None) -> Variables:
    """`variables` with a step's cotangent probes cut to those its objective nests under `name`, what one
    of its model calls sows (`dew.nn.inspection.probes_of`), a part's of a model applied in parts
    (`PARTS`), all of them where it nests none there; none with `name` None, for a call whose cotangent
    the step does not summarize (a pass under no gradient)."""
    if PROBES not in variables:
        return variables
    rest = {collection: tree for collection, tree in variables.items() if collection != PROBES}
    if name is None:
        return rest
    # A model applied in parts holds each part's probes under its name (`PARTS`).
    probes = variables[PROBES].get(PARTS, variables[PROBES])
    return {**rest, PROBES: probes.get(name, probes)}


def leaves_by_path(tree: Variables) -> dict[tuple[str, ...], jax.Array]:
    """Every leaf of a variables subtree, by its dict-key path: a node that is no mapping is one leaf, whole
    (a quantized kernel with its scales, Qwix's struct dataclasses a served model's params hold)."""
    return {tuple(entry.key for entry in path): leaf
            for path, leaf in jax.tree_util.tree_leaves_with_path(
                unboxed(tree), is_leaf=lambda node: node is not None and not isinstance(node, Mapping))}


def sown_nodes(tree, *markers: str) -> list[tuple[tuple[str, ...], Mapping]]:
    """Find every node of a sown collection that holds all of `markers`, with
    its path: the module path that sowed it.

    A sown collection nests by module path, so what a layer sowed is the
    branch that names the keys it sowed under. The search stops at such a
    branch rather than descending into the arrays below it, and walks the
    children it does descend into in sorted key order.
    """
    found: list[tuple[tuple[str, ...], Mapping]] = []

    def visit(node, path: tuple[str, ...]) -> None:
        if not isinstance(node, Mapping):
            return
        if all(marker in node for marker in markers):
            found.append((path, node))
            return
        for key in sorted(node):
            visit(node[key], (*path, key))

    visit(tree, ())
    return found


__all__ = [
    "BLOCKS_FIELD",
    "COORDINATES_FIELD",
    "EXAMPLE_ROWS",
    "FROZEN",
    "PARAMETER_ROOTS",
    "PHASE",
    "SCAN_BLOCKS",
    "TEXT_KEY",
    "VALIDITY_FIELD",
    "VALID_ROWS",
    "Batch",
    "Coordinates",
    "Path",
    "PathFilter",
    "Phase",
    "Variables",
    "coordinates",
    "dicts",
    "document_ranks",
    "document_starts",
    "everything",
    "field_coordinates",
    "flat_field",
    "freeze",
    "laid_coordinates",
    "leaves_by_path",
    "merge",
    "owned_dicts",
    "padded_rows",
    "part",
    "prefix_max",
    "prefix_min",
    "prefix_sum",
    "real_places",
    "real_slots",
    "scored_targets",
    "select",
    "shifted_span",
    "sown_nodes",
    "thaw",
    "token_id",
    "token_ids",
    "token_layout",
    "token_positions",
    "token_validity",
    "under",
    "valid_order",
    "with_validity",
    "without",
]
