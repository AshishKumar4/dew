"""The batch a decision model reads (`DecisionInputs`): a window's laid-out rows (`dew.decision.layout`'s
`Encoded`) collated into the arrays its backbone, head and loss take, packed windows of whole rows or
rows padded to the longest, with each question's place in them, its option slots compacted where a
caller fixes their count (`compacted`), and the coordinates its tokens stand at (`dew.batch.Coordinates`).
A layout says how a row is written; this module says what a batch of them is."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import struct

from dew.batch import POSITIONS_FIELD, VALIDITY_FIELD, Coordinates, Forks, coordinates
from dew.decision.layout import Encoded, question_fields
from dew.nn.inputs import pad_token_rows


def question_owners(segments, spans, questions, spans_in_context: bool, xp=np, rows=None):
    """The `[B, Q]` row each question of packed windows belongs to, 0 for a padding slot: the row the
    packer placed it in (`rows`, `PackedRows`' `question_rows`), or where none is given the row its
    span starts in, and where the spans are the context's (`DecisionInputs.spans_in_context`) the
    window's one row (`DecisionInputs.collate`). `xp` is the array namespace, NumPy for host geometry
    and `jax.numpy` in a trace, so the model's question grouping and the counted FLOPs read one owner."""
    if rows is not None:
        return xp.where(questions, xp.asarray(rows), 0).astype(segments.dtype)
    if spans_in_context:
        return xp.asarray(questions).astype(segments.dtype)
    starts = xp.asarray(spans)[..., 0]
    return xp.where(questions, xp.take_along_axis(segments, starts, axis=1), 0)


@struct.dataclass
class DecisionInputs:
    """A batch of laid-out rows: packed windows of whole rows, or rows padded to the longest.

    - `tokens` `[B, L]` ids, `segments` `[B, L]` each token's row in its
      window, numbered from 1 in window order and 0 on the padding after the
      last (`Pack`); a padded batch is windows of one row each (`collate`);
    - `positions` `[B, L]` each token's place in its row (None: counted from
      0 along it), and in the parallel layout `slots` `[B, L]` each token's
      option slot, 0 for the tokens every option shares;
    - `kinds` `[B, Q]` each question's type (`KINDS`), `questions` `[B, Q]` the
      real questions, in their rows' window order, `spans` `[B, Q, 2]` their
      instructions' (start, end) in the window;
    - `option_spans` `[B, Q, K, 2]` each option slot's (start, end), `options`
      `[B, Q, K]` the real slots;
    - `option_slots` `[B, P]` each window's real options as flat indices
      `q * K + k` into its `[Q, K]` grid, in that order (`compacted`), and
      `slotted` `[B, P]` which of them are real, or None for the grid itself;
    - `context` `[B, C]` each window's rows' context tokens end to end, which a
      backbone's cross-attention reads, `context_lengths` `[B]` how many are
      real, the padding after them, and `context_segments` `[B, C]` each one's
      row in the window as `segments` numbers it (0 for padding), which its
      row's tokens alone read; or None for rows that carry none;
    - `question_rows` `[B, Q]` each question's row in its window as the packer
      placed it, numbered as `segments` numbers its tokens (0 for a padding
      slot), or None where a question's row is the one its span starts in.

    A row of one question with one marker per option, Laya's, has spans one
    token long. A head that reads `option_slots` computes the real options
    alone, P rather than Q x K of them: a window holds a few of its grid's.
    `max_segments` bounds the rows a window holds (`Layout.bounds`).
    """

    tokens: jnp.ndarray
    segments: jnp.ndarray
    kinds: jnp.ndarray
    questions: jnp.ndarray
    spans: jnp.ndarray
    option_spans: jnp.ndarray
    options: jnp.ndarray
    positions: jnp.ndarray | None = None
    slots: jnp.ndarray | None = None
    option_slots: jnp.ndarray | None = None
    slotted: jnp.ndarray | None = None
    context: jnp.ndarray | None = None
    context_lengths: jnp.ndarray | None = None
    context_segments: jnp.ndarray | None = None
    question_rows: jnp.ndarray | None = None
    readout: jnp.ndarray | None = None
    """`[B, Q]` where a readout of labels reads each question's answer (`Laid.readout`), or None; the
    labels' ids are the readout head's own (`LabelReadout.label_ids`)."""
    blocks: jnp.ndarray | None = None
    """`[B, L]` each token's block in its row (`Encoded.blocks`), -1 on the padding, or None for rows
    read in order."""
    streams: jnp.ndarray | None = None
    """`[B, L]` each token's stream in its row (`Encoded.streams`), -1 on the padding, or None for rows
    read in order."""
    documents: jnp.ndarray | None = None
    """`[B, L]` each token's document in its window where its rows ask questions over one state
    (`Encoded.forked`): each row's shared part a document and each question's own part one continuing it,
    numbered from 1 in the window's order, 0 on the padding; None for rows of one document each."""
    parents: jnp.ndarray | None = None
    """`[B, D]` the document each of `documents`' continues, 0 for a shared part (`dew.batch.Forks`)."""
    reads: tuple[tuple[int, ...], ...] | None = struct.field(pytree_node=False, default=None)
    """What each stream's tokens read, as the layout lowers its `Reads` (`Reads.lowered`), where the rows
    stand in `streams`."""
    fragments: int = struct.field(pytree_node=False, default=1)
    """The most runs of one stream a row lays (`fragment_count`), which its tokens' key runs number."""
    max_segments: int = struct.field(pytree_node=False, default=1)
    spans_in_context: bool = struct.field(pytree_node=False, default=False)
    """Whether the questions' spans are positions in `context`, which holds the schema
    (`Schema.stream`), rather than in `tokens`."""
    whole_blocks: bool = struct.field(pytree_node=False, default=False)
    """Whether each row's one block is the whole row (`QuestionLayout.split` 'whole'), which a backbone
    reads as non-causal attention by rows (`AttentionMetadata.whole_blocks`)."""
    fork_depth: int = struct.field(pytree_node=False, default=1)
    """The longest chain of `documents` (`Forks.depth`)."""
    joint_forks: bool = struct.field(pytree_node=False, default=False)
    """Whether each row's `documents` are read together, as its siblings are (`Forks.joint`)."""

    @property
    def valid(self) -> jnp.ndarray:
        """Return the `[B, L]` real tokens."""
        return self.segments != 0

    def coordinates(self) -> Coordinates:
        """The tokens' `Coordinates`: each row a document, real tokens first, at
        most `max_segments` a window. The padding after them runs as one more
        sequence, which a sequence kernel adds to the documents itself
        (`dew.nn.attention.flash_attention`). Rows asking questions over one
        state lay out their `documents` instead, each question's continuing its
        row's shared part (`Coordinates.forks`)."""
        if self.documents is None:
            return coordinates(segments=self.segments, positions=self.positions,
                               max_segments=self.max_segments, right_padded=True)
        assert self.parents is not None, "a window's documents come with the documents they continue"
        laid = coordinates(segments=self.documents, positions=self.positions,
                           max_segments=self.parents.shape[1], right_padded=True)
        held = self.max_segments * self.fork_depth  # a shared part, or with siblings a chain, a row
        return dataclasses.replace(laid, forks=Forks(self.parents, self.fork_depth, self.joint_forks, held))

    @property
    def question_segments(self) -> jnp.ndarray:
        """Return the `[B, Q]` row each question belongs to, 0 for a padding slot (`question_owners`)."""
        return question_owners(self.segments, self.spans, self.questions, self.spans_in_context, jnp,
                               self.question_rows)

    @property
    def markers(self) -> jnp.ndarray:
        """Return the `[B, Q, K]` first token of each option slot, where a marker layout scores it."""
        return self.option_spans[..., 0]

    @classmethod
    def collate(cls, rows: Sequence[Encoded], pad: int, *, questions: int | None = None,
                options: int | None = None, option_slots: int | None = None,
                context_length: int | None = None, width: int | None = None,
                attention: tuple[tuple[tuple[int, ...], ...], int] | None = None) -> DecisionInputs:
        """Right-pad `rows` into one batch (`pad_token_rows`), one row a window, its question
        and option slots to the widest.

        `questions` and `options` widen those slots further, and `width` the
        tokens (`pad_token_rows`, which no row may exceed), to shapes a caller
        keeps fixed. `option_slots` is the real options a row may hold, a shape
        a caller keeps fixed too (`compacted`); without it, the batch's most,
        rounded up to `OPTION_TILE`, where that is fewer than the grid's slots.
        With a `context_length`, each row's context (`Encoded.context`, none
        where it holds none) is padded to that many tokens. Rows in streams are read as their layout's
        `attention` says (`Layout.attention`).
        """
        positioned = rows[0].positions is not None
        if any((row.positions is not None) != positioned for row in rows):
            raise ValueError("one batch lays its rows out one way, at their positions or in order")
        in_context = rows[0].spans_in_context
        if any(row.spans_in_context != in_context for row in rows):
            raise ValueError("one batch reads its rows' schemas one way, in the row or as its context")
        blocked = rows[0].blocks is not None
        if any((row.blocks is not None) != blocked for row in rows):
            raise ValueError("one batch reads its rows one way, in blocks or in order")
        # Parallel options' slots beside their positions; a row of questions over one state, positions alone.
        fields = {POSITIONS_FIELD: [row.positions or () for row in rows]} if positioned else {}
        if rows[0].slots is not None:
            fields["slots"] = [row.slots or () for row in rows]
        if blocked:
            fields["blocks"] = [row.blocks or () for row in rows]
        streamed = rows[0].streams is not None
        if any((row.streams is not None) != streamed for row in rows) or streamed != (attention is not None):
            raise ValueError("one batch reads its rows one way, in their layout's streams or in order")
        if streamed:
            fields["streams"] = [row.streams or () for row in rows]
        forked = rows[0].forked is not None
        if any((row.forked is not None) != forked for row in rows):
            raise ValueError("one batch lays its rows out one way, over one state or a question a row")
        if forked:
            fields["forked"] = [row.forked or () for row in rows]
        tokens, padded = pad_token_rows([row.tokens for row in rows], pad_id=pad, padding_side="right",
                                        fields=fields or None, width=width)
        valid = padded.get(VALIDITY_FIELD, np.ones(tokens.shape, bool))
        count = max(questions or 1, *(len(row.questions) for row in rows))
        documents, parents, depth, joint = None, None, 1, False
        if forked:
            from dew.decision.shared_state import collated_forks

            numbered = np.where(valid, padded["forked"], 0)
            documents, parents, depth, joint = collated_forks(rows, numbered, count)
        width = max(options or 1, *(len(laid.options) for row in rows for laid in row.questions))
        labelled = labelled_rows(rows)
        laid = [question_fields(row, count, width, labelled) for row in rows]
        stacked = {name: np.stack([fields[name] for fields in laid]) for name in laid[0]}
        along: dict[str, Any] = {name: jnp.asarray(value) for name, value in stacked.items()}
        real_options = stacked["options"]
        if option_slots is None:
            most = int(real_options.reshape(len(rows), -1).sum(axis=1).max())
            option_slots = -(-max(most, 1) // OPTION_TILE) * OPTION_TILE
            compact = option_slots < count * width
        else:
            compact = True
        indexed = [compacted(row, option_slots) for row in real_options] if compact else None
        slots_of, slotted = ((None, None) if indexed is None else
                             (jnp.asarray(np.stack([slot for slot, _ in indexed])),
                              jnp.asarray(np.stack([real for _, real in indexed]))))
        context = lengths = None
        if context_length is not None:
            held = [row.context or () for row in rows]
            if max(map(len, held)) > context_length:
                raise ValueError(f"a row's context holds {max(map(len, held))} tokens, more than the "
                                 f"{context_length} the backbone's cross-attention reads")
            context = jnp.asarray([[*tokens, *[pad] * (context_length - len(tokens))] for tokens in held],
                                  jnp.int32)
            lengths = jnp.asarray([len(tokens) for tokens in held], jnp.int32)
        return cls(tokens=jnp.asarray(tokens), segments=jnp.asarray(valid, jnp.int32),
                   **along,
                   positions=jnp.asarray(padded[POSITIONS_FIELD], jnp.int32) if positioned else None,
                   slots=jnp.asarray(padded["slots"], jnp.int32) if "slots" in fields else None,
                   option_slots=slots_of, slotted=slotted,
                   context=context, context_lengths=lengths,
                   # The padding is in no block, and no stream.
                   blocks=jnp.asarray(np.where(valid, padded["blocks"], -1), jnp.int32) if blocked else None,
                   streams=(jnp.asarray(np.where(valid, padded["streams"], -1), jnp.int32) if streamed
                            else None),
                   documents=documents, parents=parents, fork_depth=depth, joint_forks=joint,
                   reads=None if attention is None else attention[0],
                   fragments=1 if attention is None else attention[1],
                   spans_in_context=in_context)


def labelled_rows(rows: Sequence[Encoded]) -> bool:
    """Whether `rows`' options stand under labels (`Labels`), read at an answer cue; one batch reads
    its rows' options one way."""
    labelled = {laid.readout is not None for row in rows for laid in row.questions}
    if len(labelled) > 1:
        raise ValueError("one batch reads its rows' options one way, under labels or at their markers")
    return bool(labelled) and labelled.pop()


OPTION_TILE = 128
"""What a batch's option slots round up to when its caller fixes none (`DecisionInputs.collate`)."""


def compacted(options: np.ndarray, slots: int) -> tuple[np.ndarray, np.ndarray]:
    """A row's real options, `options` `[Q, K]`, as `slots` flat indices into
    its grid in row order, padded with 0, and which of them are real.

    Raises ValueError for a row of more real options than `slots`: its shape
    is the caller's to keep, and widening it would compile the step again.
    """
    real = np.flatnonzero(np.asarray(options, bool).reshape(-1)).astype(np.int32)
    if len(real) > slots:
        raise ValueError(f"a row of {len(real)} options outgrows its {slots} option slots; "
                         "raise option_slots to the most options a row holds")
    indices = np.zeros(slots, np.int32)
    indices[:len(real)] = real
    return indices, np.arange(slots) < len(real)
