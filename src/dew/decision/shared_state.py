"""Rows of a request's questions over one copy of its state (`QuestionLayout.shared_state`).

Each question's own row (`QuestionLayout._written`) writes the state and what precedes the question, then the
question's own part: its instructions, options and cue. A row over one state writes that shared part once,
its document 1, and each question's own part after it as a document continuing it (`Encoded.forked`,
`dew.batch.Forks`), each token at the position its own row lays it at, so each question reads the shared
part and itself as its own row does, a recurrent layer from the state the shared part leaves.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from dew.batch import Forks
from dew.decision.layout import (
    HOWS,
    AnswerCue,
    Encoded,
    Instructions,
    Layout,
    OptionLines,
    QuestionLayout,
    State,
)
from dew.nn.protocols import ForkedCalls

if TYPE_CHECKING:
    from flax import linen as nn

    from dew.decision.head import Head
    from dew.nn.protocols import TokenModel


def check_sharing(layout: QuestionLayout) -> None:
    """Refuse `layout`'s sharing where its rows cannot ask questions over one state: siblings or rows in
    streams with no `shared_state`, a `shared_state` under one, or several questions a row it cannot lay
    (`forkable`)."""
    if layout.shared_state is None:
        raise ValueError("siblings, and rows in streams, are of the questions a row over one state asks; "
                         "give shared_state=")
    if layout.shared_state < 1:
        raise ValueError(f"shared_state is at least 1, got {layout.shared_state}")
    if layout.shared_state > 1 and not forkable(layout):
        raise ValueError("a row of questions over one state writes its state before its question, the "
                         "question's instructions before its options and cue, and sets nothing else its "
                         "tokens read, as parallel options, blocks, reads and a question in context do")


def over_one_state(own: Sequence[Encoded], layout: QuestionLayout) -> list[Encoded]:
    """Each question's `own` row, in order, in the first row over one shared part that takes it (`takes`)
    within `layout`'s `max_len` and `shared_state` questions, or a row of its own, where it shares
    (`QuestionLayout.shares`); with siblings, each question in a row continuing the one before it
    (`merged`). Rows in streams (`QuestionLayout.streamed`) are merged so at any `shared_state`, one
    question's too, as the plans that laid them did. `own` as it stands where it shares none."""
    if layout.shared_state is None or not (layout.shares or layout.streamed):
        return list(own)
    max_len, count, streamed = layout.max_len, layout.shared_state, layout.streamed
    rows: list[list[Encoded]] = []
    for row in own:
        taken = next((held for held in rows if takes(held, row, max_len, count, empty=streamed)), None)
        if taken is None:
            rows.append([row])
        else:
            taken.append(row)
    return [merged(held, chained=layout.siblings, streamed=streamed) for held in rows]


def takes(held: Sequence[Encoded], row: Encoded, max_len: int, count: int, *, empty: bool = False) -> bool:
    """Whether the row of `held`'s questions takes `row`'s: both hold the whole state and write alike before
    their questions, fewer than `count` asked, and room for its part within `max_len`. A row writing
    nothing before its question shares nothing, unless `empty`, as rows in streams took it."""
    shared = held[0].questions[0].span[0]
    if not shared and not empty:
        return False
    whole = all(each.state_kept == each.state_tokens for each in (held[0], row))
    alike = held[0].tokens[:shared] == row.tokens[:row.questions[0].span[0]]
    return (whole and alike and len(held) < count
            and shared + sum(len(each.tokens) - shared for each in (*held, row)) <= max_len)


def merged(rows: Sequence[Encoded], *, chained: bool = False, streamed: bool = False) -> Encoded:
    """`rows`, each one question's own, as one row: their shared part once (document 1), then each one's own
    part (documents 2...), each token at its own row's position; its cuts its questions', its state's
    once. Each question's document continues the shared part, or where `chained` the question's before it
    (`Encoded.forked_parents`), which it then reads whole (`QuestionLayout.siblings`). Where `streamed`,
    the shared part is the row's stream 0 and each question's own part a stream of its own
    (`Encoded.streams`), which the layout's `attention` reads (`QuestionLayout.streamed`)."""
    shared = rows[0].questions[0].span[0]
    if not shared and not streamed:
        # Nothing before the question to share (`takes`): the row alone, one document.
        [row] = rows
        alone = (0,) * len(row.tokens)
        return dataclasses.replace(row, positions=tuple(range(len(row.tokens))),
                                   forked=(1,) * len(row.tokens), forked_parents=alone if chained else None)
    tokens, positions, documents = list(rows[0].tokens[:shared]), list(range(shared)), [1] * shared
    questions = []
    for document, row in enumerate(rows, start=2):
        moved = len(tokens) - shared
        [laid] = row.questions
        questions.append(dataclasses.replace(
            laid, span=(laid.span[0] + moved, laid.span[1] + moved),
            options=tuple((start + moved, end + moved) for start, end in laid.options),
            readout=None if laid.readout is None else laid.readout + moved))
        tokens += row.tokens[shared:]
        positions += range(shared, len(row.tokens))
        documents += [document] * (len(row.tokens) - shared)
    first = rows[0]
    row = Encoded(tuple(tokens), tuple(questions), tuple(positions), None, first.state_tokens,
                  first.state_kept, instructions_cut=sum(row.instructions_cut for row in rows),
                  options_cut=sum(row.options_cut for row in rows), state_cut=first.state_cut,
                  state_characters_cut=first.state_characters_cut,
                  options_alike=any(row.options_alike for row in rows))
    if streamed:
        # A stream is its document's number less one: the shared part's 0.
        return dataclasses.replace(row, streams=tuple(document - 1 for document in documents))
    return dataclasses.replace(
        row, forked=tuple(documents),
        forked_parents=tuple(document - 1 for document in documents) if chained else None)


def fork_depth(layout: Layout) -> int:
    """The longest chain of documents `layout`'s rows over one state lay (`Forks.depth`): one, each
    question continuing the shared part; with `QuestionLayout.siblings` a row's questions, each continuing
    the one before it, so reading them whole."""
    if isinstance(layout, QuestionLayout) and layout.siblings and layout.shared_state is not None:
        return layout.shared_state
    return 1


def joint(layout: Layout) -> bool:
    """Whether `layout`'s rows over one state read their documents together, as its siblings do
    (`QuestionLayout.siblings`, `Forks.joint`): one row, its dropout drawn once."""
    return isinstance(layout, QuestionLayout) and layout.siblings and layout.shares


def collated_forks(rows: Sequence[Encoded], documents: np.ndarray, count: int
                   ) -> tuple[jax.Array, jax.Array, int, bool]:
    """A collated batch of `rows` over one state, a window a row (`DecisionInputs.collate`): their tokens'
    `documents`, each document's parent among a row's `count` questions' and its shared part's, its
    shared part document 1 and each question's own part continuing it, or with siblings the question's
    before it (`Encoded.forked_parents`), the depth of their chains, and whether each row's documents are
    read together, as siblings are (`Forks.joint`)."""
    chained = rows[0].forked_parents is not None
    continued = np.arange(count + 1) if chained else np.arange(count + 1) > 0
    parents = np.broadcast_to(continued, (len(rows), count + 1))
    return (jnp.asarray(documents, jnp.int32), jnp.asarray(parents, jnp.int32), count if chained else 1,
            chained)


def asked_over(documents: jax.Array, parents: jax.Array, depth: int) -> jax.Array:
    """How many questions' own rows hold each document of a window `[B, D]`, as float32: a question's own
    row writes the shared part and its own part (with siblings, the questions' before it too), so a
    question's document counts itself and every question continuing it, a shared part the questions
    continuing it, and a document nothing continues, a row of one, once; 0 for a document the window
    holds none of. `documents` `[B, L]` number each token's document from 1 (`DecisionInputs.documents`),
    `parents` `[B, D]` the document each continues, 0 for a shared part, and `depth` bounds a chain of them
    (`fork_depth`). What a term over a row's own tokens weighs a document's tokens by, so rows over one
    state sum what each question's own row gives (`DecisionObjective.input_terms`)."""
    numbers = jnp.arange(1, parents.shape[1] + 1, dtype=documents.dtype)
    held = jnp.any(documents[:, :, None] == numbers, axis=1)
    asks = held & (parents > 0)
    # Each question's document counts for itself and once for each document of its chain
    # (`Forks.ancestors`).
    chains = jnp.where(asks[:, :, None], Forks(parents, depth=depth).ancestors(), 0)
    over = jnp.sum(chains[:, :, :, None] == numbers, axis=(1, 2), dtype=jnp.float32)
    return jnp.maximum(asks + over, held.astype(jnp.float32))


def stream_reads(layout: QuestionLayout) -> tuple[tuple[tuple[int, ...], ...], int]:
    """The reads of `layout`'s rows in streams over one state (`QuestionLayout.streamed`), as
    `QuestionLayout.attention` gives a row's: stream 0, the shared part, reads itself causally; each
    question's stream itself causally, the shared part whole and, with `siblings`, the questions' before
    it whole; with the streams' count less one."""
    count = layout.shared_state
    assert count is not None, "rows in streams state their shared_state (`check_sharing`)"
    questions = tuple((HOWS["causal"], HOWS["whole"],
                       *(HOWS["whole"] if layout.siblings and other < own else 0
                         for other in range(1, count + 1)))
                      for own in range(1, count + 1))
    return ((HOWS["causal"], *[0] * (count + 1)), *questions), count + 1


def over_one_state_row(row: Encoded) -> bool:
    """Whether `row` asks several questions over one state (`merged`): each a document of its own
    continuing the shared part, or a stream of its own at its own row's positions, which a head reading
    across its row's questions would join."""
    apart = row.forked is not None or (row.streams is not None and row.positions is not None)
    return apart and len(row.questions) > 1


def forkable(layout: QuestionLayout) -> bool:
    """Whether `layout`'s rows can ask several questions over one state: each writes its state before its
    question and the question's instructions first, and sets nothing else its tokens read, as parallel
    options, blocks, reads and a question in its context do."""
    if layout.parallel or layout.blocks or layout.reads is not None or layout.spans_in_context:
        return False
    kinds = [type(segment) for segment in layout.written]
    asked = [at for at, kind in enumerate(kinds) if kind in (Instructions, OptionLines, AnswerCue)]
    return (max((at for at, kind in enumerate(kinds) if kind is State), default=-1) < min(asked)
            and kinds.index(Instructions) == min(asked))


def shared_for(layout: QuestionLayout, backbone: nn.Module | TokenModel, head: Head, *,
               train: bool = False) -> QuestionLayout:
    """`layout` as `head` reads rows over `backbone`: its questions over one state where the backbone
    computes each question's documents as its own row does (`ForkedCalls`, in a training step where
    `train`, no dropout drawing) and the head reads each question apart from the others
    (`Head.reads_apart`, one question a row not asked), as many as a row holds where the layout states
    none (`QuestionLayout.shared_state`); a row a question otherwise. Siblings are no sharing to derive
    but what a row's questions read (`QuestionLayout.siblings`): kept where the backbone runs forks, as it
    reads one row over one state in training too (its dropout drawn once, as the row's streams drew it),
    and the head reads a row's questions apart; refused otherwise, by the backbone's reason."""
    if layout.siblings and layout.shares:
        refusal = (backbone.fork_refusal() if isinstance(backbone, ForkedCalls)
                   else f"a {type(backbone).__name__} runs no forks (ForkedCalls)")
        if refusal is None and not (head.reads_apart and not head.one_question):
            refusal = "its head reads across a row's questions, or one question a row"
        if refusal is not None:
            raise ValueError(f"siblings read the questions before them in their row through its documents' "
                             f"forks, which this model cannot: {refusal}")
        return layout
    reads = (head.reads_apart and not head.one_question and isinstance(backbone, ForkedCalls)
             and backbone.fork_refusal(train) is None)
    if not reads:
        return dataclasses.replace(layout, shared_state=1) if layout.shares else layout
    if layout.shared_state is None and forkable(layout):
        return dataclasses.replace(layout, shared_state=layout.max_len)
    return layout
