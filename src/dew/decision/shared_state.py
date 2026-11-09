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

from dew.decision.layout import AnswerCue, Encoded, Instructions, Layout, OptionLines, QuestionLayout, State
from dew.nn.protocols import ForkedCalls

if TYPE_CHECKING:
    from flax import linen as nn

    from dew.decision.head import Head
    from dew.nn.protocols import TokenModel


def check_sharing(layout: QuestionLayout) -> None:
    """Refuse `layout`'s sharing where its rows cannot ask questions over one state: siblings with no
    `shared_state`, a `shared_state` under one, or several questions a row it cannot lay (`forkable`)."""
    if layout.shared_state is None:
        raise ValueError("siblings are the questions a row over one state asks; give shared_state=")
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
    (`merged`). `own` as it stands where it shares none."""
    if not layout.shares or layout.shared_state is None:
        return list(own)
    max_len, count, chained = layout.max_len, layout.shared_state, layout.siblings
    rows: list[list[Encoded]] = []
    for row in own:
        taken = next((held for held in rows if takes(held, row, max_len, count)), None)
        if taken is None:
            rows.append([row])
        else:
            taken.append(row)
    return [merged(held, chained=chained) for held in rows]


def takes(held: Sequence[Encoded], row: Encoded, max_len: int, count: int) -> bool:
    """Whether the row of `held`'s questions takes `row`'s: both hold the whole state and write alike before
    their questions, fewer than `count` asked, and room for its part within `max_len`."""
    shared = held[0].questions[0].span[0]
    if not shared:
        # A row writing nothing before its question shares nothing.
        return False
    whole = all(each.state_kept == each.state_tokens for each in (held[0], row))
    alike = held[0].tokens[:shared] == row.tokens[:row.questions[0].span[0]]
    return (whole and alike and len(held) < count
            and shared + sum(len(each.tokens) - shared for each in (*held, row)) <= max_len)


def merged(rows: Sequence[Encoded], *, chained: bool = False) -> Encoded:
    """`rows`, each one question's own, as one row: their shared part once (document 1), then each one's own
    part (documents 2...), each token at its own row's position; its cuts its questions', its state's
    once. Each question's document continues the shared part, or where `chained` the question's before it
    (`Encoded.forked_parents`), which it then reads whole (`QuestionLayout.siblings`)."""
    shared = rows[0].questions[0].span[0]
    if not shared:
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
    return Encoded(tuple(tokens), tuple(questions), tuple(positions), None, first.state_tokens,
                   first.state_kept, instructions_cut=sum(row.instructions_cut for row in rows),
                   options_cut=sum(row.options_cut for row in rows), state_cut=first.state_cut,
                   state_characters_cut=first.state_characters_cut,
                   options_alike=any(row.options_alike for row in rows), forked=tuple(documents),
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


def over_one_state_row(row: Encoded) -> bool:
    """Whether `row` asks several questions over one state (`merged`): each a document of its own
    continuing the shared part, which a head reading across its row's questions would join."""
    return row.forked is not None and len(row.questions) > 1


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
