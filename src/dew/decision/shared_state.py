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

from dew.decision.layout import AnswerCue, Encoded, Instructions, OptionLines, QuestionLayout, State
from dew.nn.protocols import ForkedCalls

if TYPE_CHECKING:
    from flax import linen as nn

    from dew.decision.head import Head
    from dew.nn.protocols import TokenModel


def over_one_state(own: Sequence[Encoded], max_len: int, count: int) -> list[Encoded]:
    """Each question's `own` row, in order, in the first row over one shared part that takes it (`takes`)
    within `max_len` and `count` questions, or a row of its own."""
    rows: list[list[Encoded]] = []
    for row in own:
        taken = next((held for held in rows if takes(held, row, max_len, count)), None)
        if taken is None:
            rows.append([row])
        else:
            taken.append(row)
    return [merged(held) for held in rows]


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


def merged(rows: Sequence[Encoded]) -> Encoded:
    """`rows`, each one question's own, as one row: their shared part once (document 1), then each one's own
    part (documents 2...), each token at its own row's position; its cuts its questions', its state's
    once."""
    shared = rows[0].questions[0].span[0]
    if not shared:
        # Nothing before the question to share (`takes`): the row alone, one document.
        [row] = rows
        return dataclasses.replace(row, positions=tuple(range(len(row.tokens))),
                                   forked=(1,) * len(row.tokens))
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
                   options_alike=any(row.options_alike for row in rows), forked=tuple(documents))


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
    (`Head.reads_apart`, one question a row not asked); a row a question otherwise."""
    reads = (head.reads_apart and not head.one_question and isinstance(backbone, ForkedCalls)
             and backbone.fork_refusal(train) is None)
    return layout if reads or not layout.shares else dataclasses.replace(layout, shared_state=1)
