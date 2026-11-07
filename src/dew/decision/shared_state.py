"""Rows of a request's questions over one copy of its state (`QuestionLayout.shared_state`).

Each question's own row (`QuestionLayout._written`) writes the state and what precedes the question, then the
question's own part: its instructions, options and cue. A row over one state writes that shared part once, in
its stream 0, and each question's own part after it in a stream of its own, each token at the position its
own row lays it at, so each question reads the shared part and itself as its own row does
(`QuestionLayout.attention`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from dew.decision.layout import Encoded


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
    whole = all(each.state_kept == each.state_tokens for each in (held[0], row))
    alike = held[0].tokens[:shared] == row.tokens[:row.questions[0].span[0]]
    return (whole and alike and len(held) < count
            and shared + sum(len(each.tokens) - shared for each in (*held, row)) <= max_len)


def merged(rows: Sequence[Encoded]) -> Encoded:
    """`rows`, each one question's own, as one row: their shared part once (stream 0), then each one's own
    part (streams 1...), each token at its own row's position; its cuts its questions', its state's once."""
    shared = rows[0].questions[0].span[0]
    tokens, positions, streams = list(rows[0].tokens[:shared]), list(range(shared)), [0] * shared
    questions = []
    for stream, row in enumerate(rows, start=1):
        moved = len(tokens) - shared
        [laid] = row.questions
        questions.append(dataclasses.replace(
            laid, span=(laid.span[0] + moved, laid.span[1] + moved),
            options=tuple((start + moved, end + moved) for start, end in laid.options),
            readout=None if laid.readout is None else laid.readout + moved))
        tokens += row.tokens[shared:]
        positions += range(shared, len(row.tokens))
        streams += [stream] * (len(row.tokens) - shared)
    first = rows[0]
    return Encoded(tuple(tokens), tuple(questions), tuple(positions), None, first.state_tokens,
                   first.state_kept, instructions_cut=sum(row.instructions_cut for row in rows),
                   options_cut=sum(row.options_cut for row in rows), state_cut=first.state_cut,
                   state_characters_cut=first.state_characters_cut,
                   options_alike=any(row.options_alike for row in rows), streams=tuple(streams))


def over_one_state_row(row: Encoded) -> bool:
    """Whether `row` asks several questions over one state (`merged`): each in a stream of its own at its
    own row's positions, which a head reading across its row's questions would join."""
    return row.streams is not None and row.positions is not None and len(row.questions) > 1
