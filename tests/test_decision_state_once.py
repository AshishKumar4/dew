"""A row of a request's questions over one copy of its state (`QuestionLayout.shared_state`): what each
question's own row writes before its question, written once, and each question's own part after it, each at
the positions its own row lays it at; each question reads that shared part and itself (and, with
`siblings`, the questions before it), so it decides as its own row does, its request's tokens the state's
once and each question's."""

import copy
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import as_given, assert_as_exact, widened, without_ynn
from test_decision_context import windows_of
from test_decision_letters_context import LABELS, STATES
from test_decision_streams import SPECIALS, TOKENIZER, in_streams

from dew.batch import unboxed
from dew.decision import Choice, Example
from dew.decision.layout import (
    AnswerCue,
    DecisionInputs,
    Instructions,
    OptionLines,
    State,
    StateFirstLayout,
    Text,
)
from dew.decision.model import DecisionModel, stream_runs
from dew.decision.task import laid_out
from dew.nn.attention_masks import visibility_mask

pytestmark = pytest.mark.mesh

OWN = StateFirstLayout(
    labels=LABELS, max_len=256,
    segments=(Text("STATE:\n"), State(), Instructions({"choice": "\n{}"}),
              OptionLines(line="\n{label}. {option}", bare="\n{label}. {option}", join="",
                          noul_first="false"),
              AnswerCue(template="\nAnswer:", many="\nAnswer:")))
"""A lettered row of a question of its own, its state first."""
SHARED = dataclasses.replace(OWN, shared_state=3)
QUESTIONS = {"team": Choice("Which team handles it?", ["cards", "loans", "fraud", "none"]),
             "way": Choice("Which way is it?", ["up", "down", "left", "right"]),
             "urgent": Choice("How urgent is it?", ["now", "today", "this week", "later"])}
"""Questions of four options each, the readout's count, so each question's logits are a row of four."""
STATE_ONCE_SEEDS = 16
"""Check 1's declared seeds for rows over one state against each question's own row: at 8, as the packed
rows' of two streams (`STREAM_SEEDS`), a question of two options' rows were left unresolved over 40."""


def own_and_shared(layout=SHARED, state: str = STATES[2]):
    """The request's rows of a question each (`OWN`) and over one state (`layout`)."""
    return (OWN.rows(TOKENIZER, SPECIALS, state, QUESTIONS),
            layout.rows(TOKENIZER, SPECIALS, state, QUESTIONS))


def test_a_row_over_one_state_holds_it_once_and_each_question_as_its_own_row_lays_it():
    """The row holds what the questions' own rows write before their question once, stream 0, and each
    question's own part after it, its stream its place: each token, its position, the question's spans and
    its readout, the ones its own row holds, moved along the row."""
    own, [row] = own_and_shared()
    shared = own[0].questions[0].span[0]
    tokens, positions, streams = (np.asarray(field) for field in (row.tokens, row.positions, row.streams))
    assert list(tokens[streams == 0]) == list(own[0].tokens[:shared])
    assert len(tokens) == shared + sum(len(each.tokens) - shared for each in own)
    for stream, (alone, laid) in enumerate(zip(own, row.questions, strict=True), start=1):
        part = streams == stream
        assert list(tokens[part]) == list(alone.tokens[shared:])
        assert list(positions[part]) == list(range(shared, len(alone.tokens)))
        [mine] = alone.questions
        assert laid.name == mine.name and laid.order == mine.order and laid.label_ids == mine.label_ids
        for (start, end), (own_start, own_end) in zip((laid.span, *laid.options), (mine.span, *mine.options),
                                                     strict=True):
            assert list(tokens[start:end]) == list(alone.tokens[own_start:own_end])
            assert list(positions[start:end]) == list(range(own_start, own_end))
        assert positions[laid.readout] == mine.readout and tokens[laid.readout] == alone.tokens[mine.readout]
    assert (row.state_tokens, row.state_kept, row.state_cut) == (own[0].state_tokens, own[0].state_kept, 0)


def test_questions_that_outgrow_a_row_take_several_each_repeating_the_state():
    """A request whose questions outgrow `max_len` or `shared_state` takes rows in order, each opening with
    the shared part whole: every question asked once, as its own row lays it."""
    own = OWN.rows(TOKENIZER, SPECIALS, STATES[2], QUESTIONS)
    shared = own[0].questions[0].span[0]
    two = shared + sum(len(each.tokens) - shared for each in own[:2])
    for layout in (dataclasses.replace(SHARED, shared_state=2), dataclasses.replace(SHARED, max_len=two)):
        rows = layout.rows(TOKENIZER, SPECIALS, STATES[2], QUESTIONS)
        assert [[laid.name for laid in row.questions] for row in rows] == [["team", "way"], ["urgent"]]
        for row in rows:
            assert row.tokens[:shared] == own[0].tokens[:shared] and len(row.tokens) <= layout.max_len


def test_each_question_reads_the_shared_part_and_itself_and_with_siblings_those_before_it():
    """The model's runs of a row over one state (`stream_runs` of `Layout.attention`): a token of the shared
    part reads it up to itself; a question's token reads the shared part whole and its own part up to
    itself, and with `siblings` each question before it whole; no question reads one after it."""
    for siblings in (False, True):
        layout = dataclasses.replace(SHARED, siblings=siblings)
        _, [row] = own_and_shared(layout)
        inputs = DecisionInputs.collate([row], SPECIALS.pad, attention=layout.attention)
        runs = stream_runs(inputs.streams, inputs.segments, inputs.reads, inputs.fragments)
        length = inputs.tokens.shape[1]
        seen = np.asarray(visibility_mask(length, length, causal=False, sliding_window=None,
                                          key_runs=runs))[0, 0]
        streams = np.asarray(row.streams)
        index = np.arange(len(streams))
        for query, stream in enumerate(streams):
            read = (streams == 0) & ((index <= query) if stream == 0 else True)
            read |= (streams == stream) & (index <= query)
            if siblings and stream > 0:
                read |= (streams > 0) & (streams < stream)
            assert list(seen[query, :len(streams)]) == list(read), (siblings, query)


def examples() -> list[Example]:
    """Each state asked every question, answered with its first option."""
    return [Example(state, QUESTIONS, {name: question.options[0] for name, question in QUESTIONS.items()})
            for state in STATES]


def scored(objective, windows, variables) -> dict[str, np.ndarray]:
    """Each question's real options' logits, keyed by its request's shared part's tokens and its own
    part's: a row of its own is both, a row over one state its stream 0 and the question's stream."""
    forward = without_ynn(lambda variables, inputs: objective.model.logits(variables, inputs))
    logits = {}
    for batch in windows:
        inputs = laid_out(batch, objective.bounds.segments, attention=objective.layout.attention)
        held = np.asarray(forward(variables, inputs))
        options, owners = np.asarray(batch["options"]), np.asarray(batch["question_rows"])
        tokens, segments = np.asarray(batch["tokens"]), np.asarray(batch["segments"])
        streams = np.asarray(batch["streams"]) if "streams" in batch else np.zeros_like(tokens)
        for window, owner in enumerate(owners):
            for row in range(1, int(segments[window].max()) + 1):
                own = segments[window] == row
                for stream, slot in enumerate(np.flatnonzero(owner == row), start=1):
                    part = own & ((streams[window] == 0) | (streams[window] == stream)
                                  if "streams" in batch else True)
                    key = bytes(tokens[window][part].astype(np.uint8)).decode(errors="replace")
                    logits[key] = held[window][slot][options[window][slot]]
    return dict(sorted(logits.items()))


@pytest.mark.parametrize("siblings", [False], ids=["apart"])
def test_a_row_over_one_state_decides_each_question_as_its_own_row_does(siblings):
    """Rows of a request's questions over one state, two requests a window, decide each question's options as
    its own row does in a window of its own, through the unmodified objective (check 1, over drawn
    parameters, float64 truth)."""
    shared = in_streams(dataclasses.replace(SHARED, siblings=siblings), segments=2)
    alone = in_streams(OWN, segments=1)
    rows = examples()
    windows = {"shared": windows_of(shared, rows, count=8), "alone": windows_of(alone, rows, count=24)}
    held = max(len(np.flatnonzero(owner == 1)) for batch in windows["shared"]
               for owner in np.asarray(batch["question_rows"]))
    assert held == len(QUESTIONS), "no row holds every question"
    initialized = jax.jit(lambda key: unboxed(shared.init(key, variables={})))
    wide = copy.copy(alone)
    wide.model = DecisionModel(alone.model.backbone.clone(dtype=jnp.float64),
                               alone.model.head.clone(dtype=jnp.float64), alone.model.context_encoder)

    @as_given
    def truth(variables):
        with jax.enable_x64(new_val=True):
            return scored(wide, windows["alone"], widened(variables))

    assert_as_exact(as_given(lambda variables: scored(shared, windows["shared"], variables)),
                    as_given(lambda variables: scored(alone, windows["alone"], variables)),
                    lambda seed: (initialized(jax.random.key(seed)),), "rows over one state",
                    seeds=STATE_ONCE_SEEDS, truth_fn=truth)
