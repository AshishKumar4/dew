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
from dew.data.text import ByteTokenizer
from dew.decision import Choice, Example
from dew.decision.head import JointSchemaHead, LabelReadout
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
from dew.decision.rows import Encoding, Fit
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


def test_a_run_over_one_state_trains_and_its_task_asks_a_requests_questions_in_one_row(tmp_path):
    """A decision run of a letter readout over a causal source whose every layer reads key runs, its rows
    asking a request's questions over one state (`shared_state`), trains a step; the task loaded from its run
    lays a request's three questions in one row and answers each with one of its options. Over a source whose
    recurrent layers read no key runs (Qwen3.5's gated delta net), the run is refused, as its rows' streams
    would be read as one sequence (until a document enters from a given memory, row 1506)."""
    from run_kinds import QWEN_TINY, lettered, trained
    from test_decision_streams import GEMMA

    from dew.data import Memory
    from dew.decision import Decide, decisions

    rows = Memory([{"example": example} for example in examples()]).read("memory").pipe(decisions).batch(8)
    run = dataclasses.replace(lettered(tmp_path, drawn=False), data=rows, validation=None, shared_state=3)
    assert run.pretrained == str(QWEN_TINY)
    with pytest.raises(ValueError):
        trained(run, tmp_path, "hybrid")
    run = dataclasses.replace(run, pretrained=str(GEMMA))
    trained(run, tmp_path, "shared", trainer={"checkpoint_final": True})
    decide = Decide.from_run(str(tmp_path / "runs" / "shared"))
    assert decide.layout.shared_state == 3
    [row] = decide.layout.rows(decide.tokenizer, decide.specials, STATES[0], QUESTIONS)
    assert [laid.name for laid in row.questions] == list(QUESTIONS)
    answers = decide(STATES[0], QUESTIONS)
    assert all(answers[name].choice in question.options for name, question in QUESTIONS.items())


class Merging(ByteTokenizer):
    """Bytes, but for "B. y", one token (256): a lettered row's tokens then depend on which option its second
    slot shows, as a BPE vocabulary's do."""

    MERGED = b"B. y"

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        data, ids, at = text.encode(), [], 0
        while at < len(data):
            merged = data.startswith(self.MERGED, at)
            ids.append(256 if merged else data[at])
            at += len(self.MERGED) if merged else 1
        return ids

    def decode(self, ids) -> str:
        return b"".join(self.MERGED if int(id_) == 256 else bytes([int(id_)]) for id_ in ids).decode(
            errors="replace")


YES_NO = {"first": Choice("First?", ["no", "yes"]), "second": Choice("Second?", ["no", "yes"])}


def test_a_draw_of_option_orders_that_splits_a_row_over_one_state_lays_it_as_fit_admitted():
    """A row over one state that fits `max_len` in its questions' own option orders and would not in a
    drawn one (a vocabulary that reads "B. yes" as fewer tokens than "B. no") is laid out, as Fit admitted
    it, in one row of both questions, whatever order a training read draws (review of state-once)."""
    tokenizer = Merging()
    example = Example(STATES[0], YES_NO, {"first": "yes", "second": "yes"})
    [row] = SHARED.rows(tokenizer, SPECIALS, STATES[0], YES_NO)
    fitting = dataclasses.replace(SHARED, max_len=len(row.tokens))
    assert len(fitting.rows(tokenizer, SPECIALS, STATES[0], YES_NO, orders={"first": [1, 0]})) == 2
    encoding = Encoding(fitting, tokenizer, SPECIALS, width=4, shuffle=True)
    for seed in range(16):
        encoded = encoding(example, list(YES_NO), np.random.default_rng(seed))
        assert len(encoded["tokens"]) == len(row.tokens)
        assert int(np.sum(np.any(encoded["options"], axis=-1))) == len(YES_NO)


def test_a_none_of_the_above_that_overfills_a_row_over_one_state_is_declined_where_it_is_fitted():
    """Fit counts a "none of the above" whose added option leaves a row over one state no room as one row as
    the read's declined draw (`unfit`), the row admitted as it is (review of state-once)."""
    example = Example(STATES[0], YES_NO, {"first": "yes", "second": "yes"})
    [row] = SHARED.rows(TOKENIZER, SPECIALS, STATES[0], YES_NO)
    fitting = dataclasses.replace(SHARED, max_len=len(row.tokens))
    head = LabelReadout(features=8, label_ids=(), count=4)
    [fitted] = Fit(fitting, TOKENIZER, SPECIALS, head, none_of_the_above=1.0).map(
        {"example": example})["decision_rows"]
    assert fitted.names == tuple(YES_NO) and fitted.declined == "unfit"


def test_a_head_that_reads_across_its_rows_questions_refuses_a_row_over_one_state():
    """The joint head's own layers attend across a row's questions, which a row over one state keeps apart:
    its rows are refused where they are fitted; a letter readout, which reads each question's own answer
    state, takes them (review of state-once)."""
    [example] = examples()[:1]
    joint = JointSchemaHead(32, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)
    with pytest.raises(ValueError):
        Fit(SHARED, TOKENIZER, SPECIALS, joint).map({"example": example})
    readout = LabelReadout(features=8, label_ids=(), count=4)
    assert len(Fit(SHARED, TOKENIZER, SPECIALS, readout).map({"example": example})["decision_rows"]) == 1


def test_a_row_over_one_state_writes_its_state_before_its_question_and_its_instructions_first():
    """A layout whose rows write the state after the question (a question-first or a re-reading order), or
    the options before the instructions, shares no part of the state its questions read alike: it is refused
    with shared_state set (review of state-once)."""
    from dew.decision.layout import MarkerLayout

    first, state, instructions, options, cue = OWN.segments
    with pytest.raises(ValueError):
        dataclasses.replace(OWN, shared_state=2, segments=(first, state, options, instructions, cue))
    with pytest.raises(ValueError):
        MarkerLayout(shared_state=2)
    with pytest.raises(ValueError):
        dataclasses.replace(OWN, shared_state=2, segments=None, order="re_read")
