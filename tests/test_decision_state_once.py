"""A row of a request's questions over one copy of its state (`QuestionLayout.shared_state`): what each
question's own row writes before its question, written once, and each question's own part after it as a
document continuing it (`dew.batch.Forks`), each at the positions its own row lays it at; each question
reads that shared part and itself, so it decides as its own row does, its request's tokens the state's once
and each question's."""

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
    HOWS,
    AnswerCue,
    DecisionInputs,
    Instructions,
    OptionLines,
    QuestionLayout,
    State,
    StateFirstLayout,
    Text,
)
from dew.decision.model import DecisionModel, stream_runs
from dew.decision.rows import Encoding, Fit
from dew.decision.shared_state import fork_depth
from dew.decision.task import laid_out
from dew.nn.attention_masks import visibility_mask
from dew.objectives.base import Step

pytestmark = pytest.mark.mesh

OWN = StateFirstLayout(
    labels=LABELS, max_len=256,
    segments=(Text("STATE:\n"), State(), Instructions({"choice": "\n{}"}),
              OptionLines(line="\n{label}. {option}", bare="\n{label}. {option}", join="",
                          noul_first="false"),
              AnswerCue(template="\nAnswer:", many="\nAnswer:")))
"""A lettered row of a question of its own, its state first."""
OWN = dataclasses.replace(OWN, shared_state=1)
SHARED = dataclasses.replace(OWN, shared_state=3)
QUESTIONS = {"team": Choice("Which team handles it?", ["cards", "loans", "fraud", "none"]),
             "way": Choice("Which way is it?", ["up", "down", "left", "right"]),
             "urgent": Choice("How urgent is it?", ["now", "today", "this week", "later"])}
"""Questions of four options each, the readout's count, so each question's logits are a row of four."""
STATE_ONCE_SEEDS = 16
"""Check 1's declared seeds for rows over one state against each question's own row: at 8, as the packed
rows' of two streams (`STREAM_SEEDS`), a question of two options' rows were left unresolved over 40."""


def own_and_shared(layout=SHARED, state: str = STATES[2], questions=QUESTIONS):
    """The request's rows of a question each (`OWN`) and over one state (`layout`)."""
    return (OWN.rows(TOKENIZER, SPECIALS, state, questions),
            layout.rows(TOKENIZER, SPECIALS, state, questions))


def test_a_row_over_one_state_holds_it_once_and_each_question_as_its_own_row_lays_it():
    """The row holds what the questions' own rows write before their question once, its document 1, and
    each question's own part after it, a document continuing it in their order: each token, its position,
    the question's spans and its readout, the ones its own row holds, moved along the row."""
    own, [row] = own_and_shared()
    shared = own[0].questions[0].span[0]
    tokens, positions, documents = (np.asarray(field) for field in (row.tokens, row.positions, row.forked))
    assert list(tokens[documents == 1]) == list(own[0].tokens[:shared])
    assert len(tokens) == shared + sum(len(each.tokens) - shared for each in own)
    for document, (alone, laid) in enumerate(zip(own, row.questions, strict=True), start=2):
        part = documents == document
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


def test_each_question_reads_the_shared_part_and_itself():
    """The forks a row over one state lays out (`DecisionInputs.coordinates`): a token of the shared part
    reads it up to itself; a question's token reads the shared part whole and its own part up to itself,
    and no other question's."""
    _, [row] = own_and_shared()
    inputs = DecisionInputs.collate([row], SPECIALS.pad)
    laid = inputs.coordinates()
    assert laid.forks is not None and laid.segments is not None
    runs = laid.forks.runs(laid.segments)
    length = inputs.tokens.shape[1]
    seen = np.asarray(visibility_mask(length, length, causal=False, sliding_window=None, key_runs=runs))[0, 0]
    documents = np.asarray(row.forked)
    index = np.arange(len(documents))
    for query, document in enumerate(documents):
        read = (documents == 1) & ((index <= query) if document == 1 else True)
        read |= (documents == document) & (index <= query)
        assert list(seen[query, :len(documents)]) == list(read), query


def test_a_question_with_siblings_reads_the_questions_before_it_in_its_row_whole():
    """With siblings (`QuestionLayout.siblings`, a layout a run saved before rows forked states), each
    question reads the shared part whole, the questions before it in its row whole and itself up to each
    token, as the layout's streamed reads read it before (each own stream causal, the shared stream whole
    and with siblings the earlier questions' whole), each token at its own row's position; a row whose
    window packs it lays the same forks (`PackedRows`), and a record of the layout reads back as it is.
    The questions' documents chain, so the forks run as deep as a row's questions (`fork_depth`)."""
    from dew import registry

    chained = dataclasses.replace(SHARED, siblings=True)
    assert registry.from_record(StateFirstLayout, registry.to_record(chained, StateFirstLayout)) == chained
    _, [plain] = own_and_shared()
    _, [row] = own_and_shared(chained)
    assert (row.tokens, row.positions, row.forked) == (plain.tokens, plain.positions, plain.forked)
    inputs = DecisionInputs.collate([row], SPECIALS.pad)
    laid = inputs.coordinates()
    assert laid.forks is not None and laid.segments is not None and laid.forks.depth >= len(row.questions)
    # One row read together, its dropout drawn once, one question of it or several (`Forks.joint`).
    _, [alone] = own_and_shared(chained, questions=dict(list(QUESTIONS.items())[:1]))
    assert laid.forks.joint and DecisionInputs.collate([alone], SPECIALS.pad).coordinates().forks.joint
    runs = laid.forks.runs(laid.segments)
    length = inputs.tokens.shape[1]
    seen = np.asarray(visibility_mask(length, length, causal=False, sliding_window=None, key_runs=runs))[0, 0]
    documents = np.asarray(row.forked)
    index = np.arange(len(documents))
    for query, document in enumerate(documents):
        read = (documents == 1) & ((index <= query) if document == 1 else True)
        read |= (documents > 1) & (documents < document)
        read |= (documents == document) & (index <= query)
        assert list(seen[query, :len(documents)]) == list(read), query
    with pytest.raises(ValueError):
        dataclasses.replace(OWN, shared_state=None, siblings=True)


def test_sibling_questions_packed_a_window_read_each_their_own_requests_context():
    """Requests' rows with siblings, two a window, each lay one tree of documents, its shared part and its
    questions chained after it (`Forks.trees`), numbered as the window numbers its rows: what a token's
    cross-attention reads its context by (`CrossContext.queries`), so each question reads its own
    request's context and no other's, as the rows' streams did before rows forked."""
    from test_decision_context import windows_of

    from dew.batch import Forks

    # Windows long enough to pack two requests' rows, each holding a request's three questions.
    chained = in_streams(dataclasses.replace(SHARED, siblings=True, max_len=1024), segments=2)
    windows = windows_of(chained, examples(), count=4)
    assert any(len(set(np.asarray(batch["segments"]).ravel()) - {0}) > 1 for batch in windows)
    for batch in windows:
        forks = Forks(np.asarray(batch["parents"]), depth=fork_depth(chained.layout))
        trees = forks.trees(np.asarray(batch["documents"]), xp=np)
        np.testing.assert_array_equal(trees, np.asarray(batch["segments"]))


def test_a_layout_of_its_own_reads_that_writes_a_stream_twice_reads_by_its_streams():
    """A row whose streams read as a row over one state's do, written by a layout of its own reads in several
    runs a stream (`fragment_count`), reads every run of the shared part."""
    streams = jnp.asarray([[0, 0, 1, 1, 0]], jnp.int32)
    segments = jnp.ones((1, 5), jnp.int32)
    runs = stream_runs(streams, segments, ((HOWS["causal"], 0, 0), (HOWS["causal"], HOWS["whole"], 0)), 3)
    seen = np.asarray(visibility_mask(5, 5, causal=False, sliding_window=None, key_runs=runs))[0, 0]
    assert list(seen[3]) == [True, True, True, True, True]

def examples() -> list[Example]:
    """Each state asked every question, answered with its first option."""
    return [Example(state, QUESTIONS, {name: question.options[0] for name, question in QUESTIONS.items()})
            for state in STATES]


def scored(objective, windows, variables) -> dict[str, np.ndarray]:
    """Each question's real options' logits, keyed by its request's shared part's tokens and its own
    part's: a row of its own is both, a row over one state its shared document and the question's."""
    forward = without_ynn(lambda variables, inputs: objective.model.logits(variables, inputs))
    logits = {}
    for batch in windows:
        inputs = laid_out(batch, objective.bounds.segments, attention=objective.layout.attention)
        held = np.asarray(forward(variables, inputs))
        options, owners = np.asarray(batch["options"]), np.asarray(batch["question_rows"])
        tokens, segments = np.asarray(batch["tokens"]), np.asarray(batch["segments"])
        # Each token's part of its row: its document, or its stream where its rows stand in streams.
        marked = next((name for name in ("documents", "streams") if name in batch), None)
        parts = np.zeros_like(tokens) if marked is None else np.asarray(batch[marked])
        for window, owner in enumerate(owners):
            for row in range(1, int(segments[window].max()) + 1):
                own = segments[window] == row
                held_parts = np.unique(parts[window][own])
                for number, slot in enumerate(np.flatnonzero(owner == row), start=1):
                    part = own & (np.isin(parts[window], held_parts[[0, number]])
                                  if marked is not None else True)
                    key = bytes(tokens[window][part].astype(np.uint8)).decode(errors="replace")
                    logits[key] = held[window][slot][options[window][slot]]
    return dict(sorted(logits.items()))


def test_a_row_over_one_state_decides_each_question_as_its_own_row_does():
    """Rows of a request's questions over one state, two requests a window, decide each question's options as
    its own row does in a window of its own, through the unmodified objective (check 1, over drawn
    parameters, float64 truth)."""
    shared = in_streams(SHARED, segments=2)
    assert isinstance(shared.layout, QuestionLayout) and shared.layout.shares
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


def test_a_stated_shared_layout_read_by_an_earlier_plan_lays_its_rows_in_streams_and_decides_alike():
    """A layout that states its `shared_state`, read by a data plan from before rows forked a state
    (`dew.data.plan.SHARED_STATE`), lays its rows as that plan's runs wrote them: each window's rows in
    streams at their own rows' positions and no documents, the same rows and tokens this Dew's plan packs,
    so a run saved under such a plan rebuilds the data plan it saved (`DecisionObjective.data`,
    `QuestionLayout.streamed`); and the objective bound to that plan (`bind_plan`) decides each question
    through those rows as its own row does (check 1, float64 truth). The earlier plan's rows were laid as
    documents continuing a shared part, another plan's steps, and its checkpoint was refused (the
    promotion's final review, finding 2)."""
    from dew.data.plan import IN_ORDER

    shared = in_streams(SHARED, segments=2)
    shared.bind_plan(IN_ORDER)
    alone = in_streams(OWN, segments=1)
    rows = examples()
    earlier = windows_of(shared, rows, count=8, plan_version=IN_ORDER)
    latest = windows_of(in_streams(SHARED, segments=2), rows, count=8)
    assert len(earlier) == len(latest) > 1
    for written, forked in zip(earlier, latest, strict=True):
        assert {"streams", "positions"} <= written.keys() and not {"documents", "parents"} & written.keys()
        for field in ("tokens", "segments", "positions"):
            np.testing.assert_array_equal(written[field], forked[field])
    windows = {"shared": earlier, "alone": windows_of(alone, rows, count=24)}
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
                    lambda seed: (initialized(jax.random.key(seed)),), "rows in streams over one state",
                    seeds=STATE_ONCE_SEEDS, truth_fn=truth)


def shared_parts(batch) -> list[str]:
    """What each row of a window writes before its question, as text: its shared document where its rows
    ask over one state, the row whole otherwise (its own row, which opens with it)."""
    [tokens], [segments] = np.asarray(batch["tokens"]), np.asarray(batch["segments"])
    documents = np.asarray(batch["documents"])[0] if "documents" in batch else None
    parts = []
    for row in range(1, int(segments.max()) + 1):
        own = segments == row
        if documents is not None:
            own &= documents == documents[own].min()
        parts.append(bytes(tokens[own].astype(np.uint8)).decode(errors="replace"))
    return parts


def input_terms(objective, windows, variables, requests: list[list[str]]) -> tuple[np.ndarray, int]:
    """The rows' own next-token term (`input_lm`, an evaluation's read) over `windows`, summed by the group
    of `requests` (each group the shared parts of some requests) its rows' requests are of, and the
    targets it counts in all."""
    step = Step(jnp.asarray(0), jax.random.key(1), None, training=False)
    term = without_ynn(lambda variables, batch: objective.loss(variables, batch, step)[0].terms[-1])
    totals, mass = np.zeros(len(requests), np.float64), 0.0
    for batch in windows:
        scored_window = term(variables, batch)
        [group] = {index for index, shared in enumerate(requests)
                   for row in shared_parts(batch) if any(row.startswith(part) for part in shared)}
        totals[group] += float(np.asarray(scored_window.total, np.float64))
        mass += float(np.asarray(scored_window.mass))
    return totals, int(mass)


def test_rows_over_one_state_score_the_input_term_each_questions_own_row_scores():
    """With `input_lm`, rows asking a request's questions over one state score the term the same questions'
    own rows score: each question's tokens from the shared part and its own part before them, never
    another question's, and the shared part's targets once for each question asked over it; so the
    targets counted are the own rows', to the token, and each window's summed cross entropy its requests'
    own rows' (check 1, over drawn parameters, float64 truth). A merged row read as one causal document
    conditioned each question after its row's first on the one before it and counted the shared part once
    (the promotion's final review, finding 5: sharing is derived for a letter readout, whose input term
    then changed)."""
    shared = in_streams(SHARED, segments=2, input_lm=0.5)
    alone = in_streams(OWN, segments=1, input_lm=0.5)
    # Three requests of each state, told apart by their texts, so a window's term is a value of many.
    rows = [Example(f"{example.state} ({mark})", example.questions, example.answers)
            for mark in "abc" for example in examples()]
    windows = {"shared": windows_of(shared, rows, count=len(rows)),
               "alone": windows_of(alone, rows, count=len(rows) * len(QUESTIONS))}
    assert all("documents" in batch for batch in windows["shared"]), "a window shares no state"
    requests = [shared_parts(batch) for batch in windows["shared"]]
    assert sum(map(len, requests)) == len(rows)
    initialized = jax.jit(lambda key: unboxed(shared.init(key, variables={})))
    wide = copy.copy(alone)
    wide.model = DecisionModel(alone.model.backbone.clone(dtype=jnp.float64),
                               alone.model.head.clone(dtype=jnp.float64), alone.model.context_encoder)
    drawn = initialized(jax.random.key(0))
    assert (input_terms(shared, windows["shared"], drawn, requests)[1]
            == input_terms(alone, windows["alone"], drawn, requests)[1])

    @as_given
    def truth(variables):
        with jax.enable_x64(new_val=True):
            return input_terms(wide, windows["alone"], widened(variables), requests)[0]

    assert_as_exact(
        as_given(lambda variables: input_terms(shared, windows["shared"], variables, requests)[0]),
        as_given(lambda variables: input_terms(alone, windows["alone"], variables, requests)[0]),
        lambda seed: (initialized(jax.random.key(seed)),), "the input term of rows over one state",
        seeds=STATE_ONCE_SEEDS, truth_fn=truth)


def test_a_run_over_one_state_trains_and_its_task_asks_a_requests_questions_in_one_row(tmp_path):
    """A decision run of a letter readout, its rows asking a request's questions over one state
    (`shared_state`), trains a step over a hybrid source (Qwen3.5's gated delta net beside attention, each
    question's documents from the state the shared part leaves) and over an attention source; the task
    loaded from its run lays a request's three questions in one row and answers each with one of its
    options."""
    from run_kinds import QWEN_TINY, lettered, shared, trained
    from test_decision_streams import GEMMA

    from dew.data import Memory
    from dew.decision import Decide, decisions

    rows = Memory([{"example": example} for example in examples()]).read("memory").pipe(decisions).batch(8)
    run = dataclasses.replace(lettered(tmp_path, drawn=False), data=rows, validation=None, shared_state=3)
    assert run.pretrained == str(QWEN_TINY)
    trained(run, tmp_path, "hybrid")
    # Over a vocabulary that spells A to D as one unknown token, labels it spells apart (`run_kinds.shared`).
    run = dataclasses.replace(shared(tmp_path, drawn=False), data=rows, validation=None)
    assert run.pretrained == str(GEMMA)
    trained(run, tmp_path, "shared", trainer={"checkpoint_final": True})
    decide = Decide.from_run(str(tmp_path / "runs" / "shared"))
    assert decide.layout.shared_state == 3
    [row] = decide.layout.rows(decide.tokenizer, decide.specials, STATES[0], QUESTIONS)
    assert [laid.name for laid in row.questions] == list(QUESTIONS)
    answers = decide(STATES[0], QUESTIONS)
    assert all(answers[name].choice in question.options for name, question in QUESTIONS.items())


def test_a_run_read_by_a_plan_from_before_rows_shared_a_state_declares_a_question_a_row(tmp_path):
    """A lettered decision run whose layout states no sharing, read by a data plan from before a request's
    questions shared a state (`dew.data.plan.SHARED_STATE`), trains a question a row, and the task loaded
    from the step it saves lays a request's three questions in three rows, as it trained; the same run
    under this Dew's plan lays them in one row over one state (`DecisionObjective.bind_plan`: a saved run
    resumed under its earlier plan declared, at the step the resume saved, the sharing the code derived,
    so its task forked the requests of a run that read none)."""
    from run_kinds import lettered, trained

    from dew.data import Memory
    from dew.data.plan import IN_ORDER
    from dew.decision import Decide, decisions

    rows = Memory([{"example": example} for example in examples()]).read("memory").pipe(decisions).batch(8)
    run = dataclasses.replace(lettered(tmp_path, drawn=False), data=rows, validation=None)
    for name, version, laid in (("earlier", IN_ORDER, len(QUESTIONS)), ("shared", None, 1)):
        trained(dataclasses.replace(run, data_plan_version=version), tmp_path, name,
                trainer={"checkpoint_final": True})
        decide = Decide.from_run(str(tmp_path / "runs" / name))
        assert len(decide.layout.rows(decide.tokenizer, decide.specials, STATES[0], QUESTIONS)) == laid, name


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
    head = LabelReadout(features=8, label_ids=LABELS.ids(TOKENIZER, 4), count=4)
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
    readout = LabelReadout(features=8, label_ids=LABELS.ids(TOKENIZER, 4), count=4)
    assert len(Fit(SHARED, TOKENIZER, SPECIALS, readout).map({"example": example})["decision_rows"]) == 1


def test_a_row_over_one_state_writes_its_state_before_its_question_and_its_instructions_first():
    """A layout whose rows write the state after the question (a question-first or a re-reading order), or
    the options before the instructions, shares no part of the state its questions read alike: it is refused
    with several questions a row asked, and lays a question a row by default (review of state-once)."""
    from dew.decision.layout import MarkerLayout

    first, state, instructions, options, cue = OWN.segments
    with pytest.raises(ValueError):
        dataclasses.replace(OWN, shared_state=2, segments=(first, state, options, instructions, cue))
    with pytest.raises(ValueError):
        MarkerLayout(shared_state=2)
    with pytest.raises(ValueError):
        dataclasses.replace(OWN, shared_state=2, segments=None, order="re_read")
    # A layout that states no sharing asks a question a row, as before rows shared a state, until what
    # reads its rows states it (`shared_for`); one that states it shares.
    assert not MarkerLayout().shares and not dataclasses.replace(OWN, shared_state=None).shares
    assert dataclasses.replace(OWN, shared_state=2).shares

