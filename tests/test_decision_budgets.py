"""A decision row's budgets (row 2006). A fresh layout keeps every question whole, the state taking what
the question leaves of max_len, at real intent-classification sizes (77 and 151 intents): each
option's line stands in the row as its text, told apart from every other. Only a declared budget cuts a
question, as Laya's release declares 192 and 48 tokens, and every cut is counted by what it cut
(instructions, options, state, a context no cross-attention reads), in source examples, rows and
tokens, by a run's data counters and a server's alike. A question that alone does not fit is refused as
unfit, by training and a server alike; a cut that leaves two options alike is a row training leaves out
under its own reason, which a server answers as its release does, the cut in its usage."""

import dataclasses
from pathlib import Path

import jax
import numpy as np
import pytest
from real_tokenizers import TOKENIZERS, backbone_reading, real_tokenizer

from dew.data.dataset import Loading
from dew.data.pipeline import Refused
from dew.data.sources import Memory
from dew.data.text import ByteTokenizer, HFTokenizer
from dew.decision import Choice, Decide, Example, Noul
from dew.decision.config import DecisionRunConfig, Letters
from dew.decision.data import GroupedExamples, decisions
from dew.decision.head import DecisionHead
from dew.decision.layout import Labels, MarkerLayout, Specials, StateFirstLayout
from dew.decision.rows import Fit, _none_of_the_above

LAYA = {"head_max_len": 192, "option_tokens": 48}
"""Laya's release's declared question budgets."""
BYTES = ByteTokenizer()
BYTE_SPECIALS = Specials(begin=254, separator=253, marker=0, marker_text="\x00", pad=255)
BYTE_LABELS = Labels(names=tuple(chr(index) for index in range(33, 110)), variants=("{}",))
"""77 labels a byte vocabulary spells as one token each."""


@pytest.fixture(scope="module", params=TOKENIZERS)
def tokenized(request, tmp_path_factory) -> tuple[HFTokenizer, Path]:
    """A real tokenizer (GPT-2's in every run, Qwen3.5-0.8B's where the Hub is reachable) and a tiny
    Qwen3.5 that reads it (`backbone_reading`)."""
    directory = tmp_path_factory.mktemp(request.param) / "backbone"
    return real_tokenizer(request.param), backbone_reading(request.param, directory)


@pytest.fixture(scope="module")
def tokenizer(tokenized) -> HFTokenizer:
    return tokenized[0]


@pytest.fixture(scope="module")
def backbone(tokenized) -> Path:
    return tokenized[1]


def intents(count: int) -> Choice:
    """A question of `count` intents, each its own name, as BANKING77's and CLINC150's labels are."""
    return Choice("Which intent does the customer's message express?",
                  [f"intent_{index}_of_the_customer" for index in range(count)])


STATE = "I ordered a new card two weeks ago and it still has not arrived, can you check where it is? " * 100
"""A state longer than any row here, which the row cuts to the room its question leaves."""


@pytest.mark.parametrize("count", [77, 151])
@pytest.mark.parametrize("first", ["state", "question"])
def test_a_fresh_lettered_row_holds_every_option_of_a_large_question_whole(tokenizer, count, first):
    """A fresh lettered layout lays out a question of 77 or 151 intents whole, the state first or the
    question first: every option's line is its label and its whole name, each told apart from the
    others, nothing of the question is counted cut, and the state takes what the question leaves of
    max_len, its cut counted."""
    labels = Labels().spelled(tokenizer, count)
    layout = (StateFirstLayout if first == "state" else MarkerLayout)(max_len=2048, labels=labels)
    question = intents(count)
    [row] = layout.rows(tokenizer, Specials.of(tokenizer), STATE, {"intent": question})
    names = labels.names_for(count)
    lines = [tokenizer.decode(row.tokens[start:end]) for start, end in row.questions[0].options]
    assert lines == [f"\n{label}. {option}" for label, option in zip(names, question.options, strict=True)]
    assert not row.trimmed and set(row.cuts) == {"state"}
    assert row.state_cut == row.state_tokens - row.state_kept
    assert layout.refusal(row) is None and len(row.tokens) <= layout.max_len


@pytest.mark.parametrize("count", [77, 151])
def test_a_fresh_marker_row_holds_every_option_of_a_large_question_whole(tokenizer, count):
    """A fresh marker layout, a bidirectional backbone's, holds every option of a question of 77 or 151
    intents whole, each its own text before its marker, and nothing of the question is counted cut."""
    layout = MarkerLayout(max_len=2048)
    question = intents(count)
    [row] = layout.rows(tokenizer, Specials.of(tokenizer), STATE, {"intent": question})
    markers = [start for start, _ in row.questions[0].options]
    # Each option's text runs from its marker to the next; the last's, its own length on.
    last = len(tokenizer.encode(f" {question.options[-1]}", add_special_tokens=False))
    ends = [*markers[1:], markers[-1] + 1 + last]
    texts = [tokenizer.decode(row.tokens[start + 1:end]) for start, end in zip(markers, ends, strict=True)]
    assert texts == [f" {option}" for option in question.options]
    assert not row.trimmed and set(row.cuts) == {"state"} and layout.refusal(row) is None


def test_a_question_that_alone_passes_max_len_is_refused_as_unfit_never_cut(tokenizer):
    """A question of 151 intents in a row of 256 tokens is laid out whole and refused as unfit, by
    training and a server alike (`Layout.refusal`), never cut to fit."""
    lettered = StateFirstLayout(max_len=256, labels=Labels().spelled(tokenizer, 151))
    for layout in (lettered, MarkerLayout(max_len=256)):
        [row] = layout.rows(tokenizer, Specials.of(tokenizer), "short", {"intent": intents(151)})
        assert layout.refusal(row) == "unfit" and not row.trimmed and len(row.questions[0].options) == 151


def test_a_declared_budget_counts_each_cut_and_refuses_a_cut_that_leaves_options_alike(tokenizer):
    """Laya's declared budgets cut a question of 151 intents to the four tokens a line each, which
    leave every option its label alone: the row counts its options cut, and is refused as its options
    read alike. A question of three long options under the same budgets keeps them apart, cut to 48
    tokens each and counted, and is laid out."""
    labels = Labels().spelled(tokenizer, 151)
    declared = StateFirstLayout(max_len=2048, labels=labels, **LAYA)
    [cut] = declared.rows(tokenizer, Specials.of(tokenizer), "short", {"intent": intents(151)})
    assert cut.options_cut > 0 and cut.options_alike and declared.refusal(cut) == "options_alike"
    long = Choice("Which team handles it?", {
        team: f"{team} handles " + " and ".join(f"{team} matter number {index}" for index in range(30))
        for team in ("billing", "technical", "sales")})
    [kept] = declared.rows(tokenizer, Specials.of(tokenizer), "short", {"team": long})
    whole = StateFirstLayout(max_len=8192, labels=labels)
    whole_lines = whole.rows(tokenizer, Specials.of(tokenizer), "short", {"team": long})[0]
    assert kept.options_cut == sum(end - start for start, end in whole_lines.questions[0].options) - sum(
        end - start for start, end in kept.questions[0].options) > 0
    assert not kept.options_alike and declared.refusal(kept) is None


def test_fit_counts_each_cut_by_kind_and_none_where_the_question_is_whole():
    """Fit reports the cuts of an example's rows (`_dew_cut`: by kind, the rows each shortened and the
    tokens it removed): none of the question at a fresh layout's budgets, where a question of 77 intents
    was once cut to its labels with nothing reported; the options and instructions a declared budget
    cuts; and a context the backbone reads none of."""
    questions = {"q": Choice("Choose the banking intent", {
        f"intent_{index}": f"the full meaning of banking intent {index}" for index in range(77)})}
    example = Example("the state", questions, {"q": "intent_0"})
    fresh = StateFirstLayout(max_len=6144, labels=BYTE_LABELS)
    whole = Fit(fresh, BYTES, BYTE_SPECIALS, DecisionHead(64)).map({"example": example})
    assert whole["_dew_cut"] == {}
    declared = dataclasses.replace(fresh, head_max_len=4096, option_tokens=24)
    [row] = declared.rows(BYTES, BYTE_SPECIALS, example.state, questions)
    cut = Fit(declared, BYTES, BYTE_SPECIALS, DecisionHead(64)).map({"example": example})
    assert cut["_dew_cut"] == {"options": (1, row.options_cut)} and row.options_cut > 0
    read = dataclasses.replace(example, context="a document the backbone reads no part of")
    unread = Fit(fresh, BYTES, BYTE_SPECIALS, DecisionHead(64)).map({"example": read})
    assert unread["_dew_cut"] == {"context": (1, len("a document the backbone reads no part of"))}


def test_a_context_no_cross_attention_reads_counts_as_a_cut_of_it():
    """An example's context, which a backbone without cross-attention leaves unread, is counted as a cut
    of all its tokens, the row trained without it, where it was once dropped with no word."""
    from dew.decision.layout import context_tokens

    example = Example("short", {"q": Choice("Pick", ["a", "b"])}, {"q": "a"}, context="the account history")
    layout = StateFirstLayout(max_len=128, labels=BYTE_LABELS)
    fitted = Fit(layout, BYTES, BYTE_SPECIALS, DecisionHead(64)).map({"example": example})
    unread = len(context_tokens(BYTES, "the account history", None))
    assert unread > 0 and fitted["_dew_cut"]["context"] == (1, unread)


def _drawn(layout, example, draws: int) -> list[dict]:
    """`draws` training reads of `example`, each adding a "none of the above" (rate 1), its row as Fit
    admitted it."""
    from dew.decision.rows import Encoding

    [fitted] = Fit(layout, BYTES, BYTE_SPECIALS, DecisionHead(64)).map({"example": example})["decision_rows"]
    encoding = Encoding(layout, BYTES, BYTE_SPECIALS, width=4, none_of_the_above=1.0, shuffle=False)
    rng = np.random.default_rng(0)
    return [encoding(example, ["q"], rng, fitted.tokens) for _ in range(draws)]


def _options(row: dict) -> list[tuple[int, ...]]:
    """Each real option's text in an encoded row of Laya's markers, which writes it after its marker and
    before the next marker or the separator that ends the options."""
    tokens = list(row["tokens"])
    markers = [int(start) for (start, _), real in zip(row["option_spans"][0], row["options"][0], strict=True)
               if real]
    ends = [*markers[1:], tokens.index(BYTE_SPECIALS.separator, markers[-1])]
    return [tuple(tokens[start + 1:end]) for start, end in zip(markers, ends, strict=True)]


def test_a_none_of_the_above_draw_never_leaves_a_row_unfit():
    """A "none of the above" that would leave its question alone past the window lays the question out as
    Fit admitted it, where the encoding refused the row before; a draw whose row fits (one replacing the
    right option, shorter by it) stands (the control). The row is counted as one whose option a read
    declines, under that reason, in its corpus's counters; an example whose option fits counts none."""
    from dew.data.compile import compile
    from dew.data.pipeline import ObjectiveData

    question = Choice("Which?", ["abca", "abcb", "abcc"])
    # The question with the option added takes 95 tokens with no state, with it in place of "abca" 89.
    layout = MarkerLayout(max_len=90)
    example = Example("short", {"q": question}, {"q": "abca"})
    [admitted] = layout.rows(BYTES, BYTE_SPECIALS, example.state, example.questions)
    added, _ = _none_of_the_above(question, 0, drop=False)
    [augmented] = layout.rows(BYTES, BYTE_SPECIALS, example.state, {"q": added})
    assert layout.refusal(augmented) == "unfit" and layout.refusal(admitted) is None
    rows = _drawn(layout, example, 16)
    assert all(len(row["tokens"]) <= layout.max_len for row in rows)
    assert any(np.array_equal(row["tokens"], admitted.tokens) for row in rows)
    assert any(not np.array_equal(row["tokens"], admitted.tokens) for row in rows)
    fit = Fit(layout, BYTES, BYTE_SPECIALS, DecisionHead(64), none_of_the_above=1.0)
    roomy = Fit(MarkerLayout(max_len=128), BYTES, BYTE_SPECIALS, DecisionHead(64), none_of_the_above=1.0)
    for fitting, declined in ((fit, {"unfit": 1}), (roomy, {})):
        loader = compile(Memory([{"example": example}]).batch(1), read="memory",
                         objective=ObjectiveData(train=(fitting,)))
        [counts] = loader.counts.values()
        assert counts.accepted == 1 and counts.declined == declined


def test_a_none_of_the_above_draw_never_leaves_the_options_alike():
    """Under a declared budget an added option shares, a draw that would cut the options alike lays the
    question out as Fit admitted it: every read's options are distinct, where the same augmentation laid
    out alone leaves them alike; a draw whose options stay apart stands (the control)."""
    question = Choice("Which?", ["abca", "abcb", "abcc"])
    layout = MarkerLayout(max_len=128, head_max_len=34)
    example = Example("short", {"q": question}, {"q": "abca"})
    [admitted] = layout.rows(BYTES, BYTE_SPECIALS, example.state, example.questions)
    assert layout.refusal(admitted) is None
    added, _ = _none_of_the_above(question, 0, drop=False)
    [augmented] = layout.rows(BYTES, BYTE_SPECIALS, example.state, {"q": added})
    assert augmented.options_alike
    rows = _drawn(layout, example, 32)
    assert all(len(set(_options(row))) == len(_options(row)) for row in rows)
    assert any(not np.array_equal(row["tokens"], admitted.tokens) for row in rows)
    fit = Fit(layout, BYTES, BYTE_SPECIALS, DecisionHead(64), none_of_the_above=1.0)
    assert fit.map({"example": example})["_dew_declined"] == {"options_alike": 1}


def test_a_grouped_example_counts_the_members_it_leaves_out():
    """Grouped examples trained apart keep the members that fit and count the ones they leave out under
    each one's reason (`_dew_rejected`), where a kept sibling once hid them."""
    small = Example("short", {"q": Choice("Pick", ["a", "b"])}, {"q": "a"})
    large = Example("short", {"q": Choice("Pick", [f"option {index}" for index in range(60)])},
                    {"q": "option 0"})
    layout = StateFirstLayout(max_len=128, labels=BYTE_LABELS)
    grouped = {"example": GroupedExamples((small, large))}
    fitted = Fit(layout, BYTES, BYTE_SPECIALS, DecisionHead(64)).map(grouped)
    assert len(fitted["decision_rows"]) == 1 and fitted["_dew_rejected"] == {"unfit": 1}
    # A memory read observes its rows as it prepares them and once more as it holds them: the member
    # it left out counts once.
    from dew.data.compile import compile
    from dew.data.pipeline import ObjectiveData

    loader = compile(Memory([grouped]).batch(1), read="memory",
                     objective=ObjectiveData(train=(Fit(layout, BYTES, BYTE_SPECIALS, DecisionHead(64)),)))
    [counts] = loader.counts.values()
    assert counts.left_out == {"unfit": 1} and counts.refused == {"unfit": 1}


class Fitted(DecisionRunConfig):
    """A decision run that hands back the objective, the data and the summary it would train with."""

    def train(self, objective, dataset, **settings):
        return objective, dataset, settings["summary"]


def _run(backbone: Path, count: int, *, stated: bool = False, **budgets) -> Fitted:
    """A fresh lettered run of its own label count over a question of `count` intents, an example of
    each, its options `stated` by the data or read from its labels, at the run's own budgets but
    `budgets`."""
    options = list(intents(count).options)
    rows = [{"text": f"my message is about {option}", "label": option} for option in options]
    data = (Memory(rows).pipe(decisions, question="intent", options=options if stated else ())
            .read("memory").batch(8).loading(Loading(workers=0)))
    default = Fitted(data=data)
    trainer = dataclasses.replace(default.trainer, multi_host=False, compilation_cache_dir=None)
    # A window that holds a question of 151 intents whole under either tokenizer.
    return dataclasses.replace(default, pretrained=str(backbone), trainer=trainer, head=Letters(),
                               max_len=6144, **budgets)


@pytest.mark.parametrize("count", [77, 151])
def test_a_fresh_runs_rows_hold_their_whole_questions_and_count_each_declared_cut(backbone, count):
    """A fresh lettered run (a real tokenizer) over 77 or 151 intents lays out its questions whole at
    the run's own budgets: its layout holds every option's line as its text, and the run's data counters
    count no cut of a question and refuse no example. The same run declaring each option's line its
    shortest line's length less three tokens cuts the end every intent's name shares
    ("_of_the_customer") from each, keeping the index that tells it apart: it counts that cut of every
    example's options, in examples, rows and tokens, and refuses none. Declaring each line three tokens,
    which keep its label alone, leaves every option alike: training leaves each example out under that
    reason, counted."""
    objective, dataset, summary = _run(backbone, count).prepare().train("x")
    [counts] = dataset.counts.values()
    assert counts.accepted == count and counts.refused == {}
    assert not counts.cut_rows.keys() & {"instructions", "options"}
    question = intents(count)
    [row] = objective.layout.rows(objective.tokenizer, objective.specials, "my card", {"intent": question})
    names = objective.layout.labels.names_for(count)
    # Its readout holds every label its tokenizer letters, where the data states no option count.
    letters = Labels().capacity(objective.tokenizer)
    assert summary["capacities"]["labels"] == {"count": letters, "from": "tokenizer"} and letters >= count
    lines = [objective.tokenizer.decode(row.tokens[start:end]) for start, end in row.questions[0].options]
    assert lines == [f"\n{label}. {option}" for label, option in zip(names, question.options, strict=True)]
    shortest = min(end - start for start, end in row.questions[0].options)
    _, declared, _ = _run(backbone, count, option_tokens=shortest - 3).prepare().train("x")
    [cut] = declared.counts.values()
    assert cut.accepted == count and cut.refused == {}
    assert cut.cut_examples["options"] == cut.cut_rows["options"] == count
    assert cut.cut_removed["options"] > 0
    with pytest.raises(ValueError):
        # Every example's options read alike, which leaves the run no rows to train on.
        _run(backbone, count, option_tokens=3).prepare().train("x")


@pytest.mark.parametrize("count", [77, 151])
def test_a_runs_readout_holds_the_labels_its_datas_questions_ask(backbone, count):
    """A fresh lettered run whose data states its question's options holds as many labels as they are,
    one more where its objective adds "none of the above", and reports the count and its origin."""
    objective, _, summary = _run(backbone, count, stated=True).prepare().train("x")
    assert summary["capacities"]["labels"] == {"count": count, "from": "data"}
    assert len(objective.model.head.label_ids) == count
    noted = dataclasses.replace(_run(backbone, count, stated=True), objective=dataclasses.replace(
        DecisionRunConfig().objective, fields={"none_of_the_above": 0.25}))
    _, _, augmented = noted.prepare().train("x")
    assert augmented["capacities"]["labels"] == {"count": count + 1, "from": "data"}


def test_a_runs_readout_takes_every_label_where_a_split_leaves_its_options_unknown(backbone):
    """A run whose training options are read only as its rows are, beside a validation split that
    states two, holds every label its tokenizer letters: a split it has not measured never takes the
    fewer labels another states."""
    stated = Memory([{"text": "my card", "label": "a"}, {"text": "my loan", "label": "b"}]).pipe(
        decisions, question="intent", options=("a", "b")).read("memory").batch(8).loading(Loading(workers=0))
    run = dataclasses.replace(_run(backbone, 77), validation=stated)
    objective, _, summary = run.prepare().train("x")
    letters = Labels().capacity(objective.tokenizer)
    assert summary["capacities"]["labels"] == {"count": letters, "from": "tokenizer"} and letters >= 77


@pytest.mark.parametrize("count", [77, 151])
def test_a_server_admits_a_large_question_whole_and_counts_what_it_refuses_as_training_does(backbone, count):
    """The run's task, served whole (`Decide.scoring(whole=True)`), admits a request of 77 or 151 intents
    as it lays it out whole, its usage reporting no cut, and counts it admitted; the same task under
    Laya's declared budgets, served whole, refuses it under `cut`, counted in the server's own counters
    (`WireScoring.counts`), and answers a short request whose instructions those budgets cut with the cut
    in its usage where it is not served whole."""
    objective, _, _ = _run(backbone, count).prepare().train("x")
    task = Decide(objective.model, objective.init(jax.random.key(0)), objective.layout, objective.tokenizer,
                  objective.specials)
    body = {"state": "my new card has not arrived", "questions": {"intent": intents(count).wire()}}
    served, _ = task.scoring(details=True, whole=True)
    admitted = served.admit(body)
    assert [row.encoded.trimmed for row in admitted.rows] == [False] and served.counts.accepted == 1
    declared = dataclasses.replace(task, layout=dataclasses.replace(task.layout, **LAYA))
    strict, _ = declared.scoring(details=True, whole=True)
    with pytest.raises(Refused) as refused:
        strict.admit(body)
    assert refused.value.reason == "cut" and strict.counts.rejected == {"cut": 1}
    worded = Noul("Does the customer " + "really and truly " * 120 + "need a new card?")
    short = {"state": "my card is lost", "questions": {"holds": worded.wire()}}
    with pytest.raises(Refused) as whole:
        strict.admit(short)
    assert whole.value.reason == "cut" and strict.counts.rejected["cut"] == 2
    lenient, _ = declared.scoring(details=True)
    [row] = [row.encoded for row in lenient.admit(short).rows]
    assert set(row.cuts) == {"instructions"} and lenient.counts.cut_rows == {"instructions": 1}
    assert lenient.counts.cut_removed == {"instructions": row.instructions_cut}
