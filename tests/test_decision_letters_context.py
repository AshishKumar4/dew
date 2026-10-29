"""A lettered question row whose question is its context (row 2008): its instructions and option lines are
written to the row's context (`Instructions.stream`), which its cross-attention reads, and the letter
readout answers at the row's own cue, several such rows packed a window as one row alone reads them."""

import dataclasses
import json
from pathlib import Path

import jax
import numpy as np
import pytest
from test_decision_context import assert_packed_as_alone, opened, windows_of

from dew.batch import thaw
from dew.data.pipeline import Rejected
from dew.data.text import ByteTokenizer
from dew.decision import Choice, Decide, Example
from dew.decision.config import DecisionRunConfig, Letters
from dew.decision.head import LabelReadout, SlotReadout
from dew.decision.layout import (
    AnswerCue,
    Instructions,
    Labels,
    OptionLines,
    Segment,
    Specials,
    State,
    StateFirstLayout,
    Text,
)
from dew.decision.model import DecisionModel
from dew.decision.objective import DecisionObjective
from dew.decision.rows import Encoding, Fit
from dew.interop import Pretrained
from dew.nn.cross_attention import CrossAttention
from dew.registry import to_record

pytestmark = pytest.mark.mesh

LLAMA = Path(__file__).parent / "fixtures" / "hf" / "llama-tiny"
SPECIALS = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
TOKENIZER = ByteTokenizer()
LABELS = Labels(names=("A", "B", "C", "D"), variants=("{}",), cue="\nAnswer:")
SEGMENTS = (Text("STATE:\n"), State(), Instructions({"choice": "{}"}, stream="context"),
            OptionLines(line="\n{label}. {option}", bare="\n{label}. {option}", join="", noul_first="false",
                        stream="context"),
            AnswerCue(template="\nAnswer:", many="\nAnswer:"))
"""A lettered row with its question read through cross-attention: the state and the cue in the
row, the question and its lettered options in the context."""
LAYOUT = StateFirstLayout(labels=LABELS, segments=SEGMENTS, max_len=96)
QUESTIONS = (Choice("Which team handles it?", ["cards", "loans", "fraud"]),
             Choice("Which way is it?", ["up", "down", "left", "right"]))
STATES = ["my card is lost", "a loan request", "a strange charge", "it went up", "turn left", "down below",
          "the card again", "fraud again"]


def examples() -> list[Example]:
    """Each state asked one of `QUESTIONS` in turn, answered with its first option."""
    asked = [QUESTIONS[index % 2] for index in range(len(STATES))]
    return [Example(state, {"q": question}, {"q": question.options[0]})
            for state, question in zip(STATES, asked, strict=True)]


def lettered(segments: int, context: int = 256, *, slots: bool = False) -> DecisionObjective:
    """A lettered decision over the tiny Llama with fresh cross-attention before its layers, reading each
    row's question as its context of up to `context` tokens, at most `segments` rows a window: its letters
    read by the backbone's own rows (`LabelReadout`), or by a slot release's head (`SlotReadout`)."""
    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference", max_seq_len=512)
    crossed = source.with_cross_attention(CrossAttention((0, 1), context), key=0)
    width, dtype = DecisionModel.head_size(crossed.model)
    head = (SlotReadout(features=width, label_ids=tuple(ids[0] for ids in LABELS.ids(TOKENIZER, 4)))
            if slots else LabelReadout(features=width, label_ids=(), dtype=dtype, count=4))
    return DecisionObjective(crossed, tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT, head=head,
                             segments=segments)


def letter_probabilities(objective: DecisionObjective, variables, state: str, asked) -> list[np.ndarray]:
    """Each question of `asked`'s letter probabilities over `state`, a request of its own each."""
    task = Decide(objective.model, variables, objective.layout, objective.tokenizer, objective.specials)
    answers = [task(state, {"q": question})["q"] for question in asked]
    return [np.asarray(answer.probabilities) for answer in answers]


def text(tokens) -> str:
    return TOKENIZER.decode(list(tokens))


def test_a_lettered_row_writes_its_question_to_its_context_and_answers_at_its_own_cue():
    """The row holds the state and the cue alone, its last token the readout; its context holds the
    question's instructions and lettered options, which its spans index; the question's text and labels
    are those the row would write in its own tokens."""
    question = QUESTIONS[0]
    [row] = LAYOUT.rows(TOKENIZER, SPECIALS, "my card is lost", {"q": question})
    assert row.spans_in_context and text(row.tokens) == "STATE:\nmy card is lost\nAnswer:"
    [laid] = row.questions
    assert laid.readout == len(row.tokens) - 1
    context = text(row.context)
    assert context == "Which team handles it?\nA. cards\nB. loans\nC. fraud"
    assert text(row.context[laid.span[0]:laid.span[1]]) == "Which team handles it?"
    assert [text(row.context[start:end]) for start, end in laid.options] == ["\nA. cards", "\nB. loans",
                                                                            "\nC. fraud"]
    in_row = dataclasses.replace(LAYOUT, segments=tuple(
        dataclasses.replace(segment, stream="row") if isinstance(segment, (Instructions, OptionLines))
        else segment for segment in SEGMENTS))
    [own] = in_row.rows(TOKENIZER, SPECIALS, "my card is lost", {"q": question})
    assert text(own.tokens) == "STATE:\nmy card is lost" + context + "\nAnswer:"
    assert own.questions[0].label_ids == laid.label_ids


def test_a_letter_answer_reads_its_question_through_the_context_alone():
    """Two questions of as many options over one state lay out the same row, the state and the cue, and
    other contexts: with the cross-attention open their letters' probabilities differ, the answer reading
    the question it is asked; shut (every gate 0, as drawn) they are the same, the row reading nothing else
    of its question."""
    objective = lettered(segments=1)
    asked = (Choice("Which team handles it?", ["cards", "loans", "fraud"]),
             Choice("Which desk takes it?", ["cards", "loans", "fraud"]))
    rows = [objective.layout.rows(TOKENIZER, SPECIALS, "my card is lost", {"q": question})[0]
            for question in asked]
    assert rows[0].tokens == rows[1].tokens and rows[0].context != rows[1].context
    variables = objective.init(jax.random.key(0))
    shut = letter_probabilities(objective, opened(variables, 0.0), "my card is lost", asked)
    open_ = letter_probabilities(objective, opened(variables), "my card is lost", asked)
    assert np.array_equal(shut[0], shut[1])
    assert not np.array_equal(open_[0], open_[1])


def test_a_slot_readout_pools_each_option_line_from_the_context_that_holds_it():
    """A slot release's key pools each option's line where it stands (`SlotReadout`): in the context, for a
    row whose question is its context. With the cross-attention shut, two questions whose option lines
    alone differ lay out the same row, so only the lines' pooling tells them apart: their scores differ
    once the key reads the lines (drawn off its zero), where a pooling of the row's own states at the
    context's spans scored them alike."""
    objective = lettered(segments=1, slots=True)
    asked = (Choice("Which team handles it?", ["cards", "loans", "fraud"]),
             Choice("Which team handles it?", ["a bank", "a shop", "a home"]))
    variables = thaw(objective.init(jax.random.key(0)))

    def keyed(path, leaf):
        named = jax.tree_util.keystr(path)
        return (0.5 * jax.random.normal(jax.random.key(1), np.shape(leaf), leaf.dtype)
                if "head" in named and named.endswith("['k']") else leaf)

    variables = {**variables, "params": jax.tree_util.tree_map_with_path(keyed, variables["params"])}
    first, second = letter_probabilities(objective, variables, "my card is lost", asked)
    assert not np.array_equal(first, second)


def test_a_question_past_the_context_is_unfit_and_a_none_of_the_above_past_it_declined():
    """A question that fills the context its backbone reads exactly trains, and a "none of the above" a
    read draws for it, which lengthens the context and not the row, is declined and laid out as Fit
    admitted it, counted under `unfit`; a question past the context is counted unfit, not left to stop
    the read."""
    objective = lettered(segments=1)
    example = Example("my card is lost", {"q": QUESTIONS[0]}, {"q": "cards"})
    [row] = objective.layout.rows(TOKENIZER, SPECIALS, example.state, example.questions)
    assert row.context is not None
    length = len(row.context)
    fit = Fit(objective.layout, TOKENIZER, SPECIALS, objective.model.head, context_length=length,
              none_of_the_above=1.0)
    fitted = fit.map({"example": example})
    assert fitted["_dew_declined"] == {"unfit": 1}
    [admitted] = fitted["decision_rows"]
    encoding = Encoding(objective.layout, TOKENIZER, SPECIALS, width=4, none_of_the_above=1.0,
                        context_length=length, shuffle=False)
    rng = np.random.default_rng(0)
    drawn = [encoding(example, ["q"], rng, admitted.tokens, admitted=admitted.cuts) for _ in range(8)]
    assert all(list(laid["context"]) == list(row.context) for laid in drawn)
    short = Fit(objective.layout, TOKENIZER, SPECIALS, objective.model.head, context_length=length - 1)
    assert isinstance(short.map({"example": example}), Rejected)


LETTERED_SEEDS = 8
"""Check 1's declared seeds for packed lettered rows reading their questions in context against each row
alone, as the packed schemas' and contexts' (`test_decision_context.PACKED_CONTEXT_SEEDS`), sized by
`seeds_for` (8, 2026-10-28)."""


def test_lettered_rows_whose_question_is_their_context_pack_several_a_window_each_answering_its_own():
    """Lettered rows whose questions are their contexts pack several a window, each answering at its own
    cue (`readout`, moved by where its row's tokens start, not its context's) over its own question,
    as each row decides alone, through the unmodified objective, its cross-attention open (every gate 1),
    the real letters' logits on par from float64 (check 1, over drawn parameters)."""
    packed = lettered(segments=3)
    windows = windows_of(packed, examples(), count=4)
    readout, segments = windows[0]["readout"][0], windows[0]["segments"][0]
    owners = windows[0]["question_rows"][0]
    for question in range(int(windows[0]["questions"][0].sum())):
        # Each answer cue is its row's last token in the window.
        assert segments[readout[question]] == owners[question]
        assert readout[question] + 1 == len(segments) or segments[readout[question] + 1] != owners[question]
    assert_packed_as_alone(packed, lettered(segments=1), examples(), "lettered questions in context",
                           LETTERED_SEEDS)



def test_a_run_names_its_lettered_question_read_in_context_on_the_command_line():
    """A lettered run on the command line: its segments are one flag, its question read as the row's context
    among them, with the context layers that read it; one without them is refused, and so is a question
    whose instructions and option lines stream apart, or whose options are read at markers."""
    config = DecisionRunConfig.cli(["--pretrained", str(LLAMA), "--segments",
                                    json.dumps(to_record(SEGMENTS, tuple[Segment, ...])),
                                    "--context-layers", "0", "1", "head:letters", "--head.count", "4"])
    assert config.segments == SEGMENTS and isinstance(config.head, Letters)
    with pytest.raises(ValueError):
        dataclasses.replace(config, context_layers=None).prepare()
    apart = (*SEGMENTS[:3], dataclasses.replace(SEGMENTS[3], stream="row"), SEGMENTS[4])
    with pytest.raises(ValueError):
        StateFirstLayout(labels=LABELS, segments=apart)
    marked = (*SEGMENTS[:3], dataclasses.replace(SEGMENTS[3], marker="after"), SEGMENTS[4])
    with pytest.raises(ValueError):
        StateFirstLayout(segments=marked)
