"""A row whose tokens stand in streams of its own (`Reads`, row 2319): two streams in one pass, the state
reading the other stream (a joint row's schema, a lettered row's question) and that stream itself alone,
wherever the row lays them. The layout marks each token's stream, a window keeps each row's, and the
model lowers the record to key runs over each row's own runs of one stream (`stream_runs`), so no token
reads another row's, and each reads exactly what its stream's record says."""
import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import TINY_DECODER
from test_decision_context import assert_packed_as_alone, windows_of
from test_decision_letters_context import LABELS, QUESTIONS as LETTERED_QUESTIONS, STATES

from dew.data.text import ByteTokenizer
from dew.decision import Choice, Example, JointLayout, Schema, State, Text
from dew.decision.head import JointSchemaHead, LabelReadout
from dew.decision.layout import (
    AnswerCue,
    DecisionInputs,
    Instructions,
    Layout,
    OptionLines,
    Reads,
    Specials,
    StateFirstLayout,
)
from dew.decision.model import DecisionModel, stream_runs
from dew.decision.objective import DecisionObjective
from dew.decision.task import laid_out
from dew.interop import Pretrained
from dew.nn.attention_masks import visibility_mask
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.dex import DifferentialExtension
from dew.nn.progress import END, Progress
from dew.registry import from_record, to_record

pytestmark = pytest.mark.mesh

LLAMA = Path(__file__).parent / "fixtures" / "hf" / "llama-tiny"
SPECIALS = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
TOKENIZER = ByteTokenizer()
JOINT = JointLayout(max_len=1024, segments=(Text("STATE:\n"), State(), Schema(stream="schema"), Text("\n")),
                    reads=Reads((("row", "self", "causal"), ("row", "schema", "whole"),
                                 ("schema", "self", "causal"))))
"""A joint row's two streams: its own tokens read the row's before them and the whole schema, though the
schema follows the state, the schema's tokens the schema's before them alone."""
LETTERED = StateFirstLayout(
    labels=LABELS, max_len=256,
    segments=(Text("STATE:\n"), State(), Instructions({"choice": "{}"}, stream="question"),
              OptionLines(line="\n{label}. {option}", bare="\n{label}. {option}", join="",
                          noul_first="false"),
              AnswerCue(template="\nAnswer:", many="\nAnswer:")),
    reads=Reads((("row", "self", "causal"), ("row", "question", "whole"), ("question", "self", "causal"))))
"""A lettered row's two streams (row 2008 in one pass): the state and the cue read the whole question,
the question and its lettered options themselves alone."""
JOINT_QUESTIONS = {"pick": Choice("Which one?", ["a", "b", "c"]), "close": Choice("Closest?", ["x", "y"])}
ROWS = ["one two three", "four five", "six seven eight nine", "ten eleven", "twelve", "thirteen fourteen"]


def in_streams(layout: Layout, segments: int = 1) -> DecisionObjective:
    """A decision over the tiny Llama whose rows stand in `layout`'s streams, at most `segments` rows a
    window: a joint row's head, or a lettered row's readout of its letters."""
    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference", max_seq_len=1024)
    width, dtype = DecisionModel.head_size(source.model)
    head = (JointSchemaHead(width, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)
            if layout is JOINT else LabelReadout(features=width, label_ids=(), dtype=dtype, count=4))
    return DecisionObjective(source, tokenizer=TOKENIZER, specials=SPECIALS, layout=layout, head=head,
                             segments=segments)


def examples(layout: Layout) -> list[Example]:
    """Rows of `layout`: joint rows asked both questions, lettered rows one of two in turn."""
    if layout is JOINT:
        return [Example(state, JOINT_QUESTIONS, {"pick": "abc"[index % 3], "close": "xy"[index % 2]})
                for index, state in enumerate(ROWS)] * 2
    return [Example(state, {"q": LETTERED_QUESTIONS[index % 2]},
                    {"q": LETTERED_QUESTIONS[index % 2].options[0]}) for index, state in enumerate(STATES)]


def read_as_recorded(streams: np.ndarray, segments: np.ndarray, reads) -> np.ndarray:
    """The `[L, L]` keys each of a window's tokens reads by its row's record, from the record itself: a
    key of its own row whose stream its stream reads, whole or up to the query; the padding its own."""
    length = len(streams)
    seen = np.zeros((length, length), bool)
    for query in range(length):
        for key in range(length):
            if segments[query] != segments[key]:
                continue
            if segments[query] == 0:
                seen[query, key] = True
                continue
            reader, read = streams[query], streams[key]
            how = reads[reader][0] if reader == read else reads[reader][1 + read]
            seen[query, key] = how == 1 or (how == 2 and key <= query)
    return seen


@pytest.mark.parametrize("layout", [JOINT, LETTERED], ids=["joint", "lettered"])
def test_a_rows_tokens_stand_in_the_streams_its_segments_name_and_a_window_keeps_each_rows(layout):
    """Each token stands in its segment's stream, the other stream's tokens (the schema, the question)
    in its own and the rest in the row's, the row's own again after it; a window of rows keeps each
    row's streams and its padding in none."""
    example = examples(layout)[0]
    [row] = layout.rows(TOKENIZER, SPECIALS, example.state, example.questions)
    assert row.streams is not None and len(row.streams) == len(row.tokens)
    streams, tokens = np.asarray(row.streams), np.asarray(row.tokens)
    other, own = (TOKENIZER.decode(tokens[streams == stream].tolist()) for stream in (1, 0))
    assert example.state in own and example.state not in other
    assert "Which" in other and "Which" not in own
    assert streams[-1] == 0 and set(streams.tolist()) == {0, 1}
    window = windows_of(in_streams(layout, segments=3), examples(layout), count=1)[0]
    segments, packed = np.asarray(window["segments"]), np.asarray(window["streams"])
    assert int(segments.max()) > 1 and np.all(packed[segments == 0] == -1)


@pytest.mark.parametrize("layout", [JOINT, LETTERED], ids=["joint", "lettered"])
def test_each_token_reads_exactly_what_its_streams_record_says_in_a_packed_window(layout):
    """The key runs the model gives its backbone (`stream_runs`) are, token by token, the keys the record
    says: a row's own tokens its row's before them and the whole other stream, though it follows them,
    the other stream's its own before them, no token another row's, and the padding its own."""
    objective = in_streams(layout, segments=3)
    reads, fragments = layout.attention
    for window in windows_of(objective, examples(layout), count=3):
        inputs = laid_out(window, objective.bounds.segments, attention=layout.attention)
        assert inputs.streams is not None
        runs = stream_runs(inputs.streams, inputs.segments, reads, fragments)
        length = inputs.tokens.shape[1]
        seen = np.asarray(visibility_mask(length, length, causal=False, sliding_window=None,
                                          key_runs=runs))[:, 0]
        for row, streams, segments in zip(seen, np.asarray(inputs.streams), np.asarray(inputs.segments),
                                          strict=True):
            np.testing.assert_array_equal(row, read_as_recorded(streams, segments, reads))


def states_of(objective: DecisionObjective, variables, state: str,
              questions) -> tuple[np.ndarray, np.ndarray]:
    """The backbone's final states over a row of `state` and `questions` alone, and its tokens' streams."""
    [row] = objective.layout.rows(TOKENIZER, SPECIALS, state, questions)
    inputs = DecisionInputs.collate([row], SPECIALS.pad, attention=objective.layout.attention)
    states = np.asarray(objective.model.states(variables, inputs))[0]
    assert row.streams is not None and len(row.streams) == len(row.tokens) == len(states)
    return states, np.asarray(row.streams)


@pytest.mark.parametrize("layout", [JOINT, LETTERED], ids=["joint", "lettered"])
def test_the_state_reads_the_other_stream_after_it_and_that_stream_reads_no_state(layout):
    """Over drawn parameters, the other stream's states (the schema's, the question's) are the same
    whatever state the row holds beside it, as that stream reads none (bitwise: a key a token does not
    read adds nothing), while the state's states move with that stream, which they read though it
    follows them."""
    objective = in_streams(layout)
    variables = jax.jit(objective.init)(jax.random.key(0))
    asked = examples(layout)[0].questions
    first, streams = states_of(objective, variables, "one two three", asked)
    other, _ = states_of(objective, variables, "six five four", asked)
    read = streams == 1
    np.testing.assert_array_equal(first[read], other[read])
    assert not np.array_equal(first[~read], other[~read])
    moved_questions = (
        {"pick": Choice("Which one?", ["d", "e", "f"]), "close": Choice("Closest?", ["v", "w"])}
        if layout is JOINT else {"q": Choice("Which team handles it?", ["bards", "boans", "bra"])})
    moved, moved_streams = states_of(objective, variables, "one two three", moved_questions)
    # The row's own tokens before the other stream, the same in both rows.
    before = int(np.argmax(read))
    assert np.array_equal(moved_streams[:before], streams[:before])
    assert not np.array_equal(first[:before], moved[:before])


STREAM_SEEDS = {"joint": 16, "lettered": 8}
"""Check 1's declared seeds for packed rows in two streams against each row alone, sized by `seeds_for`
with the packed windows as `dew_fn` (16 and 8, 2026-10-30)."""


@pytest.mark.parametrize("layout", [JOINT, LETTERED], ids=["joint", "lettered"])
def test_packed_rows_in_two_streams_decide_each_row_as_the_row_alone(layout):
    """Rows in two streams, packed a window, decide each row's options as the row does in a window of its
    own, through the unmodified objective: each row's tokens read its own streams and no other row's, the
    real options' logits on par from float64 (check 1, over drawn parameters)."""
    name = "joint" if layout is JOINT else "lettered"
    assert_packed_as_alone(in_streams(layout, segments=3), in_streams(layout, segments=1), examples(layout),
                           f"packed {name} rows in two streams", STREAM_SEEDS[name])


def test_a_layouts_reads_name_only_its_streams_and_every_stream_reads_itself():
    """A record that names a stream no segment lays, a stream no record reads, a record in which a stream
    reads nothing of itself and one that reads another way than `How` are refused; a layout and its record
    round-trip through its record (`to_record`), and the model reads it lowered over the row's streams."""
    with pytest.raises(ValueError):
        dataclasses.replace(JOINT, reads=Reads((("row", "self", "causal"), ("notes", "self", "whole"))))
    with pytest.raises(ValueError):
        dataclasses.replace(JOINT, reads=None)
    with pytest.raises(ValueError):
        Reads((("row", "schema", "whole"),))
    with pytest.raises(ValueError):
        Reads((("row", "self", "sideways"),))
    assert from_record(JointLayout, to_record(JOINT, JointLayout)) == JOINT
    assert from_record(StateFirstLayout, to_record(LETTERED, StateFirstLayout)) == LETTERED
    assert JOINT.attention == (((2, 0, 1), (2, 0, 0)), 3)


def test_a_row_in_streams_reads_its_backbone_at_the_runs_progress():
    """A row in streams reads its backbone at the run's progress, as every row does (`progressed`): over
    a backbone whose layers anneal with it (DEX on every layer), the same rows' option logits mid-anneal
    differ from the run's end's, which a streams row read at the end alone would leave the same."""
    annealed = DifferentialExtension(((0, (0, 1)), (1, (1, 3))), anneal_steps=10)
    backbone = CausalTransformer(**{**TINY_DECODER, "vocab_size": 256, "max_seq_len": 256, "num_layers": 2,
                                    "dtype": jnp.float32, "attention_impl": "reference", "dex": annealed})
    width, dtype = DecisionModel.head_size(backbone)
    objective = DecisionObjective(backbone, tokenizer=TOKENIZER, specials=SPECIALS, layout=LETTERED,
                                  head=LabelReadout(features=width, label_ids=(), dtype=dtype, count=4))
    variables = objective.init(jax.random.key(0))
    example = examples(LETTERED)[0]
    rows = LETTERED.rows(TOKENIZER, SPECIALS, example.state, example.questions)
    inputs = DecisionInputs.collate(rows, SPECIALS.pad, attention=LETTERED.attention)
    assert inputs.streams is not None
    middle = objective.model.logits(variables, inputs, progress=Progress(jnp.asarray(5, jnp.int32)))
    end = objective.model.logits(variables, inputs, progress=END)
    assert not np.array_equal(np.asarray(middle), np.asarray(end))
