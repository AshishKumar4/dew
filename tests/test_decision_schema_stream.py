"""A joint row whose schema is read as its context (`Schema(stream="context")`): the schema leaves the
row for its context, the backbone reads it in a pass of its own with its own weights, its
cross-attention layers read that pass beside the row, and the head pools the questions and options
from it (`dew.nn.cross_attention.encode_context`), the row's memory and readout its own. The pass is counted
in the run's work, and the task serves the same way."""

import dataclasses
import itertools
from pathlib import Path

import jax
import numpy as np
import optax
import pytest
from decision_support import decision_rows
from recording import RecordingTracker

from dew.data.dataset import DataPartition
from dew.data.text import ByteTokenizer
from dew.decision import Choice, Decide, DecisionHead, Example, JointLayout, Noul, Schema, State, Text
from dew.decision.head import JointSchemaHead
from dew.decision.layout import Specials
from dew.decision.model import DecisionModel
from dew.decision.objective import DecisionObjective
from dew.decision.questions import ChoiceAnswer
from dew.interop import Pretrained
from dew.nn.cross_attention import CrossAttention
from dew.training import Trainer

pytestmark = pytest.mark.mesh

LLAMA = Path(__file__).parent / "fixtures" / "hf" / "llama-tiny"
SPECIALS = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
TOKENIZER = ByteTokenizer()
LAYOUT = JointLayout(max_len=256, segments=(Text("STATE:\n"), State(), Schema(stream="context"), Text("\n")))
QUESTION = {"pick": Choice("Which one?", ["a", "b", "c"])}
QUESTIONS = {"pick": Choice("Which one?", ["a", "b", "c"]), "holds": Noul("Does it hold?"),
             "close": Choice("Which letter is closest to it?", ["x", "y"])}
"""Three questions, whose schema runs far past the short row of its state."""
ROWS = [("one two three", "a"), ("four five", "b"), ("six seven eight nine", "c"), ("ten eleven", "a")] * 4


def read_in_context(length: int = 768, head=None, segments: int | None = 16, max_len: int = 256,
                    context_window: int | None = None) -> DecisionObjective:
    """A decision over the tiny Llama with fresh cross-attention before its layers, its rows' schemas
    read as their contexts, up to `length` tokens, at most `segments` rows of `max_len` a window and
    `context_window` tokens of their contexts."""
    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference", max_seq_len=1024)
    crossed = source.with_cross_attention(CrossAttention((0, 1), length), key=0)
    width, _ = DecisionModel.head_size(crossed.model)
    head = head or JointSchemaHead(width, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)
    layout = dataclasses.replace(LAYOUT, max_len=max_len)
    return DecisionObjective(crossed, tokenizer=TOKENIZER, specials=SPECIALS, layout=layout, head=head,
                             segments=segments, context_window=context_window)


LABELS = {"a": {"holds": "true", "close": "x"}, "b": {"holds": "false", "close": "y"},
          "c": {"holds": "true", "close": "y"}}
SCHEMAS = (QUESTIONS, {"pick": QUESTIONS["pick"]},
           {"close": Choice("Which letter is closest to it?", ["x", "y", "z", "w"]),
            "holds": QUESTIONS["holds"]})
"""Schemas of other questions, options and lengths, which rows in turn ask."""


def labelled() -> list[Example]:
    """`ROWS`, each asked the questions of a schema of `SCHEMAS` in turn, each answered."""
    return [Example(state, schema, {name: answer for name, answer in {"pick": label, **LABELS[label]}.items()
                                    if name in schema})
            for (state, label), schema in zip(ROWS, itertools.cycle(SCHEMAS))]


def text(tokens) -> str:
    return TOKENIZER.decode(list(tokens))


def test_a_schema_read_in_context_leaves_the_row_and_its_spans_index_the_context():
    """The row holds the state and not the schema; the context holds the schema, and each question's
    and option's span is where its text is in the context."""
    [row] = LAYOUT.rows(TOKENIZER, SPECIALS, "one two three", QUESTION)
    assert row.spans_in_context and row.context is not None
    assert "SCHEMA FIELDS" not in text(row.tokens) and "one two three" in text(row.tokens)
    [laid] = row.questions
    assert text(row.context[laid.span[0]:laid.span[1]]) == "Which one?"
    assert sorted(text(row.context[start:end]) for start, end in laid.options) == [
        '{"option_id":"a"}', '{"option_id":"b"}', '{"option_id":"c"}']


def test_a_run_reads_each_rows_schema_in_context_and_its_task_serves_alike():
    """Rows whose schemas are their contexts train, each logged step reporting the work of the
    context's own pass beside the row's, beyond the row's alone
    (`test_the_contexts_own_pass_is_counted_once_where_it_trains_nothing` holds the count itself); the
    run's task answers a request, and a question asked in other words answers otherwise, as the head
    pools it from the context's pass."""
    objective = read_in_context()
    data = decision_rows(labelled(), 8, objective=objective.data())
    [batch] = list(itertools.islice(data.train(DataPartition()), 1))
    assert "context_lengths" in batch and int(np.max(batch["context_lengths"])) > 0
    tracker = RecordingTracker()
    state = Trainer(objective, optax.adam(1e-3), key=0, tracker=tracker).fit(data, steps=1, log_every=1)
    [logged] = [scalars["train/flops"] for _, scalars in tracker.scalars if "train/loss" in scalars]
    without = {name: value for name, value in batch.items() if name != "context_lengths"}
    assert logged > objective.training_flops(None, without)

    task = objective.pipeline(state)
    assert isinstance(task, Decide)
    # Each question's span is past the short row's end, which a question's own row does not reach.
    [row] = LAYOUT.rows(TOKENIZER, SPECIALS, "one", QUESTIONS)
    assert all(laid.span[0] >= len(row.tokens) for laid in row.questions[1:])
    answered = task("one", QUESTIONS)
    assert set(answered) == set(QUESTIONS)
    asked = [task("one two three", {"pick": Choice(words, ["a", "b", "c"])})["pick"]
             for words in ("Which one?", "Pick the closest letter.")]
    assert all(isinstance(answer, ChoiceAnswer) for answer in asked)
    assert not np.array_equal(asked[0].probabilities, asked[1].probabilities)


def test_the_contexts_own_pass_is_counted_once_where_it_trains_nothing():
    """The training work of a batch whose schemas are its contexts is the frozen backbone's over its
    rows, one forward of the backbone over each context, which trains nothing and takes no gradient,
    and three passes of the cross-attention blocks and the head."""
    from dew.nn.flops import counted_flops, frozen_training_flops, parameter_flops

    objective = read_in_context(64)
    lengths, contexts = np.asarray([5, 7]), np.asarray([30, 41], np.int32)
    head_model = objective.model.head
    hidden = head_model.hidden_size
    batch = {"segments": (np.arange(8) < lengths[:, None]).astype(np.int32),
             "spans": np.zeros((2, 1, 2), np.int32), "questions": np.ones((2, 1), bool),
             "options": np.ones((2, 1, 3), bool), "context_lengths": contexts}
    backbone = objective.model.backbone
    cross = backbone.cross_flops(lengths, contexts)
    one, three = np.ones(2, np.int64), np.full(2, 3, np.int64)
    head = head_model.decision_flops(lengths, one, three, pooled=contexts)
    # The spans pool over the contexts' tokens, not the rows': each question and twice each option.
    pooling = 2 * hidden * np.sum((contexts - lengths) * (1 + 2 * 3))
    assert head - head_model.decision_flops(lengths, one, three) == pooling
    frozen = frozen_training_flops(counted_flops(backbone, lengths), parameter_flops(backbone, lengths))
    context = counted_flops(backbone, contexts)
    assert objective.training_flops(None, batch) == frozen + context + 3 * (cross + head)


def test_a_schema_read_in_context_refuses_what_cannot_read_it():
    """A request's own context beside a schema read as its context, Laya's marker head, and a backbone
    without cross-attention layers are each refused, rather than one context read in place of the
    other or the schema left unread."""
    objective = read_in_context()
    task = Decide(objective.model, objective.init(jax.random.key(0)), LAYOUT, TOKENIZER, SPECIALS)
    with pytest.raises(ValueError):
        task("one two three", QUESTION, context="a note")
    noted = [Example("one", QUESTION, {"pick": "a"}, context="a note")] * 4
    with pytest.raises(ValueError):
        list(itertools.islice(decision_rows(noted, 2, objective=objective.data()).train(DataPartition()), 1))
    width, _ = DecisionModel.head_size(objective.model.backbone)
    marked = read_in_context(head=DecisionHead(width, layers=1))
    with pytest.raises(TypeError):
        Decide(marked.model, marked.init(jax.random.key(0)), LAYOUT, TOKENIZER, SPECIALS)("one", QUESTION)
    plain = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference", max_seq_len=1024)
    bare = DecisionObjective(plain, tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT,
                             head=JointSchemaHead(width, width=16, routing_layers=1, layers=1, heads=2,
                                                  feedforward=32))
    with pytest.raises(ValueError):
        Decide(bare.model, bare.init(jax.random.key(0)), LAYOUT, TOKENIZER, SPECIALS)("one", QUESTION)


def test_a_run_names_its_schema_read_in_context_on_the_command_line():
    """A decision run's segments are one flag, the schema read as the row's context among them, with
    the context layers that read it; a schema in context without them is refused."""
    import json

    from dew.decision.config import DecisionRunConfig
    from dew.decision.layout import Segment
    from dew.registry import to_record

    segments = (Text("STATE:\n"), State(), Schema(stream="context"), Text("\n"))
    config = DecisionRunConfig.cli(["--pretrained", str(LLAMA), "--segments",
                                    json.dumps(to_record(segments, tuple[Segment, ...])),
                                    "--context-layers", "0", "1"])
    assert config.segments == segments
    with pytest.raises(ValueError):
        dataclasses.replace(config, context_layers=None).prepare()


def test_rows_whose_schema_is_their_context_pack_several_a_window_each_reading_its_own():
    """Rows whose schemas are their contexts pack several a window, as rows of an example's context do
    (row 2008): each question's row is the one the packer placed it in (`question_rows`), its span the
    place of its own text in its own row's context, numbered as that row's tokens are, the row's tokens
    its own state's; the window's contexts end to end within its context budget."""
    from test_decision_context import windows_of

    # A window of 512 tokens holds several rows of a short state, and one of 768 context tokens several of
    # their schemas.
    objective = read_in_context(segments=3, max_len=512)
    rows = labelled()
    several = 0
    for window in windows_of(objective, rows, count=8):
        segments, contexts = np.asarray(window["segments"]), np.asarray(window["context_segments"])
        tokens, context = np.asarray(window["tokens"]), np.asarray(window["context"])
        owners, spans = np.asarray(window["question_rows"]), np.asarray(window["spans"])
        assert context.shape[-1] == objective.context_window
        for batch_row in range(len(segments)):
            standing = [number for number in np.unique(segments[batch_row]) if number > 0]
            several += len(standing) > 1
            for question in np.flatnonzero(np.asarray(window["questions"])[batch_row]):
                owner, (start, end) = owners[batch_row, question], spans[batch_row, question]
                assert owner in standing and set(contexts[batch_row, start:end]) == {owner}
                instructions = {asked.instructions for schema in SCHEMAS for asked in schema.values()}
                assert text(context[batch_row, start:end]) in instructions
                real = np.asarray(window["options"])[batch_row, question]
                for first, last in np.asarray(window["option_spans"])[batch_row, question][real]:
                    assert set(contexts[batch_row, first:last]) == {owner}
                held = text(tokens[batch_row][segments[batch_row] == owner])
                assert sum(f"STATE:\n{state}\n" in held for state, _ in ROWS[:4]) == 1, held
    assert several, "no window held more than one row"


PACKED_SCHEMA_SEEDS = 128
"""Check 1's declared seeds for packed rows reading their own schemas against each row alone, sized
paired (`seeds_for` with the packed windows as `dew_fn`, 2026-10-28 on armada). The check reads 128
draws of windows of 1,024 tokens beside 2,048 of contexts, 1,540 s on a CI core, so it runs in the whole
suite alone (`whole_suite`)."""


@pytest.mark.whole_suite
def test_packed_rows_decide_each_schema_in_context_as_each_row_alone():
    """Rows whose schemas are their contexts, packed several a window, decide each row's options as the
    row does in a window of its own, through the unmodified objective, its cross-attention open (every
    gate 1): each row's head pools its own schema's questions and options from the window's contexts
    and reads its own state, the real options' logits on par from float64 (check 1, over drawn
    parameters)."""
    from test_decision_context import assert_packed_as_alone

    packed, alone = (read_in_context(segments=most, max_len=512) for most in (3, 1))
    assert_packed_as_alone(packed, alone, labelled()[:8], "packed schemas in context", PACKED_SCHEMA_SEEDS)
