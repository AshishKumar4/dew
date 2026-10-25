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


def read_in_context(length: int = 768, head=None) -> DecisionObjective:
    """A decision over the tiny Llama with fresh cross-attention before its layers, its rows' schemas
    read as their contexts, up to `length` tokens."""
    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference", max_seq_len=1024)
    crossed = source.with_cross_attention(CrossAttention((0, 1), length), key=0)
    width, _ = DecisionModel.head_size(crossed.model)
    head = head or JointSchemaHead(width, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)
    return DecisionObjective(crossed, tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT, head=head)


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
    labels = {"a": {"holds": "true", "close": "x"}, "b": {"holds": "false", "close": "y"},
              "c": {"holds": "true", "close": "y"}}
    rows = [Example(state, QUESTIONS, {"pick": label, **labels[label]}) for state, label in ROWS]
    data = decision_rows(rows, 8, objective=objective.data())
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
