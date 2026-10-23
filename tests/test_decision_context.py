"""A decision that reads a context beside its state, through gated cross-attention layers attached to a
pretrained backbone: trained as an adapter, counted in its training work, and served."""

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
from dew.decision import Choice, Decide, Example
from dew.decision.layout import Specials
from dew.decision.objective import DecisionObjective
from dew.decision.questions import ChoiceAnswer
from dew.interop import Pretrained
from dew.nn.cross_attention import CrossAttention, crossing
from dew.objectives.base import FROZEN
from dew.training import Trainer

pytestmark = pytest.mark.mesh

LLAMA = Path(__file__).parent / "fixtures" / "hf" / "llama-tiny"

TICKETS = [("charged twice for March", "billing", "a card holder since 2019"),
           ("the site is down", "technical", None),
           ("how much is the pro plan", "sales", "asked about the price list before"),
           ("refund my last invoice", "billing", "two refunds this year"),
           ("error 500 on login", "technical", "uses the mobile app"),
           ("can I upgrade", "sales", None)] * 4
"""Tickets, their team, and a note about the customer a decision may read beside the ticket: none
for some."""


SPECIALS = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
"""A byte tokenizer's specials, over the fixture's 256-token vocabulary."""

QUESTION = {"team": Choice("Which team?", ["billing", "technical", "sales"])}


def conditioned(length: int = 16, layers: tuple[int, ...] = (0, 1)) -> DecisionObjective:
    """A decision over the tiny Llama with fresh cross-attention before `layers`, its context up to
    `length` tokens, its rows laid out by bytes."""
    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    return DecisionObjective(source.with_cross_attention(CrossAttention(layers, length), key=0),
                             tokenizer=ByteTokenizer(), specials=SPECIALS)


def examples() -> list[Example]:
    return [Example(text, QUESTION, {"team": label}, context=note) for text, label, note in TICKETS]


def test_a_backbone_with_fresh_context_layers_scores_every_row_as_its_source_does():
    """The cross-attention blocks start from zero gates, so a decision over the backbone with them
    scores a row, with its context or without, exactly as the backbone alone does; the blocks train
    and the source's weights are held frozen."""
    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    plain = DecisionObjective(source, tokenizer=ByteTokenizer(), specials=SPECIALS)
    crossed = conditioned(8, (0,))
    assert crossed.context_length == 8 and plain.context_length is None
    assert crossed.variables is not None
    trained = jax.tree_util.tree_leaves_with_path(crossed.variables["params"]["backbone"])
    assert trained and all(crossing(tuple(entry.key for entry in path)) for path, _ in trained)
    assert FROZEN in crossed.variables

    asked = Decide(plain.model, plain.init(jax.random.key(0)), plain.layout, plain.tokenizer, plain.specials)
    variables = crossed.init(jax.random.key(0))
    variables = {**variables, "params": {**variables["params"], "head": asked.variables["params"]["head"]}}
    reading = Decide(crossed.model, variables, crossed.layout, crossed.tokenizer, crossed.specials)
    alone = asked("charged twice for March", QUESTION)["team"]
    for context in (None, "a card holder since 2019"):
        read = reading("charged twice for March", QUESTION, context=context)["team"]
        assert isinstance(read, ChoiceAnswer) and isinstance(alone, ChoiceAnswer)
        np.testing.assert_array_equal(read.probabilities, alone.probabilities)


def test_a_conditioned_run_logs_its_training_work_and_serves_with_its_context():
    """Examples with a context trained through context layers: every logged step reports its
    training work, and beyond what its batch's rows cost without their contexts by the
    cross-attention's work over them (`test_the_context_layers_work_is_counted_in_a_conditioned_batch`
    holds the count itself), far past the float32 sum the step logs it by; the run's task answers a
    request with its context."""
    objective = conditioned()
    data = decision_rows(examples(), 8, objective=objective.data())
    batches = list(itertools.islice(data.train(DataPartition()), 2))
    plain = [objective.training_flops(None, {name: value for name, value in batch.items()
                                             if name != "context_lengths"}) for batch in batches]
    tracker = RecordingTracker()
    trainer = Trainer(objective, optax.adam(1e-3), key=0, tracker=tracker)
    state = trainer.fit(data, steps=2, log_every=1)
    logged = [scalars["train/flops"] for _, scalars in tracker.scalars if "train/loss" in scalars]
    assert len(logged) == 2 and all(work > without for work, without in zip(logged, plain, strict=True))

    task = objective.pipeline(state)
    assert isinstance(task, Decide)
    answer = task("charged twice for March", QUESTION, context="a card holder since 2019")["team"]
    assert isinstance(answer, ChoiceAnswer) and answer.choice in ("billing", "technical", "sales")


def test_the_context_layers_work_is_counted_in_a_conditioned_batch():
    """The objective's training work for a batch whose rows carry a context is the frozen backbone's,
    three passes of its cross-attention blocks over each row's real context tokens
    (`CrossAttending.cross_flops`), and three of its head; a batch without a context counts no
    cross-attention work."""
    from dew.nn.flops import counted_flops, frozen_training_flops, parameter_flops

    objective = conditioned(8, (0,))
    lengths = np.asarray([5, 7])
    # Two windows of one row each, one question of three options a row.
    batch = {"segments": (np.arange(8) < lengths[:, None]).astype(np.int32),
             "spans": np.zeros((2, 1, 2), np.int32), "questions": np.ones((2, 1), bool),
             "options": np.ones((2, 1, 3), bool), "context_lengths": np.asarray([3, 0], np.int32)}
    np.testing.assert_array_equal(objective.real_lengths(batch), lengths)
    backbone = objective.model.backbone
    cross = backbone.cross_flops(lengths, np.asarray([3, 0]))
    assert cross > 0
    head = objective.model.head.decision_flops(lengths, np.ones(2, np.int64), np.full(2, 3, np.int64))
    frozen = frozen_training_flops(counted_flops(backbone, lengths), parameter_flops(backbone, lengths))
    assert objective.training_flops(None, batch) == frozen + 3 * (cross + head)
    plain = {name: value for name, value in batch.items() if name != "context_lengths"}
    assert objective.training_flops(None, plain) == frozen + 3 * head
