"""A decision that reads a context beside its state, through gated cross-attention layers attached to a
pretrained backbone: trained as an adapter, counted in its training work, and served."""

import csv
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from dew.data import Files, Loading
from dew.decision import Choice, Decide, decisions
from dew.decision.config import DecisionRunConfig
from dew.decision.objective import DecisionObjective
from dew.decision.questions import ChoiceAnswer
from dew.interop import Pretrained
from dew.nn.cross_attention import CrossAttention, crossing
from dew.objectives.base import FROZEN

pytestmark = pytest.mark.mesh

QWEN = Path(__file__).parent / "fixtures" / "hf" / "qwen38-dense-tiny"

TICKETS = [("charged twice for March", "billing", "a card holder since 2019"),
           ("the site is down", "technical", None),
           ("how much is the pro plan", "sales", "asked about the price list before"),
           ("refund my last invoice", "billing", "two refunds this year"),
           ("error 500 on login", "technical", "uses the mobile app"),
           ("can I upgrade", "sales", None)] * 4
"""Tickets, their team, and a note about the customer a decision may read beside the ticket: none
for some."""


def written(directory: Path) -> Path:
    table = directory / "tickets.csv"
    with table.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label", "notes"])
        writer.writerows((text, label, "" if note is None else note) for text, label, note in TICKETS)
    return table


def test_a_backbone_with_fresh_context_layers_scores_every_row_as_its_source_does():
    """The cross-attention blocks start from zero gates, so a decision over the backbone with them
    scores a row, with its context or without, exactly as the backbone alone does; the blocks train
    and the source's weights are held frozen."""
    source = Pretrained.load(QWEN, dtype="float32")
    conditioned = source.with_cross_attention(CrossAttention((0,), 8), key=0)
    plain, crossed = DecisionObjective(source), DecisionObjective(conditioned)
    assert crossed.context_length == 8 and plain.context_length is None
    assert crossed.variables is not None
    trained = jax.tree_util.tree_leaves_with_path(crossed.variables["params"]["backbone"])
    assert trained and all(crossing(tuple(entry.key for entry in path)) for path, _ in trained)
    assert FROZEN in crossed.variables

    question = {"team": Choice("Which team?", ["billing", "technical", "sales"])}
    asked = Decide(plain.model, plain.init(jax.random.key(0)), plain.layout, plain.tokenizer, plain.specials)
    head = asked.variables["params"]["head"]
    variables = crossed.init(jax.random.key(0))
    variables = {**variables, "params": {**variables["params"], "head": head}}
    reading = Decide(crossed.model, variables, crossed.layout, crossed.tokenizer, crossed.specials)
    alone = asked("charged twice for March", question)["team"].probabilities
    for context in (None, "a card holder since 2019"):
        read = reading("charged twice for March", question, context=context)["team"].probabilities
        np.testing.assert_array_equal(read, alone)


def test_a_decision_run_trains_its_context_layers_counts_their_work_and_serves_with_them(tmp_path):
    """A table's context column trained through context layers: every logged step reports its
    training work, the cross-attention's over each row's context counted in it (the objective's own
    count, which holds the blocks' pass beside the frozen backbone's), and the run's task answers a
    request with a context as it answers one without, the context read."""
    train, held = Files(str(written(tmp_path)), format="csv").holdout(0.5)

    def asked(rows):
        return (rows.pipe(decisions, question="team", context="notes").read("memory").batch(4)
                .loading(Loading(workers=0)))

    config = DecisionRunConfig.cli([
        "--pretrained", str(QWEN), "--context-layers", "0", "1", "--context-length", "16",
        "--trainer.name", "context", "--trainer.checkpoint-dir", str(tmp_path / "runs"),
        "--trainer.steps", "2", "--trainer.eval-every", "2", "--trainer.checkpoint-every", "2",
        "--trainer.log-every", "1", "--trainer.multi-host", "False",
        "--trainer.compilation-cache-dir", "None", "--model.dtype", "float32"],
        default=DecisionRunConfig(data=asked(train), validation=asked(held)))
    config.run()
    run = tmp_path / "runs" / "context"
    journal = (run / "tracking" / "scalars.jsonl").read_text().splitlines()
    logged = [json.loads(row)["scalars"] for row in journal]
    work = [scalars["train/flops"] for scalars in logged if "train/loss" in scalars]
    assert len(work) == 2 and all(flops > 0 for flops in work)

    decide = Decide.from_run(str(run))
    question = {"team": Choice("Which team?", ["billing", "technical", "sales"])}
    answer = decide("charged twice for March", question, context="a card holder since 2019")["team"]
    assert isinstance(answer, ChoiceAnswer) and answer.choice in ("billing", "technical", "sales")
    decide("charged twice for March", question)


def test_the_context_layers_work_is_counted_in_a_conditioned_batch():
    """The objective's training work for a batch whose rows carry a context is the frozen backbone's,
    three passes of its cross-attention blocks over each row's real context tokens
    (`CrossAttending.cross_flops`), and three of its head; a batch without a context counts no
    cross-attention work."""
    from dew.nn.flops import counted_flops, frozen_training_flops, parameter_flops

    source = Pretrained.load(QWEN, dtype="float32").with_cross_attention(CrossAttention((0,), 8), key=0)
    objective = DecisionObjective(source)
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
