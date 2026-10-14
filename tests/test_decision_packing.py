"""Decision rows packed into windows: each row decides as it does alone, and a run is one program."""

import copy
import itertools
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from recording import RecordingTracker
from reference_error import assert_leaves_as_exact, float64_terms

from dew.config.data import loader
from dew.data.dataset import DataPartition
from dew.data.text import ByteTokenizer
from dew.decision import (
    Choice,
    DecisionObjective,
    Example,
    JointLayout,
    LogLoss,
    Noul,
    Score,
    Specials,
    StateFirstLayout,
)
from dew.decision.data import Decisions
from dew.decision.head import DecisionHead, JointSchemaHead
from dew.decision.layout import OPTION_TILE, DecisionInputs, compacted
from dew.decision.objective import _ALONG_THE_QUESTIONS
from dew.decision.task import laid_out
from dew.interop import Pretrained
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import VALID_ROWS, Step
from dew.training.trainer import Trainer

QWEN35 = Path(__file__).parent / "fixtures" / "hf" / "qwen35-tiny"
"""Qwen3.5's architecture, tiny: three Gated DeltaNet layers (conv and recurrent state) and a gated
attention layer."""

TOKENIZER = ByteTokenizer()
SPECIALS = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
TEAM = Choice("Which team?", {"billing": "payments", "technical": "outages", "sales": "pricing"})
URGENT = Noul("Is it urgent?")
SEVERITY = Score("How severe?", ["none", "minor", "major"])
EXAMPLES = [
    Example("charged twice", {"team": TEAM}, {"team": "billing"}),
    Example("the site is down and every page answers error 500 since noon", {"team": TEAM, "urgent": URGENT},
            {"team": "technical", "urgent": "true"}),
    Example("how much is pro", {"team": TEAM, "severity": SEVERITY}, {"team": "sales"},
            {"severity": (0.7, 0.2, 0.1)}),
    Example({"ticket": "refund", "body": "please refund my second charge"}, {"urgent": URGENT},
            {"urgent": "false"}),
    Example("cannot log in", {"team": TEAM, "urgent": URGENT, "severity": SEVERITY},
            {"team": "technical", "urgent": "true", "severity": 2}),
    Example("upgrade cost", {"team": TEAM}, {"team": "sales"}),
]


def qwen35(length: int) -> Pretrained:
    return Pretrained.load(str(QWEN35), dtype="float32", attention_impl="reference", max_seq_len=length)


def attending(length: int) -> CausalTransformer:
    """A decoder of attention alone, which the parallel layout's pairwise mask needs."""
    return CausalTransformer(vocab_size=256, emb_features=32, num_layers=2, num_heads=2, mlp_features=48,
                             max_seq_len=length, attention_impl="xla")


ROWS = StateFirstLayout(max_len=256, head_max_len=64, option_tokens=8)
ARMS = {
    "joint": (JointLayout(max_len=1536), qwen35,
             lambda: JointSchemaHead(64, width=32, routing_layers=1, layers=1, heads=2, feedforward=64)),
    "marker": (ROWS, qwen35, lambda: DecisionHead(64, layers=1)),
    "parallel": (replace(ROWS, parallel=True), attending, lambda: DecisionHead(32, layers=1)),
}
"""Joint rows under the joint schema head and rows of one question under the marker head,
both over Qwen3.5; parallel options, whose pairwise mask Gated DeltaNet refuses, over attention."""


def objective_for(arm: str) -> DecisionObjective:
    layout, backbone, head = ARMS[arm]
    return DecisionObjective(backbone(layout.max_len), tokenizer=TOKENIZER, specials=SPECIALS, layout=layout,
                             head=head(), loss=LogLoss(), shuffle_options=False)


def windows(objective, rows: int) -> dict:
    """A batch of `rows` packed windows of the examples, as the run reads it."""
    data = loader(Decisions(EXAMPLES, validation=EXAMPLES), batch=rows, read="memory",
                objective=objective.data())
    reader = data.val(DataPartition())
    try:
        batch = next(reader)
    finally:
        reader.close()
    return {name: np.asarray(value) for name, value in batch.items() if name != VALID_ROWS}


def owners(batch) -> np.ndarray:
    """Each question slot's row in its window, 0 for a padding slot."""
    rows = np.take_along_axis(batch["segments"], batch["spans"][..., 0], axis=1)
    return np.where(batch["questions"], rows, 0)


def alone(batch, window: int, row: int) -> dict:
    """Row `row` of window `window` as a window of its own, its spans counted from its first token."""
    where = np.flatnonzero(batch["segments"][window] == row)
    start, end = int(where[0]), int(where[-1]) + 1
    asked = np.flatnonzero(owners(batch)[window] == row)
    one = {name: batch[name][window, asked][None] for name in _ALONG_THE_QUESTIONS}
    along = ("tokens", "positions", "slots")
    one |= {name: batch[name][window, start:end][None] for name in along if name in batch}
    one |= {"segments": np.ones((1, end - start), np.int32), "spans": one["spans"] - start,
            "option_spans": one["option_spans"] - start * one["options"][..., None]}
    if "option_slots" in batch:
        slots, real = compacted(one["options"][0], OPTION_TILE)
        one["option_slots"], one["slotted"] = slots[None], real[None]
    return one


def rows_of(batch) -> list[tuple[int, int]]:
    return [(window, row) for window, segments in enumerate(batch["segments"])
            for row in range(1, int(segments.max()) + 1)]


def decided(objective, variables, batch, segments: int):
    """The batch's logits and the gradient of its summed loss."""
    step = Step(jnp.asarray(0), jax.random.key(0), None, training=False)

    def total(variables):
        loss, _ = objective.loss(variables, batch, step)
        return loss.total

    logits = objective.model.logits(variables, laid_out(batch, segments))
    return logits, jax.grad(total)(variables)


def separately(objective, variables, batch):
    """Each row's logits on its real options, alone, and the gradient summed over the rows."""
    logits = [np.asarray(objective.model.logits(variables, laid_out(one, 1)))[one["options"]]
              for one in (alone(batch, window, row) for window, row in rows_of(batch))]
    return np.concatenate(logits), summed_gradient(objective, variables, batch)


def summed_gradient(objective, variables, batch):
    """The gradient of the rows' losses, each row alone, summed over the rows."""
    rows = [decided(objective, variables, alone(batch, window, row), 1)[1] for window, row in rows_of(batch)]
    return jax.tree.map(lambda *leaves: sum(leaves), *rows)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_a_packed_window_decides_each_row_as_the_row_alone(arm):
    """Rows packed into windows (segments, positions restarting per row, spans moved) give each
    question's logits and the parameters' gradients the rows give one at a time, through Qwen3.5's
    Gated DeltaNet conv and state and its attention, and through the head, within the float64 gate:
    no state, attention, pooled span or last-token feature crosses a row. The float64 truth is the
    rows run alone in float64."""
    objective = objective_for(arm)
    batch = windows(objective, 2)
    assert max(int(segments.max()) for segments in batch["segments"]) > 1, "no window packed two rows"
    # A sequence kernel holds as many documents as the coordinates bound, each row's first and its
    # padding among them (`dew.nn.linear._cu_seqlens`).
    laid = laid_out(batch, objective.bounds.segments).coordinates()
    assert 1 + int(np.sum(np.asarray(laid.starts)[:, 1:], axis=1).max()) <= laid.max_segments
    variables = objective.init(jax.random.key(0))
    packed_logits, packed_grads = decided(objective, variables, batch, objective.bounds.segments)
    order = [owners(batch)[window] == row for window, row in rows_of(batch)]
    packed = np.concatenate([np.asarray(packed_logits)[window][asked][batch["options"][window][asked]]
                             for (window, _), asked in zip(rows_of(batch), order, strict=True)])
    logits, grads = separately(objective, variables, batch)
    with jax.enable_x64(new_val=True):
        wide = copy.copy(objective)
        wide.substitute((objective.model.backbone.clone(dtype=jnp.float64),
                         objective.model.head.clone(dtype=jnp.float64)))
        widened = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables)
        truth, _ = separately(wide, widened, batch)
        # Each gradient leaf's terms in float64: a key bias's and the scorer's shared bias
        # gradients cancel (q.b shifts every key's logit alike; the residual gate shifts every
        # option's), and a scalar scale sums every option's terms, so their terms, not their
        # near-zero sizes, are their rounding's scale (tests/reference_error.py).
        truth_grads, terms = float64_terms(lambda held: summed_gradient(wide, held, batch), widened)
    assert_leaves_as_exact({"logits": packed, "grads": packed_grads}, {"logits": logits, "grads": grads},
                           {"logits": truth, "grads": truth_grads}, reductions=objective.layout.max_len,
                           terms={"logits": truth, "grads": terms})


def test_a_decision_run_steps_one_program_over_rows_of_every_length():
    """A token budget of packed windows: every batch is the one shape the bounds give, so rows from
    a short question to a three-question schema train through one compiled step, and the loss is
    finite on every step."""
    objective = objective_for("joint")
    layout = objective.layout
    data = loader(Decisions(EXAMPLES), tokens=8 * layout.max_len, read="memory", objective=objective.data())
    reader = data.train(DataPartition())
    try:
        shapes = {name: {np.shape(value) for value in values} for name, values in
                  _columns(itertools.islice(reader, 4)).items()}
    finally:
        reader.close()
    bounds = objective.bounds
    assert shapes["tokens"] == {(8, layout.max_len)} and shapes["kinds"] == {(8, bounds.questions)}
    assert shapes["option_slots"] == {(8, -(-bounds.options // OPTION_TILE) * OPTION_TILE)}
    tracker = RecordingTracker()
    Trainer(objective, optax.adam(1e-3), key=0, tracker=tracker).fit(data, steps=3, log_every=1)
    logged = [scalars for _, scalars in tracker.scalars if "train/loss" in scalars]
    assert [scalars["train/compiles"] for scalars in logged] == [1, 1, 1]
    assert all(np.isfinite(scalars["train/loss"]) for scalars in logged)


def test_a_batch_of_requests_holds_each_row_and_its_padding_within_the_sequence_capacity():
    """Requests of different lengths, right-padded into one batch, one row a window: the backbone's
    sequence kernel counts each row's tokens and its padding as documents (`_cu_seqlens`), every
    one within the capacity the rows' coordinates state, so no request runs in another's."""
    rows = [row for example in EXAMPLES for row in ROWS.rows(TOKENIZER, SPECIALS, example.state,
                                                             example.questions)]
    laid = DecisionInputs.collate(rows, SPECIALS.pad).coordinates()
    starts = np.asarray(laid.starts)
    assert starts[:, 1:].any(), "no request was padded"
    assert 1 + int(starts[:, 1:].sum(axis=1).max()) <= laid.max_segments


def test_a_one_question_head_reads_joint_rows_only_of_one_question():
    """The marker head reads one question a row: over joint rows of a schema that asks several, the run
    is refused before a row is read; a schema of one question reads as rows of one."""
    objective = DecisionObjective(qwen35(1536), tokenizer=TOKENIZER, specials=SPECIALS,
                                  layout=JointLayout(max_len=1536), head=DecisionHead(64, layers=1),
                                  shuffle_options=False)
    with pytest.raises(ValueError):
        loader(Decisions(EXAMPLES), batch=2, read="memory", objective=objective.data())
    single = [example for example in EXAMPLES if len(example.questions) == 1]
    data = loader(Decisions(single), batch=2, read="memory", objective=objective.data())
    reader = data.train(DataPartition())
    try:
        assert np.shape(next(reader)["tokens"])[0] == 2
    finally:
        reader.close()


@pytest.mark.parametrize("questions", [1, 2])
def test_single_option_questions_fit_the_packing_bounds(questions):
    """A fitting singleton-choice schema packs even where a two-option row cannot fit."""
    example = Example("", {f"q{index}": Choice("x", ["0"]) for index in range(questions)},
                      {f"q{index}": "0" for index in range(questions)})
    (encoded,) = JointLayout().rows(TOKENIZER, SPECIALS, example.state, example.questions)
    length = len(encoded.tokens) if questions == 1 else 500
    layout = JointLayout(max_len=length)
    assert layout.fits(encoded)
    objective = DecisionObjective(attending(length), tokenizer=TOKENIZER, specials=SPECIALS, layout=layout,
                                  head=JointSchemaHead(32, width=32, routing_layers=1, layers=1,
                                                       heads=2, feedforward=64), shuffle_options=False)
    data = loader(Decisions([example], validation=[example]), batch=1, read="memory",
                  objective=objective.data())
    (batch,) = list(data.val(DataPartition()))
    batch = {name: np.asarray(value) for name, value in batch.items()}
    np.testing.assert_array_equal(batch["tokens"][0, batch["segments"][0] == 1], encoded.tokens)
    assert int(batch["questions"].sum()) == questions
    assert int(batch["options"].sum()) == questions
    variables = objective.init(jax.random.key(0))
    step = Step(jnp.asarray(0), jax.random.key(0), None)
    loss, _ = objective.loss(variables, batch, step)
    assert float(loss.mass) == questions and float(loss.total) == 0
    probabilities = objective.evaluate(variables, batch, step).probabilities
    np.testing.assert_array_equal(np.asarray(probabilities)[batch["options"]], np.ones(questions))


def test_a_packed_loss_counts_only_real_scored_questions():
    """Different joint-row lengths and label counts contribute question charges, not token,
    window, unanswered or padding counts; a repeated held window contributes neither part."""
    examples = [
        Example("short", {"q0": Choice("x", ["a", "b"])}, {"q0": "a"}),
        Example("state " + "x" * 80,
                {"q0": Choice("pick", ["0", "1", "2"]),
                 "q1": Score("rate", ["low", "middle", "high"]),
                 "q2": Choice("unlabelled", ["u", "v"])},
                {"q0": "2"}, targets={"q1": (0.2, 0.3, 0.5)}),
        Example("last " + "x" * 160,
                {"q0": Choice("x", ["no", "yes"]), "q1": Choice("x", ["u", "v"]),
                 "q2": Noul("x"), "q3": Choice("x", ["0", "1", "2", "3"])}, {"q0": "yes"}),
    ]
    layout = JointLayout(max_len=1536)
    objective = DecisionObjective(attending(layout.max_len), tokenizer=TOKENIZER, specials=SPECIALS,
                                  layout=layout, head=JointSchemaHead(32, width=32, routing_layers=1,
                                      layers=1, heads=2, feedforward=64), shuffle_options=False)
    data = loader(Decisions(examples, validation=examples), batch=3, read="memory",
                  objective=objective.data())
    (batch,) = list(data.val(DataPartition()))
    batch = {name: np.asarray(value) for name, value in batch.items()}
    real = batch[VALID_ROWS]
    assert real.sum() == 2 and (~real).sum() == 1
    assert int(batch["segments"][real].max()) > 1
    assert (~batch["questions"]).any()
    assert int((batch["questions"] & ~batch["scored"])[real].sum()) == 4
    (repeated,) = np.flatnonzero(~real)
    assert any(np.array_equal(batch["tokens"][repeated], tokens) for tokens in batch["tokens"][real])
    variables = objective.init(jax.random.key(0))
    charges, lengths = [], []
    for example in examples:
        (encoded,) = layout.rows(TOKENIZER, SPECIALS, example.state, example.questions)
        lengths.append(len(encoded.tokens))
        inputs = DecisionInputs.collate([encoded], SPECIALS.pad)
        logits = np.asarray(objective.model.logits(variables, inputs), np.float64)[0]
        for slot, laid in enumerate(encoded.questions):
            if laid.name not in example.labels():
                continue
            shown = logits[slot, :len(laid.options)]
            shifted = shown - shown.max()
            log_probabilities = shifted - np.log(np.exp(shifted).sum())
            gold = example.distribution(laid.name)[list(laid.order)]
            charges.append(-float(np.dot(gold, log_probabilities)))
    assert len(set(lengths)) == len(examples)
    count = sum(len(example.labels()) for example in examples)
    assert len(charges) == count == 4
    step = Step(jnp.asarray(0), jax.random.key(0), None)
    loss, _ = objective.loss(variables, batch, step)
    assert float(loss.mass) == count
    assert float(loss.total) == pytest.approx(sum(charges), rel=1e-5)
    reduced, active = objective.reduce_loss(loss)
    assert bool(active)
    assert float(reduced) == pytest.approx(sum(charges) / count, rel=1e-5)
    unmarked = {name: value for name, value in batch.items() if name != VALID_ROWS}
    counted_repeats, _ = objective.loss(variables, unmarked, step)
    assert float(counted_repeats.mass) == int(batch["scored"].sum()) > count
    covered = batch | {VALID_ROWS: np.zeros_like(real)}
    empty, _ = objective.loss(variables, covered, step)
    value, active = objective.reduce_loss(empty)
    assert float(empty.mass) == float(empty.total) == float(value) == 0 and not bool(active)


def _columns(batches) -> dict[str, list]:
    columns: dict[str, list] = {}
    for batch in batches:
        for name, value in batch.items():
            columns.setdefault(name, []).append(value)
    return columns
