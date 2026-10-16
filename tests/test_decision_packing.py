"""Decision rows packed into windows: each row decides as it does alone, and a run is one program."""

import collections
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
from dew.data import Loading
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


def apart(batch) -> dict:
    """Each row of the batch's windows in a window of its own, as the packer lays out a window that
    holds one row: the row, then segment-0 padding to the window's length, and its question slots
    first. Every row's window has one shape, so the rows alone run as one batch."""
    length, questions = batch["tokens"].shape[1], batch["kinds"].shape[1]

    def widened(name, value):
        """`value` padded with zeros along the row's tokens or questions to the window's."""
        wanted = length if name in ("tokens", "positions", "slots", "segments") else questions
        return np.pad(value, [(0, 0), (0, wanted - value.shape[1]), *[(0, 0)] * (value.ndim - 2)])

    ones = []
    for window, row in rows_of(batch):
        one = {name: widened(name, value) for name, value in alone(batch, window, row).items()
               if name not in ("option_slots", "slotted")}
        if "option_slots" in batch:
            slots, real = compacted(one["options"][0], batch["option_slots"].shape[1])
            one["option_slots"], one["slotted"] = slots[None], real[None]
        ones.append(one)
    return {name: np.concatenate([one[name] for one in ones]) for name in ones[0]}


def gradient(objective, variables, batch):
    """The gradient of the batch's summed loss."""
    step = Step(jnp.asarray(0), jax.random.key(0), None, training=False)

    def total(variables):
        loss, _ = objective.loss(variables, batch, step)
        return loss.total

    return jax.grad(total)(variables)


def decided(objective, variables, batch, segments: int):
    """The batch's logits and the gradient of its summed loss."""
    return objective.model.logits(variables, laid_out(batch, segments)), gradient(objective, variables, batch)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_a_packed_window_decides_each_row_as_the_row_alone(arm):
    """Rows packed into windows (segments, positions restarting per row, spans moved) give each
    question's logits and the parameters' gradients each row gives in a window of its own, through
    Qwen3.5's Gated DeltaNet conv and state and its attention, and through the head, within the
    float64 gate: no state, attention, pooled span or last-token feature crosses a row. The float64
    truth is the rows in windows of their own in float64."""
    objective = objective_for(arm)
    assert_rows_decided_alone(objective, (objective.model.backbone.clone(dtype=jnp.float64),
                                          objective.model.head.clone(dtype=jnp.float64)))


@pytest.mark.gpu(reason="FlashAttention's packed sequences run on CUDA")
def test_a_bf16_joint_head_on_flash_reads_each_packed_row_alone(without_deterministic_ops):
    """The joint schema head in bf16 on flash: its routing and field layers read the window's tokens
    by documents, flash's packed sequences of the keys' own offsets, its padded questions and slots
    reading nothing. Each packed row's logits and the gradients are the row's alone within the
    float64 gate, the truth the head in float64 on the xla kernel. A head that read the window
    through a dense mask would not run on flash at all."""
    layout = JointLayout(max_len=1536)
    objective = DecisionObjective(
        attending(layout.max_len), tokenizer=TOKENIZER, specials=SPECIALS, layout=layout,
        head=JointSchemaHead(32, width=128, routing_layers=1, layers=1, heads=2, feedforward=64,
                             dtype=jnp.bfloat16, attention_impl="flash"),
        loss=LogLoss(), shuffle_options=False)
    head = objective.model.head.clone(dtype=jnp.float64, attention_impl="xla")
    # A bf16 scalar's gradient is one rounding draw, the extreme-value statistic the RMS rule does
    # not hold (tests/reference_error.py): the residual gate's ran from 1.2e-4 to 1.1e-3 from
    # float64 over packed and alone, flash and xla, on an A100. The head's attention projections
    # carry flash's forward and backward into many entries each.
    def attending_in_the_head(path: str) -> bool:
        return "['head']" in path and ("attn" in path or "['attention']" in path)

    assert_rows_decided_alone(objective, (objective.model.backbone.clone(dtype=jnp.float64), head),
                              held=attending_in_the_head)


def assert_rows_decided_alone(objective, wide_modules, held=lambda path: True) -> None:
    """Two packed windows of the examples decide each row's questions as the row does in a window
    of its own, logits and the gradients of the leaves `held` names (by `jax.tree_util.keystr`)
    within the float64 gate, the truth `wide_modules`, the backbone and head in float64, over the
    rows alone."""
    batch = windows(objective, 2)
    assert max(int(segments.max()) for segments in batch["segments"]) > 1, "no window packed two rows"
    # A sequence kernel runs as many documents as the coordinates bound, and a window's padding as one
    # more sequence (`dew.nn.attention.flash_attention`).
    segments = objective.bounds.segments
    laid = laid_out(batch, segments).coordinates()
    assert int(np.max(laid.segments)) <= laid.max_segments == segments
    variables = objective.init(jax.random.key(0))
    packed_logits, packed_grads = decided(objective, variables, batch, segments)
    order = [owners(batch)[window] == row for window, row in rows_of(batch)]
    packed = np.concatenate([np.asarray(packed_logits)[window][asked][batch["options"][window][asked]]
                             for (window, _), asked in zip(rows_of(batch), order, strict=True)])
    rows = apart(batch)
    logits, grads = decided(objective, variables, rows, segments)
    with jax.enable_x64(new_val=True):
        wide = copy.copy(objective)
        wide.substitute(wide_modules)
        widened = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables)
        # Each gradient leaf's terms in float64: a key bias's and the scorer's shared bias
        # gradients cancel (q.b shifts every key's logit alike; the residual gate shifts every
        # option's), and a scalar scale sums every option's terms, so their terms, not their
        # near-zero sizes, are their rounding's scale (tests/reference_error.py).
        truth_grads, terms = float64_terms(lambda held: gradient(wide, held, rows), widened)
        truth = np.asarray(wide.model.logits(widened, laid_out(rows, segments)))[rows["options"]]
    logits = np.asarray(logits)[rows["options"]]

    def kept(tree):
        return {name: leaf for name, leaf in ((jax.tree_util.keystr(path), leaf)
                                              for path, leaf in jax.tree_util.tree_leaves_with_path(tree))
                if held(name)}

    packed_grads, grads, truth_grads, terms = (kept(tree)
                                               for tree in (packed_grads, grads, truth_grads, terms))
    assert packed_grads, "no gradient leaf held"
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


def test_segments_caps_the_rows_a_window_holds_and_declares():
    """`segments` holds every packed window to that many rows, which its packer enforces and its
    windows declare to the backbone's sequence kernel (`Coordinates.max_segments`), rather than
    as many as the layout's shortest row fits."""
    layout, backbone, head = ARMS["marker"]
    capped = DecisionObjective(backbone(layout.max_len), tokenizer=TOKENIZER, specials=SPECIALS,
                               layout=layout, head=head(), loss=LogLoss(), shuffle_options=False, segments=1)
    assert capped.data().pack.segments == capped.bounds.segments == 1
    assert int(windows(objective_for("marker"), 4)["segments"].max()) > 1, "no window packed two rows"
    batch = windows(capped, 4)
    assert int(batch["segments"].max()) == 1
    assert laid_out(batch, capped.bounds.segments).coordinates().max_segments == 1


def test_a_batch_of_requests_holds_each_row_and_its_padding_within_the_sequence_capacity():
    """Requests of different lengths, right-padded into one batch, one row a window: the backbone's
    sequence kernel runs each row and its padding as two sequences (`dew.nn.inputs.cu_seqlens`),
    the row within the documents the rows' coordinates state and the padding the one more the kernel
    adds, so no request runs in another's."""
    rows = [row for example in EXAMPLES for row in ROWS.rows(TOKENIZER, SPECIALS, example.state,
                                                             example.questions)]
    laid = DecisionInputs.collate(rows, SPECIALS.pad).coordinates()
    starts = np.asarray(laid.starts)
    assert starts[:, 1:].any(), "no request was padded"
    assert 1 + int(starts[:, 1:].sum(axis=1).max()) <= laid.max_segments + 1
    assert int(np.max(laid.segments)) == laid.max_segments == 1


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


@pytest.mark.parametrize("causal", [True, False])
def test_a_joint_head_given_no_layout_reads_every_question_of_a_request_in_one_row(causal):
    """A decision objective given the joint head and no layout lays a request out
    as that head reads it, every question in one row (`JointLayout`), over a
    causal and a bidirectional backbone alike, as a run's config does: it
    scores a three-question request as the same objective given the
    JointLayout does."""
    example = EXAMPLES[4]
    backbone = CausalTransformer(vocab_size=256, emb_features=32, num_layers=2, num_heads=2, mlp_features=48,
                                 max_seq_len=2048, attention_impl="xla", causal=causal)
    head = JointSchemaHead(32, width=32, routing_layers=1, layers=1, heads=2, feedforward=64)

    def objective(**layout):
        return DecisionObjective(backbone, tokenizer=TOKENIZER, specials=SPECIALS, head=head,
                                 shuffle_options=False, **layout)

    stated, defaulted = objective(layout=JointLayout()), objective()
    # Drawn over a window the backbone holds; the weights are the same whatever the layout's length.
    variables = objective(layout=JointLayout(max_len=2048)).init(jax.random.key(0))

    def scores(objective):
        rows = objective.layout.rows(TOKENIZER, SPECIALS, example.state, example.questions)
        return np.asarray(objective.model.logits(variables, DecisionInputs.collate(rows, SPECIALS.pad)))

    expected = scores(stated)
    assert expected.shape[:2] == (1, len(example.questions))
    np.testing.assert_array_equal(scores(defaulted), expected)


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


class Counting(ByteTokenizer):
    """The byte vocabulary, counting the texts it tokenizes. It does not hash,
    as a tokenizer that compares by value need not."""

    __hash__ = None

    def __init__(self):
        super().__init__()
        self.texts: collections.Counter[str] = collections.Counter()

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        self.texts[text] += 1
        return super().encode(text, add_special_tokens=add_special_tokens)


@pytest.mark.parametrize("arm", ["joint", "marker"])
def test_a_reader_tokenizes_each_text_of_its_rows_once(arm):
    """A reader lays each record out twice, to fit it and to train on it, and
    every record asks its questions over options in a fresh order, yet each
    text is tokenized once: each template, instruction and option once for
    the run, and each state as its record is fit, which the row it trains on
    lays out again."""
    layout, _, head = ARMS[arm]
    tokenizer = Counting()
    objective = DecisionObjective(attending(layout.max_len), tokenizer=tokenizer, specials=SPECIALS,
                                  layout=layout, head=head(), loss=LogLoss())
    data = loader(Decisions(EXAMPLES), batch=2, read="memory", objective=objective.data(),
                  loading=Loading(workers=0))
    reader = data.train(DataPartition())
    try:
        assert len(list(itertools.islice(reader, 4))) == 4
    finally:
        reader.close()
    assert tokenizer.texts and set(tokenizer.texts.values()) == {1}


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
