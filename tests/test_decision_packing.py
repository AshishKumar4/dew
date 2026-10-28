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
from decision_support import decision_rows
from recording import RecordingTracker
from reference_error import as_given, assert_as_exact

from dew.batch import VALID_ROWS
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
from dew.decision.head import DecisionHead, JointSchemaHead, LabelReadout
from dew.decision.layout import OPTION_TILE, QUESTION_FIELDS, DecisionInputs, Labels, compacted
from dew.decision.rows import TRAINING_FIELDS
from dew.decision.task import laid_out
from dew.interop import Pretrained
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import Step
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
    """A decoder of attention alone, which the parallel layout's key runs need, on the attention template,
    which reads them as its spans (`DecisionModel._read`, row 1835)."""
    return CausalTransformer(vocab_size=256, emb_features=32, num_layers=2, num_heads=2, mlp_features=48,
                             max_seq_len=length, attention_impl="template")


ROWS = StateFirstLayout(max_len=256, head_max_len=64, option_tokens=8)
LABELS = Labels(variants=("{}",))
"""Labels a byte vocabulary spells as one token each: the bare letter."""
ARMS = {
    "joint": (JointLayout(max_len=1536), qwen35,
             lambda: JointSchemaHead(64, width=32, routing_layers=1, layers=1, heads=2, feedforward=64)),
    "marker": (ROWS, qwen35, lambda: DecisionHead(64, layers=1)),
    "parallel": (replace(ROWS, parallel=True), attending, lambda: DecisionHead(32, layers=1)),
    "readout": (replace(ROWS, head_max_len=128, option_tokens=24, labels=LABELS), qwen35,
                lambda: LabelReadout(64, LABELS.ids(TOKENIZER, 3))),
}
"""Joint rows under the joint schema head and rows of one question under the marker head,
both over Qwen3.5; parallel options, whose key runs Gated DeltaNet refuses, over the attention template;
options under labels read at each row's answer cue by the backbone's own logits, over Qwen3.5."""


def objective_for(arm: str) -> DecisionObjective:
    layout, backbone, head = ARMS[arm]
    return DecisionObjective(backbone(layout.max_len), tokenizer=TOKENIZER, specials=SPECIALS, layout=layout,
                             head=head(), loss=LogLoss(), shuffle_options=False)


def windows(objective, rows: int) -> dict:
    """A batch of `rows` packed windows of the examples, as the run reads it."""
    data = decision_rows(EXAMPLES, rows, EXAMPLES, objective=objective.data())
    reader = data.val(DataPartition())
    try:
        batch = next(reader)
    finally:
        reader.close()
    return {name: np.asarray(value) for name, value in batch.items() if name != VALID_ROWS}


def along(batch) -> tuple[str, ...]:
    """The batch's fields along its questions, its answer cues' where its options stand under labels."""
    return tuple(field.name for field in (*QUESTION_FIELDS, *TRAINING_FIELDS) if field.name in batch)


def owners(batch) -> np.ndarray:
    """Each question slot's row in its window, 0 for a padding slot."""
    rows = np.take_along_axis(batch["segments"], batch["spans"][..., 0], axis=1)
    return np.where(batch["questions"], rows, 0)


def alone(batch, window: int, row: int) -> dict:
    """Row `row` of window `window` as a window of its own, its spans counted from its first token."""
    where = np.flatnonzero(batch["segments"][window] == row)
    start, end = int(where[0]), int(where[-1]) + 1
    asked = np.flatnonzero(owners(batch)[window] == row)
    one = {name: batch[name][window, asked][None] for name in along(batch)}
    tokens = ("tokens", "positions", "slots")
    one |= {name: batch[name][window, start:end][None] for name in tokens if name in batch}
    one |= {"segments": np.ones((1, end - start), np.int32), "spans": one["spans"] - start,
            "option_spans": one["option_spans"] - start * one["options"][..., None]}
    if "readout" in one:
        one["readout"] = one["readout"] - start
    if "option_slots" in batch:
        slots, real = compacted(one["options"][0], OPTION_TILE)
        one["option_slots"], one["slotted"] = slots[None], real[None]
    return one


def rows_of(batch) -> list[tuple[int, int]]:
    return [(window, row) for window, segments in enumerate(batch["segments"])
            for row in range(1, int(segments.max()) + 1)]


def at_offset(batch, window: int, row: int) -> dict:
    """Row `row` of window `window` alone where the window packed it, its question slots first: the
    tokens before it one document of token 0 with no questions, so the window keeps its real tokens
    first as the layout's are, and none of the window's other rows; its tokens at their places; then
    segment-0 padding."""
    segments = batch["segments"][window]
    mine = segments == row
    before = np.arange(segments.size) < np.flatnonzero(mine)[0]
    asked = np.flatnonzero(owners(batch)[window] == row)
    one = {name: batch[name][window, asked][None] for name in along(batch)}
    one |= {name: np.where(mine.reshape(-1, *(1,) * (batch[name].ndim - 2)), batch[name][window], 0)[None]
            for name in ("tokens", "slots") if name in batch}
    if "positions" in batch:
        filler = np.cumsum(before) - 1
        one["positions"] = np.where(mine, batch["positions"][window], np.where(before, filler, 0))[None]
    return one | {"segments": (before + mine * (1 + before.any())).astype(np.int32)[None]}


def in_place(batch, placed: list[tuple[int, int]]) -> dict:
    """`batch` with each `(window, row)` of `placed`, at most one a window, alone in its window where
    the window packed it (`at_offset`), and every other window empty: the packed batch's shapes, so
    every product a row's tokens enter runs at the packed run's shapes. A window reads no other, so
    rows of different windows run alone together."""
    assert len({window for window, _ in placed}) == len(placed), "one row a window"
    one = {name: np.zeros_like(value) for name, value in batch.items()}
    for window, row in placed:
        for name, value in at_offset(batch, window, row).items():
            one[name][window, :value.shape[1]] = value[0]
        if "option_slots" in batch:
            one["option_slots"][window], one["slotted"][window] = compacted(
                one["options"][window], batch["option_slots"].shape[1])
    return one


def apart(batch) -> dict:
    """Each row of the batch's windows in a window of its own, as the packer lays out a window that
    holds one row: the row, then segment-0 padding to the longest row's length, which no real token
    reads, and its question slots first. Every row's window has one shape, so the rows alone run as
    one batch."""
    length = max(int(np.count_nonzero(batch["segments"][window] == row)) for window, row in rows_of(batch))
    questions = batch["kinds"].shape[1]

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


# The joint arm's float64 truth over the seeds its gradients need is numeric work at model size, a GPU's
# (BIBLE §8): on armada it took 705 s of CPU, the suite's whole critical path; the others take 187 s and 36 s.
@pytest.mark.parametrize("arm", [pytest.param("joint", marks=pytest.mark.gpu(
    reason="the joint arm's float64 truth at the seeds it needs")), "marker", "parallel"])
def test_a_packed_window_decides_each_row_as_the_row_alone(arm):
    """Rows packed into windows (segments, positions restarting per row, spans moved) give each
    question's logits and the parameters' gradients each row gives alone where its window packed it,
    through Qwen3.5's Gated DeltaNet conv and state and its attention, and through the head, within
    the float64 gate: no state, attention, pooled span or last-token feature crosses a row. The float64
    truth is the rows in windows of their own in float64, on the reference attention (the template takes
    fp32, bf16 and fp16).

    The reference runs the same kernels at the same shapes as the window (`rows_alone`), so this
    holds what packing does; on a GPU the delta rule runs on its port in both, and its algorithm's
    parity with fla is held at the layer, against fla's own packed sequences (test_fla_gdn,
    test_fla_kda)."""
    objective = objective_for(arm)
    assert_rows_decided_alone(objective, (objective.model.backbone.clone(dtype=jnp.float64,
                                                                         attention_impl="reference"),
                                          objective.model.head.clone(dtype=jnp.float64)), PACKING_SEEDS[arm],
                              PACKING_WINDOWS.get(arm, 2))


def test_a_labelled_readout_reads_each_packed_row_at_its_own_answer_cue():
    """Rows whose options stand under labels, packed into windows, read each question's answer at its
    row's answer cue, moved with the row: each row's label logits and the backbone's gradients are the
    row's alone where its window packed it (check 1), the readout the backbone's own logits, whose
    float64 truth is the float64 backbone's. Each question's answer position is its own row's last
    token, the cue's, in its window."""
    objective = objective_for("readout")
    batch = windows(objective, 2)
    for window, row in rows_of(batch):
        last = np.flatnonzero(batch["segments"][window] == row)[-1]
        assert np.all(batch["readout"][window][owners(batch)[window] == row] == last)
    assert_rows_decided_alone(objective, (objective.model.backbone.clone(dtype=jnp.float64),
                                          objective.model.head), PACKING_SEEDS["readout"])


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
    assert_rows_decided_alone(objective, (objective.model.backbone.clone(dtype=jnp.float64), head),
                              PACKING_SEEDS["flash"])


def asked(batch, logits) -> np.ndarray:
    """A packed batch's logits at each row's questions' options, in the order `apart` lays the rows."""
    order = [owners(batch)[window] == row for window, row in rows_of(batch)]
    return np.concatenate([np.asarray(logits)[window][slots][batch["options"][window][slots]]
                           for (window, _), slots in zip(rows_of(batch), order, strict=True)])


def rows_alone(decide, batch):
    """The logits and gradients of `batch`'s rows, each alone where its window packed it in a batch of
    the packed batch's shapes (`in_place`), by `decide(batch)`, the logits at their questions' options
    and the gradients summed over the rows: the packed run's own arithmetic, its products' shapes, the
    rule's chunks and attention's row length the window's, so the two differ only where another row
    enters, and in how the rows' gradients add. Each call holds one row of every window that has one
    left, the windows' `k`th rows together, as no window reads another."""
    every = rows_of(batch)
    rounds = [[(window, row) for window, row in every if row == index]
              for index in range(1, max(row for _, row in every) + 1)]
    logits, grads = {}, None
    for placed in rounds:
        one = in_place(batch, placed)
        row_logits, row_grads = decide(one)
        for window, row in placed:
            logits[window, row] = np.asarray(row_logits)[window][one["options"][window]]
        grads = row_grads if grads is None else jax.tree.map(jnp.add, grads, row_grads)
    return np.concatenate([logits[place] for place in every]), grads


PACKING_SEEDS = {"joint": 256, "marker": 64, "parallel": 256, "flash": 8, "readout": 8}
"""Check 1's declared seeds for each arm's packed window: the most paired sizing (`seeds_for` with the
packed window as `dew_fn`) gives on any platform the site runs on. An H100 took 256 for the joint arm
and 128 for the parallel arm (node receipts TL-h100-packing-size-*-3), an A100 128 for the joint arm
and 256 for the parallel arm (SF-a100-packing-seeds), armada 16 and 8 (run-tests
sf-packing-cpu-sizing-898ec4ac6). The marker arm's one window a draw (`PACKING_WINDOWS`) took 64 on an
H100, 32 on an A100 (node receipts TL-h100-sizing-1851-679, TL-a100-sizing-1851-679) and 8 on armada
(run-tests rows-memory-probe12-8ce8072b5), each with its confirming block, the largest declared as
docs/design/testing.md's check 1 rules (d28b1c318);
the flash arm's is an A100's, and the readout arm's 8 is an A100's
(SF-readout-sizing), an H100's and armada's. The A100's are the head's gradients whose float64
truth is zero by symmetry, which both arms err at by 0 to 3 fp32 ulps: the marker and parallel arms'
scorer_out bias, shared by a question's options, and the joint arm's attention key biases (ledger
1548). Twice the rows a draw took the same counts on an A100 (node receipts SF-packing-scale2-sizing),
each draw twice the work."""


PACKING_WINDOWS = {"marker": 1}
"""The packed windows a draw holds where an arm's site takes fewer than two: the marker arm's one window
packs several rows, and its draw at two windows took 0.50 s on armada against 0.31 at one, where the
256 draws two windows took on an A100 brought it past the suite's 240 s a test (ledger 1851)."""


def assert_rows_decided_alone(objective, wide_modules, seeds: int, count: int = 2) -> None:
    """`count` packed windows of the examples decide each row's questions as the row does alone where
    its window packed it (`rows_alone`), logits and every parameter's gradient on par with it (check 1)
    over drawn parameters, the truth `wide_modules`, the backbone and head in float64, over the rows
    in windows of their own."""
    batch = windows(objective, count)
    assert max(int(segments.max()) for segments in batch["segments"]) > 1, "no window packed two rows"
    # A sequence kernel runs as many documents as the coordinates bound, and a window's padding as one
    # more sequence (`dew.nn.attention.flash_attention`).
    segments = objective.bounds.segments
    laid = laid_out(batch, segments).coordinates()
    assert int(np.max(laid.segments)) <= laid.max_segments == segments
    rows = apart(batch)
    wide = copy.copy(objective)
    wide.substitute(wide_modules)
    decide = jax.jit(lambda variables, batch: decided(objective, variables, batch, segments))

    @as_given
    def packed(variables):
        logits, grads = decide(variables, batch)
        return asked(batch, logits), grads

    @as_given
    def alone(variables):
        return rows_alone(lambda one: decide(variables, one), batch)

    @jax.jit
    def apart_in_float64(variables):
        return wide.model.logits(variables, laid_out(rows, segments)), gradient(wide, variables, rows)

    def truth(variables):
        logits, grads = apart_in_float64(variables)
        return np.asarray(logits)[rows["options"]], grads

    # A logit sums within its row's window; a parameter's gradient over every window's tokens, in
    # another order than the rows alone.
    initialized = jax.jit(objective.init)
    assert_as_exact(packed, alone, lambda seed: (initialized(jax.random.key(seed)),), "packed window",
                    seeds=seeds, truth_fn=truth)


def test_a_decision_run_steps_one_program_over_rows_of_every_length():
    """A token budget of packed windows: every batch is the one shape the bounds give, so rows from
    a short question to a three-question schema train through one compiled step, and the loss is
    finite on every step."""
    objective = objective_for("joint")
    layout = objective.layout
    data = decision_rows(EXAMPLES, None, objective=objective.data(), tokens=8 * layout.max_len)
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
        decision_rows(EXAMPLES, 2, objective=objective.data())
    single = [example for example in EXAMPLES if len(example.questions) == 1]
    data = decision_rows(single, 2, objective=objective.data())
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
    data = decision_rows([example], 1, [example], objective=objective.data())
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
    data = decision_rows(EXAMPLES, 2, objective=objective.data())
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
    data = decision_rows(examples, 3, examples, objective=objective.data())
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
