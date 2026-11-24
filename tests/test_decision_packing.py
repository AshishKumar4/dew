"""Decision rows packed into windows: each row decides as it does alone, and a run is one program."""

import collections
import contextlib
import copy
import functools
import itertools
from dataclasses import replace
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from arm_arithmetic import ahead_on_the_host
from attention_oracle import traced_attention_terms
from decision_support import decision_rows
from float64_terms import float64_terms
from model_support import attending
from optim_support import in_float64
from recording import RecordingTracker
from reference_error import as_given, assert_as_exact, dew_path
from torch_layout import node

from dew.batch import VALID_ROWS
from dew.data.dataset import DataPartition
from dew.data.text import ByteTokenizer
from dew.decision import Choice, DecisionObjective, Example, LogLoss, Noul, Score, Specials
from dew.decision.head import DecisionHead, JointSchemaHead, LabelReadout
from dew.decision.inputs import OPTION_TILE, DecisionInputs, compacted
from dew.decision.layout import QUESTION_FIELDS, Labels, SegmentLayout
from dew.decision.rows import TRAINING_FIELDS
from dew.decision.task import laid_out
from dew.interop import Pretrained
from dew.nn.attention_blocks import NormalAttention
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.kernels import generation
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
    # Loaded on the host in a GPU tier prepare, whose compile-only GPU runs no load's checks (row 11788).
    with ahead_on_the_host():
        return Pretrained.load(str(QWEN35), dtype="float32", max_seq_len=length)


def attention_decoder(length: int) -> CausalTransformer:
    """A decoder of attention alone, which the parallel layout's key runs need."""
    return CausalTransformer(vocab_size=256, emb_features=32, num_layers=2, num_heads=2, mlp_features=48,
                             max_seq_len=length)


ROWS = SegmentLayout.state_first_row(max_len=256, head_max_len=64, option_tokens=8)
LABELS = Labels(variants=("{}",))
"""Labels a byte vocabulary spells as one token each: the bare letter."""
ARMS = {
    "joint": (SegmentLayout.joint_row(max_len=1536), qwen35,
             lambda: JointSchemaHead(64, width=32, routing_layers=1, layers=1, heads=2, feedforward=64)),
    "marker": (ROWS, qwen35, lambda: DecisionHead(64, layers=1)),
    # The parallel arm's decoder, which reads the layout's key runs as its spans on the attention template
    # (`ARM_KERNELS`; `DecisionModel._read`, row 1835).
    "parallel": (replace(ROWS, parallel=True), attention_decoder, lambda: DecisionHead(32, layers=1)),
    # A row a question, packed beside others (`shared_state=1`): rows over one state are
    # tests/test_decision_state_once.py's.
    "readout": (replace(ROWS, head_max_len=128, option_tokens=24, labels=LABELS, shared_state=1), qwen35,
                lambda: LabelReadout(64, LABELS.ids(TOKENIZER, 3))),
}
"""Joint rows under the joint schema head and rows of one question under the marker head,
both over Qwen3.5; parallel options, whose key runs Gated DeltaNet refuses, over the attention template;
options under labels read at each row's answer cue by the backbone's own logits, over Qwen3.5."""

ARM_KERNELS = {"parallel": "template"}
"""The attention kernel an arm's packed and lone rows run on (`generation.chosen(attention=...)`), the
selector's own where it names none: the parallel arm's the template, which reads its key runs as spans."""


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
    return jax.tree.map(np.asarray, {name: value for name, value in batch.items() if name != VALID_ROWS})


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
    # The row's document is 1 + whether a filler document stands before it, and its questions are its own:
    # each question's row the row's new document, or its queries would read no document's tokens.
    document = 1 + int(before.any())
    if "question_rows" in batch:
        one["question_rows"] = np.full((1, asked.size), document, np.int32)
    return one | {"segments": (before + mine * document).astype(np.int32)[None]}


def in_place(batch, placed: list[tuple[int, int]]) -> dict:
    """`batch` with each `(window, row)` of `placed`, at most one a window, alone in its window where
    the window packed it (`at_offset`), and every other window empty: the packed batch's shapes and its
    bound on a window's documents, so every product a row's tokens enter runs in the packed run's own
    program. A window reads no other, so rows of different windows run alone together."""
    assert len({window for window, _ in placed}) == len(placed), "one row a window"
    # The read's bound on its windows (`tokens_bound`) kept: a window of one row holds no more documents
    # than it bounds, and the packed run's program, its bound static, serves the rows alone too.
    one = {name: value if name == "tokens_bound" else np.zeros_like(value) for name, value in batch.items()}
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


def located(tree, path: tuple[str, ...], held) -> tuple[str, ...]:
    """Where in a parameters tree a module whose scope is `path` holds `held`, its own parameters as its
    apply read them: its model applies it at the tree's root or under one of its top-level modules, a
    backbone's and a head's scopes alike (`layers_0/self_attn` in both), so the one place whose leaves are
    the very arrays `held` holds is its, read as a trace reads them (`AttentionTerms`). A path none or
    several of whose places hold them is refused, naming it."""
    found = []
    for root in ((), *((name,) for name, child in tree.items() if isinstance(child, dict))):
        reached = tree
        for name in (*root, *path):
            reached = reached.get(name) if isinstance(reached, dict) else None
        if reached is not None and jax.tree.structure(reached) == jax.tree.structure(held) and all(
                mine is its
                for mine, its in zip(jax.tree.leaves(reached), jax.tree.leaves(held), strict=True)):
            found.append((*root, *path))
    assert len(found) == 1, f"{path} is at {len(found)} places of the parameters"
    return found[0]


READ = ("_heads", "__call__")
"""The methods of an attention `AttentionTerms` reads: each projection to heads, and the call."""


class AttentionTerms:
    """Each attention's projections' gradients' float64 terms in `objective`'s float64 model at `batch`
    (`NormalAttention`): its queries', keys' and values' terms (`traced_attention_terms`, a logit's gradient
    P (dP - delta) sized by its terms' rounding), carried through the projection that made them to its
    bias, their root sum of squares over the batch's tokens, and to its kernel, the tokens' input times
    them. A key's bias moves every logit of its query alike, so its gradient cancels to zero in float64
    (softmax's shift invariance); its rounding is of those terms, behind the sums that cancel, where
    `float64_terms` sizes a leaf by its final sum's terms alone. The attentions' inputs and outputs are
    read through Flax's method interception, the outputs' cotangents as the gradient of a zero added to
    each, in one program traced once. Each attention's place in the parameters is read as it is traced,
    the one place whose leaves are the very arrays its module reads (`located`), so the
    terms are a program of the variables a prepare traces and compiles ahead (row 7359)."""

    def __init__(self, objective, batch):
        self.names: dict = {}
        self.places: dict = {}
        with ahead_on_the_host():
            # The step's constants on the host in a GPU tier prepare (row 11788).
            step = Step(jnp.asarray(0), jax.random.key(0), None, training=False)

        def total(variables, added):
            held: dict = {}

            def read(call, args, kwargs, context):
                module = context.module
                if not isinstance(module, NormalAttention) or context.method_name not in READ:
                    return call(*args, **kwargs)
                path = module.scope.path
                self.names[path] = module.names
                if path not in self.places:
                    self.places[path] = located(variables["params"], path, module.variables["params"])
                record = held.setdefault(path, {})
                out = call(*args, **kwargs)
                if context.method_name == "_heads":
                    record[args[0]] = (args[1], out)
                    return out
                record["documents"] = (kwargs["segment_ids"], kwargs.get("key_segment_ids"))
                record["out"] = out
                return out + added[path] if path in added else out

            with nn.intercept_methods(read):
                loss, _ = objective.loss(variables, batch, step)
            return loss.total, held

        self.total = total
        self.read = jax.jit(jax.grad(total, argnums=1, has_aux=True))
        self.outputs: dict | None = None

    def __call__(self, variables) -> dict:
        """The terms carried to each attention's projections, by its place in the parameters (`located`):
        each projection's input and its heads' terms."""
        if self.outputs is None:
            # Every draw's variables are of one tree, so its attentions' outputs of one shape: their shapes
            # kept, the zeros made in each call, as one made in a trace is that trace's own.
            _, shapes = jax.eval_shape(self.total, variables, {})
            self.outputs = {path: record["out"] for path, record in shapes.items()}
        added = {path: jnp.zeros(shape.shape, shape.dtype) for path, shape in self.outputs.items()}
        cotangents, held = self.read(variables, added)
        carried = {}
        for path, record in held.items():
            query_name, key_name, value_name, out_name = self.names[path]
            (into_queries, queries), (into_keys, keys), (_, values) = (
                record[name] for name in (query_name, key_name, value_name))
            query_ids, key_ids = record["documents"]
            key_ids = query_ids if key_ids is None else key_ids
            queried = query_ids[:, None, :, None]
            keep = (queried == key_ids[:, None, None, :]) & (queried != 0)
            place = self.places[path]
            kernel = node(variables["params"], place)[out_name]["kernel"]
            heads, width = queries.shape[2:]
            dout = jnp.einsum("bsf,hdf->bshd", cotangents[path], kernel.reshape(heads, width, -1))
            _, query_terms, key_terms, value_terms = traced_attention_terms(queries, keys, values, dout, keep)
            # `into` stays on the device: `with_attention_terms` widens it there, at float64 under x64.
            carried[place] = {name: (into, terms) for name, into, terms in (
                (query_name, into_queries, query_terms), (key_name, into_keys, key_terms),
                (value_name, into_keys, value_terms))}
        return carried


def with_attention_terms(sized, carried: dict):
    """`sized`, a gradient's float64 terms (`float64_terms`), each attention projection's bias and kernel
    sized by at least the attention's terms carried to it (`AttentionTerms`), in float64 where JAX places
    them: on a GPU, its float64 units, not the host's (bible §8)."""
    with jax.enable_x64(new_val=True):
        sized = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), sized)
        for place, projections in carried.items():
            placed = node(sized["params"], place)
            for name, (into, terms) in projections.items():
                leaf = placed[name]
                squares = jnp.square(jnp.asarray(terms, jnp.float64))
                leaf["bias"] = jnp.maximum(leaf["bias"],
                                           jnp.sqrt(squares.sum((0, 1))).reshape(leaf["bias"].shape))
                leaf["kernel"] = jnp.maximum(leaf["kernel"], jnp.sqrt(jnp.einsum(
                    "bse,bshd->ehd", jnp.square(jnp.asarray(into, jnp.float64)), squares)).reshape(
                        leaf["kernel"].shape))
    return sized


def decided(objective, variables, batch, segments: int):
    """The batch's logits and the gradient of its summed loss."""
    return objective.model.logits(variables, laid_out(batch, segments)), gradient(objective, variables, batch)


# The joint arm's float64 truth over the seeds its gradients need is numeric work at model size, a GPU's
# (BIBLE §8): on armada it took 705 s of CPU, the suite's whole critical path; the others take 187 s and 36 s.
# The joint arm runs in the models' GPU flow (tests/test_gpu_models.py).
@pytest.mark.check1_shards(seeds=8, per_task=8)
def test_a_packed_window_of_marked_rows_decides_each_row_as_the_row_alone():
    """A packed window decides each row as the row alone (`packed_rows_decided_alone`) through the
    marker arm, its seeds drawn by CI's map tasks (`check1_shards`)."""
    packed_rows_decided_alone("marker")


@pytest.mark.check1_shards(seeds=64, per_task=16)
def test_a_packed_window_of_parallel_rows_decides_each_row_as_the_row_alone():
    """A packed window decides each row as the row alone (`packed_rows_decided_alone`) through the
    parallel arm, its seeds drawn by CI's map tasks (`check1_shards`): `PACKING_SEEDS`' 64, 16 a task."""
    packed_rows_decided_alone("parallel")


def test_a_packed_windows_site_prepares_its_terms_whole_and_runs_from_its_store(tmp_path):
    """The marker arm's packed window (`packed_rows_decided_alone`) prepared into a store as a GPU tier
    part's prepare compiles it (`truths_ahead.compiled_truths`): its site's terms program, the truth's
    float64 terms beside each attention's (`AttentionTerms`), traces whole and compiles, and its run
    computes its truths and terms from those programs, nothing deferred or unprepared (row 7359; the review
    of train 1058: its truth's float64 terms converted to the host, and its prepare raised)."""
    from truths_ahead import DEFERRED, UNPREPARED, compiled_truths

    for prepare in (True, False):
        with compiled_truths(tmp_path, prepare=prepare, stamp="a"):
            packed_rows_decided_alone("marker")
    assert not (tmp_path / UNPREPARED).exists() and not (tmp_path / DEFERRED).exists()


def packed_rows_decided_alone(arm: str) -> None:
    """Rows packed into windows (segments, positions restarting per row, spans moved) give each
    question's logits and the parameters' gradients each row gives alone where its window packed it,
    through Qwen3.5's Gated DeltaNet conv and state and its attention, and through the head, within
    the float64 gate: no state, attention, pooled span or last-token feature crosses a row. The float64
    truth is the rows in windows of their own in float64, on the reference attention (the template takes
    fp32, bf16 and fp16).

    The reference runs the same kernels at the same shapes as the window (`rows_alone`), so this
    holds what packing does; on a GPU the delta rule runs on its port in both, and its algorithm is held
    at the layer: against the token recurrence in float64 on packed documents (test_fla_gdn), and trained
    natively on a GPU (tests/test_gpu_recurrence.py's port cells)."""
    objective = objective_for(arm)
    assert_rows_decided_alone(objective, (objective.model.backbone.clone(dtype=jnp.float64),
                                          objective.model.head.clone(dtype=jnp.float64)), PACKING_SEEDS[arm],
                              arm, PACKING_WINDOWS.get(arm, 2), kernel=ARM_KERNELS.get(arm))


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
                                          objective.model.head), PACKING_SEEDS["readout"], "readout")


def a_bf16_joint_head_on_flash_reads_each_packed_row_alone(without_deterministic_ops):
    """The joint schema head in bf16 on flash: its routing and field layers read the window's tokens
    by documents, flash's packed sequences of the keys' own offsets, its padded questions and slots
    reading nothing; its decoder in bf16 on flash too, as a run names one kernel for every attention it
    traces (`generation.chosen(attention='flash')`, row 11938). Each packed row's logits and the
    gradients are the row's alone within the float64 gate, the truth the model in float64 on the
    reference kernel. A head that read the window through a dense mask would not run on flash at all."""
    layout = SegmentLayout.joint_row(max_len=1536)
    head = JointSchemaHead(32, width=128, routing_layers=1, layers=1, heads=2, feedforward=64,
                           dtype=jnp.bfloat16)
    objective = DecisionObjective(
        attention_decoder(layout.max_len).clone(dtype=jnp.bfloat16), tokenizer=TOKENIZER, specials=SPECIALS,
        layout=layout, head=head, loss=LogLoss(), shuffle_options=False)
    assert_rows_decided_alone(objective, (objective.model.backbone.clone(dtype=jnp.float64),
                                          objective.model.head.clone(dtype=jnp.float64)),
                              PACKING_SEEDS["flash"], "flash", kernel="flash")


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


PACKING_SEEDS = {"joint": 256, "marker": 8, "parallel": 64, "flash": 8, "readout": 8}
"""Check 1's declared seeds for each arm's packed window: the most paired sizing (`seeds_for` with the
packed window as `dew_fn`) gives on any platform the site runs on, each with its confirming block, the
largest declared as docs/design/testing.md's check 1 rules (d28b1c318). The joint arm and the flash head
run on GPUs alone, in the models' GPU flow (tests/test_gpu_models.py): an H100 took 256 for the joint arm
(node receipts TL-h100-packing-size-*-3) and an A100 128 (SF-a100-packing-seeds); the flash arm's is an
A100's. The marker and parallel arms run on armada's CPU alone (bible §7, no GPU unit tests), where
both took 8 (run-tests sf-pack-sizing at 519299480, as sf-packing-cpu-sizing-898ec4ac6 and
rows-memory-probe12-8ce8072b5 before it; on GPUs, where they no longer run, they took 64 to 256). The
readout arm's 8 is an A100's (SF-readout-sizing), an H100's and armada's. The GPUs' larger counts are
the head's gradients whose float64 truth is zero by symmetry, which both arms err at by 0 to 3 fp32
ulps: the marker and parallel arms' scorer_out bias, shared by a question's options, and the joint arm's
attention key biases (ledger 1548). Twice the rows a draw took the same counts on an A100 (node receipts
SF-packing-scale2-sizing), each draw twice the work. The parallel arm's 64 since its head runs on the
template too (row 11938: one kernel a run names, `ARM_KERNELS`), whose 64-row tiles round its scorer's
inputs otherwise: 8 read its scorer_out bias Inconclusive (pooled 1.105, [0.760, 1.694] over 40), and 64,
128 and 256 each passed on armada (run-tests IG-PAR-* at API02c), the plain pair unsizable where the site
judges by its terms (`seeds_for`'s "not the plain pair's")."""


PACKING_WINDOWS = {"marker": 1}
"""The packed windows a draw holds where an arm's site takes fewer than two: the marker arm's one window
packs several rows, and its draw at two windows took 0.50 s on armada against 0.31 at one, where the
256 draws two windows took on an A100 brought it past the suite's 240 s a test (ledger 1851)."""


def assert_rows_decided_alone(objective, wide_modules, seeds: int, arm: str, count: int = 2, *,
                              kernel: str | None = None) -> None:
    """`count` packed windows of the examples decide each row's questions as the row does alone where
    its window packed it (`rows_alone`), logits and every parameter's gradient on par with it (check 1)
    over drawn parameters, the truth `wide_modules`, the backbone and head in float64, over the rows
    in windows of their own, a leaf zero by algebra sized by the terms its sum adds; its site `arm`'s own
    (`packed window: <arm>`), as two arms' cells share a GPU flow's one store of programs. The packed and
    lone rows run their attention on `kernel` where given (`generation.chosen`), the selector's pick
    where not; the float64 truth, its parameters' draw among them, on the selector's, which takes the
    reference path for float64 (`dew.nn.attention`'s narrowing rule) and runs on the host's CPU."""
    # The windows laid out on the host in a GPU tier prepare (`ahead_on_the_host`, rows 7858, 7859: made on
    # its compile-only GPU, they left the cell unprepared).
    with ahead_on_the_host():
        batch = windows(objective, count)
        assert max(int(segments.max()) for segments in batch["segments"]) > 1, "no window packed two rows"
        # A sequence kernel runs as many documents as the coordinates bound, and a window's padding as one
        # more sequence (`dew.nn.attention.flash_attention`).
        segments = objective.bounds.segments
        laid = laid_out(batch, segments).coordinates()
        declared = batch["tokens_bound"].max_segments
        assert int(np.max(laid.segments)) <= laid.max_segments == declared <= segments
        rows = apart(batch)
    wide = copy.copy(objective)
    wide.substitute(wide_modules)
    decide = jax.jit(lambda variables, batch: decided(objective, variables, batch, segments))

    chosen = (contextlib.nullcontext if kernel is None
              else functools.partial(generation.chosen, attention=kernel))

    @as_given
    def packed(variables):
        with chosen():
            logits, grads = decide(variables, batch)
        return asked(batch, logits), grads

    # One of Dew's own paths: packing invariance, both arms one `decide` under one rewrite, one program.
    @dew_path
    @as_given
    def alone(variables):
        with chosen():
            return rows_alone(lambda one: decide(variables, one), batch)

    # The rows' float fields (the targets a smoothed label reads) in float64, as the truth computes.
    wide_rows = jax.tree.map(lambda leaf: np.asarray(leaf, np.float64)
                             if np.issubdtype(np.asarray(leaf).dtype, np.floating) else leaf, rows)

    @jax.jit
    def apart_in_float64(variables):
        return (wide.model.logits(variables, laid_out(wide_rows, segments)),
                gradient(wide, variables, wide_rows))

    @in_float64
    def truth(variables):
        # A leaf zero by algebra (a key's bias, the scorer's shared bias under the loss) cancels in its
        # sum, which the check sizes by the terms it adds (`terms`). Under a float64 run's numerics
        # (`in_float64`), whose logits and losses are float64 where the run's output dtype is float32.
        logits, grads = apart_in_float64(variables)
        return jnp.asarray(logits)[rows["options"]], grads

    carried = AttentionTerms(wide, rows)

    def terms(variables):
        # Each leaf's terms as its sums add them (`float64_terms`), an attention's projections' at least
        # the attention's own carried to them, where a key's bias cancels behind its final sum.
        _, (logits, grads) = float64_terms(truth, variables)
        return logits, with_attention_terms(grads, carried(variables))

    # A logit sums within its row's window; a parameter's gradient over every window's tokens, in
    # another order than the rows alone. Drawn unboxed: the terms read an attention's kernels as arrays
    # (`AttentionTerms`), which an init can box with their logical axes (row 4465).
    # Drawn on the selector's kernels, which the host's CPU runs (row 11788: a flash head's init refused
    # there), `kernel` the rows' alone.
    initialized = jax.jit(lambda key: nn.unbox(objective.init(key)))
    # A leaf whose float64 truth cancels, an algebraic zero's, is sized by the terms its sums add
    # (`terms`), so each arm's rounding of them is on par, on every backend.
    assert_as_exact(packed, alone, lambda seed: (initialized(jax.random.key(seed)),),
                    f"packed window: {arm}", seeds=seeds, truth_fn=truth, terms=terms)


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
    # A row a question (`shared_state=1`): a row over one state's documents are its forks'.
    alone = replace(ROWS, shared_state=1)
    rows = [row for example in EXAMPLES for row in alone.rows(TOKENIZER, SPECIALS, example.state,
                                                              example.questions)]
    laid = DecisionInputs.collate(rows, SPECIALS.pad).coordinates()
    starts = np.asarray(laid.starts)
    assert starts[:, 1:].any(), "no request was padded"
    assert 1 + int(starts[:, 1:].sum(axis=1).max()) <= laid.max_segments + 1
    assert int(np.max(laid.segments)) == laid.max_segments == 1


@attending("reference")
def test_a_one_question_head_reads_joint_rows_only_of_one_question():
    """The marker head reads one question a row: over joint rows of a schema that asks several, the run
    is refused before a row is read; a schema of one question reads as rows of one."""
    objective = DecisionObjective(qwen35(1536), tokenizer=TOKENIZER, specials=SPECIALS,
                                  layout=SegmentLayout.joint_row(max_len=1536), head=DecisionHead(64,
                                                                                                  layers=1),
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


@attending("xla")
@pytest.mark.parametrize("causal", [True, False])
def test_a_joint_head_given_no_layout_reads_every_question_of_a_request_in_one_row(causal):
    """A decision objective given the joint head and no layout lays a request out
    as that head reads it, every question in one row (`SegmentLayout.joint_row`), over a
    causal and a bidirectional backbone alike, as a run's config does: it
    scores a three-question request as the same objective given the
    the joint row does."""
    example = EXAMPLES[4]
    backbone = CausalTransformer(vocab_size=256, emb_features=32, num_layers=2, num_heads=2, mlp_features=48,
                                 max_seq_len=2048, causal=causal)
    head = JointSchemaHead(32, width=32, routing_layers=1, layers=1, heads=2, feedforward=64)

    def objective(**layout):
        return DecisionObjective(backbone, tokenizer=TOKENIZER, specials=SPECIALS, head=head,
                                 shuffle_options=False, **layout)

    stated, defaulted = objective(layout=SegmentLayout.joint_row()), objective()
    # Drawn over a window the backbone holds; the weights are the same whatever the layout's length.
    variables = objective(layout=SegmentLayout.joint_row(max_len=2048)).init(jax.random.key(0))

    def scores(objective):
        rows = objective.layout.rows(TOKENIZER, SPECIALS, example.state, example.questions)
        return np.asarray(objective.model.logits(variables, DecisionInputs.collate(rows, SPECIALS.pad)))

    expected = scores(stated)
    assert expected.shape[:2] == (1, len(example.questions))
    np.testing.assert_array_equal(scores(defaulted), expected)


@attending("xla")
@pytest.mark.parametrize("questions", [1, 2])
def test_single_option_questions_fit_the_packing_bounds(questions):
    """A fitting singleton-choice schema packs even where a two-option row cannot fit."""
    example = Example("", {f"q{index}": Choice("x", ["0"]) for index in range(questions)},
                      {f"q{index}": "0" for index in range(questions)})
    (encoded,) = SegmentLayout.joint_row().rows(TOKENIZER, SPECIALS, example.state, example.questions)
    length = len(encoded.tokens) if questions == 1 else 500
    layout = SegmentLayout.joint_row(max_len=length)
    assert layout.fits(encoded)
    head = JointSchemaHead(32, width=32, routing_layers=1, layers=1, heads=2, feedforward=64)
    objective = DecisionObjective(attention_decoder(length), tokenizer=TOKENIZER, specials=SPECIALS,
                                  layout=layout, head=head, shuffle_options=False)
    data = decision_rows([example], 1, [example], objective=objective.data())
    (batch,) = list(data.val(DataPartition()))
    batch = jax.tree.map(np.asarray, dict(batch))
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


@attending("xla")
@pytest.mark.parametrize("arm", ["joint", "marker"])
def test_a_reader_tokenizes_each_text_of_its_rows_once(arm):
    """A reader lays each record out twice, to fit it and to train on it, and
    every record asks its questions over options in a fresh order, yet each
    text is tokenized once: each template, instruction and option once for
    the run, and each state as its record is fit, which the row it trains on
    lays out again."""
    layout, _, head = ARMS[arm]
    tokenizer = Counting()
    objective = DecisionObjective(attention_decoder(layout.max_len), tokenizer=tokenizer, specials=SPECIALS,
                                  layout=layout, head=head(), loss=LogLoss())
    data = decision_rows(EXAMPLES, 2, objective=objective.data())
    reader = data.train(DataPartition())
    try:
        assert len(list(itertools.islice(reader, 4))) == 4
    finally:
        reader.close()
    assert tokenizer.texts and set(tokenizer.texts.values()) == {1}


PACKED_SEEDS = 256
"""Check 1's declared seeds for the packed charges against each example's alone, a draw a model's
variables, sized paired by `seeds_for` on armada."""


@attending("xla")
@pytest.mark.check1_shards(seeds=256, per_task=64)
def test_a_packed_loss_counts_only_real_scored_questions():
    """Different joint-row lengths and label counts contribute question charges, not token,
    window, unanswered or padding counts; a repeated held window contributes neither part. Its 256
    seeds, each two arms' runs of the objective, are drawn by CI's map tasks, 64 a task
    (`check1_shards`, row 6449)."""
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
    layout = SegmentLayout.joint_row(max_len=1536)
    objective = DecisionObjective(attention_decoder(layout.max_len), tokenizer=TOKENIZER, specials=SPECIALS,
                                  layout=layout, head=JointSchemaHead(32, width=32, routing_layers=1,
                                      layers=1, heads=2, feedforward=64), shuffle_options=False)
    data = decision_rows(examples, 3, examples, objective=objective.data())
    (batch,) = list(data.val(DataPartition()))
    batch = jax.tree.map(np.asarray, dict(batch))
    real = batch[VALID_ROWS]
    assert real.sum() == 2 and (~real).sum() == 1
    assert int(batch["segments"][real].max()) > 1
    assert (~batch["questions"]).any()
    assert int((batch["questions"] & ~batch["scored"])[real].sum()) == 4
    (repeated,) = np.flatnonzero(~real)
    assert any(np.array_equal(batch["tokens"][repeated], tokens) for tokens in batch["tokens"][real])
    variables = objective.init(jax.random.key(0))
    # Each example alone, its own row: the inputs of its row and, for each labelled question, its slot,
    # options and gold over them as the row orders them.
    alone = []
    for example in examples:
        (encoded,) = layout.rows(TOKENIZER, SPECIALS, example.state, example.questions)
        alone.append((DecisionInputs.collate([encoded], SPECIALS.pad), [
            (slot, len(laid.options), example.distribution(laid.name)[list(laid.order)])
            for slot, laid in enumerate(encoded.questions) if laid.name in example.labels()]))
    assert len({inputs.tokens.shape[-1] for inputs, _ in alone}) == len(examples)
    count = sum(len(example.labels()) for example in examples)
    assert sum(len(labelled) for _, labelled in alone) == count == 4
    step = Step(jnp.asarray(0), jax.random.key(0), None)
    loss, _ = objective.loss(variables, batch, step)
    assert float(loss.mass) == count
    assert bool(objective.reduce_loss(loss)[1])

    def packed(variables):
        loss, _ = objective.loss(variables, batch, step)
        return {"total": loss.total, "mean": objective.reduce_loss(loss)[0]}

    def each_alone(variables):
        charges = []
        for inputs, labelled in alone:
            logits = objective.model.logits(variables, inputs)[0]
            for slot, options, gold in labelled:
                log_probabilities = jax.nn.log_softmax(logits[slot, :options])
                charges.append(-jnp.dot(jnp.asarray(gold, logits.dtype), log_probabilities))
        total = jnp.sum(jnp.stack(charges))
        return {"total": total, "mean": total / count}

    init = jax.jit(objective.init)
    assert_as_exact(packed, each_alone, lambda seed: (nn.unbox(init(jax.random.key(seed))),),
                    "a packed batch's question charges against each example's alone", seeds=PACKED_SEEDS)
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


def test_a_packed_decision_run_main_saved_resumes_exactly():
    """A packed decision read main saved at ecb9577e7, one budget of windows in, resumes exactly and reads
    the windows a read of the same chain reads after its first: who lays a window out is the decision's own
    window step's (`dew.data.pipeline.LaysOutWindows`), which changes no recorded plan. A stored field on
    the pack set off its default changed every packed decision run's plan, an exact resume refused and a
    relaxed one starting its data again (train 921's review, F2)."""
    import dataclasses
    from pathlib import Path

    objective = objective_for("joint")
    data = decision_rows(EXAMPLES * 4, None, objective=objective.data(), tokens=2 * objective.layout.max_len)
    saved = (Path(__file__).with_name("fixtures") / "positions" / "main-ecb9577e7"
             / "decision_packed.position").read_bytes()
    reader = data.train(DataPartition())
    try:
        next(reader)
        expected = next(reader)
    finally:
        reader.close()
    resumed = dataclasses.replace(data, resume="exact").train(DataPartition())
    try:
        resumed.set_state(saved)
        got = next(resumed)
    finally:
        resumed.close()
    assert set(got) == set(expected)
    for name in expected:
        np.testing.assert_array_equal(np.asarray(got[name]), np.asarray(expected[name]))
