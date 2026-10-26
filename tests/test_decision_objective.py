"""`DecisionObjective`: proper scoring rules, metrics, row encoding, and training through LoRA."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from decision_support import decision_rows

from dew.artifacts import Decisions
from dew.checkpoints import Checkpoints
from dew.data.dataset import DataPartition
from dew.data.text import ByteTokenizer
from dew.decision import (
    ECE,
    Accuracy,
    Brier,
    Choice,
    Decide,
    DecisionObjective,
    Encoding,
    Example,
    LogLoss,
    Noul,
    RankedProbability,
    Specials,
    Spherical,
    StateFirstLayout,
    Weights,
)
from dew.lora import LoRA
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import FROZEN, VALID_ROWS, joined
from dew.training.trainer import Trainer

FIXTURES = Path(__file__).parent / "fixtures" / "laya"
RULES = [LogLoss(), Brier(), Spherical(), RankedProbability()]


def expected_charge(rule, logits, truth):
    """The rule's charge on `logits` averaged over outcomes drawn from
    `truth`, the options read as ordered levels."""
    count = len(truth)
    options = jnp.ones((count, count), bool)
    charges = rule.charge(jnp.tile(logits, (count, 1)), jnp.eye(count), options, jnp.ones((count,), bool))
    return float(jnp.dot(truth, charges))


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: rule.name)
def test_each_rule_is_proper(rule):
    """Charged in expectation over outcomes from q, every forecast does worse than q itself."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        truth = rng.dirichlet(np.ones(4))
        honest = expected_charge(rule, jnp.log(truth), truth)
        for _ in range(10):
            other = jnp.asarray(rng.normal(size=4))
            assert honest <= expected_charge(rule, other, truth) + 1e-6


def decisions(tops, correct):
    """Two-option questions, one per row and a second unanswered one beside it,
    whose top probability is `tops` and whose label is right where `correct`."""
    tops = np.asarray(tops)
    probabilities = np.stack([np.stack([tops, 1 - tops], axis=1), np.full((len(tops), 2), 0.5)], axis=1)
    labels = np.stack([np.where(correct, 0, 1), np.ones(len(tops), int)], axis=1)
    scored = np.stack([np.ones(len(tops), bool), np.zeros(len(tops), bool)], axis=1)
    return Decisions(probabilities=probabilities, options=np.ones((len(tops), 2, 2), bool),
                     labels=labels, targets=np.eye(2)[labels], ordinal=np.zeros((len(tops), 2), bool),
                     scored=scored)


def test_ece_and_accuracy_merge_across_batches_as_one_pass():
    """Laya's `ece_score` over 15 bins, closed on the right, with 0 in the first."""
    rng = np.random.default_rng(1)
    tops = rng.uniform(0.5, 1.0, 200)
    correct = rng.uniform(size=200) < tops
    tops[:3] = [0.6, 2 / 3, 1.0]
    metric = ECE()
    whole = metric.finalize(metric(decisions(tops, correct), {}))
    merged = metric.finalize(metric.merge(metric(decisions(tops[:70], correct[:70]), {}),
                                          metric(decisions(tops[70:], correct[70:]), {})))
    edges = np.linspace(0, 1, 16)
    laya = sum(np.mean(inside) * abs(tops[inside].mean() - correct[inside].mean())
               for lo, hi, first in zip(edges[:-1], edges[1:], [True] + [False] * 14, strict=True)
               if (inside := ((tops >= lo) if first else (tops > lo)) & (tops <= hi)).any())
    assert whole == pytest.approx(laya, abs=1e-12) and merged == pytest.approx(laya, abs=1e-12)
    accuracy = Accuracy()
    assert accuracy.finalize(accuracy(decisions(tops, correct), {})) == pytest.approx(correct.mean())


TOKENIZER = ByteTokenizer()
SPECIALS = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)
LAYOUT = StateFirstLayout(max_len=96, head_max_len=48, option_tokens=8)
INTENT = Choice("Which team?", {"billing": "payments", "technical": "outages", "sales": "pricing"})
EXAMPLES = [Example(state, {"team": INTENT}, {"team": label})
            for state, label in (("charged twice", "billing"), ("site is down", "technical"),
                                 ("how much is pro", "sales"), ("refund please", "billing"),
                                 ("error 500", "technical"), ("upgrade cost", "sales"),
                                 ("double charge", "billing"), ("cannot log in", "technical"))]


def test_the_loss_scores_a_soft_target_as_its_rule_does():
    """The log loss of a row whose gold is a distribution is the cross-entropy
    against that distribution, not against its most likely option."""
    from dew.decision.task import laid_out
    from dew.objectives.base import Step

    soft = [Example(f"charged twice {index}", {"team": INTENT}, targets={"team": (0.6, 0.3, 0.1)})
            for index in range(8)]
    objective = DecisionObjective(tiny_backbone(), tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT,
                                  shuffle_options=False)
    data = decision_rows(soft, 8, soft, objective=objective.data())
    (batch,) = list(data.val(DataPartition()))
    batch = {name: value for name, value in batch.items() if name != VALID_ROWS}
    variables = objective.init(jax.random.key(0))
    loss, _ = objective.scalar_loss(variables, batch, Step(jnp.asarray(0), jax.random.key(0), None))
    logits = np.asarray(objective.model.logits(variables, laid_out(batch, objective.bounds.segments)))
    logits = logits[np.asarray(batch["questions"])][:, :3]
    expected = -np.mean(np.sum(np.array([0.6, 0.3, 0.1]) * jax.nn.log_softmax(logits), axis=-1))
    assert float(loss) == pytest.approx(float(expected), rel=1e-5)




@pytest.mark.parametrize("rule", [LogLoss(), Brier()], ids=lambda rule: rule.name)
def test_a_held_pass_scores_a_soft_target_as_its_training_charges_it(rule):
    """Held-out rows whose gold is a distribution are scored against that
    distribution: a scoring rule's held-out metric equals the objective's own
    validation loss under the same rule, not a charge against the most likely
    option."""
    from dew.training.evaluation import Evaluation

    soft = [Example(f"charged twice {index}", {"team": INTENT}, targets={"team": (0.6, 0.3, 0.1)})
            for index in range(8)]
    objective = DecisionObjective(tiny_backbone(), tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT,
                                  loss=rule, shuffle_options=False)
    data = decision_rows(soft, 8, soft, objective=objective.data())
    result = Evaluation.run(objective, objective.init(jax.random.key(0)), data.val, key=0, metrics=(rule,),
                            loss=True)
    assert result.scores[f"val/{rule.name}"] == pytest.approx(result.scores["val/loss"], rel=1e-5)


HELD = [Example(f"{index}" if index % 3 == 0 else f"{index} " + "x" * (12 + index),
                {"team": INTENT, "urgent": Noul("Is it urgent?")},
                {"team": ("billing", "technical", "sales")[index // 3 % 3]}
                | ({"urgent": "true" if index // 4 % 2 else "false"} if index % 4 == 0 else {}))
        for index in range(18)]
"""Distinct states and question rows, some requests answering both questions and others just one."""


def test_a_held_pass_packs_its_rows_and_scores_every_row_once():
    """Held question rows share windows, four windows a batch: their exact identities
    appear once, repeats count nothing, each question decides as it does alone,
    and the pass's log loss is the mean of its questions'."""
    from test_decision_packing import alone, owners, rows_of

    from dew.objectives.base import Step
    from dew.training.distributed import MeshSpec
    from dew.training.evaluation import Evaluation

    layout = replace(LAYOUT, max_len=192)
    objective = DecisionObjective(tiny_backbone().clone(max_seq_len=layout.max_len), tokenizer=TOKENIZER,
                                  specials=SPECIALS, layout=layout,
                                  shuffle_options=False)
    variables = objective.init(jax.random.key(0))
    step = Step(jnp.asarray(0), jax.random.key(0), None)
    data = decision_rows(HELD, 4, HELD, objective=objective.data())
    expected = {}
    for index, example in enumerate(HELD):
        asked = {name: example.questions[name] for name in example.labels()}
        for encoded in objective.layout.rows(TOKENIZER, SPECIALS, example.state, asked):
            (laid,) = encoded.questions
            expected[encoded.tokens] = (index, laid)
    seen, windows_seen = {}, set()
    shared, repeats = False, 0
    for batch in data.val(DataPartition()):
        batch = {name: np.asarray(value) for name, value in batch.items()}
        found = np.asarray(objective.evaluate(variables, batch, step).probabilities)
        # A batch that repeats windows to fill itself marks its real ones.
        real = batch.get(VALID_ROWS, np.ones(len(batch["tokens"]), bool))
        shared |= bool((batch["segments"][real].max(axis=1) > 1).any())
        repeats += int((~real).sum())
        windows_seen.update(tuple(tokens) for tokens in batch["tokens"][real])
        assert all(tuple(tokens) in windows_seen for tokens in batch["tokens"][~real])
        for window, row in rows_of(batch):
            if not real[window]:
                continue
            one = alone(batch, window, row)
            (slot,) = np.flatnonzero(owners(batch)[window] == row)
            key = tuple(one["tokens"][0].tolist())
            index, laid = expected[key]
            identity = index, laid.name
            assert identity not in seen
            width = len(laid.options)
            seen[identity] = found[window, slot, :width], int(batch["labels"][window, slot])
            np.testing.assert_array_equal(one["spans"][0, 0], laid.span)
            gold = HELD[index].distribution(laid.name)[list(laid.order)]
            np.testing.assert_array_equal(one["targets"][0, 0, :width], gold)
            # Within float32 rounding: alone, a row's attention sums over none of its window's keys.
            own = objective.evaluate(variables, one, step).probabilities[0, 0, :width]
            np.testing.assert_allclose(seen[identity][0], own, rtol=1e-5)
    assert shared and repeats > 0
    assert set(seen) == {(index, laid.name) for index, laid in expected.values()}
    mesh = MeshSpec().build(jax.local_devices()[:1])
    result = Evaluation.run(objective, variables, data.val, key=0, metrics=[LogLoss()], loss=True, mesh=mesh)
    expected = -np.mean([np.log(probabilities[label]) for probabilities, label in seen.values()])
    assert result.scores["val/log_loss"] == pytest.approx(expected, rel=1e-5)
    assert result.scores["val/loss"] == pytest.approx(expected, rel=1e-5)


def tiny_backbone() -> CausalTransformer:
    return CausalTransformer(vocab_size=256, emb_features=32, num_layers=2, num_heads=2, mlp_features=48,
                             max_seq_len=128, attention_impl="xla")


def leaves(tree, *path):
    for name in path:
        tree = tree[name]
    return jax.tree.leaves(tree)


def test_lora_trains_its_factors_and_the_head_over_a_fixed_base_and_reloads_from_the_run(tmp_path):
    """The backbone runs under its adapter, applied as its own module: the
    factors and the head move, every base weight stays bitwise where it
    was, and the saved run loads as a Decide task with identical
    probabilities."""
    model = tiny_backbone()
    base = model.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32))
    adapter = LoRA(rank=4, modules=("q_proj", "v_proj")).apply(model, base, key=1)
    objective = DecisionObjective(adapter, tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT,
                                  loss=LogLoss() + 0.5 * Brier())
    data = decision_rows(EXAMPLES, 8, EXAMPLES, objective=objective.data())
    checkpoints = Checkpoints(str(tmp_path), keep=1)
    start = objective.init(jax.random.key(2))
    state = Trainer(objective, optax.adam(1e-2), key=0, checkpoints=checkpoints).fit(
        data, steps=3, log_every=100, checkpoint_every=3, eval_every=3, metrics=[Accuracy(), ECE()])
    checkpoints.wait()

    trained = state.variables
    frozen = zip(leaves(start, FROZEN, "backbone"), leaves(trained, FROZEN, "backbone"), strict=True)
    for before, after in frozen:
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    factors = [leaf for path, leaf in jax.tree_util.tree_leaves_with_path(trained["params"]["backbone"])
               if "lora_B" in jax.tree_util.keystr(path)]
    assert factors and all(np.abs(np.asarray(leaf)).max() > 0 for leaf in factors)
    moved = [not np.array_equal(np.asarray(a), np.asarray(b))
             for a, b in zip(leaves(start, "params", "head"), leaves(trained, "params", "head"), strict=True)]
    assert all(moved)

    live = objective.pipeline(state)
    reloaded = Decide.from_run(str(tmp_path))
    asked = {"team": INTENT, "urgent": Noul("Is it urgent?")}
    first, second = live("charged twice!", asked), reloaded("charged twice!", asked)
    for answer, again in zip(first.values(), second.values(), strict=True):
        np.testing.assert_array_equal(answer.probabilities, again.probabilities)


@pytest.mark.parametrize("head", ["transformed", "adapted", "fp8", "fp8-adapted"])
def test_a_joint_head_trains_on_the_rows_of_a_head_no_one_matrix_is(head, tmp_path):
    """A joint head's lexical prior gathers its backbone's vocabulary rows where no one matrix gives
    the logits, so no output table exists: past a nonlinear prediction head (dense, GELU and norm
    before the tied matrix) the embedding's rows; with an adapter on the head the rows of the head
    its factors update; and with the head's kernel frozen in FP8 under an adapter, its codes
    decoded, updated where the adapter targets it: each as `merge` folds them into one matrix. A
    step through the head moves the head and the backbone's part the rows come from."""
    from dew.decision import JointLayout, JointSchemaHead
    from dew.lora import unadapted
    from dew.nn.kernels.generation import device_generation
    from dew.quantization import InputQuantization

    model = tiny_backbone().clone(head_transform="gelu" if head == "transformed" else None,
                                  tie_embeddings=head == "transformed", max_seq_len=1536)
    variables = model.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32))
    if head == "transformed":
        given, backbone, held = model, model, variables
        expected = variables["params"]["embed_tokens"]["embedding"]
        moving = ("params", "backbone", "embed_tokens")
    else:
        modules = ("q_proj",) if head == "fp8" else ("lm_head", "q_proj")
        fp8 = (InputQuantization(frozen_fp8=("lm_head",), measured_on=device_generation())
               if head.startswith("fp8") else None)
        adapter = LoRA(rank=4, modules=modules).apply(model, variables, key=1, fp8_base=fp8)
        # B starts at zero; drawn, the rows show its update.
        held = jax.tree_util.tree_map_with_path(
            lambda path, leaf: jax.random.normal(jax.random.key(3), leaf.shape, leaf.dtype) / 8
            if "lora_B" in jax.tree_util.keystr(path) else leaf, adapter.variables)
        given, backbone = replace(adapter, variables=held), adapter.model
        expected = unadapted(backbone).apply(given.merge(held), method="output_table").rows()
        moving = ("params", "backbone")
    assert backbone.apply(held, method="output_table") is None
    np.testing.assert_array_equal(backbone.apply(held, method="vocabulary_rows"), expected)

    objective = DecisionObjective(given, variables=joined({"backbone": held}), tokenizer=TOKENIZER,
                                  specials=SPECIALS, layout=JointLayout(max_len=1536),
                                  head=JointSchemaHead(32, width=16, routing_layers=1, layers=1, heads=2,
                                                       feedforward=32))
    data = decision_rows(EXAMPLES, 8, EXAMPLES, objective=objective.data())
    trainer = Trainer(objective, optax.adam(1e-2), key=0, checkpoints=Checkpoints(str(tmp_path)))
    initial = trainer.initial_state()
    # The step donates the state it is handed; the host keeps what it started from.
    started = jax.device_get(initial.variables)
    state = trainer.fit(data, steps=1, log_every=100, state=initial)
    for path in (("params", "head"), moving):
        moved = [not np.array_equal(np.asarray(before), np.asarray(after)) for before, after in
                 zip(leaves(started, *path), leaves(state.variables, *path), strict=True)]
        assert any(moved), path


def test_a_held_out_row_fewer_than_a_batch_counts_once_in_the_loss_and_the_metrics():
    """The validation pass fills a part-full batch with repeats it marks, so
    one held-out row under a batch of eight is scored once: the pass's loss
    and accuracy are that row's alone."""
    from dew.objectives.base import Step
    from dew.training.evaluation import Evaluation

    objective = DecisionObjective(tiny_backbone(), tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT)
    data = decision_rows(EXAMPLES, 8, EXAMPLES[:1], objective=objective.data())
    variables = objective.init(jax.random.key(0))
    result = Evaluation.run(objective, variables, data.val, key=0, metrics=(Accuracy(),), loss=True)
    alone = decision_rows(EXAMPLES, 1, EXAMPLES[:1], objective=objective.data())
    (row,) = list(alone.val(DataPartition()))
    row = {name: value for name, value in row.items() if name != VALID_ROWS}
    loss, _ = objective.scalar_loss(variables, row, Step(jnp.asarray(0), jax.random.key(0), None))
    decisions = objective.evaluate(variables, row, Step(jnp.asarray(0), jax.random.key(0), None))
    assert result.records == 1
    assert result.scores["val/loss"] == pytest.approx(float(loss), rel=1e-6)
    right = np.argmax(decisions.probabilities[0, 0]) == row["labels"][0, 0]
    assert result.scores["val/accuracy"] == float(right)


@pytest.mark.parametrize("separate_tokenizer", [False, True], ids=["shared-source", "separate-tokenizer"])
def test_a_saved_vision_run_answers_image_rows(tmp_path, separate_tokenizer):
    """A shared decision record restores the backbone's image processor.

    A local multimodal source under Dew's own head trains and checkpoints
    without a released decision format. Its reloaded task answers an image
    row with the live task's probabilities, and those differ from the text
    row's, so the saved processor and vision tower both reach the answer.
    The task's tokenizer may live in a different, tokenizer-only directory;
    that directory is not the identity of its image processor.
    """
    from PIL import Image

    from dew.data.text import HFTokenizer
    from dew.decision import JointLayout
    from dew.interop import Pretrained

    directory = Path(__file__).parent / "fixtures" / "hf" / "qwen38-dense-tiny"
    source = Pretrained.load(directory, dtype="float32", attention_impl="xla")
    tokenizer_directory = directory
    if separate_tokenizer:
        tokenizer_directory = tmp_path / "tokenizer"
        tokenizer_directory.mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            shutil.copyfile(directory / name, tokenizer_directory / name)
    tokenizer = HFTokenizer(str(tokenizer_directory), local_files_only=True)
    layout = JointLayout(max_len=1024)
    base = DecisionObjective(source, tokenizer=tokenizer, layout=layout)
    task = Decide(base.model, base.init(jax.random.key(0)), layout, tokenizer, base.specials,
                  processor=source.processor)
    objective, live, reloaded = saved_and_reloaded(task, tmp_path / "run")
    images = [Image.fromarray(array) for array in np.load(directory / "images.npy")]
    asked = {"team": INTENT}
    first = live("charged twice!", asked, images=images)["team"].probabilities
    second = reloaded("charged twice!", asked, images=images)["team"].probabilities

    assert reloaded.processor is not None
    np.testing.assert_array_equal(second, first)
    assert reloaded.processor.source == str(directory)
    assert objective.inference_record()["processor"] == str(directory)
    assert not np.array_equal(second, reloaded("charged twice!", asked)["team"].probabilities)


def saved_and_reloaded(task, run):
    """`task`'s decision objective, the task one checkpointed step over EXAMPLES trains, and that task
    as `Decide.from_run` reads it back from the run directory `run`."""
    objective = DecisionObjective(task)
    checkpoints = Checkpoints(str(run), keep=1)
    state = Trainer(objective, optax.adam(1e-3), key=0, checkpoints=checkpoints).fit(
        decision_rows(EXAMPLES, 8, objective=objective.data()), steps=1, log_every=100, checkpoint_every=1)
    checkpoints.wait()
    return objective, objective.pipeline(state), Decide.from_run(str(run))


def test_a_saved_byte_decision_run_records_no_tokenizer_backed_processor(tmp_path):
    """A text task records its own byte vocabulary, not its source's tokenizer.

    The pretrained decoder brings a tokenizer-backed processor, but the
    decision task lays out text with ByteTokenizer. Saving that task keeps
    no media processor to reload, and never treats `byte` as a Hub repo.
    """
    from model_support import tokenized_qwen3

    from dew.interop import Pretrained

    source = Pretrained.load(tokenized_qwen3(tmp_path / "source"), dtype="float32", attention_impl="xla",
                             max_seq_len=128)
    base = DecisionObjective(source, tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT)
    task = Decide(base.model, base.init(jax.random.key(0)), LAYOUT, TOKENIZER, SPECIALS,
                  processor=source.processor)
    objective, live, reloaded = saved_and_reloaded(task, tmp_path / "run")

    assert source.processor is not None
    assert objective.inference_record()["processor"] is None
    assert isinstance(reloaded.tokenizer, ByteTokenizer)
    assert reloaded.processor is None
    asked = {"team": INTENT}
    np.testing.assert_array_equal(reloaded("charged twice!", asked)["team"].probabilities,
                                  live("charged twice!", asked)["team"].probabilities)


def test_a_saved_one_question_head_refuses_multi_question_requests(tmp_path):
    """A joint layout may train and batch many one-question rows, but neither the live
    nor the saved task may read several question types as one question's embedding."""
    from dew.decision import DecisionHead, JointLayout, Score

    layout = JointLayout(max_len=1024)
    pick = Choice("x", ["0", "1"])
    examples = [Example(state, {"q0": pick}, {"q0": label})
                for state, label in (("first", "0"), ("second", "1"))]
    objective = DecisionObjective(tiny_backbone().clone(max_seq_len=layout.max_len), tokenizer=TOKENIZER,
                                  specials=SPECIALS, layout=layout, head=DecisionHead(32, layers=1),
                                  shuffle_options=False)
    data = decision_rows(examples, 8, examples, objective=objective.data())
    (batch,) = list(data.val(DataPartition()))
    assert int(np.asarray(batch["segments"]).max()) == 2
    checkpoints = Checkpoints(str(tmp_path), keep=1)
    state = Trainer(objective, optax.adam(1e-3), key=0, checkpoints=checkpoints).fit(
        data, steps=1, log_every=100, checkpoint_every=1)
    checkpoints.wait()
    live, reloaded = objective.pipeline(state), Decide.from_run(str(tmp_path))
    requests = [(example.state, example.questions) for example in examples]
    first, second = live.batch(requests), reloaded.batch(requests)
    # Each task is asked at one pass's shapes at a time: a pass of one row and a pass of two run
    # different shapes, whose matmuls a GPU computes with kernels it picks by shape, so a request
    # alone and the same request batched part by rounding (1.5e-7 on an A100), while a row's
    # logits in a pass do not depend on the other rows in it.
    for request, (answer, usage), (again, restored_usage) in zip(requests, first, second, strict=True):
        np.testing.assert_array_equal(answer["q0"].probabilities, again["q0"].probabilities)
        np.testing.assert_array_equal(live(*request)["q0"].probabilities,
                                      reloaded(*request)["q0"].probabilities)
        assert usage == restored_usage
    score = Score("rate", ["low", "high"])
    for task in (live, reloaded):
        for questions in ({"pick": pick, "score": score}, {"score": score, "flag": Noul("x")}):
            with pytest.raises(ValueError):
                task("first", questions)


def test_a_saved_head_reloads_in_the_numerics_it_trained_in(tmp_path):
    """A head configured apart from its backbone, here computing in bfloat16
    over float32 states at jax's highest matmul precision, is recorded with its
    numerics, so the reloaded task answers as the live one rather than in the
    dtype the backbone would give a fresh head."""
    from dew.decision import DecisionHead

    head = DecisionHead(32, layers=1, dtype=jnp.bfloat16, precision=jax.lax.Precision.HIGHEST)
    objective = DecisionObjective(tiny_backbone(), tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT,
                                  head=head)
    data = decision_rows(EXAMPLES, 8, objective=objective.data())
    checkpoints = Checkpoints(str(tmp_path), keep=1)
    state = Trainer(objective, optax.adam(1e-3), key=0, checkpoints=checkpoints).fit(
        data, steps=1, log_every=100, checkpoint_every=1)
    checkpoints.wait()
    live, reloaded = objective.pipeline(state), Decide.from_run(str(tmp_path))
    asked = {"team": INTENT}
    np.testing.assert_array_equal(reloaded("charged twice!", asked)["team"].probabilities,
                                  live("charged twice!", asked)["team"].probabilities)


@pytest.mark.parametrize("precision", [jax.lax.Precision.HIGHEST, jax.lax.DotAlgorithmPreset.F32_F32_F32,
                                       "tensorfloat32", None], ids=str)
def test_a_head_records_the_matmul_precision_it_computes_at(precision):
    """A head's precision, a jax enum, a dot algorithm preset or a name, is written
    into the run's JSON record as a name and read back as the same precision."""
    from dew.decision.head import DecisionHead, Head
    from dew.nn.precision import precision_names

    record = json.loads(json.dumps(DecisionHead(32, layers=1, precision=precision).record()))
    assert precision_names(Head.from_record(record, 32).precision) == precision_names(precision)


@pytest.mark.parametrize("record", [{"features": 32, "layers": 1}, {"name": "DecisionHead", "layers": 1}],
                         ids=["unnamed", "flat"])
def test_a_head_record_names_its_head_and_holds_its_fields(record):
    """A run records a head as its name and its fields (`Head.record`); a
    record of another shape is refused rather than read as Laya's head."""
    from dew.decision.head import Head

    with pytest.raises(ValueError):
        Head.from_record(record, 32)


def test_a_calibration_comes_back_with_the_weights_it_was_fitted_on(tmp_path):
    """A calibration saved into a run reloads with the step it was fitted
    on; trained further, the run's latest weights refuse it rather than
    reuse it stale."""
    objective = DecisionObjective(tiny_backbone(), tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT)
    data = decision_rows(EXAMPLES, 8, EXAMPLES, objective=objective.data())
    checkpoints = Checkpoints(str(tmp_path), keep=2)
    trainer = Trainer(objective, optax.adam(1e-3), key=0, checkpoints=checkpoints)
    state = trainer.fit(data, steps=1, log_every=100, checkpoint_every=1)
    checkpoints.wait()
    calibrated = objective.pipeline(state).calibrated(data.val, type_minimum=1, bucket_minimum=1)
    assert calibrated.weights == Weights(1, ema=False)
    with pytest.raises(TypeError):
        objective.pipeline(state, processor=None)
    calibrated.save(str(tmp_path))
    reloaded = Decide.from_run(str(tmp_path))
    assert reloaded.calibration == calibrated.calibration
    # One run's weights average no runs; a file written before the field reads as one that names none.
    assert json.loads((tmp_path / "decide.json").read_text())["weights"] == {"step": 1, "ema": False,
                                                                           "averaged": []}

    trainer.fit(data, steps=2, log_every=100, checkpoint_every=1)
    checkpoints.wait()
    with pytest.raises(ValueError):
        Decide.from_run(str(tmp_path))
    assert Decide.from_run(str(tmp_path), step=1).calibration == calibrated.calibration
    with pytest.raises(ValueError):
        Decide.from_pretrained(FIXTURES / "tiny").save(str(tmp_path))


def test_diffusion_gemma_reads_each_row_once_whole_and_reloads_from_the_run(tmp_path):
    """DiffusionGemma is a backbone like any other: its causal encoder reads
    each clean row once (`hidden_states`), so a StateFirstLayout is chosen
    for it, and the saved run rebuilds it with identical probabilities."""
    from dew.interop import diffusion_gemma

    directory = Path(__file__).parent / "fixtures" / "hf" / "diffusion-gemma-denoise-tiny"
    text = {**json.loads((directory / "config.json").read_text()), "vocab_size": 256}
    model = diffusion_gemma.build({"model_type": "diffusion_gemma", "canvas_length": 4, "text_config": text},
                                  dtype="float32", attention_impl="xla")
    chosen = DecisionObjective(model, tokenizer=TOKENIZER, specials=SPECIALS).layout
    assert isinstance(chosen, StateFirstLayout)
    objective = DecisionObjective(model, tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT)
    data = decision_rows(EXAMPLES, 8, EXAMPLES, objective=objective.data())
    checkpoints = Checkpoints(str(tmp_path), keep=1)
    state = Trainer(objective, optax.adam(1e-2), key=0, checkpoints=checkpoints).fit(
        data, steps=2, log_every=100, checkpoint_every=2)
    checkpoints.wait()
    live, reloaded = objective.pipeline(state), Decide.from_run(str(tmp_path))
    np.testing.assert_array_equal(live("charged twice!", {"team": INTENT})["team"].probabilities,
                                  reloaded("charged twice!", {"team": INTENT})["team"].probabilities)


def test_a_joint_run_cuts_its_state_to_the_state_budget_and_a_layout_refuses_a_budget_it_lacks():
    """--max-state-tokens reaches a fresh joint layout, whose rows keep that much
    state under a `max_len` sized for their schema; a layout refuses a budget it
    has no field for, a row per question the state's and a joint row a question's."""
    from dew.decision import JointLayout
    from dew.decision.config import _fresh_layout

    layout = _fresh_layout(causal=True, one_question=False, budgets={"max_len": 3072, "max_state_tokens": 16})
    assert layout == JointLayout(max_len=3072, max_state_tokens=16)
    # A budget of no state tokens is a budget, which a run's unset options leave alone.
    bare = _fresh_layout(causal=True, one_question=False,
                         budgets={"max_len": 3072, "max_state_tokens": 0, "option_tokens": None})
    assert bare == JointLayout(max_len=3072, max_state_tokens=0)
    (stateless,) = bare.rows(TOKENIZER, SPECIALS, "x" * 500, {"team": INTENT})
    assert stateless.state_kept == 0
    for budgets in ({"max_state_tokens": -1}, {"max_len": 0}):
        with pytest.raises(ValueError):
            _fresh_layout(causal=True, one_question=False, budgets=budgets)
    (row,) = layout.rows(TOKENIZER, SPECIALS, "x" * 500, {"team": INTENT})
    (whole,) = JointLayout(max_len=3072).rows(TOKENIZER, SPECIALS, "x" * 500, {"team": INTENT})
    assert row.state_kept == 16 and len(whole.tokens) - len(row.tokens) == whole.state_kept - 16
    with pytest.raises(ValueError):
        _fresh_layout(causal=True, one_question=False, budgets={"max_len": 3072, "head_max_len": 96})
    with pytest.raises(ValueError):
        _fresh_layout(causal=True, one_question=True, budgets={"max_len": 1024, "max_state_tokens": 16})


def test_a_hub_tokenizer_lays_rows_out_from_the_revision_its_identity_names(monkeypatch, tmp_path):
    """A decision step holds an HFTokenizer, which resolves a hub name once:
    the step's identity hashes the files of the resolved revision, and the
    step lays its rows out from those same files, here and in a worker it is
    pickled to, never from the name again."""
    import pickle
    import shutil

    import huggingface_hub

    from dew.data.plan import StepIdentity
    from dew.data.text import HFTokenizer

    # The Hub cache keeps a commit's files in a directory named for it.
    files = tmp_path / "snapshots" / ("a" * 40)
    shutil.copytree(Path(__file__).parent / "fixtures" / "tokenizers" / "gpt2", files)
    resolved = []

    def download(repo, revision, allow_patterns, local_files_only):
        resolved.append((repo, revision))
        return str(files)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    encoding = Encoding(LAYOUT, HFTokenizer("acme/tokenizer"), SPECIALS, width=4)
    assert StepIdentity.of(encoding).dependencies["tokenizer"]["revision"] == "a" * 40
    sent = pickle.loads(pickle.dumps(encoding))
    local = Encoding(LAYOUT, HFTokenizer(str(files)), SPECIALS, width=4)
    np.testing.assert_array_equal(sent(EXAMPLES[0], ("team",), None)["tokens"],
                                  local(EXAMPLES[0], ("team",), None)["tokens"])
    assert resolved == [("acme/tokenizer", None)]


@pytest.mark.parametrize("joint", [False, True])
def test_a_head_reads_none_of_its_rows_padding(joint):
    """A batch's rows of unequal length pad the shorter: Laya's head and the joint head give the same
    logits whatever states the backbone leaves on that padding, NaN included, as a padded or diverged
    row's can be."""
    from dew.decision import DecisionInputs, JointLayout, JointSchemaHead
    from dew.objectives.base import part

    layout = JointLayout(max_len=1024) if joint else StateFirstLayout(max_len=512)
    head = JointSchemaHead(hidden_size=32, width=32, routing_layers=1, layers=1, heads=2,
                           feedforward=64) if joint else None
    objective = DecisionObjective(tiny_backbone().clone(max_seq_len=layout.max_len), tokenizer=TOKENIZER,
                                  specials=SPECIALS, layout=layout, head=head)
    rows = [layout.rows(TOKENIZER, SPECIALS, text, {"team": INTENT})[0]
            for text in ("short", "a much longer state than the first")]
    inputs = DecisionInputs.collate(rows, SPECIALS.pad)
    padding = ~np.asarray(inputs.valid)
    assert padding.any() and not padding.all(axis=1).any()
    variables = objective.init(jax.random.key(0))
    model = objective.model
    states = np.asarray(model.states(variables, inputs))
    diverged = np.where(padding[..., None], np.nan, states).astype(states.dtype)
    table = model.table(variables) if model.head.reads_table else None

    def logits(given):
        return np.asarray(model.head.apply(part(variables, "head"), jnp.asarray(given), inputs, table))

    np.testing.assert_array_equal(logits(diverged), logits(states))


@pytest.mark.parametrize("joint", [False, True])
def test_the_abstract_state_draws_a_missing_head_at_the_shapes_init_draws(joint):
    """A missing head drawn in a trace of the state (the abstract state a restore builds) has the
    shapes a concrete init gives it. The joint head reads the backbone's output table as well."""
    from dew.decision import JointLayout, JointSchemaHead

    layout = JointLayout(max_len=96) if joint else LAYOUT
    head = JointSchemaHead(hidden_size=32, width=32, routing_layers=1, layers=1, heads=2,
                           feedforward=64) if joint else None
    objective = DecisionObjective(tiny_backbone(), tokenizer=TOKENIZER, specials=SPECIALS, layout=layout,
                                  head=head)
    backbone = {"params": {"backbone": objective.model.backbone.init(
        jax.random.key(1), np.zeros((1, layout.max_len), np.int32))["params"]}}
    drawn = objective.complete_variables(jax.random.key(0), backbone)
    again = jax.eval_shape(objective.complete_variables, jax.random.key(0), backbone)
    assert jax.tree.map(np.shape, again) == jax.tree.map(np.shape, drawn)
