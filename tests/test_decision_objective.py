"""`DecisionObjective`: proper scoring rules, metrics, row encoding, and training through LoRA."""

import itertools
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.artifacts import Decisions
from dew.checkpoints import Checkpoints
from dew.data import DataPartition
from dew.data.text import ByteTokenizer
from dew.decision import (
    AURC,
    ECE,
    NONE_OF_THE_ABOVE,
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
    Score,
    Specials,
    Spherical,
    StateFirstLayout,
    Weights,
)
from dew.lora import LoRA
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import FROZEN, VALID_ROWS
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


def test_rules_add_and_scale_and_rps_reads_only_levels():
    logits = jnp.asarray([[2.0, 0.0, -1.0], [0.5, 0.5, 0.0]])
    target = jnp.eye(3)[jnp.asarray([0, 2])]
    options = jnp.ones((2, 3), bool)
    ordinal = jnp.asarray([True, False])
    combined = LogLoss() + 0.5 * Brier()
    np.testing.assert_allclose(combined.charge(logits, target, options, ordinal),
                               LogLoss().charge(logits, target, options, ordinal)
                               + 0.5 * Brier().charge(logits, target, options, ordinal), rtol=1e-6)
    assert combined.name == "log_loss+0.5*brier"
    assert float(RankedProbability().charge(logits, target, options, ordinal)[1]) == 0.0
    with pytest.raises(ValueError, match="positive"):
        LogLoss() + -1.0 * Brier()


def test_a_rule_ignores_slots_past_a_rows_options():
    logits = jnp.asarray([[1.0, 0.0, 99.0]])
    options = jnp.asarray([[True, True, False]])
    target = jnp.asarray([[1.0, 0.0, 0.0]])
    for rule in RULES:
        narrow = rule.charge(logits[:, :2], target[:, :2], options[:, :2], jnp.asarray([True]))
        np.testing.assert_allclose(rule.charge(logits, target, options, jnp.asarray([True])), narrow,
                                   rtol=1e-6)


def decisions(tops, correct):
    """Two-option questions, one per row and a second unanswered one beside it,
    whose top probability is `tops` and whose label is right where `correct`."""
    tops = np.asarray(tops)
    probabilities = np.stack([np.stack([tops, 1 - tops], axis=1), np.full((len(tops), 2), 0.5)], axis=1)
    labels = np.stack([np.where(correct, 0, 1), np.ones(len(tops), int)], axis=1)
    scored = np.stack([np.ones(len(tops), bool), np.zeros(len(tops), bool)], axis=1)
    return Decisions(probabilities=probabilities, options=np.ones((len(tops), 2, 2), bool),
                     labels=labels, ordinal=np.zeros((len(tops), 2), bool), scored=scored)


def test_aurc_counts_a_group_of_equal_confidences_whole():
    """Two answers at 0.9, one right, then one right at 0.6: the 0.9 group's
    risk of a half covers two answers, the last a third covers one, as
    Laya's `aurc` weighs levels; and the dataset's order does not matter."""
    metric = AURC()
    first = metric(decisions([0.9, 0.9, 0.6], [True, False, True]), {})
    swapped = metric(decisions([0.9, 0.9, 0.6], [False, True, True]), {})
    expected = (0.5 * 2 + (1 / 3) * 1) / 3
    assert metric.finalize(first) == pytest.approx(expected)
    assert metric.finalize(swapped) == pytest.approx(expected)


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


def test_an_encoding_shuffles_a_choice_and_moves_its_label_with_it():
    encoding = Encoding(LAYOUT, TOKENIZER, SPECIALS, width=4)
    rng = np.random.default_rng(0)
    rows = [encoding(EXAMPLES[0], ("team",), rng) for _ in range(30)]
    labels = {int(row["labels"][0]) for row in rows}
    assert labels == {0, 1, 2}  # the right option lands in every slot
    for row in rows:
        marker = int(row["option_spans"][0, int(row["labels"][0]), 0])
        option = bytes(int(token) for token in row["tokens"][marker - 18:marker]).decode()
        assert "billing" in option
    plain = encoding(EXAMPLES[0], ("team",), None)
    assert int(plain["labels"][0]) == 0 and plain["options"][0].tolist() == [True, True, True, False]
    assert plain["scored"].tolist() == [True]


def test_an_encoding_keeps_levels_in_order_and_adds_none_of_the_above():
    rating = Example("fine", {"q": Score("How bad?", ["calm", "upset", "furious"])}, {"q": 2})
    encoding = Encoding(LAYOUT, TOKENIZER, SPECIALS, width=4, none_of_the_above=1.0)
    rng = np.random.default_rng(3)
    assert all(int(encoding(rating, ("q",), rng)["labels"][0]) == 2 for _ in range(10))
    unshuffled = replace(encoding, shuffle=False)
    seen = set()
    for _ in range(40):
        row = unshuffled(EXAMPLES[0], ("team",), rng)
        assert int(row["options"].sum()) in (3, 4)
        seen.add((int(row["options"].sum()), int(row["labels"][0])))
    # Without its right option the row's answer is the added one, last; with
    # it, still billing, first.
    assert seen == {(3, 2), (4, 0)}
    assert NONE_OF_THE_ABOVE == "none of the above"


def test_a_soft_target_follows_its_options_through_the_shuffle():
    """A target distribution is laid out in the order the row shows the
    options, its most likely option is the row's label, and a choice whose
    gold is a distribution gains no "none of the above" option."""
    soft = Example("charged twice", {"team": INTENT}, targets={"team": (0.6, 0.3, 0.1)})
    encoding = Encoding(LAYOUT, TOKENIZER, SPECIALS, width=4, none_of_the_above=1.0)
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(20):
        row = encoding(soft, ("team",), rng)
        assert int(row["options"].sum()) == 3
        shown = row["targets"][0, :3]
        assert sorted(shown.tolist()) == pytest.approx([0.1, 0.3, 0.6])
        assert int(row["labels"][0]) == int(np.argmax(shown))
        for slot in range(3):
            marker = int(row["option_spans"][0, slot, 0])
            text = bytes(int(token) for token in row["tokens"][marker - 24:marker]).decode()
            key = {0.6: "billing", 0.3: "technical", 0.1: "sales"}[round(float(shown[slot]), 1)]
            # The layout keeps each option's first `option_tokens` bytes.
            assert text.endswith(f" {key}: {INTENT.criteria[key]}"[:LAYOUT.option_tokens])
        seen.add(tuple(np.round(shown, 1).tolist()))
    assert len(seen) > 1
    with pytest.raises(ValueError, match="distribution over the question's 3 options"):
        Example("x", {"team": INTENT}, targets={"team": (0.5, 0.5)})
    with pytest.raises(ValueError, match="both an answer and a target"):
        Example("x", {"team": INTENT}, {"team": "billing"}, {"team": (1.0, 0.0, 0.0)})


def test_the_loss_scores_a_soft_target_as_its_rule_does():
    """The log loss of a row whose gold is a distribution is the cross-entropy
    against that distribution, not against its most likely option."""
    from dew.decision.task import laid_out
    from dew.objectives.base import Step

    soft = [Example(f"charged twice {index}", {"team": INTENT}, targets={"team": (0.6, 0.3, 0.1)})
            for index in range(8)]
    objective = DecisionObjective(tiny_backbone(), tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT,
                                  shuffle_options=False)
    data = objective.dataset(soft, batch=8, validation=soft)
    (batch,) = list(data.val(DataPartition()))
    batch = {name: value for name, value in batch.items() if name != VALID_ROWS}
    variables = objective.init(jax.random.key(0))
    loss, _ = objective.scalar_loss(variables, batch, Step(jnp.asarray(0), jax.random.key(0), None))
    logits = np.asarray(objective.model.logits(variables, laid_out(batch)))[:, 0, :3]
    expected = -np.mean(np.sum(np.array([0.6, 0.3, 0.1]) * jax.nn.log_softmax(logits), axis=-1))
    assert float(loss) == pytest.approx(float(expected), rel=1e-5)


def test_a_joint_row_asks_its_questions_in_a_fresh_order():
    """Clef's layout reads every question in one row; each read asks them in
    a new order, and each slot's target is its own question's."""
    from dew.decision import JointLayout

    example = Example("charged twice", {"team": INTENT, "urgent": Noul("Is it urgent?")},
                      {"team": "billing", "urgent": "true"})
    encoding = Encoding(JointLayout(max_len=4096), TOKENIZER, SPECIALS, width=3, questions=2)
    rng = np.random.default_rng(0)
    orders = set()
    for _ in range(12):
        row = encoding(example, ("team", "urgent"), rng)
        widths = tuple(int(count) for count in row["options"].sum(axis=-1))
        orders.add(widths)
        for slot, width in enumerate(widths):
            target = row["targets"][slot, :width]
            assert target.sum() == 1.0 and int(row["labels"][slot]) == int(np.argmax(target))
    assert orders == {(3, 2), (2, 3)}


def test_a_mixture_fills_each_step_at_its_weights():
    """Two sets of very different lengths, mixed three to one: every batch of
    eight holds six rows of the first and two of the second."""
    from dew.decision import Weighted

    small = [Example(f"zz {index}", {"team": INTENT}, {"team": "sales"}) for index in range(3)]
    objective = DecisionObjective(tiny_backbone(), tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT)
    data = objective.dataset({"large": Weighted(EXAMPLES * 5, 3.0), "small": Weighted(small, 1.0)},
                             batch=8, validation=EXAMPLES)
    stream = iter(data.train(DataPartition()))
    for _ in range(4):
        batch = next(stream)
        firsts = [bytes([int(row[0])]).decode() for row in np.asarray(batch["tokens"])]
        assert firsts.count("z") == 2
    # A pass reads every set at least once: the large one's 40 rows at three quarters of each step.
    assert data.records == 54


def test_a_bucketed_stream_cuts_each_batch_just_past_its_longest_row():
    """Rows of forty lengths, two to a batch: each batch is cut at the next
    multiple of eight past its longest row, a window of batches holds the
    uncut stream's rows, and cutting a batch leaves its loss as it was."""
    from dew.objectives.base import Step

    rows = [Example("x" * size, {"team": INTENT}, {"team": "sales"}) for size in range(1, 41)]

    def read(**fields):
        objective = DecisionObjective(tiny_backbone(), tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT,
                                      shuffle_options=False, **fields)
        return objective, list(itertools.islice(objective.dataset(rows, batch=2).train(DataPartition()), 64))

    def held(batches):
        return sorted(tuple(row[valid]) for batch in batches for row, valid
                      in zip(np.asarray(batch["tokens"]), np.asarray(batch["valid"]), strict=True))

    plain, whole = read()
    bucketed, cut = read(bucket=8)
    assert bucketed.inputs is None
    for batch in cut:
        size, longest = batch["tokens"].shape[1], int(np.sum(batch["valid"], axis=1).max())
        assert size % 8 == 0 and longest <= size < longest + 8
    assert held(cut) == held(whole)
    batch = whole[0]
    size = -(-int(np.sum(batch["valid"], axis=1).max()) // 8) * 8
    trimmed = {name: value[:, :size] if name in ("tokens", "valid", "positions", "slots") else value
               for name, value in batch.items()}
    variables = plain.init(jax.random.key(0))
    step = Step(jnp.asarray(0), jax.random.key(0), None)
    np.testing.assert_allclose(plain.scalar_loss(variables, trimmed, step)[0],
                               plain.scalar_loss(variables, batch, step)[0], rtol=1e-6)


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
    data = objective.dataset(EXAMPLES, batch=8, validation=EXAMPLES)
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


def test_a_held_out_row_fewer_than_a_batch_counts_once_in_the_loss_and_the_metrics():
    """The validation pass fills a part-full batch with repeats it marks, so
    one held-out row under a batch of eight is scored once: the pass's loss
    and accuracy are that row's alone."""
    from dew.objectives.base import Step
    from dew.training.evaluation import Evaluation

    objective = DecisionObjective(tiny_backbone(), tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT)
    data = objective.dataset(EXAMPLES, batch=8, validation=EXAMPLES[:1])
    variables = objective.init(jax.random.key(0))
    result = Evaluation.run(objective, variables, data.val, key=0, metrics=(Accuracy(),), loss=True)
    alone = objective.dataset(EXAMPLES, batch=1, validation=EXAMPLES[:1])
    (row,) = list(alone.val(DataPartition()))
    row = {name: value for name, value in row.items() if name != VALID_ROWS}
    loss, _ = objective.scalar_loss(variables, row, Step(jnp.asarray(0), jax.random.key(0), None))
    decisions = objective.evaluate(variables, row, Step(jnp.asarray(0), jax.random.key(0), None))
    assert result.records == 1
    assert result.scores["val/loss"] == pytest.approx(float(loss), rel=1e-6)
    right = np.argmax(decisions.probabilities[0, 0]) == row["labels"][0, 0]
    assert result.scores["val/accuracy"] == float(right)


def test_a_calibration_comes_back_with_the_weights_it_was_fitted_on(tmp_path):
    """A calibration saved into a run reloads with the step it was fitted
    on; trained further, the run's latest weights refuse it rather than
    reuse it stale."""
    objective = DecisionObjective(tiny_backbone(), tokenizer=TOKENIZER, specials=SPECIALS, layout=LAYOUT)
    data = objective.dataset(EXAMPLES, batch=8, validation=EXAMPLES)
    checkpoints = Checkpoints(str(tmp_path), keep=2)
    trainer = Trainer(objective, optax.adam(1e-3), key=0, checkpoints=checkpoints)
    state = trainer.fit(data, steps=1, log_every=100, checkpoint_every=1)
    checkpoints.wait()
    calibrated = objective.pipeline(state).calibrated(data.val, type_minimum=1, bucket_minimum=1)
    assert calibrated.weights == Weights(1, ema=False)
    with pytest.raises(TypeError, match="takes no processor"):
        objective.pipeline(state, processor=None)
    calibrated.save(str(tmp_path))
    reloaded = Decide.from_run(str(tmp_path))
    assert reloaded.calibration == calibrated.calibration
    assert json.loads((tmp_path / "decide.json").read_text())["weights"] == {"step": 1, "ema": False}

    trainer.fit(data, steps=2, log_every=100, checkpoint_every=1)
    checkpoints.wait()
    with pytest.raises(ValueError, match="fitted on step 1"):
        Decide.from_run(str(tmp_path))
    assert Decide.from_run(str(tmp_path), step=1).calibration == calibrated.calibration
    with pytest.raises(ValueError, match="no run checkpoint"):
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
    data = objective.dataset(EXAMPLES, batch=8, validation=EXAMPLES)
    checkpoints = Checkpoints(str(tmp_path), keep=1)
    state = Trainer(objective, optax.adam(1e-2), key=0, checkpoints=checkpoints).fit(
        data, steps=2, log_every=100, checkpoint_every=2)
    checkpoints.wait()
    live, reloaded = objective.pipeline(state), Decide.from_run(str(tmp_path))
    np.testing.assert_array_equal(live("charged twice!", {"team": INTENT})["team"].probabilities,
                                  reloaded("charged twice!", {"team": INTENT})["team"].probabilities)


def test_a_joint_rows_fit_is_read_from_its_schema_as_laying_it_out_finds():
    """A joint row lays out exactly when its schema leaves room, whatever its
    state, which the layout cuts: `fits` reads the schema alone and agrees
    with `rows` on every case, a long state, a schema at the edge and one past it."""
    from dew.decision import JointLayout
    from dew.decision.config import _fitting
    from dew.decision.data import Weighted

    layout = JointLayout(max_len=900)
    wide = Choice("Which code?", {f"code-{index}": f"the {index}th code" for index in range(60)})
    narrow = Choice("Which code?", {f"code-{index}": f"the {index}th code" for index in range(12)})
    cases = [Example("x" * 5000, {"team": INTENT}), Example("short", {"code": wide}),
             Example("short", {"code": narrow}), Example("y" * 3000, {"team": INTENT, "code": narrow})]

    def laid_out(example):
        try:
            layout.rows(TOKENIZER, SPECIALS, example.state, example.questions)
        except ValueError:
            return False
        return True

    expected = [laid_out(example) for example in cases]
    assert expected == [True, False, True, True]
    assert [layout.fits(TOKENIZER, example.questions) for example in cases] == expected
    objective = DecisionObjective(tiny_backbone(), tokenizer=TOKENIZER, specials=SPECIALS, layout=layout)
    kept, held, unfit = _fitting(objective, {"rows": Weighted(cases, 1.0)}, cases[:2])
    assert kept["rows"].examples == [case for case, fit in zip(cases, expected, strict=True) if fit]
    assert unfit == {"rows": 1, "held out": 1} and held == cases[:1]
