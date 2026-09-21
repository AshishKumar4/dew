"""Knowledge distillation from a frozen teacher: the MaxText loss, its
schedules, the feature pairs, and the objective through the trainer.

The arithmetic is checked against MaxText 0.2.4's
`CombinedDistillationStrategy.compute_loss`, transcribed in float64 torch
in tools/distillation_reference.py and run on two tiny fixed-seed decoders
into tests/fixtures/distillation: the loss, its reported terms, and the
gradient with respect to the student's parameters, which the test chains
from the reference's logit and feature cotangents through the models with
`jax.vjp`. The rest runs through the objective on the same decoders: the
weights at the schedules' ends, a projection only where widths differ, the
refusal of a vocabulary mismatch, and a real trainer run on CPU where the
teacher never moves and a resumed run lands where the straight one does.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.traverse_util import unflatten_dict

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives import DistillationObjective
from dew.objectives.base import Step, scalar_loss
from dew.objectives.distillation import PROJECTIONS, TEACHER
from dew.objectives.lm import TEXT_KEY, LMObjective
from dew.training import Checkpoints, Layout, MeshSpec, Trainer

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "distillation"
META = json.loads((FIXTURES / "meta.json").read_text())
SEQ = META["seq_len"]
VOCAB = META["vocab"]
TINY_SHARD = 256


def fixture() -> dict:
    with np.load(FIXTURES / "fixture.npz") as data:
        return dict(data.items())


def tree(arrays: dict, prefix: str) -> dict:
    """The variables the fixture stores under `prefix`, nested again."""
    return unflatten_dict({tuple(name[len(prefix):].split("/")): jnp.asarray(value)
                           for name, value in arrays.items() if name.startswith(prefix)})


def models():
    return CausalTransformer(**META["student"]), CausalTransformer(**META["teacher"])


def fixture_objective(arrays: dict, **overrides) -> tuple[DistillationObjective, dict]:
    """The objective over the fixture's variables, its projections included."""
    student_model, teacher_model = models()
    student = LMObjective(student_model, SEQ, pad_id=META["pad_id"], ema_decay=None,
                          pretrained=tree(arrays, "student/"))
    teacher = LMObjective(teacher_model, SEQ, ema_decay=None, pretrained=tree(arrays, "teacher/"))
    settings = {"alpha": META["alpha"], "temperature": META["temperature"], "beta": META["beta"],
                "features": [tuple(pair) for pair in META["pairs"]], **overrides}
    objective = DistillationObjective(student, teacher, **settings)
    params = objective.init(jax.random.key(0))
    if PROJECTIONS in params["params"]:
        projections = {name: jnp.asarray(arrays[name]) for name in params["params"][PROJECTIONS]}
        params["params"] = {**params["params"], PROJECTIONS: projections}
    return objective, params


def fixture_batch(arrays: dict) -> dict:
    return {TEXT_KEY: jnp.asarray(arrays["tokens"])}


def step_at(index=0, key=1, ema=None) -> Step:
    return Step(step=jnp.asarray(index), key=jax.random.key(key), ema=ema)


def scaled_close(actual, expected, tolerance: float) -> None:
    """The fp32 parity bound: the largest error, scaled by the largest reference value."""
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    scale = max(float(np.abs(expected).max()), 1.0)
    assert float(np.abs(actual - expected).max()) <= tolerance * scale


# --------------------------------------------------------------------------
# The loss against the reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["cosine", "l2"])
def test_the_loss_and_its_terms_match_the_maxtext_reference(kind):
    """Loss and every reported term within 1e-4 of the float64 reference,
    scaled; observed at most 1.2e-7 on CPU. Padding sits in the batch, so the
    mask is part of what agrees."""
    arrays = fixture()
    objective, params = fixture_objective(arrays, feature_loss=kind)

    loss, aux = scalar_loss(objective, params, fixture_batch(arrays), step_at())

    scaled_close(loss, arrays[f"{kind}/loss"], 1e-4)
    for metric, name in (("ce", "hard_loss"), ("distill/soft_loss", "soft_loss"),
                         ("distill/kl", "kl_div_at_T"), ("distill/feature", "feature"),
                         ("distill/teacher_loss", "teacher_loss")):
        scaled_close(aux.metrics[metric], arrays[f"{kind}/{name}"], 1e-4)


@pytest.mark.parametrize("kind", ["cosine", "l2"])
def test_the_student_gradient_matches_the_reference_cotangents(kind):
    """The gradient with respect to every trainable leaf, the projections
    included, within 1e-4 scaled of the reference's logit and feature
    cotangents pulled back through the student; observed at most 1.8e-7."""
    arrays = fixture()
    objective, params = fixture_objective(arrays, feature_loss=kind)
    batch = fixture_batch(arrays)

    def student_outputs(trainable):
        _, _, prediction = objective.student.predict(
            {"params": {name: value for name, value in trainable.items() if name != PROJECTIONS}},
            batch, step_at(), train=True, layers=[s for _, s in META["pairs"]])
        projected = jnp.stack([state.astype(jnp.float32) @ trainable[PROJECTIONS][f"projection_{index}"]
                               for index, state in enumerate(prediction.hidden)])
        return prediction.logits, projected

    (logits, _), pullback = jax.vjp(student_outputs, params["params"])
    scaled_close(logits, arrays["student_logits"], 1e-5)
    expected, = pullback((jnp.asarray(arrays[f"{kind}/d_student_logits"], jnp.float32),
                          jnp.asarray(arrays[f"{kind}/d_student_features"], jnp.float32)))
    actual = jax.grad(lambda trainable: scalar_loss(
        objective, {**params, "params": trainable}, batch, step_at())[0])(params["params"])

    expected_leaves = dict(jax.tree_util.tree_leaves_with_path(expected))
    actual_leaves = dict(jax.tree_util.tree_leaves_with_path(actual))
    assert actual_leaves.keys() == expected_leaves.keys()
    for path, leaf in expected_leaves.items():
        assert float(jnp.abs(leaf).max()) > 0, path
        scaled_close(actual_leaves[path], leaf, 1e-4)


def test_maxtext_anneals_are_optax_schedules():
    """`compute_schedule`'s linear and cosine anneals from start to end over
    max_steps are `optax.linear_schedule` and `optax.cosine_decay_schedule`
    with alpha = end / start, at every fixture step; within 1e-6."""
    arrays = fixture()
    start, end, steps = (META["schedule"][name] for name in ("start", "end", "max_steps"))
    linear = optax.linear_schedule(start, end, steps)
    cosine = optax.cosine_decay_schedule(start, steps, alpha=end / start)
    for index, step in enumerate(arrays["schedule/steps"]):
        assert float(linear(step)) == pytest.approx(float(arrays["schedule/linear"][index]), abs=1e-6)
        assert float(cosine(step)) == pytest.approx(float(arrays["schedule/cosine"][index]), abs=1e-6)


# --------------------------------------------------------------------------
# Schedules, endpoints and the feature pairs through the objective
# --------------------------------------------------------------------------


def test_a_scheduled_weight_reads_the_step():
    """alpha annealed 0.6 to 0.1 over 100 steps: at step 0 the loss is the
    constant-0.6 loss, at step 100 the constant-0.1 one, and a step in
    between is neither."""
    arrays = fixture()
    batch = fixture_batch(arrays)
    scheduled, params = fixture_objective(arrays, alpha=optax.linear_schedule(0.6, 0.1, 100))
    at_start = float(scalar_loss(scheduled, params, batch, step_at(0))[0])
    at_end = float(scalar_loss(scheduled, params, batch, step_at(100))[0])
    between = float(scalar_loss(scheduled, params, batch, step_at(50))[0])

    first, _ = fixture_objective(arrays, alpha=0.6)
    last, _ = fixture_objective(arrays, alpha=0.1)
    assert at_start == pytest.approx(float(scalar_loss(first, params, batch, step_at(0))[0]), rel=1e-6)
    assert at_end == pytest.approx(float(scalar_loss(last, params, batch, step_at(0))[0]), rel=1e-6)
    assert min(at_start, at_end) < between < max(at_start, at_end)


def test_alpha_zero_without_features_is_the_students_own_loss():
    """The endpoints of the mix: at alpha 0 with no pairs the loss is the
    student's loss to the bit, and at alpha 1 it is T^2 times the KL."""
    arrays = fixture()
    batch = fixture_batch(arrays)
    hard, params = fixture_objective(arrays, alpha=0.0, beta=0.0, features=())
    assert PROJECTIONS not in params["params"]
    own, _ = scalar_loss(hard.student, hard.student_variables(params), batch, step_at())
    mixed, _ = scalar_loss(hard, params, batch, step_at())
    assert float(mixed) == float(own)

    soft, _ = fixture_objective(arrays, alpha=1.0, beta=0.0, features=())
    loss, aux = scalar_loss(soft, params, batch, step_at())
    assert float(loss) == pytest.approx(float(aux.metrics["distill/soft_loss"]), rel=1e-6)
    assert float(loss) == pytest.approx(META["temperature"] ** 2 * float(aux.metrics["distill/kl"]), rel=1e-6)


def test_a_pair_of_equal_widths_needs_no_projection():
    """Teacher and student of the same width: no `distillation` leaves in
    the params, and the feature term still reaches the loss."""
    student_model = CausalTransformer(**{**META["student"], "num_layers": 3})
    student = LMObjective(student_model, SEQ, ema_decay=None)
    teacher = LMObjective(CausalTransformer(**META["student"]), SEQ, ema_decay=None)
    objective = DistillationObjective(student, teacher, features=[(1, 2)], beta=1.0)
    params = objective.init(jax.random.key(0))
    assert PROJECTIONS not in params["params"]

    batch = fixture_batch(fixture())
    with_feature, aux = scalar_loss(objective, params, batch, step_at())
    without, _ = scalar_loss(DistillationObjective(student, teacher), params, batch, step_at())
    assert float(aux.metrics["distill/feature"]) > 0
    assert float(with_feature) == pytest.approx(float(without) + float(aux.metrics["distill/feature"]), rel=1e-6)


def test_a_layer_the_model_does_not_have_is_named_at_init():
    student = LMObjective(CausalTransformer(**META["student"]), SEQ, ema_decay=None)
    teacher = LMObjective(CausalTransformer(**META["teacher"]), SEQ, ema_decay=None)
    with pytest.raises(ValueError, match=r"no layer 5; it has layers \[0, 1\]"):
        DistillationObjective(student, teacher, features=[(0, 5)]).init(jax.random.key(0))


def test_beta_without_pairs_and_weights_outside_their_range_are_refused():
    student_model, teacher_model = models()
    student = LMObjective(student_model, SEQ, ema_decay=None)
    teacher = LMObjective(teacher_model, SEQ, ema_decay=None)
    with pytest.raises(ValueError, match="none were named"):
        DistillationObjective(student, teacher, beta=0.5)
    with pytest.raises(ValueError, match="alpha=1.5"):
        DistillationObjective(student, teacher, alpha=1.5)
    with pytest.raises(ValueError, match="temperature=0"):
        DistillationObjective(student, teacher, temperature=0.0)


def test_a_vocabulary_mismatch_is_refused_with_the_reason():
    student = LMObjective(CausalTransformer(**META["student"]), SEQ, ema_decay=None)
    teacher = LMObjective(CausalTransformer(**{**META["teacher"], "vocab_size": VOCAB + 5}), SEQ,
                          ema_decay=None)
    objective = DistillationObjective(student, teacher)
    params = objective.init(jax.random.key(0))
    with pytest.raises(ValueError, match="share a tokenizer and a vocabulary"):
        scalar_loss(objective, params, fixture_batch(fixture()), step_at())


def test_a_student_with_the_router_balance_loss_is_refused():
    """`aux_loss_alpha`'s router terms normalise per router, not per token,
    so the seam refuses them at the first trace and names the way out."""
    arrays = fixture()
    student = LMObjective(CausalTransformer(**META["student"]), SEQ, ema_decay=None, aux_loss_alpha=0.01)
    teacher = LMObjective(CausalTransformer(**META["teacher"]), SEQ, ema_decay=None)
    objective = DistillationObjective(student, teacher)
    with pytest.raises(ValueError, match="balance_rate"):
        scalar_loss(objective, objective.init(jax.random.key(0)), fixture_batch(arrays), step_at())


# --------------------------------------------------------------------------
# Through the trainer
# --------------------------------------------------------------------------


class Data:
    """The one fixed batch, endlessly, as the trainer's dataset contract."""

    def __init__(self, batch):
        self.batch = batch["text"].shape[0]
        self._batch = batch

    def train(self):
        while True:
            yield self._batch

    val = None
    steps_per_epoch = None


def make_trainer(arrays, tmp_path=None):
    objective, _ = fixture_objective(arrays)
    return Trainer(objective, optax.adam(3e-3), key=jax.random.key(0), mesh=MeshSpec(),
                   layout=Layout(min_shard=TINY_SHARD),
                   checkpoints=None if tmp_path is None else Checkpoints(str(tmp_path / "distill")))


@pytest.mark.mesh
def test_the_teacher_never_moves_and_the_student_learns(tmp_path):
    """Five steps on one batch: the teacher collection is bitwise what init
    gave, the projections and the student moved, the loss on that batch fell,
    and a run killed at step 3 and resumed to 5 lands where the straight run
    lands."""
    arrays = fixture()
    # The fixture's two rows on every one of the eight data-parallel devices.
    batch = {TEXT_KEY: jnp.tile(fixture_batch(arrays)[TEXT_KEY], (4, 1))}
    trainer = make_trainer(arrays)
    initial = trainer.initial_state()
    before = float(scalar_loss(trainer.objective, initial.params, batch, step_at())[0])

    state = trainer.fit(Data(batch), steps=5, log_every=1)

    for path, leaf in jax.tree_util.tree_leaves_with_path(state.params[TEACHER]):
        assert np.array_equal(np.asarray(leaf), np.asarray(dict(
            jax.tree_util.tree_leaves_with_path(initial.params[TEACHER]))[path])), path
    for name, kernel in state.params["params"][PROJECTIONS].items():
        assert not np.array_equal(np.asarray(kernel), np.asarray(initial.params["params"][PROJECTIONS][name])), name
    after = float(scalar_loss(trainer.objective, state.params, batch, step_at())[0])
    assert after < before, (before, after)

    make_trainer(arrays, tmp_path).fit(Data(batch), steps=3, log_every=1)
    resumed = make_trainer(arrays, tmp_path).fit(Data(batch), steps=5, log_every=1)
    for straight, again in zip(jax.tree.leaves(state.params), jax.tree.leaves(resumed.params), strict=True):
        np.testing.assert_array_equal(np.asarray(straight), np.asarray(again))
    assert Checkpoints(str(tmp_path / "distill")).latest == 5


def test_the_student_tree_is_what_a_plain_student_run_reads():
    """`student_variables` strips the teacher and the projections, and the
    student objective scores that tree as it scores its own."""
    arrays = fixture()
    objective, params = fixture_objective(arrays)
    own = objective.student_variables(params)
    assert TEACHER not in own and PROJECTIONS not in own["params"]
    batch = fixture_batch(arrays)
    _, aux = scalar_loss(objective, params, batch, step_at())
    loss, _ = scalar_loss(objective.student, own, batch, step_at())
    assert float(loss) == pytest.approx(float(aux.metrics["ce"]), rel=1e-6)
    assert objective.ema is None
