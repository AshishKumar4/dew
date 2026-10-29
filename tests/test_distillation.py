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

import dataclasses
import functools
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.traverse_util import unflatten_dict
from reference_error import as_given, assert_as_exact, assert_as_exact_as_the_reference, widened

from dew.batch import TEXT_KEY
from dew.data.dataset import DataPartition, Loader, replayed
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives import DistillationObjective
from dew.objectives.base import Step
from dew.objectives.distillation import PROJECTIONS, TEACHER
from dew.objectives.lm import LMObjective
from dew.objectives.routing import ExpertBalance
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
                          variables=tree(arrays, "student/"))
    teacher = LMObjective(teacher_model, SEQ, ema_decay=None, variables=tree(arrays, "teacher/"))
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


def teacher_outputs(arrays: dict, variables=None) -> tuple[np.ndarray, np.ndarray]:
    """The teacher's logits and its paired layers' outputs, `[L, B, T, D]`, from the float32
    forward the objective runs, over `variables` (by default the fixture's teacher). The fixture
    stores none: the outputs its generator saw were another XLA's rounding of that forward (up to
    2.7e-6 off in states of about 8), which, fed to the reference, would leave that rounding in
    Dew's distance from float64 alone. A float64 truth's widened variables run in float32 too:
    the teacher's outputs are the reference's data, not what it computes."""
    variables = {"params": tree(arrays, "teacher/")["params"]} if variables is None else variables
    variables = jax.tree.map(lambda leaf: np.asarray(leaf, np.float32), variables)
    with jax.enable_x64(new_val=False):
        logits, states = teacher_forward()(variables, jnp.asarray(arrays["tokens"][:, :-1]))
        return np.asarray(logits), np.stack([np.asarray(state) for state in states])


@functools.cache
def teacher_forward():
    """The teacher's float32 forward `teacher_outputs` reads, compiled once for every draw."""
    from tools.distillation_reference import dew_outputs

    teacher, layers = models()[1], [layer for layer, _ in META["pairs"]]
    return jax.jit(lambda variables, ids: dew_outputs(teacher, variables, ids, layers))


DISTILLATION_SEEDS = {"terms": 64, "gradient": 16}
"""Check 1's declared seeds for the loss's terms and the student's gradient, sized by `seeds_for`
(2026-10-21, on armada)."""


def drawn_parameters(params):
    """`draw(seed)` over the objective's whole tree, the teacher's and the projections included:
    seed 0 the fixture's, each other seed every leaf moved by a relative 1e-2 of its own."""
    def draw(seed):
        if seed == 0:
            return (params,)
        keys = iter(jax.random.split(jax.random.key(seed), len(jax.tree.leaves(params))))
        return (jax.tree.map(lambda leaf: np.asarray(
            leaf * (1 + 1e-2 * jax.random.normal(next(keys), leaf.shape, leaf.dtype)), leaf.dtype), params),)

    return draw


def reference_run(arrays: dict, kind: str, outputs, teacher, dtype: str, *, cotangents: bool):
    """The reference transcription run in torch at the `dtype` named, on the student's `outputs`,
    its logits and projected features, beside the `teacher`'s (`teacher_outputs`): its loss, its
    terms, and, where `cotangents`, its cotangents on those outputs (None otherwise)."""
    import torch

    from tools.distillation_reference import distillation_loss

    targets = arrays["tokens"][:, 1:]
    logits, features = (torch.tensor(np.asarray(value), requires_grad=cotangents) for value in outputs)
    teacher_logits, teacher_features = (torch.tensor(value) for value in teacher)
    loss, terms = distillation_loss(
        logits, teacher_logits, torch.tensor(targets, dtype=torch.int64),
        torch.tensor((targets != META["pad_id"]).astype(np.float32)), features, teacher_features,
        META["alpha"], META["temperature"], META["beta"], kind, dtype=getattr(torch, dtype))
    reported = {name: value.item() for name, value in terms.items()}
    if not cotangents:
        return loss.item(), reported, None
    d_logits, d_features = torch.autograd.grad(loss, (logits, features))
    return loss.item(), reported, (d_logits.numpy(), d_features.numpy())


def student_outputs(model, trainable, ids):
    """The student's logits and its paired layers' outputs projected, as the reference reads them
    (`tools/distillation_reference.py:dew_outputs`)."""
    from tools.distillation_reference import dew_outputs

    variables = {"params": {name: value for name, value in trainable.items() if name != PROJECTIONS}}
    logits, states = dew_outputs(model, variables, ids, [layer for _, layer in META["pairs"]])
    projections = trainable[PROJECTIONS]
    return logits, jnp.stack([state @ projections[f"projection_{index}"]
                              for index, state in enumerate(states)])


@functools.cache
def student_forward(*, wide: bool):
    """The student's outputs (`student_outputs`), compiled once for every draw: in float64 where
    `wide`, over float64 leaves under `jax.enable_x64`."""
    model = models()[0]
    model = model.clone(dtype=jnp.float64) if wide else model
    return jax.jit(lambda trainable, ids: student_outputs(model, trainable, ids))


@functools.cache
def student_pullback(*, wide: bool):
    """The pullback of the student's outputs (`student_forward`) of `cotangents`, compiled once."""
    model = models()[0]
    model = model.clone(dtype=jnp.float64) if wide else model
    return jax.jit(lambda trainable, ids, cotangents: jax.vjp(
        lambda leaves: student_outputs(model, leaves, ids), trainable)[1](cotangents)[0])


# --------------------------------------------------------------------------
# The loss against the reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["cosine", "l2"])
def test_the_loss_and_its_terms_match_the_maxtext_reference(kind):
    """Loss and every reported term on par with the reference's own float32
    run over drawn students, teachers and projections (check 1): the truth
    runs the reference in float64 on the student's float64 forward, the
    reference in float32 on its float32 one, both beside the teacher's
    forward that Dew runs (`teacher_outputs`). Padding sits in the batch, so
    the mask is part of what agrees."""
    arrays = fixture()
    objective, params = fixture_objective(arrays, feature_loss=kind)
    ids = jnp.asarray(arrays["tokens"][:, :-1])
    named = {"loss": "loss", "ce": "hard_loss", "distill/soft_loss": "soft_loss", "distill/kl": "kl_div_at_T",
             "distill/feature": "feature", "distill/teacher_loss": "teacher_loss"}

    @jax.jit
    def dew(params):
        loss, aux = objective.scalar_loss(params, fixture_batch(arrays), step_at())
        return {metric: {**aux.metrics, "loss": loss}[metric] for metric in named}

    @as_given
    def reference(params, dtype="float32"):
        wide = dtype == "float64"
        trainable = widened(params["params"]) if wide else params["params"]
        outputs = student_forward(wide=wide)(trainable, ids)
        loss, terms, _ = reference_run(arrays, kind, outputs, teacher_outputs(arrays, params[TEACHER]), dtype,
                                       cotangents=False)
        return {metric: np.float64({**terms, "loss": loss}[name]) for metric, name in named.items()}

    assert_as_exact(dew, reference, drawn_parameters(params), f"{kind} distillation terms",
                    seeds=DISTILLATION_SEEDS["terms"], truth_fn=lambda params: reference(params, "float64"))


@pytest.mark.parametrize("kind", ["cosine", "l2"])
def test_the_student_gradient_matches_the_reference_cotangents(kind):
    """The gradient with respect to every trainable leaf, the projections
    included, on par with the reference's own float32 run over drawn
    students, teachers and projections (check 1): the truth pulls the
    reference's float64 cotangents at the student's float64 forward back
    through that forward, and the reference pulls its float32 cotangents at
    the float32 forward back through the student in float32, both beside the
    teacher's forward that Dew runs (`teacher_outputs`)."""
    arrays = fixture()
    objective, params = fixture_objective(arrays, feature_loss=kind)
    batch = fixture_batch(arrays)
    ids = jnp.asarray(arrays["tokens"][:, :-1])

    @jax.jit
    def dew(params):
        return jax.grad(lambda trainable: objective.scalar_loss(
            {**params, "params": trainable}, batch, step_at())[0])(params["params"])

    @as_given
    def pulled(params, dtype="float32"):
        wide = dtype == "float64"
        trainable = widened(params["params"]) if wide else params["params"]
        outputs = student_forward(wide=wide)(trainable, ids)
        teacher = teacher_outputs(arrays, params[TEACHER])
        _, _, cotangents = reference_run(arrays, kind, outputs, teacher, dtype, cotangents=True)
        return student_pullback(wide=wide)(trainable, ids, tuple(jnp.asarray(value) for value in cotangents))

    assert_as_exact(dew, pulled, drawn_parameters(params), f"{kind} student gradient",
                    seeds=DISTILLATION_SEEDS["gradient"],
                    truth_fn=as_given(lambda params: pulled(params, "float64")))
    # The fixture's own draw: the student's logits as near its float64 forward as the fixture's,
    # which the reference generator wrote from the same forward.
    outputs = student_forward(wide=False)(params["params"], ids)
    with jax.enable_x64(new_val=True):
        wide_logits = np.asarray(student_forward(wide=True)(widened(params["params"]), ids)[0])
    assert_as_exact_as_the_reference(np.asarray(outputs[0]), arrays["student_logits"], wide_logits, "logits")
    truth = pulled(params)
    assert jax.tree.structure(dew(params)) == jax.tree.structure(truth)
    assert all(float(np.abs(leaf).max()) > 0 for leaf in jax.tree.leaves(truth))


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
    at_start = float(scheduled.scalar_loss(params, batch, step_at(0))[0])
    at_end = float(scheduled.scalar_loss(params, batch, step_at(100))[0])
    between = float(scheduled.scalar_loss(params, batch, step_at(50))[0])

    first, _ = fixture_objective(arrays, alpha=0.6)
    last, _ = fixture_objective(arrays, alpha=0.1)
    assert at_start == pytest.approx(float(first.scalar_loss(params, batch, step_at(0))[0]), rel=1e-6)
    assert at_end == pytest.approx(float(last.scalar_loss(params, batch, step_at(0))[0]), rel=1e-6)
    assert min(at_start, at_end) < between < max(at_start, at_end)


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
    with_feature, aux = objective.scalar_loss(params, batch, step_at())
    without, _ = DistillationObjective(student, teacher).scalar_loss(params, batch, step_at())
    assert float(aux.metrics["distill/feature"]) > 0
    assert float(with_feature) == pytest.approx(
        float(without) + float(aux.metrics["distill/feature"]), rel=1e-6
    )


def test_a_validation_pass_scores_with_the_teacher_the_trainer_substituted():
    """A validation loss is compiled once per set of modules the objective
    runs: once the trainer swaps the teacher (`substitute`, as remat does),
    the next pass scores with the new one, as an objective built over it
    does, and not with the program compiled for the old one."""
    from dew.training import Evaluation

    arrays = fixture()
    objective, params = fixture_objective(arrays, features=[], beta=0.0)
    tokens = np.asarray(arrays["tokens"])
    # One row per device of the lane's eight.
    batch = {TEXT_KEY: np.resize(tokens, (8, *tokens.shape[1:]))}

    def scored(held):
        return Evaluation.run(held, params, lambda partition: iter([batch]), key=3,
                              loss=True).scores["val/loss"]

    before = scored(objective)
    softened = [program.module.clone(final_logit_softcap=2.0) if not program.trained else program.module
                for program in objective.program_key()]
    objective.substitute(softened)
    built, _ = fixture_objective(arrays, features=[], beta=0.0)
    built.substitute(softened)
    after = scored(objective)

    assert after != before
    assert after == scored(built)


def test_a_distillation_step_fuses_no_head_and_has_none_to_unfuse():
    """The step reads the student's whole logits (`predict`), so a tiled student head is no fused
    head: the program says so and the fit ladder has no rung to unfuse."""
    student, teacher = (LMObjective(model, SEQ, ema_decay=None, head_tile=(4, 16)) for model in models())
    objective = DistillationObjective(student, teacher)
    assert student.program_key()[0].head_fused
    assert not any(program.head_fused for program in objective.program_key())
    assert objective.unfuse_head() is None


def test_a_layer_the_model_does_not_have_is_named_at_init():
    student = LMObjective(CausalTransformer(**META["student"]), SEQ, ema_decay=None)
    teacher = LMObjective(CausalTransformer(**META["teacher"]), SEQ, ema_decay=None)
    with pytest.raises(ValueError):
        DistillationObjective(student, teacher, features=[(0, 5)]).init(jax.random.key(0))


def test_beta_without_pairs_and_weights_outside_their_range_are_refused():
    student_model, teacher_model = models()
    student = LMObjective(student_model, SEQ, ema_decay=None)
    teacher = LMObjective(teacher_model, SEQ, ema_decay=None)
    with pytest.raises(ValueError):
        DistillationObjective(student, teacher, beta=0.5)
    with pytest.raises(ValueError):
        DistillationObjective(student, teacher, alpha=1.5)
    with pytest.raises(ValueError):
        DistillationObjective(student, teacher, temperature=0.0)


@pytest.mark.parametrize("student_vocab, teacher_vocab", [(VOCAB, VOCAB + 5), (1, 3)])
def test_a_vocabulary_mismatch_is_refused_with_the_reason(student_vocab, teacher_vocab):
    """A one-id student would broadcast against the teacher's columns into a
    wrong KL instead of failing, so the mismatch is refused by name."""
    student = LMObjective(CausalTransformer(**{**META["student"], "vocab_size": student_vocab}), SEQ,
                          ema_decay=None, head_chunks=1)
    teacher = LMObjective(CausalTransformer(**{**META["teacher"], "vocab_size": teacher_vocab}), SEQ,
                          ema_decay=None, head_chunks=1)
    objective = DistillationObjective(student, teacher)
    params = objective.init(jax.random.key(0))
    tokens = jnp.zeros_like(fixture_batch(fixture())[TEXT_KEY])
    with pytest.raises(ValueError):
        objective.scalar_loss(params, {TEXT_KEY: tokens}, step_at())


def test_a_student_with_the_router_balance_loss_is_refused():
    """The balance loss's router terms normalize per router, not per token,
    so the seam refuses them at the first trace and names the way out."""
    arrays = fixture()
    student = LMObjective(CausalTransformer(**META["student"]), SEQ, ema_decay=None,
                          balance=ExpertBalance(aux_loss_alpha=0.01))
    teacher = LMObjective(CausalTransformer(**META["teacher"]), SEQ, ema_decay=None)
    objective = DistillationObjective(student, teacher)
    with pytest.raises(ValueError):
        objective.scalar_loss(objective.init(jax.random.key(0)), fixture_batch(arrays), step_at())


# --------------------------------------------------------------------------
# Through the trainer
# --------------------------------------------------------------------------


def fixed(batch):
    """The one fixed batch, endlessly, as the trainer's dataset contract,
    declared to replay so a resumed run continues it."""
    return Loader(replayed(lambda partition: itertools.repeat(batch)), None, batch=batch[TEXT_KEY].shape[0],
                  records=None)


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
    before = float(trainer.objective.scalar_loss(initial.variables, batch, step_at())[0])

    state = trainer.fit(fixed(batch), steps=5, log_every=1)

    for path, leaf in jax.tree_util.tree_leaves_with_path(state.variables[TEACHER]):
        assert np.array_equal(np.asarray(leaf), np.asarray(dict(
            jax.tree_util.tree_leaves_with_path(initial.variables[TEACHER]))[path])), path
    for name, kernel in state.variables["params"][PROJECTIONS].items():
        assert not np.array_equal(
            np.asarray(kernel), np.asarray(initial.variables["params"][PROJECTIONS][name])
        ), name
    after = float(trainer.objective.scalar_loss(state.variables, batch, step_at())[0])
    assert after < before, (before, after)

    make_trainer(arrays, tmp_path).fit(fixed(batch), steps=3, log_every=1)
    resumed = make_trainer(arrays, tmp_path).fit(fixed(batch), steps=5, log_every=1)
    for straight, again in zip(jax.tree.leaves(state.variables),
                               jax.tree.leaves(resumed.variables), strict=True):
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
    _, aux = objective.scalar_loss(params, batch, step_at())
    loss, _ = objective.student.scalar_loss(own, batch, step_at())
    assert float(loss) == pytest.approx(float(aux.metrics["ce"]), rel=1e-6)
    assert objective.ema is None


def test_a_saved_distillation_loads_as_its_student_and_reads_the_students_data(tmp_path):
    """A distillation reads its batches through the student's data steps, so a token budget the
    student's loader takes is one it takes too, and a saved run of it loads (`dew.pipeline`) as the
    student's task, generating what the student's own task over the trained student tree does."""
    from token_support import write_token_corpus

    import dew
    from dew.data import presets
    from dew.data.compile import compile
    from dew.data.sources.tokens import TokenFiles

    arrays = fixture()
    objective, _ = fixture_objective(arrays)
    corpus = write_token_corpus(tmp_path / "corpus", np.arange(4 * SEQ * 8, dtype=np.uint16) % 50)
    windows = TokenFiles(corpus.train).pipe(presets.token_windows, seq_len=SEQ, batch=None)
    loaded = [compile(windows.batch(tokens=2 * SEQ), read="memory", objective=chosen.data())
              for chosen in (objective.student, objective)]
    batches = [next(data.train(DataPartition())) for data in loaded]
    np.testing.assert_array_equal(batches[0][TEXT_KEY], batches[1][TEXT_KEY])
    # The fixture's two rows on every data-parallel device.
    batch = {TEXT_KEY: jnp.tile(fixture_batch(arrays)[TEXT_KEY], (max(1, jax.device_count() // 2), 1))}
    state = Trainer(objective, optax.adam(3e-3), key=0, checkpoints=Checkpoints(str(tmp_path / "run"))).fit(
        fixed(batch), steps=2, log_every=1, checkpoint_every=2)
    task = dew.pipeline(str(tmp_path / "run"))
    own = objective.student.build_task(objective.student_variables(state.variables))
    prompt = np.asarray(batch[TEXT_KEY])[:1, :4]
    np.testing.assert_array_equal(task(prompt, 3, key=0).tokens, own(prompt, 3, key=0).tokens)
    # The load restores the student alone, onto a mesh as on the host, its projections packed as a
    # loaded task holds them: the teacher and the distillation's projections stay unread.
    from dew.inference import TextGeneration
    from dew.inference.projections import inference_projections

    student = jax.tree.structure(own.bind(inference_projections(own.model, own.variables)).variables)
    for loaded in (task, TextGeneration.from_run(str(tmp_path / "run"), mesh=MeshSpec())):
        assert jax.tree.structure(loaded.variables) == student


def test_a_distillation_trains_for_a_count_of_its_students_tokens():
    """A run's length in tokens (`fit(tokens=)`) counts the student's, the batches both models
    score: the run stops at the first step that reaches the count, as the student's own run does."""
    arrays = fixture()
    objective, _ = fixture_objective(arrays)
    # The fixture's two rows on every data-parallel device.
    batch = {TEXT_KEY: jnp.tile(fixture_batch(arrays)[TEXT_KEY], (max(1, jax.device_count() // 2), 1))}
    tokens = int(np.sum(objective.student.scored_tokens(batch)))
    distilled = Trainer(objective, optax.sgd(0.0), key=0).fit(fixed(batch), tokens=3 * tokens, log_every=1)
    alone = Trainer(objective.student, optax.sgd(0.0), key=0).fit(fixed(batch), tokens=3 * tokens,
                                                                  log_every=1)
    assert int(distilled.step) == int(alone.step) == 3


def test_selected_distillation_weights_each_sampled_tokens_score_by_its_reverse_kl_reward():
    """On-policy distillation at the tokens the batch holds (`divergence='selected'`, alpha 1): its
    gradient is the score function of each sampled token weighted by its log-ratio
    log pi_s(y) - log pi_t(y), the teacher frozen, as the hand formula over the two models' own
    log-likelihoods gives it, bit for bit; the reported estimate is that log-ratio's mean."""
    arrays = fixture()
    objective, params = fixture_objective(arrays, alpha=1.0, temperature=1.0, beta=0.0, features=[],
                                          divergence="selected")
    batch = fixture_batch(arrays)
    student, teacher = objective.student, objective.teacher

    def by_hand(student_params):
        _, _, own = student.predict(objective.student_variables({**params, "params": student_params}), batch,
                                    step_at(), train=False)
        _, _, held = teacher.predict(params[TEACHER], batch, step_at(), train=False)
        ratio = jax.lax.stop_gradient(held.losses - own.losses)
        weights = own.weights.astype(jnp.float32)
        return jnp.sum(ratio * -own.losses * weights) / jnp.sum(weights)

    found = jax.grad(lambda p: objective.scalar_loss({**params, "params": p}, batch, step_at())[0])(
        params["params"])
    expected = jax.grad(by_hand)(params["params"])
    jax.tree.map(np.testing.assert_array_equal, objective.student_variables({"params": found})["params"],
                 objective.student_variables({"params": expected})["params"])
    with pytest.raises(ValueError):
        DistillationObjective(student, teacher, divergence="selected", temperature=2.0)


@pytest.mark.parametrize("scores", [True, False], ids=["scores and states", "states alone"])
def test_a_teacher_position_given_non_finitely_is_counted_and_never_trained_on(scores):
    """A teacher whose paired layers' states, and its scores too or not, are NaN at one position (row
    0's last) leaves that position out of every distillation term, the soft one and the features': the
    loss and the student's gradient (its projections' included) equal, bit for bit, those the intact
    teacher gives with that position's label padded out, and the excluded weight is reported."""
    arrays = fixture()
    objective, params = fixture_objective(arrays, alpha=1.0)
    batch = fixture_batch(arrays)
    padded = {TEXT_KEY: batch[TEXT_KEY].at[0, SEQ].set(META["pad_id"])}
    original = objective.teacher.predict

    def broken(variables, held, step, **kwargs):
        statistics, aux, prediction = original(variables, held, step, **kwargs)

        def failed(value):
            return value.at[0, SEQ - 1].set(jnp.nan)

        hidden = tuple(failed(state) for state in prediction.hidden)
        if not scores:
            return statistics, aux, dataclasses.replace(prediction, hidden=hidden)
        return statistics, aux, dataclasses.replace(
            prediction, logits=failed(prediction.logits), losses=failed(prediction.losses), hidden=hidden)

    def distilled(batch):
        def total(trained):
            loss, aux = objective.loss({**params, "params": trained}, batch, step_at())
            return loss.total, aux.metrics

        (loss, metrics), gradient = jax.value_and_grad(total, has_aux=True)(params["params"])
        return loss, metrics["distill/teacher_nonfinite"], gradient

    intact = distilled(padded)
    objective.teacher.predict = broken
    failed = distilled(batch)
    assert len(objective.features) == 2
    assert (float(intact[1]), float(failed[1])) == (0.0, 1.0)
    np.testing.assert_array_equal(failed[0], intact[0])
    jax.tree.map(np.testing.assert_array_equal, failed[2], intact[2])


def test_a_teacher_with_another_tokenizer_of_the_same_size_is_refused(tmp_path):
    """Two tokenizers whose vocabularies are the same size but map two strings to each other's ids are
    refused: the term compares one id's probabilities on both sides. The same tokenizer twice is not."""
    import shutil

    student, teacher = (LMObjective(model, SEQ, ema_decay=None) for model in models())
    chat = Path(__file__).resolve().parent / "fixtures" / "tokenizers" / "tiny-chat"
    swapped = tmp_path / "swapped"
    shutil.copytree(chat, swapped)
    spec = json.loads((swapped / "tokenizer.json").read_text())
    vocabulary = spec["model"]["vocab"]
    first, second = sorted(vocabulary, key=vocabulary.get)[100:102]
    vocabulary[first], vocabulary[second] = vocabulary[second], vocabulary[first]
    (swapped / "tokenizer.json").write_text(json.dumps(spec))
    DistillationObjective(student, teacher, tokenizers=(str(chat), str(chat)))
    with pytest.raises(ValueError):
        DistillationObjective(student, teacher, tokenizers=(str(chat), str(swapped)))
