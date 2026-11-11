"""Training a looped model at every pass count at once (`pass_weights`, ledger 5480/5481) and its pass count
grown phase by phase (`Trainer`'s `phase_passes`): each loss and gradient is the passes' own models' at
their weights, on par from float64 (check 1), and a phased run steps each phase at its own count."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from decision_support import decision_rows
from flax import linen as nn
from model_support import float64_twin
from reference_error import assert_as_exact

from dew.batch import PHASE, Phase
from dew.data.dataset import DataPartition
from dew.data.text import ByteTokenizer
from dew.decision import Choice, DecisionObjective, Example, Specials, StateFirstLayout
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.protocols import EXITS
from dew.objectives.base import Step
from dew.objectives.lm.objective import LMObjective
from dew.objectives.passes import PassWeights, exiting

PASS_SEEDS = 64
"""Draws of each check-1 site here: tiny models, whose losses' errors settle well inside it."""

STEPS = 3
"""Passes the looped block runs."""

INTENT = Choice("Which team?", {"billing": "payments", "technical": "outages", "sales": "pricing"})
EXAMPLES = [Example(state, {"team": INTENT}, {"team": label})
            for state, label in (("charged twice", "billing"), ("site is down", "technical"),
                                 ("how much is pro", "sales"), ("refund please", "billing"),
                                 ("error 500", "technical"), ("upgrade cost", "sales"),
                                 ("double charge", "billing"), ("cannot log in", "technical"))]


def looped(implicit: bool = False, steps: int = STEPS) -> CausalTransformer:
    """A tiny decoder whose middle layer runs `steps` passes before its last layer, a coda every pass
    exits through; `implicit` takes each later pass by the θ-method (`Loop.implicit`), which reads each
    pass's state un-normed (no `step_norm`)."""
    return CausalTransformer(vocab_size=256, emb_features=32, num_layers=3, num_heads=2, mlp_features=48,
                             max_seq_len=128, attention_impl="xla",
                             loop=Loop(steps, layers=(1, 2), implicit=implicit, step_norm=not implicit))


def lm_loss(model, weights, tokens):
    """The loss and gradient of the LM objective over `model` with `weights`, `tokens` its one batch."""
    objective = LMObjective(model, tokens.shape[1] - 1, pass_weights=weights)

    def run(variables):
        def loss(params):
            return objective.scalar_loss({**variables, "params": params}, {"text": tokens},
                                         Step(jnp.asarray(0), jax.random.key(0), None))[0]
        return jax.value_and_grad(loss)(variables["params"])
    return run


def by_passes(model, weights, tokens):
    """The same loss as each pass count's model's own (`with_passes(k)`) at its weight, every `k`, its
    gradient theirs alike: what pass weights mean."""
    def run(variables):
        found = [lm_loss(model.with_passes(k, exits=False), None, tokens)(variables)
                 for k in range(1, len(weights) + 1)]
        loss = sum(weight * value for weight, (value, _) in zip(weights, found, strict=True))
        grad = jax.tree.map(lambda *leaves: sum(w * leaf for w, leaf in zip(weights, leaves, strict=True)),
                            *(gradient for _, gradient in found))
        return loss, grad
    return run


@pytest.mark.parametrize("implicit", [False, True], ids=["explicit", "theta"])
@pytest.mark.parametrize("weights", ["mean", (0.1, 0.3, 0.6)], ids=["mean", "stated"])
def test_an_lm_loss_weighing_its_passes_is_each_pass_models_loss_at_its_weight(weights, implicit):
    """`pass_weights` over a looped decoder whose block a coda follows, each pass leaving through it as the
    last does, under the explicit step and the θ-method's: the loss and every parameter's gradient (θ's
    among them) on par from float64 with each `with_passes(k)` model's own loss at its weight, the passes'
    own models run apart (check 1)."""
    model = looped(implicit)
    stated = PassWeights.of(weights).over(STEPS)
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, 256, (2, 17)), jnp.int32)
    init = jax.jit(model.init)

    def draw(seed):
        return (nn.unbox(init(jax.random.key(seed), tokens[:, :-1])),)

    label = f"pass-weighed LM loss ({weights}, {'θ' if implicit else 'explicit'})"
    assert_as_exact(jax.jit(lm_loss(model, weights, tokens)), jax.jit(by_passes(model, stated, tokens)), draw,
                    label, seeds=PASS_SEEDS, truth_fn=jax.jit(by_passes(float64_twin(model), stated, tokens)))


def test_weights_on_the_last_pass_alone_are_the_models_own_loss():
    """Weights all on the last pass score its output alone, the model's own loss: the same program as
    no pass weights, so its loss and gradient are the same bits."""
    model = looped()
    tokens = jnp.asarray(np.random.default_rng(1).integers(0, 256, (2, 17)), jnp.int32)
    variables = nn.unbox(jax.jit(model.init)(jax.random.key(0), tokens[:, :-1]))
    last = jax.jit(lm_loss(model, (0.0, 0.0, 1.0), tokens))(variables)
    plain = jax.jit(lm_loss(model, None, tokens))(variables)
    assert jax.tree.all(jax.tree.map(lambda a, b: bool(np.array_equal(a, b)), last, plain))


SPECIALS = Specials(begin=None, separator=10, marker=0, marker_text="\x00", pad=255)


def decision_objective(model, weights):
    """A decision objective over the looped `model` with `weights`, its rows the eight intents'."""
    layout = StateFirstLayout(max_len=96, head_max_len=48, option_tokens=8)
    return DecisionObjective(model, tokenizer=ByteTokenizer(), specials=SPECIALS, layout=layout,
                             shuffle_options=False, pass_weights=weights)


def test_a_decision_loss_weighing_its_passes_is_each_pass_models_loss_at_its_weight():
    """`pass_weights` over a decision model's looped backbone, a consistency term beside the label loss
    over a group of two questions, one negated (`test_decision_consistency.NEGATED`), whose divergence is
    nonzero: the loss and every parameter's gradient on par from float64 with each `with_passes(k)`
    backbone's own decision loss at its weight (check 1)."""
    from test_decision_consistency import LAYOUT, NEGATED

    from dew.decision import JointSchemaHead

    model = looped().clone(max_seq_len=LAYOUT.max_len)
    weights = (0.2, 0.3, 0.5)
    head = JointSchemaHead(32, width=16, routing_layers=1, layers=1, heads=2, feedforward=32)

    def objective_of(backbone, stated, head=head):
        return DecisionObjective(backbone, tokenizer=ByteTokenizer(), specials=SPECIALS, layout=LAYOUT,
                                 head=head, consistency=0.5, pass_weights=stated)

    weighed = objective_of(model, weights)
    reader = decision_rows([NEGATED, NEGATED], None, objective=weighed.data(), tokens=LAYOUT.max_len)
    [batch] = list(itertools.islice(reader.train(DataPartition()), 1))
    step = Step(jnp.asarray(0), jax.random.key(0), None)
    init = jax.jit(weighed.init)
    stats, _ = weighed.loss(nn.unbox(init(jax.random.key(0))), batch, step)
    assert float(stats.terms[1].total) > 0.0, "the group's divergence has support"

    def loss_of(objective):
        def run(variables):
            return jax.value_and_grad(lambda params: objective.scalar_loss(
                {**variables, "params": params}, batch, step)[0])(variables["params"])
        return run

    def apart(backbone, head=head):
        def run(variables):
            found = [loss_of(objective_of(backbone.with_passes(k, exits=False), None, head))(variables)
                     for k in range(1, STEPS + 1)]
            loss = sum(weight * value for weight, (value, _) in zip(weights, found, strict=True))
            grad = jax.tree.map(
                lambda *leaves: sum(w * leaf for w, leaf in zip(weights, leaves, strict=True)),
                *(gradient for _, gradient in found))
            return loss, grad
        return run

    def draw(seed):
        return (nn.unbox(init(jax.random.key(seed))),)

    assert_as_exact(jax.jit(loss_of(weighed)), jax.jit(apart(model)), draw, "pass-weighed decision loss",
                    seeds=PASS_SEEDS, truth_fn=jax.jit(apart(float64_twin(model), float64_twin(head))))


def test_a_validation_reports_each_pass_beside_the_run():
    """A pass-weighed run's validation reports each pass's metrics and loss under `val/pass_<k>/` beside the
    run's own, whose metrics stay the last pass's: the last pass's accuracy is the run's, and each pass's
    the accuracy of that pass's own model (`with_passes(k)`). Each pass's loss is the training loss's
    term, which `test_a_decision_loss_weighing_its_passes_is_each_pass_models_loss_at_its_weight` holds."""
    from dew.decision import Accuracy
    from dew.training.evaluation import Evaluation

    model = looped()
    objective = decision_objective(model, "mean")
    data = decision_rows(EXAMPLES, 8, EXAMPLES, objective=objective.data())
    variables = objective.init(jax.random.key(0))
    metrics = (Accuracy(),)
    scores = Evaluation.run(objective, variables, data.val, key=0, metrics=metrics, loss=True).scores
    assert {f"val/pass_{k}/loss" for k in range(1, STEPS + 1)} <= set(scores)
    assert {f"val/pass_{k}/accuracy" for k in range(1, STEPS + 1)} <= set(scores)
    assert scores[f"val/pass_{STEPS}/accuracy"] == scores["val/accuracy"]
    for k in range(1, STEPS + 1):
        own = decision_objective(model.with_passes(k, exits=False), None)
        alone = Evaluation.run(own, variables, data.val, key=0, metrics=metrics, loss=True).scores
        assert scores[f"val/pass_{k}/accuracy"] == alone["val/accuracy"]


@pytest.mark.parametrize("stated", [(0.5, 0.6, 0.0), (0.5, 0.5), "max", (-0.1, 0.5, 0.6)],
                         ids=["over one", "two of three passes", "no rule", "negative"])
def test_pass_weights_that_mean_no_mixture_of_the_loops_passes_are_refused(stated):
    """Weights that do not sum to one, name another count of passes than the loop runs, name no rule or
    weigh a pass below zero are refused when the objective is built."""
    with pytest.raises(ValueError):
        LMObjective(looped(), 16, pass_weights=stated)


def test_pass_weights_over_a_model_with_no_loop_are_refused():
    """A model that runs no loop has one pass, its output: pass weights over it are refused, naming it."""
    plain = CausalTransformer(vocab_size=256, emb_features=32, num_layers=2, num_heads=2, mlp_features=48,
                              max_seq_len=128, attention_impl="xla")
    with pytest.raises(ValueError):
        LMObjective(plain, 16, pass_weights="mean")


def test_a_phased_run_steps_each_phase_at_its_own_pass_count():
    """A run whose loop runs 1 pass in its first phase and 3 in its second (`phase_passes`) steps each phase
    as a run of the model with that many passes steps it: from the same state over the same batch, its loss
    and its stepped parameters the same bits as that run's, each phase its own program over the one state;
    between steps the objective holds the last phase's model, the one the run hands on and validates."""
    from dew.training import Layout, Trainer

    model = looped()
    tokens = jnp.asarray(np.random.default_rng(2).integers(0, 256, (jax.device_count() * 2, 17)), jnp.int32)
    batch = {"text": tokens}

    def trainer_of(model, **phased):
        return Trainer(LMObjective(model, 16), optax.sgd(0.1), key=jax.random.key(0),
                       layout=Layout(min_shard=1, tolerance=1.0), **phased)

    phased = trainer_of(model, phase_passes=(1, STEPS))
    for phase, passes in enumerate((1, STEPS)):
        # The phase the batch was read in, as a phased stream marks it (`dew.batch.PHASE`).
        assert phased.phase_passes.at({**batch, PHASE: Phase(phase)}) == passes
        state, _, _ = phased.place()
        stepped, loss, *_ = phased.compile(state, batch)(state, batch)
        assert phased.objective.model.loop.steps == STEPS
        alone = trainer_of(model.with_passes(passes))
        start, _, _ = alone.place()
        own, own_loss, *_ = alone.compile(start, batch)(start, batch)
        assert np.array_equal(np.asarray(loss), np.asarray(own_loss))
        assert jax.tree.all(jax.tree.map(lambda a, b: bool(np.array_equal(a, b)),
                                         jax.device_get(stepped.variables), jax.device_get(own.variables)))


def cut_loop_loss(model, weights, tokens):
    """The stated program of pass weights under a cut: the cut loop's every exit, one forward, each through
    the head as its last pass's output is, their cross entropies weighed before the mean; the loss and its
    gradient."""
    passing = exiting(model)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]

    def run(variables):
        def loss(params):
            held = {**variables, "params": params}
            hidden, sown = passing.apply(held, inputs, method="hidden_states", mutable=[EXITS])
            earlier, _ = passing.exit_passes(sown)
            terms = [optax.softmax_cross_entropy_with_integer_labels(
                passing.apply(held, states, method="logits_from_hidden"), targets)
                for states in (*earlier, hidden)]
            return jnp.mean(sum(weight * term for weight, term in zip(weights, terms, strict=True)))
        return jax.value_and_grad(loss)(variables["params"])
    return run


def test_an_lm_loss_weighing_a_cut_loops_passes_is_its_exits_loss_at_their_weights():
    """Under a cut (`backprop_steps=1` of 3 passes) the passes' terms are the cut loop's exits, one forward,
    a held-back pass's exit training its coda and the head alone: the loss and every parameter's gradient on
    par from float64 with that stated program, the exits read off the loop's `exits` (check 1)."""
    model = looped().clone(loop=Loop(STEPS, layers=(1, 2), backprop_steps=1))
    weights = (0.2, 0.3, 0.5)
    tokens = jnp.asarray(np.random.default_rng(3).integers(0, 256, (2, 17)), jnp.int32)
    init = jax.jit(model.init)

    def draw(seed):
        return (nn.unbox(init(jax.random.key(seed), tokens[:, :-1])),)

    assert_as_exact(jax.jit(lm_loss(model, weights, tokens)), jax.jit(cut_loop_loss(model, weights, tokens)),
                    draw, "pass-weighed LM loss under a cut", seeds=PASS_SEEDS,
                    truth_fn=jax.jit(cut_loop_loss(float64_twin(model), weights, tokens)))


def test_a_pass_count_a_phase_whose_last_is_not_the_models_is_refused():
    """The last phase's count is the one the model holds and the run hands on: another is refused."""
    from dew.training import Trainer

    with pytest.raises(ValueError):
        Trainer(LMObjective(looped(), 16), optax.sgd(0.1), key=jax.random.key(0), phase_passes=(1, 2))


def test_a_phases_objective_is_its_own_whatever_objectives_it_holds():
    """A distillation of a looped student (its student and teacher objectives inside it): the objective a
    first phase's step is built from runs its student one pass while the run's own, which validation and
    the handed-on model read, still runs three, neither sharing the other's student or teacher."""
    from dew.objectives.distillation import DistillationObjective
    from dew.training.phase_passes import PhasePasses

    teacher = CausalTransformer(vocab_size=256, emb_features=32, num_layers=2, num_heads=2, mlp_features=48,
                                max_seq_len=128, attention_impl="xla")
    run = DistillationObjective(LMObjective(looped(), 16, ema_decay=None),
                                LMObjective(teacher, 16, ema_decay=None), alpha=0.5, temperature=1.0)
    phases = PhasePasses.of(run, (1, STEPS))
    first = phases.objective_for(1)
    assert first.student.model.loop.steps == 1 and run.student.model.loop.steps == STEPS
    assert first.student is not run.student and first.teacher is not run.teacher
    assert phases.objective_for(1) is first


def corpus(tag: int):
    """Thirty-two rows of 17 tokens, read on in batches of eight, each `tag`'s own."""
    from dew.data import Memory

    rows = [{"text": (np.arange(17, dtype=np.int32) * (tag + 3) + index) % 256} for index in range(32)]
    return Memory(rows).repeat().batch(8)


def remade(state, batch, key):
    """A rollout that makes its batch afresh of the rows it reads, as a sampler's does: none of the read's
    own fields (its phase among them) on what it makes."""
    return {"text": np.asarray(batch["text"])}


@pytest.mark.parametrize("rollout", [None, remade], ids=["read", "rollout"])
def test_a_phased_fit_steps_each_phase_as_a_run_of_its_own_pass_count_does(tmp_path, rollout):
    """A fit over two phases of data on the step clock, its loop one pass in the first (two steps) and
    three in the second (`phase_passes`), each step's count read off its batch's phase: its parameters
    after four steps are the same bits as a one-pass run's two steps resumed by a three-pass run's two. A
    rollout's batch, made afresh of the read's rows, is stepped at the phase its read was (`carried`)."""
    import optax
    from test_data_mixtures import READ, assert_same, parameters

    from dew.data import phases
    from dew.data.compile import compile
    from dew.training import Checkpoints, Layout, Trainer

    data = compile(phases((corpus(1), 2), (corpus(2), None)), read="indexed", loading=READ)
    model = looped()

    def trainer_of(model, directory, **phased):
        return Trainer(LMObjective(model, 16), optax.sgd(0.1), key=jax.random.key(0),
                       layout=Layout(min_shard=1, tolerance=1.0), rollout=rollout,
                       checkpoints=Checkpoints(str(directory), keep=8), **phased)

    whole = trainer_of(model, tmp_path / "phased", phase_passes=(1, STEPS)).fit(
        data, steps=4, log_every=100, checkpoint_every=None)
    trainer_of(model.with_passes(1), tmp_path / "apart").fit(data, steps=2, log_every=100, checkpoint_every=1)
    apart = trainer_of(model, tmp_path / "apart").fit(data, steps=4, log_every=100, checkpoint_every=1)
    assert int(apart.step) == 4
    assert_same(parameters(whole), parameters(apart), "the phased fit is not its phases' runs")


class Flops:
    """A tracker of each logged step's `train/flops`."""

    def __init__(self):
        self.at: dict[int, float] = {}

    def log(self, scalars, step: int) -> None:
        if "train/flops" in scalars:
            self.at[step] = scalars["train/flops"]

    def artifact(self, value, step: int) -> None:
        pass

    def close(self) -> None:
        pass


def test_a_phased_runs_work_counts_each_step_at_its_phases_pass_count():
    """A fit over two phases, its loop one pass in the first and three in the second (`phase_passes`),
    counts each step's work (`train/flops`, which train/mfu divides) as a run of its phase's pass count
    counts the same step: the first phase's at one pass, not the model the run hands on."""
    import optax
    from test_data_mixtures import READ

    from dew.data import phases
    from dew.data.compile import compile
    from dew.training import Layout, Trainer

    def counted(model, data, steps, **phased):
        flops = Flops()
        Trainer(LMObjective(model, 16), optax.sgd(0.1), key=jax.random.key(0), tracker=flops,
                layout=Layout(min_shard=1, tolerance=1.0), **phased).fit(
            compile(data, read="indexed", loading=READ), steps=steps, log_every=1, checkpoint_every=None)
        return flops.at

    phased = counted(looped(), phases((corpus(1), 2), (corpus(2), None)), 4, phase_passes=(1, STEPS))
    one = counted(looped().with_passes(1), corpus(1), 1)
    three = counted(looped(), corpus(2), 1)
    assert one[1] < three[1]
    assert [phased[step] for step in (1, 2, 3, 4)] == [one[1], one[1], three[1], three[1]]
