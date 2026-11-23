"""A looped model's pass count drawn per microbatch from a bounded law
(`dew.training.phase_passes.SampledPasses`, row 7660)."""

import dataclasses
import math
import time

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from generation_support import lowers
from model_support import float64_twin
from reference_error import assert_as_exact, dew_path
from scipy import integrate, optimize, special, stats
from test_pass_weights import PASS_SEEDS, drawn, looped, phased_trainer

from dew.batch import LOOP_PASSES
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.protocols import EXITS
from dew.objectives.base import Ratio, Step
from dew.objectives.lm.objective import LMObjective
from dew.optim import MuonClip
from dew.training import FitPolicy, UpdateBatch
from dew.training.compiler import DrawingStep
from dew.training.interval import DISPATCHED_MICROBATCHES, Interval
from dew.training.phase_passes import PhasePasses, SampledPasses

LAWS = [SampledPasses(mean=2.0, dispersion=0.5, cap=4), SampledPasses(mean=4.0, dispersion=0.5, cap=8),
        SampledPasses(mean=3.0, dispersion=0.0, cap=6), SampledPasses(mean=32.0, dispersion=0.0, cap=3),
        SampledPasses(mean=32.0, dispersion=1.0, cap=3), SampledPasses(mean=1e6, dispersion=0.1, cap=2)]
"""The note's law (mean 2, dispersion 0.5, cap 4), a wider one, a Poisson count (no dispersion), and two
whose cap holds little of their unbounded mass (a Poisson count of mean 32 is at most 3 with chance
~2e-11: a count redrawn past the cap would be drawn ~5e10 times), and one whose counts' mass lies where its
rate is drawn far below its mean (each count's mode near a log rate of 6.6, its mean's 13.8, 47 of its
deviations away)."""

DRAWS = 2_500
"""The microbatches a seed's key draws a law's counts at."""

SEEDS = 8
"""The keys a law's shares are read over (`sampling_gate.assert_draws_as_declared`)."""


SPAN = 40
"""The standard deviations either side of a count's mode past which its integrand over the log rate,
log-concave and at least as narrow as the log rate's normal density, is 0 beside its peak in float64."""


def probabilities(law: SampledPasses) -> np.ndarray:
    """The bounded law's probability at each count 1 to its cap, from its statement alone: the Poisson's
    closed form where it has no dispersion, its log probabilities normalized in log space; else the Poisson's
    probability integrated over the log-normal rate (scipy's adaptive quadrature in float64), each count's
    integrand about its own mode (scipy's root of its log's derivative) scaled by its peak, over `SPAN` of
    the log rate's deviations either side; conditioned on the cap."""
    location = math.log(law.mean - 1) - law.dispersion ** 2 / 2
    if law.dispersion == 0:
        logs = stats.poisson.logpmf(np.arange(law.cap), math.exp(location))
    else:
        logs = []
        for n in range(law.cap):
            # A rate past float64's range is a probability of 0 at every count, as the largest rate's is.
            def log_integrand(x, n=n):
                return (n * x - math.exp(min(x, 709.0)) - special.gammaln(n + 1)
                        - ((x - location) / law.dispersion) ** 2 / 2)

            def slope(x, n=n):
                return n - math.exp(min(x, 709.0)) - (x - location) / law.dispersion ** 2

            reach = SPAN * law.dispersion
            mode = optimize.brentq(slope, min(location, math.log(max(n, 1))) - reach,
                                   max(location, math.log(max(n, 1))) + reach, xtol=1e-15)
            peak = log_integrand(mode)
            # Breakpoints a peak's width apart near it, and doubling out to the span below it, so the
            # quadrature samples a peak far narrower than the span.
            width = 1 / math.sqrt(math.exp(mode) + law.dispersion ** -2)
            points = sorted({mode + step * width for step in range(-60, 61)}
                            | {mode - 2.0 ** power * width for power in range(6, 64)
                               if 2.0 ** power * width < reach})
            logs.append(peak + math.log(integrate.quad(
                lambda x, f=log_integrand, peak=peak: math.exp(f(x) - peak), mode - reach, mode + reach,
                points=points, limit=4000)[0]))
        logs = np.asarray(logs)
    unbounded = np.exp(logs - logs.max())
    return unbounded / unbounded.sum()


@pytest.mark.parametrize("law", LAWS, ids=lambda law: f"{law.mean}-{law.dispersion}-{law.cap}")
def test_a_sampled_pass_count_follows_its_bounded_law(law):
    """Over many microbatches' positions under each of several keys, each count from 1 to the cap, the
    cap's tail included, is drawn at the bounded law's probability, as a fair draw of it would be (the
    sampling gate); no count is past the cap or below one."""
    from sampling_gate import assert_draws_as_declared

    def counts_of(seed: int) -> np.ndarray:
        drawn = np.array([law.count((seed, 7), position) for position in range(DRAWS)])
        assert drawn.min() >= 1 and drawn.max() <= law.cap
        return np.bincount(drawn - 1, minlength=law.cap)

    assert_draws_as_declared(counts_of, probabilities(law), f"law {law}", seeds=SEEDS)


@pytest.mark.parametrize(("law", "count"), [(SampledPasses(mean=1e20, dispersion=0.0, cap=4), 4),
                                            (SampledPasses(mean=2.0, dispersion=1e-323, cap=1), 1),
                                            (SampledPasses(mean=2.0, dispersion=1e154, cap=3), 1),
                                            (SampledPasses(mean=2.0, dispersion=1e300, cap=3), 1)],
                         ids=["vast-mean", "subnormal-dispersion", "vast-dispersion",
                              "unsquarable-dispersion"])
def test_a_sampled_pass_count_at_its_laws_extremes_draws_its_one_count(law, count):
    """A law whose every draw is one count: a Poisson count of mean 1e20, below 4 with chance ~1e-60, past
    any uniform's least step, draws its cap (its counts' common term, rounded alike, read as four equal ones);
    a cap of one under a dispersion too small to step through draws 1 (which divided by a step of 0); and a
    dispersion so vast its rate is past its least value but with chance exp(-dispersion² / 8), 0 in float64,
    draws 1 too, where its peak's width (its square overflowing) or its log rate's mean was out of range."""
    assert {law.count((0, 7), position) for position in range(DRAWS)} == {count}


def test_a_sampled_pass_count_of_a_vanishing_dispersion_is_its_means_poisson():
    """A law of a dispersion near 0 (1e-300) draws as the Poisson count of its mean, conditioned on its cap (a
    rate of 1: counts 1, 2 and 3 at 0.4, 0.4 and 0.2), as a fair draw of it would be (the sampling gate)."""
    from sampling_gate import assert_draws_as_declared

    law = SampledPasses(mean=2.0, dispersion=1e-300, cap=3)
    poisson = stats.poisson.pmf(np.arange(law.cap), 1.0)

    def counts_of(seed: int) -> np.ndarray:
        return np.bincount([law.count((seed, 7), position) - 1 for position in range(DRAWS)],
                           minlength=law.cap)

    assert_draws_as_declared(counts_of, poisson / poisson.sum(), f"law {law}", seeds=SEEDS)


def test_a_broad_laws_draws_are_its_statements():
    """A law whose counts' integrands over the log rate are far narrower than the rate's spread (mean e^450,
    dispersion 30, cap 256: count 200's peak 0.0024 of a deviation wide) draws, at each position, the count
    its statement's cumulative (`probabilities`, scipy's quadrature about each count's mode in the log rate
    itself) gives at the draw's own uniform (`SampledPasses.count`'s Philox). The two cumulatives agree to
    ~1e-12, so two fair readings part only where a uniform falls that near a step between counts: ~5e-6
    likely over the draws. One that missed a peak's tails (row 8274's review) moved a count's mass by 15%."""
    law = SampledPasses(mean=math.exp(450), dispersion=30.0, cap=256)
    stated = np.cumsum(probabilities(law))
    stated[-1] = 1.0
    for position in range(DRAWS):
        uniform = np.random.Generator(np.random.Philox(np.random.SeedSequence([0, 7, position]))).random()
        assert law.count((0, 7), position) == 1 + int(np.searchsorted(stated, uniform, side="right"))


def test_a_sampled_pass_count_is_its_key_and_position_alone():
    """A draw is a function of the run's key and the microbatch's global position alone: drawn again, in
    any order, it is the same count; another key draws another stream."""
    law = LAWS[0]
    positions = list(range(256))
    forward = [law.count((0, 7), position) for position in positions]
    assert [law.count((0, 7), position) for position in reversed(positions)] == forward[::-1]
    assert [law.count((0, 8), position) for position in positions] != forward


@pytest.mark.parametrize("fields", [{"mean": 1.0, "dispersion": 0.5, "cap": 4},
                                    {"mean": 2.0, "dispersion": -0.1, "cap": 4},
                                    {"mean": 2.0, "dispersion": 0.5, "cap": 0},
                                    {"mean": 2.0, "dispersion": 0.5, "cap": 4, "law": "geometric"}])
def test_a_sampled_pass_count_outside_its_law_is_refused(fields):
    """A mean at or below one count, a negative dispersion, a cap below one pass or an unnamed law
    states no bounded law, and is refused."""
    with pytest.raises(ValueError):
        SampledPasses(**fields)



CAP = 3
"""The cap of the trainer cells' law, the passes their tiny looped decoder holds
(`test_pass_weights.looped`)."""

TRAINED = SampledPasses(mean=2.0, dispersion=0.5, cap=CAP)
"""The trainer cells' law."""


def token_draw(seed: int = 2) -> np.ndarray:
    """Two rows a device of 17 tokens, int32, on the host: a run handed them places them as it steps,
    compiling nothing of its own to make them."""
    rows = jax.device_count() * 2
    return np.random.default_rng(seed).integers(0, 256, (rows, 17), dtype=np.int32)


def token_batch(seed: int = 2) -> dict:
    """`token_draw` on the default device."""
    return {"text": jnp.asarray(token_draw(seed))}


def words(key) -> tuple[int, ...]:
    """A key's words, as a draw reads them."""
    return tuple(int(word) for word in np.asarray(jax.random.key_data(key)).ravel())


WEIGHED = {"last": {}, "mean": {"pass_weights": "mean"}, "gated": {"exit_entropy": 0.1}}
"""The ways a looped decoder's loss reads its passes: its last pass's output alone, the mean of every pass's
exit, and its exit gate's distribution over them (`exit_entropy`, the model gating its exits)."""


def weighed_model(weighing: str, trained: int | None = None) -> CausalTransformer:
    """The trainer cells' decoder at its cap, through its last `trained` passes (every pass where None), each
    pass exiting where `weighing` reads every pass's exit, through a gate where it is `gated`."""
    exits = weighing != "last"
    model = looped(steps=CAP).with_passes(CAP, backprop_steps=trained, exits=exits or None,
                                          exit_gate=(weighing == "gated") or None)
    assert model is not None
    return model


def counted_gradient(objective, batch):
    """The loss and gradient of `objective` over `batch`, one batch, run at the count the call gives its rows
    (`LOOP_PASSES`, a traced input), or at its model's own where it gives None."""
    def run(variables, passes=None):
        given = batch if passes is None else {**batch, LOOP_PASSES: passes}

        def loss(params):
            return objective.scalar_loss({**variables, "params": params}, given,
                                         Step(jnp.asarray(0), jax.random.key(0), None))[0]
        return jax.value_and_grad(loss)(variables["params"])
    return run


def loss_at(model, tokens, **weighed):
    """`counted_gradient` of `model`'s LM objective (`weighed` its passes' weighing) over `tokens`."""
    return counted_gradient(LMObjective(model, tokens.shape[1] - 1, **weighed), {"text": tokens})


@pytest.mark.parametrize("count", range(1, CAP + 1))
@pytest.mark.parametrize(("weighing", "trained"),
                         [("last", None), ("last", 2), ("mean", None), ("gated", None)],
                         ids=["last", "last-two-trained", "mean", "gated"])
def test_the_caps_program_at_a_drawn_count_is_that_counts_model(weighing, trained, count):
    """The one program a sampled run steps every count by, the loop's cap's, its rows given a count
    (`LOOP_PASSES`, traced): its loss and every parameter's gradient on par from float64 with the model
    of that count's own (`with_passes`), through its last two passes where it trains through them, its
    passes weighed by its last alone, their mean and its exit gate (check 1, row 11951's design)."""
    model = weighed_model(weighing, trained)
    fixed = model.with_passes(count, backprop_steps=trained and min(trained, count))
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, 256, (2, 17)), jnp.int32)
    weighed = WEIGHED[weighing]
    passes = np.full((tokens.shape[0],), count, np.int32)
    initial = drawn(model.init, tokens[:, :-1])

    def draw(seed):
        return (*initial(seed), passes)

    def own(variables, _):
        return loss_at(fixed, tokens, **weighed)(variables)

    def own_in_float64(variables, _):
        return loss_at(float64_twin(fixed), tokens, **weighed)(variables)

    # Each count's own model is Dew's (`dew_path`): an invariance claim, held against it compiled alike.
    assert_as_exact(jax.jit(loss_at(model, tokens, **weighed)), dew_path(jax.jit(own)), draw,
                    f"the cap's program at {count} passes ({weighing}, trained {trained})", seeds=PASS_SEEDS,
                    truth_fn=jax.jit(own_in_float64))


@pytest.mark.parametrize("count", range(1, CAP + 1))
@pytest.mark.parametrize("weighing", ["mean", "gated"])
def test_a_pass_its_drawn_count_leaves_unrun_moves_no_loss_and_no_gradient(weighing, count):
    """The cap's program at a drawn count reads each pass the count leaves unrun nowhere: the loss's
    cotangent on its exit's states, its exit-gate logit and its loss terms is exactly zero, where every
    pass the count runs before its last moves the loss, its mean or its gate weighing them (row 11951's
    design). A leaked share, however small, would move it."""
    model = weighed_model(weighing)
    tokens = jnp.asarray(np.random.default_rng(0).integers(0, 256, (2, 17)), jnp.int32)
    (variables,) = drawn(model.init, tokens[:, :-1])(0)
    passes = jnp.full((tokens.shape[0],), count, jnp.int32)
    held: dict[str, tuple] = {}

    def loss(offsets):
        class Offset(LMObjective):
            """The LM objective with each earlier pass's exit states, gate logits and loss terms offset by
            `offsets`, where given, and their shapes held, as the exit objective reads them."""

            def _exit_scores(self, params, gathered, targets, last, before=None):
                exits = gathered[EXITS]
                read = {"states": tuple(exits["states"]), "gate logit": tuple(exits.get("logits", ()))}
                if offsets is not None:
                    read = {name: tuple(value + offset.astype(value.dtype)
                                        for value, offset in zip(read[name], offsets[name], strict=True))
                            for name in read}
                moved = {**exits, "states": read["states"]}
                if "logits" in exits:
                    moved["logits"] = read["gate logit"]
                terms, logits, right, ran = super()._exit_scores(params, {**gathered, EXITS: moved},
                                                                 targets, last, before)
                *earlier_terms, final = terms
                held.update({name: tuple(value.shape for value in values) for name, values in read.items()},
                            terms=tuple(term.shape for term in earlier_terms))
                if offsets is not None:
                    earlier_terms = [term + offset.astype(term.dtype)
                                     for term, offset in zip(earlier_terms, offsets["terms"], strict=True)]
                return (*earlier_terms, final), logits, right, ran

        objective = Offset(model, tokens.shape[1] - 1, **WEIGHED[weighing])
        return objective.scalar_loss(variables, {"text": tokens, LOOP_PASSES: passes},
                                     Step(jnp.asarray(0), jax.random.key(0), None))[0]

    jax.eval_shape(loss, None)
    assert len(held["states"]) == len(held["terms"]) == CAP - 1, held
    offsets = {name: tuple(jnp.zeros(shape, jnp.float32) for shape in shapes)
               for name, shapes in held.items()}
    cotangents = jax.device_get(jax.jit(jax.grad(loss))(offsets))
    moved = ("states", "terms") if weighing == "mean" else ("states", "gate logit", "terms")
    for name, values in cotangents.items():
        for index, cotangent in enumerate(values):
            reached = bool(np.any(np.asarray(cotangent) != 0))
            if index + 1 >= count:
                assert not reached, f"pass {index + 1} unrun at {count}: its {name} moved the loss"
            elif name in moved:
                assert reached, f"pass {index + 1} of {count}: its {name} moved nothing"


@pytest.mark.parametrize("count", range(1, CAP + 1))
def test_a_pass_its_drawn_count_leaves_unrun_routes_no_token_to_any_expert(count):
    """A looped mixture's routers at the cap's program, its rows given a count (`LOOP_PASSES`), count, to
    the slot, the experts the model of that count selects (`ExpertBalance.effects`, the counts its bias
    moves by), and their balance loss and z-loss are that model's: a pass the count leaves unrun, whose
    routers sow a zero state's choice, credits no expert and scores no position (`ROUTER_RAN`, row 11951's
    design). A skipped pass's choice, counted, would add its slots to the counts."""
    from test_loop import tiny

    from dew.nn.backbones.decoder_block import Mixture
    from dew.objectives.routing import ExpertBalance

    model = tiny(mixture=Mixture(experts=4, top_k=2, bias=True),
                 loop=Loop(CAP, step_norm=False, layers=(1, 2)))
    fixed = model.with_passes(count)
    ids = jnp.asarray(np.random.default_rng(0).integers(0, 32, (2, 12)), jnp.int32)
    (variables,) = drawn(model.init, ids[:, :-1])(0)
    balance = ExpertBalance(balance_rate=0.01, aux_loss_alpha=0.1, router_z_loss=0.01)
    step = Step(jnp.zeros((), jnp.int32), jax.random.key(3), None)

    def scored(model, batch):
        objective = LMObjective(model, ids.shape[1] - 1, balance=balance)
        loss, aux = objective.scalar_loss(variables, batch, step)
        return float(loss), jax.device_get(aux.effects)

    capped, capped_counts = scored(model, {"text": ids, LOOP_PASSES: np.full((2,), count, np.int32)})
    own, own_counts = scored(fixed, {"text": ids})
    for path, counted in jax.tree_util.tree_leaves_with_path(capped_counts):
        np.testing.assert_array_equal(counted, own_counts_at(own_counts, path), err_msg=f"{path} at {count}")
    # The unrun passes' terms add exact zeros, so the loss is the count's own to the bit.
    assert capped == own, (count, capped, own)


def own_counts_at(tree, path):
    """`tree`'s leaf at `path` (`jax.tree_util` keys)."""
    for key in path:
        tree = tree[key.key]
    return tree


def test_each_microbatch_of_a_sampled_run_is_dealt_its_drawn_count():
    """A run drawing its loop's pass count per microbatch, two microbatches an update (`accumulation` 2),
    deals each microbatch its key's draw at its position, one program stepping them all at every count it
    draws: more than one count among eight, each step's loss finite, every microbatch counted. The program
    at each count is that count's model (`test_the_caps_program_at_a_drawn_count_is_that_counts_model`),
    and each microbatch's work is its count's
    (`test_a_sampled_steps_work_is_its_microbatches_at_their_counts`)."""
    batch = token_batch()
    sampled = phased_trainer(looped(steps=CAP), 0, None, sampled_passes=TRAINED,
                             update=UpdateBatch(batches=2))
    state, _, _ = sampled.place()
    step = sampled.compile(state, batch)
    assert isinstance(step, DrawingStep)
    interval = Interval(time.time(), None, microbatches=0)
    drawn_counts = []
    for index in range(8):
        step.drawn(interval, index)
        drawn_counts.extend(step.counts)
        state, loss, *_ = step(state, batch)
        assert np.isfinite(float(loss)), (index, step.counts)
    assert drawn_counts == [TRAINED.count(words(sampled.key), position) for position in range(8)]
    assert len(set(drawn_counts)) > 1 and interval.microbatches == 8


@pytest.mark.parametrize("cap", [2, CAP])
def test_a_sampled_run_of_an_implicit_loop_in_bf16_reads_its_caps_narrow_copies_at_every_count(cap):
    """A bf16 decoder whose implicit loop's pass count is drawn per microbatch, its parameters read through
    narrow copies (row 10770): at its cap the loop's layers sum their cotangents over its passes and are read
    whole, and at one pass they are cast once and could be copied; and at a cap of two its one later pass
    casts θ once, which one pass reads nowhere. The window reads the cap's copies at every count, the
    loop's layers whole, θ through its copy where a pass reads it, so it plans and steps each count drawn,
    where it refused at its first step."""
    from dew.nn.kernels.generation import chosen

    model = dataclasses.replace(looped(implicit=True, steps=cap), dtype=jnp.bfloat16)
    batch = token_batch()
    law = dataclasses.replace(TRAINED, cap=cap)
    with chosen(narrow_copies="on"):
        sampled = phased_trainer(model, 0, None, sampled_passes=law, update=UpdateBatch(batches=2))
        state, _, _ = sampled.place()
        step = sampled.compile(state, batch)
        interval = Interval(time.time(), None, microbatches=0)
        drawn = []
        for index in range(8):
            step.drawn(interval, index)
            drawn.extend(step.counts)
            state, loss, *_ = step(state, batch)
            assert np.isfinite(float(loss)), (index, step.counts)
        # The copies a run of the cap's passes alone reads, the loop's layers whole among them.
        alone = phased_trainer(model, 0, None, update=UpdateBatch(batches=2))
        start, _, _ = alone.place()
        stepped, *_ = alone.compile(start, batch)(start, batch)

    def copied(compute) -> list[str]:
        flat = jax.tree_util.tree_flatten_with_path(compute)[0]
        return sorted(jax.tree_util.keystr(path) for path, _ in flat)

    assert len(set(drawn)) > 1 and state.compute is not None, drawn
    assert copied(state.compute) == copied(stepped.compute) != [], copied(state.compute)


def test_a_sampled_runs_draws_go_on_through_another_split_and_a_resume():
    """The draws continue one stream of the run's dispatched microbatches (`Interval.microbatches`): ten
    steps of four microbatches, then, resumed from a checkpoint's control, five of two, draw at positions 0
    to 49, not where a step and its split would place them; a checkpoint that counted none resumes at its
    step's microbatches, at the split its step reads."""
    key = jax.random.key(0)
    four, two = (DrawingStep(split, TRAINED, words(key), None) for split in (4, 2))
    interval = Interval(time.time(), None, microbatches=0)
    drawn = []
    for index in range(10):
        four.drawn(interval, index)
        drawn += four.counts
    control = {DISPATCHED_MICROBATCHES: interval.microbatches}
    resumed = Interval.resumed(dict(control), last_saved=10)
    for index in range(10, 15):
        two.drawn(resumed, index)
        drawn += two.counts
    assert drawn == [TRAINED.count(words(key), position) for position in range(50)]
    lacking = Interval.resumed({}, last_saved=7)
    two.drawn(lacking, 7)
    assert two.counts == [TRAINED.count(words(key), position) for position in (14, 15)]
    assert lacking.microbatches == 16


def test_a_sampled_step_called_by_hand_draws_each_calls_counts_itself():
    """A sampled run's step as `Trainer.compile` returns it, called by hand with no fit drawing for it (a
    script's loop, `dew inspect`'s sampled window), draws each call's counts itself (row 8268): its first
    call's at its state's step's microbatches, each next call's past the last, the counts the run's key
    draws there, a call a fit drew for (`drawn`) running the fit's; its calls compile nothing, every count
    run by the one program at its cap (row 11951's design: a program a count, compiled as each was first
    drawn)."""
    from dew.telemetry.compiles import Compiles

    batch = token_batch()
    sampled = phased_trainer(looped(steps=CAP), 0, None, sampled_passes=TRAINED, fit=FitPolicy(split=2))
    state, _, _ = sampled.place()
    step = sampled.compile(state, batch)
    assert isinstance(step, DrawingStep) and step.split == 2
    drawn = []
    with Compiles("step") as compiles:
        for _ in range(3):
            state, *_ = step(state, batch)
            drawn += step.counts
    assert drawn == [TRAINED.count(words(sampled.key), position) for position in range(6)]
    assert len(set(drawn)) > 1 and compiles.count == 0, (drawn, compiles.programs)
    step.drawn(Interval(time.time(), None, microbatches=40), int(state.step))
    step(state, batch)
    assert step.counts == [TRAINED.count(words(sampled.key), position) for position in (40, 41)]


def test_a_sampled_run_whose_variables_are_on_the_host_is_refused_as_it_is_built():
    """A run drawing its pass count per microbatch whose stated layout keeps its variables on the host (a
    host-master run, which no plan of the devices steps) is refused as the trainer is built (rows 8271,
    8400), and on a part whose table prices the streamed step (an H100's) a run of the default layout is
    never offered the streamed host layout (`streaming.refusal`), where the same run drawing no count is
    (row 8441: on the CPU's own table no part prices it, which refused both alike)."""
    from test_host_training import assumed_on_cpu

    from dew.training import Layout, Trainer, streaming

    with pytest.raises(ValueError):
        Trainer(LMObjective(looped(steps=CAP), 16), optax.sgd(0.1), key=jax.random.key(0),
                layout=Layout(host=("variables",), min_shard=1, tolerance=1.0), sampled_passes=TRAINED)
    drawing, fixed = (Trainer(LMObjective(looped(steps=CAP), 16), optax.adam(0.1), key=jax.random.key(0),
                              **given) for given in ({"sampled_passes": TRAINED}, {}))
    with assumed_on_cpu("sm90"):
        assert streaming.refusal(fixed, stated=False, ahead=False) is None
        assert streaming.refusal(drawing, stated=False, ahead=False) is not None


def test_a_sampled_run_resumed_steps_as_the_run_never_stopped(tmp_path, caplog):
    """A sampled run of four steps over data read in phases (`PhasedStream`), stopped after two and resumed,
    lands, bit for bit, where the run never stopped lands: each microbatch draws its count whatever its
    data's phase, and its resume draws on from the microbatches its checkpoint counted. A fresh run and a
    resume of a checkpoint that counted its microbatches say nothing of a checkpoint lacking them (row
    8276)."""
    from test_data_mixtures import assert_same, parameters
    from test_pass_weights import phased_draw, phased_loader

    from dew.training import Checkpoints

    tokens, _ = phased_draw(0)

    def fitted(directory, steps):
        return phased_trainer(looped(steps=CAP), 0, None, sampled_passes=TRAINED,
                          checkpoints=Checkpoints(str(directory), keep=8)).fit(
            phased_loader(tokens), steps=steps, log_every=100, checkpoint_every=1)

    def noted() -> int:
        """How many times a fit said its checkpoint counted none of its microbatches (`DrawingStep.drawn`)."""
        return len([record for record in caplog.records if record.name == "dew.training.compiler"
                    and record.funcName == "drawn"])

    with caplog.at_level("INFO", logger="dew.training.compiler"):
        whole = fitted(tmp_path / "whole", 4)
        fitted(tmp_path / "resumed", 2)
        resumed = fitted(tmp_path / "resumed", 4)
        assert noted() == 0
        # A checkpoint of a run drawing no count, which counted no microbatch: said once as the sampled
        # run resumes it, its draws on from its step's microbatches (row 8442).
        phased_trainer(looped(steps=CAP), 0, None, checkpoints=Checkpoints(str(tmp_path / "fixed"), keep=8)
                       ).fit(phased_loader(tokens), steps=2, log_every=100, checkpoint_every=1)
        taken = fitted(tmp_path / "fixed", 4)
    assert noted() == 1 and int(taken.step) == 4
    assert int(resumed.step) == 4
    assert_same(parameters(whole), parameters(resumed), "the resumed sampled run is not the run unstopped")
    # A resume drawing by another law, or by none, would draw another schedule of passes from the same key
    # and positions: its checkpoint's sampled law is the computation it trained, and the resume is refused
    # (row 10783).
    for law in (dataclasses.replace(TRAINED, mean=2.5), None):
        with pytest.raises(ValueError):
            phased_trainer(looped(steps=CAP), 0, None, sampled_passes=law,
                           checkpoints=Checkpoints(str(tmp_path / "resumed"), keep=8)).fit(
                phased_loader(tokens), steps=5, log_every=100, checkpoint_every=1, resume="require")


def unnormed() -> CausalTransformer:
    """A tiny decoder whose middle layer a loop of `CAP` passes runs, its queries and keys un-normed, whose
    scale QK-Clip bounds."""
    return CausalTransformer(vocab_size=256, emb_features=32, num_layers=3, num_heads=2, mlp_features=48,
                             max_seq_len=128, attention_impl="xla", qk_norm=False,
                             loop=Loop(CAP, layers=(1, 2)))


def test_a_sampled_run_under_qk_clip_pools_its_counts_logit_maxima_in_one_window():
    """A sampled run under QK-Clip, two microbatches an update, whose looped layer's attention a microbatch
    calls as many times as its count: its window holds each layer's maxima as one call, their largest
    (`compact_qk`'s `calls`), so every count's step pools into it, and the run steps."""
    from test_pass_weights import phased_draw, phased_loader

    from dew.optim import scale_by_qk_clip
    from dew.training import Layout, Trainer

    tokens, _ = phased_draw(0)
    trainer = Trainer(LMObjective(unnormed(), 16), optax.chain(scale_by_qk_clip(1.0), optax.sgd(0.1)),
                      key=jax.random.key(0), layout=Layout(min_shard=1, tolerance=1.0),
                      update=UpdateBatch(batches=2), sampled_passes=TRAINED)
    assert int(trainer.fit(phased_loader(tokens), steps=4, log_every=100).step) == 4


@pytest.mark.parametrize("sampled_first", [False, True], ids=["fixed-first", "sampled-first"])
def test_a_fixed_and_a_sampled_run_of_one_config_each_trace_their_own_window(sampled_first):
    """Two trainers of one optimizer config under QK-Clip (muonclip) in one process, one at the loop's own
    count and one drawing it per microbatch, two microbatches an update, each compiled over one raw batch:
    each traces its own step, a fixed run's window a looped layer's QK calls apart and a sampled run's as
    one (`compact_qk`), so neither reads the other's from the process's cache of steps
    (`Trainer._step_key`), in either order, and both step."""
    from dew.optim import OptimConfig
    from dew.training import Layout, Trainer

    config = OptimConfig(optimizer=MuonClip(), learning_rate=0.01)
    batch = token_batch()

    def stepped(**sampling) -> None:
        trainer = Trainer(LMObjective(unnormed(), 16), config, key=jax.random.key(0),
                          layout=Layout(min_shard=1, tolerance=1.0), update=UpdateBatch(batches=2),
                          **sampling)
        trainer.scheduled(1)
        state, _, _ = trainer.place()
        step = trainer.compile(state, batch)
        if isinstance(step, DrawingStep):
            step.drawn(Interval(time.time(), None, microbatches=0), 0)
        step(state, batch)

    runs = [{}, {"sampled_passes": TRAINED}]
    for options in (runs[::-1] if sampled_first else runs):
        stepped(**options)


def test_a_phased_run_keeps_its_loops_horizon_at_every_count():
    """A run whose loop trains through its last two passes (`backprop_steps`) and runs one pass in its first
    phase and three in its second steps each phase as a run of the model at that count steps it, the first
    through its one pass and the second through its last two, bit for bit: the count a phase clamped the
    horizon to is not the next phase's."""
    from dew.batch import PHASE, Phase

    model = looped(steps=CAP).with_passes(CAP, backprop_steps=2)
    assert model is not None
    batch = token_batch()
    phased = phased_trainer(model, 0, (1, CAP))
    for phase, passes in enumerate((1, CAP)):
        assert phased.phase_passes.at({**batch, PHASE: Phase(phase)}) == passes
        state, _, _ = phased.place()
        stepped, loss, *_ = phased.compile(state, batch)(state, batch)
        alone = phased_trainer(model.with_passes(passes, backprop_steps=min(2, passes)), 0, None)
        start, _, _ = alone.place()
        own, own_loss, *_ = alone.compile(start, batch)(start, batch)
        np.testing.assert_array_equal(np.asarray(loss), np.asarray(own_loss), err_msg=f"phase {phase}")
        assert jax.tree.all(jax.tree.map(lambda a, b: bool(np.array_equal(a, b)),
                                         jax.device_get(stepped.variables), jax.device_get(own.variables)))


def test_a_sampled_steps_work_is_its_microbatches_at_their_counts():
    """A sampled step's work (`TRAINING_FLOPS`, which train/mfu divides) is the sum of its microbatches'
    rows' work at their drawn counts (invariant 4): each count's column of the batch's counted work is the
    work a run of that count counts its rows (`prepare_counts`), and the step's metric sums its two
    microbatches' rows of their counts' columns, exactly; its hardware FLOPs are the one program's at its
    cap, reported as a bound on its counts' (`StepFlops.bound`; row 11951's design)."""
    from dew.training.model_flops import TRAINING_FLOPS, prepare_counts

    model = looped(steps=CAP)
    host = {"text": token_draw()}
    sampled = phased_trainer(model, 0, None, sampled_passes=TRAINED, fit=FitPolicy(split=2))
    counted = prepare_counts(sampled.objective, host, count_model=True,
                             counts=sampled.phase_passes.counted())
    for count in range(1, CAP + 1):
        alone = phased_trainer(model.with_passes(count), 0, None)
        np.testing.assert_array_equal(counted[TRAINING_FLOPS][:, count - 1],
                                      prepare_counts(alone.objective, host, count_model=True)[TRAINING_FLOPS])
    batch = {name: jnp.asarray(value) for name, value in counted.items()}
    state, _, _ = sampled.place()
    step = sampled.compile(state, batch)
    assert isinstance(step, DrawingStep)
    interval = Interval(time.time(), None, microbatches=0)
    flops = step.drawn(interval, 0)
    _, _, metrics, *_ = step(state, batch)
    work = counted[TRAINING_FLOPS]
    expected = sum(jnp.sum(jnp.asarray(work[part::2, count - 1]), dtype=jnp.float32)
                   for part, count in enumerate(step.counts))
    np.testing.assert_array_equal(np.asarray(metrics[TRAINING_FLOPS]), np.asarray(expected))
    capped = sampled.step_flops
    assert flops is not None and capped is not None
    assert (flops.model, flops.hardware, flops.bound) == (capped.model, capped.hardware, "_upper_bound"), (
        flops)


class Replayed(LMObjective):
    """The LM objective, its statistics its loss's total and mass as a pair, no `Ratio`: a window replays
    them (`pooled_by_mean`), whatever the step's pass count."""

    def loss(self, variables, batch, step):
        stats, aux = super().loss(variables, batch, step)
        return (stats.total, stats.mass), aux

    def reduce_loss(self, stats):
        total, mass = stats
        return super().reduce_loss(Ratio(total, mass))


@pytest.mark.parametrize("grouped", [False, True], ids=["own-window", "diloco"])
def test_a_sampled_run_whose_window_replays_its_microbatches_is_refused(grouped):
    """A loss whose statistics are no mean its window pools replays each microbatch of an update at the
    count closing it: a sampled run of two microbatches a step is refused (row 7884), its own window and
    DiLoCo's groups' alike (`TrainState.window`), the statistics alike at every count; read one microbatch
    an update, it steps."""
    from dew.training import Layout, Trainer
    from dew.training.diloco import DiLoCo

    def trainer(split):
        grouping = {"diloco": DiLoCo(every=2, rate=0.7, momentum=0.9)} if grouped else {}
        return Trainer(Replayed(looped(steps=CAP), 16), optax.sgd(0.1), key=jax.random.key(0),
                       layout=Layout(min_shard=1, tolerance=1.0), sampled_passes=TRAINED,
                       fit=FitPolicy(split=split), **grouping)

    batch = token_batch()
    replayed = trainer(2)
    state, _, _ = replayed.place()
    with pytest.raises(ValueError):
        replayed.compile(state, batch)
    whole = trainer(1)
    state, _, _ = whole.place()
    assert isinstance(whole.compile(state, batch), DrawingStep)


@pytest.mark.parametrize("fields", [{"loop_steps_mean": 2.0},
                                    {"loop_steps_sample": "lognormal-poisson", "loop_steps_mean": 2.0,
                                     "loop_steps_max": 4},
                                    {"loop_steps_sample": "lognormal-poisson", "loop_steps_mean": 2.0,
                                     "loop_steps_dispersion": 0.5, "loop_steps_max": 4, "loop_steps": 4},
                                    {"loop_steps_sample": "lognormal-poisson", "loop_steps_mean": 1.0,
                                     "loop_steps_dispersion": 0.5, "loop_steps_max": 4}],
                         ids=["a law unnamed", "a law half stated", "beside a stated count",
                              "outside its law"])
def test_a_decoder_run_states_a_sampled_pass_count_whole_and_once(fields):
    """A run's sampled pass count (`--loop-steps-sample`) is stated whole, its law's mean, dispersion and cap
    with its name, beside no stated count (the cap the model holds), and within its law; stated so, the run
    hands its trainer the law and its model the cap."""
    from dew.config import DecoderRunConfig

    with pytest.raises(ValueError):
        DecoderRunConfig(**fields)
    stated = DecoderRunConfig(loop_steps_sample="lognormal-poisson", loop_steps_mean=2.0,
                              loop_steps_dispersion=0.5, loop_steps_max=4)
    assert stated.sampled_passes() == SampledPasses(2.0, 0.5, 4) and stated.handed_passes == 4


class _Drawing:
    """A rollout drawing nothing: the batch it is handed (`dew.training.rollout.Rollout`)."""

    def __call__(self, state, batch, key):
        return batch


class UncountedLM(LMObjective):
    """A language model's loss of its own that hands its model no count of passes (a custom objective's)."""

    @property
    def reads_pass_counts(self) -> bool:
        return False


def test_a_pass_count_drawn_per_microbatch_for_an_objective_that_reads_none_is_refused():
    """A run drawing its pass count per microbatch hands each microbatch its count in its batch
    (`LOOP_PASSES`), which an objective whose loss reads no count (`Objective.reads_pass_counts`), a custom
    one's over a looped decoder, would leave unread, running every pass at the cap whatever was drawn: such a
    trainer is refused as it is built, and the same objective at a phase's count is not."""
    from dew.training import Layout, Trainer

    objective = UncountedLM(looped(steps=CAP), 16)
    with pytest.raises(ValueError):
        Trainer(objective, optax.sgd(0.1), key=jax.random.key(0), layout=Layout(min_shard=1, tolerance=1.0),
                sampled_passes=TRAINED)
    Trainer(objective, optax.sgd(0.1), key=jax.random.key(0), layout=Layout(min_shard=1, tolerance=1.0),
            phase_passes=(CAP,))


def counted_objective(name: str, at) -> tuple:
    """The objective `name` over the trainer cells' looped decoder at its cap (`at`, each looped model it
    reads made at a count), its one batch, and whether its step reads its frozen reference (`Step.ema`)."""
    from dew.diffusion.discrete import MDLM
    from dew.objectives.diffusion.masked import MaskedDiffusionObjective
    from dew.objectives.distillation import DistillationObjective
    from dew.objectives.rl.preference import DPOObjective

    tokens = jnp.asarray(np.random.default_rng(0).integers(0, 255, (2, 17)), jnp.int32)
    if name == "dpo":
        from test_dpo import WIDTH, pair_batch

        return DPOObjective(at(looped(steps=CAP)), WIDTH - 1, beta=0.5), pair_batch(), True
    if name == "masked-diffusion":
        model = at(looped(steps=CAP).clone(causal=False))
        return (MaskedDiffusionObjective(model, MDLM(mask_id=255)(), seq_len=16, ema_decay=None),
                {"text": tokens[:, :16]}, False)
    assert name == "distillation"
    student, teacher = (LMObjective(at(looped(steps=CAP)), 16, ema_decay=None) for _ in range(2))
    return DistillationObjective(student, teacher, alpha=0.5), {"text": tokens}, False


@pytest.mark.parametrize("count", range(1, CAP + 1))
@pytest.mark.parametrize("name", ["dpo", "masked-diffusion", "distillation"])
def test_an_objectives_cap_program_at_a_drawn_count_is_that_counts_objective(name, count):
    """Every objective a sampled run trains over a looped decoder runs its cap's one program at the count its
    microbatch drew (`LOOP_PASSES`, traced), as a language model's does: a preference pair's policy and
    reference, a masked diffusion's denoiser, and a distillation's student and teacher, each loss and every
    parameter's gradient on par from float64 with the objective over the models of that count's own
    (`with_passes`; review of row 11951: each was refused, which main trained at every count)."""
    capped, batch, referenced = counted_objective(name, lambda model: model)
    fixed, _, _ = counted_objective(name, lambda model: model.with_passes(count))
    # A run drawing its count per microbatch trains it, its every count by the cap's one program.
    PhasePasses.of(capped, None, TRAINED)
    rows = len(next(iter(jax.tree.leaves(batch))))
    passes = np.full((rows,), count, np.int32)
    initial = drawn(capped.init)

    def gradient(objective, given_passes=None):
        given = batch if given_passes is None else {**batch, LOOP_PASSES: given_passes}

        def run(variables):
            # The reference a preference loss reads is the drawn variables, frozen for both sides alike.
            step = Step(jnp.asarray(0), jax.random.key(0), variables if referenced else None)

            def loss(params):
                return objective.scalar_loss({**variables, "params": params}, given, step)[0]
            return jax.value_and_grad(loss)(variables["params"])
        return run

    def counted(variables):
        return gradient(capped, passes)(variables)

    def own(variables):
        return gradient(fixed)(variables)

    def own_in_float64(variables):
        twin, _, _ = counted_objective(name, lambda model: float64_twin(model.with_passes(count)))
        return gradient(twin)(variables)

    assert_as_exact(jax.jit(counted), dew_path(jax.jit(own)), initial,
                    f"{name}'s cap program at {count} passes", seeds=PASS_SEEDS,
                    truth_fn=jax.jit(own_in_float64))


def decision_loss_at(model, weighed):
    """`counted_gradient` of a decision head's objective over the looped decoder `model` (`weighed` its
    passes' weighing), over the two-option tickets of `test_decision_objective`, its rows given `passes` (a
    traced count, `LOOP_PASSES`), or run at the model's own count where None."""
    from decision_support import decision_rows
    from test_decision_objective import EXAMPLES

    from dew.data import ByteTokenizer
    from dew.data.dataset import DataPartition
    from dew.decision import DecisionObjective, Specials
    from dew.decision.layout import SegmentLayout

    objective = DecisionObjective(model, tokenizer=ByteTokenizer(),
                                  specials=Specials(None, 10, 0, "\x00", 255),
                                  layout=SegmentLayout.state_first_row(max_len=32, head_max_len=24,
                                                                       option_tokens=6), **weighed)
    batch = next(decision_rows(EXAMPLES, 2, objective=objective.data()).train(DataPartition()))
    return objective, batch, counted_gradient(objective, batch)


DECIDED = {**WEIGHED, "input-lm": {"input_lm": 0.5}}
"""A decision loss's readings of its backbone's passes (`WEIGHED`), and its last pass's beside the rows' own
next-token term (`input_lm`), whose causal read of the rows runs the backbone again."""


@pytest.mark.parametrize("count", range(1, CAP + 1))
@pytest.mark.parametrize("weighing", sorted(DECIDED))
def test_a_decision_heads_cap_program_at_a_drawn_count_is_that_counts_model(weighing, count):
    """A decision run drawing its pass count per microbatch steps every count by its looped backbone's cap's
    one program, its rows given the count (`LOOP_PASSES`, traced), as a language model's does: its loss and
    every parameter's gradient on par from float64 with the decision head over the backbone of that count's
    own (`with_passes`), its passes weighed by its last alone, their mean and its exit gate, and its rows'
    own next-token term read at that count too (`input_lm`; check 1; row 11951's design, which a decision
    run, row 9691's, keeps)."""
    model = weighed_model("last" if weighing == "input-lm" else weighing)
    fixed = model.with_passes(count)
    weighed = DECIDED[weighing]
    capped, batch, at_cap = decision_loss_at(model, weighed)
    passes = np.full((len(batch["labels"]),), count, np.int32)
    initial = capped.init

    def draw(seed):
        return initial(jax.random.key(seed)), passes

    def own(variables, _):
        return decision_loss_at(fixed, weighed)[2](variables)

    def own_in_float64(variables, _):
        return decision_loss_at(float64_twin(fixed), weighed)[2](variables)

    # Each count's own model is Dew's (`dew_path`): an invariance claim, held against it compiled alike.
    assert_as_exact(jax.jit(at_cap), dew_path(jax.jit(own)), draw,
                    f"a decision head at {count} passes ({weighing})", seeds=PASS_SEEDS,
                    truth_fn=jax.jit(own_in_float64))


@pytest.mark.parametrize("count", range(1, CAP + 1))
def test_a_decision_heads_cap_program_whose_backbone_encodes_its_context_runs_both_passes_at_the_count(count):
    """A decision run whose backbone encodes each row's context in a pass of its own (`context_encoder=
    "backbone"`), read by cross-attention before its first layer, drawing its pass count per microbatch:
    the context's pass and the answering pass both run the backbone's loop at the drawn count
    (`BackboneCall.loop_passes`), so its loss and every gradient are on par from float64 with the decision
    over the backbone of that count's own (`with_passes`, whose both passes run its count; check 1). A
    context encoded at the cap beside an answer at the count is another model's, which this holds apart."""
    from decision_support import decision_rows
    from test_decision_context import QUESTION, ROWS

    from dew.data import ByteTokenizer
    from dew.data.dataset import DataPartition
    from dew.decision import DecisionObjective, Example, Specials
    from dew.nn.cross_attention import CrossAttention

    def objective(over):
        return DecisionObjective(over, tokenizer=ByteTokenizer(), specials=Specials(None, 10, 0, "\x00", 255),
                                 context_encoder="backbone")

    model = looped(steps=CAP).clone(max_seq_len=512, cross_attention=CrossAttention((0,), 32))
    fixed = model.with_passes(count)
    capped = objective(model)
    rows = [Example(text, QUESTION, {"pick": label}, context=note) for text, label, note in ROWS[:4]]
    batch = next(decision_rows(rows, 2, objective=capped.data()).train(DataPartition()))
    passes = np.full((len(batch["labels"]),), count, np.int32)

    def draw(seed):
        # Its tree unboxed, as a step holds it, which the context's pass reads leaf by leaf.
        return flax.linen.unbox(capped.init(jax.random.key(seed))), passes

    def own(variables, _):
        return counted_gradient(objective(fixed), batch)(variables)

    def own_in_float64(variables, _):
        # The backbone in float64 as a decision's truth reads it (`test_decision_context.in_float64`).
        return counted_gradient(objective(fixed.clone(dtype=jnp.float64)), batch)(variables)

    # Each count's own model is Dew's (`dew_path`): an invariance claim, held against it compiled alike.
    assert_as_exact(jax.jit(counted_gradient(capped, batch)), dew_path(jax.jit(own)), draw,
                    f"a context-encoding decision at {count} passes", seeds=PASS_SEEDS,
                    truth_fn=jax.jit(own_in_float64))


def test_a_sampled_objective_names_its_loss_s_forward_and_a_counted_run_refuses_one_that_takes_no_count():
    """The forward a run drawing its count per microbatch checks takes the count is the one its
    objective's loss reads a module by (`Objective.counted_forward`, `chunked.read_method`); and an LM whose
    adaptation steps its decoder's own layers chunk by chunk (`ChunkAdaptation.layers`) through its decode
    cache, whose loop runs no count given per call, reads no count, so such a run is refused (row 12300)."""
    from dew.nn.backbones.decoder_stack import Loop
    from dew.objectives.lm.adaptation import ChunkAdaptation
    from dew.training import Layout, Trainer

    model = looped(steps=CAP)
    assert LMObjective(model, 16).counted_forward(model) == "hidden_states"
    adapted = LMObjective(model, 16, adaptation=ChunkAdaptation(window=4, rate=0.5, layers=(0,)))
    assert not adapted.reads_pass_counts and isinstance(model.loop, Loop)
    with pytest.raises(ValueError):
        Trainer(adapted, optax.sgd(0.1), key=jax.random.key(0), layout=Layout(min_shard=1, tolerance=1.0),
                sampled_passes=TRAINED)


@pytest.mark.parametrize("loop", [{"shared_cache": True}, {"exit_threshold": 0.5, "exit_gate": True}],
                         ids=["shared-cache", "exit-threshold"])
def test_a_loop_that_chooses_its_passes_itself_is_refused_a_count_drawn_per_microbatch(loop):
    """A loop sharing one cache across its passes, or handing each token on at an exit threshold, chooses
    its passes itself, which the one program at its cap runs at no count given per call (`run_loop`): a run
    drawing its count per microbatch over it is refused as it is built, and the same loop at a phase's count
    is not (review of row 11951: admitted, it failed at its first trace)."""
    from dew.training import Layout, Trainer

    model = looped(steps=CAP).clone(loop=Loop(CAP, layers=(1, 2), **loop))
    with pytest.raises(ValueError):
        Trainer(LMObjective(model, 16), optax.sgd(0.1), key=jax.random.key(0),
                layout=Layout(min_shard=1, tolerance=1.0), sampled_passes=TRAINED)
    Trainer(LMObjective(model, 16), optax.sgd(0.1), key=jax.random.key(0),
            layout=Layout(min_shard=1, tolerance=1.0), phase_passes=(CAP,))


def test_a_counted_mean_weighs_the_passes_that_ran_in_the_terms_precision():
    """A count given per call weighs the passes that ran alike (`PassWeights.ran`), each share `1/c` in the
    terms' own dtype: a float64 loss's share of three passes is float64's third, not float32's rounded
    (review of row 11951)."""
    from dew.objectives.passes import PassWeights

    weights = PassWeights.of("mean")
    assert weights is not None
    with jax.enable_x64(new_val=True):
        shares = weights.ran(tuple(jnp.asarray([True, True, False])), jnp.float64)
        assert [share.dtype for share in shares] == [jnp.dtype(jnp.float64)] * 4
        assert [float(share) for share in shares] == [1 / 3, 1 / 3, 0.0, 1 / 3]


def multimodal_looped():
    """A float32 multimodal wrapper (a tiny vision tower and projector, gemma3's fusion) over the trainer
    cells' looped decoder (`looped`), its image placeholder id 3."""
    from dew.nn.multimodal import MultimodalTransformer
    from dew.nn.vision import GemmaProjector, SiglipVision

    return MultimodalTransformer(
        looped(steps=CAP), SiglipVision(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2,
                                         image_size=8, patch_size=4),
        GemmaProjector(text_width=32, patches_per_side=2, tokens_per_side=1), family="gemma3",
        image_token_id=3, dtype=jnp.float32)


@pytest.mark.parametrize("count", range(1, CAP + 1))
def test_a_multimodal_cap_program_at_a_drawn_count_is_that_counts_model_over_its_image_rows(count):
    """A multimodal language run drawing its pass count per microbatch steps every count by its wrapper's
    cap's one program, its rows given the count (`LOOP_PASSES`, traced), its forward handing the count to
    its language model (`MultimodalTransformer.hidden_states`): over rows whose image fills a placeholder,
    the vision tokens read under the count, its loss and every parameter's gradient on par from float64 with
    the wrapper of that count's own (`with_passes`; check 1)."""
    from test_tokens_embedded import media_rows

    from dew.nn.inputs import ModelInputs

    model = multimodal_looped()
    fixed = model.with_passes(count)
    assert fixed is not None
    rows = np.tile(np.asarray([[2, 3, 4, 5, 6, 7, 8, 9, 10]], np.int32), (2, 1))
    images = np.tile(np.asarray([[-1, 0, -1, -1, -1, -1, -1, -1, -1]], np.int32), (2, 1))
    text = media_rows(rows, images)
    assert isinstance(text, ModelInputs)
    batch = {"text": text}

    def gradient(over):
        return counted_gradient(LMObjective(over, 8, head_chunks=1), batch)

    passes = np.full((2,), count, np.int32)
    initial = LMObjective(model, 8, head_chunks=1).init

    def draw(seed):
        return initial(jax.random.key(seed)), passes

    def own(variables, _):
        return gradient(fixed)(variables)

    def own_in_float64(variables, _):
        return gradient(float64_twin(fixed))(variables)

    # Each count's own model is Dew's (`dew_path`): an invariance claim, held against it compiled alike.
    assert_as_exact(jax.jit(gradient(model)), dew_path(jax.jit(own)), draw,
                    f"a multimodal cap program at {count} passes", seeds=PASS_SEEDS,
                    truth_fn=jax.jit(own_in_float64))


def test_a_pass_count_drawn_per_microbatch_over_a_loop_whose_forward_takes_none_is_refused():
    """A looped module whose forward its loss reads by names no count of passes a call runs (`loop_passes`),
    an adapter's handing its keywords on, which states no count it reads, would run every pass at its cap
    whatever a microbatch draws: a run drawing its count per microbatch over it is refused as it is built,
    and the same module at a phase's count is not (review of row 11951: a forward taking any keyword was
    read as taking the count where the decoder's did)."""
    from dew.training import Layout, Trainer

    class Uncounted(CausalTransformer):
        def hidden_states(self, tokens, **kwargs):
            return super().hidden_states(tokens, **{key: value for key, value in kwargs.items()
                                                    if key != "loop_passes"})

    base = looped(steps=CAP)
    model = Uncounted(**{field.name: getattr(base, field.name) for field in dataclasses.fields(base)
                         if field.init and field.name not in ("parent", "name")})
    with pytest.raises(ValueError):
        Trainer(LMObjective(model, 16), optax.sgd(0.1), key=jax.random.key(0),
                layout=Layout(min_shard=1, tolerance=1.0), sampled_passes=TRAINED)
    Trainer(LMObjective(model, 16), optax.sgd(0.1), key=jax.random.key(0),
            layout=Layout(min_shard=1, tolerance=1.0), phase_passes=(CAP,))


def test_a_sampled_run_under_autopgle_profiles_and_steps_its_counted_program():
    """A run drawing its pass count per microbatch under JAX's AutoPGLE (`jax_enable_pgle`) profiles the
    program it planned, its cap's counted one, on its fit's trials over the batch that program reads, each
    microbatch's count among its rows (`StepCompiler.counted_trials`), and steps on (review of row 11951:
    the trials held the batch read before its counts, which the profiled program refused before step 1)."""
    from test_pass_weights import phased_draw, phased_loader
    from test_plan import autopgle

    tokens, _ = phased_draw(0)
    with autopgle(2):
        state = phased_trainer(looped(steps=CAP), 0, None, sampled_passes=TRAINED).fit(
            phased_loader(tokens), steps=2, log_every=100)
    assert int(state.step) == 2


def test_a_sampled_run_resumes_at_the_rung_its_checkpoint_climbed_to(monkeypatch):
    """A run drawing its pass count per microbatch, at one microbatch a step, whose devices hold its step
    recomputing all its layers (`full`) and not at its own rung, climbs there and records that rung for its
    batch's layout (`checkpoint_rung`); a run resuming that rung (`resume_rung`) on devices that hold either
    compiles at the rung it holds, which the fit's compile report states (`Trainer.plan`'s first probe), not
    its own (review of row 11951: recorded under the counted layout, the resume searched from the ladder's
    foot)."""
    from test_pass_weights import phased_draw, phased_loader
    from test_plan import LIMIT, free

    tokens, _ = phased_draw(0)

    def fitted(model, rung=None):
        trainer = phased_trainer(model, 0, None, sampled_passes=TRAINED, fit=FitPolicy(split=1))
        if rung is not None:
            trainer._compiler.resume_rung(rung)
        trainer.fit(phased_loader(tokens), steps=1, log_every=100)
        return trainer

    def needed(model) -> int:
        free(monkeypatch, LIMIT)
        return LIMIT - fitted(model).plan["probes"][0]["headroom"]

    own, full = needed(looped(steps=CAP)), needed(looped(steps=CAP).clone(remat="full"))
    assert full < own, (full, own)
    free(monkeypatch, (own + full) // 2)
    climbed = fitted(looped(steps=CAP))
    assert climbed.plan["remat"] is not None, climbed.plan
    free(monkeypatch, LIMIT)
    resumed = fitted(looped(steps=CAP), climbed._compiler.checkpoint_rung())
    first = resumed.plan["probes"][0]
    assert first["remat"] == climbed.plan["remat"] and first["fits"], (first, climbed.plan)


@pytest.mark.parametrize("target", [pytest.param("sm90", marks=lowers("sm90"))])
def test_a_sampled_step_lowered_for_a_gpu_is_its_counted_program(target):
    """A run drawing its pass count per microbatch lowered for a GPU this host is not (`lower_for`) lowers
    the program it steps, its cap's, reading each microbatch's count among its rows (`LOOP_PASSES`), not a
    fixed count's (review of row 11951: lowered before its counts were laid in)."""
    from dew.training._lowering import Lowered

    trainer = phased_trainer(looped(steps=CAP), 0, None, sampled_passes=TRAINED, lower_for=target)
    state = jax.eval_shape(trainer.initial_state, trainer.objective.initializer, trainer.key)
    with pytest.raises(Lowered) as lowered:
        trainer.compile(state, {"text": jax.ShapeDtypeStruct((jax.device_count() * 2, 17), jnp.int32)})
    inputs = [jax.tree_util.keystr(path) for path, _ in
              jax.tree_util.tree_flatten_with_path(lowered.value.lowered.args_info)[0]]
    assert any(LOOP_PASSES in path for path in inputs), inputs


def test_a_rollout_beside_a_pass_count_drawn_per_microbatch_is_refused():
    """A rollout samples a step's rows and behaviour log-probabilities from the one policy it holds, at the
    run's cap, while a run drawing its pass count per microbatch trains each microbatch at its own count, so
    it would train on rows a policy it does not run sampled: such a trainer is refused as it is built, and
    one at a phase's counts is not (row 8272)."""
    model = looped(steps=CAP).with_passes(CAP)
    assert model is not None
    with pytest.raises(ValueError):
        phased_trainer(model, 0, None, sampled_passes=TRAINED, rollout=_Drawing())
    phased_trainer(model, 0, (CAP,), rollout=_Drawing())


