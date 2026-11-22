"""A looped model's pass count drawn per microbatch from a bounded law
(`dew.training.phase_passes.SampledPasses`, row 7660)."""

import dataclasses
import math
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import raw_leaf
from scipy import integrate, optimize, special, stats
from test_pass_weights import looped, phased_trainer

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_stack import Loop
from dew.objectives.base import Ratio
from dew.objectives.lm.objective import LMObjective
from dew.training.compiler import DrawingStep
from dew.training.interval import DISPATCHED_MICROBATCHES, Interval
from dew.training.phase_passes import SampledPasses

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


@pytest.mark.parametrize("trained", [None, 2], ids=["every-pass", "last-two"])
def test_each_microbatch_of_a_sampled_run_steps_as_a_run_of_its_drawn_count(trained):
    """A run drawing its loop's pass count per microbatch, two microbatches an update (`accumulation` 2),
    steps each microbatch as a run of the model at its drawn count steps it (invariant 1): from the same
    state over the same batch, its loss, its metrics and its stepped state, an update's window pooled
    over microbatches of other counts and committed by the second's, the same bits as that count's run's,
    each count's program its own over the one state. A loop training through its last two passes
    (`backprop_steps`) trains a count below them through them all. Its counts are its key's draws at the
    microbatches' positions, and it draws more than one count."""
    model = looped(steps=CAP).with_passes(CAP, backprop_steps=trained)
    assert model is not None
    batch = token_batch()
    sampled = phased_trainer(model, 0, None, sampled_passes=TRAINED, accumulation=2)
    state, _, _ = sampled.place()
    step = sampled.compile(state, batch)
    assert isinstance(step, DrawingStep)
    interval = Interval(time.time(), None, microbatches=0)
    alone = {count: phased_trainer(model.with_passes(count, backprop_steps=trained and min(trained, count)),
                                   0, None, accumulation=2) for count in range(1, CAP + 1)}
    programs: dict = {}
    drawn = []
    for index in range(8):
        step.drawn(interval, index, state)
        (count,) = step.counts
        drawn.append(count)
        placement = jax.tree.map(lambda leaf: leaf.sharding, state)
        given = jax.device_get(state)
        state, loss, metrics, *_ = step(state, batch)
        if count not in programs:
            start, _, _ = alone[count].place()
            programs[count] = alone[count].compile(start, batch)
        own, own_loss, own_metrics, *_ = programs[count](jax.device_put(given, placement), batch)
        np.testing.assert_array_equal(np.asarray(loss), np.asarray(own_loss), err_msg=f"microbatch {index}")
        for name, value in jax.device_get(metrics).items():
            np.testing.assert_array_equal(value, jax.device_get(own_metrics)[name], err_msg=f"{index} {name}")
        ours, theirs = (jax.tree_util.tree_flatten_with_path(jax.device_get(jax.tree.map(raw_leaf, tree)))[0]
                        for tree in (state, own))
        unlike = [jax.tree_util.keystr(path) for (path, a), (_, b) in zip(ours, theirs, strict=True)
                  if not np.array_equal(a, b)]
        assert not unlike, f"microbatch {index} at {count} passes: {unlike}"
    assert drawn == [TRAINED.count(words(sampled.key), position) for position in range(8)]
    assert len(set(drawn)) > 1 and interval.microbatches == 8


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
        sampled = phased_trainer(model, 0, None, sampled_passes=law, accumulation=2)
        state, _, _ = sampled.place()
        step = sampled.compile(state, batch)
        interval = Interval(time.time(), None, microbatches=0)
        drawn = []
        for index in range(8):
            step.drawn(interval, index, state)
            drawn.extend(step.counts)
            state, loss, *_ = step(state, batch)
            assert np.isfinite(float(loss)), (index, step.counts)
        # The copies a run of the cap's passes alone reads, the loop's layers whole among them.
        alone = phased_trainer(model, 0, None, accumulation=2)
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
    flops = dict.fromkeys(range(1, CAP + 1))
    # Every count's plan in hand, so a draw admits none (`DrawingStep.admit`): the draws alone are read.
    four, two = (DrawingStep(dict.fromkeys(flops), flops, split, TRAINED, words(key)) for split in (4, 2))
    interval = Interval(time.time(), None, microbatches=0)
    drawn = []
    for index in range(10):
        four.drawn(interval, index, None)
        drawn += four.counts
    control = {DISPATCHED_MICROBATCHES: interval.microbatches}
    resumed = Interval.resumed(dict(control), last_saved=10)
    for index in range(10, 15):
        two.drawn(resumed, index, None)
        drawn += two.counts
    assert drawn == [TRAINED.count(words(key), position) for position in range(50)]
    lacking = Interval.resumed({}, last_saved=7)
    two.drawn(lacking, 7, None)
    assert two.counts == [TRAINED.count(words(key), position) for position in (14, 15)]
    assert lacking.microbatches == 16


def test_a_sampled_step_admits_each_count_on_its_loop_as_its_draws_first_ask_it():
    """A run that draws its pass count per microbatch builds its step with no count below its cap admitted:
    each call admits on the loop the counts it draws that none admitted yet (`DrawingStep.plans`), while
    from a fit's first draw every count below the cap compiles off the loop (`LaterCounts`), so a count
    drawn once its compile there is done is admitted compiling no step program on the loop (rows 7959,
    8067: every count compiled on the loop before the first step)."""
    from dew.telemetry.compiles import Compiles

    batch = token_batch()
    sampled = phased_trainer(looped(steps=CAP), 0, None, sampled_passes=TRAINED, accumulation=2)
    state, _, _ = sampled.place()
    step = sampled.compile(state, batch)
    assert isinstance(step, DrawingStep) and set(step.plans) == {CAP}
    interval = Interval(time.time(), None, microbatches=0)
    step.drawn(interval, 0, state)
    state, *_ = step(state, batch)
    first = set(step.plans)
    for count in range(1, CAP):
        step.waited(count)
    with Compiles("step") as compiles:
        for index in range(1, 32):
            if len(step.plans) == CAP:
                break
            step.drawn(interval, index, state)
            state, *_ = step(state, batch)
    assert first < set(step.plans) == set(range(1, CAP + 1)), (first, set(step.plans))
    assert compiles.count == 0, compiles.programs


def test_a_sampled_step_called_by_hand_draws_each_calls_counts_itself():
    """A sampled run's step as `Trainer.compile` returns it, called by hand with no fit drawing for it (a
    script's loop, `dew inspect`'s sampled window), draws each call's counts itself (row 8268): its first
    call's at its state's step's microbatches, each next call's past the last, the counts the run's key
    draws there, a call a fit drew for (`drawn`) running the fit's; with no fit to wait for them as it ends,
    it compiles no count off its loop (the review of rows 7959, 8067: by-hand calls left a thread compiling
    every count past their caller's end)."""
    batch = token_batch()
    sampled = phased_trainer(looped(steps=CAP), 0, None, sampled_passes=TRAINED, split=2)
    state, _, _ = sampled.place()
    step = sampled.compile(state, batch)
    assert isinstance(step, DrawingStep) and step.split == 2
    drawn = []
    for _ in range(3):
        state, *_ = step(state, batch)
        drawn += step.counts
    assert drawn == [TRAINED.count(words(sampled.key), position) for position in range(6)]
    # No fit waits for counts compiling off the loop as a caller's calls end, so these compiled none.
    assert not [thread for thread in threading.enumerate() if thread.name.startswith("dew-later-counts")]
    step.drawn(Interval(time.time(), None, microbatches=40), int(state.step), state)
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
                      key=jax.random.key(0), layout=Layout(min_shard=1, tolerance=1.0), accumulation=2,
                      sampled_passes=TRAINED)
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

    config = OptimConfig(optimizer="muonclip", learning_rate=0.01)
    batch = token_batch()

    def stepped(**sampling) -> None:
        trainer = Trainer(LMObjective(unnormed(), 16), config, key=jax.random.key(0),
                          layout=Layout(min_shard=1, tolerance=1.0), accumulation=2, **sampling)
        trainer.scheduled(1)
        state, _, _ = trainer.place()
        step = trainer.compile(state, batch)
        if isinstance(step, DrawingStep):
            step.drawn(Interval(time.time(), None, microbatches=0), 0, state)
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
    microbatches' rows of their counts' columns, exactly; its programs' FLOPs are each microbatch's count's
    a microbatch, a run of that count's own at the same split."""
    from dew.training.model_flops import TRAINING_FLOPS, prepare_counts

    model = looped(steps=CAP)
    host = {"text": token_draw()}
    sampled = phased_trainer(model, 0, None, sampled_passes=TRAINED, split=2)
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
    flops = step.drawn(interval, 0, state)
    _, _, metrics, *_ = step(state, batch)
    work = counted[TRAINING_FLOPS]
    expected = sum(jnp.sum(jnp.asarray(work[part::2, count - 1]), dtype=jnp.float32)
                   for part, count in enumerate(step.counts))
    np.testing.assert_array_equal(np.asarray(metrics[TRAINING_FLOPS]), np.asarray(expected))
    own = {}
    for count in set(step.counts):
        alone = phased_trainer(model.with_passes(count), 0, None, split=2)
        start, _, _ = alone.place()
        alone.compile(start, batch)
        own[count] = alone.step_flops
    first, second = (own[count] for count in step.counts)
    assert flops is not None and first is not None and second is not None
    assert flops == 0.5 * first + 0.5 * second


def test_each_count_of_a_sampled_run_steps_at_its_caps_plan(monkeypatch):
    """On devices with room for the one-pass step read whole and for the cap's only in two microbatches, a
    sampled run plans its cap's step at two (`compile`), and admits every other count's at that plan as
    its draws first ask it (`StepCompiler.admitted_at`), the one-pass step's at two microbatches where its
    own search plans it whole: one window, of the cap's split, over every count's step."""
    from test_plan import LIMIT, free, free_until_placed

    from dew.nn.kernels import options

    batch = token_batch()

    def needed(model, split):
        free(monkeypatch, LIMIT)
        trainer = phased_trainer(model, 0, None, split=split)
        state, _, _ = trainer.place()
        trainer.compile(state, batch)
        options.unhold()
        return LIMIT - trainer.plan["probes"][0]["headroom"]

    model = looped(steps=CAP)
    once, whole, halves = needed(model.with_passes(1), 1), needed(model, 1), needed(model, 2)
    assert max(once, halves) < whole, (once, halves, whole)
    free(monkeypatch, (max(once, halves) + whole) // 2)
    sampled = phased_trainer(model, 0, None, sampled_passes=TRAINED)
    state, _, _ = sampled.place()
    step = sampled.compile(state, batch)
    assert isinstance(step, DrawingStep) and step.split == 2
    # From here the devices hold what the steps place, the window and copies the first builds among them,
    # which an admission over the state its call steps charges once (review of rows 7959, 8067: over the
    # state as the step was built, its fit check charged them again beside the devices that held them).
    free_until_placed(monkeypatch, (max(once, halves) + whole) // 2)
    # Each count admitted as its calls first draw it (`DrawingStep.admit`), until every count is.
    interval = Interval(time.time(), None, microbatches=0)
    for index in range(64):
        if len(step.plans) == CAP:
            break
        step.drawn(interval, index, state)
        state, *_ = step(state, batch)
    assert {count: plan.split for count, plan in step.plans.items()} == dict.fromkeys(range(1, CAP + 1), 2)
    alone = phased_trainer(model.with_passes(1), 0, None)
    start, _, _ = alone.place()
    alone.compile(start, batch)
    assert alone.plan["split"] == 1


def test_a_sampled_count_whose_step_returns_other_outputs_is_refused():
    """A count whose step returns outputs other than its cap's (here a metric named by its count) cannot
    pass the one state and pool its step's outputs with the other counts': its admission is refused."""
    class ByCount(LMObjective):
        def loss(self, variables, batch, step):
            stats, aux = super().loss(variables, batch, step)
            named = {**aux.metrics, f"passes_{self.model.loop.steps}": jnp.zeros(())}
            return stats, dataclasses.replace(aux, metrics=named)

    from dew.training import Layout, Trainer

    trainer = Trainer(ByCount(looped(steps=CAP), 16), optax.sgd(0.1), key=jax.random.key(0),
                      layout=Layout(min_shard=1, tolerance=1.0), sampled_passes=TRAINED)
    state, _, _ = trainer.place()
    with pytest.raises(ValueError):
        trainer.compile(state, token_batch())


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
                       layout=Layout(min_shard=1, tolerance=1.0), sampled_passes=TRAINED, split=split,
                       **grouping)

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


def test_a_sampled_runs_step_compiled_ahead_holds_every_count(tmp_path):
    """A run that draws its pass count per microbatch, its step compiled ahead for compile-only GPUs
    (`compile_ahead`), holds beside its cap's step every other count's at the cap's plan, each its own program
    (`AheadOfTime.counts`, row 7924), saved and loaded with it; a run handed it takes each count's
    (`AheadOfTime.at_count`), a count it holds none of refused."""
    from test_compile_ahead import gpus

    from dew.training import MeshSpec
    from dew.training.ahead import AheadOfTime, compile_ahead

    trainer = phased_trainer(looped(steps=CAP), 0, None, sampled_passes=TRAINED,
                             mesh=MeshSpec().build(gpus("sm90")))
    ahead = compile_ahead(trainer, {"text": token_draw()})
    assert sorted(ahead.counts) == list(range(1, CAP))
    programs = {count: step.program for count, (step, _) in ahead.counts.items()}
    assert len({ahead.program, *programs.values()}) == CAP
    ahead.save(tmp_path / "step.dew-aot")
    loaded = AheadOfTime.load(tmp_path / "step.dew-aot")
    assert {count: step.program for count, (step, _) in loaded.counts.items()} == programs
    assert all(loaded.at_count(count).program == program for count, program in programs.items())
    with pytest.raises(ValueError):
        loaded.at_count(CAP)

