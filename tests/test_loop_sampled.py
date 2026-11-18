"""A looped model's pass count drawn per microbatch from a bounded law
(`dew.training.phase_passes.SampledPasses`, row 7660)."""

import dataclasses
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from scipy import integrate, stats
from test_pass_weights import looped, phased_trainer

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_stack import Loop
from dew.objectives.base import Ratio
from dew.objectives.lm.objective import LMObjective
from dew.training.compiler import DrawingStep
from dew.training.interval import DISPATCHED_MICROBATCHES, Interval
from dew.training.phase_passes import SampledPasses

LAWS = [SampledPasses(mean=2.0, dispersion=0.5, cap=4), SampledPasses(mean=4.0, dispersion=0.5, cap=8),
        SampledPasses(mean=3.0, dispersion=0.0, cap=6)]
"""The note's law (mean 2, dispersion 0.5, cap 4), a wider one, and a Poisson count (no dispersion)."""

DRAWS = 2_500
"""The microbatches a seed's key draws a law's counts at."""

SEEDS = 8
"""The keys a law's shares are read over (`sampling_gate.assert_draws_as_declared`)."""


SPAN = 39
"""The standard deviations either side of the log rate's mean past which its normal density is 0 in
float64 (exp(-39² / 2) underflows past the least subnormal), so a quadrature over them misses nothing."""


def probabilities(law: SampledPasses) -> np.ndarray:
    """The bounded law's probability at each count 1 to its cap, from its statement alone: the Poisson's
    closed form where it has no dispersion, else the Poisson's probability integrated over the log-normal
    rate (scipy's adaptive quadrature in float64, over the log rate's `SPAN`), conditioned on the cap."""
    location = math.log(law.mean - 1) - law.dispersion ** 2 / 2
    if law.dispersion == 0:
        unbounded = stats.poisson.pmf(np.arange(law.cap), math.exp(location))
    else:
        reach = SPAN * law.dispersion
        unbounded = np.array([integrate.quad(
            lambda x, n=n: stats.poisson.pmf(n, np.exp(x)) * stats.norm.pdf(x, location, law.dispersion),
            location - reach, location + reach, points=[location])[0] for n in range(law.cap)])
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


def token_batch(seed: int = 2) -> dict:
    """Two rows a device of 17 tokens."""
    rows = jax.device_count() * 2
    return {"text": jnp.asarray(np.random.default_rng(seed).integers(0, 256, (rows, 17)), jnp.int32)}


def held(leaf):
    """A state's leaf as its bits compare: a key's words (`jax.random.key_data`), any other leaf itself."""
    return jax.random.key_data(leaf) if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key) else leaf


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
        step.drawn(TRAINED, sampled.key, interval, index)
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
        ours, theirs = (jax.tree_util.tree_flatten_with_path(jax.device_get(jax.tree.map(held, tree)))[0]
                        for tree in (state, own))
        unlike = [jax.tree_util.keystr(path) for (path, a), (_, b) in zip(ours, theirs, strict=True)
                  if not np.array_equal(a, b)]
        assert not unlike, f"microbatch {index} at {count} passes: {unlike}"
    assert drawn == [TRAINED.count(words(sampled.key), position) for position in range(8)]
    assert len(set(drawn)) > 1 and interval.microbatches == 8


def test_a_sampled_runs_draws_go_on_through_another_split_and_a_resume():
    """The draws continue one stream of the run's dispatched microbatches (`Interval.microbatches`): ten
    steps of four microbatches, then, resumed from a checkpoint's control, five of two, draw at positions 0
    to 49, not where a step and its split would place them; a checkpoint that counted none resumes at its
    step's microbatches, at the split its step reads."""
    key = jax.random.key(0)
    flops = dict.fromkeys(range(1, CAP + 1))
    four, two = DrawingStep({}, flops, 4), DrawingStep({}, flops, 2)
    interval = Interval(time.time(), None, microbatches=0)
    drawn = []
    for index in range(10):
        four.drawn(TRAINED, key, interval, index)
        drawn += four.counts
    control = {DISPATCHED_MICROBATCHES: interval.microbatches}
    resumed = Interval.resumed(dict(control), last_saved=10)
    for index in range(10, 15):
        two.drawn(TRAINED, key, resumed, index)
        drawn += two.counts
    assert drawn == [TRAINED.count(words(key), position) for position in range(50)]
    lacking = Interval.resumed({}, last_saved=7)
    two.drawn(TRAINED, key, lacking, 7)
    assert two.counts == [TRAINED.count(words(key), position) for position in (14, 15)]
    assert lacking.microbatches == 16


def test_a_sampled_run_resumed_steps_as_the_run_never_stopped(tmp_path):
    """A sampled run of four steps over data read in phases (`PhasedStream`), stopped after two and resumed,
    lands, bit for bit, where the run never stopped lands: each microbatch draws its count whatever its
    data's phase, and its resume draws on from the microbatches its checkpoint counted."""
    from test_data_mixtures import assert_same, parameters
    from test_pass_weights import phased_draw, phased_loader

    from dew.training import Checkpoints

    tokens, _ = phased_draw(0)

    def fitted(directory, steps):
        return phased_trainer(looped(steps=CAP), 0, None, sampled_passes=TRAINED,
                          checkpoints=Checkpoints(str(directory), keep=8)).fit(
            phased_loader(tokens), steps=steps, log_every=100, checkpoint_every=1)

    whole = fitted(tmp_path / "whole", 4)
    fitted(tmp_path / "resumed", 2)
    resumed = fitted(tmp_path / "resumed", 4)
    assert int(resumed.step) == 4
    assert_same(parameters(whole), parameters(resumed), "the resumed sampled run is not the run unstopped")


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
            step.drawn(TRAINED, trainer.key, Interval(time.time(), None, microbatches=0), 0)
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
    host = {"text": np.asarray(token_batch()["text"])}
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
    flops = step.drawn(TRAINED, sampled.key, interval, 0)
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
    sampled run plans its cap's step at two (`compile`), and admits every other count's at that plan
    (`StepCompiler.admitted_at`), the one-pass step's at two microbatches where its own search plans it
    whole: one window, of the cap's split, over every count's step."""
    from test_plan import LIMIT, free

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
