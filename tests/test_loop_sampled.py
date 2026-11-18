"""A looped model's pass count drawn per microbatch from a bounded law
(`dew.training.phase_passes.SampledPasses`, row 7660)."""

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from scipy import integrate, stats
from test_pass_weights import looped

from dew.objectives.lm.objective import LMObjective
from dew.training.compiler import SampledStep
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


def trainer_of(model, **options):
    """A trainer of `model`'s LM over 16 tokens, the phased runs' of test_pass_weights."""
    from dew.training import Layout, Trainer

    return Trainer(LMObjective(model, 16), optax.sgd(0.1), key=jax.random.key(0),
                   layout=Layout(min_shard=1, tolerance=1.0), **options)


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


def test_each_microbatch_of_a_sampled_run_steps_as_a_run_of_its_drawn_count():
    """A run drawing its loop's pass count per microbatch, two microbatches an update (`accumulation` 2),
    steps each microbatch as a run of the model at its drawn count steps it (invariant 1): from the same
    state over the same batch, its loss, its metrics and its stepped state, an update's window pooled
    over microbatches of other counts and committed by the second's, the same bits as that count's run's,
    each count's program its own over the one state. Its counts are its key's draws at the microbatches'
    positions, and it draws more than one count."""
    model = looped(steps=CAP)
    batch = token_batch()
    sampled = trainer_of(model, sampled_passes=TRAINED, accumulation=2)
    state, _, _ = sampled.place()
    step = sampled.compile(state, batch)
    assert isinstance(step, SampledStep)
    interval = Interval(time.time(), None, microbatches=0)
    alone = {count: trainer_of(model.with_passes(count), accumulation=2) for count in range(1, CAP + 1)}
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
    four, two = SampledStep({}, flops, 4), SampledStep({}, flops, 2)
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


class Tokens:
    """`tokens` every step, a stream that keeps its place (`dew.data.dataset.Checkpointable`): the steps it
    handed over."""

    def __init__(self, tokens):
        self.tokens, self.steps = jnp.asarray(tokens), 0

    def __iter__(self):
        return self

    def __next__(self):
        self.steps += 1
        return {"text": self.tokens}

    def get_state(self) -> bytes:
        return str(self.steps).encode()

    def set_state(self, state: bytes) -> None:
        self.steps = int(state)


def test_a_sampled_run_resumed_steps_as_the_run_never_stopped(tmp_path):
    """A sampled run of four steps stopped after two and resumed lands, bit for bit, where the run never
    stopped lands: its resume draws on from the microbatches its checkpoint counted."""
    from test_data_mixtures import assert_same, parameters

    from dew.data.dataset import Loader
    from dew.training import Checkpoints

    tokens = np.asarray(token_batch(0)["text"])
    loader = Loader(train=lambda partition: Tokens(tokens), val=None, records=None, batch=len(tokens))

    def fitted(directory, steps):
        return trainer_of(looped(steps=CAP), sampled_passes=TRAINED,
                          checkpoints=Checkpoints(str(directory), keep=8)).fit(
            loader, steps=steps, log_every=100, checkpoint_every=1)

    whole = fitted(tmp_path / "whole", 4)
    fitted(tmp_path / "resumed", 2)
    resumed = fitted(tmp_path / "resumed", 4)
    assert int(resumed.step) == 4
    assert_same(parameters(whole), parameters(resumed), "the resumed sampled run is not the run unstopped")


def test_a_sampled_steps_work_is_its_microbatches_at_their_counts():
    """A sampled step's work (`TRAINING_FLOPS`, which train/mfu divides) is the sum of its microbatches'
    rows' work at their drawn counts (invariant 4): each count's column of the batch's counted work is the
    work a run of that count counts its rows (`prepare_counts`), and the step's metric sums its two
    microbatches' rows of their counts' columns, exactly; its programs' FLOPs are each microbatch's count's
    a microbatch, a run of that count's own at the same split."""
    from dew.training.model_flops import TRAINING_FLOPS, prepare_counts

    model = looped(steps=CAP)
    host = {"text": np.asarray(token_batch()["text"])}
    sampled = trainer_of(model, sampled_passes=TRAINED, split=2)
    counted = prepare_counts(sampled.objective, host, count_model=True,
                             at_counts=sampled.phase_passes.counted_at())
    for count in range(1, CAP + 1):
        alone = trainer_of(model.with_passes(count))
        np.testing.assert_array_equal(counted[TRAINING_FLOPS][:, count - 1],
                                      prepare_counts(alone.objective, host, count_model=True)[TRAINING_FLOPS])
    batch = {name: jnp.asarray(value) for name, value in counted.items()}
    state, _, _ = sampled.place()
    step = sampled.compile(state, batch)
    assert isinstance(step, SampledStep)
    interval = Interval(time.time(), None, microbatches=0)
    flops = step.drawn(TRAINED, sampled.key, interval, 0)
    _, _, metrics, *_ = step(state, batch)
    work = counted[TRAINING_FLOPS]
    expected = sum(jnp.sum(jnp.asarray(work[part::2, count - 1]), dtype=jnp.float32)
                   for part, count in enumerate(step.counts))
    np.testing.assert_array_equal(np.asarray(metrics[TRAINING_FLOPS]), np.asarray(expected))
    own = {}
    for count in set(step.counts):
        alone = trainer_of(model.with_passes(count), split=2)
        start, _, _ = alone.place()
        alone.compile(start, batch)
        own[count] = alone.step_flops
    first, second = (own[count] for count in step.counts)
    assert flops is not None and first is not None and second is not None
    assert flops == 0.5 * first + 0.5 * second
