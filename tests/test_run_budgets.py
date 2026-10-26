"""A run's length as a count of examples, a pause partway through its budget, and a run whose data
runs out before its length."""

import collections
import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from recording import RecordingTracker
from reference_error import assert_as_exact

from dew.config import OptimConfig, RunConfig, TrainerConfig
from dew.data import Memory
from dew.data.compile import compile
from dew.data.dataset import Closeable, DataPartition, Loader
from dew.data.pipeline import Pipeline
from dew.objectives import Aux, Objective
from dew.objectives.base import VALID_ROWS
from dew.optim import Linear
from dew.training import Best, Checkpoints, Pause, Trainer

FEATURES = 4
RECORDS = 10
BATCH = 8
EXAMPLES = 41
"""Four passes over the records and one more record, which fill five batches and one of a single record:
three updates of two steps."""
ACCUMULATION = 2
PEAK = 0.2

EXAMPLES_SEEDS = 64
"""Check 1's declared seeds for an examples run against the reference's Adam, sized by `seeds_for`."""
PAUSE_SEEDS = 8
"""Check 1's declared seeds for a paused run against the whole one, sized by `seeds_for`."""


class Pulled(Objective):
    """A weight pulled toward each batch's real rows: the mean over them and their coordinates of
    (w - x)^2, whose gradient over an update's window is 2 (w - the mean of its rows) / FEATURES."""

    def fresh_variables(self, key, held):
        return {"params": {"w": jnp.zeros(FEATURES)}}

    def loss(self, variables, batch, step):
        return self.row_mean(jnp.square(variables["params"]["w"] - batch["x"]), batch), Aux({})


def records(seed: int) -> tuple[np.ndarray]:
    """The corpus of one draw: RECORDS rows of FEATURES coordinates from a unit normal."""
    return (np.random.default_rng(seed).normal(size=(RECORDS, FEATURES)).astype(np.float32),)


def chain(x: np.ndarray) -> Pipeline:
    return Pipeline(Memory({"x": x})).batch(BATCH)


def descended(x):
    """The losses of each step and the weight after them, of Adam on `Pulled` over the first EXAMPLES
    records of `x` read in order, pass after pass, in batches of BATCH (the last holding the one
    record left) and updates of ACCUMULATION batches, at a rate falling linearly from PEAK to 0 over
    the updates, in `x`'s own dtype: the reference arm, and in float64 the truth. A step reports its
    batch's loss, and the step that closes an update the loss of the whole update's rows."""
    rows = x[np.arange(EXAMPLES) % RECORDS]
    batches = [rows[start:start + BATCH] for start in range(0, EXAMPLES, BATCH)]
    updates = len(batches) // ACCUMULATION
    w = first = second = jnp.zeros(FEATURES, x.dtype)
    losses = []
    for update in range(updates):
        window = batches[update * ACCUMULATION:(update + 1) * ACCUMULATION]
        held = jnp.concatenate(window)
        losses += [jnp.mean(jnp.square(w - batch)) for batch in window[:-1]]
        losses.append(jnp.mean(jnp.square(w - held)))
        gradient = 2 * jnp.sum(w - held, axis=0) / (len(held) * FEATURES)
        first, second = 0.9 * first + 0.1 * gradient, 0.999 * second + 0.001 * jnp.square(gradient)
        moved = (first / (1 - 0.9 ** (update + 1))) / (jnp.sqrt(second / (1 - 0.999 ** (update + 1))) + 1e-8)
        w = w - PEAK * (1 - update / updates) * moved
    return {"w": w, "loss": jnp.stack(losses)}


def trainer(directory=None, tracker=None) -> Trainer:
    return Trainer(Pulled(), OptimConfig(optimizer="adam", learning_rate=Linear(peak_value=PEAK)), key=0,
                   accumulation=ACCUMULATION, tracker=tracker,
                   checkpoints=None if directory is None else Checkpoints(str(directory)))


def losses(tracker: RecordingTracker) -> list[float]:
    return [scalars["train/loss"] for _, scalars in tracker.scalars if "train/loss" in scalars]


def trained(x, directory=None, stops: tuple[Pause | None, ...] = (None,)):
    """The losses of every step and the final weight of an examples run over `x`, fitted once under
    each of `stops` in turn, each fit resuming the one before from `directory`."""
    stepped, state = [], None
    for stop in stops:
        tracker = RecordingTracker()
        state = trainer(directory, tracker).fit(chain(x), examples=EXAMPLES, log_every=1, stop=stop)
        stepped += losses(tracker)
    assert state is not None
    return {"w": np.asarray(state.variables["params"]["w"]), "loss": np.asarray(stepped)}


def ids_read(loader: Loader, steps: int, field: str) -> collections.Counter:
    """The real values of `field` in the first `steps` training batches of `loader`, which ends there."""
    stream = loader.train(DataPartition())
    read: collections.Counter = collections.Counter()
    try:
        for _ in range(steps):
            batch = next(stream)
            real = np.asarray(batch.get(VALID_ROWS, np.ones(len(batch[field]), bool)), bool)
            values = np.asarray(batch[field])[real]
            read.update(values[values > 0].tolist())
        with pytest.raises(StopIteration):
            next(stream)
    finally:
        if isinstance(stream, Closeable):
            stream.close()
    return read


@pytest.mark.parametrize("examples", [7, 23])
def test_an_examples_budget_reads_each_record_once_and_its_read_ends_there(examples):
    """A read bounded to its first `examples` records, shuffled pass after pass, hands each out once
    for each pass it reaches, within its steps, and ends: a count under one pass reads each record of
    it once, and one over two reads every record two or three times, the third time `examples` mod
    RECORDS of them."""
    ids = np.arange(1, RECORDS + 1, dtype=np.int32)
    loader = compile(Pipeline(Memory({"id": ids})).shuffle(3).batch(BATCH), examples=examples)
    read = ids_read(loader, loader.example_steps(examples), "id")
    passes, rest = divmod(examples, RECORDS)
    assert sum(read.values()) == examples
    assert collections.Counter(read.values()) == collections.Counter(
        {passes + 1: rest, passes: RECORDS - rest} if passes else {1: rest})


def test_an_examples_budget_counts_documents_before_they_are_packed():
    """Documents packed into windows as they arrive are counted before the pack: the read bounded to 13
    of 5 documents of 1 to 5 tokens holds each document's tokens once for each time the bound reaches
    it, in the windows of the steps one pass over it counts, and ends."""
    documents = [{"text": np.full(length, length, np.int32)} for length in range(1, 6)]
    loader = compile(Pipeline(Memory(documents)).pack(8).batch(2), examples=13)
    read = ids_read(loader, loader.example_steps(13), "text")
    reached = collections.Counter(1 + index % 5 for index in range(13))
    assert read == {token: count * token for token, count in reached.items()}


def test_an_examples_run_trains_each_record_once_under_the_schedule_its_plan_sizes():
    """An examples run takes the steps its data plan counts, sizes its schedule's horizon from them and
    trains on each record as many times as the bound reaches it: its losses and weight match SGD on
    the same records at a rate falling to 0 over those updates (check 1)."""
    assert_as_exact(trained, descended, records, "examples run", seeds=EXAMPLES_SEEDS)


def test_a_paused_run_resumed_trains_on_as_one_that_never_paused(tmp_path):
    """A run paused halfway through its budget and resumed without the pause lands where the whole
    run does, step for step and in its weight (check 1): the budget, not the pause, sized its
    schedule, and its checkpoint held its data position."""
    directories = iter(range(1 << 20))
    assert_as_exact(lambda x: trained(x, tmp_path / str(next(directories)), (Pause(0.5), None)), trained,
                    records, "paused run", seeds=PAUSE_SEEDS, truth_fn=descended)


def test_a_run_paused_where_it_already_stands_trains_nothing_more(tmp_path):
    """A paused run fitted again with the same pause is already where it pauses: it returns the
    checkpointed state at the pause, two of the three updates in, and trains no further."""
    (x,) = records(0)
    first = trainer(tmp_path).fit(chain(x), examples=EXAMPLES, stop=Pause(0.5))
    again = trainer(tmp_path).fit(chain(x), examples=EXAMPLES, stop=Pause(0.5))
    assert int(first.step) == int(again.step) == 2 * ACCUMULATION
    assert Checkpoints(str(tmp_path)).latest == 2 * ACCUMULATION


def test_a_pause_on_a_step_that_ranks_a_weights_only_best_writes_the_whole_state(tmp_path):
    """A pause whose step also ranks a best kept as weights alone writes that step's whole state and
    data position, which the run resumed without the pause goes on from to the end of its budget."""
    (x,) = records(0)
    objective = Pulled()

    def fitted(stop):
        return Trainer(objective, OptimConfig(optimizer="adam", learning_rate=Linear(peak_value=PEAK)), key=0,
                       accumulation=ACCUMULATION, checkpoints=Checkpoints(str(tmp_path))).fit(
            chain(x), examples=EXAMPLES, stop=stop, eval_every=2 * ACCUMULATION,
            best=Best(objective.loss, weights_only=True))

    assert int(fitted(Pause(0.5)).step) == Checkpoints(str(tmp_path)).latest == 2 * ACCUMULATION
    assert int(fitted(None).step) == 3 * ACCUMULATION


def test_an_examples_count_whose_last_steps_make_no_whole_update_is_refused():
    """Thirty-five records fill five batches, which leave a step no update takes, so the run is refused
    before it trains rather than read records it never learns from."""
    (x,) = records(0)
    with pytest.raises(ValueError):
        trainer().fit(chain(x), examples=35)


def test_a_stream_that_runs_out_on_an_evaluated_step_scores_that_step_once():
    """A stream of three batches under a run of five steps, evaluated at every step, ends at step 3,
    whose evaluation on its cadence is the run's last: each step's split is scored once."""
    (x,) = records(0)
    tracker = RecordingTracker()
    batch = {"x": x[:BATCH]}
    finite = Loader(lambda partition: iter([batch] * 3), lambda partition: iter([batch]), None, BATCH)
    Trainer(Pulled(), OptimConfig(optimizer="adam", learning_rate=0.1), key=0, tracker=tracker).fit(
        finite, steps=5, eval_every=1, log_every=1)
    assert [step for step, scalars in tracker.scalars if "val/loss" in scalars] == [1, 2, 3]


def test_a_stream_that_runs_out_after_a_ranked_step_ends_on_a_whole_checkpoint(tmp_path):
    """A one-batch stream under a run of ten steps, its step ranking a best kept as weights alone,
    ends at step 1 on a checkpoint the run resumes from: the whole state and its data position."""
    (x,) = records(0)
    objective = Pulled()
    finite = Loader(lambda partition: iter([{"x": x[:BATCH]}]), None, None, BATCH)
    Trainer(objective, OptimConfig(optimizer="adam", learning_rate=0.1), key=0,
            checkpoints=Checkpoints(str(tmp_path))).fit(
        finite, steps=10, eval_every=1, best=Best(objective.loss, weights_only=True))
    assert Checkpoints(str(tmp_path)).latest == 1


def test_a_run_resumed_where_its_finite_stream_ran_out_ends_there_with_nothing_to_read(tmp_path):
    """A one-pass read of the corpus under a run asked for more steps than it holds ends where the read
    runs out, on a checkpoint; resumed with a higher cap it opens the read where it stopped, finds it
    has run out and ends at the same step, its state untouched, rather than refusing a stream that holds
    no batch."""
    (x,) = records(0)

    def run(steps):
        trainer = Trainer(Pulled(), OptimConfig(optimizer="adam", learning_rate=0.1), key=0,
                          checkpoints=Checkpoints(str(tmp_path)))
        return trainer.fit(compile(Pipeline(Memory({"x": x})).repeat(1).batch(BATCH), read="memory"),
                           steps=steps)

    first, again = run(10), run(12)
    assert 0 < int(first.step) == int(again.step) < 10
    np.testing.assert_array_equal(again.variables["params"]["w"], first.variables["params"]["w"])


def test_a_stream_that_runs_out_between_cadences_scores_only_the_suites_its_last_step_did_not():
    """A one-batch stream under a run of five steps ends at step 1, where the suite scored every step
    already reported and the one scored every other step did not: the run's end scores that one alone,
    each suite once at step 1."""
    from dew.training import EvalSuite

    (x,) = records(0)
    tracker = RecordingTracker()
    batch = {"x": x[:BATCH]}

    def held(partition):
        return iter([batch])

    finite = Loader(lambda partition: iter([batch]), None, None, BATCH)
    Trainer(Pulled(), OptimConfig(optimizer="adam", learning_rate=0.1), key=0, tracker=tracker).fit(
        finite, steps=5, log_every=1, validation={"often": EvalSuite(held, every=1),
                                                   "rare": EvalSuite(held, every=2)})
    reported = [(step, name) for step, scalars in tracker.scalars for name in ("often", "rare")
                if f"{name}/loss" in scalars]
    assert sorted(reported) == [(1, "often"), (1, "rare")]


def test_a_stream_that_runs_out_with_no_final_checkpoint_writes_its_weights_only_best_whole(tmp_path):
    """A one-batch stream under a run of five steps that writes no final checkpoint ends at step 1,
    whose closing evaluation ranks its first best, kept as weights alone: the stream is closed by then,
    so the run reads no further batch, and the best it writes is the whole state of its last step."""
    (x,) = records(0)
    objective = Pulled()
    finite = Loader(lambda partition: iter([{"x": x[:BATCH]}]), lambda partition: iter([{"x": x[:BATCH]}]),
                    None, BATCH)
    Trainer(objective, OptimConfig(optimizer="adam", learning_rate=0.1), key=0,
            checkpoints=Checkpoints(str(tmp_path))).fit(
        finite, steps=5, checkpoint_final=False, best=Best(objective.loss, weights_only=True))
    assert Checkpoints(str(tmp_path)).latest == 1


def test_a_run_config_counts_examples_and_pauses_as_a_fit_does(tmp_path):
    """A run config's `examples` and `stop` reach the fit, its record holding both: paused halfway it
    stands two updates in, and run again without the pause it goes on to the steps its examples fill."""
    (x,) = records(0)
    optim = OptimConfig(optimizer="adam", learning_rate=Linear(peak_value=PEAK))
    config = RunConfig(optim=optim, trainer=TrainerConfig(
        checkpoint_dir=str(tmp_path), examples=EXAMPLES, stop=Pause(0.5), accumulation=ACCUMULATION,
        log_every=1, eval_every=None, checkpoint_every=None))
    assert RunConfig.from_dict(config.to_dict()) == config
    flags = RunConfig.cli(["--trainer.examples", str(EXAMPLES), "--trainer.stop.fraction", "0.5"]).trainer
    assert (flags.examples, flags.stop) == (EXAMPLES, Pause(0.5))
    paused = config.train(Pulled(), chain(x), name="budget")
    resumed = dataclasses.replace(config, trainer=dataclasses.replace(config.trainer, stop=None)).train(
        Pulled(), chain(x), name="budget")
    assert (int(paused.step), int(resumed.step)) == (2 * ACCUMULATION, 3 * ACCUMULATION)
