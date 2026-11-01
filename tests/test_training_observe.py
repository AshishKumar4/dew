"""`Trainer.fit(observe=Observe(every, call))`: values of the state taken inside one fit, at a
cadence of their own and at the run's last step, in place of a fit called once per stage.

An observation is copied home before the next step donates the state it was taken from, so the
EMA tree a call returns is a snapshot. A resumed fit drops what a failed run observed past its
restored step, and a fit that finds the run complete observes its final state once.
"""

import jax
import numpy as np
import optax
import pytest
from affine_run import Regression, dataset
from recording import preempting

from dew.training import Checkpoints, Layout, Observe, Trainer
from dew.training.runtime import Preempted


def trainer(directory=None, tracker=None) -> Trainer:
    return Trainer(Regression(), optax.sgd(0.1), key=jax.random.key(0), layout=Layout(min_shard=1),
                   checkpoints=None if directory is None else Checkpoints(str(directory)), tracker=tracker)


def averaged(state):
    return state.averaged


def kernels(observe: Observe) -> dict[int, np.ndarray]:
    return {step: value["params"]["Dense_0"]["kernel"] for step, value in observe.values}


def kernel(state) -> np.ndarray:
    return np.asarray(state.averaged["params"]["Dense_0"]["kernel"])


def test_an_observer_keeps_each_cadenced_steps_average_and_the_last_as_a_staged_fit_does():
    """Every second step's EMA and the last step's, though five is off the cadence, each a host
    copy equal to what a fit stopped at that step returns: what tutorials took by calling fit
    once a stage."""
    observe = Observe(2, averaged)
    final = trainer().fit(dataset(), steps=5, log_every=1, observe=observe)
    assert [step for step, _ in observe.values] == [2, 4, 5]
    assert all(isinstance(leaf, np.ndarray) for _, value in observe.values for leaf in jax.tree.leaves(value))
    for step in (2, 4):
        staged = trainer().fit(dataset(), steps=step, log_every=1)
        np.testing.assert_array_equal(kernels(observe)[step], kernel(staged))
    np.testing.assert_array_equal(kernels(observe)[5], kernel(final))
    assert not np.array_equal(kernels(observe)[2], kernels(observe)[4])


def test_a_resumed_fit_continues_its_observers_ledger_and_a_complete_run_is_observed_once(tmp_path):
    """A fit resumed at step 4 drops what a failed run observed past it, steps it will not observe
    again among them, and observes on; a fit that finds the run complete calls the observer once
    for its final state, and not again when asked again."""
    observe = Observe(2, averaged)
    trainer(tmp_path / "run").fit(dataset(), steps=4, log_every=1, checkpoint_every=2, observe=observe)
    assert [step for step, _ in observe.values] == [2, 4]
    stale = jax.tree.map(np.zeros_like, observe.values[-1][1])
    observe.values += [(5, stale), (7, stale)]
    resumed = trainer(tmp_path / "run").fit(dataset(), steps=6, log_every=1, checkpoint_every=2,
                                            observe=observe)
    assert [step for step, _ in observe.values] == [2, 4, 6]
    np.testing.assert_array_equal(kernels(observe)[6], kernel(resumed))

    calls = []
    fresh = Observe(2, lambda state: calls.append(int(state.step)) or averaged(state))
    for _ in range(2):
        trainer(tmp_path / "run").fit(dataset(), steps=6, log_every=1, observe=fresh)
    assert calls == [6] and [step for step, _ in fresh.values] == [6]
    np.testing.assert_array_equal(kernels(fresh)[6], kernels(observe)[6])


def test_a_preempted_fit_observes_the_step_it_stopped_at(tmp_path):
    """A preemption stops the run off the cadence, after step 4: that step, the one its
    checkpoint holds, is the run's last and is observed before fit raises."""
    observe = preempting(3, every=10, call=averaged)
    with pytest.raises(Preempted):
        trainer(tmp_path / "run").fit(dataset(), steps=20, log_every=1, observe=observe)
    assert [step for step, _ in observe.values] == [4]
    assert Checkpoints(str(tmp_path / "run")).latest == 4


def test_a_failing_observation_fails_the_fit_and_a_cadence_below_one_is_refused():
    def failing(state):
        raise ZeroDivisionError("the observer's own failure")

    with pytest.raises(ZeroDivisionError):
        trainer().fit(dataset(), steps=2, log_every=1, observe=Observe(1, failing))
    with pytest.raises(ValueError):
        Observe(0, averaged)
