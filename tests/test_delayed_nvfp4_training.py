"""A delayed-scaling NVFP4 run's amax histories as its training steps move them (row 12721): each projection
the recipe takes reads its history and records its rows' largest magnitude a call; the trainer takes the
largest across a step's calls and microbatches and rolls it into the history once per committed update."""

import numpy as np
from delayed_nvfp4 import DELAYED, data, delayed_lm, histories

from dew.checkpoints import Checkpoints
from dew.nn.kernels import generation


def test_a_split_step_rolls_the_largest_its_microbatches_recorded_into_each_history_once_an_update():
    """A step read whole and the same step read in two microbatches roll the same largest magnitude into
    each history, newest first (the window's largest is the largest of its microbatches', so the history
    does not depend on the split), once an update: after one step each history holds one recorded value and
    zeros, and after the next, the first step's value moved one place on, behind the second's."""
    with generation.run_precision(DELAYED):
        whole, halves = delayed_lm(split=1), delayed_lm(split=2)
        started = histories(whole.place()[0])
        first = [whole.fit(data(), steps=1), halves.fit(data(), steps=1)]
        # The split run's next step, from where its first left its histories.
        second = halves.fit(data(), state=first[1], steps=2)
    assert started and all(not history.any() for history in started), started
    for one, other in zip(histories(first[0]), histories(first[1]), strict=True):
        np.testing.assert_array_equal(one, other)
        assert one[0] > 0 and not one[1:].any(), one
    for after, before in zip(histories(second), histories(first[1]), strict=True):
        np.testing.assert_array_equal(after[1], before[0])
        assert after[0] > 0 and not after[2:].any(), after


def test_a_resumed_run_restores_its_amax_histories(tmp_path):
    """A delayed-scaling NVFP4 run's histories are state its checkpoint keeps: a run resumed from it reads
    them as the run that saved them held them, and its next step rolls them on."""
    with generation.run_precision(DELAYED):
        saved = delayed_lm(1, Checkpoints(str(tmp_path))).fit(data(), steps=2, checkpoint_every=1)
        fresh = delayed_lm(1, Checkpoints(str(tmp_path)))
        restored = fresh.place()[0]
        resumed = fresh.fit(data(), steps=3, checkpoint_every=1)
    for one, other in zip(histories(saved), histories(restored), strict=True):
        np.testing.assert_array_equal(one, other)
    for after, before in zip(histories(resumed), histories(saved), strict=True):
        np.testing.assert_array_equal(after[1:3], before[:2])
        assert after[0] > 0, after
