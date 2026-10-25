"""The learning gate (tests/learning_gate.py): a trainer at its margin passes at the gate's level, one with
room passes, one short of a claim fails, and a site that cannot be judged is refused."""

import numpy as np
import pytest
from learning_gate import FEWEST_RUNS, LEVEL, LearningFailure, assert_learns, t_statistics, threshold, within

VERDICTS = 2_000_000
"""The verdicts the level's measurement takes, LEVEL times which is 200 expected passes."""


def passes_at_the_margin(draw, runs: int) -> int:
    """How many of VERDICTS verdicts on `runs` runs a claim passes, its contrasts `draw(rng, shape)`."""
    rng = np.random.default_rng([0x1EA2, runs])
    passed = 0
    for start in range(0, VERDICTS, 250_000):
        contrasts = draw(rng, (runs, min(250_000, VERDICTS - start)))
        passed += int(np.sum(t_statistics(contrasts) >= threshold(runs)))
    return passed


@pytest.mark.parametrize("runs", [FEWEST_RUNS, 8, 16])
def test_a_trainer_at_its_margin_passes_at_the_gates_level(runs):
    """Runs whose contrasts are normal about zero, a trainer meeting its claim exactly at the old margin
    on average, pass with chance LEVEL: over two million verdicts, the passes within six of their
    standard deviations of LEVEL's share."""
    passed = passes_at_the_margin(lambda rng, shape: rng.standard_normal(shape), runs)
    assert abs(passed - LEVEL * VERDICTS) <= 6 * np.sqrt(LEVEL * VERDICTS), passed


def test_a_folded_contrast_would_pass_a_trainer_at_its_margin_far_more_often():
    """The margin less a normal run's absolute departure, shifted to mean zero, over eight runs: its long
    left tail passes a trainer at the margin more than twice as often as LEVEL, which is why a two-sided
    margin states its two one-sided contrasts (`within`)."""
    expected = np.sqrt(2 / np.pi)
    passed = passes_at_the_margin(lambda rng, shape: expected - np.abs(rng.standard_normal(shape)), 8)
    assert passed > 2 * LEVEL * VERDICTS, passed


def test_a_trainer_with_room_passes_and_one_short_of_a_claim_fails():
    """Eight runs meeting a claim with room pass; a second claim a run's spread short of its margin on
    average fails, named, and a claim met by runs that all agree passes."""
    def runs(shift):
        def contrasts(run):
            rng = np.random.default_rng([0x1EA3, run])
            return {"room": 0.03 + 0.005 * rng.standard_normal(),
                    "short": shift + 0.005 * rng.standard_normal(), "exact": 0.01}
        return contrasts

    assert_learns(runs(0.03), "room", runs=8)
    with pytest.raises(LearningFailure, match="short"):
        assert_learns(runs(-0.005), "short", runs=8)


def test_within_states_a_margin_as_two_one_sided_contrasts():
    """A value a quarter above its target within a margin of a half: a quarter short of the upper margin,
    three quarters past the lower; a value past the margin makes its side negative."""
    assert sorted(within("mean", 0.75, 0.5, 0.5).values()) == [0.25, 0.75]
    assert sorted(within("mean", 1.25, 0.5, 0.5).values()) == [-0.25, 1.25]


def test_a_site_that_cannot_be_judged_is_refused():
    """Fewer than FEWEST_RUNS runs, runs that state other claims and a contrast that is not finite are
    refused, not judged."""
    with pytest.raises(ValueError):
        assert_learns(lambda run: {"claim": 1.0}, "few", runs=FEWEST_RUNS - 1)
    with pytest.raises(ValueError):
        assert_learns(lambda run: {f"claim {min(run, 1)}": 1.0}, "claims", runs=4)
    with pytest.raises(ValueError):
        assert_learns(lambda run: {"claim": float("nan") if run == 2 else 1.0}, "nan", runs=4)
