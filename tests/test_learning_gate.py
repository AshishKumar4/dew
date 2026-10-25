"""The learning gate (tests/learning_gate.py): its verdicts fail a trainer as often as the t law says, a
trainer with room passes, one short of a claim fails, its sizing declares the units that keep a trainer
like its pilot's false reds under FALSE_RED, and a site that cannot be judged is refused."""

import numpy as np
import pytest
from learning_gate import (
    FALSE_RED,
    FEWEST_UNITS,
    LearningFailure,
    assert_learns,
    false_red,
    t_statistics,
    threshold,
    units_for,
    within,
)
from scipy import stats

VERDICTS = 400_000
"""The verdicts a rate's measurement takes."""


def failed(draw, units: int, needed: float) -> int:
    """How many of VERDICTS verdicts on `units` units fail a claim held to `needed`, its contrasts
    `draw(rng, shape)`."""
    rng = np.random.default_rng([0x1EA2, units])
    count = 0
    for start in range(0, VERDICTS, 100_000):
        contrasts = draw(rng, (units, min(100_000, VERDICTS - start)))
        count += int(np.sum(~(t_statistics(contrasts) >= needed)))
    return count


@pytest.mark.parametrize("units, effect", [(FEWEST_UNITS, 6.0), (16, 3.0), (64, 1.2)])
def test_a_verdict_fails_a_trainer_as_often_as_the_noncentral_t_law_says(units, effect):
    """Units whose contrasts are normal at `effect` times their spread fail the gate's threshold as often
    as `false_red` says, the noncentral t law's chance at that effect: over 400,000 verdicts, within six
    of the count's standard deviations. At an effect of zero the same law is the central one, which
    the threshold's own definition at LEVEL is."""
    expected = false_red(np.asarray([effect]), units)
    count = failed(lambda rng, shape: effect + rng.standard_normal(shape), units, threshold(units))
    assert 0.01 < expected < 0.99
    assert abs(count - expected * VERDICTS) <= 6 * np.sqrt(expected * (1 - expected) * VERDICTS), count


def test_a_folded_contrast_passes_a_claim_at_its_margin_more_often_than_the_t_law_says():
    """The margin less a normal unit's absolute departure, shifted to mean zero, over eight units, held to
    the t law's quantile at one in a thousand: its long left tail passes more than twice as often as
    the normal contrasts `within` makes do, which is why a two-sided margin is two one-sided claims."""
    needed = float(stats.t.isf(1e-3, 7))
    mean = np.sqrt(2 / np.pi)
    folded = VERDICTS - failed(lambda rng, shape: mean - np.abs(rng.standard_normal(shape)), 8, needed)
    normal = VERDICTS - failed(lambda rng, shape: rng.standard_normal(shape), 8, needed)
    assert folded > 2 * normal > 0, (folded, normal)


def test_a_trainer_with_room_passes_and_one_short_of_a_claim_fails():
    """Sixteen units meeting a claim with room pass; a second claim a unit's spread short of its margin
    on average fails, named; a claim every unit meets alike passes."""
    def units(shift):
        def contrasts(numbers):
            rng = np.random.default_rng([0x1EA3, *numbers.tolist()])
            return {"room": 0.05 + 0.005 * rng.standard_normal(numbers.size),
                    "short": shift + 0.005 * rng.standard_normal(numbers.size),
                    "exact": np.full(numbers.size, 0.01)}
        return contrasts

    assert_learns(units(0.05), "room", units=16)
    with pytest.raises(LearningFailure, match="short"):
        assert_learns(units(-0.005), "short", units=16)


def test_within_states_a_margin_as_two_one_sided_contrasts():
    """Values a quarter above and three quarters above their target within a margin of a half: a quarter
    and minus a quarter short of the upper margin, three quarters and one and a quarter past the lower."""
    sides = within("mean", np.asarray([0.75, 1.25]), 0.5, 0.5)
    assert sorted(sides) == ["mean above its lower margin", "mean below its upper margin"]
    np.testing.assert_array_equal(sides["mean below its upper margin"], [0.25, -0.25])
    np.testing.assert_array_equal(sides["mean above its lower margin"], [0.75, 1.25])


def test_sizing_declares_the_fewest_units_that_keep_a_trainer_like_its_pilots_false_reds_under_false_red():
    """From a pilot of 4096 units at an effect of one and of two, `units_for` declares the least units at
    which `false_red` at the pilot's effects is at most FALSE_RED, one fewer above it; a pilot that
    misses a claim on average is refused."""
    rng = np.random.default_rng(0x1EA4)
    pilot = {"one": 1.0 + rng.standard_normal(4096), "two": 2.0 + rng.standard_normal(4096)}
    units = units_for(pilot, "pilot")
    effects = np.asarray([pilot[name].mean() / pilot[name].std(ddof=1) for name in sorted(pilot)])
    assert false_red(effects, units) <= FALSE_RED < false_red(effects, units - 1)
    with pytest.raises(AssertionError):
        units_for({"missed": -0.1 + rng.standard_normal(4096)}, "missed")


def test_a_site_that_cannot_be_judged_is_refused():
    """Fewer than FEWEST_UNITS units, contrasts of another count than the units and a contrast that is not
    finite are refused, not judged."""
    with pytest.raises(ValueError):
        assert_learns(lambda numbers: {"claim": np.ones(numbers.size)}, "few", units=FEWEST_UNITS - 1)
    with pytest.raises(ValueError):
        assert_learns(lambda numbers: {"claim": np.ones(numbers.size + 1)}, "count", units=FEWEST_UNITS)
    with pytest.raises(ValueError):
        assert_learns(lambda numbers: {"claim": np.where(numbers == 2, np.nan, 1.0)}, "nan",
                      units=FEWEST_UNITS)
