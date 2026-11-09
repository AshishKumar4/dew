"""dew.parity's check 1 over a library's own draws: `seeds_for` declares a site's seeds and `judge` judges
them, each draw ours, the reference's and the float64 truth's."""

import numpy as np
import pytest

from dew import parity


def rounded(factor: float = 1.0):
    """`draw(seed)`: 256 float64 values, the reference's and ours each off them by its own normal error of
    1e-6, ours `factor` times larger, both held in float32."""
    def draw(seed: int):
        rng = np.random.default_rng([1305, seed])
        truth = rng.standard_normal(256)
        ours = (truth + factor * 1e-6 * rng.standard_normal(256)).astype(np.float32)
        theirs = (truth + 1e-6 * rng.standard_normal(256)).astype(np.float32)
        return {"x": ours}, {"x": theirs}, {"x": truth}
    return draw


def test_an_equally_exact_arm_passes_at_its_declared_seeds_and_one_three_times_off_fails():
    """An arm as far from float64 as the reference passes check 1 at the count `seeds_for` declares, and
    one three times as far fails it."""
    seeds = parity.seeds_for(rounded(), "equally exact")
    parity.judge(rounded(), seeds, "equally exact")
    with pytest.raises(parity.GateFailure):
        parity.judge(rounded(3.0), seeds, "three times off")


def test_fewer_seeds_than_check_one_judges_are_inconclusive_whatever_their_ratio():
    """A run of fewer than FEWEST_SEEDS seeds is Inconclusive, even of an arm that matches the reference
    exactly."""
    def same(seed: int):
        ours, _, truth = rounded()(seed)
        return ours, ours, truth

    with pytest.raises(parity.Inconclusive):
        parity.judge(same, parity.FEWEST_SEEDS - 1, "too few")


def test_a_site_its_sizing_cannot_resolve_is_inconclusive():
    """A site whose arms' errors swing by orders of magnitude from draw to draw, apart, resolves no count
    within the most seeds its sizing may take: Inconclusive, which a caller trying candidates in turn
    reads as the gate's own finding."""
    def swinging(seed: int):
        rng = np.random.default_rng([1305, 7, seed])
        truth = rng.standard_normal(256)
        ours, theirs = (truth + np.exp(3 * rng.standard_normal()) * 1e-6 * rng.standard_normal(256)
                        for _ in range(2))
        return {"x": ours.astype(np.float32)}, {"x": theirs.astype(np.float32)}, {"x": truth}

    with pytest.raises(parity.Inconclusive):
        parity.seeds_for(swinging, "swinging", most=parity.FEWEST_SEEDS)
    # A recording that holds no seeds resolves no count either.
    with pytest.raises(parity.Inconclusive):
        parity.seeds_resolving(np.empty((0, 1)), ["x"], "nothing recorded")
