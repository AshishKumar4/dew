"""Check 1's qualification run (tests/check1_wire.py `QUALIFY`): a site sizes itself and is held to the
count it declares."""

import json

import jax.numpy as jnp
import numpy as np
import pytest
from check1_wire import QUALIFY
from reference_error import StaleSeeds, as_given, assert_as_exact, seeds_for

from dew.parity_stages import staged_units

LABEL = "a reversed dot"


def site():
    """A site's arms, truth and draw: the float32 dot of two vectors of 64, the same dot of the vectors
    reversed, which rounds apart from it and as far from float64, and NumPy's float64 dot."""
    def draw(seed):
        rng = np.random.default_rng([0x51, seed])
        return rng.standard_normal(64).astype(np.float32), rng.standard_normal(64).astype(np.float32)

    reversed_dot = as_given(lambda x, w: np.asarray(jnp.dot(x[::-1], w[::-1])))
    truth = as_given(lambda x, w: np.dot(x.astype(np.float64), w.astype(np.float64)))
    return jnp.dot, reversed_dot, truth, draw


pytestmark = pytest.mark.usefixtures("unqualified")


def test_a_qualification_run_passes_a_site_at_the_count_it_sizes_and_records_it(monkeypatch, tmp_path):
    """Under `QUALIFY` a site that declares the count its own sizing takes passes, and the run's file holds
    its label, the count it declares and the count it sized."""
    dew, reference, truth, draw = site()
    sized = seeds_for(reference, draw, LABEL, dew_fn=dew, truth_fn=truth)
    monkeypatch.setenv(QUALIFY, str(tmp_path / "qualified.jsonl"))
    assert_as_exact(dew, reference, draw, LABEL, seeds=sized, truth_fn=truth)
    (record,) = [json.loads(line) for line in (tmp_path / "qualified.jsonl").read_text().splitlines()]
    assert (record["label"], record["declared"], record["sized"]) == (LABEL, sized, sized)


def test_a_qualification_run_fails_a_site_that_declares_fewer_seeds_than_it_sizes(monkeypatch, tmp_path):
    """Under `QUALIFY` a site that declares fewer seeds than its own sizing takes here raises StaleSeeds
    before its verdict: a count sized on other arithmetic. Outside a qualification run no site sizes
    itself: the same call draws its declared seeds alone."""
    dew, reference, truth, draw = site()
    sized = seeds_for(reference, draw, LABEL, dew_fn=dew, truth_fn=truth)
    drawn = []

    def counted(seed):
        drawn.append(seed)
        return draw(seed)

    assert_as_exact(dew, reference, counted, LABEL, seeds=sized, truth_fn=truth)
    assert max(drawn) < staged_units(sized)
    monkeypatch.setenv(QUALIFY, str(tmp_path / "qualified.jsonl"))
    with pytest.raises(StaleSeeds):
        assert_as_exact(dew, reference, draw, LABEL, seeds=sized // 2, truth_fn=truth)
