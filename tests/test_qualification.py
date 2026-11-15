"""Check 1's qualification run (tests/check1_wire.py `QUALIFY`): a site sizes itself and is held to the
count it declares."""

import json

import jax.numpy as jnp
import numpy as np
import pytest
from check1_wire import LATER, QUALIFY
from map_judge import LaterUnits, site_id
from reference_error import StaleSeeds, as_given, assert_as_exact, seeds_for, stored_truths

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


def test_a_site_judged_from_its_prepared_truths_sizes_itself_in_its_run(monkeypatch, tmp_path):
    """A site judged under a prepared store (a GPU flow's: its prepare writes its verdict's truths, its run
    reads them) sizes itself in its run under `QUALIFY`, its sizing draws' truths its own: it records the
    count it sizes unprepared, and raises StaleSeeds where it declares fewer."""
    dew, reference, truth, draw = site()
    sized = seeds_for(reference, draw, LABEL, dew_fn=dew, truth_fn=truth)

    def run(declared: int) -> None:
        store = tmp_path / str(declared)
        with stored_truths(store, prepare=True, stamp="here"):
            assert_as_exact(dew, reference, draw, LABEL, seeds=declared, truth_fn=truth)
        with monkeypatch.context() as qualifying, stored_truths(store, prepare=False, stamp="here"):
            qualifying.setenv(QUALIFY, str(tmp_path / "qualified.jsonl"))
            assert_as_exact(dew, reference, draw, LABEL, seeds=declared, truth_fn=truth)

    run(sized)
    with pytest.raises(StaleSeeds):
        run(sized // 2)
    records = [json.loads(line) for line in (tmp_path / "qualified.jsonl").read_text().splitlines()]
    found = [(record["declared"], record["sized"]) for record in records]
    assert found == [(sized, sized), (sized // 2, sized)]


@pytest.mark.parametrize("repeated", [False, True], ids=["drawn apart", "a draw repeated"])
def test_a_prepared_site_its_first_stage_leaves_undecided_asks_for_its_second_stage(tmp_path, monkeypatch,
                                                                                    repeated):
    """A site under prepared truths (a GPU flow's) whose first stage leaves it undecided (this one sizes far
    above the 64 it declares): its prepare stores its first stage's draws alone; its run asks a later job
    for its second stage where that stage begins (`map_judge.ask_later`: its ask, every unit its stages
    take, appended to `check1_wire.LATER`, and `LaterUnits`), its first stage's inputs drawn apart or one
    drawn twice (its store then holding one truth fewer than its stage's draws); and a follow-up given that
    ask (`check1_wire.ASKED`'s sites) prepares every draw and judges it over both stages."""
    dew, reference, truth, drawn = site()

    def draw(seed):
        return drawn(0 if repeated and seed == 1 else seed)

    first, follow_up, asks = tmp_path / "first", tmp_path / "follow-up", tmp_path / "later.jsonl"
    with stored_truths(first, prepare=True, stamp="here"):
        assert_as_exact(dew, reference, draw, LABEL, seeds=64, truth_fn=truth)
    assert len(list(first.rglob("*.truth"))) == (63 if repeated else 64)
    monkeypatch.setenv(LATER, str(asks))
    with stored_truths(first, prepare=False, stamp="here"), pytest.raises(LaterUnits):
        assert_as_exact(dew, reference, draw, LABEL, seeds=64, truth_fn=truth)
    (ask,) = [json.loads(line) for line in asks.read_text().splitlines()]
    assert (ask["site"], ask["from"], ask["to"]) == (site_id(LABEL), 64, staged_units(64))
    with stored_truths(follow_up, prepare=True, stamp="here", later={ask["site"]}):
        assert_as_exact(dew, reference, draw, LABEL, seeds=64, truth_fn=truth)
    with stored_truths(follow_up, prepare=False, stamp="here", later={ask["site"]}):
        assert_as_exact(dew, reference, draw, LABEL, seeds=64, truth_fn=truth)
