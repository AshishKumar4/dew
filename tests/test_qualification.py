"""Check 1's qualification run (tests/check1_wire.py `QUALIFY`): a site sizes itself and is held to the
count it declares."""

import json

import jax.numpy as jnp
import numpy as np
import pytest
from check1_wire import LATER, QUALIFY
from reference_error import StaleSeeds, as_given, as_recorded, assert_as_exact, seeds_for
from truths_ahead import compiled_truths

from dew.parity import SIZING_FROM
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


def test_a_site_judged_with_the_gates_own_terms_is_sized_as_a_plain_pair(monkeypatch, tmp_path):
    """A site asking for the gate's own terms of its truth (`terms=True`, its truth program's walk), whose
    sizing reads no terms, is sized as the plain pair's: declaring fewer seeds than its sizing takes, it
    raises StaleSeeds under `QUALIFY`, where it was recorded unsized and could never go stale (spec
    alignment, 9044)."""
    dew, reference, _, draw = site()

    def truth(x, w):
        return jnp.dot(x, w)

    sized = seeds_for(reference, draw, LABEL, dew_fn=dew, truth_fn=truth)
    monkeypatch.setenv(QUALIFY, str(tmp_path / "qualified.jsonl"))
    with pytest.raises(StaleSeeds):
        assert_as_exact(dew, reference, draw, LABEL, seeds=sized // 2, truth_fn=truth, terms=True)


def test_a_site_judged_from_its_prepared_truths_sizes_itself_in_its_run(monkeypatch, tmp_path):
    """A site judged under programs compiled ahead (a GPU flow's: its prepare compiles its truths' programs,
    its run computes its truths with them) sizes itself in its run under `QUALIFY`, its sizing draws' truths
    its own: it records the count it sizes unprepared, and raises StaleSeeds where it declares fewer."""
    dew, reference, truth, draw = site()
    sized = seeds_for(reference, draw, LABEL, dew_fn=dew, truth_fn=truth)

    def run(declared: int) -> None:
        store = tmp_path / str(declared)
        with compiled_truths(store, prepare=True, stamp="here"):
            assert_as_exact(dew, reference, draw, LABEL, seeds=declared, truth_fn=truth)
        with monkeypatch.context() as qualifying, compiled_truths(store, prepare=False, stamp="here"):
            qualifying.setenv(QUALIFY, str(tmp_path / "qualified.jsonl"))
            assert_as_exact(dew, reference, draw, LABEL, seeds=declared, truth_fn=truth)

    run(sized)
    with pytest.raises(StaleSeeds):
        run(sized // 2)
    records = [json.loads(line) for line in (tmp_path / "qualified.jsonl").read_text().splitlines()]
    found = [(record["declared"], record["sized"]) for record in records]
    assert found == [(sized, sized), (sized // 2, sized)]


def test_an_accuracy_claims_site_sizes_its_bound_as_written_too(monkeypatch, tmp_path):
    """A site whose reference is written apart from Dew (an accuracy claim, its reference compiled as written
    too, `reference_error._written_twin`) sizes under `QUALIFY` both pairs it is judged by, Dew against its
    reference compiled alike and against it as written (`seeds_for`'s `written`), and declares the larger,
    run live and from programs its prepare compiled under `QUALIFY`, the twin's at its sizing draws' own
    shapes too (pilots of another length than its verdict's draws): the as-written head's review, its sizing
    took the pair compiled alike alone and its prepare compiled no twin at a sizing draw's shape."""
    _, _, truth, verdict_draw = site()

    def draw(seed):
        # A pilot (from SIZING_FROM on) of 48 values, a verdict's of 64.
        x, w = verdict_draw(seed)
        return (x[:48], w[:48]) if seed >= SIZING_FROM else (x, w)

    def dot(x, w):
        return jnp.dot(x, w)

    def reference(x, w):
        # Its own, written apart from Dew: the dot of `x` at bfloat16's precision, far from the truth beside
        # Dew's, so each pair resolves at few seeds whichever way it compiles.
        return jnp.dot(x.astype(jnp.bfloat16).astype(jnp.float32), w)

    alike = seeds_for(reference, draw, LABEL, dew_fn=dot, truth_fn=truth)
    written = seeds_for(reference, draw, LABEL, dew_fn=dot, truth_fn=truth, written=True)
    declared = max(alike, written)
    store = tmp_path / "store"
    monkeypatch.setenv(QUALIFY, str(tmp_path / "qualified.jsonl"))
    with compiled_truths(store, prepare=True, stamp="here"):
        assert_as_exact(dot, reference, draw, LABEL, seeds=declared, truth_fn=truth)
    assert_as_exact(dot, reference, draw, LABEL, seeds=declared, truth_fn=truth)
    with compiled_truths(store, prepare=False, stamp="here"):
        assert_as_exact(dot, reference, draw, LABEL, seeds=declared, truth_fn=truth)
    records = [json.loads(line) for line in (tmp_path / "qualified.jsonl").read_text().splitlines()]
    assert [(record["sized"], record["sized as written"]) for record in records] == [(declared, written)] * 2


def test_a_compiled_sites_run_draws_its_second_stage_from_its_programs(tmp_path, monkeypatch):
    """A site under programs compiled ahead (a GPU flow's, `truths_ahead.compiled_truths`) whose first stage
    leaves it undecided (this one sizes far above the 64 it declares) draws its second stage in its run, its
    truths computed by the program its prepare compiled for its draws' shape: no later job is asked
    (`check1_wire.LATER` set, nothing appended to it), and it passes as a site computing its own truths
    passes."""
    dew, reference, _, drawn = site()
    seeds = []

    def draw(seed):
        seeds.append(seed)
        return drawn(seed)

    def truth(x, w):
        return jnp.dot(x, w)

    asks, programs = tmp_path / "later.jsonl", tmp_path / "programs"
    monkeypatch.setenv(LATER, str(asks))
    with compiled_truths(programs, prepare=True, stamp="here"):
        assert_as_exact(dew, reference, draw, LABEL, seeds=64, truth_fn=truth)
    seeds.clear()
    with compiled_truths(programs, prepare=False, stamp="here"):
        assert_as_exact(dew, reference, draw, LABEL, seeds=64, truth_fn=truth)
    assert not asks.exists() and max(seeds) >= 64


def test_a_judged_recording_beside_a_compiled_reference_prepares_no_sizing_draw(monkeypatch, tmp_path):
    """A site whose Dew arm is a recording without pilots (a judged recording, `as_recorded`) beside a
    reference its prepare compiles prepares under `QUALIFY` at its verdict's draws alone: the pair's pilot
    budget, none, leaves it unsized there as its qualification leaves it (train 1430's review: the prepare
    read the reference's budget alone and drew a sizing seed the recording does not hold)."""
    _, _, truth, live = site()

    def recorded_draw(seed):
        # A recording's draws: its verdict's seeds alone, none from SIZING_FROM on (tests/recordings.py).
        if seed >= SIZING_FROM:
            raise AssertionError(f"seed {seed} is past the recording's draws")
        return live(seed)

    dew = as_recorded(as_given(lambda x, w: np.asarray(jnp.dot(x, w))))
    monkeypatch.setenv(QUALIFY, str(tmp_path / "qualified.jsonl"))
    with compiled_truths(tmp_path / "store", prepare=True, stamp="here"):
        assert_as_exact(dew, lambda x, w: jnp.dot(x[::-1], w[::-1]), recorded_draw, LABEL, seeds=16,
                        truth_fn=truth)
