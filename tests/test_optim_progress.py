"""A schedule over a run's progress (row 2009): a run bounded in examples counts no updates ahead, so a
schedule whose horizon is the run reads the share of its bound the run has consumed. Where every update
consumes the same examples, progress and updates move together, and each law on progress is the law over
the steps that run takes, piece for piece; a law read in float64 is that law's to its own rounding
(`assert_computes_the_oracle`)."""

import jax
import numpy as np
import pytest
from reference_error import assert_computes_the_oracle, chain_roundings

from dew.optim import Cosine, Exponential, Linear, OneCycle, Power, PowerTail, ProgressState, RateState, _Rate

STEPS = 40
"""The updates of the lockstep run: each consumes one fortieth of its bound."""


def progressed(schedule, steps: int = STEPS) -> tuple[np.ndarray, int]:
    """`schedule`'s rate at each update of a run that consumes `1 / steps` of its bound an update, read
    as a clocked run's solver reads it (`_Rate`, `ProgressState`), in float64, and its chain of roundings
    (`chain_roundings`)."""
    with jax.enable_x64(new_val=True):
        rate = _Rate(schedule, None, np.float64, clocked=True)
        state = rate.init()
        assert isinstance(state, ProgressState)
        roundings = chain_roundings(jax.make_jaxpr(lambda progress: rate(state, progress=progress))(0.5))
        values = []
        for update in range(steps):
            progress = np.float64(update / steps)
            values.append(float(rate(state, progress=progress)))
            state = rate.update(state, progress=progress)
    return np.asarray(values), roundings


def stepped(schedule, steps: int = STEPS) -> tuple[np.ndarray, int]:
    """`schedule`'s rate at each update of a run of `steps` updates, read as a run counted in steps reads
    it (`_Rate`, `RateState`), in float64, and its chain of roundings (`chain_roundings`)."""
    with jax.enable_x64(new_val=True):
        rate = _Rate(schedule, steps, np.float64)
        state = rate.init()
        roundings = chain_roundings(jax.make_jaxpr(rate)(state))
        values = []
        for _ in range(steps):
            values.append(float(rate(state)))
            state = rate.update(state)
        return np.asarray(values), roundings


@pytest.mark.parametrize("schedule", [
    Cosine(peak_value=0.3, warmup_steps=6, end_value=0.01, init_value=0.02),
    Cosine(peak_value=0.3, warmup_steps="10%", end_value=0.01),
    Cosine(peak_value=0.3, warmup_steps=0),
    Linear(peak_value=0.3, warmup_steps=5, decay_start=12, end_value=0.05),
    Linear(peak_value=0.3, warmup_steps="25%", end_value=0.05),
    Power(peak_value=0.3, warmup_steps=4, a=2.0, tail=PowerTail(start=24, end_value=0.01)),
    Exponential(init_value=0.3, end_value=0.01, offset=0.02),
], ids=["cosine", "cosine-share", "cosine-bare", "linear", "linear-share", "power-tail", "exponential"])
def test_a_schedule_on_progress_is_its_law_over_the_steps_a_lockstep_run_takes(schedule):
    """Each law read at a run's progress, where every update consumes one fortieth of the bound, is the
    same law over a run of forty updates: warmups counted in updates or as shares, a decay from where its
    warmup ended, a constant to `decay_start`, a power law's tail from `tail.start`, a geometric decay."""
    assert schedule.run_relative()
    (read, own), (values, theirs) = progressed(schedule), stepped(schedule)
    assert_computes_the_oracle(read, values, type(schedule).__name__, roundings=own + theirs)


def test_a_one_cycle_on_progress_peaks_at_its_share_and_ends_at_the_runs_end():
    """A one-cycle schedule on progress rises over `warmup_fraction` of the run and falls to its end, its
    peak at that share and its last value at the run's end (torch's are a step before each), each half a
    cosine between them (float64, against the law written out)."""
    schedule = OneCycle(peak_value=0.2, init_value=0.02, end_value=1e-4, warmup_fraction=0.25)
    at = np.arange(STEPS) / STEPS

    def anneal(start, stop, through):
        return start * 0.5 * (1 + np.cos(np.pi * through)) + stop * 0.5 * (1 - np.cos(np.pi * through))

    law = np.where(at <= 0.25, anneal(0.02, 0.2, at / 0.25), anneal(0.2, 1e-4, (at - 0.25) / 0.75))
    read, roundings = progressed(schedule)
    assert_computes_the_oracle(read, law, "one cycle on progress", roundings=roundings)


def test_a_schedule_whose_every_horizon_is_its_own_reads_its_updates_in_a_clocked_run():
    """A schedule whose decay ends at a step it names reads its updates whatever the run's clock: a clocked
    run's solver holds its state as any run's does (`RateState`), and its rates are the step law's."""
    schedule = Cosine(peak_value=0.3, warmup_steps=4, decay_steps=30)
    assert not schedule.run_relative()
    with jax.enable_x64(new_val=True):
        rate = _Rate(schedule, None, np.float64, clocked=True)
        state = rate.init()
        assert isinstance(state, RateState)
        values = []
        for _ in range(STEPS):
            values.append(float(rate(state)))
            state = rate.update(state)
    stepped_values, roundings = stepped(schedule)
    assert_computes_the_oracle(np.asarray(values), stepped_values, "update-horizoned", roundings=roundings)
