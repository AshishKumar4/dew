"""Check 1's paired sizing (tests/reference_error.py `seeds_for` with `dew_fn`): arms that share their
upstream rounding, each one's error heavy-tailed across draws, resolve paired where the reference
compared with itself does not, and a fault still fails at the count they declare."""

import jax
import numpy as np
import pytest
from reference_error import GateFailure, Inconclusive, as_given, as_recorded, assert_as_exact, seeds_for


@pytest.fixture(scope="module")
def heavy():
    """Sums' terms scaled per draw by a heavy-tailed factor (a Pareto of index one, plus one), the null
    runs' heavy-sums class's draw (tools/gate_null_runs.py) at a smaller size, loaded once: 48 rows a
    draw, which the pair resolves past its first count, confirmed on a second block (at 32 rows a
    draw it resolves on none within 4096 seeds, and the sizing refuses it)."""
    from tool_support import load

    return load("gate_null_runs").heavy_sums(rows=48, terms=128)[1]



def test_the_row_controls_judge_rows_wider_than_two():
    """The worst-row rule's null controls (tools/gate_null_runs.py's `row_controls`) judge rows of eight
    values, their worst rows' shares held as check 1's draws hold them (`parity._Draws.shares`: seed, leaf,
    kind, arm): each run's statistic, the least factor that would have failed it, finite and positive, where
    the controls' shares of no kind axis raised past rows of two (row 10829)."""
    from tool_support import load

    verdicts, statistics = load("gate_null_runs").row_controls(width=8, rows=4, seeds=8, runs=2, chunk=0)
    assert len(verdicts) == 2 and np.isfinite(statistics).all() and min(statistics) > 0

def _summed(order):
    """The rows' float32 sums with their terms in `order`'s permutation, one rounding a term."""
    @as_given
    def summed(x):
        terms = np.asarray(x, np.float32)[:, np.random.default_rng(order).permutation(x.shape[1])]
        return {"sum": np.cumsum(terms, axis=1, dtype=np.float32)[:, -1]}
    return summed


@as_given
def _truth(x):
    return {"sum": np.sum(np.asarray(x, np.float64), axis=1)}


def test_a_shared_heavy_tail_resolves_paired_and_not_against_itself(heavy):
    reference, twin = _summed(0), _summed(1)
    seeds = seeds_for(reference, heavy, "heavy sums", truth_fn=_truth, dew_fn=twin)
    # The reference compared with itself resolves no count within the pair's: the site left undecided.
    with pytest.raises(Inconclusive):
        seeds_for(reference, heavy, "heavy sums", truth_fn=_truth, most=seeds)
    assert_as_exact(twin, reference, heavy, "heavy sums", seeds=seeds, truth_fn=_truth)


def test_a_fault_fails_at_the_paired_count(heavy):
    """A sum in float32 whose every term is first rounded to bfloat16's precision fails at the count the
    equally exact pair declares."""
    import ml_dtypes

    reference, twin = _summed(0), _summed(1)
    seeds = seeds_for(reference, heavy, "heavy sums", truth_fn=_truth, dew_fn=twin)

    @as_given
    def coarse(x):
        return twin(np.asarray(x).astype(ml_dtypes.bfloat16).astype(np.float32))

    with pytest.raises(AssertionError) as raised:
        assert_as_exact(coarse, reference, heavy, "heavy sums", seeds=seeds, truth_fn=_truth)
    assert raised.type is GateFailure


def test_a_recorded_reference_is_refused_a_paired_sizing(heavy):
    """A recorded reference holds its recorded seeds alone (`indexed_recording` marks it `recorded`),
    so a paired sizing, whose draws start at SIZING_FROM, refuses it."""
    reference = _summed(0)

    @as_recorded
    def recorded(x):
        return reference(x)

    with pytest.raises(ValueError):
        seeds_for(recorded, heavy, "recorded sums", truth_fn=_truth, dew_fn=_summed(1))


def test_a_recorded_dew_arm_is_refused_a_paired_sizing(heavy):
    """A judged recording, Dew's arm read from a recording without pilots, holds its recorded seeds alone as
    a recorded reference does, so a paired sizing of it is refused too: the pair's pilot budget is both
    arms' (train 1430's review: it was read off the reference alone)."""
    dew = _summed(1)

    @as_recorded
    def recorded(x):
        return dew(x)

    with pytest.raises(ValueError):
        seeds_for(_summed(0), heavy, "recorded Dew sums", truth_fn=_truth, dew_fn=recorded)


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="the arm's mark is a CPU's")
def test_a_sizing_holds_a_library_reference_to_its_call_on_the_executable_it_runs():
    """A qualification run's sizing (`seeds_for`) of a site no prepare reached holds a library reference
    (`truths_ahead.library_arm`, marked for a CPU) to its call on the executable it runs, as the verdict
    does, before any count is drawn: XLA's own dot marked as a library's call is refused (`NotTheLibrary`;
    train 1472's review: the sizing ran it unchecked, and could decide its count before the verdict's
    check)."""
    import jax.numpy as jnp
    from contract_support import normal_drawn
    from truths_ahead import NotTheLibrary, library_arm

    def product(rows, matrix):
        return jnp.dot(rows, matrix)

    truth = as_given(lambda rows, matrix: np.asarray(rows, np.float64) @ np.asarray(matrix, np.float64))
    reference = library_arm(jax.jit(product), "__a_library_call", platform="cpu")
    with pytest.raises(NotTheLibrary):
        seeds_for(reference, normal_drawn(0x14096, (16, 8), (8, 4)), "a library reference sized",
                  truth_fn=truth, dew_fn=jax.jit(product))
