"""Check 1's paired sizing (tests/reference_error.py `seeds_for` with `dew_fn`): arms that share their
upstream rounding, each one's error heavy-tailed across draws, resolve paired where the reference
compared with itself does not, and a fault still fails at the count they declare."""

import numpy as np
import pytest
from reference_error import GateFailure, as_given, assert_as_exact, seeds_for


@pytest.fixture(scope="module")
def heavy():
    """Sums' terms scaled per draw by a heavy-tailed factor (a Pareto of index one, plus one), the null
    runs' heavy-sums class's draw (tools/gate_null_runs.py) at a smaller size, loaded once: 48 rows a
    draw, which the pair resolves past its first count, confirmed on a second block (at 32 rows a
    draw it resolves on none within 4096 seeds, and the sizing refuses it)."""
    from tool_support import load

    return load("gate_null_runs").heavy_sums(rows=48, terms=128)[1]


def _summed(order):
    """The rows' float32 sums with their terms in `order`'s permutation, one rounding a term."""
    @as_given
    def summed(x):
        terms = np.asarray(x, np.float32)[:, np.random.default_rng(order).permutation(x.shape[1])]
        return {"sum": np.cumsum(terms, axis=1, dtype=np.float32)[:, -1]}
    return summed


def _truth(x):
    return {"sum": np.sum(np.asarray(x, np.float64), axis=1)}


def test_a_shared_heavy_tail_resolves_paired_and_not_against_itself(heavy):
    reference, twin = _summed(0), _summed(1)
    seeds = seeds_for(reference, heavy, "heavy sums", truth_fn=_truth, dew_fn=twin)
    with pytest.raises(AssertionError) as raised:
        seeds_for(reference, heavy, "heavy sums", truth_fn=_truth, most=seeds)
    assert raised.type is AssertionError
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


def test_a_sizing_evaluates_each_draw_once_however_many_doublings_it_takes(heavy):
    """A sizing that doubles past its first count runs each arm and the truth once on each draw it
    takes, paired (its draws at a count are its predecessor's and the seeds it adds, twice the count with
    the confirming block) and the reference against itself (each block's first half against its second,
    the reference once on each); besides, each block draws its first seed again and runs each arm again on
    it, the gate's check that they are functions of their seed and inputs (`reference_error._of_its_seed`,
    `_of_its_inputs`), once a block, never the truth."""
    import collections

    calls = collections.Counter()

    def drawn(draw):
        def counted_draw(seed):
            calls["draw"] += 1
            return draw(seed)
        return counted_draw

    def counted(name, fn):
        @as_given
        def run(x):
            calls[name] += 1
            return fn(x)
        return run

    reference, twin = counted("reference", _summed(0)), counted("twin", _summed(1))
    truth = counted("truth", _truth)
    seeds = seeds_for(reference, drawn(heavy), "heavy sums", truth_fn=truth, dew_fn=twin)
    assert seeds > 8
    blocks = calls["draw"] - 2 * seeds
    assert blocks > 1 and calls == {"draw": 2 * seeds + blocks, "reference": 2 * seeds + blocks,
                                    "twin": 2 * seeds + blocks, "truth": 2 * seeds}
    calls.clear()

    def light(seed):
        return (np.random.default_rng(seed).standard_normal((32, 128)).astype(np.float32),)

    seeds = seeds_for(reference, drawn(light), "light sums", truth_fn=truth)
    blocks = calls["draw"] - 4 * seeds
    assert blocks > 0 and calls == {"draw": 4 * seeds + blocks, "reference": 4 * seeds + blocks,
                                    "truth": 4 * seeds}


def test_a_recorded_reference_is_refused_a_paired_sizing(heavy):
    """A recorded reference holds its recorded seeds alone (`indexed_recording` marks it `recorded`),
    so a paired sizing, whose draws start at SIZING_FROM, refuses it."""
    reference = _summed(0)

    def recorded(x):
        return reference(x)
    recorded.recorded = True

    with pytest.raises(ValueError):
        seeds_for(recorded, heavy, "recorded sums", truth_fn=_truth, dew_fn=_summed(1))
