"""The gate's float64 truths held to recorded ones (tests/recorded_truths.py, `reference_error.certifier`):
a truths recording's function, and a host loop's truth by the programs it states."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import UnrecordedTruth, as_given, assert_as_exact, indexed_recording
from test_reference_error import _squared_case

pytestmark = pytest.mark.usefixtures("unqualified")
"""The gate's own tests: their sites are its subjects, declared at the counts they test it at."""


def test_a_truths_recording_holds_a_float64_path_to_the_references_function_below_float32_noise():
    """A truths recording (`record_truths`) keeps the reference's float64 truth alone at a few draws: the
    reference's function computed in float64 is held to it at every draw (`assert_truths_as_recorded`), and
    another function is refused, one a part in 2**40 off, far below any float32 run's rounding, among
    them."""
    from recorded_truths import assert_truths_as_recorded
    from tool_support import load

    recorder = load("recorded_references")
    draw, evaluate = _squared_case(700)
    recorded = recorder.record_truths("site", draw, evaluate, 4)

    def squared(inputs):
        return {"y": jnp.asarray(inputs["x"], jnp.float64) ** 2 - 1}

    def off(inputs):
        return {"y": (jnp.asarray(inputs["x"], jnp.float64) ** 2 - 1) * (1 + 2.0 ** -40)}

    with jax.enable_x64(new_val=True):
        assert_truths_as_recorded(recorded, "site", squared)
        with pytest.raises(UnrecordedTruth):
            assert_truths_as_recorded(recorded, "site", off)


def test_a_host_loops_float64_truth_is_held_by_the_programs_it_states():
    """A float64 truth run as given, a host loop of compiled rows (each row's values over its own valid
    length, read from the draw on the host, as a decoder's rows run), states the programs a draw runs
    (`as_given`'s `programs`), which the certifier traces to count the draw's roundings: it is held to a
    summarized recording at every seed, and another function, a part in 2**30 off, is refused."""
    from tool_support import load

    recorder = load("recorded_references")

    def draw(seed):
        rng = np.random.default_rng([5970, seed])
        # Each row's length is the draw's, read on the host, alike in every draw as a recording's leaves are.
        return {"x": rng.standard_normal((3, 48)).astype(np.float32), "lengths": np.asarray([40, 32, 24])}

    def evaluate(inputs):
        rows = [inputs["x"][row, :length] for row, length in enumerate(inputs["lengths"])]
        return ({f"row{row}": values * values - np.float32(1) for row, values in enumerate(rows)},
                {f"row{row}": values.astype(np.float64) ** 2 - 1 for row, values in enumerate(rows)})

    summed = jax.jit(lambda values: values * values - 1)

    def rows_of(inputs, scale=1.0):
        return [(summed, (jnp.asarray(inputs["x"][row, :length], jnp.float64) * scale,))
                for row, length in enumerate(inputs["lengths"])]

    def truth_of(scale):
        def run(inputs):
            with jax.enable_x64(new_val=True):
                return {f"row{row}": np.asarray(program(*held))
                        for row, (program, held) in enumerate(rows_of(inputs, scale))}
        return as_given(run, programs=lambda inputs: rows_of(inputs, scale))

    recorded = recorder.record_case("rows", draw, evaluate, store_inputs=False, summary=True)
    seeds = int(recorded["rows.seeds"])
    drawn, reference, truth = indexed_recording(recorded, "rows", seeds, inputs_of=draw,
                                                truth_fn=truth_of(1.0))
    def rows(index, inputs):
        return {f"row{row}": np.asarray(summed(jnp.asarray(inputs["x"][row, :length])))
                for row, length in enumerate(inputs["lengths"])}

    dew = as_given(rows)
    assert_as_exact(dew, reference, drawn, "rows", seeds=seeds, truth_fn=truth)
    _, _, off = indexed_recording(recorded, "rows", seeds, inputs_of=draw, truth_fn=truth_of(1 + 2.0 ** -30))
    with pytest.raises(UnrecordedTruth):
        off(np.int32(0), draw(0))
