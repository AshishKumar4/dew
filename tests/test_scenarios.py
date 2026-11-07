"""The gate's scenarios (tests/scenarios.py, row 4514), each validated by a null control it passes and an
injected fault it refuses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import GateFailure
from scenarios import (
    assert_differentiates_past_integers,
    assert_examples_independent,
    assert_masked_rows_unread,
    assert_restartable,
    assert_transforms_as_exact,
    boundary_sizes,
)


def test_a_blocks_boundary_sizes_are_its_edges_each_once():
    assert boundary_sizes(8) == (0, 1, 7, 8, 9)
    assert boundary_sizes(1) == (0, 1, 2)


def softplus_sum(x):
    """A smooth function of a batch of rows, row by row."""
    return jnp.sum(jax.nn.softplus(x), axis=-1)


@jax.custom_jvp
def skewed(x):
    """`softplus_sum` whose forward derivative is one percent off: the fault."""
    return softplus_sum(x)


@skewed.defjvp
def _skewed_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    return skewed(x), jnp.sum(jax.nn.sigmoid(x) * t, axis=-1) * 1.01


def draw(seed):
    rng = np.random.default_rng([4514, seed])
    x, t = (jnp.asarray(rng.standard_normal((6, 32)), jnp.float32) for _ in range(2))
    return (x,), (t,), jnp.asarray(rng.standard_normal(6), jnp.float32)


def test_each_transform_is_held_by_check_one_and_a_wrong_forward_derivative_is_refused():
    """Null: a function against itself passes at every transform. Fault: a forward derivative one percent
    off fails at the transforms that take it (jvp, jvp of grad), and not at those that do not."""
    assert_transforms_as_exact(softplus_sum, softplus_sum, draw, "softplus", seeds=8)
    for name in ("jvp", "jvp of grad"):
        with pytest.raises(GateFailure):
            assert_transforms_as_exact(skewed, softplus_sum, draw, "skewed", seeds=8, names=(name,))
    assert_transforms_as_exact(skewed, softplus_sum, draw, "skewed", seeds=8, names=("primal", "vmap"))


def test_integer_and_boolean_inputs_differentiate_as_constants_and_a_seam_reading_their_tangent_fails():
    """Null: a gather by integer rows under a boolean mask differentiates with float0 tangents and
    cotangents for both. Fault: a seam whose forward rule multiplies the integer's tangent fails to run."""
    x, rows, keep = jnp.ones((5, 3)), jnp.asarray([0, 2, 4]), jnp.asarray([True, False, True])

    def gathered(x, rows, keep):
        return jnp.where(keep[:, None], x[rows] * 2.0, 0.0)

    assert_differentiates_past_integers(gathered, (x, rows, keep), "a masked gather")

    @jax.custom_jvp
    def scaled(x, rows):
        return x[rows]

    @scaled.defjvp
    def _scaled_jvp(primals, tangents):
        (x, rows), (dx, drows) = primals, tangents
        return scaled(x, rows), dx[rows] * drows

    with pytest.raises(TypeError):
        assert_differentiates_past_integers(scaled, (x, rows), "a seam reading an integer's tangent")


def test_a_masked_row_no_kept_output_reads_passes_and_one_multiplied_away_is_refused():
    """Null: a masked mean that selects with `where` reads no masked row. Fault: one that multiplies the mask
    in carries a NaN or an infinity of a masked row into every kept output (NaN times zero)."""
    x = jnp.asarray(np.random.default_rng(1).standard_normal((4, 8)), jnp.float32)
    mask = jnp.asarray([False, True, False, True])

    def masked(inputs):
        return (mask,)

    def selected(x):
        return jnp.sum(jnp.where(mask[:, None], 0.0, x), axis=0), jnp.ones(8, bool)

    def multiplied(x):
        return jnp.sum(x * (~mask)[:, None], axis=0), jnp.ones(8, bool)

    assert_masked_rows_unread(selected, (x,), masked, "selected")
    with pytest.raises(AssertionError):
        assert_masked_rows_unread(multiplied, (x,), masked, "multiplied")


def test_examples_independent_of_their_batch_pass_and_a_batch_statistic_is_refused():
    """Null: a row-wise function gives each example's output whatever its batch. Fault: normalizing by the
    batch's mean makes an example's output depend on its neighbours, which a part's own mean moves far past
    rounding."""
    def draw(seed):
        return (jnp.asarray(np.random.default_rng([4549, seed]).standard_normal((6, 4)), jnp.float32),)

    assert_examples_independent(lambda x: jnp.tanh(x) * 2, draw, "row-wise", seeds=8, parts=3)
    # Refused at the reversal, whose mean sums its rows in another order, or by check 1 over the parts.
    with pytest.raises(AssertionError):
        assert_examples_independent(lambda x: x - jnp.mean(x, axis=0), draw, "batch mean", seeds=8, parts=3)
    # A batch maximum is the same whatever the rows' order, so the reversal holds it bit for bit: the parts
    # alone refuse it, each part's maximum its own.
    with pytest.raises(AssertionError):
        assert_examples_independent(lambda x: x - jnp.max(x, axis=0), draw, "batch maximum", seeds=8, parts=3)


def test_a_run_restarted_midstream_ends_where_one_run_ends_and_hidden_state_is_refused():
    """Null: a running sum's state is all it carries. Fault: a step counting its calls outside its state
    restarts its count."""
    inputs = [jnp.full((3,), float(i)) for i in range(6)]
    assert_restartable(jax.jit(lambda state, x: state + x), jnp.zeros(3), inputs, "a running sum", at=4)
    calls = []

    def counted(state, x):
        calls.append(None)
        return state + x * len(calls)

    with pytest.raises(AssertionError):
        assert_restartable(counted, jnp.zeros(3), inputs, "a hidden count", at=4)
