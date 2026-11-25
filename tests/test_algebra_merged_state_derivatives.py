"""Higher-order derivative parity for `means.merged_state` (docs/design/algebra.md §6.12), split out of
`tests/test_algebra_scaled.py` and `tests/test_algebra_merged_state.py` so every file stays well under the
train CI wall bound even on an unsplit run (row 12434)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_algebra_merged_state import (
    _apart,
    _assert_site_on_par,
    _moved,
    _on_par,
    _second,
)
from test_algebra_scaled import WAYS, _state

DERIVATIVE_SITES = {
    "along every operand": lambda: (_moved(_state(1), (0, 1, 2)), _moved(_apart(1), (0, 1, 2))),
    "its gradient's derivative": lambda: (_second(_state(1)), _second(_apart(1))),
}
"""The merged state's heaviest derivative sites (`_state`, `_apart`), split into their own file."""


@pytest.mark.parametrize("site", DERIVATIVE_SITES)
@pytest.mark.parametrize("way", WAYS)
def test_a_merged_states_higher_order_sites_are_on_par_with_its_parts(way, site):
    """`means.merged_state` moved along every operand and its gradient's derivative along the totals, on par
    with the separate merged output and total from float64 (`_on_par`), at each draw (`MERGED_DRAWS`)."""
    _assert_site_on_par(DERIVATIVE_SITES, way, site)


def _gradients_derivatives(form):
    """The derivatives of a two-state merge's gradient as to its totals (its states' anchors 0, one channel)
    under a cotangent of its output and its total: along the outputs, along the totals forward over
    reverse and reverse over reverse (the attention merge's second-order site's)."""
    def ran(outputs, totals, along_outputs, along_totals):
        def gradient(o, w):
            def objective(each):
                output, total, _ = form(o[:, None], each, jnp.zeros_like(each))
                return 0.75 * output[0] + 0.5 * total
            return jax.grad(objective)(w)

        mixed = jax.jvp(lambda o: gradient(o, totals), (outputs,), (along_outputs,))[1]
        forward = jax.jvp(lambda w: gradient(outputs, w), (totals,), (along_totals,))[1]
        reverse = jax.grad(lambda w: jnp.vdot(gradient(outputs, w), along_totals))(totals)
        return mixed, forward, reverse
    return ran


@pytest.mark.parametrize("totals", [[1.25, 0.0], [1.25, 0.5]], ids=["a zero mass", "two masses"])
@pytest.mark.parametrize("way", WAYS)
def test_a_merged_states_gradients_derivatives_are_on_par_with_its_parts(way, totals):
    """The derivatives of a merged state's gradient (`means.merged_state`), forward and reverse over its
    reverse, on par with the separate merged output's and total's from float64 (`_on_par`), at a state of no
    total beside one of a drawn mass too (the attention merge's zero-mass draw, row 7124, where a rule that
    moved every operand read the anchors' zero tangent as a term and missed by its whole value)."""
    def drawn(seed):
        rng = np.random.default_rng(seed)
        directions = (np.asarray(rng.standard_normal(2), np.float32) for _ in range(2))
        return (np.asarray(rng.uniform(-1, 1, 2), np.float32), np.asarray(totals, np.float32), *directions)

    _on_par(_gradients_derivatives(_state(0)), _gradients_derivatives(_apart(0)), way, drawn,
            f"a merged state's gradient's derivatives against its parts', {totals}")
