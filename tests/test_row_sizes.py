"""The gate's one sizing of a row (`dew.parity._row_sizes`): check 1 and one draw's rows size a row, and a
leaf, alike (testing.md, "Check 1, on par"; ledger 2442)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from float64_terms import float64_terms
from reference_error import assert_as_exact

from dew.parity import GateFailure, row_errors

FAINT = 0
"""The row whose operands are 2^-50 of the others': its terms far under RESOLUTION of its leaf's, and above
nothing of a leaf whose truth cancels to float64's zero."""


def faint_draw(seed):
    """A [32, 64] by [64, 4] dot's operands, row FAINT of the first 2^-50 of the others."""
    rng = np.random.default_rng([2442, seed])
    x = rng.standard_normal((32, 64))
    x[FAINT] *= 2.0**-50
    return jnp.asarray(x, jnp.float32), jnp.asarray(rng.standard_normal((64, 4)), jnp.float32)


def cancelled(cut: int, *, dropped: bool = False):
    """A leaf whose every row is a dot less itself, the second summed in two parts split at `cut`: zero in
    float64, an ulp or so of each row's own terms off it in float32, every row alike; `dropped`, one term
    of the faint row's second dot left out."""
    def arm(x, w):
        whole = x @ w
        parts = x[:, :cut] @ w[:cut] + x[:, cut:] @ w[cut:]
        if dropped:
            parts = parts.at[FAINT].add(-x[FAINT, 0] * w[0])
        return {"rows": whole - parts}
    return arm


def test_a_leaf_is_sized_as_its_rows_are_on_both_of_the_gates_paths():
    """One draw's rows (`row_errors`) size the faint row of a leaf that cancels at zero: its terms are under
    RESOLUTION of the leaf's terms, every other row at its terms. Check 1 sizes it so too: two arms equally
    exact pass, the faint row held to the reference's residue on it, and an arm that drops one of that
    row's terms, a residue the size of the row, fails by the rule for rows of float64 zero, where a row
    sized at its terms would have failed by the sized rows' rule."""
    x, w = faint_draw(0)
    with jax.enable_x64(new_val=True):
        wide = jnp.asarray(x, jnp.float64), jnp.asarray(w, jnp.float64)
        truth, terms = float64_terms(cancelled(32), *wide)
    sizes, _ = row_errors(np.asarray(cancelled(32)(x, w)["rows"]), truth["rows"], terms["rows"])
    assert sizes[FAINT] == 0 and np.all(np.delete(sizes, FAINT) > 0)

    assert_as_exact(cancelled(32), cancelled(16), faint_draw, "equally exact", seeds=16, terms=True)
    with pytest.raises(GateFailure) as failed:
        assert_as_exact(cancelled(32, dropped=True), cancelled(16), faint_draw, "a dropped term", seeds=16,
                        terms=True)
    assert [finding.rule for finding in failed.value.findings] == ["zero rows"]
