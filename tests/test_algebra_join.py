"""The derivation's forms joined to the engine (`dew.algebra.forms.rewritten`, docs/design/algebra.md §6.9):
attention's walk and a recurrence's chunks through forms whose math the engine rewrote before evaluating,
each held to the float64 truth (check 1) as the derived forms are (tests/test_algebra_derive.py,
tests/test_algebra_scan.py), in the fast class and the accurate one."""

import jax
import jax.numpy as jnp
import pytest
import test_algebra_derive as walks
import test_algebra_scan as scans
from reference_error import assert_as_exact

from dew import algebra
from dew.algebra import derive, forms, scan

ACCURACY = {"fast": False, "accurate": True}


@pytest.mark.parametrize("accurate", ACCURACY.values(), ids=ACCURACY)
@pytest.mark.parametrize("name", ["attention", "causal", "products"])
def test_the_rewritten_walk_computes_the_declared_function(name, accurate):
    """Attention's derived walk with each form's math rewritten by the engine: on par with the
    whole-operand function, against the float64 truth (check 1)."""
    function, exponent = walks.DECLARED[name]
    derived = forms.rewritten(derive.derive(walks.declared(function, exponent)), accurate=accurate,
                              saturation=algebra.SATURATION)
    assert_as_exact(jax.jit(walks.walked_whole(derived)), jax.jit(walks.whole(function)), walks.draw,
                    f"{name} rewritten", seeds=8)


@pytest.mark.parametrize("accurate", ACCURACY.values(), ids=ACCURACY)
def test_the_rewritten_ssd_chunks_compute_the_token_scan(accurate):
    """Mamba-2's derived chunks, their forms' math rewritten by the engine: outputs and final state on par
    with the token-by-token scan (check 1)."""
    derived = forms.rewritten(scan.derive_scan(scans.DECLARED), accurate=accurate,
                              saturation=algebra.SATURATION)

    def chunked(x, dt, B, C):
        values = scans.operands(x, dt, B, C)
        outputs, state = scans.walk(derived, values, range(scans.TOKENS // scans.CHUNK), derived.init({}))
        return jnp.concatenate(outputs), derived.finish(state, values)

    assert_as_exact(jax.jit(chunked), jax.jit(scans.tokenwise), scans.draw, "ssd rewritten", seeds=8)


def test_the_rewritten_delta_rules_adjoint_is_the_token_scans_gradient():
    """The gated delta rule's chunk adjoint through rewritten forms, walked back: each token operand's
    gradient on par with the token-by-token scan's (check 1)."""
    derived = forms.rewritten(
        scan.derive_scan(scan.Scan(scans.gated_delta, scans.gated_readout, scans.GATED, walked="time",
                                   state=((scans.KEY, scans.VALUE), jnp.float32))),
        accurate=False, saturation=algebra.SATURATION)
    scans.assert_delta_adjoint(derived, scans.gated_delta, scans.gated_draw, "gated delta rewritten", seeds=8)
