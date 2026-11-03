"""The derivation joined to the engine (docs/design/algebra.md §6.9): attention's walk derived in one e-graph
with the engine's rule sets (`dew.algebra.derive`), and a recurrence's chunks through forms whose math the
engine rewrote before evaluating (`dew.algebra.forms.rewritten`), each held to the float64 truth (check 1) as
the derived forms are (tests/test_algebra_derive.py, tests/test_algebra_scan.py), in the fast class and the
accurate one."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest
import test_algebra_derive as walks
import test_algebra_scan as scans
from reference_error import assert_as_exact

from dew import algebra
from dew.algebra import derive, forms, scan
from dew.nn.kernels.templates import report

ACCURACY = {"fast": False, "accurate": True}


@pytest.mark.parametrize("accurate", ACCURACY.values(), ids=ACCURACY)
@pytest.mark.parametrize("name", ["attention", "causal", "products"])
def test_the_rewritten_walk_computes_the_declared_function(name, accurate):
    """Attention's walk derived beside the engine's sets its numerics admit: on par with the whole-operand
    function, against the float64 truth (check 1)."""
    function, exponent = walks.DECLARED[name]
    derived = derive.derive(walks.declared(function, exponent), accurate=accurate,
                            saturation=algebra.SATURATION)
    assert_as_exact(jax.jit(walks.walked_whole(derived)), jax.jit(walks.whole(function)), walks.draw,
                    f"{name} rewritten", seeds=8)


@pytest.mark.parametrize("accurate", ACCURACY.values(), ids=ACCURACY)
def test_the_rewritten_ssd_chunks_compute_the_token_scan(accurate):
    """Mamba-2's derived chunks, their forms' math rewritten by the engine: outputs and final state on par
    with the token-by-token scan (check 1)."""
    derived = forms.rewritten(scan.derive_scan(scans.DECLARED, saturation=algebra.SATURATION),
                              accurate=accurate, saturation=algebra.SATURATION)

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
                                   state=((scans.KEY, scans.VALUE), jnp.float32)),
                         saturation=algebra.SATURATION),
        accurate=False, saturation=algebra.SATURATION)
    scans.assert_delta_adjoint(derived, scans.gated_delta, scans.gated_draw, "gated delta rewritten", seeds=8)


def scaled_by_a_power_of_two(q, k, v, rows, keys):
    """Attention whose logits' scale is a power of two (64 wide, 1/8): a scale the engine may move onto
    either operand of its contraction exactly."""
    del rows, keys
    s = (q @ k.T) * 0.125
    m = jnp.max(s, axis=-1)
    p = jnp.exp(s - m[:, None])
    return (p @ v) / jnp.sum(p, axis=-1)[:, None]


def test_the_engine_rewrites_a_derived_walk_and_it_computes_the_declared_function():
    """A derived walk whose logits' scale is a power of two, its queries' and keys' ranges declared (the
    draws' bound): the engine's rule that moves a power-of-two scale onto an operand of the contraction fires
    in the derivation's e-graph (`linear`, its report in `report.rewrites`), its new operand bounded by the
    range facts (§6.3), and the walk still computes the whole-operand function, against the float64 truth
    (check 1)."""
    declared = dataclasses.replace(walks.declared(scaled_by_a_power_of_two), ranges={"q": 8.0, "k": 8.0})
    derived = derive.derive(declared, accurate=False, saturation=algebra.SATURATION)
    q, k, v = walks.draw(0)
    with report.refusals():
        jax.make_jaxpr(walks.walked_whole(derived))(q, k, v)
        fired = [made.fired for made in report.rewrites()]
    assert any(counts.get("linear", 0) > 0 for counts in fired), fired
    assert_as_exact(jax.jit(walks.walked_whole(derived)), jax.jit(walks.whole(scaled_by_a_power_of_two)),
                    walks.draw, "power-of-two scale rewritten", seeds=8)
