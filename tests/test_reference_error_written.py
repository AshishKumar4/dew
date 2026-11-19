"""Check 1's accuracy claim held as written (`reference_error._written_twin`, docs/design/testing.md's check
1): Dew's arm against its reference compiled alike and against the same reference compiled as written,
without XLA's rewrites and fusions, so a rewrite both arms carry fails (row 6909); and every float64 truth
compiled as written."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import GateFailure, assert_as_exact
from truths_ahead import compiled_truths


def _cancelling(seed):
    """Three outputs, +-2^127 (1 + u) and 3 2^-126 (1 + u'), of equal masses."""
    rng = np.random.default_rng(seed)
    large = np.float32(2.0 ** 127 * (1 + rng.uniform()))
    tiny = np.float32(3.0 * 2.0 ** -126 * (1 + rng.uniform()))
    return np.asarray([large, -large, tiny], np.float32), np.ones(3, np.float32)


def _combined(outputs, totals):
    """The outputs' convex combination by their masses."""
    return jnp.sum(totals / jnp.sum(totals) * outputs)[None, None]


def test_a_rewrite_both_arms_would_carry_fails_dews_arm_against_its_reference_as_written():
    """The convex combination of three equal masses whose outputs are +-2^127 (1 + u) and 3 2^-126 (1 + u'),
    one function as both arms: compiled with every XLA pass, as Dew's arm is, its fused sum takes each
    product unrounded beside its rounded opposite and leaves some 3.6e30 where the truth is 1.2e-38. On
    par with the reference compiled alike, both wrong (row 6909), Dew's arm fails against the reference
    compiled as written (`_written_twin`), which sums them as written, near the truth, by far more than
    PARITY to the power WRITTEN."""
    with pytest.raises(AssertionError) as raised:
        assert_as_exact(_combined, _combined, _cancelling, "a combination XLA's fusions rewrite", seeds=8)
    assert raised.type is GateFailure


def test_a_truth_its_site_jits_runs_as_written():
    """A float64 truth its site passes jitted runs as the gate runs every truth, as written
    (`_truth_program`): the combination of equal masses of +-2^1023 (1 + u) and 3 2^-1022 (1 + u'), whose
    fused sum XLA's passes leave some 2^1023 2^-53 off, is NumPy's evaluation of it as written."""
    from reference_error import _truth_program

    def combined(outputs, totals):
        return jnp.sum(totals / jnp.sum(totals) * outputs)

    with jax.enable_x64(new_val=True):
        rng = np.random.default_rng(0)
        large, tiny = 2.0 ** 1023 * (1 + rng.uniform()), 3.0 * 2.0 ** -1022 * (1 + rng.uniform())
        outputs, totals = np.asarray([large, -large, tiny]), np.ones(3)
        value = float(_truth_program(jax.jit(combined))(outputs, totals))
    assert value == np.sum(totals / np.sum(totals) * outputs)


def test_a_reference_compiled_without_ynnpacks_fusions_is_held_as_written_too():
    """The combination as its reference compiled by its site without YNNPACK's fusions (`without_ynn`), a
    policy and not a claim: written apart from Dew, its own function is compiled as written too
    (`_written_twin`), and Dew's arm fails against it as against the plain function."""
    from reference_error import without_ynn

    with pytest.raises(GateFailure):
        assert_as_exact(_combined, without_ynn(_combined), _cancelling, "a combination XLA's fusions rewrite",
                        seeds=8)




def test_a_truth_its_terms_size_runs_as_written():
    """A truth evaluated with its terms (`float64_terms`, a site's `terms`) runs as written, as every truth
    the gate runs: the float64 combination of equal masses of +-2^1023 (1 + u) and 3 2^-1022 (1 + u') is
    NumPy's evaluation of it as written."""
    from float64_terms import float64_terms

    def combined(outputs, totals):
        return jnp.sum(totals / jnp.sum(totals) * outputs)

    with jax.enable_x64(new_val=True):
        rng = np.random.default_rng(0)
        large, tiny = 2.0 ** 1023 * (1 + rng.uniform()), 3.0 * 2.0 ** -1022 * (1 + rng.uniform())
        outputs, totals = np.asarray([large, -large, tiny]), np.ones(3)
        value = float(float64_terms(combined, outputs, totals)[0])
    assert value == np.sum(totals / np.sum(totals) * outputs)


def _product(x, w):
    """A float32 product of drawn matrices."""
    return x @ w


def _drawn_matrices(seed):
    rng = np.random.default_rng([0x6909, seed])
    return (rng.standard_normal((16, 32)).astype(np.float32), rng.standard_normal((32, 8)).astype(np.float32))


def test_a_truth_jitted_with_compiler_options_of_its_own_runs_as_written(tmp_path):
    """A float64 truth its site passes as a jit of compiler options of its own runs as every truth does, its
    own function compiled as written (`arm_forms._own_function`): nested inside the gate's jit of options,
    JAX refused it, a configured truth's site raising before it drew, live and in its prepare alike (the
    as-written head's review)."""
    from arm_forms import WITHOUT_YNN

    configured = jax.jit(_product, compiler_options=WITHOUT_YNN)
    label = "a product beside its configured truth"
    assert_as_exact(_product, _product, _drawn_matrices, label, seeds=8, truth_fn=configured)
    # Prepared and run from its store as a GPU tier part's (`truths_ahead.compiled_truths`) too.
    with compiled_truths(tmp_path, prepare=True, stamp="here"):
        assert_as_exact(_product, _product, _drawn_matrices, label, seeds=8, truth_fn=configured)
    with compiled_truths(tmp_path, prepare=False, stamp="here"):
        assert_as_exact(_product, _product, _drawn_matrices, label, seeds=8, truth_fn=configured)
