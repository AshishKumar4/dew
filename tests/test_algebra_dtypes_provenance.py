"""Per-term e-graph dtype pricing (rows 12974, 13033), structural body-cache equation provenance (row 13020),
guarded reuse hoisting over softmax and standalone roots (M-DC-14), and missing-DebugInfo hygiene."""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import source_info_util
from reference_error import assert_as_exact
from test_algebra_program import _bf16_rms

from dew import algebra
from dew.algebra import program, terms
from dew.nn.kernels import generation
from dew.nn.kernels.templates import report as kernel_report
from dew.nn.programs import equations


def _rows(seed: int, *shapes: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    return tuple(rng.standard_normal(shape).astype(np.float32) for shape in shapes)


def test_a_norm_of_norms_backward_reuses_a_reduction_through_a_cotangent_sum():
    from jax._src.ad_util import add_jaxvals_p

    def norm(values, axis):
        return jnp.sqrt(jnp.sum(values * values, axis=axis, keepdims=True))

    def loss(rows):
        positive = 2 + jnp.tanh(rows)
        return norm(positive / norm(positive, -1), 0).sum()

    gradient = jax.grad(loss)
    rewritten = algebra.rewritten(gradient)
    rows = jax.ShapeDtypeStruct((32, 128), jnp.float32)
    original, selected = (list(equations(jax.make_jaxpr(arm)(rows).jaxpr))
                          for arm in (gradient, rewritten))
    assert sum(eq.primitive is jax.lax.reduce_sum_p for eq in selected) < sum(
        eq.primitive is jax.lax.reduce_sum_p for eq in original)
    assert any(eq.primitive is add_jaxvals_p for eq in selected)
    assert_as_exact(jax.jit(rewritten), jax.jit(gradient), lambda seed: _rows(seed, rows.shape),
                    "norm of norms backward", seeds=8)


@pytest.mark.mesh(devices=2)
def test_cotangent_reductions_keep_their_explicit_placements():
    from jax._src.ad_util import add_jaxvals_p
    from jax.core import ShapedArray
    from jax.sharding import AxisType, PartitionSpec as P

    def reduced(a, b):
        x = jax.lax.broadcast_in_dim(jnp.tanh(a), (128,), (), out_sharding=P("data"))
        y = jax.lax.broadcast_in_dim(jnp.tanh(b), (128,), (), out_sharding=P(None))
        return add_jaxvals_p.bind(jax.lax.reduce_sum(x, (0,)), jax.lax.reduce_sum(y, (0,)))

    mesh = jax.make_mesh((2,), ("data",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    with jax.set_mesh(mesh):
        inputs = jnp.float32(0.5), jnp.float32(1.0)
        expected = jax.jit(reduced)(*inputs)
        compute = algebra.rewritten(reduced)
        actual = jax.jit(compute)(*inputs)
        np.testing.assert_array_equal(actual, expected)
        selected = equations(jax.make_jaxpr(compute)(*inputs).jaxpr)
        add, = (eq for eq in selected if eq.primitive is add_jaxvals_p)
        assert [atom.aval.shape for atom in add.invars if isinstance(atom.aval, ShapedArray)] == [(), ()]


def test_structural_closed_preserves_equation_source_info_and_nested_body_provenance():
    """`program._STRUCTURAL_CLOSED` declines whole-body substitution of a modified body across differing
    equation `source_info` (`name_stack` / traceback) or nested body provenance (row 13020), while still
    reusing exact-provenance retraces and unchanged bodies."""
    shapes = (jax.ShapeDtypeStruct((256, 256), jnp.bfloat16), jax.ShapeDtypeStruct((256,), jnp.bfloat16))

    def make_inner(scope_name: str):
        @jax.jit
        def inner(y):
            with jax.named_scope(scope_name):
                return y + jnp.bfloat16(1.0)
        return inner

    inner_left = make_inner("inner_left_scope")
    inner_right = make_inner("inner_right_scope")

    def fn_left(x, weight):
        with jax.named_scope("left_branch"):
            return inner_left(_bf16_rms(x, weight)) + jnp.bfloat16(2.0)

    def fn_right(x, weight):
        with jax.named_scope("right_branch"):
            return inner_right(_bf16_rms(x, weight)) + jnp.bfloat16(2.0)

    cl = jax.make_jaxpr(fn_left)(*shapes)
    cr = jax.make_jaxpr(fn_right)(*shapes)
    assert terms.body_key(cl) == terms.body_key(cr)
    assert terms.provenance_key(cl) != terms.provenance_key(cr)
    with generation.assumed("sm90"):
        out_l = program._closed(cl, accurate=False, saturation=algebra.SATURATION, kernels=True)
        out_r = program._closed(cr, accurate=False, saturation=algebra.SATURATION, kernels=True)
    assert out_l is not cl and out_r is not cr and out_l is not out_r
    assert out_l.debug_info == cl.debug_info and out_r.debug_info == cr.debug_info
    add_l = [eq for eq in out_l.jaxpr.eqns if eq.primitive is jax.lax.add_p]
    add_r = [eq for eq in out_r.jaxpr.eqns if eq.primitive is jax.lax.add_p]
    assert add_l and add_r
    assert "left_branch" in str(add_l[-1].source_info.name_stack)
    assert "right_branch" in str(add_r[-1].source_info.name_stack)
    assert source_info_util.user_frame(add_l[-1].source_info.traceback) != source_info_util.user_frame(
        add_r[-1].source_info.traceback)
    pjit_l = next(
        eq for eq in out_l.jaxpr.eqns if eq.primitive.name == "jit" and eq.params.get("name") == "inner")
    pjit_r = next(
        eq for eq in out_r.jaxpr.eqns if eq.primitive.name == "jit" and eq.params.get("name") == "inner")
    sub_l, sub_r = pjit_l.params["jaxpr"], pjit_r.params["jaxpr"]
    assert "inner_left_scope" in str(sub_l.jaxpr.eqns[0].source_info.name_stack)
    assert "inner_right_scope" in str(sub_r.jaxpr.eqns[0].source_info.name_stack)

    def trace_once():
        def fresh_fn(x, weight):
            with jax.named_scope("left_branch"):
                return inner_left(_bf16_rms(x, weight)) + jnp.bfloat16(2.0)
        return jax.make_jaxpr(fresh_fn)(*shapes)

    c1, c2 = trace_once(), trace_once()
    assert c1 is not c2 and terms.provenance_key(c1) == terms.provenance_key(c2)
    with generation.assumed("sm90"), kernel_report.recording():
        r1 = program._closed(c1, accurate=False, saturation=algebra.SATURATION, kernels=True)
        r2 = program._closed(c2, accurate=False, saturation=algebra.SATURATION, kernels=True)
    assert r2 is r1

    def noop_a(x):
        with jax.named_scope("noop_a"):
            return x + jnp.float32(1.0)

    def noop_b(x):
        with jax.named_scope("noop_b"):
            return x + jnp.float32(1.0)

    s1 = jax.ShapeDtypeStruct((16,), jnp.float32)
    na, nb = jax.make_jaxpr(noop_a)(s1), jax.make_jaxpr(noop_b)(s1)
    oa = program._closed(na, accurate=False, saturation=algebra.SATURATION, kernels=False)
    ob = program._closed(nb, accurate=False, saturation=algebra.SATURATION, kernels=False)
    assert oa is na and ob is nb
    assert "noop_b" in str(ob.jaxpr.eqns[0].source_info.name_stack)


def test_reuse_hoists_softmax_and_standalone_root_and_emits_no_debug_info_warning():
    """`_SoftmaxTrack` proves `sum(exp(x - max(x))) >= 1` finite/bounded (rejecting misaligned axes and lossy
    casts), `reuse` hoists `jax.nn.softmax` broadcast division to `integer_pow(-1)` and standalone
    `x / sqrt(v)` to `x * rsqrt(v)` without pre-existing siblings while holding Check 1, and rewriting emits
    zero JAX `missing a DebugInfo` DeprecationWarnings."""
    def softmax_sum(x):
        peak = jax.lax.stop_gradient(jnp.max(x, axis=-1, keepdims=True))
        return jnp.sum(jnp.exp(x - peak), axis=-1)

    def bad_axis_sum(x):
        peak = jnp.max(x, axis=0)
        return jnp.sum(jnp.exp(x - peak[:, None]), axis=-1)

    def lossy_max_sum(x):
        peak = jnp.max(x, axis=-1, keepdims=True).astype(jnp.bfloat16).astype(jnp.float32)
        return jnp.sum(jnp.exp(x - peak), axis=-1)

    sq = jax.ShapeDtypeStruct((16, 16), jnp.float32)
    c_good = jax.make_jaxpr(softmax_sum)(sq)
    held_g, apart_g = terms.proven(c_good.jaxpr)
    out_g = c_good.jaxpr.outvars[0]
    assert held_g[out_g] is not None and apart_g[out_g].positive and apart_g[out_g].floor == 0
    for bad_fn in (bad_axis_sum, lossy_max_sum):
        c_bad = jax.make_jaxpr(bad_fn)(sq)
        _, apart_b = terms.proven(c_bad.jaxpr)
        assert apart_b[c_bad.jaxpr.outvars[0]].floor is None

    def plain_softmax(x):
        peak = jnp.max(x, axis=-1, keepdims=True)
        exps = jnp.exp(x - peak)
        return exps / jnp.sum(exps, axis=-1, keepdims=True)

    def standalone_root(x, z):
        denom = jnp.mean(jnp.square(jnp.tanh(z)), axis=-1, keepdims=True) + jnp.float32(1e-5)
        return jnp.tanh(x) / jnp.sqrt(denom)

    def roots_in_branches(x, z):
        # The other arm accepts an unbounded divisor, so reuse must leave its square root and division.
        return jax.lax.cond(x[0, 0] > 0, standalone_root,
                            lambda x, z: jnp.tanh(x) / jnp.sqrt(z), x, z)

    def draw_roots(seed):
        x, z = _rows(seed, (16, 64), (16, 64))
        return x, np.abs(z) + np.float32(0.125)

    x0, z0 = draw_roots(0)
    with algebra.rules.enabling_dev("reuse"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            sm_before = list(equations(jax.make_jaxpr(plain_softmax)(x0).jaxpr))
            sm_after = list(equations(jax.make_jaxpr(algebra.rewritten(plain_softmax))(x0).jaxpr))
            rt_before = list(equations(jax.make_jaxpr(roots_in_branches)(x0, z0).jaxpr))
            rt_after = list(equations(jax.make_jaxpr(algebra.rewritten(roots_in_branches))(x0, z0).jaxpr))
            with generation.assumed("sm90"):
                rms_shapes = (jax.ShapeDtypeStruct((256, 256), jnp.bfloat16),
                              jax.ShapeDtypeStruct((256,), jnp.bfloat16))
                program._closed(jax.make_jaxpr(_bf16_rms)(*rms_shapes), accurate=False,
                                saturation=algebra.SATURATION, kernels=True)
        debug_warns = [w for w in caught if "DebugInfo" in str(w.message)]
        assert not debug_warns, [str(w.message) for w in debug_warns]
        assert any(eq.primitive is jax.lax.div_p for eq in sm_before)
        assert not any(eq.primitive is jax.lax.div_p for eq in sm_after)
        assert any(eq.primitive is jax.lax.integer_pow_p and eq.params.get("y") == -1 for eq in sm_after)
        assert sum(eq.primitive is jax.lax.sqrt_p for eq in rt_before) == 2
        assert sum(eq.primitive is jax.lax.sqrt_p for eq in rt_after) == 1
        assert (sum(eq.primitive is jax.lax.div_p for eq in rt_before)
                - sum(eq.primitive is jax.lax.div_p for eq in rt_after)) == 1
        assert any(eq.primitive is jax.lax.rsqrt_p for eq in rt_after)
        assert_as_exact(jax.jit(algebra.rewritten(plain_softmax)), jax.jit(plain_softmax),
                        lambda seed: _rows(seed, (16, 64)), "reuse softmax hoist", seeds=8)
        assert_as_exact(jax.jit(algebra.rewritten(roots_in_branches)), jax.jit(roots_in_branches),
                        draw_roots, "reuse roots in guarded and unguarded branches", seeds=8)
