"""Structural rewrite fingerprints, cache admission, and retained program lifetime."""

import dataclasses
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew import algebra
from dew.algebra import program, terms
from dew.nn.kernels import generation


def test_jaxpr_parameter_key_hashes_structure_and_isolates_opaques_and_consts():
    """`terms.parameter_key` hashes `jcore.Jaxpr` structurally (`terms.jaxpr_digest`), keeping pure jaxprs
    JSON-safe (`_stated._plain`) and cross-process stable (`terms.opaque`) while distinguishing different
    arithmetic, constants, and opaque `custom_vjp` `WrappedFun` closures."""
    from dew import _stated

    shape = jax.ShapeDtypeStruct((4, 8), jnp.float32)
    c1 = jax.make_jaxpr(lambda x: jnp.sin(x) + 1.0)(shape)
    c2 = jax.make_jaxpr(lambda x: jnp.sin(x) + 1.0)(shape)
    c3 = jax.make_jaxpr(lambda x: jnp.cos(x) + 1.0)(shape)
    k1, k2, k3 = terms.parameter_key(c1), terms.parameter_key(c2), terms.parameter_key(c3)
    assert k1 == k2 and k1 != k3
    assert terms.jaxpr_digest(c1) == terms.jaxpr_digest(c2)
    assert terms.opaque(k1) == k1
    assert isinstance(_stated._plain(k1, "pure_jaxpr"), list)

    # Closed-over constants differentiate `parameter_key` even when the jaxpr body matches structurally.
    arr_a, arr_b = np.ones((4, 8), np.float32), np.full((4, 8), 2.0, np.float32)
    ca = jax.make_jaxpr(lambda x: x + arr_a)(shape)
    cb = jax.make_jaxpr(lambda x: x + arr_b)(shape)
    assert terms.jaxpr_digest(ca) == terms.jaxpr_digest(cb)
    assert terms.parameter_key(ca) != terms.parameter_key(cb)

    # Distinct `custom_vjp` rules embed distinct opaque `WrappedFun`s that never collide and refuse `_plain`.
    def make_vjp(scale: float):
        @jax.custom_vjp
        def f(x):
            return x * 2.0

        f.defvjp(lambda x: (x * 2.0, None), lambda _, g: (g * scale,))
        return jax.make_jaxpr(f)(shape)

    v1, v2 = make_vjp(2.0), make_vjp(3.0)
    kv1, kv2 = terms.parameter_key(v1), terms.parameter_key(v2)
    assert kv1 != kv2
    assert terms.opaque(kv1) == terms.opaque(kv2)
    with pytest.raises(_stated.Unestablished):
        _stated._plain(kv1, "custom_vjp_jaxpr")


def test_repeated_primitive_slots_keep_distinct_primitive_identities():
    from jax.extend import core

    def negate_twice(x):
        return jax.lax.neg(jax.lax.neg(x))

    traced = jax.make_jaxpr(negate_twice)(jax.ShapeDtypeStruct((8,), jnp.float32))
    first, second = traced.eqns
    other = core.Primitive(second.primitive.name)
    changed = traced.replace(jaxpr=traced.jaxpr.replace(eqns=[first, second.replace(primitive=other)]))
    assert terms.jaxpr_digest(traced, in_process=True) != terms.jaxpr_digest(changed, in_process=True)


def test_in_process_digest_distinguishes_which_link_each_equation_reads():
    shape = jax.ShapeDtypeStruct((), jnp.float32)

    def step(x, y):
        both = jax.lax.add(x, y)
        return jax.lax.sub(both, x)

    def other(x, y):
        both = jax.lax.add(x, y)
        return jax.lax.sub(both, y)

    first, clone, different = (jax.make_jaxpr(fn)(shape, shape) for fn in (step, jax.jit(step), other))
    key = terms.jaxpr_digest(first, in_process=True)
    assert key == terms.jaxpr_digest(clone.eqns[0].params["jaxpr"], in_process=True)
    assert key != terms.jaxpr_digest(different, in_process=True)


def test_process_digest_keeps_repeated_descriptors_in_equation_order():
    def formed(x, first, second):
        return jax.lax.reduce_precision(jax.lax.reduce_precision(x, 8, first), 8, second)

    shape = jax.ShapeDtypeStruct((4,), jnp.float32)
    direct = jax.make_jaxpr(lambda x: formed(x, first=4, second=8))(shape)
    reboxed = jax.make_jaxpr(lambda x: formed(x, 4, 8))(shape)
    reversed_ = jax.make_jaxpr(lambda x: formed(x, first=8, second=4))(shape)
    assert terms.body_key(direct) == terms.body_key(reboxed)
    assert terms.body_key(direct) != terms.body_key(reversed_)


def test_equal_string_parameters_with_distinct_encodings_stay_distinct():
    class Mode(str):
        direction: int

        def __new__(cls, direction):
            value = super().__new__(cls, "fold")
            value.direction = direction
            return value

        def __repr__(self):
            return f"Mode({self.direction})"

    first, second = Mode(1), Mode(-1)
    shape = jax.ShapeDtypeStruct((4,), jnp.float32)
    traced = jax.make_jaxpr(lambda x: jax.lax.neg(jax.lax.neg(x)))(shape)
    head, tail = traced.eqns
    head = head.replace(params={"mode": first})
    same = traced.replace(jaxpr=traced.jaxpr.replace(eqns=[head, tail.replace(params={"mode": first})]))
    changed = traced.replace(jaxpr=traced.jaxpr.replace(eqns=[head, tail.replace(params={"mode": second})]))
    assert terms.body_key(same) != terms.body_key(changed)


def _bf16_rms(x, w):
    """Plain RMSNorm in fp32 statistics for structural body-cache tests."""
    mean_sq = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x.astype(jnp.float32) * jax.lax.rsqrt(mean_sq + 1e-6)).astype(x.dtype) * w


def _named_bf16_rms(name: str):
    """Freshly named RMSNorm function so each `make_jaxpr` carries distinct `debug_info`."""
    def norm(x, w):
        return _bf16_rms(x, w)

    norm.__name__ = name
    return norm


def test_structural_closed_reuses_const_free_body_preserving_debug_info_and_reports():
    """Fresh structurally identical const-free bodies hit `program._STRUCTURAL_CLOSED`, replaying routes and
    rewrite reports while keeping each body's own `debug_info` and computing identical values."""
    from jax._src import core as jax_core

    from dew.nn.kernels.templates import report as kernel_report

    shapes = (jax.ShapeDtypeStruct((256, 256), jnp.bfloat16), jax.ShapeDtypeStruct((256,), jnp.bfloat16))
    ca = jax.make_jaxpr(_named_bf16_rms("norm_a"))(*shapes)
    cb = jax.make_jaxpr(_named_bf16_rms("norm_b"))(*shapes)
    assert ca is not cb and ca.debug_info != cb.debug_info
    with generation.assumed("sm90"):
        with generation.routes() as routes_a, kernel_report.recording():
            out_a = program._closed(ca, accurate=False, saturation=algebra.SATURATION, kernels=True)
        with generation.routes() as routes_b, kernel_report.recording():
            out_b = program._closed(cb, accurate=False, saturation=algebra.SATURATION, kernels=True)
    assert out_a is not ca and out_b is not cb
    assert out_a.debug_info == ca.debug_info
    assert out_b.debug_info == cb.debug_info
    assert any(r.choice == "rms_norm" and r.kernel != "xla" for r in routes_a)
    assert any(r.choice == "rms_norm" and r.kernel != "xla" for r in routes_b)
    jax_core.check_jaxpr(out_b.jaxpr)

    # `rules.head_primitives()` and `rules.heads()` stay aligned with and without dev rule sets enabled.
    base_prims = algebra.rules.head_primitives()
    assert "add" in base_prims and "div" not in base_prims
    with algebra.rules.enabling_dev("reuse"):
        reuse_prims = algebra.rules.head_primitives()
        assert "div" in reuse_prims
        assert algebra.rules.heads("div", [None, "sqrt"])
        assert not algebra.rules.heads("div", [None, None])


def test_structural_closed_isolates_callbacks_device_consts_and_prng_impls():
    """`program._STRUCTURAL_CLOSED` and `terms.body_key` distinguish `pure_callback` closures from the same
    factory, same-tag `define_prng_impl` specs with distinct implementations, and decline structural sharing
    without touching device memory when a nested `pjit` holds a concrete `jax.Array`."""
    from jax.core import ShapedArray, eval_jaxpr
    from jax.extend import random as jrandom

    def make_cb_body(offset: float):
        def bump(x):
            return x + np.asarray(offset, dtype=x.dtype)

        def body(x, w):
            normed = _bf16_rms(x, w)
            return jax.pure_callback(bump, jax.ShapeDtypeStruct(normed.shape, normed.dtype), normed)

        return body

    shapes = (jax.ShapeDtypeStruct((256, 256), jnp.bfloat16), jax.ShapeDtypeStruct((256,), jnp.bfloat16))
    cb1, cb5 = jax.make_jaxpr(make_cb_body(1.0))(*shapes), jax.make_jaxpr(make_cb_body(5.0))(*shapes)
    assert terms.body_key(cb1) != terms.body_key(cb5)
    with generation.assumed("sm90"):
        r1 = program._closed(cb1, accurate=False, saturation=algebra.SATURATION, kernels=True)
        r5 = program._closed(cb5, accurate=False, saturation=algebra.SATURATION, kernels=True)
    assert r1 is not cb1 and r5 is not cb5 and r1.jaxpr is not r5.jaxpr
    eq1 = next(eq for eq in r1.jaxpr.eqns if eq.primitive.name == "pure_callback")
    eq5 = next(eq for eq in r5.jaxpr.eqns if eq.primitive.name == "pure_callback")
    assert eq1.params["callback"] is not eq5.params["callback"]
    x_in, w_in = jnp.ones((256, 256), jnp.bfloat16), jnp.ones((256,), jnp.bfloat16)
    cpu1 = program._closed(cb1, accurate=False, saturation=algebra.SATURATION, kernels=True)
    cpu5 = program._closed(cb5, accurate=False, saturation=algebra.SATURATION, kernels=True)
    assert float(eval_jaxpr(cpu1.jaxpr, cpu1.consts, x_in, w_in)[0][0, 0]) == 2.0
    assert float(eval_jaxpr(cpu5.jaxpr, cpu5.consts, x_in, w_in)[0][0, 0]) == 6.0

    # Nested `jit` holding a concrete device `jax.Array` declines structural sharing without host copy.
    captured = jnp.ones((256,), jnp.bfloat16)

    @jax.jit
    def inner_with_const(x):
        return x + captured

    def outer_const_free(x, w):
        return inner_with_const(_bf16_rms(x, w))

    nested_closed = jax.make_jaxpr(outer_const_free)(*shapes)
    assert not nested_closed.consts
    assert terms.body_key(nested_closed) is None

    # Same-tag `define_prng_impl` specs with distinct `random_bits` implementations get distinct keys.
    def make_prng(fill: int):
        return jrandom.define_prng_impl(
            key_shape=(2,),
            seed=lambda seed: jnp.zeros((2,), jnp.uint32),
            split=lambda key, shape: jnp.broadcast_to(key, (*shape, 2)),
            random_bits=lambda key, bit_width, shape: jnp.full(shape, fill, jnp.uint32),
            fold_in=lambda key, word: key,
            name="shared_test_name",
            tag="shared_test_tag",
        )
    impl_a, impl_b = make_prng(11), make_prng(29)
    key_a, key_b = jax.random.key(0, impl=impl_a), jax.random.key(0, impl=impl_b)

    def draw_and_norm(k, x, w):
        return jax.random.bits(k, (4,), jnp.uint32), _bf16_rms(x, w)

    p_a = jax.make_jaxpr(draw_and_norm)(jax.typeof(key_a), *shapes)
    p_b = jax.make_jaxpr(draw_and_norm)(jax.typeof(key_b), *shapes)
    assert terms.body_key(p_a) != terms.body_key(p_b)
    with generation.assumed("sm90"):
        rp_a = program._closed(p_a, accurate=False, saturation=algebra.SATURATION, kernels=True)
        rp_b = program._closed(p_b, accurate=False, saturation=algebra.SATURATION, kernels=True)
    assert rp_a is not p_a and rp_b is not p_b and rp_a.jaxpr is not rp_b.jaxpr
    a0, b0 = rp_a.in_avals[0], rp_b.in_avals[0]
    assert isinstance(a0, ShapedArray) and isinstance(b0, ShapedArray) and a0.dtype != b0.dtype
    cpu_a = program._closed(p_a, accurate=False, saturation=algebra.SATURATION, kernels=True)
    cpu_b = program._closed(p_b, accurate=False, saturation=algebra.SATURATION, kernels=True)
    assert int(eval_jaxpr(cpu_a.jaxpr, cpu_a.consts, key_a, x_in, w_in)[0][0]) == 11
    assert int(eval_jaxpr(cpu_b.jaxpr, cpu_b.consts, key_b, x_in, w_in)[0][0]) == 29


def test_structural_closed_keys_on_drawing_context_in_both_orders():
    """Fresh structurally identical bodies traced under `generation.drawing()` and live `assumed('sm90')`
    never share a `_STRUCTURAL_CLOSED` entry in either cache order."""
    from dew.nn.kernels.templates import report as kernel_report

    shapes = (jax.ShapeDtypeStruct((256, 256), jnp.bfloat16), jax.ShapeDtypeStruct((256,), jnp.bfloat16))
    d1, n1 = (jax.make_jaxpr(_named_bf16_rms(name))(*shapes) for name in ("draw_first", "real_after"))
    n2, d2 = (jax.make_jaxpr(_named_bf16_rms(name))(*shapes) for name in ("real_first", "draw_after"))
    with generation.assumed("sm90"):
        with generation.drawing(), generation.routes() as draw_r1, kernel_report.recording():
            out_d1 = program._closed(d1, accurate=False, saturation=algebra.SATURATION, kernels=True)
        with generation.routes() as real_r1, kernel_report.recording():
            out_n1 = program._closed(n1, accurate=False, saturation=algebra.SATURATION, kernels=True)
        with generation.routes() as real_r2, kernel_report.recording():
            out_n2 = program._closed(n2, accurate=False, saturation=algebra.SATURATION, kernels=True)
        with generation.drawing(), generation.routes() as draw_r2, kernel_report.recording():
            out_d2 = program._closed(d2, accurate=False, saturation=algebra.SATURATION, kernels=True)
    assert not any(r.choice == "rms_norm" and r.kernel != "xla" for r in draw_r1)
    assert any(r.choice == "rms_norm" and r.kernel != "xla" for r in real_r1)
    assert any(r.choice == "rms_norm" and r.kernel != "xla" for r in real_r2)
    assert not any(r.choice == "rms_norm" and r.kernel != "xla" for r in draw_r2)
    assert out_d1 is d1 and out_n1 is not n1 and out_n2 is not n2 and out_d2 is d2


def test_body_key_handles_float0_dag_scaling_equation_context_and_scoped_array_gc():
    """`terms.body_key` and `_STRUCTURAL_CLOSED` handle `float0` tangent dtypes from integer gradients, stay
    linear on depth-30 shared nested-jit DAGs, preserve equation-local `cur_abstract_mesh` contexts in both
    cache orders, and never pin a `jax.Array` held only in a scoped `ContextVar`."""
    import contextvars
    import gc

    from jax.sharding import AbstractMesh, AxisType

    import dew

    grad_int = jax.grad(lambda x, i: (x * i).sum(), allow_int=True)
    f0_closed = jax.make_jaxpr(grad_int)(
        jax.ShapeDtypeStruct((4,), jnp.float32), jax.ShapeDtypeStruct((4,), jnp.int32))
    assert terms.body_key(f0_closed) is not None and terms.jaxpr_digest(f0_closed, in_process=True)[0]
    assert program._closed(f0_closed, accurate=False, saturation=algebra.SATURATION, kernels=True) is not None
    got_g = dew.jit(grad_int)(jnp.ones((4,), jnp.float32), jnp.arange(4, dtype=jnp.int32))
    np.testing.assert_array_equal(np.asarray(got_g), np.arange(4, dtype=jnp.float32))

    def build_dag(offset: float):
        fn = jax.jit(lambda x: x + offset)
        for _ in range(30):
            prev = fn
            fn = jax.jit(lambda x, f=prev: f(x) + f(x + 1.0))
        return jax.make_jaxpr(fn)(jax.ShapeDtypeStruct((4,), jnp.float32))

    dag_a, dag_b, dag_diff = build_dag(1.0), build_dag(1.0), build_dag(2.0)
    ka, kb, kd = terms.body_key(dag_a), terms.body_key(dag_b), terms.body_key(dag_diff)
    assert ka is not None and ka == kb and ka != kd
    assert terms.jaxpr_digest(dag_a, in_process=True) == terms.jaxpr_digest(dag_b, in_process=True)
    assert terms.jaxpr_digest(dag_a, in_process=True) != terms.jaxpr_digest(dag_diff, in_process=True)

    mesh_a = AbstractMesh((2,), ("data",), axis_types=(AxisType.Explicit,))
    mesh_b = AbstractMesh((4,), ("data",), axis_types=(AxisType.Explicit,))
    shapes = (jax.ShapeDtypeStruct((256, 256), jnp.bfloat16), jax.ShapeDtypeStruct((256,), jnp.bfloat16))

    def make_scoped_body(mesh: AbstractMesh, name: str):
        def body(x, w):
            normed = _bf16_rms(x, w)
            with jax.sharding.use_abstract_mesh(mesh):
                return normed + jnp.asarray(1.0, normed.dtype)
        body.__name__ = name
        return jax.make_jaxpr(body)(*shapes)

    for first_mesh, second_mesh, tag in ((mesh_a, mesh_b, "ab"), (mesh_b, mesh_a, "ba")):
        c1, c2 = make_scoped_body(first_mesh, f"ctx_{tag}_1"), make_scoped_body(second_mesh, f"ctx_{tag}_2")
        assert terms.body_key(c1) != terms.body_key(c2) and terms.jaxpr_digest(c1) != terms.jaxpr_digest(c2)
        with generation.assumed("sm90"):
            r1 = program._closed(c1, accurate=False, saturation=algebra.SATURATION, kernels=True)
            r2 = program._closed(c2, accurate=False, saturation=algebra.SATURATION, kernels=True)
        assert r1.jaxpr.eqns[-1].ctx.cur_abstract_mesh == first_mesh
        assert r2.jaxpr.eqns[-1].ctx.cur_abstract_mesh == second_mesh

    scoped_var: contextvars.ContextVar = contextvars.ContextVar("test_scoped_device_array", default=None)

    def run_in_scope() -> weakref.ref:
        buf = jnp.ones((16, 16), jnp.float32)
        ref, token = weakref.ref(buf), scoped_var.set(buf)
        try:
            traced = jax.make_jaxpr(lambda x: x + 1.0)(jax.ShapeDtypeStruct((4,), jnp.float32))
            program._closed(traced, accurate=False, saturation=algebra.SATURATION, kernels=True)
        finally:
            scoped_var.reset(token)
        return ref

    arr_ref = run_in_scope()
    gc.collect()
    assert arr_ref() is None


def test_body_key_distinguishes_callable_dataclasses_and_norm_rejects_bad_layouts():
    """Executable callable dataclasses (factory-made, `eq=False`, cyclic `eq=False`, and frozen cyclic
    `compare=False` in bare nested `jax.jit`) never hit `RecursionError` or cross-hit `_STRUCTURAL_CLOSED`;
    `_NormTrack` rejects permuting reshapes, singleton expansion, and lossy `bf16->fp16` casts."""
    import dew

    def make_factory_callback(offset: float):
        @dataclasses.dataclass(frozen=True)
        class OffsetCallback:
            def __call__(self, arr):
                return arr + np.float32(offset)
        return OffsetCallback()

    @dataclasses.dataclass(frozen=True, eq=False)
    class IdentityCallback:
        def __call__(self, arr):
            return arr + np.float32(1.0 if self is id_first else 2.0)

    @dataclasses.dataclass(eq=False)
    class CyclicCallback:
        offset: float
        parent: object = None
        def __call__(self, arr):
            return arr + np.float32(self.offset)

    @dataclasses.dataclass(frozen=True)
    class FrozenCyclic:
        offset: float
        parent: list = dataclasses.field(default_factory=list, compare=False, repr=False)
        def __call__(self, arr):
            return arr + np.float32(self.offset)

    id_first, id_second = IdentityCallback(), IdentityCallback()
    cyc1, cyc2 = CyclicCallback(1.0), CyclicCallback(2.0)
    cyc1.parent, cyc2.parent = cyc1, cyc2
    fc1, fc2 = FrozenCyclic(1.0), FrozenCyclic(2.0)
    fc1.parent.append(fc1)
    fc2.parent.append(fc2)
    rms_shapes = (jax.ShapeDtypeStruct((256, 256), jnp.bfloat16), jax.ShapeDtypeStruct((256,), jnp.bfloat16))
    x_in, w_in = jnp.ones((256, 256), jnp.bfloat16), jnp.ones((256,), jnp.bfloat16)
    exp2, exp3 = np.full((256, 256), 2.0, np.float32), np.full((256, 256), 3.0, np.float32)
    def make_body(cb, *, nested: bool = False, bare: bool = False):
        inner = jax.jit(lambda y: jax.pure_callback(cb, jax.ShapeDtypeStruct(y.shape, y.dtype), y))
        return lambda x, w: (lambda y: inner(y) if nested else jax.pure_callback(
            cb, jax.ShapeDtypeStruct(y.shape, y.dtype), y))(
            x.astype(jnp.float32) if bare else _bf16_rms(x, w).astype(jnp.float32))
    cases = ((make_factory_callback(1.0), make_factory_callback(2.0), True, False, False),
             (id_first, id_second, False, False, False), (cyc1, cyc2, False, True, False),
             (fc1, fc2, False, True, True))
    for cb1, cb2, shareable, is_nested, is_bare in cases:
        f1, f2 = (make_body(cb, nested=is_nested, bare=is_bare) for cb in (cb1, cb2))
        c1, c2 = jax.make_jaxpr(f1)(*rms_shapes), jax.make_jaxpr(f2)(*rms_shapes)
        k1, k2 = terms.body_key(c1), terms.body_key(c2)
        assert (k1 is not None and k2 is not None and k1 != k2) if shareable else (k1 is None and k2 is None)
        assert terms.jaxpr_digest(c1, in_process=True) != terms.jaxpr_digest(c2, in_process=True)
        assert shareable or terms.jaxpr_digest(c1) != terms.jaxpr_digest(c2)
        with generation.assumed("sm90"):
            r1 = program._closed(c1, accurate=False, saturation=algebra.SATURATION, kernels=True)
            r2 = program._closed(c2, accurate=False, saturation=algebra.SATURATION, kernels=True)
        assert r1.jaxpr != r2.jaxpr
        np.testing.assert_array_equal(np.asarray(dew.jit(f1)(x_in, w_in)), exp2)
        np.testing.assert_array_equal(np.asarray(dew.jit(f2)(x_in, w_in)), exp3)
    bad_norms = (
        (lambda x: x * jax.lax.rsqrt(jax.lax.reshape(
            jnp.sum(jnp.square(x), axis=-1, keepdims=True) + jnp.float32(1e-6),
            (4, 2, 2, 1), dimensions=(0, 2, 1, 3))), (4, 2, 2, 8), jnp.float32),
        (lambda x: x * jax.lax.broadcast_in_dim(jax.lax.rsqrt(
            jnp.sum(jnp.square(x), axis=-1, keepdims=True) + jnp.float32(1e-6)),
            (4, 2, 2, 64), (0, 1, 2, 3)), (4, 2, 2, 1), jnp.float32),
        (lambda x: x.astype(jnp.float32) * jax.lax.rsqrt(
            jnp.sum(jnp.square(x.astype(jnp.float16).astype(jnp.float32)), axis=-1, keepdims=True)
            + jnp.float32(2.0 ** -100)), (4, 2, 2, 64), jnp.bfloat16),
        (lambda x: x.astype(jnp.float32) * jax.lax.rsqrt(
            jnp.sum(jnp.square(x), axis=-1, keepdims=True).astype(jnp.float32)
            + jnp.float32(2.0 ** -100)), (4, 2, 2, 64), jnp.float16),
    )
    for bad_fn, shape, dt in bad_norms:
        closed = jax.make_jaxpr(bad_fn)(jax.ShapeDtypeStruct(shape, dt))
        held, apart = terms.proven(closed.jaxpr, targets=(closed.jaxpr.outvars[0],), deep=True)
        assert held[closed.jaxpr.outvars[0]] is None and not apart[closed.jaxpr.outvars[0]].unit_norm
