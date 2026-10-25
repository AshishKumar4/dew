"""Dew's CPU policy for float32 products (`dew.nn.precision.accumulated`) in the suite's wide CPU
arithmetic (`dew.nn.precision.WIDE_CPU_ARITHMETIC`, which tests/lane_environment.py sets): a CPU sums
them in float64 and rounds once, through every product a layer forms, its gradient's and tangent's
included, and a bias's gradient (`dew.nn.precision.spread`), with x64 off; 16-bit products and other
backends keep their own arithmetic. A library process, without the setting, traces the plain
products."""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import equations, with_gradients

from dew.nn.dense import Dense
from dew.nn.precision import (
    FP32_SUMS,
    WIDE_CPU_ARITHMETIC,
    accumulated,
    float32_dot_general,
    float32_einsum,
    float32_matmul,
)

pytestmark = pytest.mark.skipif(jax.default_backend() != "cpu", reason="the policy is the CPU's")

CANCELLING = np.asarray([2.0 ** 24, 1.0, -(2.0 ** 24)], np.float32)
"""Terms whose float32 sum, in the order given, loses the 1 (2^24 + 1 rounds to 2^24), and whose
float64 sum is 1 exactly."""


def products():
    """Each public product over `CANCELLING` against ones, as a [1, 3] by [3, 1] contraction."""
    lhs, rhs = CANCELLING[None], np.ones((3, 1), np.float32)
    return {
        "dot_general": lambda: float32_dot_general(lhs, rhs, (((1,), (0,)), ((), ()))),
        "einsum": lambda: float32_einsum("ik,kj->ij", lhs, rhs),
        "matmul": lambda: float32_matmul(lhs, rhs),
    }


@pytest.mark.parametrize("name", sorted(products()))
def test_a_float32_product_sums_in_float64_and_rounds_once(name):
    """2^24 + 1 - 2^24 is 1: the float64 sum rounded once, where float32 accumulation in that order
    gives 0; under jit and without, x64 off."""
    assert not jax.config.jax_enable_x64
    product = products()[name]
    assert np.asarray(product()).tolist() == [[1.0]]
    result = jax.jit(product)()
    assert result.dtype == jnp.float32 and np.asarray(result).tolist() == [[1.0]]


def test_a_layers_gradient_and_tangent_sum_in_float64():
    """A Dense layer forming its product as Dew's layers do (`float32_dot_general`): its kernel
    gradient sums the inputs' products over the batch in float64, and its output tangent sums the
    input tangent's, both 1 exactly over the cancelling terms."""
    layer = Dense(1, use_bias=False, dot_general=float32_dot_general)
    x = CANCELLING[:, None]
    params = {"params": {"kernel": np.ones((1, 1), np.float32)}}
    _, gradient = with_gradients(lambda params: layer.apply(params, x))(params, np.ones((3, 1), np.float32))
    assert np.asarray(gradient["params"]["kernel"]).tolist() == [[1.0]]
    rows = Dense(1, use_bias=False, dot_general=float32_dot_general)
    weights = {"params": {"kernel": np.ones((3, 1), np.float32)}}
    _, tangent = jax.jvp(lambda x: rows.apply(weights, x), (np.zeros((1, 3), np.float32),),
                         (CANCELLING[None],))
    assert np.asarray(tangent).tolist() == [[1.0]]


def test_a_bias_gradient_sums_its_positions_in_float64():
    """A bias broadcast over the positions it adds to (`dew.nn.precision.biased`) takes the sum of
    their cotangents in float64, rounded once: 1 exactly over the cancelling terms, where a float32
    sum in that order gives 0; its forward adds the bias itself to every position, and its tangent
    the bias's own."""
    layer = Dense(1)
    x = np.zeros((3, 1), np.float32)
    params = {"params": {"kernel": np.ones((1, 1), np.float32), "bias": np.asarray([0.5], np.float32)}}
    applied = jax.jit(with_gradients(lambda params: layer.apply(params, x)))
    out, gradient = applied(params, CANCELLING[:, None])
    assert np.asarray(out).tolist() == [[0.5]] * 3
    assert np.asarray(gradient["params"]["bias"]).tolist() == [1.0]
    tangent = {"params": {"kernel": np.zeros((1, 1), np.float32), "bias": np.asarray([0.25], np.float32)}}
    _, moved = jax.jvp(lambda params: layer.apply(params, x), (params,), (tangent,))
    assert np.asarray(moved).tolist() == [[0.25]] * 3


def test_16_bit_products_keep_their_float32_accumulation():
    """bfloat16 products, exact in float32, accumulate as the backend's dot accumulates them, and
    come back in the dtype asked."""
    lhs = jnp.asarray([[1.0, 2.0, 3.0]], jnp.bfloat16)
    rhs = jnp.ones((3, 1), jnp.bfloat16)
    dims = (((1,), (0,)), ((), ()))
    for preferred in (None, jnp.float32):
        ours = float32_dot_general(lhs, rhs, dims, preferred_element_type=preferred)
        theirs = jax.lax.dot_general(lhs, rhs, dims, preferred_element_type=preferred)
        assert ours.dtype == theirs.dtype
        np.testing.assert_array_equal(np.asarray(ours, np.float32), np.asarray(theirs, np.float32))


def test_a_float32_product_returns_the_dtype_asked():
    """A float32 product asked for a float32 result returns one, rounded once from the float64 sum."""
    lhs, rhs = CANCELLING[None], np.ones((3, 1), np.float32)
    dims = (((1,), (0,)), ((), ()))
    result = jax.jit(lambda: float32_dot_general(lhs, rhs, dims, preferred_element_type=jnp.float32))()
    assert result.dtype == jnp.float32 and float(result[0, 0]) == 1.0


@pytest.mark.parametrize("assumed_first", [True, False])
def test_the_policy_is_the_executing_platforms_whatever_target_a_trace_assumed(assumed_first):
    """One jitted product, run on a CPU inside and outside `generation.assumed` in either order,
    gives the CPU policy's float64 sum each time: a trace's assumed target chooses kernels, never the
    sum a CPU executes, so no cached trace carries another call's policy."""
    from dew.nn.kernels.generation import assumed

    product = jax.jit(lambda lhs, rhs: float32_dot_general(lhs, rhs, (((1,), (0,)), ((), ()))))
    lhs, rhs = CANCELLING[None], np.ones((3, 1), np.float32)
    results = []
    for inside in (assumed_first, not assumed_first):
        with assumed("sm80") if inside else contextlib.nullcontext():
            results.append(float(product(lhs, rhs)[0, 0]))
    assert results == [1.0, 1.0]


@pytest.mark.parametrize("stated_first", [True, False])
def test_a_scope_that_states_fp32_sums_gets_them(stated_first):
    """One jitted product, run on a CPU with fp32 sums stated (`FP32_SUMS`) and without, in either
    order, sums the cancelling terms in fp32 in the stated scope (to zero, the 1 lost to 2**24) and in
    float64 outside it (to 1): the statement keys the compiled program, as a precision does."""
    product = jax.jit(lambda lhs, rhs: float32_dot_general(lhs, rhs, (((1,), (0,)), ((), ()))))
    lhs, rhs = CANCELLING[None], np.ones((3, 1), np.float32)
    results = {}
    for stated in (stated_first, not stated_first):
        with jax.default_matmul_precision(FP32_SUMS) if stated else contextlib.nullcontext():
            results[stated] = float(product(lhs, rhs)[0, 0])
    assert results == {True: 0.0, False: 1.0}


def test_a_bias_spread_over_a_sharded_axis_keeps_its_sharding():
    """A batched bias, `[2, 1, 4]` spread to `[2, 3, 4]`, its last axis split over an Explicit mesh,
    adds itself to every position and takes the cancelling cotangents' float64 sum, 1, under jit:
    the spread drops only its size-one axis, never merging a split one into another."""
    from jax.sharding import AxisType, NamedSharding, PartitionSpec

    from dew.nn.precision import spread

    mesh = jax.make_mesh((2,), ("model",), axis_types=(AxisType.Explicit,), devices=jax.devices()[:2])
    with jax.set_mesh(mesh):
        value = jax.device_put(np.full((2, 1, 4), 0.5, np.float32),
                               NamedSharding(mesh, PartitionSpec(None, None, "model")))
        cotangent = np.broadcast_to(CANCELLING[None, :, None], (2, 3, 4)).copy()
        out, gradient = jax.jit(with_gradients(lambda value: spread(value, (2, 3, 4))))(value, cotangent)
    assert np.asarray(out).tolist() == np.full((2, 3, 4), 0.5).tolist()
    assert np.asarray(gradient).tolist() == np.ones((2, 1, 4)).tolist()


def test_another_threads_quantized_trace_leaves_this_threads_products_dews():
    """While one thread traces a quantized model's method (`QUANTIZED_TRACE`, whose products Qwix
    owns), a product another thread compiles, and the same program after that trace ends, sum in
    float64: the ownership is the tracing thread's alone."""
    import threading

    from dew.cache import QUANTIZED_TRACE

    product = jax.jit(lambda lhs, rhs: float32_einsum("ik,kj->ij", lhs, rhs))
    lhs, rhs = CANCELLING[None], np.ones((3, 1), np.float32)
    inside, done, results = threading.Event(), threading.Event(), []

    def quantized():
        with QUANTIZED_TRACE("int8"):
            inside.set()
            done.wait()

    held = threading.Thread(target=quantized)
    held.start()
    inside.wait()
    results.append(float(product(lhs, rhs)[0, 0]))
    done.set()
    held.join()
    results.append(float(product(lhs, rhs)[0, 0]))
    assert results == [1.0, 1.0]


@pytest.mark.parametrize("precision", ["F32_F32_F32", jax.lax.DotAlgorithmPreset.F32_F32_F32,
                                       jax.lax.Precision.DEFAULT], ids=["preset-name", "preset", "default"])
def test_a_product_that_names_its_precision_still_sums_in_float64(precision):
    """A product bound to an fp32 algorithm or precision of its own, outside a scope that states fp32
    sums, sums in float64 on a CPU, forward and gradient: the float64 products run at the precision
    that keeps them float64."""
    lhs, rhs = CANCELLING[None], np.ones((3, 1), np.float32)
    product = jax.jit(with_gradients(
        lambda lhs, rhs: float32_einsum("ik,kj->ij", lhs, rhs, precision=precision)))
    out, _, into_rhs = product(lhs, rhs, np.ones((1, 1), np.float32))
    assert np.asarray(out).tolist() == [[1.0]]
    assert np.asarray(into_rhs).tolist() == CANCELLING[:, None].tolist()


def test_a_mixed_product_converts_its_operands_as_the_product_does():
    """An int32 operand multiplies a float32 one as float32, as JAX converts it: 2**24 + 1 becomes
    2**24, so the terms cancel to 0 exactly, which the float64 sum keeps; the float operand's gradient
    is the converted integers, the integer held fixed. Asked for a float64 result, the product
    converts both to float64 and keeps the 1, as JAX's does."""
    ints = np.asarray([2 ** 24 + 1, -(2 ** 24)], np.int32)
    floats = np.ones(2, np.float32)
    assert float(jax.jit(lambda ints, floats: float32_einsum("i,i->", ints, floats))(ints, floats)) == 0.0
    gradient = jax.jit(jax.grad(lambda floats: float32_einsum("i,i->", ints, floats)))(floats)
    assert np.asarray(gradient).tolist() == [2.0 ** 24, -(2.0 ** 24)]
    with jax.enable_x64(new_val=True):
        wide = jax.jit(lambda ints, floats: float32_einsum("i,i->", ints, floats,
                                                         preferred_element_type=jnp.float64))(ints, floats)
        assert wide.dtype == jnp.float64 and float(wide) == 1.0
    # With x64 off a float64 result is a float32 one, whose products sum in float64.
    lhs, rhs = CANCELLING[None], np.ones((3, 1), np.float32)
    narrowed = jax.jit(lambda lhs, rhs: float32_einsum("ik,kj->ij", lhs, rhs,
                                                       preferred_element_type=jnp.float64))(lhs, rhs)
    assert narrowed.dtype == jnp.float32 and np.asarray(narrowed).tolist() == [[1.0]]
    assert float(accumulated(jnp.dot, 2.0, 3.0, preferred_element_type=jnp.float32)) == 6.0


def doubling(kind: str):
    """A dot stating its own derivative, twice the dot's: as a `jax.custom_jvp` or a `jax.custom_vjp`
    (Qwix's kind, when it quantizes a model directly)."""
    def dot(x, y, preferred_element_type=None):
        return jnp.dot(x, y, preferred_element_type=preferred_element_type)

    if kind == "custom_jvp":
        doubled = jax.custom_jvp(dot, nondiff_argnums=(2,))

        @doubled.defjvp
        def _(preferred_element_type, primals, tangents):
            (x, y), (dx, dy) = primals, tangents
            return dot(x, y), 2 * (jnp.dot(dx, y) + jnp.dot(x, dy))
        return doubled
    doubled = jax.custom_vjp(dot, nondiff_argnums=(2,))

    def forward(x, y, preferred_element_type):
        return dot(x, y), (x, y)

    def backward(preferred_element_type, saved, cotangent):
        x, y = saved
        return 2 * jnp.dot(cotangent, y.T), 2 * jnp.dot(x.T, cotangent)

    doubled.defvjp(forward, backward)
    return doubled


@pytest.mark.parametrize("kind", ["custom_jvp", "custom_vjp"])
def test_a_product_that_states_its_own_derivative_keeps_it(kind):
    """A product with a derivative of its own (a `jax.custom_jvp`, or a `jax.custom_vjp` as Qwix's
    are when it quantizes a model directly) is its library's: `accumulated`'s gradient through it is
    its own, twice the dot's here, where a transpose of its primal would give the dot's."""
    doubled = doubling(kind)
    lhs, rhs = CANCELLING[None], np.ones((3, 1), np.float32)
    gradient = jax.jit(jax.grad(lambda rhs: jnp.sum(accumulated(doubled, lhs, rhs))))(rhs)
    assert np.asarray(gradient).tolist() == [[2.0 ** 25], [2.0], [-(2.0 ** 25)]]


def test_a_portable_recurrence_product_sums_in_float64():
    """The recurrence's portable products (`recurrence_algebra.portable` and the weighed `paired`)
    are real CPU execution's, and sum in float64: 1 exactly over the cancelling terms."""
    from dew.nn.kernels.recurrence_algebra import paired, portable

    lhs, rhs = CANCELLING[None], np.ones((3, 1), np.float32)
    assert np.asarray(jax.jit(lambda lhs, rhs: portable("product", lhs, rhs))(lhs, rhs)).tolist() == [[1.0]]
    left, right = CANCELLING[None, :], np.ones((1, 3), np.float32)
    decay = np.ones((1, 1, 3), np.float32)
    weighed = jax.jit(lambda *operands: paired("product", *operands))(left, right, decay)
    assert np.asarray(weighed).tolist() == [[1.0]]


@pytest.mark.parametrize("wide", [True, False], ids=["test setting", "library"])
def test_wide_cpu_arithmetic_is_the_test_settings_alone(wide, monkeypatch):
    """Each site of the wide CPU arithmetic, a float32 product and a bias's spread, a row's logsumexp
    and a float64 division, traces a platform choice in a process running Dew's test setting
    (`WIDE_CPU_ARITHMETIC`), and in a library process, which sets nothing, the plain computation:
    no platform choice and no float64 beside a float32 computation. Each is traced afresh, as a
    process with either setting traces it."""
    from dew.diffusion.schedules.common import divided
    from dew.nn.protocols import OutputTable
    from dew.objectives.lm.chunked import chunked_cross_entropy

    if wide:
        monkeypatch.setenv(WIDE_CPU_ARITHMETIC, "1")
    else:
        monkeypatch.delenv(WIDE_CPU_ARITHMETIC, raising=False)
    lhs, rhs = np.ones((4, 3), np.float32), np.ones((3, 2), np.float32)
    layer = Dense(2)
    variables = layer.init(jax.random.key(0), lhs)
    states, table = np.ones((2, 5, 8), np.float32), np.ones((7, 8), np.float32)
    targets = np.zeros((2, 5), np.int32)

    def traced(fn, *arguments):
        """The equations of `fn` traced afresh on `arguments` (a new function, so no trace is reused)."""
        return list(equations(jax.make_jaxpr(lambda *values: fn(*values))(*arguments)))

    def float64(found) -> bool:
        return any(var.aval.dtype == jnp.float64 for equation in found
                   for var in (*equation.invars, *equation.outvars) if hasattr(var, "aval"))

    def chosen(found) -> bool:
        return any(equation.primitive.name == "platform_index" for equation in found)

    sites = {
        "product": traced(lambda a, b: float32_dot_general(a, b, (((1,), (0,)), ((), ()))), lhs, rhs),
        "bias": traced(jax.grad(lambda params: jnp.sum(layer.apply({"params": params}, lhs))),
                       variables["params"]),
        "logsumexp": traced(lambda states, table: chunked_cross_entropy(
            states, OutputTable(table, vocab_major=True), targets, 1, tile=None)[0], states, table),
    }
    with jax.enable_x64(new_val=True):
        division = traced(divided, np.ones(3, np.float64), np.float64(3.0))
    for name, found in sites.items():
        assert chosen(found) == wide, name
        assert float64(found) == wide, name
    assert chosen(division) == wide
