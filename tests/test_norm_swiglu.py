"""The interpreted fused norms and SiLU products against XLA and float64.

Every operand gradient is checked, including the fp32 weight's column sum
and Dew's bf16 cotangent roundings. Ragged tiles and enough rows to make
each norm program accumulate multiple blocks exercise the masked paths.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.ad_checkpoint import print_saved_residuals
from reference_error import assert_as_exact_as_the_reference

from dew.nn import attention, kernels
from dew.nn.attention import RMSNorm, forward_mode_attention, rms_normalized, unweighted_rmsnorm
from dew.nn.moe import gated_product


def relative(got, want):
    want = np.asarray(want, np.float64)
    return np.max(np.abs(np.asarray(got, np.float64) - want)) / max(np.max(np.abs(want)), 1.0)


def evaluated(function, operands, seed):
    output, backward = jax.vjp(function, *operands)
    return (output, *backward(seed))


def compare(functions, operands, seed, labels):
    xla, pallas, exact = functions
    graph = str(jax.make_jaxpr(pallas)(*operands))
    assert "rms_norm_forward" in graph or "swiglu_forward" in graph
    reference = jax.jit(functools.partial(evaluated, xla))(operands, seed)
    actual = jax.jit(functools.partial(evaluated, pallas))(operands, seed)
    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(new_val=True), jax.default_device(cpu):
        wide = jax.tree.map(lambda x: jnp.asarray(np.asarray(x, np.float64)), operands)
        truth = evaluated(exact, wide, jnp.asarray(np.asarray(seed, np.float64)))
    for label, mine, theirs, want in zip(labels, actual, reference, truth, strict=True):
        if mine is None:
            assert theirs is None and want is None
            continue
        assert mine.dtype == theirs.dtype, label
        assert relative(mine, theirs) < 1e-5, label
        assert_as_exact_as_the_reference(mine, theirs, want, label)


def norm_functions(dtype, offset, after_cast):
    def norm(implementation, output_dtype, x, scale):
        return rms_normalized(x, scale, 1e-5, output_dtype, offset, after_cast, fp32_statistics=True,
                              implementation=implementation)

    return (functools.partial(norm, "xla", dtype), functools.partial(norm, "pallas", dtype),
            functools.partial(norm, "xla", jnp.float64))


def norm_operands(shape, dtype, scale_dtype):
    x = jax.random.normal(jax.random.key(23), shape).astype(dtype)
    scale = None if scale_dtype is None else (
        0.7 + 0.2 * jax.random.normal(jax.random.key(24), (shape[-1],))).astype(scale_dtype)
    seed = jax.random.normal(jax.random.key(25), shape).astype(dtype)
    return (x, scale), seed


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("scale_dtype", [None, jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("offset", [False, True])
@pytest.mark.parametrize("after_cast", [False, True])
def test_rms_variants_and_every_gradient(dtype, scale_dtype, offset, after_cast):
    operands, seed = norm_operands((2, 3, 31), dtype, scale_dtype)
    compare(norm_functions(dtype, offset, after_cast), operands, seed, ("y", "x", "scale"))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("shape", [(7, 1000), (3, 4096), (3, 257, 17)])
def test_rms_ragged_widths_and_many_rows(dtype, shape):
    operands, seed = norm_operands(shape, dtype, jnp.float32)
    compare(norm_functions(dtype, offset=True, after_cast=True), operands, seed, ("y", "x", "scale"))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_rms_weight_gradient_accumulates_a_full_step_of_rows(dtype):
    # Repeated tokens hold the row arithmetic fixed while the column sum
    # grows to Qwen3.5-0.8B's size, isolating partial-sum error from bf16 midpoint flips.
    (row, scale), seed = norm_operands((1, 1000), dtype, jnp.float32)
    x = jnp.broadcast_to(row, (24576, 1000))
    seed = jnp.broadcast_to(seed, x.shape)
    compare(norm_functions(dtype, offset=True, after_cast=True), (x, scale), seed, ("y", "x", "scale"))


@pytest.mark.parametrize("input_dtype,output_dtype", [(jnp.bfloat16, jnp.float32),
                                                       (jnp.float32, jnp.bfloat16)])
@pytest.mark.parametrize("after_cast", [False, True])
def test_rms_output_dtype_is_independent_of_input(input_dtype, output_dtype, after_cast):
    operands, seed = norm_operands((5, 37), input_dtype, jnp.float32)
    compare(norm_functions(output_dtype, offset=False, after_cast=after_cast), operands,
            seed.astype(output_dtype), ("y", "x", "scale"))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 1000), (3, 4096), (3, 257, 17)])
def test_swiglu_forward_and_both_gradients(dtype, shape):
    operands = tuple(jax.random.normal(jax.random.key(index), shape).astype(dtype) for index in (31, 32))
    seed = jax.random.normal(jax.random.key(33), shape).astype(dtype)
    xla, pallas = (gated_product("swiglu", implementation=name) for name in ("xla", "pallas"))
    compare((xla, pallas, xla), operands, seed, ("y", "gate", "up"))


def test_zero_rows_have_finite_norm_gradients():
    x = jnp.zeros((5, 1000), jnp.bfloat16)
    functions = norm_functions(jnp.bfloat16, offset=False, after_cast=True)
    expected = evaluated(functions[0], (x, None), jnp.ones_like(x))
    actual = jax.jit(functools.partial(evaluated, functions[1]))((x, None), jnp.ones_like(x))
    for mine, theirs in zip(actual[:2], expected[:2], strict=True):
        np.testing.assert_array_equal(mine, theirs)
        assert np.all(np.isfinite(np.asarray(mine, np.float32)))


@pytest.mark.parametrize("offset,after_cast,with_scale", [(False, True, True), (True, False, True),
                                                         (False, True, False)])
def test_rms_module_uses_the_same_fused_vjp(offset, after_cast, with_scale):
    x = jax.random.normal(jax.random.key(4), (2, 5, 31))
    module = RMSNorm(scale_offset=offset, scale_after_cast=after_cast, with_scale=with_scale,
                     implementation="pallas")
    params = module.init(jax.random.key(5), x)
    portable = module.clone(implementation="xla")
    mine = jax.jit(jax.value_and_grad(lambda p, x: jnp.sum(module.apply(p, x)), argnums=(0, 1)))(params, x)
    theirs = jax.jit(jax.value_and_grad(lambda p, x: jnp.sum(portable.apply(p, x)),
                                       argnums=(0, 1)))(params, x)
    for actual, expected in zip(jax.tree.leaves(mine), jax.tree.leaves(theirs), strict=True):
        assert relative(actual, expected) < 1e-5
    assert "rms_norm_forward" in str(jax.make_jaxpr(module.apply)(params, x))


@pytest.mark.parametrize("statistics,dtype,scale_shape,width,reason", [
    (False, jnp.float32, (31,), 31, "input-dtype statistics"),
    (True, jnp.float64, (31,), 31, "float64"),
    (True, jnp.float32, (1,), 31, "broadcast affine"),
    (True, jnp.float32, (8193,), 8193, "8192 features"),
])
def test_rms_refusals_fall_back_and_log_once(monkeypatch, caplog, statistics, dtype,
                                           scale_shape, width, reason):
    from dew.nn.kernels import norm_swiglu

    monkeypatch.setitem(kernels.KERNELS, "rms_norm", {"cpu": "pallas"})
    monkeypatch.setattr(kernels.generation, "_logged", set())
    with jax.enable_x64(new_val=True):
        x, scale = jnp.ones((2, width), dtype), jnp.ones(scale_shape, dtype)
        assert norm_swiglu.rms_kernel("pallas", x, scale, dtype, statistics) == "xla"
        assert norm_swiglu.rms_kernel("pallas", x, scale, dtype, statistics) == "xla"
        mine = rms_normalized(x, scale, 1e-5, dtype, scale_offset=False, scale_after_cast=True,
                               fp32_statistics=statistics, implementation="pallas")
        theirs = rms_normalized(x, scale, 1e-5, dtype, scale_offset=False, scale_after_cast=True,
                                 fp32_statistics=statistics, implementation="xla")
        np.testing.assert_array_equal(mine, theirs)
    logged = [record.getMessage() for record in caplog.records if record.name == kernels.generation.__name__]
    assert len(logged) == 1 and reason in logged[0]


def test_swiglu_refuses_broadcast_mixed_and_wide_dtypes():
    from dew.nn.kernels import norm_swiglu

    with jax.enable_x64(new_val=True):
        for gate, up in ((jnp.ones((2, 17)), jnp.ones((17,))),
                         (jnp.ones((2, 17)), jnp.ones((2, 17), jnp.bfloat16)),
                         (jnp.ones((2, 17), jnp.float64), jnp.ones((2, 17), jnp.float64))):
            assert norm_swiglu.swiglu_kernel("pallas", gate, up) == "xla"
            np.testing.assert_array_equal(gated_product("swiglu", implementation="pallas")(gate, up),
                                          gated_product("swiglu", implementation="xla")(gate, up))


def test_cpu_auto_and_forward_mode_are_portable():
    from dew.nn.kernels import norm_swiglu

    x = jnp.ones((2, 17), jnp.float32)
    assert norm_swiglu.rms_kernel("auto", x, None, x.dtype, fp32_statistics=True) == "xla"
    assert norm_swiglu.swiglu_kernel("auto", x, x) == "xla"
    with forward_mode_attention():
        assert norm_swiglu.rms_kernel("pallas", x, None, x.dtype, fp32_statistics=True) == "xla"
        assert norm_swiglu.swiglu_kernel("pallas", x, x) == "xla"
        jax.jvp(lambda x: rms_normalized(x, None, 1e-5, x.dtype, scale_offset=False,
                                         scale_after_cast=False, fp32_statistics=True,
                                         implementation="pallas"), (x,), (x,))
        jax.jvp(gated_product("swiglu", implementation="pallas"), (x, x), (x, x))


@pytest.mark.parametrize("name", ["geglu", "geglu_exact"])
def test_other_gates_keep_their_arithmetic(name):
    gate, up = jnp.arange(17, dtype=jnp.float32) / 7 - 1, jnp.ones(17)
    np.testing.assert_array_equal(gated_product(name, implementation="pallas")(gate, up),
                                  gated_product(name, implementation="xla")(gate, up))


def test_selection_uses_the_generation_table(monkeypatch):
    monkeypatch.setitem(kernels.KERNELS, "rms_norm", {"sm80": "pallas"})
    monkeypatch.setitem(kernels.KERNELS, "swiglu", {"sm80": "pallas"})
    monkeypatch.setattr(kernels.generation, "device_generation", lambda: "sm80")
    monkeypatch.setattr(kernels.generation, "triton_runs", lambda: True)
    x = jnp.ones((2, 17), jnp.float32)
    graph = str(jax.make_jaxpr(lambda x: attention.rms_normalized(
        x, None, 1e-5, x.dtype, scale_offset=False, scale_after_cast=False, fp32_statistics=True))(x))
    assert "rms_norm_forward" in graph
    assert "swiglu_forward" in str(jax.make_jaxpr(gated_product("swiglu"))(x, x))


def test_unknown_implementation_is_rejected():
    x = jnp.ones((2, 17))
    with pytest.raises(ValueError, match="implementation"):
        rms_normalized(x, None, 1e-5, x.dtype, scale_offset=False, scale_after_cast=False,
                        fp32_statistics=True, implementation="other")
    with pytest.raises(ValueError, match="implementation"):
        gated_product("swiglu", implementation="other")


def test_inverse_cast_variant_refuses_the_normalized_activation_kernel(monkeypatch, caplog):
    monkeypatch.setitem(kernels.KERNELS, "rms_norm", {"cpu": "pallas"})
    monkeypatch.setattr(kernels.generation, "_logged", set())
    x = jax.random.normal(jax.random.key(15), (3, 31)).astype(jnp.bfloat16)
    reference = evaluated(lambda x: unweighted_rmsnorm(x, 1e-5, implementation="xla"), (x,), x)
    for _ in range(2):
        actual = evaluated(lambda x: unweighted_rmsnorm(x, 1e-5, implementation="pallas"), (x,), x)
        for mine, theirs in zip(actual, reference, strict=True):
            np.testing.assert_array_equal(mine, theirs)
    logged = [record.getMessage() for record in caplog.records if record.name == kernels.generation.__name__]
    assert len(logged) == 1 and "rounds the inverse" in logged[0]


@pytest.mark.parametrize("operation", ["rms_norm", "swiglu"])
def test_fused_residuals_keep_no_full_width_fp32_activation(operation, capsys):
    x = jnp.ones((2, 3, 31), jnp.bfloat16)
    if operation == "rms_norm":
        function = norm_functions(jnp.bfloat16, offset=False, after_cast=True)[1]
        operands = (x, jnp.ones((31,), jnp.float32))
    else:
        function, operands = gated_product("swiglu", implementation="pallas"), (x, x)
    print_saved_residuals(function, *operands)
    saved = capsys.readouterr().out
    assert "bf16[2,3,31]" in saved and "f32[2,3,31]" not in saved, saved


@pytest.mark.mesh(devices=2)
def test_split_mesh_operands_refuse_both_local_kernels():
    from dew.nn.kernels import norm_swiglu
    from dew.training import MeshSpec

    x = jnp.ones((2, 31), jnp.float32)
    with jax.set_mesh(MeshSpec(fsdp=2).build(jax.devices()[:2])):
        assert norm_swiglu.rms_kernel("pallas", x, None, x.dtype, fp32_statistics=True) == "xla"
        assert norm_swiglu.swiglu_kernel("pallas", x, x) == "xla"
    assert norm_swiglu.rms_kernel("pallas", x, None, x.dtype, fp32_statistics=True) == "pallas"
    assert norm_swiglu.swiglu_kernel("pallas", x, x) == "pallas"
