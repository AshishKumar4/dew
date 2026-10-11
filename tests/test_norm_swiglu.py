"""The interpreted fused norms and SiLU products against XLA and float64.

Every operand gradient is checked, including the fp32 weight's column sum
and Liger's fp32-intermediate VJP contract. Ragged tiles and enough rows to make
each norm program accumulate multiple blocks exercise the masked paths.
The fp32 activation/input-gradient fixtures retain a 1e-5 relative bound
from XLA. Every output, including dscale, keeps the project's 2x-XLA RMS
budget against the unrounded float64 function. A separate dscale check
isolates accumulation: float64 sums dy times the identical rounded forward
coefficient. Dew must be at least as close to that exact sum as XLA plus
one final output rounding; at 24,576 rows it must also stay within 2 ulp.
This does not reward XLA's lucky cancellation of coefficient-rounding
error. The bf16 VJPs keep wider intermediates than XLA's autodiff,
so their contract is distance from float64, not bitwise XLA agreement.
Forwards retain each model's cast sites. Small-shape c53 fixtures also hold
the fused VJPs to Liger's measured float64 distances.
"""

import functools
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.ad_checkpoint import print_saved_residuals
from reference_error import assert_as_exact_as_the_reference, distance

from dew.nn import attention, kernels
from dew.nn.attention import RMSNorm, forward_mode_attention, rms_normalized, unweighted_rmsnorm
from dew.nn.moe import ExpertMLP, gated_product


def relative(got, want):
    want = np.asarray(want, np.float64)
    return np.max(np.abs(np.asarray(got, np.float64) - want)) / max(np.max(np.abs(want)), 1.0)


def evaluated(function, operands, seed):
    output, backward = jax.vjp(function, *operands)
    return (output, *backward(seed))


def rounding_report(labels, actual, reference, truth):
    """On failure, compare every output's error to the same float64 truth.

    A GPU reference's column reduction can be less accurate than the fused
    reduction. Reporting both, not just relative-vs-XLA, distinguishes that
    from a kernel regression in one paid verify; assertion bounds stay unchanged.
    """
    report = {"backend": jax.default_backend(), "outputs": {}}
    names = {"x": "dx", "scale": "dscale", "gate": "dgate", "up": "dup"}
    for label, mine, theirs, exact in zip(labels, actual, reference, truth, strict=True):
        if mine is None:
            continue
        mine64, reference64, exact64 = (np.asarray(value, np.float64) for value in (mine, theirs, exact))
        report["outputs"][names.get(label, label)] = {
            "shape": list(mine.shape), "dtype": str(mine.dtype),
            "dew_rms_from_float64": distance(mine64, exact64),
            "xla_rms_from_float64": distance(reference64, exact64),
            "dew_max_from_float64": float(np.max(np.abs(mine64 - exact64))),
            "xla_max_from_float64": float(np.max(np.abs(reference64 - exact64))),
            "relative_vs_xla": float(relative(mine64, reference64)),
        }
    return "NORM_SWIGLU_FLOAT64_DIAGNOSTIC\n" + json.dumps(report, indent=2)


def scale_coefficient(x, dtype, after_cast: bool):
    """The fused forward's normalized coefficient, before its optional weight."""
    output = dtype if after_cast else jnp.float32
    return rms_normalized(x, None, 1e-5, output, scale_offset=False, scale_after_cast=False,
                           fp32_statistics=True, implementation="pallas")


def assert_scale_accumulation(mine, reference, coefficient, seed):
    """Float64 exact sum of identical rounded factors, not the unrounded RMS function."""
    terms = np.asarray(coefficient, np.float64) * np.asarray(seed, np.float64)
    exact_sum = np.sum(terms.reshape(-1, terms.shape[-1]), axis=0, dtype=np.float64)
    rounded = jnp.asarray(exact_sum, mine.dtype)
    above = np.asarray(jnp.nextafter(rounded, jnp.full_like(rounded, jnp.inf)), np.float64)
    below = np.asarray(jnp.nextafter(rounded, jnp.full_like(rounded, -jnp.inf)), np.float64)
    nearest = np.asarray(rounded, np.float64)
    ulp = np.maximum(above - nearest, nearest - below)
    final_rounding = float(np.sqrt(np.mean(np.square(0.5 * ulp))))
    actual_error, reference_error = distance(mine, exact_sum), distance(reference, exact_sum)
    detail = (f"dscale accumulation: dew={actual_error:.12g}, xla={reference_error:.12g}, "
              f"final rounding={final_rounding:.12g}, rows={terms.size // terms.shape[-1]}")
    assert actual_error <= reference_error + final_rounding, detail
    if terms.size // terms.shape[-1] >= 24576:
        assert np.all(np.abs(np.asarray(mine, np.float64) - exact_sum) <= 2 * ulp), detail


def compare(functions, operands, seed, labels, *, coefficient=None):
    xla, pallas, exact = functions
    graph = str(jax.make_jaxpr(pallas)(*operands))
    assert "rms_norm_forward" in graph or "swiglu_forward" in graph
    differentiated = str(jax.make_jaxpr(functools.partial(evaluated, pallas))(operands, seed))
    assert "rms_norm_backward" in differentiated or "swiglu_backward" in differentiated
    reference = jax.jit(functools.partial(evaluated, xla))(operands, seed)
    actual = jax.jit(functools.partial(evaluated, pallas))(operands, seed)
    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(new_val=True), jax.default_device(cpu):
        wide = jax.tree.map(lambda x: jnp.asarray(np.asarray(x, np.float64)), operands)
        truth = evaluated(exact, wide, jnp.asarray(np.asarray(seed, np.float64)))
    try:
        for label, mine, theirs, want in zip(labels, actual, reference, truth, strict=True):
            if mine is None:
                assert theirs is None and want is None
                continue
            assert mine.dtype == theirs.dtype, label
            if label == "scale":
                assert coefficient is not None
                assert_scale_accumulation(mine, theirs, coefficient, seed)
            elif mine.dtype == jnp.float32 and reference[0].dtype == jnp.float32:
                assert relative(mine, theirs) < 1e-5, label
            assert_as_exact_as_the_reference(mine, theirs, want, label)
    except AssertionError as error:
        report = rounding_report(labels, actual, reference, truth)
        print(report, flush=True)
        error.add_note(report)
        raise


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
    compare(norm_functions(dtype, offset, after_cast), operands, seed, ("y", "x", "scale"),
            coefficient=scale_coefficient(operands[0], dtype, after_cast))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("shape", [(7, 1000), (3, 4096), (3, 257, 17)])
def test_rms_ragged_widths_and_many_rows(dtype, shape):
    operands, seed = norm_operands(shape, dtype, jnp.float32)
    compare(norm_functions(dtype, offset=True, after_cast=True), operands, seed, ("y", "x", "scale"),
            coefficient=scale_coefficient(operands[0], dtype, after_cast=True))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_rms_weight_gradient_accumulates_a_full_step_of_rows(dtype):
    # Repeated tokens hold the row arithmetic fixed while the column sum
    # grows to Qwen3.5-0.8B's size, isolating partial-sum error from bf16 midpoint flips.
    (row, scale), seed = norm_operands((1, 1000), dtype, jnp.float32)
    x = jnp.broadcast_to(row, (24576, 1000))
    seed = jnp.broadcast_to(seed, x.shape)
    compare(norm_functions(dtype, offset=True, after_cast=True), (x, scale), seed, ("y", "x", "scale"),
            coefficient=scale_coefficient(x, dtype, after_cast=True))


def test_random_bf16_gradients_keep_fp32_intermediates_near_float64():
    """The real-row stress holds every output and gradient to XLA's float64
    RMS budget, plus the independent exact-sum accumulation check. BF16
    gradients deliberately do not reproduce XLA's intermediate roundings."""
    operands, seed = norm_operands((24576, 17), jnp.bfloat16, jnp.float32)
    xla, pallas, exact = norm_functions(jnp.bfloat16, offset=True, after_cast=True)
    reference = jax.jit(functools.partial(evaluated, xla))(operands, seed)
    actual = jax.jit(functools.partial(evaluated, pallas))(operands, seed)
    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(new_val=True), jax.default_device(cpu):
        wide = jax.tree.map(lambda x: jnp.asarray(np.asarray(x, np.float64)), operands)
        truth = evaluated(exact, wide, jnp.asarray(np.asarray(seed, np.float64)))
    try:
        for label, mine, theirs, want in zip(("y", "x", "scale"), actual, reference, truth, strict=True):
            assert_as_exact_as_the_reference(mine, theirs, want, label)
            if label == "scale":
                assert_scale_accumulation(mine, theirs, scale_coefficient(
                    operands[0], jnp.bfloat16, after_cast=True), seed)
    except AssertionError as error:
        report = rounding_report(("y", "x", "scale"), actual, reference, truth)
        print(report, flush=True)
        error.add_note(report)
        raise


@pytest.mark.parametrize("input_dtype,output_dtype", [(jnp.bfloat16, jnp.float32),
                                                       (jnp.float32, jnp.bfloat16)])
@pytest.mark.parametrize("after_cast", [False, True])
def test_rms_output_dtype_is_independent_of_input(input_dtype, output_dtype, after_cast):
    operands, seed = norm_operands((5, 37), input_dtype, jnp.float32)
    compare(norm_functions(output_dtype, offset=False, after_cast=after_cast), operands,
            seed.astype(output_dtype), ("y", "x", "scale"),
            coefficient=scale_coefficient(operands[0], output_dtype, after_cast))


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


def test_expert_jvp_refuses_swiglu_once_and_matches_xla(monkeypatch, caplog):
    x = jax.random.normal(jax.random.key(73), (2, 3, 8))
    indices = jnp.zeros((2, 3, 1), jnp.int32)
    weights = jnp.ones((2, 3, 1), jnp.float32)
    expert = ExpertMLP(num_experts=2, hidden_features=16, out_features=8, implementation="xla")
    params = expert.init(jax.random.key(74), x, weights, indices)

    def function(x):
        return expert.apply(params, x, weights, indices)

    reference = jax.jvp(function, (x,), (jnp.ones_like(x),))
    monkeypatch.setitem(kernels.KERNELS, "swiglu", {"sm80": "pallas"})
    monkeypatch.setattr(kernels.generation, "device_generation", lambda: "sm80")
    monkeypatch.setattr(kernels.generation, "triton_runs", lambda: True)
    monkeypatch.setattr(kernels.generation, "_logged", set())
    for _ in range(2):
        actual = jax.jvp(function, (x,), (jnp.ones_like(x),))
        for mine, theirs in zip(actual, reference, strict=True):
            np.testing.assert_array_equal(mine, theirs)
    logged = [record.getMessage() for record in caplog.records if record.name == kernels.generation.__name__]
    assert len(logged) == 1 and "JVP trace" in logged[0]


def test_norm_jvp_without_a_context_uses_xla():
    x = jax.random.normal(jax.random.key(75), (2, 31))
    xla, pallas, _ = norm_functions(jnp.float32, offset=True, after_cast=True)
    for mine, theirs in zip(jax.jvp(lambda x: pallas(x, None), (x,), (x,)),
                            jax.jvp(lambda x: xla(x, None), (x,), (x,)), strict=True):
        np.testing.assert_array_equal(mine, theirs)


# Same-session c53, A100-SXM4-40GB, Liger 0.8.4 / torch 2.11.0+cu130:
# dew-aux-c53-norm-swiglu-20261010-job-1, result.json, accuracy shape [257,127].
LIGER_RMS_ERRORS = {
    "rms_norm": (0.0017684225615251462, 0.0012577999558486422, 0.03934815194557525),
    "swiglu": (0.0014667377536021014, 0.0010115300863026556, 0.0009935534908141895),
}


@pytest.mark.parametrize("operation", ["rms_norm", "swiglu"])
def test_c53_gradients_are_as_close_to_float64_as_xla_and_measured_liger(operation):
    """The measured operands and independent float64 equations from c53.

    CPU and CUDA can differ by a few fp32 ulps inside a reduction or exp.
    Four fp32 epsilons of the truth's RMS allow that backend floor, not an
    extra bf16 rounding: the old rounded VJPs miss these bounds by ~40%.
    """
    shape = (257, 127)
    x = jnp.asarray(np.random.default_rng(61).standard_normal(shape, dtype=np.float32), jnp.bfloat16)
    seed = jnp.asarray(np.random.default_rng(62).standard_normal(shape, dtype=np.float32), jnp.bfloat16)
    second = np.random.default_rng(63).standard_normal(
        (shape[-1],) if operation == "rms_norm" else shape, dtype=np.float32)
    if operation == "rms_norm":
        scale = jnp.asarray(np.float32(0.7) + np.float32(0.2) * second)
        operands = (x, scale)

        def function(implementation):
            return lambda x, scale: rms_normalized(
                x, scale, 1e-6, jnp.bfloat16, scale_offset=False, scale_after_cast=True,
                fp32_statistics=True, implementation=implementation)
    else:
        operands = (x, jnp.asarray(second, jnp.bfloat16))

        def function(implementation):
            return gated_product("swiglu", implementation=implementation)

    reference = jax.jit(functools.partial(evaluated, function("xla")))(operands, seed)
    actual = jax.jit(functools.partial(evaluated, function("pallas")))(operands, seed)
    first, second = (np.asarray(value, np.float64) for value in operands)
    dy = np.asarray(seed, np.float64)
    if operation == "rms_norm":
        inverse = 1 / np.sqrt(np.mean(first * first, axis=-1, keepdims=True) + 1e-6)
        normalized, m = first * inverse, dy * second
        truth = (normalized * second,
                 inverse * (m - normalized * np.mean(m * normalized, axis=-1, keepdims=True)),
                 np.sum(dy * normalized, axis=0))
    else:
        sigmoid = 1 / (1 + np.exp(-first))
        activated = first * sigmoid
        truth = (activated * second, dy * second * (sigmoid + activated * (1 - sigmoid)), dy * activated)
    for index, (mine, theirs, want, measured) in enumerate(zip(
            actual, reference, truth, LIGER_RMS_ERRORS[operation], strict=True)):
        floor = 4 * np.finfo(np.float32).eps * np.sqrt(np.mean(np.square(want)))
        assert distance(mine, want) <= min(distance(theirs, want), measured) + floor, (operation, index)
