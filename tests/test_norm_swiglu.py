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

import contextlib
import functools
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from reference_error import assert_as_exact_as_the_reference, distance, equations, float64_terms, widened
from tuned import each_config

from dew.nn import attention
from dew.nn.activations import gated_product, packed_gated_product
from dew.nn.attention import RMSNorm, gated_rms_normalized, rms_normalized, stream_rms_normalized
from dew.nn.kernels import norm_swiglu, tuning

pytestmark = pytest.mark.kernel


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


def traced(fn, *args):
    """The equations of `fn`'s traced program, its sub-programs' in turn."""
    return list(equations(jax.make_jaxpr(fn)(*args)))


def kernels(fn, *args) -> set[str]:
    """The Pallas kernels `fn`'s traced program launches, by name."""
    return {str(equation.params["name"]) for equation in traced(fn, *args)
            if equation.primitive.name == "pallas_call"}


def compare(functions, operands, seed, labels, *, coefficient=None):
    pallas = functions[1]
    assert kernels(pallas, *operands) & {"rms_norm_forward", "swiglu_forward"}
    differentiated = kernels(functools.partial(evaluated, pallas), operands, seed)
    assert differentiated & {"rms_norm_backward", "swiglu_backward"}
    reference, (truth, terms) = oracles(functions, operands, seed)
    actual = jax.jit(functools.partial(evaluated, pallas))(operands, seed)
    gated(labels, actual, reference, truth, terms, seed, coefficient)


def oracles(functions, operands, seed):
    """XLA's outputs and gradients, and float64 truth with its terms (`float64`)."""
    xla, _, exact = functions
    return jax.jit(functools.partial(evaluated, xla))(operands, seed), float64(exact, operands, seed)


def float64(function, operands, seed):
    """`function`'s outputs and gradients under `seed`, evaluated in float64 on the CPU, and each one's
    float64 terms (`float64_terms`), what its sums add, by which the gate sizes its rows."""
    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(new_val=True), jax.default_device(cpu):
        return float64_terms(functools.partial(evaluated, function), *widened((operands, seed)))


def gated(labels, actual, reference, truth, terms, seed, coefficient):
    """Every output and gradient within the float64 gate."""
    try:
        for label, mine, theirs, want, summed in zip(labels, actual, reference, truth, terms, strict=True):
            if mine is None:
                assert theirs is None and want is None
                continue
            assert mine.dtype == theirs.dtype, label
            if label == "scale":
                assert coefficient is not None
                assert_scale_accumulation(mine, theirs, coefficient, seed)
            elif mine.dtype == jnp.float32 and reference[0].dtype == jnp.float32:
                assert relative(mine, theirs) < 1e-5, label
            assert_as_exact_as_the_reference(mine, theirs, want, label, terms=summed)
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
@pytest.mark.parametrize("case", ["random-2x3x31", "random-3x257x17", "random-5x2x128", "aligned-1x128",
                                  "large-1x128", "small-1x128"])
def test_the_unweighted_norm_rounds_its_inverse_and_takes_every_gradient_fused(dtype, case):
    """`attention.unweighted_rmsnorm` fused (`norm_swiglu.rms_norm`'s `inverse_rounded`): the inverse RMS
    reduced in fp32 and rounded to the input's dtype before it multiplies, as DeepSeek-V4's and GLM's
    unweighted norms round it, and its reverse rounded where XLA's is; its output and input gradient
    within the float64 gate of XLA's, random rows, a row of 1.5 under a cotangent of ones, whose two
    gradient contributions cancel, and rows of 0.5 under cotangents of 1.5 * 2**119, finite through
    XLA's transposes, and of 2**-140, which they do not halve."""
    def unweighted(x):
        return attention.unweighted_rmsnorm(x, 1e-6, implementation="xla")

    def fused(x):
        return norm_swiglu.rms_norm(x, None, 1e-6, x.dtype, offset=False, after_cast=False,
                                    inverse_rounded=True)

    kind, shape = case.split("-")
    shape = tuple(int(size) for size in shape.split("x"))
    if kind == "aligned":
        x, seed = jnp.full(shape, 1.5, dtype), jnp.ones(shape, dtype)
    elif kind in ("large", "small"):
        cotangent = 1.5 * 2.0 ** 119 if kind == "large" else 2.0 ** -140
        x, seed = jnp.full(shape, 0.5, dtype), jnp.full(shape, cotangent, dtype)
    else:
        x = jax.random.normal(jax.random.key(26), shape).astype(dtype)
        seed = jax.random.normal(jax.random.key(27), shape).astype(dtype)
    compare((unweighted, fused, unweighted), (x,), seed, ("y", "x"))


def test_an_unweighted_norm_takes_the_fused_kernel_and_keeps_a_staged_jvp():
    """Traced for an A100, whose row names the fused RMSNorm, `unweighted_rmsnorm` (Mamba2's,
    Nemotron-H's, DeepSeek-V4's and GLM-5's) takes the kernel, differentiated as one custom JVP
    (`generation.linear_tangent`): a jvp of a jit that staged it is the portable norm's, and its
    gradient runs the kernel's backward."""
    from dew.nn.kernels import generation

    x = jax.random.normal(jax.random.key(28), (4, 64)).astype(jnp.bfloat16)
    tangent = jax.random.normal(jax.random.key(29), (4, 64)).astype(jnp.bfloat16)
    with generation.assumed("sm80"), generation.routes() as found:
        jax.make_jaxpr(lambda x: attention.unweighted_rmsnorm(x, 1e-6))(x)
    (route,) = [route for route in found if route.choice == "rms_norm"]
    assert (route.kernel, route.preferred, route.refusal) == ("pallas", "pallas", None)

    def norm(implementation):
        return jax.jit(lambda x: attention.unweighted_rmsnorm(x, 1e-6, implementation=implementation))

    staged = jax.jvp(norm("pallas"), (x,), (tangent,))
    portable = jax.jvp(norm("xla"), (x,), (tangent,))
    np.testing.assert_array_equal(np.asarray(staged[1]), np.asarray(portable[1]))

    def pulled(x):
        return jax.grad(lambda x: norm("pallas")(x).astype(jnp.float32).sum())(x)

    assert "rms_norm_backward" in kernels(pulled, x)
    assert pulled(x).dtype == x.dtype


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
    (x, scale), seed = full_step(dtype)
    compare(norm_functions(dtype, offset=True, after_cast=True), (x, scale), seed, ("y", "x", "scale"),
            coefficient=scale_coefficient(x, dtype, after_cast=True))


def full_step(dtype):
    """E1's 24,576 rows of one repeated 1000-wide token, so the weight
    gradient's sums reach a step's length (`test_rms_weight_gradient...`)."""
    (row, scale), seed = norm_operands((1, 1000), dtype, jnp.float32)
    x = jnp.broadcast_to(row, (24576, 1000))
    return (x, scale), jnp.broadcast_to(seed, x.shape)


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", [norm_swiglu.RmsForward, norm_swiglu.RmsBackward])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("operands", ["full step", "wide rows", "narrow rows"])
def test_every_config_of_the_norm_kernels_holds_the_float64_gate(kernel, dtype, operands):
    """Each config the tuner may pick, at a step's rows, at rows wider than
    a tile holds two of, and at rows narrow enough for its tallest tiles:
    the space declares only configs proven here."""
    (x, scale), seed = {"full step": lambda: full_step(dtype),
                        "wide rows": lambda: norm_operands((7, 4096), dtype, jnp.float32),
                        "narrow rows": lambda: norm_operands((4099, 256), dtype, jnp.float32)}[operands]()
    functions = norm_functions(dtype, offset=True, after_cast=True)
    reference, (truth, terms) = oracles(functions, (x, scale), seed)
    configs = each_config(kernel, functools.partial(evaluated, functions[1]), (x, scale), seed)
    assert len(configs) > 1
    for config in configs:
        with tuning.pinned({kernel: config}):
            actual = jax.jit(functools.partial(evaluated, functions[1]))((x, scale), seed)
            # The factors this config's forward rounded, which its weight gradient sums.
            coefficient = scale_coefficient(x, dtype, after_cast=True)
        try:
            gated(("y", "x", "scale"), actual, reference, truth, terms, seed, coefficient)
        except AssertionError as error:
            error.add_note(f"{kernel.__name__} at {config}")
            raise


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", [norm_swiglu.SwigluForward, norm_swiglu.SwigluBackward])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_every_config_of_the_swiglu_kernels_holds_the_float64_gate(kernel, dtype):
    """Each block and warp count, over a ragged tail of elements."""
    shape = (3, 257, 17)
    operands = tuple(jax.random.normal(jax.random.key(index), shape).astype(dtype) for index in (31, 32))
    seed = jax.random.normal(jax.random.key(33), shape).astype(dtype)
    xla, pallas = (gated_product("swiglu", implementation=name) for name in ("xla", "pallas"))
    reference, (truth, terms) = oracles((xla, pallas, xla), operands, seed)
    configs = each_config(kernel, functools.partial(evaluated, pallas), operands, seed)
    assert len(configs) > 1
    for config in configs:
        with tuning.pinned({kernel: config}):
            actual = jax.jit(functools.partial(evaluated, pallas))(operands, seed)
        try:
            gated(("y", "gate", "up"), actual, reference, truth, terms, seed, None)
        except AssertionError as error:
            error.add_note(f"{kernel.__name__} at {config}")
            raise


def test_random_bf16_gradients_keep_fp32_intermediates_near_float64():
    """The real-row stress holds every output and gradient to XLA's float64
    RMS budget, plus the independent exact-sum accumulation check. BF16
    gradients deliberately do not reproduce XLA's intermediate roundings."""
    operands, seed = norm_operands((24576, 17), jnp.bfloat16, jnp.float32)
    xla, pallas, exact = norm_functions(jnp.bfloat16, offset=True, after_cast=True)
    reference = jax.jit(functools.partial(evaluated, xla))(operands, seed)
    actual = jax.jit(functools.partial(evaluated, pallas))(operands, seed)
    truth, terms = float64(exact, operands, seed)
    try:
        for label, mine, theirs, want, summed in zip(("y", "x", "scale"), actual, reference, truth, terms,
                                                     strict=True):
            assert_as_exact_as_the_reference(mine, theirs, want, label, terms=summed)
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


def stream_seam(implementation: str, dtype, joined: bool, offset: bool = True, after_cast: bool = True):
    """The pre-norm seam `(stream, normed)` over `x`, plus `branch` where `joined`."""
    def seam(x, branch, scale):
        return stream_rms_normalized(x, branch if joined else None, scale, 1e-5, dtype, offset, after_cast,
                                     fp32_statistics=True, implementation=implementation)
    return seam


@pytest.mark.parametrize("joined", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 31), (3, 257, 17), (5, 1000)])
def test_stream_seam_takes_the_add_and_the_stream_cotangent(joined, dtype, shape):
    """The kernel's sum is XLA's to the bit; the norm and every gradient,
    the stream's cotangent folded into dx, hold the float64 gate."""
    (x, scale), dy = norm_operands(shape, dtype, jnp.float32)
    branch = jax.random.normal(jax.random.key(26), shape).astype(dtype)
    ds = jax.random.normal(jax.random.key(27), shape).astype(dtype)
    operands, seed = (x, branch, scale), (ds, dy)
    fused = stream_seam("pallas", dtype, joined)
    assert "rms_norm_forward" in kernels(fused, *operands)
    assert "rms_norm_backward" in kernels(functools.partial(evaluated, fused), operands, seed)
    reference = jax.jit(functools.partial(evaluated, stream_seam("xla", dtype, joined)))(operands, seed)
    actual = jax.jit(functools.partial(evaluated, fused))(operands, seed)
    truth, terms = float64(stream_seam("xla", jnp.float64, joined), operands, seed)
    (stream, normed), *gradients = actual
    (reference_stream, reference_normed), *reference_gradients = reference
    np.testing.assert_array_equal(stream, reference_stream)
    labels = ("y", "x", "branch", "scale")
    for label, mine, theirs, want, summed in zip(labels, (normed, *gradients),
                                                 (reference_normed, *reference_gradients),
                                                 (truth[0][1], *truth[1:]), (terms[0][1], *terms[1:]),
                                                 strict=True):
        if label == "branch" and not joined:
            assert not np.any(np.asarray(mine)) and not np.any(np.asarray(theirs))
            continue
        assert mine.dtype == theirs.dtype, label
        if label == "scale":
            assert_scale_accumulation(mine, theirs, scale_coefficient(stream, dtype, after_cast=True), dy)
        assert_as_exact_as_the_reference(mine, theirs, want, label, terms=summed)


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 4, 31), (2, 64, 16, 128)])
@pytest.mark.parametrize("gate_dtype", ["same", jnp.bfloat16])
def test_gated_norm_and_every_gradient_hold_the_float64_gate(activation, dtype, shape, gate_dtype):
    """Qwen3-Next's gated norm: the kernel's output and its input, gate and
    weight gradients each within the gate of XLA's float64 distance, its
    gate in the input's dtype or in bf16 beside an fp32 input, as a decode
    step's recurrence hands it over."""
    x, gate, dy = (jax.random.normal(jax.random.key(index), shape).astype(dtype) for index in (51, 52, 53))
    gate = gate if gate_dtype == "same" else gate.astype(gate_dtype)
    scale = 0.7 + 0.2 * jax.random.normal(jax.random.key(54), shape[-1:])
    operands = (x, gate, scale)

    def gated(implementation, output_dtype):
        return lambda x, gate, scale: gated_rms_normalized(x, gate, scale, 1e-6, output_dtype, activation,
                                                           implementation=implementation)

    fused = gated("pallas", dtype)
    assert "rms_norm_forward" in kernels(fused, *operands)
    assert "rms_norm_backward" in kernels(functools.partial(evaluated, fused), operands, dy)
    reference = jax.jit(functools.partial(evaluated, gated("xla", dtype)))(operands, dy)
    actual = jax.jit(functools.partial(evaluated, fused))(operands, dy)
    truth, terms = float64(gated("xla", jnp.float64), operands, dy)
    for label, mine, theirs, want, summed in zip(("y", "x", "gate", "scale"), actual, reference, truth, terms,
                                                 strict=True):
        assert mine.dtype == theirs.dtype, label
        assert_as_exact_as_the_reference(mine, theirs, want, label, terms=summed)


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("output", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 4, 31), (2, 1, 16, 128)])
def test_a_decode_steps_gated_norm_of_an_fp32_input_by_a_bf16_gate_holds_the_float64_gate(activation, output,
                                                                                           shape):
    """A decode step's gated norm (Qwen3-Next's, Kimi's): the recurrence's fp32 output gated by its bf16
    projection, the gate read in fp32 as XLA widens it, cast once to `output`; its output and its input,
    gate and weight gradients, each in its operand's dtype, within the float64 gate of XLA's."""
    x, gate, dy = (jax.random.normal(jax.random.key(index), shape) for index in (61, 62, 63))
    gate, dy = gate.astype(jnp.bfloat16), dy.astype(output)
    scale = 0.7 + 0.2 * jax.random.normal(jax.random.key(64), shape[-1:])
    operands = (x, gate, scale)

    def gated(implementation, dtype):
        return lambda x, gate, scale: gated_rms_normalized(x, gate, scale, 1e-6, dtype, activation,
                                                           implementation=implementation)

    fused = gated("pallas", output)
    assert "rms_norm_forward" in kernels(fused, *operands)
    assert "rms_norm_backward" in kernels(functools.partial(evaluated, fused), operands, dy)
    reference = jax.jit(functools.partial(evaluated, gated("xla", output)))(operands, dy)
    actual = jax.jit(functools.partial(evaluated, fused))(operands, dy)
    truth, terms = float64(gated("xla", jnp.float64), operands, dy)
    for label, mine, theirs, want, summed in zip(("y", "x", "gate", "scale"), actual, reference, truth, terms,
                                                 strict=True):
        assert mine.dtype == theirs.dtype, label
        assert_as_exact_as_the_reference(mine, theirs, want, label, terms=summed)


def test_a_decode_steps_gated_norm_takes_the_fused_kernel_where_its_row_names_it():
    """Traced for an A100, whose row names the fused RMSNorm, the gated norm of a decode step (Qwen3-Next's,
    Kimi's): an fp32 recurrence output gated by its bf16 projection reports the kernel, not a refusal."""
    from dew.nn.kernels import generation

    x, gate, scale = jnp.ones((2, 4, 128), jnp.float32), jnp.ones((2, 4, 128), jnp.bfloat16), jnp.ones((128,))
    with generation.assumed("sm80"), generation.routes() as found:
        jax.make_jaxpr(lambda x, gate, scale: gated_rms_normalized(
            x, gate, scale, 1e-6, jnp.bfloat16, "silu"))(x, gate, scale)
    (route,) = [route for route in found if route.choice == "rms_norm"]
    assert (route.kernel, route.preferred, route.refusal) == ("pallas", "pallas", None)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 1000), (3, 4096), (3, 257, 17)])
def test_swiglu_forward_and_both_gradients(dtype, shape):
    operands = tuple(jax.random.normal(jax.random.key(index), shape).astype(dtype) for index in (31, 32))
    seed = jax.random.normal(jax.random.key(33), shape).astype(dtype)
    xla, pallas = (gated_product("swiglu", implementation=name) for name in ("xla", "pallas"))
    compare((xla, pallas, xla), operands, seed, ("y", "gate", "up"))


@pytest.mark.parametrize("implementation", ["pallas", "xla"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 1000), (3, 4096), (3, 257, 17)])
def test_a_packed_projections_swiglu_is_the_apart_one_read_in_place(implementation, dtype, shape):
    """One packed projection's `[..., 2, H]` output, gate then up: the
    kernel's packed launch and XLA's product read the halves where they lie
    and give the apart product's output and both gradients bit for bit, the
    gradient in the packed layout, so the apart product's float64 gate
    holds for them."""
    gate_up = jax.random.normal(jax.random.key(34), (*shape[:-1], 2, shape[-1])).astype(dtype)
    seed = jax.random.normal(jax.random.key(35), shape).astype(dtype)
    packed = functools.partial(evaluated, packed_gated_product("swiglu", implementation=implementation))
    if implementation == "pallas":
        assert "swiglu_backward" in kernels(packed, (gate_up,), seed)
    output, dpacked = jax.jit(packed)((gate_up,), seed)
    apart = jax.jit(functools.partial(evaluated, gated_product("swiglu", implementation=implementation)))(
        (gate_up[..., 0, :], gate_up[..., 1, :]), seed)
    for mine, theirs in zip((output, dpacked[..., 0, :], dpacked[..., 1, :]), apart, strict=True):
        np.testing.assert_array_equal(mine, theirs)


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
    assert "rms_norm_forward" in kernels(module.apply, params, x)


def test_swiglu_refuses_broadcast_mixed_and_wide_dtypes():
    from dew.nn.kernels import norm_swiglu

    with jax.enable_x64(new_val=True):
        for gate, up in ((jnp.ones((2, 17)), jnp.ones((17,))),
                         (jnp.ones((2, 17)), jnp.ones((2, 17), jnp.bfloat16)),
                         (jnp.ones((2, 17), jnp.float64), jnp.ones((2, 17), jnp.float64))):
            assert norm_swiglu.swiglu_kernel("pallas", gate, up) == "xla"
            np.testing.assert_array_equal(gated_product("swiglu", implementation="pallas")(gate, up),
                                          gated_product("swiglu", implementation="xla")(gate, up))


@pytest.mark.parametrize("named", [None, "chosen", "call"])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("packed", [False, True])
def test_a_swiglu_takes_the_kernel_its_target_measured_at_its_width(dtype, named, packed):
    """On an A100's row (`generation.assumed`), a bf16 SiLU product with no kernel named takes the
    Pallas kernel the row measured, and an fp32 one, which no row measured and XLA fuses with its
    neighbours, takes XLA as its own preference; a kernel named by `chosen` or per call runs at
    either width. Each product, apart or packed, reports the route it took."""
    from dew.nn.kernels import generation, norm_swiglu

    operands = (jnp.ones((2, 2, 4096), dtype),) if packed else (jnp.ones((2, 4096), dtype),) * 2
    resolve = norm_swiglu.packed_swiglu_kernel if packed else norm_swiglu.swiglu_kernel
    kernel = "pallas" if named or dtype == jnp.bfloat16 else "xla"
    named_by = generation.chosen(swiglu="pallas") if named == "chosen" else contextlib.nullcontext()
    with generation.assumed("sm80"), named_by, generation.routes() as found:
        assert resolve("pallas" if named == "call" else "auto", *operands) == kernel
    (route,) = found
    assert (route.kernel, route.preferred, route.refusal) == (kernel, kernel, None)


def test_a_swiglu_of_a_bf16_gate_over_an_fp32_up_is_an_fp32_product_xla_takes():
    """Gemma-4's per-layer input gates a bf16 projection over its fp32 input: an fp32 product, which
    XLA fuses (`_measured_width`), so on an A100 it takes XLA as its own preference, reporting no
    refusal, and computes as XLA's product does."""
    from dew.nn.kernels import generation

    gate, up = jnp.ones((2, 63, 8), jnp.bfloat16), jnp.ones((2, 63, 8), jnp.float32)
    with generation.assumed("sm80"), generation.routes() as found:
        assert norm_swiglu.swiglu_kernel("auto", gate, up) == "xla"
    (route,) = found
    assert (route.kernel, route.preferred, route.refusal) == ("xla", "xla", None)


@pytest.mark.gpu
@pytest.mark.parametrize("implementation", ["auto", "pallas"])
def test_a_gpu_hosts_cpu_runs_the_seams_its_placement_chooses(without_deterministic_ops, implementation):
    """On a GPU host, the norm, the SiLU product and attention placed on the CPU, eagerly under the
    default device and in a JIT on a CPU mesh, as a host-layout run places its state, run there:
    'auto' takes no GPU kernel, and an explicit Pallas kernel interprets."""
    cpu = jax.devices("cpu")[0]

    def seams(x):
        y = rms_normalized(x, None, 1e-5, x.dtype, scale_offset=False, scale_after_cast=False,
                           fp32_statistics=True, implementation=implementation)
        y = gated_product("swiglu", implementation=implementation)(y, y).reshape(1, 4, 2, 64)
        return attention.attention_kernel(y, y, y).astype(jnp.float32).sum()

    x = jax.device_put(jnp.linspace(-1, 1, 512, dtype=jnp.bfloat16).reshape(4, 128), cpu)
    with jax.default_device(cpu):
        eager = jax.value_and_grad(seams)(x)
    with jax.set_mesh(Mesh(np.array([cpu]), ("data",))):
        jitted = jax.jit(jax.value_and_grad(seams))(x)
    assert {device for value in (*eager, *jitted) for device in value.devices()} == {cpu}


@pytest.mark.parametrize("name", ["geglu", "geglu_exact"])
def test_other_gates_keep_their_arithmetic(name):
    gate, up = jnp.arange(17, dtype=jnp.float32) / 7 - 1, jnp.ones(17)
    np.testing.assert_array_equal(gated_product(name, implementation="pallas")(gate, up),
                                  gated_product(name, implementation="xla")(gate, up))


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("split", ["fsdp", "tensor"])
def test_seam_and_product_run_per_shard_as_on_one_device(split):
    """Named logical axes place the kernels on a mesh: rows split, the norm's
    features whole, the weight's gradient summed over the shards, and a
    packed SwiGLU's pair of halves whole."""
    from dew.nn.sharding import MLP_HIDDEN, RESIDUAL
    from dew.training import MeshSpec

    (x, scale), dy = norm_operands((4, 8, 31), jnp.float32, jnp.float32)
    branch, ds = (jax.random.normal(jax.random.key(index), x.shape) for index in (28, 29))
    gate, up, dout = (jax.random.normal(jax.random.key(index), (4, 8, 24)) for index in (30, 31, 32))

    def run(axes, hidden):
        seam = evaluated(lambda x, branch, scale: stream_rms_normalized(
            x, branch, scale, 1e-5, x.dtype, scale_offset=True, scale_after_cast=True, fp32_statistics=True,
            implementation="pallas", axes=axes),
            (x, branch, scale), (ds, dy))
        product = evaluated(gated_product("swiglu", implementation="pallas", axes=hidden), (gate, up), dout)
        packed = evaluated(packed_gated_product("swiglu", implementation="pallas", axes=hidden),
                           (jnp.stack([gate, up], -2),), dout)
        return seam, product, packed

    alone = jax.jit(functools.partial(run, RESIDUAL, MLP_HIDDEN))()
    with jax.set_mesh(MeshSpec(**{split: 2}).build(jax.devices()[:2])):
        graph = traced(functools.partial(run, RESIDUAL, MLP_HIDDEN))
        assert sum(equation.primitive.name == "shard_map" for equation in graph) >= 2
        assert {"rms_norm_forward", "swiglu_forward"} <= {str(equation.params["name"]) for equation in graph
                                                          if equation.primitive.name == "pallas_call"}
        meshed = jax.jit(functools.partial(run, RESIDUAL, MLP_HIDDEN))()
    for mine, theirs in zip(jax.tree.leaves(meshed), jax.tree.leaves(alone), strict=True):
        assert relative(mine, theirs) < 1e-6


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
