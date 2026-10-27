"""The fused norms and SiLU products against XLA and float64, each kernel called itself (`fused_seam`),
interpreted on a CPU: off a GPU the selectors take XLA's arithmetic, a kernel named per call too.

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
from jax.sharding import Mesh
from reference_error import (
    assert_as_exact,
    assert_as_exact_as_the_reference,
    distance,
    equations,
    float64_terms,
    widened,
    with_gradients,
)
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
    """`function`'s output and its operands' gradients under the cotangent `seed`, pulled back through
    the gate (`reference_error.with_gradients`)."""
    return with_gradients(function)(*operands, seed)


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
    return fused_seam(1e-5, output, offset=False, after_cast=False)(x, None, None)[1]


def fused_seam(epsilon: float, dtype, offset: bool, after_cast: bool, *, axes=None):
    """The fused pre-norm seam called itself (`norm_swiglu.stream_rms_norm`), interpreted on a CPU, with
    `stream_rms_normalized`'s XLA arithmetic its portable path: the program a GPU's selector runs."""
    def portable(x, branch, scale):
        return stream_rms_normalized(x, branch, scale, epsilon, dtype, offset, after_cast,
                                     fp32_statistics=True, implementation="xla")

    return lambda x, branch, scale: norm_swiglu.stream_rms_norm(x, branch, scale, epsilon, dtype, offset,
                                                               after_cast, portable, axes=axes)


def fused_gated(epsilon: float, dtype, activation: str):
    """The fused gated norm called itself (`norm_swiglu.gated_rms_norm`), as `fused_seam` is."""
    def portable(x, gate, scale):
        return gated_rms_normalized(x, gate, scale, epsilon, dtype, activation, implementation="xla")

    return lambda x, gate, scale: norm_swiglu.gated_rms_norm(x, gate, scale, epsilon, dtype, activation,
                                                            portable)


def fused_unweighted(epsilon: float):
    """`attention.unweighted_rmsnorm` fused (`norm_swiglu.rms_norm`'s `inverse_rounded`), as `fused_seam`
    is."""
    def portable(x, scale):
        return attention.unweighted_rmsnorm(x, epsilon, implementation="xla")

    return lambda x: norm_swiglu.rms_norm(x, None, epsilon, x.dtype, offset=False, after_cast=False,
                                          portable=portable, inverse_rounded=True)


def fused_swiglu(*, axes=None):
    """The fused SiLU product called itself (`norm_swiglu.swiglu`), as `fused_seam` is."""
    return lambda gate, up: norm_swiglu.swiglu(gate, up, gated_product("swiglu", implementation="xla"),
                                               axes=axes)


def fused_packed(*, axes=None):
    """The fused SiLU product of one packed projection called itself (`norm_swiglu.packed_swiglu`)."""
    return lambda gate_up: norm_swiglu.packed_swiglu(
        gate_up, packed_gated_product("swiglu", implementation="xla"), axes=axes)


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


SEEDS = {(2, 3, 31): 32, (3, 257, 17): 32, (5, 1000): 32, (2, 3, 4, 31): 64, (2, 64, 16, 128): 8,
         (2, 1, 16, 128): 32, (7, 1000): 32, (3, 4096): 32, (5, 37): 32, (24576, 1000): 8, (2, 31): 32,
         "composed": 32, "gates": 8}
"""Each shape's declared seeds for check 1 (`assert_as_exact`), sized by `seeds_for`: the fewest for which
XLA compared with itself resolves the factor in every output and gradient (on armada)."""


def assert_drawn_as_exact(fused, xla, exact, draw, leaves, label, seeds):
    """Check 1 over `draw`'s seeds: the kernel's outputs and gradients (`leaves` of `evaluated`) on par with
    XLA's, truth XLA's float64 run on the CPU."""
    def truth(operands, seed):
        with jax.default_device(jax.devices("cpu")[0]):
            return leaves(evaluated(exact, operands, seed))

    assert_as_exact(jax.jit(lambda operands, seed: leaves(evaluated(fused, operands, seed))),
                    jax.jit(lambda operands, seed: leaves(evaluated(xla, operands, seed))),
                    draw, label, seeds=seeds, truth_fn=truth)


def compare(functions, operands, seed, labels, *, coefficient=None, draw=None, seeds=None, judged=None):
    """Every output's dtype and accumulation on `operands`; their accuracy
    held by check 1 over `draw`'s `seeds` (`assert_drawn_as_exact`), or, for operands no draw
    redraws (a row of one value, a cotangent at a format's edge), by the one-draw rule on them. Check 1
    holds the `judged` labels, every label by default, and the one-draw rule the rest."""
    pallas = functions[1]
    reference, (truth, terms) = oracles(functions, operands, seed)
    actual = jax.jit(functools.partial(evaluated, pallas))(operands, seed)
    held = () if draw is None else (judged or labels)
    gated(labels, actual, reference, truth, terms, seed, coefficient,
          one_draw=[label for label in labels if label not in held])
    if draw is not None:
        def leaves(result):
            return {label: value for label, value in zip(labels, result, strict=True)
                    if value is not None and label in (judged or labels)}

        xla, _, exact = functions
        assert_drawn_as_exact(pallas, xla, exact, draw, leaves, " ".join(labels), seeds)


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


def gated(labels, actual, reference, truth, terms, seed, coefficient, one_draw=None):
    """Every output and gradient's dtype and the weight gradient's accumulation, and the `one_draw`
    labels (every label by default) within the one-draw rule."""
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
            if one_draw is None or label in one_draw:
                assert_as_exact_as_the_reference(mine, theirs, want, label, terms=summed)
    except AssertionError as error:
        report = rounding_report(labels, actual, reference, truth)
        print(report, flush=True)
        error.add_note(report)
        raise


def norm_functions(dtype, offset, after_cast):
    """XLA's norm into `dtype`, the fused one (`fused_seam`'s, adding no branch) and XLA's in float64."""
    def norm(output_dtype, x, scale):
        return rms_normalized(x, scale, 1e-5, output_dtype, offset, after_cast, fp32_statistics=True,
                              implementation="xla")

    seam = fused_seam(1e-5, dtype, offset, after_cast)
    return (functools.partial(norm, dtype), lambda x, scale: seam(x, None, scale)[1],
            functools.partial(norm, jnp.float64))


def drawn_key(index: int, draw: int):
    """The key of an operand's `index` in draw `draw` of a check-1 comparison, draw 0 the key itself."""
    return jax.random.key(index) if draw == 0 else jax.random.fold_in(jax.random.key(index), draw)


def norm_operands(shape, dtype, scale_dtype, draw: int = 0):
    x = jax.random.normal(drawn_key(23, draw), shape).astype(dtype)
    scale = None if scale_dtype is None else (
        0.7 + 0.2 * jax.random.normal(drawn_key(24, draw), (shape[-1],))).astype(scale_dtype)
    seed = jax.random.normal(drawn_key(25, draw), shape).astype(dtype)
    return (x, scale), seed


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("scale_dtype", [None, jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("offset", [False, True])
@pytest.mark.parametrize("after_cast", [False, True])
def test_rms_variants_and_every_gradient(dtype, scale_dtype, offset, after_cast):
    operands, seed = norm_operands((2, 3, 31), dtype, scale_dtype)
    compare(norm_functions(dtype, offset, after_cast), operands, seed, ("y", "x", "scale"),
            coefficient=scale_coefficient(operands[0], dtype, after_cast),
            draw=lambda draw: norm_operands((2, 3, 31), dtype, scale_dtype, draw), seeds=SEEDS[(2, 3, 31)])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("case", ["random-2x3x31", "random-3x257x17", "random-5x2x128", "aligned-1x128",
                                  "large-1x128", "small-1x128"])
def test_the_unweighted_norm_rounds_its_inverse_and_takes_every_gradient_fused(dtype, case):
    """`attention.unweighted_rmsnorm` fused (`norm_swiglu.rms_norm`'s `inverse_rounded`): the inverse RMS
    reduced in fp32 and rounded to the input's dtype before it multiplies, as DeepSeek-V4's and GLM's
    unweighted norms round it, and its reverse rounded where XLA's is; its output and input gradient
    within the float64 gate of XLA's, random rows, a row of 1.5 under a cotangent of ones, whose two
    gradient contributions cancel, and rows of 0.5 under cotangents of 1.5 * 2**119, finite through
    XLA's transposes, and of a subnormal (2**-140 in fp32, 2**-130 in bf16), which they do not halve."""
    def unweighted(x):
        return attention.unweighted_rmsnorm(x, 1e-6, implementation="xla")

    kind, shape = case.split("-")
    shape = tuple(int(size) for size in shape.split("x"))
    if kind == "aligned":
        x, seed = jnp.full(shape, 1.5, dtype), jnp.ones(shape, dtype)
    elif kind in ("large", "small"):
        cotangent = 1.5 * 2.0 ** 119 if kind == "large" else 2.0 ** (-140 if dtype == jnp.float32 else -130)
        x, seed = jnp.full(shape, 0.5, dtype), jnp.full(shape, cotangent, dtype)
    else:
        x = jax.random.normal(jax.random.key(26), shape).astype(dtype)
        seed = jax.random.normal(jax.random.key(27), shape).astype(dtype)
    compare((unweighted, fused_unweighted(1e-6), unweighted), (x,), seed, ("y", "x"))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("shape", [(7, 1000), (3, 4096), (3, 257, 17)])
def test_rms_ragged_widths_and_many_rows(dtype, shape):
    operands, seed = norm_operands(shape, dtype, jnp.float32)
    compare(norm_functions(dtype, offset=True, after_cast=True), operands, seed, ("y", "x", "scale"),
            coefficient=scale_coefficient(operands[0], dtype, after_cast=True),
            draw=lambda draw: norm_operands(shape, dtype, jnp.float32, draw), seeds=SEEDS[shape])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_rms_weight_gradient_accumulates_a_full_step_of_rows(dtype):
    """Repeated tokens hold the row arithmetic fixed while the column sum grows to Qwen3.5-0.8B's size,
    isolating partial-sum error from bf16 midpoint flips: the weight gradient holds check 1 over the
    step's draws; the output and input gradient, the full step's launch, hold the one-draw rule."""
    (x, scale), seed = full_step(dtype)
    compare(norm_functions(dtype, offset=True, after_cast=True), (x, scale), seed, ("y", "x", "scale"),
            coefficient=scale_coefficient(x, dtype, after_cast=True),
            draw=lambda draw: full_step(dtype, draw), seeds=SEEDS[(24576, 1000)], judged=("scale",))


def test_a_weight_gradient_past_fp32s_range_is_an_infinity_as_xlas():
    """A weight gradient whose column sum overflows fp32 is +inf, as XLA's sum and the shared
    finalization give it (`precision.compensated_total`), not the NaN the compensated pair's residual
    past the infinity (inf - inf) would make: four rows of ones under cotangents of 1e38."""
    xla, fused, _ = norm_functions(jnp.float32, offset=False, after_cast=False)
    x, scale = jnp.ones((4, 128), jnp.float32), jnp.ones((128,), jnp.float32)
    seed = jnp.full(x.shape, 1e38, jnp.float32)

    def weight_gradient(norm):
        return jax.vjp(lambda scale: norm(x, scale), scale)[1](seed)[0]

    expected = weight_gradient(xla)
    assert np.isposinf(np.asarray(expected)).all()
    np.testing.assert_array_equal(np.asarray(weight_gradient(fused)), np.asarray(expected))


def full_step(dtype, draw: int = 0):
    """24,576 rows of one repeated 1000-wide token, so the weight gradient's sums reach a step's
    length (`test_rms_weight_gradient...`)."""
    (row, scale), seed = norm_operands((1, 1000), dtype, jnp.float32, draw)
    x = jnp.broadcast_to(row, (24576, 1000))
    return (x, scale), jnp.broadcast_to(seed, x.shape)


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", [norm_swiglu.RmsForward, norm_swiglu.RmsBackward])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("operands", ["full step", "wide rows", "narrow rows"])
def test_every_config_of_the_norm_kernels_holds_the_float64_gate(kernel, dtype, operands):
    """Each config the tuner may pick, at a step's rows, at rows wider than
    a tile holds two of, and at rows narrow enough for its tallest tiles:
    its variants are only configs proven here (`KernelOp.variants`)."""
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
    xla, pallas = gated_product("swiglu", implementation="xla"), fused_swiglu()
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
    def draw(draw):
        operands, seed = norm_operands((5, 37), input_dtype, jnp.float32, draw)
        return operands, seed.astype(output_dtype)

    operands, seed = draw(0)
    compare(norm_functions(output_dtype, offset=False, after_cast=after_cast), operands, seed,
            ("y", "x", "scale"), coefficient=scale_coefficient(operands[0], output_dtype, after_cast),
            draw=draw, seeds=SEEDS[(5, 37)])


def stream_seam(dtype, joined: bool, *, fused: bool = False):
    """The pre-norm seam `(stream, normed)` over `x`, plus `branch` where `joined`: XLA's, or `fused` the
    kernel's (`fused_seam`)."""
    seam = fused_seam(1e-5, dtype, offset=True, after_cast=True) if fused else (
        lambda x, branch, scale: stream_rms_normalized(x, branch, scale, 1e-5, dtype, scale_offset=True,
                                                       scale_after_cast=True, fp32_statistics=True,
                                                       implementation="xla"))
    return lambda x, branch, scale: seam(x, branch if joined else None, scale)


@pytest.mark.parametrize("joined", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 31), (3, 257, 17), (5, 1000)])
def test_stream_seam_takes_the_add_and_the_stream_cotangent(joined, dtype, shape):
    """The kernel's sum is XLA's to the bit, and the weight's gradient accumulates as it should, on the
    first draw; the norm and every gradient, the stream's cotangent folded into dx, hold check 1 over
    the shape's seeds."""
    def draw(seed):
        keys = jax.random.split(jax.random.fold_in(jax.random.key(23), seed), 5)
        x, dy, branch, ds = (jax.random.normal(keys[index], shape).astype(dtype) for index in (0, 2, 3, 4))
        return (x, branch, 0.7 + 0.2 * jax.random.normal(keys[1], shape[-1:])), (ds, dy)

    def leaves(result):
        (_, normed), dx, dbranch, dscale = result
        return {"y": normed, "x": dx, "scale": dscale, **({"branch": dbranch} if joined else {})}

    operands, seed = draw(0)
    fused = stream_seam(dtype, joined, fused=True)
    reference = jax.jit(functools.partial(evaluated, stream_seam(dtype, joined)))(operands, seed)
    (stream, _), *gradients = jax.jit(functools.partial(evaluated, fused))(operands, seed)
    (reference_stream, _), *reference_gradients = reference
    np.testing.assert_array_equal(stream, reference_stream)
    if not joined:
        assert not np.any(np.asarray(gradients[1])) and not np.any(np.asarray(reference_gradients[1]))
    assert [gradient.dtype for gradient in gradients] == [gradient.dtype for gradient in reference_gradients]
    assert_scale_accumulation(gradients[2], reference_gradients[2],
                              scale_coefficient(stream, dtype, after_cast=True), seed[1])
    assert_drawn_as_exact(fused, stream_seam(dtype, joined), stream_seam(jnp.float64, joined), draw, leaves,
                          "stream seam", SEEDS[shape])


def assert_gated_norm_holds(draw, activation, output, name, seeds):
    """The fused gated norm keeps XLA's output and gradient dtypes on `draw(0)`, and holds check 1 for its
    output and its input, gate and weight gradients, in `output`, over `seeds` draws."""
    def gated(dtype):
        return lambda x, gate, scale: gated_rms_normalized(x, gate, scale, 1e-6, dtype, activation,
                                                           implementation="xla")

    def leaves(result):
        return dict(zip(("y", "x", "gate", "scale"), result, strict=True))

    operands, dy = draw(0)
    fused = fused_gated(1e-6, output, activation)
    actual = jax.jit(functools.partial(evaluated, fused))(operands, dy)
    reference = jax.jit(functools.partial(evaluated, gated(output)))(operands, dy)
    assert [value.dtype for value in actual] == [value.dtype for value in reference]
    assert_drawn_as_exact(fused, gated(output), gated(jnp.float64), draw, leaves, name, seeds)


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 4, 31), (2, 64, 16, 128)])
@pytest.mark.parametrize("gate_dtype", ["same", jnp.bfloat16])
def test_gated_norm_and_every_gradient_hold_the_float64_gate(activation, dtype, shape, gate_dtype):
    """Qwen3-Next's gated norm: the kernel's output and its input, gate and weight gradients on par
    with XLA's over the shape's seeds (check 1), its gate in the input's dtype or in bf16 beside an
    fp32 input, as a decode step's recurrence hands it over."""
    def draw(seed):
        keys = jax.random.split(jax.random.fold_in(jax.random.key(51), seed), 4)
        x, gate, dy = (jax.random.normal(key, shape).astype(dtype) for key in keys[:3])
        gate = gate if gate_dtype == "same" else gate.astype(gate_dtype)
        return (x, gate, 0.7 + 0.2 * jax.random.normal(keys[3], shape[-1:])), dy

    assert_gated_norm_holds(draw, activation, dtype, "gated norm", SEEDS[shape])


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("output", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 4, 31), (2, 1, 16, 128)])
def test_a_decode_steps_gated_norm_of_an_fp32_input_by_a_bf16_gate_holds_the_float64_gate(activation, output,
                                                                                           shape):
    """A decode step's gated norm (Qwen3-Next's, Kimi's): the recurrence's fp32 output gated by its bf16
    projection, the gate read in fp32 as XLA widens it, cast once to `output`; its output and its input,
    gate and weight gradients, each in its operand's dtype, on par with XLA's over the shape's seeds."""
    def draw(seed):
        keys = jax.random.split(jax.random.fold_in(jax.random.key(61), seed), 4)
        x, gate, dy = (jax.random.normal(key, shape) for key in keys[:3])
        scale = 0.7 + 0.2 * jax.random.normal(keys[3], shape[-1:])
        return (x, gate.astype(jnp.bfloat16), scale), dy.astype(output)

    assert_gated_norm_holds(draw, activation, output, "decode gated norm", SEEDS[shape])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 1000), (3, 4096), (3, 257, 17)])
def test_swiglu_forward_and_both_gradients(dtype, shape):
    operands = tuple(jax.random.normal(jax.random.key(index), shape).astype(dtype) for index in (31, 32))
    seed = jax.random.normal(jax.random.key(33), shape).astype(dtype)
    xla = gated_product("swiglu", implementation="xla")
    compare((xla, fused_swiglu(), xla), operands, seed, ("y", "gate", "up"))


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 1000), (3, 4096), (3, 257, 17)])
def test_a_packed_projections_swiglu_is_the_apart_one_read_in_place(fused, dtype, shape):
    """One packed projection's `[..., 2, H]` output, gate then up: the
    kernel's packed launch and XLA's product read the halves where they lie
    and give the apart product's output and both gradients bit for bit, the
    gradient in the packed layout, so the apart product's float64 gate
    holds for them."""
    gate_up = jax.random.normal(jax.random.key(34), (*shape[:-1], 2, shape[-1])).astype(dtype)
    seed = jax.random.normal(jax.random.key(35), shape).astype(dtype)
    packed = functools.partial(evaluated, fused_packed() if fused else packed_gated_product(
        "swiglu", implementation="xla"))
    output, dpacked = jax.jit(packed)((gate_up,), seed)
    product = fused_swiglu() if fused else gated_product("swiglu", implementation="xla")
    apart = jax.jit(functools.partial(evaluated, product))((gate_up[..., 0, :], gate_up[..., 1, :]), seed)
    for mine, theirs in zip((output, dpacked[..., 0, :], dpacked[..., 1, :]), apart, strict=True):
        np.testing.assert_array_equal(mine, theirs)


@pytest.mark.parametrize("offset,after_cast,with_scale", [(False, True, True), (True, False, True),
                                                         (False, True, False)])
def test_rms_module_uses_the_same_fused_vjp(offset, after_cast, with_scale):
    """The module naming the fused norm: its values and gradients are XLA's within 1e-5: the kernel's on a
    GPU, and on a CPU, where the selector takes XLA's arithmetic, XLA's own."""
    x = jax.random.normal(jax.random.key(4), (2, 5, 31))
    module = RMSNorm(scale_offset=offset, scale_after_cast=after_cast, with_scale=with_scale,
                     implementation="pallas")
    params = module.init(jax.random.key(5), x)
    portable = module.clone(implementation="xla")

    def differentiated(module):
        return jax.value_and_grad(lambda p, x: jnp.sum(module.apply(p, x)), argnums=(0, 1))

    mine = jax.jit(differentiated(module))(params, x)
    theirs = jax.jit(differentiated(portable))(params, x)
    for actual, expected in zip(jax.tree.leaves(mine), jax.tree.leaves(theirs), strict=True):
        assert relative(actual, expected) < 1e-5


def test_a_swiglu_named_onto_the_kernel_takes_broadcast_mixed_and_wide_operands():
    """On an A100, a SiLU product named onto the kernel runs where its operands broadcast, mix dtypes or
    are float64, into their broadcast shape and promoted dtype."""
    from dew.nn.kernels import generation

    with jax.enable_x64(new_val=True), generation.assumed("sm80"):
        for gate, up in ((jnp.ones((2, 17)), jnp.ones((17,))),
                         (jnp.ones((2, 17)), jnp.ones((2, 17), jnp.bfloat16)),
                         (jnp.ones((2, 17), jnp.float64), jnp.ones((2, 17), jnp.float64))):
            product = gated_product("swiglu", implementation="pallas")(gate, up)
            assert (product.shape, product.dtype) == ((2, 17), jnp.promote_types(gate.dtype, up.dtype))


def test_a_weak_scalar_gate_over_a_bf16_up_promotes_as_its_product_does():
    """A weak fp32 scalar gate over a strongly typed bf16 up promotes to bf16 under strict promotion, as
    the operands' own product does, so on an A100 the product runs into bf16 where promoting the two
    dtypes alone would raise."""
    from dew.nn.kernels import generation

    gate, up = jnp.asarray(1.0), jnp.ones((2, 64), jnp.bfloat16)
    with jax.numpy_dtype_promotion("strict"), generation.assumed("sm80"):
        product = jax.eval_shape(gated_product("swiglu"), gate, up)
    assert (product.shape, product.dtype) == ((2, 64), jnp.bfloat16)


@pytest.mark.gpu
@pytest.mark.parametrize("implementation", ["auto", "pallas"])
def test_a_gpu_hosts_cpu_runs_the_seams_its_placement_chooses(without_deterministic_ops, implementation):
    """On a GPU host, the norm, the SiLU product and attention placed on the CPU, eagerly under the
    default device and in a JIT on a CPU mesh, as a host-layout run places its state, run there: no
    GPU kernel runs off the GPU, so 'auto' and a kernel named per call alike take XLA's arithmetic."""
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
    """A GeGLU named onto the kernel, which takes SiLU products alone, computes its own gate: on par with
    XLA's product over the seeds (check 1), truth that product in float64."""
    product = gated_product(name, implementation="xla")
    assert_as_exact(jax.jit(gated_product(name, implementation="pallas")), jax.jit(product),
                    functools.partial(fused_operands, "swiglu", jnp.float32), name, seeds=SEEDS["gates"],
                    truth_fn=product)


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
        seam = evaluated(fused_seam(1e-5, x.dtype, offset=True, after_cast=True, axes=axes),
                         (x, branch, scale), (ds, dy))
        product = evaluated(fused_swiglu(axes=hidden), (gate, up), dout)
        packed = evaluated(fused_packed(axes=hidden), (jnp.stack([gate, up], -2),), dout)
        return seam, product, packed

    alone = jax.jit(functools.partial(run, RESIDUAL, MLP_HIDDEN))()
    with jax.set_mesh(MeshSpec(**{split: 2}).build(jax.devices()[:2])):
        graph = traced(functools.partial(run, RESIDUAL, MLP_HIDDEN))
        assert sum(equation.primitive.name == "shard_map" for equation in graph) >= 2
        meshed = jax.jit(functools.partial(run, RESIDUAL, MLP_HIDDEN))()
    for mine, theirs in zip(jax.tree.leaves(meshed), jax.tree.leaves(alone), strict=True):
        assert relative(mine, theirs) < 1e-6


def fused_function(name, fused: bool):
    """A fused kernel's function by `name`, the kernel's where `fused` (called itself, `fused_seam`), else
    its public function's XLA arithmetic, over (x, other, scale) or (gate, up), its operands
    `fused_operands`: each of the norms, the inverse-rounded one and a gated one without a weight
    ('weightless') among them, and both SiLU products."""
    if name == "swiglu":
        return fused_swiglu() if fused else gated_product("swiglu", implementation="xla")
    if name == "packed":
        product = fused_packed() if fused else packed_gated_product("swiglu", implementation="xla")
        return lambda gate, up: product(jnp.stack([gate, up], -2))
    if name == "unweighted":
        return lambda x, other, scale: (fused_unweighted(1e-6)(x) if fused else
                                        attention.unweighted_rmsnorm(x, 1e-6, implementation="xla"))
    if name in ("rms", "stream"):
        # The rms norm offsets its weight and rounds before it; the stream does neither.
        rounded = name == "rms"

        def seam(x, other, scale):
            branch = None if rounded else other
            if fused:
                return fused_seam(1e-5, x.dtype, offset=rounded, after_cast=rounded)(x, branch, scale)
            return stream_rms_normalized(x, branch, scale, 1e-5, x.dtype, scale_offset=rounded,
                                         scale_after_cast=rounded, fp32_statistics=True, implementation="xla")

        return (lambda x, other, scale: seam(x, other, scale)[1]) if rounded else seam
    return lambda x, other, scale: (fused_gated(1e-6, x.dtype, "silu")(x, other, scale) if fused else
                                    gated_rms_normalized(x, other, scale, 1e-6, x.dtype, "silu",
                                                         implementation="xla"))


def fused_operands(name, dtype, draw: int = 0):
    """`fused_function(name)`'s operands at `dtype`, a bf16 weight where the norm takes one."""
    first, second = (jax.random.normal(drawn_key(index, draw), (4, 96)).astype(dtype) for index in (76, 77))
    if name in ("swiglu", "packed"):
        return first, second
    if name == "weightless":
        return first, second, None
    return first, second, (0.7 + 0.2 * jax.random.normal(drawn_key(79, draw), (96,))).astype(jnp.bfloat16)


@pytest.mark.parametrize("composition", ["jvp_grad", "grad_jvp", "jvp_staged", "vmap_grad", "grad_vmap"])
@pytest.mark.parametrize("name, dtype", [("swiglu", jnp.bfloat16), ("packed", jnp.float32),
                                         ("rms", jnp.bfloat16), ("stream", jnp.float32),
                                         ("unweighted", jnp.bfloat16), ("gated", jnp.float32),
                                         ("weightless", jnp.bfloat16)])
def test_a_fused_kernel_differentiates_at_every_order_through_its_portable_arithmetic(name, dtype,
                                                                                     composition):
    """Each fused norm and SiLU product (interpreted here) differentiated at a second order, forward
    over reverse and reverse over forward, staged by a jit, a gradient mapped over rows and the gradient
    of a map over rows (the linear map's batch, `generation.linear_map`), by the first operand alone and
    under a score that squares the output, so the cotangent varies: each is the portable arithmetic's
    own composition, within the float64 gate of it."""
    def draw(draw):
        # The score's cotangent, one, held in the draw as the gate takes it.
        operands = fused_operands(name, dtype, draw)
        return (operands[0], jax.random.normal(drawn_key(78, draw), operands[0].shape).astype(dtype),
                jnp.float32(1), *operands[1:])

    def composed(kernel: bool):
        fused = fused_function(name, kernel)

        def score(first, *rest):
            out = fused(first, *rest)
            # At least float32, so the float64 truth scores in float64.
            return sum(jnp.square(leaf.astype(jnp.promote_types(leaf.dtype, jnp.float32))).sum()
                       for leaf in jax.tree.leaves(out))

        def gradient(function, one):
            # The first operand's gradient, through the gate's pullback of the score's one at its dtype.
            def run(first, *rest):
                score_dtype = jax.eval_shape(function, first, *rest).dtype
                return with_gradients(function)(first, *rest, jnp.asarray(one, score_dtype))[1]

            return run

        def run(first, tangent, one, *rest):
            if composition == "jvp_grad":
                return jax.jvp(lambda first: gradient(score, one)(first, *rest), (first,), (tangent,))[1]
            if composition == "grad_jvp":
                def along(first):
                    return jax.jvp(lambda first: score(first, *rest), (first,), (tangent,))[1]

                return gradient(along, one)(first)
            if composition == "jvp_staged":
                return jax.jvp(jax.jit(lambda first: fused(first, *rest)), (first,), (tangent,))[1]
            # The rows map, a norm's weight is every row's.
            rows = (0, 0, None)[:1 + len(rest)]
            if composition == "grad_vmap":
                return gradient(lambda first, *rest: jax.vmap(score, in_axes=rows)(first, *rest).sum(), one)(
                    first, *rest)
            return jax.vmap(gradient(score, one), in_axes=rows)(first, *rest)

        return run

    # Exact by construction (a stream's tangent is its addends'), or on par over the draws: check 1,
    # which holds Dew exact where the reference is on every draw.
    assert_as_exact(jax.jit(composed(kernel=True)), jax.jit(composed(kernel=False)), draw,
                    f"{name} {composition}", seeds=SEEDS["composed"], truth_fn=composed(kernel=False))


def test_a_fused_norms_jvp_is_its_kernels_value_and_the_portable_tangent():
    """`jax.jvp` through the fused norm, with no forward-mode context, staged by a jit or not: its
    value is the kernel's own output, and its tangent within the float64 gate of the XLA norm's
    (`generation.linear_tangent`)."""
    def draw(draw):
        return tuple(jax.random.normal(drawn_key(index, draw), (2, 31)) for index in (75, 74))

    def tangent_of(norm):
        return lambda x, direction: jax.jvp(lambda x: norm(x, None), (x,), (direction,))[1]

    x, direction = draw(0)
    xla, pallas, exact = norm_functions(jnp.float32, offset=True, after_cast=True)
    fused = jax.jit(lambda x: pallas(x, None))
    # The eager JVP, traced op by op, and the staged one, each as it is called.
    for norm, staged in ((lambda x: pallas(x, None), False), (fused, True)):
        value, _ = jax.jvp(norm, (x,), (direction,))
        np.testing.assert_array_equal(value, fused(x))
        tangent = tangent_of(lambda x, _, norm=norm: norm(x))
        assert_as_exact(jax.jit(tangent) if staged else tangent, jax.jit(tangent_of(xla)),
                        draw, "jvp", seeds=SEEDS[(2, 31)], truth_fn=tangent_of(exact))


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

        def function(fused):
            if fused:
                seam = fused_seam(1e-6, jnp.bfloat16, offset=False, after_cast=True)
                return lambda x, scale: seam(x, None, scale)[1]
            return lambda x, scale: rms_normalized(
                x, scale, 1e-6, jnp.bfloat16, scale_offset=False, scale_after_cast=True,
                fp32_statistics=True, implementation="xla")
    else:
        operands = (x, jnp.asarray(second, jnp.bfloat16))

        def function(fused):
            return fused_swiglu() if fused else gated_product("swiglu", implementation="xla")

    reference = jax.jit(functools.partial(evaluated, function(fused=False)))(operands, seed)
    actual = jax.jit(functools.partial(evaluated, function(fused=True)))(operands, seed)
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
