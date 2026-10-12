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
from reference_error import assert_as_exact_as_the_reference, distance

from dew.nn import kernels
from dew.nn.attention import (
    RMSNorm,
    forward_mode_attention,
    gated_rms_normalized,
    rms_normalized,
    stream_rms_normalized,
)
from dew.nn.moe import gated_product

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
    assert "rms_norm_forward" in str(jax.make_jaxpr(fused)(*operands))
    assert "rms_norm_backward" in str(jax.make_jaxpr(functools.partial(evaluated, fused))(operands, seed))
    reference = jax.jit(functools.partial(evaluated, stream_seam("xla", dtype, joined)))(operands, seed)
    actual = jax.jit(functools.partial(evaluated, fused))(operands, seed)
    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(new_val=True), jax.default_device(cpu):
        wide = jax.tree.map(lambda x: jnp.asarray(np.asarray(x, np.float64)), operands)
        truth = evaluated(stream_seam("xla", jnp.float64, joined), wide,
                          jax.tree.map(lambda x: jnp.asarray(np.asarray(x, np.float64)), seed))
    (stream, normed), *gradients = actual
    (reference_stream, reference_normed), *reference_gradients = reference
    np.testing.assert_array_equal(stream, reference_stream)
    labels = ("y", "x", "branch", "scale")
    for label, mine, theirs, want in zip(labels, (normed, *gradients),
                                         (reference_normed, *reference_gradients),
                                         (truth[0][1], *truth[1:]), strict=True):
        if label == "branch" and not joined:
            assert not np.any(np.asarray(mine)) and not np.any(np.asarray(theirs))
            continue
        assert mine.dtype == theirs.dtype, label
        if label == "scale":
            assert_scale_accumulation(mine, theirs, scale_coefficient(stream, dtype, after_cast=True), dy)
        assert_as_exact_as_the_reference(mine, theirs, want, label)


@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 4, 31), (2, 64, 16, 128)])
def test_gated_norm_and_every_gradient_hold_the_float64_gate(activation, dtype, shape):
    """Qwen3-Next's gated norm: the kernel's output and its input, gate and
    weight gradients each within the gate of XLA's float64 distance."""
    x, gate, dy = (jax.random.normal(jax.random.key(index), shape).astype(dtype) for index in (51, 52, 53))
    scale = 0.7 + 0.2 * jax.random.normal(jax.random.key(54), shape[-1:])
    operands = (x, gate, scale)

    def gated(implementation, output_dtype):
        return lambda x, gate, scale: gated_rms_normalized(x, gate, scale, 1e-6, output_dtype, activation,
                                                           implementation=implementation)

    fused = gated("pallas", dtype)
    assert "rms_norm_forward" in str(jax.make_jaxpr(fused)(*operands))
    assert "rms_norm_backward" in str(jax.make_jaxpr(functools.partial(evaluated, fused))(operands, dy))
    reference = jax.jit(functools.partial(evaluated, gated("xla", dtype)))(operands, dy)
    actual = jax.jit(functools.partial(evaluated, fused))(operands, dy)
    cpu = jax.devices("cpu")[0]
    with jax.enable_x64(new_val=True), jax.default_device(cpu):
        wide = jax.tree.map(lambda x: jnp.asarray(np.asarray(x, np.float64)), operands)
        truth = evaluated(gated("xla", jnp.float64), wide, jnp.asarray(np.asarray(dy, np.float64)))
    for label, mine, theirs, want in zip(("y", "x", "gate", "scale"), actual, reference, truth, strict=True):
        assert mine.dtype == theirs.dtype, label
        assert_as_exact_as_the_reference(mine, theirs, want, label)


def small_block():
    from dew.nn.backbones.decoder_block import BlockWiring, DecoderBlock, GatedMLP
    from dew.nn.mixers import AttentionMixer, MixerContext

    features = 32
    mixer = AttentionMixer().build(MixerContext(
        emb_features=features, num_heads=4, num_kv_heads=2, head_dim=8, max_seq_len=16))
    block = DecoderBlock(mixer=mixer, feedforward=functools.partial(
        GatedMLP, hidden_features=64, out_features=features), emb_features=features, scale_offset=True,
        wiring=BlockWiring())
    x = jax.random.normal(jax.random.key(41), (2, 16, features))
    params = block.init(jax.random.key(42), x)
    seed = jax.random.normal(jax.random.key(43), x.shape)
    step = jax.jit(jax.value_and_grad(lambda params, x: jnp.sum(block.apply(params, x) * seed),
                                      argnums=(0, 1)))
    return params, x, step


def test_decoder_block_seams_keep_the_block_to_the_bit(monkeypatch):
    """XLA's seams are the old block's add, read and norm, bit for bit."""
    from dew.nn.backbones import decoder_block

    params, x, step = small_block()
    seams = step(params, x)

    class Unseamed:
        """No norm owns a seam: the block writes, reads and norms as before."""

    monkeypatch.setattr(decoder_block, "SeamNorm", Unseamed)
    *_, old = small_block()
    for mine, theirs in zip(jax.tree.leaves(seams), jax.tree.leaves(old(params, x)), strict=True):
        np.testing.assert_array_equal(mine, theirs)


def test_decoder_block_seams_run_the_fused_norm(monkeypatch):
    """Both of a pre-norm block's norms run the fused seam: the input norm
    passes the stream through, the post-attention norm takes the attention
    branch's add, and the reverse pass folds each stream cotangent in."""
    from dew.nn.kernels import norm_swiglu

    monkeypatch.setattr(norm_swiglu, "rms_kernel", lambda *args, **kwargs: "xla")
    params, x, step = small_block()
    reference = step(params, x)
    monkeypatch.setattr(norm_swiglu, "rms_kernel", lambda *args, **kwargs: "pallas")
    fused, joins = norm_swiglu.stream_rms_norm, []

    def recorded(x, branch, *args, **kwargs):
        joins.append(branch is not None)
        return fused(x, branch, *args, **kwargs)

    monkeypatch.setattr(norm_swiglu, "stream_rms_norm", recorded)
    params, x, step = small_block()
    joins.clear()
    jax.make_jaxpr(step)(params, x)
    # The input norm and the attention's query and key norms read; the MLP's norm joins.
    assert joins.count(True) == 1 and joins.count(False) == 3, joins
    for mine, theirs in zip(jax.tree.leaves(step(params, x)), jax.tree.leaves(reference), strict=True):
        assert relative(mine, theirs) < 1e-5


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 1000), (3, 4096), (3, 257, 17)])
def test_swiglu_forward_and_both_gradients(dtype, shape):
    operands = tuple(jax.random.normal(jax.random.key(index), shape).astype(dtype) for index in (31, 32))
    seed = jax.random.normal(jax.random.key(33), shape).astype(dtype)
    xla, pallas = (gated_product("swiglu", implementation=name) for name in ("xla", "pallas"))
    compare((xla, pallas, xla), operands, seed, ("y", "gate", "up"))


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


def measured_on_this_host(monkeypatch, choice: str) -> None:
    """This host's generation measures the Pallas kernel fastest for `choice`, so a call the
    kernel turns down is logged (`ran_kernel`), as on a GPU whose target row names it."""
    host = kernels.device_generation()
    monkeypatch.setitem(kernels.TARGETS, host, kernels.GpuTarget(host, host, kernels={choice: "pallas"}))


def test_packed_halves_keep_xla_product_and_say_why(monkeypatch, caplog):
    """A packed gate_up projection's halves stay XLA's product, which reads
    them in place; the kernel would copy each half out first."""
    from dew.nn.backbones.decoder_block import GatedMLP

    measured_on_this_host(monkeypatch, "swiglu")
    monkeypatch.setattr(kernels.generation, "_logged", set())
    x = jax.random.normal(jax.random.key(81), (2, 5, 8))
    for packed in (False, True):
        mlp = GatedMLP(hidden_features=16, out_features=8, packed=packed)
        params = mlp.init(jax.random.key(82), x)
        graph = str(jax.make_jaxpr(mlp.apply)(params, x))
        assert ("swiglu_forward" in graph) == (not packed and jax.default_backend() == "gpu"), packed
    with monkeypatch.context() as patched:
        patched.setattr(kernels.generation, "triton_runs", lambda: True)
        assert kernels.norm_swiglu.swiglu_kernel("auto", x, x, packed=True) == "xla"
    logged = [record.getMessage() for record in caplog.records if record.name == kernels.generation.__name__]
    assert any("packed projection" in message for message in logged), logged


def test_swiglu_refuses_broadcast_mixed_and_wide_dtypes():
    from dew.nn.kernels import norm_swiglu

    with jax.enable_x64(new_val=True):
        for gate, up in ((jnp.ones((2, 17)), jnp.ones((17,))),
                         (jnp.ones((2, 17)), jnp.ones((2, 17), jnp.bfloat16)),
                         (jnp.ones((2, 17), jnp.float64), jnp.ones((2, 17), jnp.float64))):
            assert norm_swiglu.swiglu_kernel("pallas", gate, up) == "xla"
            np.testing.assert_array_equal(gated_product("swiglu", implementation="pallas")(gate, up),
                                          gated_product("swiglu", implementation="xla")(gate, up))


def test_auto_takes_the_generation_row_and_forward_mode_is_portable():
    """Auto takes the device's measured kernel on a GPU Triton runs, XLA on a CPU."""
    from dew.nn.kernels import norm_swiglu

    def expected(choice):
        on_gpu = kernels.generation.triton_runs()
        return kernels.generation.measured_kernel(choice, "xla") if on_gpu else "xla"

    x = jnp.ones((2, 17), jnp.float32)
    assert norm_swiglu.rms_kernel("auto", x, None, x.dtype, fp32_statistics=True) == expected("rms_norm")
    assert norm_swiglu.swiglu_kernel("auto", x, x) == expected("swiglu")
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


@pytest.mark.mesh(devices=2)
def test_split_mesh_operands_refuse_both_local_kernels_without_axes():
    from dew.nn.kernels import norm_swiglu
    from dew.training import MeshSpec

    x = jnp.ones((2, 31), jnp.float32)
    with jax.set_mesh(MeshSpec(fsdp=2).build(jax.devices()[:2])):
        assert norm_swiglu.rms_kernel("pallas", x, None, x.dtype, fp32_statistics=True) == "xla"
        assert norm_swiglu.swiglu_kernel("pallas", x, x) == "xla"
    assert norm_swiglu.rms_kernel("pallas", x, None, x.dtype, fp32_statistics=True) == "pallas"
    assert norm_swiglu.swiglu_kernel("pallas", x, x) == "pallas"


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("split", ["fsdp", "tensor"])
def test_seam_and_product_run_per_shard_as_on_one_device(split):
    """Named logical axes place both kernels on a mesh: rows split, the norm's
    features whole, and the weight's gradient summed over the shards."""
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
        return seam, product

    alone = jax.jit(functools.partial(run, RESIDUAL, MLP_HIDDEN))()
    with jax.set_mesh(MeshSpec(**{split: 2}).build(jax.devices()[:2])):
        graph = str(jax.make_jaxpr(functools.partial(run, RESIDUAL, MLP_HIDDEN))())
        assert graph.count("shard_map") >= 2 and "rms_norm_forward" in graph and "swiglu_forward" in graph
        meshed = jax.jit(functools.partial(run, RESIDUAL, MLP_HIDDEN))()
    for mine, theirs in zip(jax.tree.leaves(meshed), jax.tree.leaves(alone), strict=True):
        assert relative(mine, theirs) < 1e-6


@pytest.mark.mesh(devices=2)
def test_a_narrow_weight_gradient_sums_its_shards_before_it_rounds():
    """Rows split over two shards: each shard's bf16-rounded sum would lose
    the 1/256 that the whole column's fp32 sum keeps (1 + 1/256 rounds to 1)."""
    from dew.nn.sharding import RESIDUAL
    from dew.training import MeshSpec

    x = jnp.ones((4, 1, 1), jnp.float32)
    dy = jnp.asarray([1.0, 1 / 256, -1.0, 0.0], jnp.float32).reshape(x.shape)
    scale = jnp.ones((1,), jnp.bfloat16)

    def dscale(x, scale):
        return jax.vjp(lambda scale: rms_normalized(
            x, scale, 0.0, x.dtype, scale_offset=False, scale_after_cast=False, fp32_statistics=True,
            implementation="pallas", axes=RESIDUAL), scale)[1](dy)[0]

    with jax.set_mesh(MeshSpec(fsdp=2).build(jax.devices()[:2])):
        assert "shard_map" in str(jax.make_jaxpr(dscale)(x, scale))
        meshed = jax.jit(dscale)(x, scale)
    assert meshed.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(meshed, np.float32), [1 / 256])


@pytest.mark.mesh(devices=2)
def test_any_varying_operand_refuses_both_kernels_in_a_checked_shard_map():
    """A Pallas call states no varying type for its result, so a stream,
    weight, branch, gate or up that varies inside a checking shard_map keeps
    XLA, even where the first operand is replicated."""
    from jax.sharding import PartitionSpec

    from dew.nn.kernels import norm_swiglu
    from dew.training import MeshSpec

    mesh = MeshSpec(fsdp=2).build(jax.devices()[:2])
    replicated, rows = jnp.ones((2, 8), jnp.float32), jnp.ones((4, 8), jnp.float32)
    chosen = {}

    def local(replicated, varying):
        chosen["branch"] = norm_swiglu.rms_kernel("pallas", replicated, None, jnp.float32,
                                                  fp32_statistics=True, branch=varying)
        chosen["gate"] = norm_swiglu.gated_rms_kernel("pallas", replicated, varying, None, jnp.float32,
                                                      "silu")
        chosen["up"] = norm_swiglu.swiglu_kernel("pallas", replicated, varying)
        return varying

    jax.make_jaxpr(jax.shard_map(local, mesh=mesh, in_specs=(PartitionSpec(), PartitionSpec("fsdp")),
                                 out_specs=PartitionSpec("fsdp")))(replicated, rows)
    assert chosen == {"branch": "xla", "gate": "xla", "up": "xla"}, chosen


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
