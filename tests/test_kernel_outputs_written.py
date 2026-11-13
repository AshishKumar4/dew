"""Every output a launch declares is written on every path (the unwritten-output audit, Main 11-08: an output
some program skips trips `jax_debug_nans` wherever the launch runs alone, and holds whatever memory held).
The fused norms' and SiLU products' Triton launches, the TPU's GEMM launches, interpreted on a CPU over
finite operands, forward and reverse, at the shapes that take their less common paths: a row past the norm's
register tile, a residual stream, a gate, an activation after the norm, ragged groups with an empty one and
rows past them; and Mosaic GPU's grouped GEMM launches, the class's one gate. `kernel_outputs`'
`launched_outputs` runs each launch alone and reads every output it declares, those its caller slices away
too; an element it never wrote holds the interpreter's NaN."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from kernel_outputs import launched_outputs, unwritten

from dew.nn import norms
from dew.nn.activations import gated_product, packed_gated_product
from dew.nn.kernels import generation, grouped_matmul, norm_swiglu
from dew.nn.kernels.templates.gemm import mosaic_gpu, pricing
from dew.nn.norms import gated_rms_normalized, stream_rms_normalized


def pulled(function, *operands):
    """`function`'s value and its operands' gradients at an all-ones cotangent: its forward's launches
    and its reverse's."""
    value, pull = jax.vjp(function, *operands)
    return value, pull(jax.tree.map(jnp.ones_like, value))


def assert_written(function, *operands, launches: int = 1):
    """Every launch `function(*operands)` runs writes every output it declares, at least `launches` of
    them."""
    found = launched_outputs(function, *operands)
    assert len(found) >= launches, [name for name, _ in found]
    assert unwritten(found) == []


def drawn(*shapes, dtype=jnp.bfloat16):
    rng = np.random.default_rng(4880)
    return tuple(jnp.asarray(rng.standard_normal(shape), dtype) for shape in shapes)


def seam(*, epilogue=None):
    def portable(x, branch, scale):
        return stream_rms_normalized(x, branch, scale, 1e-5, jnp.bfloat16, scale_offset=False,
                                     scale_after_cast=False, fp32_statistics=True, implementation="xla",
                                     activation=epilogue)

    return lambda x, branch, scale: norm_swiglu.stream_rms_norm(x, branch, scale, 1e-5, jnp.bfloat16,
                                                               offset=False, after_cast=False,
                                                               portable=portable, epilogue=epilogue)


@pytest.mark.parametrize("width", [96, norm_swiglu.NARROW_WIDTH + 8], ids=["narrow", "wide"])
@pytest.mark.parametrize("form", ["plain", "stream", "activated"])
def test_the_norms_launches_write_every_output(form, width):
    """The fused RMSNorm's forward (output, stream, mean square) and reverse (input's and stream's
    cotangents, each program's weight-gradient partial row and its compensation) write every element, a
    row narrower than a program's rows and one past the register tile, with a residual stream or an
    activation after the norm."""
    x, branch = drawn((5, width), (5, width))
    scale = jnp.asarray(np.random.default_rng(1).uniform(0.5, 1.5, width), jnp.float32)
    epilogue = jax.nn.swish if form == "activated" else None
    if form == "stream":
        assert_written(lambda x, branch, scale: pulled(seam(), x, branch, scale), x, branch, scale,
                       launches=2)
    else:
        assert_written(lambda x, scale: pulled(lambda x, scale: seam(epilogue=epilogue)(x, None, scale), x,
                                               scale), x, scale, launches=2)


def test_the_gated_norms_launches_write_every_output():
    """The gated RMSNorm's launches write every output, the gate's cotangent among them."""
    x, gate = drawn((7, 64), (7, 64))
    scale = jnp.ones((64,), jnp.float32)

    def portable(x, gate, scale):
        return gated_rms_normalized(x, gate, scale, 1e-6, jnp.bfloat16, "silu", implementation="xla")

    def fused(x, gate, scale):
        return norm_swiglu.gated_rms_norm(x, gate, scale, 1e-6, jnp.bfloat16, "silu", portable)

    assert_written(lambda *operands: pulled(fused, *operands), x, gate, scale, launches=2)


def test_the_unweighted_norms_launches_write_every_output():
    """An unweighted RMSNorm (its rounded inverse, no weight: no partial rows) writes every output."""
    (x,) = drawn((6, 128))

    def fused(x):
        return norm_swiglu.rms_norm(x, None, 1e-6, jnp.bfloat16, offset=False, after_cast=False,
                                    portable=lambda x, scale: norms.unweighted_rmsnorm(x, 1e-6,
                                                                                      implementation="xla"),
                                    inverse_rounded=True)

    assert_written(lambda x: pulled(fused, x), x, launches=2)


@pytest.mark.parametrize("packed", [False, True], ids=["apart", "packed"])
def test_the_silu_products_launches_write_every_output(packed):
    """The fused SiLU product's launches, apart and over a packed projection, write every output, at a
    size no block divides."""
    if packed:
        (gate_up,) = drawn((3, 2, 1000))

        def fused(gate_up):
            return norm_swiglu.packed_swiglu(gate_up, packed_gated_product("swiglu", implementation="xla"))

        assert_written(lambda gate_up: pulled(fused, gate_up), gate_up, launches=2)
        return
    gate, up = drawn((3, 1000), (3, 1000))

    def apart(gate, up):
        return norm_swiglu.swiglu(gate, up, gated_product("swiglu", implementation="xla"))

    assert_written(lambda gate, up: pulled(apart, gate, up), gate, up, launches=2)


@pytest.mark.parametrize("product", ["gmm", "tgmm"])
def test_the_tpus_gemm_launches_write_every_output(product):
    """The template's grouped product and kernel gradient on a TPU core (`mosaic_gpu.gmm`, `tgmm` on the
    MXU's bundle), interpreted, write every output: groups ragged, one empty, rows past them (each row past
    the routed ones its zero accumulator's), and a grouped product of two outputs."""
    sizes = jnp.asarray([100, 0, 150], jnp.int32)
    tiles = mosaic_gpu.Tiles(128, 128, 128, 2, 1, family='mxu')
    options = {"tiles": tiles, "multiprocessors": 1, "interpret": grouped_matmul.INTERPRETED}
    with generation.assumed("v5e"):
        if product == "gmm":
            lhs, rhs = drawn((300, 256), (3, 256, 256))
            launch = functools.partial(mosaic_gpu.gmm, transpose_rhs=False, out_dtype=jnp.float32, **options)
            assert_written(lambda lhs, rhs: launch(lhs, rhs, sizes), lhs, rhs)
            assert_written(lambda lhs, rhs: launch(lhs, rhs, sizes, epilogue=lambda acc: (acc, 2 * acc),
                                                   out_dtypes=(jnp.float32, jnp.bfloat16)), lhs, rhs)
        else:
            lhs, rhs = drawn((300, 256), (300, 256))
            assert_written(lambda lhs, rhs: mosaic_gpu.tgmm(lhs, rhs, sizes, out_dtype=jnp.float32,
                                                            **options), lhs, rhs)


@pytest.mark.parametrize("product", ["gmm", "tgmm"])
def test_the_gpus_grouped_gemm_launches_write_every_output(product):
    """Mosaic GPU's grouped product and kernel gradient (`mosaic_gpu.gmm`, `tgmm`, `plgpu.kernel`'s
    launches), interpreted as an H100's, write every output: groups ragged, one empty, rows past them."""
    from dew.nn.kernels.templates.gemm import mosaic_gpu

    sizes = jnp.asarray([70, 0, 129, 51], jnp.int32)
    tiles = mosaic_gpu.Tiles(128, 128, 64, 2, 1, warpgroups=2, family="wgmma")
    options = {"tiles": tiles, "out_dtype": jnp.float32, "multiprocessors": 4,
               "interpret": grouped_matmul.INTERPRETED}
    with generation.assumed("sm90"):
        if product == "gmm":
            lhs, rhs = drawn((300, 128), (4, 128, 256))
            assert_written(lambda lhs, rhs: mosaic_gpu.gmm(lhs, rhs, sizes, transpose_rhs=False, **options),
                           lhs, rhs)
        else:
            lhs, rhs = drawn((300, 256), (300, 128))
            assert_written(lambda lhs, rhs: mosaic_gpu.tgmm(lhs, rhs, sizes, **options), lhs, rhs)


@pytest.mark.parametrize("product", ["balanced gmm", "balanced tgmm", "scattered gmm"])
def test_the_gpus_partial_gemm_launches_write_every_output(product):
    """Mosaic GPU's launches whose programs share a tile or store only some rows, interpreted as an H100's,
    write every output they declare: a dense product and kernel gradient whose last wave is balanced over 8
    multiprocessors (`pricing.balanced_tail`, each tile's partials in a workspace the launch takes in
    place), and a grouped product scattering its rows, not whole 8-row stores, through a permutation."""
    from dew.nn.kernels.templates.gemm import mosaic_gpu

    tiles = mosaic_gpu.Tiles(128, 128, 64, 3, 1, warpgroups=2, family="wgmma")
    target = dataclasses.replace(generation.gpu_target("sm90"), multiprocessors=8, measured={})
    options = {"tiles": tiles, "out_dtype": jnp.float32, "multiprocessors": 8,
               "interpret": grouped_matmul.INTERPRETED}
    with generation.assumed(target):
        if product == "balanced gmm":
            lhs, rhs = drawn((1408, 128), (1, 128, 128))
            assert pricing.balanced_tail(tiles, (11, 1), 2, (2, 2), target, 8) is not None
            sizes = jnp.asarray([1408], jnp.int32)
            assert_written(lambda lhs, rhs: mosaic_gpu.gmm(lhs, rhs, sizes, transpose_rhs=False, dense=True,
                                                           **options), lhs, rhs)
        elif product == "balanced tgmm":
            lhs, rhs = drawn((320, 128), (320, 1408))
            assert pricing.balanced_tail(tiles, (1, 11), 5, (2, 2), target, 8) is not None
            sizes = jnp.asarray([320], jnp.int32)
            assert_written(lambda lhs, rhs: mosaic_gpu.tgmm(lhs, rhs, sizes, dense=True, **options), lhs, rhs)
        else:
            # 203 rows, which the scattered product pads to whole 8-row stores (`Scattered`'s -1 entries).
            lhs, rhs = drawn((203, 128), (4, 128, 256))
            slots = jnp.asarray(np.random.default_rng(2).permutation(203), jnp.int32)
            assert_written(lambda lhs, rhs: mosaic_gpu.gmm(lhs, rhs, jnp.asarray([70, 0, 69, 51], jnp.int32),
                                                           transpose_rhs=False, scatter=slots, **options),
                           lhs, rhs)


@pytest.mark.parametrize("far", [False, True], ids=["plain", "a-row-rewalked"])
def test_a_decode_steps_attention_launches_write_every_output(far):
    """A decode step through the attention template (`attention_decode.attend`, interpreted as an H100's):
    its plain launch, its slices' merge, and its masked rewalk into the plain launch's outputs in place,
    which declares none of its own (a step whose rows need no rewalk, and one whose first row's query is
    2^16 times a drawn one, past the fold's reach) write every output they declare (M-NS-27's red)."""
    from dew.nn.kernels.templates import attention as template, attention_decode

    query, key, value = drawn((2, 1, 8, 64), (2, 256, 2, 64), (2, 256, 2, 64))
    if far:
        query = query.at[0].multiply(2.0 ** 16)
    spans = template.spans(1, 256, 2, lengths=jnp.asarray([200, 37], jnp.int32),
                           starts=jnp.asarray([0, 16], jnp.int32))
    with generation.assumed("sm90"):
        assert_written(lambda q, k, v: attention_decode.attend(q, k, v, spans, interpret=True), query, key,
                       value)


def test_attentions_training_launches_write_every_output_over_an_empty_row():
    """The attention template's training step (its forward keeping the backward's residuals, and the
    backward), interpreted as an H100's, over two rows of 64 queries, the first seeing no key (a zero-length
    document, `lengths` 0) and the second every one: every launch writes every output it declares, an
    empty row's among them, those inside the walks' loops too (`kernel_outputs.LOOPS`)."""
    from dew.nn.kernels.templates import attention as template, attention_mosaic

    query, key, value, dout = drawn((2, 64, 4, 64), (2, 64, 2, 64), (2, 64, 2, 64), (2, 64, 4, 64))
    spans = template.spans(64, 64, 2, lengths=jnp.asarray([0, 64], jnp.int32))

    def step(query, key, value, dout):
        out, *residuals = attention_mosaic.forward(query, key, value, spans, None,
                                                   residuals=template.Kept.SPLIT, interpret=True)
        grads = attention_mosaic.backward(query, key, value, spans, None, dout, residuals=tuple(residuals),
                                          interpret=True)
        return out, grads

    with generation.assumed("sm90"):
        assert_written(step, query, key, value, dout, launches=2)


def test_the_attention_certify_launch_writes_every_record():
    """The attention forward's certify form (`attention_mosaic.certified`, interpreted as an H100's) over a
    causal call: its records past each query block's walked key tiles hold the empty state its refs start
    from, so every output it declares is written."""
    from dew.nn.kernels.templates import attention as template, attention_mosaic

    query, key, value = drawn((1, 256, 2, 64), (1, 256, 2, 64), (1, 256, 2, 64))
    spans = template.spans(256, 256, 1, causal=True)
    with generation.assumed("sm90"):
        assert_written(lambda q, k, v: attention_mosaic.certified(q, k, v, spans, interpret=True)[1],
                       query, key, value)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_an_unwritten_output_is_found(dtype):
    """The audit's own check: a launch that leaves its second output's last row unwritten is reported, its
    first output, written whole, is not; in bfloat16 too, which NumPy's floating kinds do not hold."""
    from jax.experimental import pallas as pl

    def kernel(x_ref, whole_ref, partial_ref):
        whole_ref[...] = x_ref[...]
        partial_ref[:3] = x_ref[:3]

    def launch(x):
        return pl.pallas_call(kernel, out_shape=(jax.ShapeDtypeStruct((4, 8), dtype),) * 2,
                              interpret=True)(x)[0]

    found = launched_outputs(launch, jnp.ones((4, 8), dtype))
    assert [(gap.output, gap.elements) for gap in unwritten(found)] == [(1, 8)]


def test_a_launch_inside_a_loop_or_a_branch_is_run_trip_by_trip():
    """The audit runs a loop's trips and a branch's chosen body as the program does (`kernel_outputs.LOOPS`),
    each launch inside run alone: a launch leaving its last row unwritten found in a scan's every step, a
    while loop's every trip and a cond's taken branch."""
    from jax.experimental import pallas as pl

    def kernel(x_ref, o_ref):
        o_ref[:3] = x_ref[:3]

    def launch(x):
        return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((4, 8), jnp.float32), interpret=True)(x)

    x = jnp.ones((4, 8), jnp.float32)
    programs = {"scan": lambda x: jax.lax.scan(lambda c, _: (launch(c), None), x, length=3)[0],
                "while": lambda x: jax.lax.fori_loop(0, 2, lambda _, c: launch(c), x),
                "cond": lambda x: jax.lax.cond(x[0, 0] > 0, launch, lambda x: x, x)}
    trips = {"scan": 3, "while": 2, "cond": 1}
    for name, program in programs.items():
        found = launched_outputs(program, x)
        assert len(found) == trips[name] and len(unwritten(found)) == trips[name], name
