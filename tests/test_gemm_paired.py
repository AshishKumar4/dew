"""The template's paired product (`gemm.gemm`'s `paired`: two matrices of one shape multiplied by the same
rows, one launch, its epilogue reading both fp32 accumulators): its check 1 against the two products
combined, through every mode of its epilogue, at a named compute dtype and without an epilogue, and its
lowering; the hooked cases it shares are test_gemm_template's."""

import contextlib
import functools
from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import COMPILES_ON_CPU, generations, lowers, runs
from jax.sharding import Mesh
from reference_error import ahead_arm, assert_as_exact, with_gradients
from test_gemm_template import CASES, SEEDS, _dew, _draw, _drop, _rounded_through

from dew.nn.kernels import gemm, generation, grouped_launch, xla_products
from dew.nn.kernels.templates import mosaic_sm90, resources
from dew.nn.kernels.templates.gemm import chained, derivation, mosaic_gpu, tiling


def swiglu(gate, up):
    """A SwiGLU epilogue on a paired product's two fp32 accumulators, each rounded to bf16 first and the
    activated gate again, as `activations.gated_product` rounds them."""
    gate, up = gate.astype(jnp.bfloat16), up.astype(jnp.bfloat16)
    return (jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16) * up).astype(jnp.float32)


def test_a_grouped_call_no_cooperative_tile_holds_derives_one_consumer_alone():
    """Where no two-consumer tile holds a grouped call, its derivation falls back to one consumer, which
    runs on one multiprocessor: a pair is two consumers' (`tiling.Tiles`), so the fallback derives a
    tile rather than refusing one it built."""

    tile = derivation.derived((mosaic_sm90.WGMMA,), rows=512, depth=2048, width=1536, dtype=jnp.bfloat16,
                              multiprocessors=132, target=generation.TARGETS["sm90"],
                              candidate=lambda tile: tile.warpgroups == 1,
                              launch_of=lambda tile: resources.Launch(lambda stages: resources.Scope(),
                                                                      tile.warpgroups, 0), grouped=True)
    assert tile is not None and (tile.warpgroups, tile.cluster) == (1, 1)


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90", "sm100"))
@pytest.mark.parametrize("grouped", [False, True], ids=["dense", "grouped"])
def test_the_templates_paired_product_holds_check_1(generation_name, grouped):
    """A gate and an up projection as the template's paired product (`gemm`'s `paired`): its value on
    warpgroup MMA, which takes hooks (one load of the rows for both, their SwiGLU in its epilogue), and every
    gradient (the rows', both matrices') through the paired product's own reverse mode on the routes the
    selector takes (its two accumulators recomputed, the input gradient's two products summed wide: the
    SwiGLU's fp32 cotangents multiply wide, which warpgroup MMA's 16-bit products do not take), each as
    exact as the unfused program's from float64 truth. On mma.sync too the paired product is one launch
    (row 2100). The gradients run eagerly, as the Mosaic interpreters' ordered callbacks need."""
    rows, depth, width = 136, 128, 256
    sizes = jnp.asarray([70, 66], jnp.int32) if grouped else None
    groups = 2 if grouped else 1

    def draw(seed):
        rng = np.random.default_rng(seed)
        shape = (groups, depth, width) if grouped else (depth, width)
        return (jnp.asarray(rng.normal(size=(rows, depth)), jnp.bfloat16),
                jnp.asarray(rng.normal(size=shape) / 8, jnp.bfloat16),
                jnp.asarray(rng.normal(size=shape) / 8, jnp.bfloat16),
                jnp.asarray(rng.normal(size=(rows, width)), jnp.bfloat16))

    def dew(x, gate, up):
        return gemm.gemm(x, gate, sizes, paired=up, epilogue=swiglu, out_dtype=jnp.bfloat16,
                             interpret_on_cpu=True)

    def plain(x, gate, up, wide=jnp.float32, out=jnp.bfloat16):
        def projected(matrix):
            if not grouped:
                return x.astype(wide) @ matrix.astype(wide)
            return jnp.concatenate([x[:70].astype(wide) @ matrix[0].astype(wide),
                                    x[70:].astype(wide) @ matrix[1].astype(wide)])
        g, u = projected(gate), projected(up)
        # The program rounds as the epilogue does (`swiglu`); float64 truth rounds nothing.
        return (swiglu(g, u) if wide == jnp.float32 else jax.nn.silu(g) * u).astype(out)

    def valued(f):
        """`f` of the drawn inputs but the cotangent."""
        return lambda x, gate, up, cotangent: f(x, gate, up)

    label = f"paired {'grouped' if grouped else 'dense'} product on {generation_name}"
    truth = functools.partial(plain, wide=jnp.float64, out=jnp.float64)
    with generation.assumed(generation_name):
        # Named, the value is the one paired launch's on either family.
        with generation.chosen(grouped_matmul="mosaic"), generation.routes() as routes:
            assert_as_exact(jax.jit(valued(dew)), jax.jit(valued(plain)), draw, label, seeds=SEEDS,
                            truth_fn=valued(truth))
        assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {"mosaic"}
        assert_as_exact(with_gradients(dew), jax.jit(with_gradients(plain)), draw, f"{label}, gradients",
                        seeds=SEEDS, truth_fn=with_gradients(truth))


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
def test_a_paired_products_epilogue_reads_its_fp32_accumulators_in_every_mode(generation_name):
    """A paired product's epilogue (a product of its two accumulators, fp32 out) at a named compute dtype,
    whole on warpgroup MMA or apart on mma.sync: its value, every reverse gradient and its forward-mode
    tangent read the two fp32 accumulators, as the composed program's (two portable projections summed
    wide, their product), so its derivatives are of one function: each on par with that program's from
    float64 truth (check 1)."""
    rows, depth, width = 136, 128, 256
    sizes = jnp.asarray([70, 66], jnp.int32)

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(rows, depth)), jnp.float32),
                *(jnp.asarray(rng.normal(size=(2, depth, width)) / 8, jnp.float32) for _ in range(2)),
                jnp.asarray(rng.normal(size=(rows, width)), jnp.float32))

    def dew(x, gate, up):
        return gemm.gemm(x, gate, sizes, paired=up, epilogue=lambda g, u: g * u, compute=jnp.bfloat16,
                             out_dtype=jnp.float32, precision="default", interpret_on_cpu=True)

    def plain(x, gate, up, dtype=jnp.bfloat16):
        g, u = (xla_products.portable_projection(x, matrix, sizes, dtype, "default",
                                                   out_dtype=jnp.promote_types(dtype, jnp.float32))
                for matrix in (gate, up))
        return g * u

    def tangent(f):
        """`f`'s forward-mode tangent along the drawn direction (the cotangent's draw, the rows')."""
        return lambda x, gate, up, direction: jax.jvp(f, (x, gate, up),
                                                      (direction[:, :depth], gate, up))[1]

    def truth(x, gate, up):
        # The law's operands rounded to the compute dtype, everything after them exact.
        return plain(*(_rounded_through(array, jnp.bfloat16) for array in (x, gate, up)), dtype=jnp.float64)

    with generation.assumed(generation_name):
        assert_as_exact(with_gradients(dew), jax.jit(with_gradients(plain)), draw,
                        f"paired epilogue's gradients ({generation_name})", seeds=SEEDS,
                        truth_fn=with_gradients(truth))
        assert_as_exact(tangent(dew), jax.jit(tangent(plain)), draw,
                        f"paired epilogue's tangent ({generation_name})", seeds=SEEDS,
                        truth_fn=tangent(truth))


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
def test_a_paired_product_at_a_named_compute_dtype_holds_check_1(generation_name):
    """A gate and an up projection of fp32 rows and masters multiplied in bf16 (`gemm`'s `compute`, the MoE
    projection's law: the operands rounded, the products summed wide and rounded once, each tangent
    through the rounded operands) as one paired product without an epilogue: its two outputs and every
    gradient (the rows' in fp32, both masters' in fp32) as exact as the two unfused projections'
    (`xla_products.portable_projection`) from float64 truth."""
    rows, depth, width = 136, 128, 256
    sizes = jnp.asarray([70, 66], jnp.int32)

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(rows, depth)), jnp.float32),
                *(jnp.asarray(rng.normal(size=(2, depth, width)) / 8, jnp.float32) for _ in range(2)),
                tuple(jnp.asarray(rng.normal(size=(rows, width)), jnp.bfloat16) for _ in range(2)))

    def dew(x, gate, up):
        return gemm.gemm(x, gate, sizes, paired=up, compute=jnp.bfloat16, out_dtype=jnp.bfloat16,
                             out_dtypes=(jnp.bfloat16, jnp.bfloat16), interpret_on_cpu=True)

    def plain(x, gate, up, dtype=jnp.bfloat16):
        return tuple(xla_products.portable_projection(x, matrix, sizes, dtype, None)
                     for matrix in (gate, up))

    def truth(x, gate, up):
        # The law's operands rounded to the compute dtype, everything after them exact.
        return plain(*(_rounded_through(array, jnp.bfloat16) for array in (x, gate, up)), dtype=jnp.float64)

    with generation.assumed(generation_name):
        assert_as_exact(with_gradients(dew), jax.jit(with_gradients(plain)), draw,
                        f"paired product at a named compute dtype ({generation_name})", seeds=SEEDS,
                        truth_fn=with_gradients(truth))


@pytest.mark.parametrize(("generation_name", "out"), [pytest.param("sm90", jnp.bfloat16, marks=runs("sm90")),
                                                    (None, jnp.float32)], ids=["sm90", "XLA"])
def test_the_templates_paired_product_without_an_epilogue_holds_check_1(generation_name, out):
    """A paired product without an epilogue, each matrix's product an output of its own, on the route the
    selector takes: in bf16 on an sm90, and in XLA's program in fp32, whose fp32 cotangents its
    reverse mode keeps wide; its values and every gradient (the rows', both matrices') as exact as the
    unfused program's from float64 truth. The gradients run eagerly, as the Mosaic interpreters' ordered
    callbacks need."""
    rows, depth, width = 136, 128, 256

    def draw(seed):
        rng = np.random.default_rng(seed)
        return (jnp.asarray(rng.normal(size=(rows, depth)), jnp.bfloat16),
                *(jnp.asarray(rng.normal(size=(depth, width)) / 8, jnp.bfloat16) for _ in range(2)),
                tuple(jnp.asarray(rng.normal(size=(rows, width)), out) for _ in range(2)))

    def dew(x, gate, up):
        return gemm.gemm(x, gate, None, paired=up, out_dtype=out, out_dtypes=(out, out),
                             interpret_on_cpu=generation_name is not None)

    def plain(x, gate, up, wide=jnp.float32, cast=out):
        # The unfused program's products, its operands' own dtype summed wide (XLA's tensor-core product
        # on a GPU, as the template's); float64 truth rounds nothing.
        if wide == jnp.float64:
            return tuple((x.astype(wide) @ matrix.astype(wide)).astype(cast) for matrix in (gate, up))
        return tuple(jax.lax.dot(x, matrix, preferred_element_type=wide).astype(cast)
                     for matrix in (gate, up))

    with generation.assumed(generation_name):
        assert_as_exact(with_gradients(dew), jax.jit(with_gradients(plain)), draw,
                        f"paired product's two outputs ({generation_name})", seeds=SEEDS,
                        truth_fn=with_gradients(functools.partial(plain, wide=jnp.float64, cast=jnp.float64)))


@lowers("sm90")
def test_a_hook_that_broadcasts_a_side_operand_lowers_for_hopper():
    """A side operand given back with its dimension of one, broadcast across its tile
    (`hooks._unsqueezed`), which Mosaic's layout inference lays out where it found no layout for the
    dimension of one itself (GMMGATE-h100-8): a Hopper call's forward and kernel gradient with such a
    prologue lower on the template's launch, named."""
    drawn = [jax.ShapeDtypeStruct(array.shape, array.dtype) for array in _draw(0)]
    dew = _drop(_dew(CASES["broadcast axes"]))

    def loss(*drawn):
        return sum(jnp.sum(out.astype(jnp.float32)) for out in dew(*drawn))

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with (generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh),
          jax.default_matmul_precision("default"), generation.chosen(grouped_matmul="mosaic"),
          generation.routes() as routes):
        jax.jit(jax.grad(loss, argnums=range(len(drawn)))).trace(*drawn).lower(lowering_platforms=("cuda",))
    assert {route.kernel for route in routes if route.choice == "grouped_matmul"} == {"mosaic"}


@lowers("sm90")
def test_a_paired_products_input_gradient_is_one_launch_over_its_joined_depths():
    """A gate and an up projection's paired product with its two bf16 outputs (a mixture of experts' pair,
    `moe.paired_projections`) differentiated for its rows on an H100, the template named: its input gradient,
    the two cotangents through their matrices, traces one launch, the two products summed in its one
    accumulator over their joined depths (`gemm`'s `joined`), where it traced two, the second adding the
    first's wide product in its epilogue: lm-moe's start certified and built both, at two tiles (row 6783)."""
    from dew.nn.kernels import tuning

    rows, depth, width = 256, 128, 256
    shapes = (jax.ShapeDtypeStruct((rows, depth), jnp.bfloat16),
              *(jax.ShapeDtypeStruct((depth, width), jnp.bfloat16) for _ in range(2)))

    def loss(x, gate, up):
        gated, upped = gemm.gemm(x, gate, None, paired=up, out_dtype=jnp.bfloat16,
                                 out_dtypes=(jnp.bfloat16, jnp.bfloat16))
        return jnp.sum(swiglu(gated, upped).astype(jnp.float32))

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with (generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh),
          jax.default_matmul_precision("default"), generation.chosen(grouped_matmul="mosaic"),
          tuning.seen() as launches):
        jax.jit(jax.grad(loss)).trace(*shapes).lower(lowering_platforms=("cuda",))
    input_gradients = [bound for _, bound in launches.taken
                       if bound.arguments.get("trans_rhs") and bound.arguments.get("kernel") is not None]
    assert len(input_gradients) == 1, len(input_gradients)


@pytest.mark.parametrize("generation_name", generations("sm90"))
def test_a_joined_input_gradient_is_as_finite_as_its_two_products(generation_name):
    """A paired product's input gradient over its joined depths (`gemm`'s `joined`) sums each product apart
    and adds the two at the finish: where each product's terms cancel to zero inside fp32's range, the joined
    gradient is zero too, never a sum of the two products' partials past the range (base review F2: +2^126
    terms of the gate's and the up projection's, each product's own cancelled by its -2^126 ones a depth step
    later, would meet at 2^128 in one running sum)."""
    rows, depth, width = 128, 128, 256
    big = 2.0 ** 63
    matrix = np.zeros((depth, width), np.float32)
    matrix[0, [0, 1]], matrix[0, [128, 129]] = big, -big
    cotangent = np.zeros((rows, width), np.float32)
    cotangent[:, [0, 1, 128, 129]] = big
    x = jnp.zeros((rows, depth), jnp.bfloat16)
    gate = up = jnp.asarray(matrix, jnp.bfloat16)
    cotangents = (jnp.asarray(cotangent, jnp.bfloat16),) * 2

    def paired(x):
        return gemm.gemm(x, gate, None, paired=up, out_dtype=jnp.bfloat16,
                         out_dtypes=(jnp.bfloat16, jnp.bfloat16), interpret_on_cpu=True)

    with (generation.assumed(generation_name), jax.default_matmul_precision("default"),
          generation.chosen(grouped_matmul="mosaic")):
        _, pullback = jax.vjp(paired, x)
        (d_x,) = pullback(cotangents)
    assert np.all(np.asarray(d_x, np.float32) == 0), np.asarray(d_x, np.float32)[0, :4]


WIDTHS = (512, 128, 128)
"""A query's, a key's and a value's widths: matrices side by side, the first several column tiles."""


def _side_by_side_draw(dense: bool, depth: int, sizes: tuple):
    """A draw of bf16 rows, matrices of `WIDTHS` (one group where `dense`), and fp32 cotangents."""
    groups = 1 if dense else len(sizes)
    rows = sum(sizes)

    def draw(seed):
        keys = jax.random.split(jax.random.key(seed), 2 + 2 * len(WIDTHS))
        matrices = tuple((jax.random.normal(key, (groups, depth, width)) / np.sqrt(depth)).astype(
            jnp.bfloat16) for key, width in zip(keys[2:2 + len(WIDTHS)], WIDTHS, strict=True))
        cotangents = tuple(jax.random.normal(key, (rows, width), jnp.float32)
                           for key, width in zip(keys[2 + len(WIDTHS):], WIDTHS, strict=True))
        return jax.random.normal(keys[0], (rows, depth)).astype(jnp.bfloat16), matrices, cotangents

    return draw


def _apart(rows, matrices, sizes: tuple, dense: bool):
    """Each matrix's product with its group's rows in XLA (`[groups, depth, width]` matrices), fp32 out of
    16-bit products, float64 out of float64's."""
    out = jnp.promote_types(rows.dtype, jnp.float32)
    if dense:
        return tuple(jnp.dot(rows, matrix[0], preferred_element_type=out) for matrix in matrices)
    hot = jax.nn.one_hot(np.repeat(np.arange(len(sizes)), sizes), len(sizes), dtype=out)
    return tuple(jnp.einsum("rg,rk,gkn->rn", hot, rows.astype(out), matrix.astype(out))
                 for matrix in matrices)


@pytest.mark.parametrize("dense", [True, False], ids=["dense", "grouped"])
def test_matrices_side_by_side_are_each_ones_product_in_one_launch(dense):
    """Three matrices of one depth and widths 512, 128 and 128 (a query's, a key's and a value's) side by
    side in one launch (`gmm`'s `beside`, `Beside`), interpreted on an H100's rings: each output the rows
    times its own matrix, on par with XLA's products apart against float64, a dense call's and a grouped
    call's (ragged groups, one empty), each column tile reading and storing its own matrix's (row 6794: a
    join without its kernels concatenated or its output split); an A100's pipeline walk refuses them."""
    tiles = tiling.Tiles(128, 128, 64, 3, 1, warpgroups=2, family="wgmma")
    sizes = (200,) if dense else (70, 0, 130)
    draw = _side_by_side_draw(dense, 256, sizes)

    def launch(rows, matrices, cotangents):
        del cotangents
        group_sizes = None if dense else jnp.asarray(sizes, jnp.int32)
        return mosaic_gpu.gmm(rows, matrices[0], group_sizes, tiles=tiles, transpose_rhs=False,
                              out_dtype=jnp.float32, multiprocessors=4, interpret=grouped_launch.INTERPRETED,
                              dense=dense, beside=matrices[1:])

    def apart(rows, matrices, cotangents):
        del cotangents
        return _apart(rows, matrices, sizes, dense)

    with generation.assumed("sm90"):
        assert_as_exact(launch, jax.jit(apart), draw, "matrices side by side, one launch", seeds=SEEDS)
    rows, matrices, _ = draw(0)
    with generation.assumed("sm80"), pytest.raises(ValueError):
        mosaic_gpu.gmm(rows, matrices[0], None if dense else jnp.asarray(sizes, jnp.int32),
                       tiles=tiling.Tiles(64, 128, 32, 3, 1, family="mma.sync"), transpose_rhs=False,
                       out_dtype=jnp.float32, multiprocessors=4, interpret=grouped_launch.INTERPRETED,
                       dense=dense, beside=matrices[1:])


@pytest.mark.parametrize("kernel", ["mosaic", "xla"])
@pytest.mark.parametrize("dense", [True, False], ids=["dense", "grouped"])
def test_projections_side_by_side_are_on_par_with_their_gradients(kernel, dense):
    """`gemm.gemm` of bf16 rows by a query's, a key's and a value's matrices side by side (`beside`; widths
    512, 128, 128 at a depth of 200, which no step divides, each matrix's padded) on an H100, interpreted,
    its launch Mosaic GPU's one side-by-side launch or XLA's products, fp32 outputs: the outputs, the rows'
    gradient (the three cotangents' products chained in one accumulator) and each matrix's on par with
    XLA's products apart against float64, a dense call's and a grouped call's (row 6794)."""
    sizes = (200,) if dense else (70, 0, 130)
    group_sizes = None if dense else jnp.asarray(sizes, jnp.int32)
    draw = _side_by_side_draw(dense, 200, sizes)

    def together(rows, matrices):
        first, *beside = (matrix[0] for matrix in matrices) if dense else matrices
        return gemm.gemm(rows, first, group_sizes, beside=tuple(beside), out_dtype=jnp.float32,
                         interpret_on_cpu=True)

    def through(rows, matrices, cotangents):
        with generation.assumed("sm90"), generation.chosen(grouped_matmul=kernel):
            return with_gradients(together)(rows, matrices, cotangents)

    def apart(rows, matrices, cotangents):
        return with_gradients(lambda rows, matrices: _apart(rows, matrices, sizes, dense))(rows, matrices,
                                                                                            cotangents)

    assert_as_exact(jax.jit(through), jax.jit(apart), draw, f"projections side by side {kernel}", seeds=SEEDS)


def test_side_by_side_gradients_read_a_wide_cotangent_wide():
    """`gemm.gemm` of float16 rows by two float16 matrices side by side into float32 outputs, each cotangent
    70000 (past float16's 65504): every gradient finite, each matrix's 64 x 70000 / 128 rounded once to
    float16, and the rows' the sum of both cotangents' products, 70000 x 192 / 1024, rounded once (the
    cotangents worked wide, as every product's reverse pass reads them, `_worked`)."""
    rows = jnp.full((64, 128), 1 / 128, jnp.float16)
    first, second = (jnp.full((128, width), 1 / 1024, jnp.float16) for width in (128, 64))
    cotangents = tuple(jnp.full((64, width), 70000.0, jnp.float32) for width in (128, 64))

    def together(rows, first, second):
        return gemm.gemm(rows, first, beside=(second,), out_dtype=jnp.float32)

    d_rows, *gradients = jax.vjp(together, rows, first, second)[1](cotangents)
    for gradient in gradients:
        expected = np.full(gradient.shape, np.float16(64 * 70000 / 128))
        np.testing.assert_array_equal(np.asarray(gradient), expected)
    np.testing.assert_array_equal(np.asarray(d_rows), np.full((64, 128), np.float16(70000.0 * 192 / 1024)))


@pytest.mark.parametrize(("generation_name", "hidden", "widths", "rows", "dtype"), [
    ("sm90", 4096, (4096, 1024, 1024), 12288, jnp.bfloat16),
    ("sm90", 2560, (8192, 1024, 1024), 12288, jnp.bfloat16),
    ("sm90", 2560, (8192, 1024, 1024), 2048, jnp.bfloat16),
    ("sm90", 4096, (4096, 1024, 1024), 12288, jnp.float16),
    ("sm80", 4096, (4096, 1024, 1024), 12288, jnp.bfloat16)],
    ids=["h100-8b-4x3072", "h100-4b-4x3072", "h100-4b-1x2048", "h100-8b-fp16", "a100-8b-4x3072"])
def test_projections_side_by_side_admit_legal_hopper_shapes_and_refuse_ampere(generation_name, hidden, widths,
                                                                              rows, dtype):
    """A layer's query, key and value over `dtype` rows (`gemm.side_by_side`): on an assumed H100 with the
    template named explicitly, every valid Hopper geometry is admitted as `mosaic`; on an assumed A100,
    whose cp.async rings take no matrices side by side, none is admitted and naming the template raises
    `KernelRefused` (row 9769)."""
    from dew.nn.kernels.selection import KernelRefused

    matrices = tuple(jax.ShapeDtypeStruct((hidden, width), dtype) for width in widths)
    lhs = jax.ShapeDtypeStruct((rows, hidden), dtype)
    with generation.assumed(generation_name):
        if generation_name == "sm80":
            assert gemm.side_by_side(lhs, matrices, dtype, None, whole=True) is None
            with generation.chosen(grouped_matmul="mosaic"), pytest.raises(KernelRefused):
                gemm.side_by_side(lhs, matrices, dtype, None, whole=True)
        else:
            with generation.chosen(grouped_matmul="mosaic"):
                route = gemm.side_by_side(lhs, matrices, dtype, None, whole=True)
            assert route is not None and route.kernel == "mosaic", route

def _overflow_witness(depth: int, rank: int) -> tuple[np.ndarray, ...]:
    """LoRA operands (`x [256, depth]`, `W [depth, 256]`, `A [depth, rank]`, `B [rank, 256]`) whose first
    output's terms over the concatenated depth are -2^127 (the base's `x[0, 0] · W[0, 0]`, 2^63 times
    -2^64) then +2^127 twice (the rank's, each 2^63 times 2^64): equal depths take `A` the identity and the
    rank's terms at `x`'s columns 0 and 64; a 127-deep base `A`'s first row's two ones, the rank's terms its
    rows 0 and 1, at the concatenated depth's offsets 127 and 128 (base review 3f38bf20d F1, F2)."""
    x, w = np.zeros((256, depth), np.float32), np.zeros((depth, 256), np.float32)
    b = np.zeros((rank, 256), np.float32)
    x[0, 0], w[0, 0] = 2.0 ** 63, -(2.0 ** 64)
    if depth == rank:
        a = np.eye(depth, dtype=np.float32)
        x[0, 64] = 2.0 ** 63
        b[0, 0] = b[64, 0] = 2.0 ** 64
    else:
        a = np.zeros((depth, rank), np.float32)
        a[0, 0] = a[0, 1] = 1.0
        b[0, 0] = b[1, 0] = 2.0 ** 64
    return x, w, a, b


@pytest.mark.parametrize(("depth", "rank", "out_dtype", "precision"), [
    (128, 128, jnp.bfloat16, None), (127, 64, jnp.float32, jax.lax.Precision.HIGHEST)],
    ids=["equal-depths", "ragged-base"])
def test_an_adapted_products_running_sum_crosses_its_base_and_rank_as_one(depth, rank, out_dtype, precision):
    """LoRA's adapted product (`low_rank.low_rank`, `gemm.joined_depths`) of `_overflow_witness`'s operands on
    an assumed H100, the template named and interpreted: its first output's terms -2^127, +2^127, +2^127 sum
    to 2^127 in one running sum over the concatenated depth, where a pair's two accumulators (equal depths)
    or padding between the depths moving both positive terms into one exact-sum step (a 127-deep base)
    overflowed to infinity (base review 3f38bf20d F1, F2)."""
    from dew.nn.kernels.low_rank import low_rank

    operands = (jnp.asarray(each, jnp.bfloat16) for each in _overflow_witness(depth, rank))
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic", low_rank="fused"):
        out = low_rank(*operands, 1.0, out_dtype=out_dtype, precision=precision, interpret_on_cpu=True)
    assert float(np.asarray(out, np.float32)[0, 0]) == 2.0 ** 127


def test_side_by_side_forward_and_gradients_match_independent_projections():
    """Matrices side by side through `gemm.gemm` with the template named explicitly on an assumed H100
    (interpreted) match independent `gemm.gemm` projections in both forward values and input/matrix gradients
    (rows 9665, 9768, 9769)."""
    rng = np.random.default_rng(17)
    rows = jnp.asarray(rng.integers(-2, 3, size=(128, 128)), jnp.bfloat16)
    first = jnp.asarray(rng.integers(-2, 3, size=(128, 128)), jnp.bfloat16)
    second = jnp.asarray(rng.integers(-2, 3, size=(128, 64)), jnp.bfloat16)

    def together(x, a, b):
        outs = gemm.gemm(x, a, beside=(b,), out_dtype=jnp.float32, interpret_on_cpu=True)
        return jnp.sum(outs[0]) + jnp.sum(outs[1]), outs

    def apart(x, a, b):
        o0 = gemm.gemm(x, a, out_dtype=jnp.float32, interpret_on_cpu=True)
        o1 = gemm.gemm(x, b, out_dtype=jnp.float32, interpret_on_cpu=True)
        return jnp.sum(o0) + jnp.sum(o1), (o0, o1)

    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        (_, outs_t), grads_t = jax.value_and_grad(together, argnums=(0, 1, 2), has_aux=True)(
            rows, first, second)
        (_, outs_a), grads_a = jax.value_and_grad(apart, argnums=(0, 1, 2), has_aux=True)(
            rows, first, second)
    for got, exp in zip((*outs_t, *grads_t), (*outs_a, *grads_a), strict=True):
        np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(exp, np.float32))

def test_projections_side_by_side_whose_forward_a_pin_refuses_run_apart():
    """Projections of widths 256, 128 and 128 over 12,288 bf16 rows on an assumed H100, the template pinned
    at a 256-column Hopper tile: their forward side by side is the launch's, which the pin refuses for their
    narrower widths, so `gemm.side_by_side` takes none (head review of 4d6c7a69c F5: an input gradient alone
    admitted them, and the forward raised `Unpinnable`); unpinned, the template named, it takes them."""
    from dew.nn.kernels import grouped_matmul, tuning

    lhs = jax.ShapeDtypeStruct((12288, 256), jnp.bfloat16)
    matrices = tuple(jax.ShapeDtypeStruct((256, width), jnp.bfloat16) for width in (256, 128, 128))
    pin = tiling.Tiles(128, 256, 64, 2, 1, warpgroups=2, family="wgmma")
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        assert gemm.side_by_side(lhs, matrices, jnp.bfloat16, None, whole=True) is not None
        with tuning.pinned({grouped_matmul.MosaicGmm: pin}):
            assert gemm.side_by_side(lhs, matrices, jnp.bfloat16, None, whole=True) is None


@pytest.mark.parametrize("widths", [(512, 128), (512, 128, 128)], ids=["two-outputs", "three-outputs"])
def test_side_by_side_projections_of_unequal_widths_take_their_input_gradient_chained(widths):
    """`gemm.gemm` of bf16 rows by grouped matrices side by side of widths 512 and 128 (and a third of 128),
    its launch the template's on an H100, interpreted, its first two outputs reached by the reverse pass:
    the rows' gradient is the reached cotangents' products summed and rounded once, its two products of
    unequal widths no joined product (whose matrices are of one shape) but the input gradient's chain."""
    rng = np.random.default_rng(11)
    sizes = jnp.asarray([70, 130], jnp.int32)
    rows = jnp.asarray(rng.integers(-3, 4, size=(200, 256)), jnp.bfloat16)
    matrices = tuple(jnp.asarray(rng.integers(-3, 4, size=(2, 256, width)), jnp.bfloat16) for width in widths)

    def reached(rows):
        outs = gemm.gemm(rows, matrices[0], sizes, beside=matrices[1:], out_dtype=jnp.bfloat16,
                         interpret_on_cpu=True)
        return sum(jnp.sum(outs[at].astype(jnp.float32)) for at in (0, 1))

    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        d_rows = jax.grad(reached)(rows)
    group_of = np.repeat(np.arange(2), [70, 130])
    exact = sum(np.asarray(matrices[at], np.float64)[group_of].sum(axis=-1) for at in (0, 1))
    rounded = jnp.asarray(exact, jnp.float32).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(d_rows, np.float32), np.asarray(rounded, np.float32))


@pytest.mark.parametrize(("gpu", "rows", "depth", "sizes"),
                         [("sm90", 2, 16384, (1, 1)), ("sm120", 1, 128, None)],
                         ids=["h100-decode-split", "sm120-decode-swap"])
def test_side_by_side_decode_steps_derive_a_tile_that_takes_them(gpu, rows, depth, sizes):
    """`gemm.gemm` of matrices side by side at a decode step's few rows, its launch the template's,
    interpreted: on an H100 two one-row groups over a 16384 deep contraction (whose tile a depth split
    would take), on an RTX PRO 6000 a dense one-row call (whose tile a swap would take), bf16 outputs. Each
    output is its matrix's product with the rows, its exact fp32 sum rounded once (small integers): the
    derived tile one whose split and swap the side-by-side launch runs (neither), judged as derived, not as
    the run of a split or a swap."""
    rng = np.random.default_rng(17)
    lhs = jnp.asarray(rng.integers(-2, 3, size=(rows, depth)), jnp.bfloat16)
    shape = (depth, 128) if sizes is None else (len(sizes), depth, 128)
    matrices = tuple(jnp.asarray(rng.integers(-2, 3, size=shape), jnp.bfloat16) for _ in range(2))
    group_sizes = None if sizes is None else jnp.asarray(sizes, jnp.int32)
    with generation.assumed(gpu), generation.chosen(grouped_matmul="mosaic"):
        outs = gemm.gemm(lhs, matrices[0], group_sizes, beside=matrices[1:], out_dtype=jnp.bfloat16,
                         interpret_on_cpu=True)
    wide = np.asarray(lhs, np.float64)
    for out, matrix in zip(outs, matrices, strict=True):
        each = np.asarray(matrix, np.float64)
        exact = wide @ each if sizes is None else np.stack([wide[at] @ each[at] for at in range(rows)])
        rounded = jnp.asarray(exact, jnp.float32).astype(jnp.bfloat16)
        np.testing.assert_array_equal(np.asarray(out, np.float32), np.asarray(rounded, np.float32))


def test_side_by_side_input_gradient_sums_in_the_plain_programs_order():
    """`gemm.gemm` of a float32 row by three one-entry matrices side by side, 2^126, 2^126 and -2^126, each
    cotangent 2: the rows' gradient is 2^127, as the plain program's reverse pass sums the three products
    (the last first), where the forward's order overflows at 2^128 (the base review's F1)."""
    rows = jnp.zeros((1, 1), jnp.float32)
    first, second, third = (jnp.full((1, 1), value, jnp.float32)
                            for value in (2.0 ** 126, 2.0 ** 126, -2.0 ** 126))
    cotangents = (jnp.full((1, 1), 2.0, jnp.float32),) * 3

    def together(rows):
        return gemm.gemm(rows, first, beside=(second, third), out_dtype=jnp.float32)

    def apart(rows):
        return tuple(gemm.gemm(rows, matrix, out_dtype=jnp.float32) for matrix in (first, second, third))

    (got,) = jax.vjp(together, rows)[1](cotangents)
    (expected,) = jax.vjp(apart, rows)[1](cotangents)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(got), np.full((1, 1), 2.0 ** 127, np.float32))


@pytest.mark.parametrize("kernel", [None, "xla"], ids=["derived", "xla"])
def test_a_matrix_of_no_columns_side_by_side_has_an_empty_output_in_its_place(kernel):
    """`gemm.gemm` of bf16 rows by a matrix of no columns beside one of 128, on an H100 as derived
    (interpreted) or XLA's: the first output empty, the second the rows' product with its matrix exactly
    (small integers), each gradient that of the second product alone (the base review's F2)."""
    rng = np.random.default_rng(19)
    rows = jnp.asarray(rng.integers(-2, 3, size=(128, 128)), jnp.bfloat16)
    empty = jnp.zeros((128, 0), jnp.bfloat16)
    matrix = jnp.asarray(rng.integers(-2, 3, size=(128, 128)), jnp.bfloat16)
    cotangent = jnp.asarray(rng.integers(-2, 3, size=(128, 128)), jnp.float32)
    chosen = {} if kernel is None else {"grouped_matmul": kernel}

    def together(rows, empty, matrix):
        return gemm.gemm(rows, empty, beside=(matrix,), out_dtype=jnp.float32, interpret_on_cpu=True)

    def alone(rows, matrix):
        return gemm.gemm(rows, matrix, out_dtype=jnp.float32, interpret_on_cpu=True)

    with generation.assumed("sm90"), generation.chosen(**chosen):
        (none, out), pull = jax.vjp(together, rows, empty, matrix)
        d_rows, d_empty, d_matrix = pull((jnp.zeros((128, 0), jnp.float32), cotangent))
        expected, pull_alone = jax.vjp(alone, rows, matrix)
        d_rows_alone, d_matrix_alone = pull_alone(cotangent)
    assert none.shape == (128, 0) and d_empty.shape == (128, 0)
    for got, want in ((out, expected), (d_rows, d_rows_alone), (d_matrix, d_matrix_alone)):
        np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))


@COMPILES_ON_CPU
@pytest.mark.parametrize("dense", [True, False], ids=["dense", "grouped"])
def test_matrices_side_by_side_compile_for_an_h100_within_its_resources(dense):
    """Three matrices side by side (widths 512, 128, 128) in one launch on an H100's rings, compiled by Mosaic
    for a compile-only H100 through ptxas: within its resources, no register spilled, a dense call's L2
    prefetch of each program's first tile reading that tile's own matrix (`BesideRefs.columns`), a grouped
    call's none."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    tiles = tiling.Tiles(128, 128, 64, 4, 1, warpgroups=2, family="wgmma")
    groups = 1 if dense else 3
    shaped = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    operands = (shaped((4096, 2048)), *(shaped((groups, 2048, width)) for width in (512, 128, 128)),
                jax.ShapeDtypeStruct((groups,), jnp.int32))

    def launch(rows, first, second, third, sizes):
        return mosaic_gpu.gmm(rows, first, None if dense else sizes, tiles=tiles, transpose_rhs=False,
                              out_dtype=jnp.bfloat16, multiprocessors=132, dense=dense,
                              beside=(second, third))

    case = KernelCase(f"matrices side by side {'dense' if dense else 'grouped'}", lambda: (launch, operands))
    with generation.assumed("sm90"):
        found = compile_mosaic_kernels([case], "sm90")
    assert found and all(record.get("kernels") for record in found), found
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))

CHAINED = (256, 128, 128)
"""A query's, a key's and a value's widths: an input gradient's three products, chained end to end."""

FEATURES = 256
"""Their rows' width, the input gradient's: a tile chaining three products holds its replay's tile, a
quarter of its columns and whole column units (`grouped_matmul.MosaicGmm._replays`), at 256 columns."""


def _chained_overflow(dtype=jnp.bfloat16) -> tuple:
    """Rows of zeros and matrices of `CHAINED` widths by `FEATURES`, cotangents of 2^63 at a product's
    first columns: the value's products (the transpose order's first) +2^126 at feature 0's two first
    columns, 2^127 in all; the key's +2^126 there and -2^126 two columns into its second 64-step, 0 in
    all; the query's none. Each product's own sums and the total (2^127) lie in float32's range; one running
    sum over the joined contraction meets 2^128 at the key's first step (base review 0711d5f57 F2). Host
    arrays, NumPy's, in `dtype`: a GPU tier's prepare computes on no device."""
    big = 2.0 ** 63
    matrices = [np.zeros((FEATURES, width), np.float32) for width in CHAINED]
    matrices[2][0, [0, 1]] = big
    matrices[1][0, [0, 1]], matrices[1][0, [64, 65]] = big, -big
    cotangents = [np.zeros((128, width), np.float32) for width in CHAINED]
    for cotangent in cotangents[1:]:
        cotangent[:, [0, 1, 64, 65]] = big
    return (np.zeros((128, FEATURES), dtype), tuple(each.astype(dtype) for each in matrices),
            tuple(each.astype(dtype) for each in cotangents))


def _chained_gradient(interpret: bool = False) -> Callable:
    """The rows' gradient through three projections side by side (`gemm.gemm`'s `beside`), the template
    named: a function of the rows, the query's, key's and value's matrices and their cotangents, its input
    gradient's products one chained call where a launch takes it (`gemm._one_launch`, `_chained_taken`),
    interpreted where `interpret` asks on a CPU."""
    def chained(x, wq, wk, wv, dq, dk, dv):
        def projected(x):
            return gemm.gemm(x, wq, beside=(wk, wv), out_dtype=x.dtype, interpret_on_cpu=interpret)

        with generation.chosen(grouped_matmul="mosaic"):
            return jax.vjp(projected, x)[1]((dq, dk, dv))[0]

    return chained


def _plain_gradient(x, wq, wk, wv, dq, dk, dv):
    """`_chained_gradient`'s plain program: each projection apart, its cotangent through its matrix, their
    products added in its reverse pass's order (XLA's)."""
    def projected(x):
        return tuple(jnp.dot(x, w, preferred_element_type=jnp.promote_types(x.dtype, jnp.float32))
                     .astype(x.dtype) for w in (wq, wk, wv))

    return jax.vjp(projected, x)[1]((dq, dk, dv))[0]


@contextlib.contextmanager
def _chained_taken() -> Iterator[None]:
    """Within the block, the launches its gradient arms bind recorded (`tuning.seen`: where a GPU tier's
    prepare traces them in its target's scope, and where its run replays the records of the programs it
    prepared, `truths_ahead.CompiledTruths.ran`), their input gradient's products one chained call
    (`gemm._one_launch`), not the products apart, on leaving it (base review d0bc7dee6 F2; 68e1f210f F1: a
    trace before the prepare's scope saw no GPU generation)."""
    from dew.nn.kernels import tuning

    with tuning.seen() as launches:
        yield
    # Every input-gradient launch one of the three products with the other two chained to it: a product
    # run apart beside a chain of two passed where its sum cancelled (base review 5028c8148 F1).
    transposed = [bound for _, bound in launches.taken if bound.arguments.get("trans_rhs")]
    assert transposed and all(len(bound.arguments.get("chained") or ()) == 2 for bound in transposed), (
        launches.taken)


CHAINED_CASES = ("projections", "unequal scales", "cancelling")
"""The chained input gradient's check 1 draws (`a_chained_input_gradient_holds_check_1`): q, k and v's
cotangents at one scale; at scales 2^10 apart, so one product's sum dwarfs another's in the one accumulator;
and k's product cancelling q's across their boundary (k's matrix q's negated, its cotangent q's plus 2^-6 of
noise), so the joined sum carries q's partials into k's cancellation (base review c3b97e20f F1, F7)."""


def _chained_draw(case: str):
    """`a_chained_input_gradient_holds_check_1`'s draw of `case` (`CHAINED_CASES`): the rows, three matrices
    (hidden 256, widths 256, 256 and 128) and their cotangents, bf16, NumPy's (a GPU tier's prepare computes
    on no device)."""
    rows, hidden, widths = 256, 256, (256, 256, 128)

    def draw(seed):
        rng = np.random.default_rng([seed, CHAINED_CASES.index(case)])
        x = rng.normal(size=(rows, hidden))
        matrices = [rng.normal(size=(hidden, width)) / np.sqrt(hidden) for width in widths]
        cotangents = [rng.normal(size=(rows, width)) for width in widths]
        if case == "unequal scales":
            cotangents = [each * scale for each, scale in zip(cotangents, (1.0, 2.0 ** 10, 2.0 ** -10),
                                                               strict=True)]
        if case == "cancelling":
            matrices[1] = -matrices[0]
            cotangents[1] = cotangents[0] + 2.0 ** -6 * cotangents[1]
        return tuple(each.astype(jnp.bfloat16) for each in (x, *matrices, *cotangents))

    return draw


def a_chained_input_gradient_holds_check_1(case: str, interpret: bool = False) -> None:
    """Three projections side by side (`gemm.gemm`'s `beside`), their input gradient one launch over their
    joined depths (`gemm._one_launch`, the template named), on par from float64 with the plain program's
    (each projection apart, its cotangent through its matrix, their products added in its reverse pass's
    order: XLA's), over `CHAINED_CASES`' draws (check 1; base review c3b97e20f F7): on this GPU (the GEMM
    flows' Hopper and Blackwell cells), or interpreted (`interpret`). The chained launch's contract is a
    GEMM's over the joined depth (`chained.guarded_walk`), held here."""
    with _chained_taken():
        assert_as_exact(jax.jit(_chained_gradient(interpret)), jax.jit(_plain_gradient), _chained_draw(case),
                        f"a chained input gradient: {case}", seeds=SEEDS, truth_fn=_plain_gradient)


@pytest.mark.parametrize("case", CHAINED_CASES)
@pytest.mark.parametrize("generation_name", generations("sm90", "sm100"))
def test_a_chained_input_gradient_holds_check_1(generation_name, case):
    """`a_chained_input_gradient_holds_check_1` interpreted as a Hopper's and a Blackwell's walk runs it."""
    with generation.assumed(generation_name):
        a_chained_input_gradient_holds_check_1(case, interpret=True)


def a_chained_input_gradient_is_as_finite_as_its_products_sum(interpret: bool = False) -> None:
    """Three projections side by side, their input gradient one launch over their joined depths
    (`_chained_gradient`, `_chained_taken`): where one running sum over the joined contraction passes
    float32's largest while each product's own sum and the total lie within it (`_chained_overflow`), the
    launch flags the tile and walks it again with its sums bounded (`mosaic_gpu.guarded_walk`): the gradient
    is the plain program's, 2^127 at feature 0 and zero elsewhere, where one running sum gave infinities
    (row 8282). On this GPU (the GEMM flows' Hopper and Blackwell cells, base review d0bc7dee6 F2), each
    arm compiled ahead (`ahead_arm`), or interpreted (`interpret`)."""
    rows, matrices, cotangents = _chained_overflow()
    operands = (rows, *matrices, *cotangents)
    arm = ahead_arm(jax.jit(_chained_gradient(interpret)), "a chained input gradient's overflow")
    with _chained_taken():
        got = arm(*operands)
    plain = ahead_arm(jax.jit(_plain_gradient), "a chained input gradient's overflow, apart")(*operands)
    expected = np.zeros((128, FEATURES), np.float32)
    expected[:, 0] = 2.0 ** 127
    np.testing.assert_array_equal(np.asarray(plain, np.float32), expected)
    np.testing.assert_array_equal(np.asarray(got, np.float32), expected)


@pytest.mark.parametrize("generation_name", generations("sm90", "sm100"))
def test_a_chained_input_gradient_is_as_finite_as_its_products_sum(generation_name):
    """`a_chained_input_gradient_is_as_finite_as_its_products_sum` interpreted as a Hopper's and a
    Blackwell's walk runs it."""
    with generation.assumed(generation_name):
        a_chained_input_gradient_is_as_finite_as_its_products_sum(interpret=True)


WALKED = tiling.Tiles(128, 256, 64, 2, 1, warpgroups=2, family="wgmma")
"""The chained launch's tile the protocol checker walks: 128 x 256 x 64 over a two-stage ring."""


def _chained_walk(*, launch: int = 0, flagged: bool = True) -> tuple:
    """The programs (`ring_protocol.programs`) of the overflow witness's chained launch on an H100
    (`mosaic_gpu.gmm`'s `chained`: three products, each step one product's) at `WALKED`, traced as the GPU
    builds it: its `launch`-th, the flagged walk (0) or the bounded one (1), the bounded walk's flags given
    (its tile's every share flagged where `flagged`, none elsewhere: the tiles it walks)."""
    from ring_protocol import programs

    _, matrices, cotangents = _chained_overflow(jnp.bfloat16)
    products = [(cotangent, matrix[None]) for cotangent, matrix in zip(cotangents[::-1], matrices[::-1],
                                                                       strict=True)]
    chain, operands = _chained_call(products, None, jnp.bfloat16, WALKED, interpret=False)

    def chained(*operands):
        with generation.assumed("sm90"):
            return chain(*operands)

    def flags(index, aval):
        # The flags, int32 `[row tiles, column tiles, row shares]` (`chained.flags_declared`).
        del index
        if aval.dtype != jnp.int32 or aval.ndim != 3:
            return None
        return np.full(aval.shape, int(flagged), np.int32)

    return programs(chained, *operands, threads=WALKED.warpgroups + 1, launch=launch, known=flags)


def test_a_chained_launchs_walk_holds_every_property_in_every_interleaving():
    """The chained input gradient's launch (`_chained_walk`'s first): every interleaving of its warpgroups
    and its loads' landings explored (`ring_protocol.explore`), each slot given back once the products
    reading it retire. Each product's issue counted as a step's group though one runs a step, the walk
    waited for three groups where one was in flight, and step 2 refilled step 0's slot while step 0's product
    read it (base review of f86cb5b03, F2)."""
    from ring_protocol import explore

    assert explore(*_chained_walk()) > 0


@pytest.mark.parametrize("walked", ["flagged", "clear", "overcounted"])
def test_a_chained_launchs_bounded_walk_holds_every_property_in_every_interleaving(walked):
    """The chained launch's second, bounded walk (`chained.guarded_walk`, `tiling.GUARDED_GMM`;
    `_chained_walk`'s second), its flags given: its tile flagged, every interleaving of its warpgroups and its
    loads' landings explored, each slot given back once the products reading it retire; clear, its consumers
    issue no product. Its steps each issue one product's groups (`Issuer.issued_one_of`); counted as every
    product's, its waits leave a step's slot in flight past its release, which the explorer finds (base review
    2e4651b2e F2, row 8680; 3150e8dfc F5)."""
    from ring_protocol import Violation, explore

    from dew.nn.kernels.templates import mosaic

    original = mosaic.Issuer.issued_one_of

    def every_case_counted(self, which, accumulators, issue):
        before = mosaic.commits()
        issued = original(self, which, accumulators, issue)
        mosaic._COMMITTED.set(before + (mosaic.commits() - before) * len(accumulators))
        return issued

    with pytest.MonkeyPatch.context() as patch:
        if walked == "overcounted":
            patch.setattr(mosaic.Issuer, "issued_one_of", every_case_counted)
        threads, arrivals = _chained_walk(launch=1, flagged=walked != "clear")
    products = sum(event.kind == "product" for program in threads for event in program)
    if walked == "clear":
        # No product, and no producer waits for a release its idle consumers never send (base review
        # 156a4595a F3).
        assert products == 0, products
        assert explore(threads, arrivals) > 0
        return
    assert products > 0
    if walked == "overcounted":
        with pytest.raises(Violation):
            explore(threads, arrivals)
        return
    assert explore(threads, arrivals) > 0


def test_a_ping_pong_chained_launch_flags_its_tiles():
    """The overflow witness's three products through one chained launch on a Hopper's ping-pong tile
    (`Tiles.pingpong`: each consumer a whole tile in turn), interpreted: its one finishing consumer writes its
    tile's one flag (`Tiles.sharing`), the flagged tile walked again, the plain program's 2^127 at feature 0
    (base review c3b97e20f F5: flags written by consumer left a tile's other slot unwritten)."""
    _, matrices, cotangents = _chained_overflow(jnp.float32)
    products = [(np.asarray(d), np.asarray(w)[None]) for d, w in zip(cotangents[::-1], matrices[::-1],
                                                                       strict=True)]
    pingpong = tiling.Tiles(64, 256, 64, 3, 1, warpgroups=2, family="wgmma", pingpong=True)
    got = _chained_launch(products, None, jnp.bfloat16, tiles=pingpong)
    expected = _plain_sum(products, None, jnp.bfloat16)
    assert expected[0, 0] == 2.0 ** 127 and not expected[:, 1:].any()
    np.testing.assert_array_equal(got, expected)


def _chained_bound(depths: tuple, *, groups: int = 2, features: int = FEATURES, rows: int = 256):
    """A grouped bf16 input gradient's chained call (`gemm._one_launch`'s chained binding) of products of
    `depths`, in walk order: each its cotangent `[rows, depth]` by its matrix `[groups, features, depth]`,
    transposed."""
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    first, *rest = depths
    args = (bf16((rows, first)), bf16((groups, features, first)), jax.ShapeDtypeStruct((groups,), jnp.int32))
    chained = tuple((bf16((rows, depth)), bf16((groups, features, depth))) for depth in rest)
    return args, {"trans_rhs": True, "out_dtype": "bfloat16", "interpret": False, "chained": chained}


@pytest.mark.parametrize("depths", [(128, 128, 128), (80, 128, 256), (128, 0, 128), (64,) * 9, (64,) * 65],
                         ids=["three", "unaligned", "empty", "nine", "sixty-five"])
def test_a_chained_call_is_derived_as_it_is_launched(depths):
    """On an assumed H100, a chained call's tile is one its launch holds legal or none (base review
    c3b97e20f): the derivation and the launch judge the call as bound (F3: a first depth of 80 was derived
    on padded to 128 and refused as bound), a tile holds its replay's narrower tile too (F2: nine products
    narrowed a 64-column tile to four columns, 65 to none, dividing by zero), and a product of no depth takes
    no chain (F4: its operand has no extent a TMA descriptor takes). Three whole products chain."""
    from dew.nn.kernels import grouped_matmul

    launch = grouped_matmul.MosaicGmm()
    args, options = _chained_bound(depths)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        ba, target = launch.bind(*args, **options), generation.gpu_target()
        tile = launch.derived(ba, target)
        narrow = tiling.Tiles(128, 64, 64, 2, 1, warpgroups=2, family='wgmma')
        assert tile is None or launch.legal(ba, tile, target), tile
        if depths == (128, 128, 128):
            assert tile is not None and tile.block_n % chained.narrowing(len(depths)) == 0, tile
        if depths in ((128, 0, 128), (64,) * 9, (64,) * 65):
            assert not launch.legal(ba, narrow, target)


@pytest.mark.parametrize("generation_name", generations("sm90"))
def test_a_chained_input_gradient_of_non_finite_cotangents_is_the_plain_programs(generation_name):
    """The chained input gradient where a cotangent holds an infinity and a NaN (each its own row): the launch
    flags their tiles, and the bounded walk gives the plain program's infinities and NaNs there, every other
    entry the products' exact sum (small integers) rounded once, as one accumulator rounds it."""
    rng = np.random.default_rng(8282)
    rows = np.zeros((256, FEATURES), jnp.bfloat16)
    matrices = [rng.integers(-3, 4, size=(FEATURES, width)).astype(np.float32) for width in CHAINED]
    cotangents = [rng.integers(-3, 4, size=(256, width)).astype(np.float32) for width in CHAINED]
    exact = sum(cotangent.astype(np.float64) @ matrix.T.astype(np.float64)
                for cotangent, matrix in zip(cotangents, matrices, strict=True))
    cotangents[0][5, 3], cotangents[2][200, 7] = np.inf, np.nan
    operands = (rows, *(each.astype(jnp.bfloat16) for each in (*matrices, *cotangents)))
    with generation.assumed(generation_name):
        with _chained_taken():
            got = np.asarray(_chained_gradient(interpret=True)(*operands), np.float32)
        plain = np.asarray(_plain_gradient(*operands), np.float32)
    finite = np.isfinite(plain)
    assert not finite[5].all() and not finite[200].any() and finite[[0, 6, 199, 201]].all()
    np.testing.assert_array_equal(np.isnan(got), np.isnan(plain))
    np.testing.assert_array_equal(got[~finite], plain[~finite])
    rounded = np.asarray(jnp.asarray(exact, jnp.float32).astype(jnp.bfloat16), np.float32)
    np.testing.assert_array_equal(got[finite], rounded[finite])


def _chained_call(products: list, sizes, dtype, tiles=None, *, interpret=grouped_launch.INTERPRETED,
                  multiprocessors: int = 4) -> tuple[Callable, tuple]:
    """`products`' `(cotangent, matrix)` pairs (NumPy's, a matrix `[groups, features, width]`, or abstract:
    `jax.ShapeDtypeStruct`s, given as they are) as one chained launch (`mosaic_gpu.gmm`'s `chained`, the
    first the call's own) over `multiprocessors`, interpreted as `interpret` says, at `tiles` (the template's
    128 x 256 tile where None: three products' replay a quarter of it, whole column units,
    `chained.replay_tile`), its groups' rows `sizes` (None: dense), in `dtype`: the launch, a function of its
    operands, and those operands, each pair's in turn; the one fixture of a chained launch its checks vary
    (base review 3150e8dfc F3)."""
    tiles = tiles or tiling.Tiles(128, 256, 64, 2, 1, warpgroups=2, family="wgmma")
    group_sizes = None if sizes is None else jnp.asarray(sizes, jnp.int32)

    def launch(*operands):
        (d, w), *rest = zip(operands[0::2], operands[1::2], strict=True)
        return mosaic_gpu.gmm(d, w, group_sizes, tiles=tiles, transpose_rhs=True, out_dtype=dtype,
                              multiprocessors=multiprocessors, interpret=interpret, dense=sizes is None,
                              chained=tuple(rest))

    return launch, tuple(each if isinstance(each, jax.ShapeDtypeStruct) else jnp.asarray(each, dtype)
                         for pair in products for each in pair)


def _chained_launch(products: list, sizes, dtype, generation_name: str = "sm90", tiles=None) -> np.ndarray:
    """`_chained_call`'s launch run on its operands as `generation_name` runs it, its output."""
    launch, operands = _chained_call(products, sizes, dtype, tiles)
    with generation.assumed(generation_name):
        return np.asarray(launch(*operands), np.float32)


def _plain_sum(products: list, sizes, dtype) -> np.ndarray:
    """The plain program's chained sum: each product's float32 sum over its own depth (its group's matrix
    at each row), added in order, rounded once to `dtype`."""
    rows = products[0][0].shape[0]
    group_of = np.zeros(rows, int) if sizes is None else np.repeat(np.arange(len(sizes)), sizes)
    with np.errstate(invalid="ignore", over="ignore"):
        total = None
        for d, w in products:
            d32 = np.asarray(jnp.asarray(d, dtype), np.float32)
            w32 = np.asarray(jnp.asarray(w, dtype), np.float32)
            product = np.einsum("rw,rfw->rf", d32, w32[group_of], dtype=np.float32)
            total = product if total is None else total + product
    return np.asarray(jnp.asarray(total).astype(dtype), np.float32)


@pytest.mark.parametrize("sizes", [(64, 64), (128,)], ids=["two-groups-one-tile", "one-group-unused-entry"])
def test_a_flagged_tile_is_its_own_whatever_shares_its_rows(sizes):
    """A grouped chained launch whose first group's tile overflows its one running sum (`_chained_overflow`'s
    products on the first 64 rows, zeros elsewhere) where another tile of the same rows stays finite (a
    second group's at the same row tile, or the table's unused entry finishing zeros): each tile's flag is its
    table entry's own (`chained.flag_row`), so the overflowing tile is walked again and gives the plain sum,
    2^127 at feature 0; one flag a row tile let the finite tile's clear it (237e8e558 F1)."""
    _, matrices, cotangents = _chained_overflow(jnp.float32)
    products = []
    for d, w in zip(cotangents, matrices, strict=True):
        rows = np.zeros((128, w.shape[1]), np.float32)
        rows[:64] = np.asarray(d)[:64]
        products.append((rows, np.stack([np.asarray(w)] * len(sizes))))
    products = products[::-1]  # the transpose order: the value's first
    got = _chained_launch(products, sizes, jnp.float32)
    expected = _plain_sum(products, sizes, jnp.float32)
    assert expected[0, 0] == 2.0 ** 127 and np.isfinite(expected).all()
    np.testing.assert_array_equal(got, expected)


def test_a_tiles_flag_reads_only_the_rows_it_stores():
    """A grouped fp16 chained launch of three products, its groups [1, 127] in one row tile: the first
    product's cotangent 2^15 at a column on group 1's rows, group 0's matrix 2^15 there, so group 0's table
    entry multiplies rows it never stores into fp16's infinity, while every row it stores, and group 1's, is
    finite (group 1's matrix one there). The launch's per-tile flags (its int32 output) are all clear, no tile
    walked again, and its gradient is the plain program's; flagging the discarded rows set group 0's (base
    review d0bc7dee6 F3)."""
    from kernel_outputs import launched_outputs

    products = [(np.zeros((128, width), np.float32), np.zeros((2, FEATURES, width), np.float32))
                for width in CHAINED]
    products[0][0][1:, 0] = 2.0 ** 15
    products[0][1][0, 0, 0], products[0][1][1, 0, 0] = 2.0 ** 15, 1.0
    sizes = (1, 127)
    launch, operands = _chained_call(products, sizes, jnp.float16)
    with generation.assumed("sm90"):
        found = launched_outputs(launch, *operands)
    flags = [output for name, outputs in found if tiling.GUARDED_GMM not in name for output in outputs
             if output.dtype == np.int32]
    assert flags and not any(np.asarray(each).any() for each in flags), flags
    expected = _plain_sum(products, sizes, jnp.float16)
    assert expected[1, 0] == 2.0 ** 15 and np.isfinite(expected).all()
    np.testing.assert_array_equal(_chained_launch(products, sizes, jnp.float16), expected)


def test_a_flagged_tiles_walk_keeps_its_operands_least_values():
    """A dense fp16 chained launch of three products whose tile is flagged by an infinite cotangent entry
    (row 0's), another row of it the least fp16 value (2^-24) times 2^15: the bounded walk sums each product
    over its own depth on its operands as given, so that row's gradient is 2^-9, as the plain program's; rows
    scaled down in their own dtype lost it to zero (237e8e558 F2)."""
    widths = (256, 128, 128)
    products = [(np.zeros((128, width), np.float32), np.zeros((1, FEATURES, width), np.float32))
                for width in widths]
    products[0][1][0, 0, 0] = 2.0 ** 15
    products[0][0][1, 0] = 2.0 ** -24
    products[0][0][0, 0] = np.inf
    got = _chained_launch(products, None, jnp.float16)
    expected = _plain_sum(products, None, jnp.float16)
    assert expected[1, 0] == 2.0 ** -9 and not np.isfinite(expected[0]).all()
    np.testing.assert_array_equal(got, expected)


def test_a_chained_sum_overflowing_its_output_dtype_is_walked_again():
    """A dense fp16 chained launch whose products' own sums are 65504, then 2^30 and -2^30 in two steps (0),
    then 0: the plain program's total, 65504, is fp16's largest, where the joined fp32 sum ends at a finite
    65536 (2^30 + 65504 rounds to 2^30 + 65536), an fp16 infinity once stored. The launch flags its tile by
    the value it stores, not its fp32 sum, so the bounded walk gives the plain program's 65504 (base review
    b1ab251ac F1)."""
    products = [(np.zeros((128, width), np.float32), np.zeros((1, FEATURES, width), np.float32))
                for width in (64, 128, 64)]
    products[0][0][0, 0], products[0][1][0, 0, 0] = 65504.0, 1.0
    products[1][0][0, [0, 64]] = 2.0 ** 15
    products[1][1][0, 0, 0], products[1][1][0, 0, 64] = 2.0 ** 15, -2.0 ** 15
    got = _chained_launch(products, None, jnp.float16)
    expected = _plain_sum(products, None, jnp.float16)
    assert expected[0, 0] == 65504.0 and np.isfinite(expected).all()
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("products", [9, 65])
def test_a_direct_chained_launch_whose_replay_no_tile_holds_is_refused(products):
    """`mosaic_gpu.gmm` called with chained products directly, at a legal 128 x 64 Hopper tile: nine 64-deep
    products narrow its replay to four columns, 65 to none, no whole column unit of the family, so the launch
    refuses the call (`chained.replay_tile`, the one replay plan selection reads too) rather than lower a
    replay WGMMA cannot run (base review f86cb5b03 F1)."""
    tiles = tiling.Tiles(128, 64, 64, 2, 1, warpgroups=2, family="wgmma")
    shaped = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    launch, operands = _chained_call([(shaped((256, 64)), shaped((1, 128, 64))) for _ in range(products)],
                                     None, jnp.bfloat16, tiles, interpret=False, multiprocessors=132)
    with generation.assumed("sm90"), pytest.raises(ValueError):
        jax.eval_shape(launch, *operands)


def test_a_launch_pinned_at_a_tile_no_chain_holds_keeps_its_products_apart():
    """A grouped bf16 input gradient of three projections side by side (rows `[256, 256]`, groups
    `[128, 128]`, matrices `[2, 256, 128]`), the template pinned at a legal 128 x 64 Hopper tile whose
    chain's replay would narrow it to 16 columns: the products run apart at the pin, no chained launch, the
    gradient the exact sum (base review 2e4651b2e F1: the chain was chosen at a derived tile, and its call
    refused the pin, `Unpinnable`)."""
    from dew.nn.kernels import grouped_matmul, tuning

    rng = np.random.default_rng(21)
    sizes = jnp.asarray([128, 128], jnp.int32)
    x = jnp.asarray(rng.integers(-2, 3, size=(256, 256)), jnp.bfloat16)
    matrices = [jnp.asarray(rng.integers(-2, 3, size=(2, 256, 128)), jnp.bfloat16) for _ in range(3)]
    cotangents = [jnp.asarray(rng.integers(-2, 3, size=(256, 128)), jnp.bfloat16) for _ in range(3)]

    def projected(x):
        return gemm.gemm(x, matrices[0], sizes, beside=tuple(matrices[1:]), out_dtype=jnp.bfloat16,
                         interpret_on_cpu=True)

    pin = tiling.Tiles(128, 64, 64, 2, 1, warpgroups=2, family="wgmma")
    with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"),
          tuning.pinned({grouped_matmul.MosaicGmm: pin}), tuning.seen() as launches):
        got = jax.vjp(projected, x)[1](tuple(cotangents))[0]
    assert not any(bound.arguments.get("chained") for _, bound in launches.taken), launches.taken
    group_of = np.repeat([0, 1], 128)
    expected = sum(np.einsum("rw,rfw->rf", np.asarray(d, np.float32), np.asarray(w, np.float32)[group_of])
                   for d, w in zip(cotangents, matrices, strict=True))
    np.testing.assert_array_equal(np.asarray(got, np.float32), expected)


def test_a_chained_walks_steps_read_each_from_its_own_product():
    """The chained products' step views (`mosaic_gpu.ChainedRefs.view`) at each product's boundary steps,
    `starts[p] - 1` and `starts[p]`, from a balanced tail's first step: the step reads the product it falls
    in (`which`) at that product's own step, its last before the next product's first."""
    walk = chained.Chained(starts=(0, 4, 6), depths=(256, 128, 128))
    first = 1
    view = chained.ChainedRefs(walk, ("q", "k", "v")).view(lambda gmem: gmem, lambda j: (0, first + j),
                                                                   first)
    read = {}
    for step in (2, 3, 4, 5):  # global steps 3 and 4, then 5 and 6
        which = int(view.which(step))
        ref, index_map = view.sources[which]
        read[first + step] = (ref, index_map(step))
    assert read == {3: ("q", (0, 3)), 4: ("k", (0, 0)), 5: ("k", (0, 1)), 6: ("v", (0, 0))}


@lowers("sm90")
@pytest.mark.parametrize("generation_name", ["sm90", "sm100"])
def test_a_chained_input_gradient_lowers_as_one_call(generation_name):
    """Three projections side by side differentiated for their rows on an H100 and a B200, the template
    named: their input gradient is one bound call (`gemm`'s `chained`), its flagged walk and its bounded walk
    over the flagged tiles lowering through Mosaic GPU for CUDA, where it was three, each after the first
    reading and writing the fp32 sum (row 8282)."""
    from dew.nn.kernels import tuning

    shapes = (jax.ShapeDtypeStruct((256, FEATURES), jnp.bfloat16),
              *(jax.ShapeDtypeStruct((FEATURES, width), jnp.bfloat16) for width in CHAINED))

    def loss(x, *matrices):
        outs = gemm.gemm(x, matrices[0], beside=matrices[1:], out_dtype=jnp.bfloat16)
        return sum(jnp.sum(out.astype(jnp.float32)) for out in outs)

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), generation_name)
    with (generation.assumed(generation_name), jax.sharding.use_abstract_mesh(mesh),
          jax.default_matmul_precision("default"), generation.chosen(grouped_matmul="mosaic"),
          tuning.seen() as launches):
        jax.jit(jax.grad(loss)).trace(*shapes).lower(lowering_platforms=("cuda",))
    input_gradients = [bound for _, bound in launches.taken
                       if bound.arguments.get("trans_rhs") and bound.arguments.get("kernel") is not None]
    assert len(input_gradients) == 1 and input_gradients[0].arguments.get("chained"), input_gradients


@COMPILES_ON_CPU
@pytest.mark.parametrize("dense", [True, False], ids=["dense", "grouped"])
def test_a_chained_input_gradient_compiles_for_an_h100_within_its_resources(dense):
    """An 8B decoder's q, k and v cotangents through their matrices (widths 4096, 1024, 1024 into 4096
    features) chained in one launch on an H100's rings, its flagged walk and its bounded walk over the flagged
    tiles (the rows' prologue scaling them, the epilogue scaling the sums back: `chained.guarded_walk`),
    compiled by Mosaic for a compile-only H100 through ptxas: each within its resources, no register
    spilled."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    # The tile the derivation takes for three products: its replay's a quarter of its columns, whole column
    # units (`grouped_matmul.MosaicGmm._replays`).
    tiles = tiling.Tiles(128, 256, 64, 4, 1, warpgroups=2, family="wgmma")
    groups = 1 if dense else 3
    shaped = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    widths = (4096, 1024, 1024)
    operands = (*(shaped((4096, width)) for width in widths),
                *(shaped((groups, 4096, width)) for width in widths),
                jax.ShapeDtypeStruct((groups,), jnp.int32))

    def launch(q, k, v, wq, wk, wv, sizes):
        return mosaic_gpu.gmm(v, wv, None if dense else sizes, tiles=tiles, transpose_rhs=True,
                              out_dtype=jnp.bfloat16, multiprocessors=132, dense=dense,
                              chained=((k, wk), (q, wq)))

    case = KernelCase(f"chained input gradient {'dense' if dense else 'grouped'}", lambda: (launch, operands))
    with generation.assumed("sm90"):
        found = compile_mosaic_kernels([case], "sm90")
    assert found and all(len(record.get("kernels") or ()) >= 2 for record in found), found
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))


def test_joined_depths_walks_lora_in_place_without_concatenating_base():
    """`low_rank.low_rank` / `gemm.joined_depths` walks the base `(x, W)` and LoRA `(t, B)` in place via
    `mosaic_gpu.gmm`'s `chained` launch on an assumed H100 without concatenating `x` or `W`, and forward +
    VJP hold check 1 against float64 truth (row 1304)."""
    from dew.nn.kernels.low_rank import low_rank
    from dew.nn.programs import equations

    rows, depth, width, rank = 128, 256, 128, 16

    def draw(seed):
        rng = np.random.default_rng(1304 + seed)
        x = jnp.asarray(rng.standard_normal((rows, depth)) / np.sqrt(depth), jnp.bfloat16)
        w = jnp.asarray(rng.standard_normal((depth, width)) / np.sqrt(depth), jnp.bfloat16)
        a = jnp.asarray(rng.standard_normal((depth, rank)) / np.sqrt(depth), jnp.bfloat16)
        b = jnp.asarray(rng.standard_normal((rank, width)) / np.sqrt(rank), jnp.bfloat16)
        cot = jnp.asarray(rng.standard_normal((rows, width)), jnp.bfloat16)
        return x, w, a, b, cot

    x0, w0, a0, b0, _ = draw(0)
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        traced = jax.make_jaxpr(
            lambda x, w, a, b: low_rank(x, w, a, b, 0.5, out_dtype=jnp.bfloat16, interpret_on_cpu=True)
        )(x0, w0, a0, b0)
        assert not any(
            eqn.primitive.name == "concatenate"
            and any(getattr(v.aval, "shape", None) in ((rows, depth), (depth, width)) for v in eqn.invars)
            for eqn in equations(traced.jaxpr)
        )

    def dew_vjp(x, w, a, b, cot):
        with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
            out, pull = jax.vjp(
                lambda xi, wi, ai, bi: low_rank(xi, wi, ai, bi, 0.5, out_dtype=jnp.bfloat16,
                                                interpret_on_cpu=True),
                x, w, a, b)
            return out, pull(cot)
    def ref_vjp(x, w, a, b, cot, *, dtype=jnp.float32):
        def fwd(xi, wi, ai, bi):
            t = (0.5 * jax.lax.dot(xi, ai, preferred_element_type=dtype)).astype(
                jnp.bfloat16 if dtype == jnp.float32 else dtype)
            y = (jax.lax.dot(xi, wi, preferred_element_type=dtype)
                 + jax.lax.dot(t, bi, preferred_element_type=dtype))
            return y.astype(jnp.bfloat16 if dtype == jnp.float32 else dtype)
        if dtype == jnp.float64:
            out, pull = jax.vjp(fwd, x.astype(dtype), w.astype(dtype), a.astype(dtype), b.astype(dtype))
            return out, pull(cot.astype(dtype))
        out, pull = jax.vjp(fwd, x, w, a, b)
        return out, pull(cot)

    assert_as_exact(dew_vjp, jax.jit(ref_vjp), draw, "zero-copy chained low_rank", seeds=8,
                    truth_fn=functools.partial(ref_vjp, dtype=jnp.float64))


def test_grouped_low_rank_walks_in_place_and_holds_check_1_and_compiles():
    """Grouped MoE `low_rank.low_rank` (`group_sizes` given, row 1304) walks base experts `(x, W)` and LoRA
    `(t, B)` in place via `mosaic_gpu.gmm`'s grouped `chained` launch without concatenating base weights,
    holds check 1 forward + VJP (full and frozen-base) with an empty expert group, and compiles spill-free on
    `sm90`."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels

    from dew.nn.kernels.low_rank import low_rank
    from dew.nn.programs import equations

    groups, rows, depth, width, rank = 3, 128, 256, 128, 16
    sizes = jnp.asarray([48, 0, 80], jnp.int32)

    def draw(seed):
        rng = np.random.default_rng(13040 + seed)
        x = jnp.asarray(rng.standard_normal((rows, depth)) / np.sqrt(depth), jnp.bfloat16)
        w = jnp.asarray(rng.standard_normal((groups, depth, width)) / np.sqrt(depth), jnp.bfloat16)
        a = jnp.asarray(rng.standard_normal((groups, depth, rank)) / np.sqrt(depth), jnp.bfloat16)
        b = jnp.asarray(rng.standard_normal((groups, rank, width)) / np.sqrt(rank), jnp.bfloat16)
        cot = jnp.asarray(rng.standard_normal((rows, width)), jnp.bfloat16)
        return x, w, a, b, cot

    from jax.core import ShapedArray

    x0, w0, a0, b0, c0 = draw(0)
    with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
        traced = jax.make_jaxpr(
            lambda x, w, a, b: low_rank(x, w, a, b, 0.5, sizes, out_dtype=jnp.bfloat16,
                                        interpret_on_cpu=True)
        )(x0, w0, a0, b0)
        assert not any(
            eqn.primitive.name == "concatenate"
            and any(isinstance(v.aval, ShapedArray)
                    and v.aval.shape in ((rows, depth), (groups, depth, width))
                    for v in eqn.invars)
            for eqn in equations(traced.jaxpr)
        )
        # Frozen-kernel VJP (differentiating x, a, b only) produces valid adapter gradients.
        _, pull_frozen = jax.vjp(
            lambda xi, ai, bi: low_rank(xi, w0, ai, bi, 0.5, sizes, out_dtype=jnp.bfloat16,
                                        interpret_on_cpu=True),
            x0, a0, b0)
        dx_f, da_f, db_f = pull_frozen(c0)
        assert dx_f.shape == x0.shape and da_f.shape == a0.shape and db_f.shape == b0.shape

    def dew_vjp(x, w, a, b, cot):
        with generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"):
            out, pull = jax.vjp(
                lambda xi, wi, ai, bi: low_rank(xi, wi, ai, bi, 0.5, sizes, out_dtype=jnp.bfloat16,
                                                interpret_on_cpu=True),
                x, w, a, b)
            return out, pull(cot)

    def ref_vjp(x, w, a, b, cot, *, dtype=jnp.float32):
        def fwd(xi, wi, ai, bi):
            t = (0.5 * jax.lax.ragged_dot(xi, ai, sizes, preferred_element_type=dtype)).astype(
                jnp.bfloat16 if dtype == jnp.float32 else dtype)
            y = (jax.lax.ragged_dot(xi, wi, sizes, preferred_element_type=dtype)
                 + jax.lax.ragged_dot(t, bi, sizes, preferred_element_type=dtype))
            return y.astype(jnp.bfloat16 if dtype == jnp.float32 else dtype)

        if dtype == jnp.float64:
            out, pull = jax.vjp(fwd, x.astype(dtype), w.astype(dtype), a.astype(dtype), b.astype(dtype))
            return out, pull(cot.astype(dtype))
        out, pull = jax.vjp(fwd, x, w, a, b)
        return out, pull(cot)

    assert_as_exact(dew_vjp, jax.jit(ref_vjp), draw, "grouped zero-copy chained low_rank", seeds=8,
                    truth_fn=functools.partial(ref_vjp, dtype=jnp.float64))

    if generation.compile_only_gpus("sm90"):
        bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
        specs = (bf16((rows, depth)), bf16((groups, depth, width)), bf16((groups, depth, rank)),
                 bf16((groups, rank, width)), jax.ShapeDtypeStruct((groups,), jnp.int32))

        def compiled_moe_lora(x, w, a, b, s):
            with generation.chosen(grouped_matmul="mosaic"):
                return low_rank(x, w, a, b, 0.5, s, out_dtype=jnp.bfloat16)

        with generation.assumed("sm90"):
            records = compile_mosaic_kernels(
                [KernelCase("grouped_low_rank_sm90", lambda: (compiled_moe_lora, specs))], "sm90")
        for rec in records:
            assert not rec.get("error"), rec
            assert rec["kernels"], rec
            assert all(k.get("spilled") == 0 for k in rec["kernels"]), rec


@pytest.mark.parametrize("grouped", [False, True])
def test_an_adapted_products_gradient_keeps_a_wide_cotangent(grouped):
    """An adapted product of fp16 rows with an fp32 output pulls its cotangent back at the cotangent's width:
    `2**16`, past fp16's largest, against a kernel of `2**-10` gives the rows exactly 64, not the inf of a
    cotangent rounded to fp16 first, dense and grouped (row 1304)."""
    from dew.nn.kernels.low_rank import low_rank

    half = functools.partial(jnp.zeros, dtype=jnp.float16)
    x = jnp.ones((8, 16), jnp.float16)
    kernel = jnp.full((2, 16, 8) if grouped else (16, 8), 2.0 ** -10, jnp.float16)
    a, b = half((2, 16, 4) if grouped else (16, 4)), half((2, 4, 8) if grouped else (4, 8))
    sizes = jnp.asarray([4, 4], jnp.int32) if grouped else None
    _, pull = jax.vjp(lambda rows: low_rank(rows, kernel, a, b, 1.0, sizes, out_dtype=jnp.float32), x)
    (d_x,) = pull(jnp.zeros((8, 8), jnp.float32).at[0, 0].set(2.0 ** 16))
    want = np.zeros((8, 16), np.float16)
    want[0] = 64.0
    np.testing.assert_array_equal(np.asarray(d_x), want)

    # An fp32 kernel's `1 + 2**-12` the forward multiplied as fp16's 1: against -1 its gradient cancels.
    exact = jnp.zeros(kernel.shape, jnp.float32).at[..., 0, 0].set(1.0 + 2.0 ** -12).at[..., 0, 1].set(-1.0)
    _, pull = jax.vjp(lambda rows: low_rank(rows, exact, a, b, 1.0, sizes, out_dtype=jnp.float32), x)
    (d_x,) = pull(jnp.zeros((8, 8), jnp.float32).at[0, 0].set(1.0).at[0, 1].set(1.0))
    assert np.asarray(d_x)[0, 0] == 0.0, np.asarray(d_x)[0, :2]


@pytest.mark.mesh(devices=2)
@pytest.mark.parametrize("grouped", [False, True])
def test_an_adapted_product_in_a_checked_map_sums_its_replicated_operands_gradients(grouped):
    """An adapted product inside a `shard_map` over two CPU devices that checks the axes its values vary over
    (`check_vma`), each device its own rows and every device the same kernel, factors and scale: their
    gradients are each device's partial summed over the rows' axis, invariant, equal to the base product and
    its update written plainly, exactly, over small integers (row 1304)."""
    from dew.nn.kernels.low_rank import low_rank

    rng = np.random.default_rng(1304)

    def ints(*shape):
        return jnp.asarray(rng.integers(-2, 3, shape), jnp.bfloat16)

    stacked = (2,) if grouped else ()
    x, kernel = ints(8, 16), ints(*stacked, 16, 8)
    a, b, scale = ints(*stacked, 16, 4), ints(*stacked, 4, 8), jnp.float32(1.0)
    # Each device's 4 rows, two to each group.
    sizes = jnp.asarray([2, 2], jnp.int32) if grouped else None
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("data",))
    data, whole = jax.sharding.PartitionSpec("data"), jax.sharding.PartitionSpec()

    def plain(rows, w, a_, b_, s):
        rows, w, a_, b_ = (operand.astype(jnp.float32) for operand in (rows, w, a_, b_))
        if sizes is None:
            return rows @ w + s * ((rows @ a_) @ b_)
        # Each row's group's matrices, gathered.
        owner = jnp.repeat(jnp.arange(2), sizes, total_repeat_length=rows.shape[0])
        update = jnp.einsum('rk,rkw->rw', jnp.einsum('rd,rdk->rk', rows, a_[owner]), b_[owner])
        return jnp.einsum('rd,rdw->rw', rows, w[owner]) + s * update

    def adapted(rows, w, a_, b_, s):
        return low_rank(rows, w, a_, b_, s, sizes, out_dtype=jnp.float32)

    def gradients(product):
        def each_device(rows, w, a_, b_, s):
            return jax.grad(lambda *ops: jnp.sum(product(rows, *ops)), argnums=(0, 1, 2, 3))(w, a_, b_, s)

        return jax.jit(jax.shard_map(each_device, mesh=mesh, in_specs=(data, whole, whole, whole, whole),
                                     out_specs=whole, check_vma=True))(x, kernel, a, b, scale)

    for got, want in zip(gradients(adapted), gradients(plain), strict=True):
        np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))


@pytest.mark.mesh(devices=2)
def test_an_adapted_products_replicated_gradient_sums_before_it_rounds():
    """A replicated fp16 kernel's gradient in a checked `shard_map` over two CPU devices is their partials
    summed wide, then rounded once: `65536` (fp16's inf alone) and `-65504` sum to 32 (row 1304)."""
    from dew.nn.kernels.low_rank import low_rank

    x = jnp.zeros((2, 8), jnp.float16).at[:, 0].set(1.0)
    kernel, a, b = (jnp.zeros(shape, jnp.float16) for shape in ((8, 8), (8, 4), (4, 8)))
    cotangent = jnp.zeros((2, 8), jnp.float32).at[0, 0].set(65536.0).at[1, 0].set(-65504.0)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:2]), ("data",))
    data, whole = jax.sharding.PartitionSpec("data"), jax.sharding.PartitionSpec()

    def each_device(rows, w, dy):
        _, pull = jax.vjp(lambda w_: low_rank(rows, w_, a, b, 1.0, out_dtype=jnp.float32), w)
        return pull(dy)[0]

    d_kernel = jax.jit(jax.shard_map(each_device, mesh=mesh, in_specs=(data, whole, data), out_specs=whole,
                                     check_vma=True))(x, kernel, cotangent)
    want = np.zeros((8, 8), np.float16)
    want[0, 0] = 32.0
    np.testing.assert_array_equal(np.asarray(d_kernel), want)
