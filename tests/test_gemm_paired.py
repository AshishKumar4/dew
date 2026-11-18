"""The template's paired product (`gemm.gemm`'s `paired`: two matrices of one shape multiplied by the same
rows, one launch, its epilogue reading both fp32 accumulators): its check 1 against the two products
combined, through every mode of its epilogue, at a named compute dtype and without an epilogue, and its
lowering; the hooked cases it shares are test_gemm_template's."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import COMPILES_ON_CPU, generations, lowers, runs
from jax.sharding import Mesh
from reference_error import assert_as_exact, with_gradients
from test_gemm_template import CASES, SEEDS, _dew, _draw, _drop, _rounded_through

from dew.nn.kernels import gemm, generation, grouped_launch, xla_products
from dew.nn.kernels.templates import mosaic_sm90, resources
from dew.nn.kernels.templates.gemm import derivation, mosaic_gpu


def swiglu(gate, up):
    """A SwiGLU epilogue on a paired product's two fp32 accumulators, each rounded to bf16 first and the
    activated gate again, as `activations.gated_product` rounds them."""
    gate, up = gate.astype(jnp.bfloat16), up.astype(jnp.bfloat16)
    return (jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16) * up).astype(jnp.float32)


def test_a_grouped_call_no_cooperative_tile_holds_derives_one_consumer_alone():
    """Where no two-consumer tile holds a grouped call, its derivation falls back to one consumer, which
    runs on one multiprocessor: a pair is two consumers' (`mosaic_gpu.Tiles`), so the fallback derives a
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
    tiles = mosaic_gpu.Tiles(128, 128, 64, 3, 1, warpgroups=2, family="wgmma")
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
                       tiles=mosaic_gpu.Tiles(64, 128, 32, 3, 1, family="mma.sync"), transpose_rhs=False,
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

    tiles = mosaic_gpu.Tiles(128, 128, 64, 4, 1, warpgroups=2, family="wgmma")
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

