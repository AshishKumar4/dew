"""The template's paired product (`gemm.gemm`'s `paired`: two matrices of one shape multiplied by the same
rows, one launch, its epilogue reading both fp32 accumulators): its check 1 against the two products
combined, through every mode of its epilogue, at a named compute dtype and without an epilogue, and its
lowering; the hooked cases it shares are test_gemm_template's."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import generations, lowers, runs
from jax.sharding import Mesh
from reference_error import assert_as_exact, with_gradients
from test_gemm_template import CASES, SEEDS, _dew, _draw, _drop, _rounded_through

from dew.nn.kernels import gemm, generation, grouped_matmul, xla_products
from dew.nn.kernels.templates import mosaic_sm90, resources
from dew.nn.kernels.templates.gemm import mosaic_gpu, triton_dense


def swiglu(gate, up):
    """A SwiGLU epilogue on a paired product's two fp32 accumulators, each rounded to bf16 first and the
    activated gate again, as `activations.gated_product` rounds them."""
    gate, up = gate.astype(jnp.bfloat16), up.astype(jnp.bfloat16)
    return (jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16) * up).astype(jnp.float32)


PAIRED_TILE = grouped_matmul.Tile(64, 128, 64, 4, 2)


def test_a_grouped_call_no_cooperative_tile_holds_derives_one_consumer_alone():
    """Where no two-consumer tile holds a grouped call, its derivation falls back to one consumer, which
    runs on one multiprocessor: a pair is two consumers' (`mosaic_gpu.Tiles`), so the fallback derives a
    tile rather than refusing one it built."""

    tile = mosaic_gpu.derived((mosaic_sm90.WGMMA,), rows=512, depth=2048, width=1536, dtype=jnp.bfloat16,
                              multiprocessors=132, target=generation.TARGETS["sm90"],
                              candidate=lambda tile: tile.warpgroups == 1,
                              launch_of=lambda tile: resources.Launch(lambda stages: resources.Scope(),
                                                                      tile.warpgroups, 0), grouped=True)
    assert tile is not None and (tile.warpgroups, tile.cluster) == (1, 1)


def test_a_paired_product_holds_check_1_against_its_two_products_combined():
    """The two-source dense kernel (one `lhs` tile, two matrices, two accumulators) at a tile, with rows,
    depth and width no tile divides: its SwiGLU of the two products on par with the plain two products'
    combined by the same epilogue over the draws (check 1), truth the two products' SwiGLU in float64."""
    rows, depth, width = 200, 300, 140
    shapes = ((rows, depth), (depth, width), (depth, width))

    def draw(seed):
        keys = jax.random.split(jax.random.fold_in(jax.random.key(7), seed), 3)
        return tuple(jax.random.normal(key, shape).astype(jnp.bfloat16)
                     for key, shape in zip(keys, shapes, strict=True))

    def run(lhs, gate, up):
        [paired] = triton_dense.gmm(lhs, gate, paired=up, epilogue=swiglu, out_dtypes=("bfloat16",),
                                    tile=PAIRED_TILE, transpose_rhs=False, interpret=True)
        return paired

    def plain(lhs, gate, up):
        def product(matrix):
            return jnp.dot(lhs, matrix, preferred_element_type=jnp.float32)
        return swiglu(product(gate), product(up)).astype(jnp.bfloat16)

    def truth(lhs, gate, up):
        return jax.nn.silu(lhs @ gate) * (lhs @ up)

    paired = run(*draw(0))
    assert paired.shape == (rows, width) and paired.dtype == jnp.bfloat16
    assert_as_exact(run, jax.jit(plain), draw, "paired product", seeds=SEEDS, truth_fn=truth)


@pytest.mark.parametrize("generation_name", generations("sm80", "sm90"))
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


@pytest.mark.parametrize("generation_name", ["sm80", "sm86", "sm90"])
def test_a_paired_dense_product_lowers_for_cuda(generation_name):
    """The two-source dense kernel lowers for CUDA on the CPU."""
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    args = (bf16((1021, 200)), bf16((200, 300)), bf16((200, 300)))

    def paired(lhs, gate, up):
        return triton_dense.gmm(lhs, gate, paired=up, epilogue=swiglu, out_dtypes=("bfloat16",),
                                tile=PAIRED_TILE, transpose_rhs=False, interpret=False)

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), generation_name)
    with generation.assumed(generation_name), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(paired).trace(*args).lower(lowering_platforms=("cuda",))


def test_a_paired_product_without_an_epilogue_is_refused():
    lhs = jnp.zeros((64, 64), jnp.bfloat16)
    with pytest.raises(ValueError):
        triton_dense.gmm(lhs, lhs, paired=lhs, out_dtypes=("bfloat16",), tile=PAIRED_TILE,
                         transpose_rhs=False, interpret=True)


@lowers("sm90")
def test_a_hook_that_broadcasts_a_side_operand_lowers_for_hopper():
    """A side operand given back with its dimension of one, broadcast across its tile
    (`grouped_matmul._unsqueezed`), which Mosaic's layout inference lays out where it found no layout for the
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
