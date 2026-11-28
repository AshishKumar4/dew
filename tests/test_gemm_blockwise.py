"""The blockwise-scaled FP8 product (`gemm.blockwise_fp8_product`, DeepSeek-V3's blockwise FP8, row 4518)
on the CPU: the template's step-scaled walk, interpreted as an H100's, on par with the scaled blocks summed in
XLA (check 1), from the float64 sum of the dequantized blocks (tests/blockwise_matrix.py's `truth`)."""

import functools

import blockwise_matrix
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from blockwise_matrix import CUBLASLT_PRECISION
from reference_error import assert_as_exact, equations, with_gradients

from dew.nn.kernels import gemm, generation, grouped_quantized, projections, tuning
from dew.nn.kernels.templates.gemm import tiling


@pytest.mark.parametrize("groups", ["1", "4"])
@pytest.mark.parametrize("columns", ["128", "1"])
def test_a_blockwise_fp8_product_through_the_template_is_on_par(columns, groups):
    """On an H100, interpreted, the product of FP8 rows and each group's FP8 matrix, each 128-deep block's
    partial times its rows' and its group's column scales (a weight block's, or each column's) before it is
    added, through the template named (`mosaic_gpu.gmm`'s `step_scales`), on par with XLA's (check 1), from
    float64; the routes it records name the template."""
    case = {"columns": columns, "groups": groups}
    sizes = blockwise_matrix._sizes(int(groups))

    def product(kernel, *operands):
        with generation.chosen(blockwise_fp8=kernel):
            return gemm.blockwise_fp8_product(*operands[:2], sizes, *operands[2:], out_dtype=jnp.float32,
                                              interpret_on_cpu=True)

    with generation.assumed("sm90"), generation.routes() as routes:
        assert_as_exact(jax.jit(functools.partial(product, "mosaic")),
                        jax.jit(functools.partial(product, "xla")),
                        lambda seed: blockwise_matrix._operands(case, seed), f"blockwise {columns} {groups}",
                        seeds=blockwise_matrix.SEEDS, truth_fn=blockwise_matrix.truth(sizes))
    assert ("blockwise_fp8", "mosaic") in {(route.choice, route.kernel) for route in routes}


def _every_block(code: float) -> np.ndarray:
    """A row of `blockwise_matrix.DEPTH` codes, `code` first in each 128-deep block, every other zero."""
    row = np.zeros(blockwise_matrix.DEPTH, np.float32)
    row[0::128] = code
    return row


def _edge_product(row_codes, column_codes, scales: tuple[float, float], dtype, interpret_on_cpu: bool
                  ) -> np.ndarray:
    """The blockwise FP8 product through the template of rows each `row_codes` (`blockwise_matrix.DEPTH`
    codes in `dtype`) and one matrix whose columns are each `column_codes`, every block's row and column
    scales `scales`. Its routes name the template: a selector falling back to XLA, whose arm scales each
    partial in `precision.scaled_partial`'s orders, would read the same exact value of no kernel (row
    13292)."""
    blocks = blockwise_matrix.DEPTH // 128
    rows = np.broadcast_to(np.asarray(row_codes, np.float32), (blockwise_matrix.ROWS, blockwise_matrix.DEPTH))
    matrices = np.broadcast_to(np.asarray(column_codes, np.float32),
                               (1, blockwise_matrix.WIDTH, blockwise_matrix.DEPTH))
    row_scales = np.full((blocks, blockwise_matrix.ROWS), scales[0], np.float32)
    column_scales = np.full((1, blocks, blockwise_matrix.WIDTH), scales[1], np.float32)
    with generation.chosen(blockwise_fp8="mosaic"), generation.routes() as routes:
        out = jax.jit(lambda *ops: gemm.blockwise_fp8_product(
            *ops[:2], blockwise_matrix._sizes(1), *ops[2:], out_dtype=jnp.float32,
            interpret_on_cpu=interpret_on_cpu))(rows.astype(dtype), matrices.astype(dtype), row_scales,
                                                column_scales)
    assert ("blockwise_fp8", "mosaic") in {(route.choice, route.kernel) for route in routes}
    return np.asarray(out)


def a_blockwise_fp8_product_at_far_apart_scales_is_exact(interpret_on_cpu: bool = False) -> None:
    """A blockwise FP8 product through the template of codes 1.5 (one a 128-deep block in each row and
    column: each block's partial 2.25) at a subnormal row scale 2^-128 and a column scale 2^127: each block's
    scaled partial is 1.125 exactly, so the product is its blocks' count times 1.125, where the partial
    times either scale first underflows or overflows (the scales' own product first, row 13274). A GPU keeps
    the subnormal scale; a CPU's XLA flushes it."""
    out = _edge_product(_every_block(1.5), _every_block(1.5), (2.0 ** -128, 2.0 ** 127), jnp.float8_e4m3fn,
                        interpret_on_cpu)
    np.testing.assert_array_equal(out, np.full(out.shape, blockwise_matrix.DEPTH // 128 * 1.125, np.float32))


def a_blockwise_fp8_product_at_large_finite_scales_is_finite(interpret_on_cpu: bool = False) -> None:
    """A blockwise FP8 product through the template of codes 2^-3 (one a 128-deep block in each row and
    column: each block's partial 2^-6) at row and column scales 2^64: each block's scaled partial is 2^122,
    finite, so the product is its blocks' count times 2^122, exactly, though the two scales' own product
    overflows (row 13257)."""
    out = _edge_product(_every_block(2.0 ** -3), _every_block(2.0 ** -3), (2.0 ** 64, 2.0 ** 64),
                        jnp.float8_e4m3fn, interpret_on_cpu)
    np.testing.assert_array_equal(out, np.full(out.shape, blockwise_matrix.DEPTH // 128 * 2.0 ** 122,
                                               np.float32))


E5M2_EDGES = {
    "row first": ((32768.0, 32768.0, 40960.0, 16384.0), (32768.0, 32768.0, 32768.0, 8192.0),
                  ((2.0 ** 23 + 3) * 2.0 ** 25, 9942050.0 * 2.0 ** 25)),
    "column first": ((32768.0, 32768.0, 40960.0, 16384.0), (32768.0, 32768.0, 32768.0, 8192.0),
                     ((2.0 ** 23 + 3) * 2.0 ** 81, 9942050.0 * 2.0 ** -31)),
}
"""E5M2 steps whose scaled partial is fp32's largest (row 13291), each its leading codes in each row and in
each column and its row and column scales: a partial (27/16) 2^31, past any E4M3 step's 2^25, times the row
scale first, where the scales' rounded product first rounds past fp32's overflow midpoint; and times the
column scale first, the row scale's intermediate past fp32's largest (`precision.scaled_partial`'s orders,
XLA's arm's)."""


def _e5m2_edge(edge: str) -> tuple[np.ndarray, np.ndarray, tuple[float, float], np.float32]:
    """`E5M2_EDGES[edge]`'s codes as rows of `blockwise_matrix.DEPTH` (every code past them zero), its scales,
    and its one scaled partial, fp32's largest (the portable arm's, `precision.scaled_partial`'s)."""
    leading, columns, scales = E5M2_EDGES[edge]
    rows, matrix = np.zeros(blockwise_matrix.DEPTH, np.float32), np.zeros(blockwise_matrix.DEPTH, np.float32)
    rows[:len(leading)], matrix[:len(columns)] = leading, columns
    return rows, matrix, scales, np.finfo(np.float32).max


def an_e5m2_blockwise_product_at_fp32s_largest_is_finite(edge: str, interpret_on_cpu: bool = False) -> None:
    """A blockwise E5M2 product through the template whose first 128-deep block holds `E5M2_EDGES[edge]`'s
    codes at its scales: each entry fp32's largest, `precision.scaled_partial`'s, XLA's arm's (row 13291)."""
    rows, matrix, scales, want = _e5m2_edge(edge)
    out = _edge_product(rows, matrix, scales, jnp.float8_e5m2, interpret_on_cpu)
    np.testing.assert_array_equal(out, np.full(out.shape, want, np.float32))


def an_e5m2_blockwise_kernel_gradient_at_fp32s_largest_is_finite(edge: str, interpret_on_cpu: bool = False
                                                                 ) -> None:
    """A blockwise E5M2 kernel gradient through the template, one group of one 128-row block whose depth
    rows each hold `E5M2_EDGES[edge]`'s row codes and its columns its column codes, at its scales: each entry
    fp32's largest, as the product's (`mosaic_gpu.tgmm`'s step scaling, row 13291). Its routes name the
    template (row 13292)."""
    rows, matrix, (row_scale, column_scale), want = _e5m2_edge(edge)
    lhs_t = np.broadcast_to(rows[:128], (GRADIENT_DEPTH, 128)).astype(jnp.float8_e5m2)
    rhs_t = np.broadcast_to(matrix[:128], (GRADIENT_WIDTH, 128)).astype(jnp.float8_e5m2)
    scales = (np.full((1, GRADIENT_DEPTH), row_scale, np.float32), np.full((1, GRADIENT_WIDTH), column_scale,
                                                                            np.float32))
    with generation.chosen(blockwise_fp8="mosaic"), generation.routes() as routes:
        out = jax.jit(lambda *ops: gemm.blockwise_fp8_kernel_gradient(
            *ops[:2], np.asarray([1], np.int32), *ops[2:], out_dtype=jnp.float32,
            interpret_on_cpu=interpret_on_cpu))(lhs_t, rhs_t, *scales)
    assert ("blockwise_fp8", "mosaic") in {(route.choice, route.kernel) for route in routes}
    np.testing.assert_array_equal(np.asarray(out), np.full(out.shape, want, np.float32))


@pytest.mark.parametrize("edge", list(E5M2_EDGES))
@pytest.mark.parametrize("witness", [an_e5m2_blockwise_product_at_fp32s_largest_is_finite,
                                     an_e5m2_blockwise_kernel_gradient_at_fp32s_largest_is_finite],
                         ids=["product", "kernel gradient"])
def test_an_e5m2_blockwise_walk_at_fp32s_largest_is_finite(witness, edge):
    """Each E5M2 edge witness interpreted on a CPU as an H100 runs it (the GPU's native cells,
    tests/test_gpu_gemm.py, run them compiled)."""
    with generation.assumed("sm90"):
        witness(edge, interpret_on_cpu=True)


def test_a_blockwise_fp8_product_at_large_finite_scales_is_finite():
    """`a_blockwise_fp8_product_at_large_finite_scales_is_finite` interpreted on a CPU as an H100 runs it
    (the GPU's native cell, tests/test_gpu_gemm.py, runs it compiled)."""
    with generation.assumed("sm90"):
        a_blockwise_fp8_product_at_large_finite_scales_is_finite(interpret_on_cpu=True)


def test_a_blockwise_fp8_products_launch_reads_its_pin():
    """The blockwise FP8 product through the template named on an H100 launches through its launch
    (`grouped_quantized.MosaicStepScaledGmm`, `tuning.KernelOp.__call__`): a pin of its kind is read, one it
    cannot take refused by name (`tuning.Unpinnable`: an mma.sync tile, which no FP8 product of an H100
    takes), where the product called its mainloop beside the launch and ignored the pin."""
    case = {"columns": "128", "groups": "1"}
    sizes = blockwise_matrix._sizes(1)
    operands = blockwise_matrix._operands(case, 0)
    untakeable = tiling.Tiles(128, 128, 128, 2, 1, family='mma.sync')

    def product(*operands):
        return gemm.blockwise_fp8_product(*operands[:2], sizes, *operands[2:], out_dtype=jnp.float32,
                                          interpret_on_cpu=True)

    with (generation.assumed("sm90"), generation.chosen(blockwise_fp8="mosaic"),
          tuning.pinned({grouped_quantized.MosaicStepScaledGmm: untakeable}),
          pytest.raises(tuning.Unpinnable)):
        jax.eval_shape(product, *operands)


def test_a_blockwise_fp8_kernel_gradient_through_the_template_is_on_par():
    """`a_blockwise_kernel_gradient_is_on_par` on an H100, interpreted."""
    with generation.assumed("sm90"):
        a_blockwise_kernel_gradient_is_on_par(interpret=True)


GRADIENT_BLOCKS = (1, 0, 2, 1)
"""The kernel gradient cells' groups' 128-row blocks: an empty group among them."""
GRADIENT_DEPTH, GRADIENT_WIDTH = 256, 256


def _gradient_operands(seed: int) -> tuple:
    """A kernel gradient's FP8 operands (rows contiguous: `[depth, rows]`, `[width, rows]`) and each 128-row
    block's depth rows' and columns' scales."""
    rng = np.random.default_rng(seed)
    rows = 128 * sum(GRADIENT_BLOCKS)
    # NumPy's, as a site's draws are: a GPU tier's prepare computes on no device.
    return (rng.normal(size=(GRADIENT_DEPTH, rows)).astype(jnp.float8_e4m3fn),
            rng.normal(size=(GRADIENT_WIDTH, rows)).astype(jnp.float8_e4m3fn),
            rng.uniform(0.5, 2.0, size=(rows // 128, GRADIENT_DEPTH)).astype(np.float32),
            rng.uniform(0.5, 2.0, size=(rows // 128, GRADIENT_WIDTH)).astype(np.float32))


def _gradient(kernel: str, interpret: bool = False):
    """The blockwise FP8 kernel gradient of `_gradient_operands`, its selector naming `kernel`."""
    blocks = np.asarray(GRADIENT_BLOCKS, np.int32)

    def gradient(*operands):
        with generation.chosen(blockwise_fp8=kernel):
            return gemm.blockwise_fp8_kernel_gradient(*operands[:2], blocks, *operands[2:],
                                                      out_dtype=jnp.float32, interpret_on_cpu=interpret)
    return gradient


def _blocked(lhs_t, rhs_t, row_scales, column_scales, wide, precision=None):
    """Each 128-row block's partial in `wide` at `precision`, times its scales, added in order into its
    group's sum: the kernel gradient's definition, written out (a truth's in float64)."""
    out = jnp.zeros((len(GRADIENT_BLOCKS), GRADIENT_DEPTH, GRADIENT_WIDTH), wide)
    owners = np.repeat(np.arange(len(GRADIENT_BLOCKS)), np.asarray(GRADIENT_BLOCKS))
    for block, owner in enumerate(owners):
        rows = slice(128 * block, 128 * (block + 1))
        partial = jnp.dot(lhs_t[:, rows], rhs_t[:, rows].T, precision=precision, preferred_element_type=wide)
        out = out.at[owner].add(partial * row_scales[block, :, None].astype(wide)
                                * column_scales[block, None, :].astype(wide))
    return out


def _gradient_truth(lhs_t, rhs_t, *scales):
    wide = jnp.float64
    return _blocked(lhs_t.astype(wide), rhs_t.astype(wide), *scales, wide)


def _gradient_cublaslt(*operands):
    """cuBLASLt's FP8 GEMM on the kernel gradient's codes a block at a time (`blockwise_matrix.cublaslt_arm`:
    its blocks' dots of the FP8 codes into fp32 at `CUBLASLT_PRECISION`, cuBLASLt's fast accumulation), each
    partial scaled and added in order. The precision read as a name of this module's: a test module an arm
    reads through is no value its prepare's statement holds (`truths_ahead.arm_role`), and the arm would run
    unprepared."""
    return _blocked(*operands, jnp.float32, precision=CUBLASLT_PRECISION)


def a_blockwise_kernel_gradient_is_on_par(interpret: bool = False):
    """Each group's rows of FP8 operands (rows contiguous) contracted over its whole 128-row blocks, each
    block's partial times its depth rows' and its columns' scales before it is added to its group's sum,
    through the template named (`mosaic_gpu.tgmm`'s `k_major` and `step_scales`), on par with cuBLASLt's FP8
    GEMM on the same codes a block at a time (check 1, `_gradient_cublaslt`: block-scaled FP8 on Hopper is
    held at parity with cuBLASLt), from float64; an empty group's gradient is zero. On this GPU, or
    `interpret`ed (test_gpu_gemm's cell)."""
    reference = blockwise_matrix.cublaslt_arm(_gradient_cublaslt, _gradient_operands(0))
    with generation.routes() as routes:
        assert_as_exact(jax.jit(_gradient("mosaic", interpret)), reference, _gradient_operands,
                        "blockwise kernel gradient", seeds=blockwise_matrix.SEEDS,
                        truth_fn=_gradient_truth)
    assert ("blockwise_fp8", "mosaic") in {(route.choice, route.kernel) for route in routes}

def a_blockwise_kernel_gradients_portable_arm_sums_exactly():
    """On this GPU, under its lane's flags (the CUDA lane's deterministic ops, which keep XLA's Triton GEMMs
    off), the kernel gradient's portable arm (`blockwise_fp8` chosen `xla`, `gemm._blocks_summed`) on par
    with the FP8 codes' values multiplied in fp32 at the highest precision (check 1), from float64: each
    block's products exact and summed into fp32, as the FP8 contract's reference arm states them
    (docs/design/testing.md, "FP8 products"). Its dot of the widened codes was XLA's to fold into cuBLASLt's
    FP8 GEMM there, 4.43e-5 from float64 against 4.05e-8 on a GH200 (M-EM-17c)."""
    def values(*operands):
        lhs_t, rhs_t, *scales = operands
        return _blocked(lhs_t.astype(jnp.float32), rhs_t.astype(jnp.float32), *scales, jnp.float32,
                        jax.lax.Precision.HIGHEST)

    assert_as_exact(jax.jit(_gradient("xla")), jax.jit(values), _gradient_operands,
                    "blockwise kernel gradient's portable arm", seeds=blockwise_matrix.SEEDS,
                    truth_fn=_gradient_truth)


def test_a_bank_quantized_by_blocks_is_each_blocks_codes_under_its_power_of_two_scale():
    """`projections.blocks_quantized` of a bank of matrices whose 128 x 128 blocks span ten decades (one block
    of zeros) is, computed on the host in NumPy, each block's values over its scale rounded to E4M3 (to
    nearest, ties to even, clamped to 448) under its scale: the least power of two at or above its largest
    magnitude, floored at 1e-4, times fp32's 1/448 (DeepGEMM's `per_block_cast_to_fp8` at ue8m0 scales, as
    the kernels' `fast_round_scale` computes it), bit for bit (row 4518)."""
    import ml_dtypes
    from reference_error import ran_here

    rng = np.random.default_rng(4518)
    decades = 10.0 ** rng.integers(-6, 4, (2, 2, 1, 3, 1))
    decades[0, 0, :, 0] = 0
    w = (rng.standard_normal((2, 2, 128, 3, 128)) * decades).astype(np.float32).reshape(2, 256, 384)
    tiles = w.reshape(2, 2, 128, 3, 128)
    over = np.maximum(np.abs(tiles).max(axis=(2, 4), keepdims=True), np.float32(1e-4)) * np.float32(1 / 448)
    fraction, exponent = np.frexp(over)
    scale = np.ldexp(np.float32(1), np.where(fraction == 0.5, exponent - 1, exponent)).astype(np.float32)
    codes = np.clip(tiles / scale, -448, 448).astype(ml_dtypes.float8_e4m3fn).reshape(w.shape)
    # Every way run before any is judged (`ran_here`).
    found = [run(w) for run in ran_here(projections.blocks_quantized, "a bank quantized by blocks")]
    for found_codes, found_scales in found:
        np.testing.assert_array_equal(np.asarray(found_codes).view(np.uint8), codes.view(np.uint8))
        np.testing.assert_array_equal(np.asarray(found_scales), scale[:, :, 0, :, 0])


def test_a_blockwise_fp8_projections_training_step_is_on_par():
    """On an H100, interpreted, one training step through the blockwise FP8 recipe
    (`projections.blockwise_fp8_projection`: its output and both gradients), the template named, on par with
    the recipe through XLA's arms (check 1), from the recipe in float64 (tests/blockwise_matrix.py's
    `recipe_truth`): the step-level parity a preset's projections hold (row 2513)."""
    group_blocks = jnp.asarray([1, 0, 2], jnp.int32)

    def step(kernel):
        def projection(x, w):
            with generation.chosen(blockwise_fp8=kernel):
                return projections.blockwise_fp8_projection(x, w, group_blocks, out_dtype=jnp.float32,
                                                     interpret_on_cpu=True)
        return with_gradients(projection)

    with generation.assumed("sm90"):
        assert_as_exact(jax.jit(step("mosaic")), jax.jit(step("xla")), blockwise_matrix.step_operands,
                        "blockwise fp8 training step",
                        seeds=blockwise_matrix.SEEDS,
                        truth_fn=with_gradients(blockwise_matrix.recipe_truth(group_blocks)))


def test_a_biased_recipe_projection_rounds_once_after_its_bias():
    """A biased projection the 'fp8' recipe takes (`projections.recipe_projection`'s `bias`, audit A1) adds
    each group's bias to its rows' fp32 sums and rounds the whole once: bitwise its unbiased fp32 product plus
    the bias, rounded to bfloat16, where a bias cancelling all but 2^-10 of each sum leaves that remainder,
    which rounding the product to bfloat16 first lost."""
    from dew.nn.precision import FP8, projection_role

    rng = np.random.default_rng(9)
    rows = jnp.asarray(rng.normal(size=(200, 256)), jnp.float32)
    matrices = jnp.asarray(rng.normal(size=(2, 256, 128)) / 16, jnp.float32)
    sizes = jnp.asarray([120, 80], jnp.int32)

    def projected(bias, out_dtype):
        with generation.run_precision(FP8), projection_role("experts"):
            return projections.recipe_projection(rows, matrices, sizes, out_dtype=out_dtype, bias=bias)

    summed = projected(None, jnp.float32)
    group = np.repeat([0, 1], [120, 80])
    bias = jnp.asarray(np.stack([-np.asarray(summed)[group == g].mean(0) for g in (0, 1)]), jnp.float32)
    out = projected(bias, jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(out), np.asarray((summed + bias[group]).astype(jnp.bfloat16)))


def test_blockwise_fp8_named_mosaic_uninterpreted_dispatches_portable_on_cpu():
    """In a CPU process assuming an H100 with `blockwise_fp8="mosaic"` named and `interpret_on_cpu=False`,
    both `blockwise_fp8_product` and `blockwise_fp8_kernel_gradient` dispatch through `_dispatched` to their
    portable arms on CPU instead of lowering Mosaic uninterpreted (row 8646)."""
    case = {"columns": "128", "groups": "1"}
    sizes = blockwise_matrix._sizes(1)
    lhs, kernel, row_scales, col_scales = blockwise_matrix._operands(case, 0)
    rng = np.random.default_rng(19)
    blocks = jnp.asarray([1], jnp.int32)
    lhs_t = jnp.asarray(rng.normal(size=(128, 128)) / 4, jnp.float8_e4m3fn)
    rhs_t = jnp.asarray(rng.normal(size=(128, 128)) / 4, jnp.float8_e4m3fn)
    grad_row_scales = jnp.ones((1, 128), jnp.float32)
    grad_col_scales = jnp.ones((1, 128), jnp.float32)
    from dew.nn.kernels import xla_products

    cpu = jax.devices("cpu")[0]
    committed = tuple(jax.device_put(a, cpu) for a in (lhs, kernel, sizes, row_scales, col_scales))
    assert xla_products.committed_device(committed) == cpu
    assert xla_products.committed_device((np.zeros(4), jnp.zeros(4))) is None
    with generation.assumed("sm90"), jax.default_device(cpu):
        with generation.chosen(blockwise_fp8="mosaic"):
            got_fwd = gemm.blockwise_fp8_product(*committed, out_dtype=jnp.float32, interpret_on_cpu=False)
            got_grad = gemm.blockwise_fp8_kernel_gradient(lhs_t, rhs_t, blocks, grad_row_scales,
                                                          grad_col_scales, out_dtype=jnp.float32,
                                                          interpret_on_cpu=False)
        with generation.chosen(blockwise_fp8="xla"):
            ref_fwd = gemm.blockwise_fp8_product(*committed, out_dtype=jnp.float32, interpret_on_cpu=False)
            ref_grad = gemm.blockwise_fp8_kernel_gradient(lhs_t, rhs_t, blocks, grad_row_scales,
                                                          grad_col_scales, out_dtype=jnp.float32,
                                                          interpret_on_cpu=False)
    np.testing.assert_array_equal(np.asarray(got_fwd), np.asarray(ref_fwd))
    np.testing.assert_array_equal(np.asarray(got_grad), np.asarray(ref_grad))


def test_restored_blockwise_cublaslt_references_prepare_on_sm90(tmp_path, monkeypatch):
    """Both restored blockwise FP8 cells (`blockwise_matrix.a_blockwise_fp8_product_is_on_par` and
    `a_blockwise_kernel_gradient_is_on_par`) with cuBLASLt as `reference` prepare cleanly under a
    compile-only sm90 GPU-tier prepare with zero unprepared sites (row 12062)."""
    import pickle

    import generation_support
    import model_flows
    import tier_wire
    from tier_wire import DEFERRED
    from truths_ahead import AT_RUN, HELD, SITE, UNPREPARED, prepared

    monkeypatch.setenv(model_flows.TOPOLOGY, str(generation_support.node_topology(tmp_path / "topology.bin",
                                                                                   gpu="sm90")))
    monkeypatch.setenv(tier_wire.PREPARE, "1")
    store = tmp_path / "store"
    with prepared(store, stamp="blockwise-12062", target=generation_support.node_target()):
        blockwise_matrix.a_blockwise_fp8_product_is_on_par(blockwise_matrix.CASES[0])
        a_blockwise_kernel_gradient_is_on_par(interpret=False)
    for record in (UNPREPARED, AT_RUN, DEFERRED):
        assert not (store / record).exists(), (store / record).read_text()
    arm_sites = [pickle.loads((s.parent / HELD).read_bytes()) for s in store.rglob(SITE)]
    with_ref = [h for h in arm_sites
                if any(k.startswith("dew ") for k in h) and any(k.startswith("reference ") for k in h)]
    assert len(with_ref) == 2, [list(h.keys()) for h in arm_sites]


@pytest.mark.parametrize("rows", [256, 384])
def test_a_blockwise_chain_through_the_template_is_its_two_projections_apart(rows):
    """An ungated MLP's two blockwise FP8 projections chained (`projections.blockwise_chain`), on an H100,
    interpreted, through the template named: the down projection's rows quantized in the up projection's
    epilogue and stored there (E4M3 codes and 1 x 128 scales `[hidden / 128, rows]`, transposed,
    `quantized.reduced`), its value and every gradient bitwise the two projections apart with the
    activation between them in XLA, as a recipe's run takes them today (row 8547); and the up projection's
    launch the template's, its scales output in its form (`Hooks.out_forms`). 384 rows end in a share past
    the last row tile's pair of 128 (its stores whole shares, the tensor's bound clipping them)."""
    group_blocks = jnp.asarray([rows // 128], jnp.int32)
    rng = np.random.default_rng(8547)
    x = jnp.asarray(rng.normal(size=(rows, 256)), jnp.bfloat16)
    up = jnp.asarray(rng.normal(size=(1, 256, 384)) / 16, jnp.bfloat16)
    down = jnp.asarray(rng.normal(size=(1, 384, 256)) / 16, jnp.bfloat16)
    cotangent = jnp.asarray(rng.normal(size=(rows, 256)), jnp.bfloat16)

    def chained(x, up, down):
        return projections.blockwise_chain(x, up, down, jax.nn.gelu, out_dtype=jnp.bfloat16,
                                           interpret_on_cpu=True)

    def apart(x, up, down):
        hidden = projections.blockwise_fp8_projection(x, up, group_blocks, out_dtype=jnp.bfloat16,
                                                      interpret_on_cpu=True)
        return projections.blockwise_fp8_projection(jax.nn.gelu(hidden).astype(jnp.bfloat16), down,
                                                    group_blocks, out_dtype=jnp.bfloat16,
                                                    interpret_on_cpu=True)

    with generation.assumed("sm90"), generation.chosen(blockwise_fp8="mosaic"), generation.routes() as routes:
        fused = jax.jit(with_gradients(chained))(x, up, down, cotangent)
        separate = jax.jit(with_gradients(apart))(x, up, down, cotangent)
    blockwise_matrix.assert_bitwise_leaves(("value", "d_x", "d_up", "d_down"), fused, separate)
    assert ("blockwise_fp8", "mosaic") in {(route.choice, route.kernel) for route in routes}


def test_a_blockwise_fp8_epilogues_codes_and_scales_are_the_recipes():
    """The blockwise quantizing epilogue (`quantized.blockwise_fp8_epilogue`) on XLA's route of a product:
    its stored product rounded once, and its codes and transposed 1 x 128 scales of the activated product,
    bitwise `projections.rows_quantized` of the activation of that rounded product (row 8547), including
    rows of zeros (amax floored at 1e-4), a block past E4M3's range, and blocks holding an infinity and a
    NaN beside finite and zero values."""
    from dew.nn.kernels.templates.gemm import quantized

    rng = np.random.default_rng(8550)
    values = rng.normal(size=(128, 256)).astype(np.float32)
    values[3] = 0.0
    values[5, :128] *= 1e4
    values[7, 0], values[7, 1:4] = np.inf, (1.0, 0.0, -2.0)  # an infinite block's codes: the recipe's
    values[9, 130], values[9, 131:134] = np.nan, (1.0, 0.0, -2.0)  # a NaN block's
    acc = jnp.asarray(values)
    rounded, codes, scales = quantized.blockwise_fp8_epilogue(jax.nn.relu)(acc)
    stored_scales = quantized.stored(scales, (128, True))
    want_codes, want_scales = projections.rows_quantized(jax.nn.relu(acc.astype(jnp.bfloat16)).astype(
        jnp.bfloat16))
    np.testing.assert_array_equal(np.asarray(rounded).view(np.uint16),
                                  np.asarray(acc.astype(jnp.bfloat16)).view(np.uint16))
    np.testing.assert_array_equal(np.asarray(codes).view(np.uint8), np.asarray(want_codes).view(np.uint8))
    np.testing.assert_array_equal(np.asarray(stored_scales).view(np.uint32),
                                  np.asarray(want_scales.T).view(np.uint32))


@pytest.mark.parametrize("rows", [200, 256])
def test_an_ungated_mlps_recipe_chain_is_its_two_recipe_projections_apart(rows):
    """Under a run's blockwise FP8 recipe, an ungated MLP's chain (`projections.recipe_chain`) of any rows
    count, laid in whole 128-row blocks and the padding dropped: its value and every gradient bitwise the two
    `recipe_projection` calls apart with the activation between them in XLA (row 8547); and outside the
    recipe, None, which the two apart take."""
    from dew.nn.precision import FP8, projection_role

    rng = np.random.default_rng(8551)
    x = jnp.asarray(rng.normal(size=(rows, 256)), jnp.bfloat16)
    up = jnp.asarray(rng.normal(size=(256, 384)) / 16, jnp.bfloat16)
    down = jnp.asarray(rng.normal(size=(384, 256)) / 16, jnp.bfloat16)
    cotangent = jnp.asarray(rng.normal(size=(rows, 256)), jnp.bfloat16)

    def chained(x, up, down):
        return projections.recipe_chain(x, up, down, jax.nn.gelu, out_dtype=jnp.bfloat16)

    def apart(x, up, down):
        hidden = projections.recipe_projection(x, up, out_dtype=jnp.bfloat16)
        return projections.recipe_projection(jax.nn.gelu(hidden).astype(jnp.bfloat16), down,
                                             out_dtype=jnp.bfloat16)

    assert chained(x, up, down) is None
    with generation.run_precision(FP8), projection_role("mlp"):
        fused = jax.jit(with_gradients(chained))(x, up, down, cotangent)
        separate = jax.jit(with_gradients(apart))(x, up, down, cotangent)
    blockwise_matrix.assert_bitwise_leaves(("value", "d_x", "d_up", "d_down"), fused, separate)


def _swiglu(gate, up):
    """A gated MLP's SwiGLU of its rounded gate and up, in fp32."""
    return jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)


@pytest.mark.parametrize("rows", [256, 384])
def test_a_gated_blockwise_chain_through_the_template_is_its_three_projections_apart(rows):
    """A gated MLP's three blockwise FP8 projections chained (`projections.blockwise_chain`'s `gate`), on an
    H100, interpreted, through the template named: the gate's product stored and read by the up projection's
    epilogue as a tile beside its sum, their SwiGLU quantized there as the down projection's rows; its value
    and every gradient bitwise the three projections apart with SwiGLU between them in XLA (row 12689)."""
    group_blocks = jnp.asarray([rows // 128], jnp.int32)
    rng = np.random.default_rng(12689)
    x = jnp.asarray(rng.normal(size=(rows, 256)), jnp.bfloat16)
    gate, up = (jnp.asarray(rng.normal(size=(1, 256, 384)) / 16, jnp.bfloat16) for _ in range(2))
    down = jnp.asarray(rng.normal(size=(1, 384, 256)) / 16, jnp.bfloat16)
    cotangent = jnp.asarray(rng.normal(size=(rows, 256)), jnp.bfloat16)

    def chained(x, gate, up, down):
        return projections.blockwise_chain(x, up, down, _swiglu, out_dtype=jnp.bfloat16, gate=gate,
                                           interpret_on_cpu=True)

    def apart(x, gate, up, down):
        def projected(rows, matrix):
            return projections.blockwise_fp8_projection(rows, matrix, group_blocks, out_dtype=jnp.bfloat16,
                                                        interpret_on_cpu=True)

        hidden = _swiglu(projected(x, gate), projected(x, up)).astype(jnp.bfloat16)
        return projected(hidden, down)

    with generation.assumed("sm90"), generation.chosen(blockwise_fp8="mosaic"), generation.routes() as routes:
        fused = jax.jit(with_gradients(chained))(x, gate, up, down, cotangent)
        separate = jax.jit(with_gradients(apart))(x, gate, up, down, cotangent)
    blockwise_matrix.assert_bitwise_leaves(("value", "d_x", "d_gate", "d_up", "d_down"), fused, separate)
    assert ("blockwise_fp8", "mosaic") in {(route.choice, route.kernel) for route in routes}


def test_a_gated_mlps_recipe_chain_is_its_three_recipe_projections_apart():
    """Under a run's blockwise FP8 recipe, a gated MLP's chain (`projections.recipe_chain`'s `gate`) of 200
    rows: its value and every gradient bitwise its three `recipe_projection` calls apart with SwiGLU between
    them in XLA (row 12689)."""

    rng = np.random.default_rng(12690)
    x = jnp.asarray(rng.normal(size=(200, 256)), jnp.bfloat16)
    gate, up = (jnp.asarray(rng.normal(size=(256, 384)) / 16, jnp.bfloat16) for _ in range(2))
    down = jnp.asarray(rng.normal(size=(384, 256)) / 16, jnp.bfloat16)
    cotangent = jnp.asarray(rng.normal(size=(200, 256)), jnp.bfloat16)

    _assert_gated_chain_is_apart(_swiglu, x, gate, up, down, cotangent)



@pytest.mark.parametrize("activation", ["nonzero at zero", "nonfinite derivative at zero"])
def test_a_padded_gated_recipe_chain_keeps_its_padding_out_of_every_gradient(activation):
    """A gated recipe chain of one row (127 padding rows in its 128-row block) through an activation whose
    padding is not inert, nonzero at zero (`where(gate == 0, 256, up)`, which would set the down kernel
    gradient's token-block scale and round the real row away) or of a nonfinite derivative there
    (`sqrt(gate) + up`): its value and every gradient bitwise the three recipe projections apart, the
    padding held out of the backward (`_chain_backward`'s `live`, row 12689's review)."""

    act = {"nonzero at zero": lambda gate, up: jnp.where(gate == 0, 256.0, up),
           "nonfinite derivative at zero": lambda gate, up: jnp.sqrt(gate) + up}[activation]
    rng = np.random.default_rng(12693)
    x = jnp.asarray(np.abs(rng.normal(size=(1, 256))), jnp.bfloat16)
    gate = jnp.asarray(np.abs(rng.normal(size=(256, 384))) / 16, jnp.bfloat16)
    up = jnp.asarray(rng.normal(size=(256, 384)) / 4096, jnp.bfloat16)
    down = jnp.asarray(rng.normal(size=(384, 256)) / 16, jnp.bfloat16)
    cotangent = jnp.asarray(rng.normal(size=(1, 256)), jnp.bfloat16)

    _assert_gated_chain_is_apart(act, x, gate, up, down, cotangent)


def _assert_gated_chain_is_apart(activation, x, gate, up, down, cotangent) -> None:
    """A gated recipe chain's value and every gradient (`projections.recipe_chain`'s `gate`) bitwise its
    three `recipe_projection` calls apart with `activation` between them in XLA, each finite, under a run's
    blockwise FP8 recipe (row 12689)."""
    from dew.nn.precision import FP8, projection_role

    def chained(x, gate, up, down):
        return projections.recipe_chain(x, up, down, activation, out_dtype=jnp.bfloat16, gate=gate)

    def apart(x, gate, up, down):
        def projected(rows, matrix):
            return projections.recipe_projection(rows, matrix, out_dtype=jnp.bfloat16)

        return projected(activation(projected(x, gate), projected(x, up)).astype(jnp.bfloat16), down)

    with generation.run_precision(FP8), projection_role("mlp"):
        fused = jax.jit(with_gradients(chained))(x, gate, up, down, cotangent)
        separate = jax.jit(with_gradients(apart))(x, gate, up, down, cotangent)
    blockwise_matrix.assert_bitwise_leaves(("value", "d_x", "d_gate", "d_up", "d_down"), fused, separate,
                                           finite=True)


@pytest.mark.parametrize("activation", ["nonzero at zero", "nonfinite derivative at zero"])
def test_a_padded_recipe_chain_keeps_its_padding_out_of_every_gradient(activation):
    """A recipe chain of one row (127 padding rows in its 128-row block) through an activation whose padding
    is not inert, nonzero at zero (`where(value == 0, 256, value)`, which would set the down kernel gradient's
    token-block scale and round the real row away) or of a nonfinite derivative there (`sqrt`): its value and
    every gradient bitwise the two recipe projections apart, the padding held out of the backward
    (`_chain_backward`'s `live`, row 12689's review)."""
    from dew.nn.precision import FP8, projection_role

    act = {"nonzero at zero": lambda value: jnp.where(value == 0, 256.0, value / 4096),
           "nonfinite derivative at zero": jnp.sqrt}[activation]
    rng = np.random.default_rng(12693)
    x = jnp.asarray(np.abs(rng.normal(size=(1, 256))), jnp.bfloat16)
    up = jnp.asarray(np.abs(rng.normal(size=(256, 384))) / 16, jnp.bfloat16)
    down = jnp.asarray(rng.normal(size=(384, 256)) / 16, jnp.bfloat16)
    cotangent = jnp.asarray(rng.normal(size=(1, 256)), jnp.bfloat16)

    def chained(x, up, down):
        return projections.recipe_chain(x, up, down, act, out_dtype=jnp.bfloat16)

    def apart(x, up, down):
        hidden = projections.recipe_projection(x, up, out_dtype=jnp.bfloat16)
        return projections.recipe_projection(act(hidden).astype(jnp.bfloat16), down, out_dtype=jnp.bfloat16)

    with generation.run_precision(FP8), projection_role("mlp"):
        fused = jax.jit(with_gradients(chained))(x, up, down, cotangent)
        separate = jax.jit(with_gradients(apart))(x, up, down, cotangent)
    blockwise_matrix.assert_bitwise_leaves(("value", "d_x", "d_up", "d_down"), fused, separate, finite=True)


@pytest.mark.parametrize("recipe", ["blockwise_fp8", "nvfp4"])
@pytest.mark.parametrize("gated", [False, True], ids=["ungated", "gated"])
def test_a_recipe_chains_products_carry_the_names_a_remat_policy_saves(recipe, gated):
    """A recipe chain given `names` (`blockwise_chain`, `nvfp4_chain`; `recipe_chain`'s, which `GatedMLP`
    passes) names its first projections' rounded products, the residuals its backward reads, as the
    projections apart name their outputs: differentiated under a remat policy saving those names, its
    program runs no more products than without remat, its backward recomputing neither first projection nor
    its quantizing epilogue; unnamed, the same policy recomputes them (row 12758)."""
    rng = np.random.default_rng(12758)
    x = jnp.asarray(rng.normal(size=(128, 128)), jnp.bfloat16)
    firsts = [jnp.asarray(rng.normal(size=(1, 128, 256)) / 16, jnp.bfloat16) for _ in range(1 + gated)]
    down = jnp.asarray(rng.normal(size=(1, 256, 128)) / 16, jnp.bfloat16)

    def activation(*values):
        return jax.nn.silu(values[0]) * values[-1] if gated else jax.nn.relu(values[0])

    def total(names):
        def run(x, *matrices):
            *gate, up, down = matrices
            options = {"gate": gate[0] if gate else None, "names": names, "out_dtype": jnp.bfloat16}
            if recipe == "nvfp4":
                scales = (jnp.float32(2.0 ** -6), jnp.float32(2.0 ** -8))
                out = projections.nvfp4_chain(x, up, down, activation, tensor_scales=scales, **options)
            else:
                out = projections.blockwise_chain(x, up, down, activation, **options)
            return jnp.sum(out.astype(jnp.float32))
        return run

    def products(fn) -> int:
        # The products the differentiated program runs, its sub-programs' too.
        graph = jax.make_jaxpr(jax.grad(fn, argnums=tuple(range(3 + gated))))(x, *firsts, down)
        return sum(equation.primitive.name in ("dot_general", "ragged_dot_general")
                   for equation in equations(graph, platform="cpu"))

    policy = jax.checkpoint_policies.save_only_these_names("gate", "up")
    plain = products(total(("gate", "up")))
    assert products(jax.checkpoint(total(("gate", "up")), policy=policy)) == plain
    assert products(jax.checkpoint(total(None), policy=policy)) > plain
