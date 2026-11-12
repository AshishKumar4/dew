"""The GEMM template and the Mosaic toolkit on the GPU (G14, the owner's rule: no unit tests on GPUs), run in
one process (`gpu_flow`), each cell its own file's property:

- A dense product's exact sums survive one row tile through Triton's MMA accumulation (test_gemm_template).
- CUTLASS runs the grouped projection as exactly as XLA wherever it is chosen, over every grouping that
  routes a row (test_grouped_gemm).
- The blockwise FP8 kernel gradient (rows contiguous, step scales) holds check 1 on an H100
  (test_gemm_blockwise).
- The template takes gpt-oss's 2880-wide projection, its ragged stores' one-row rungs, as exactly as XLA's
  (test_grouped_gemm; row 4474).
- A ring streams tiles of their own trip counts, consumers feeding their own ring split a step, and every
  read path of a warpgroup accumulator lands its products under a sign op on them (row 4346), on an H100
  (test_mosaic_toolkit).
- The shared GEMM mainloop runs on Blackwell's tcgen05, exactly, on a B200 or a B300, sm100 and sm103 alike,
  and its epilogues (a column bias, a per-row scale, two outputs of two dtypes) hold check 1 at each
  accumulator width and on a two-CTA pair (test_gemm_tcgen05).
- The NVFP4 product's packed kernel (its mainloop decoding each stage in place) holds every order on an
  H100 and an A100 (test_contracts_gemm).
- A grouped product whose producer gathers its rows from the tokens by a table (cp.async, row 1376) holds
  check 1 on an H100, plain and paired, the paired one's rows stored where a permutation says; a kernel
  gradient reading its tokens and its cotangent through tables is bitwise its gathered rows' on an H100;
  and on a B200 tcgen05's load warp gathers by TMA's four-row gathers, bitwise the rows gathered first
  (test_gemm_gather).
- The template's grouped product of 16-bit operands sums into fp32 as exactly as XLA's at depths 128 and
  512, its sums the tensor cores' own at the default precision (test_grouped_gemm; M-EM-3).
- The toolkit's warpgroup scan walks a linear recurrence forward and reversed, each position's state or
  the one before it, as float64 does (test_warpgroup_scan).
- GEMM's check-1 matrix (tests/kernel_matrix.py): the template route's training step on par with the reference
  path's, dense and grouped, each matrix layout, hook and dtype its served pairs name (gemm_matrix).
- The fused GEMM's check-1 matrix (tests/kernel_matrix.py): a region's training step through the fuser's
  prologue and epilogue on par with the same region's stages in XLA (fused_gemm_matrix).
- The joined projections' check-1 matrix (tests/kernel_matrix.py): a gated MLP's gate and up projections, or a
  gated delta net's b and a, run as one GEMM, the layer's training step on par with the projections apart
  (join_matrix).
- The NVFP4, MX, blockwise FP8 and adapted products' check-1 matrices (tests/kernel_matrix.py). Each matrix
  runs here over the combinations of the axes a GPU runs natively, its kernel or form and precision, and the
  cases a red found on a GPU (`kernel_matrix.native_cells`); its other pairs are the CPU's.
"""

import functools

import jax
import jax.numpy as jnp
import kernel_matrix
import numpy as np
import pytest
import test_contracts_gemm as contracts_gemm
import test_gemm_blockwise as gemm_blockwise
import test_gemm_gather as gemm_gather
import test_gemm_tcgen05 as gemm_tcgen05
import test_gemm_template as gemm_template
import test_gemm_widths as gemm_widths
import test_grouped_gemm as grouped_gemm
import test_mosaic_toolkit as mosaic_toolkit
import test_warpgroup_scan as warpgroup_scan
from gpu_flow import Cell, run_cells, when
from jax.experimental.pallas import fuser
from reference_error import assert_as_exact

from dew.nn.kernels import fused_gemm, generation
from dew.nn.norms import rms_normalized

pytestmark = pytest.mark.gpu(reason="the GEMM template's and the Mosaic toolkit's kernels, compiled")


def _fused_projection_training():
    """One routed projection update, including an empty expert and rows beyond the routed ones. The
    input cotangent exercises both paths through RMS normalization, and the updated parameters run
    another forward. The reference is the plain program main's GEMM tests take: the RMS norm as a model
    writes it and XLA's ragged product (`test_gemm_template`), outside the template."""
    rows, depth, width = 256, 256, 256
    sizes = (0, 100, 150)
    groups = jnp.asarray(sizes, jnp.int32)

    def native(x, params):
        normalized = rms_normalized(x, params["norm"], 1e-6, x.dtype, scale_offset=False,
                                    scale_after_cast=False, fp32_statistics=True, implementation="xla")
        projected = fused_gemm.fusible_gemm(normalized, params["weight"].astype(x.dtype), groups)
        return jax.nn.silu(projected + params["bias"]) + x[:, :width].astype(jnp.float32)

    def reference(x, params):
        work = jnp.promote_types(x.dtype, jnp.float32)
        normalized = gemm_template.rms_norm(x, params["norm"])
        matrices = params["weight"].astype(x.dtype).astype(work)
        # Rows past the routed ones take no expert: ragged_dot leaves them zero.
        projected = jax.lax.ragged_dot(normalized.astype(work), matrices, groups, preferred_element_type=work)
        return jax.nn.silu(projected + params["bias"]) + x.astype(work)[:, :width]

    def step(forward):
        def update(x, params, targets):
            def loss(x, params):
                values = forward(x, params)
                return jnp.mean(jnp.square(values - targets)), values

            (value, outputs), (dx, gradients) = jax.value_and_grad(
                loss, argnums=(0, 1), has_aux=True)(x, params)
            updated = jax.tree.map(lambda p, g: p - 0.25 * g, params, gradients)
            after = forward(x, updated)
            return value, outputs, dx, gradients, updated, jnp.mean(jnp.square(after - targets))
        return update

    def draw(seed):
        rng = np.random.default_rng(seed)
        x = jnp.asarray(rng.normal(size=(rows, depth)), jnp.bfloat16).at[:2].set(0)
        # Master weights begin at representable operand values; the update itself stays in fp32.
        weight = jnp.asarray(rng.normal(size=(len(sizes), depth, width)) / np.sqrt(depth),
                             jnp.bfloat16).astype(jnp.float32)
        params = {"weight": weight, "norm": jnp.asarray(rng.uniform(0.8, 1.2, depth), jnp.float32),
                  "bias": jnp.asarray(rng.normal(size=width) * 0.1, jnp.float32)}
        return x, params, jnp.asarray(rng.normal(size=(rows, width)), jnp.float32)

    return step(fuser.fuse(native)), step(reference), draw


def a_fused_projection_trains_its_producer_and_epilogue():
    native, reference, draw = _fused_projection_training()
    with generation.chosen(gemm_prologue="fused", gemm_epilogue="fused"):
        assert_as_exact(jax.jit(native), jax.jit(reference), draw,
                        "fused projection producer and epilogue update", seeds=8, truth_fn=reference)


def test_the_gemm_kernels_run_on_this_gpu():
    """Every GEMM and toolkit cell above on this GPU (`gpu_flow.run_cells`), each one's failure kept and all
    raised together; the toolkit's on an H100 alone."""
    run_cells([
        Cell("a fused projection trains its producer and epilogue",
             a_fused_projection_trains_its_producer_and_epilogue, "sm90"),
        Cell("a dense product's exact sums survive one row tile",
             when(generation.triton_runs(), "needs a GPU the Pallas kernels compile for",
                  gemm_template.a_dense_products_exact_sums_survive_one_row_tile)),
        *(Cell(f"CUTLASS runs the projection as exactly as XLA where chosen: {sizes}",
               functools.partial(grouped_gemm.cutlass_runs_the_projection_as_exactly_as_xla_where_chosen,
                                 sizes))
          for sizes in grouped_gemm.SIZES if any(grouped_gemm.SIZES[sizes])),
        *(Cell(f"the template sums 16-bit products into fp32 as exactly as XLA: depth {depth}",
               functools.partial(grouped_gemm.the_template_sums_16_bit_products_into_fp32_as_exactly_as_xla,
                                 depth))
          for depth in (128, 512)),
        Cell("the template takes a gpt-oss width as exactly as XLA",
             grouped_gemm.the_template_takes_a_gpt_oss_width_as_exactly_as_xla),
        *(Cell(f"the template runs widths no tile divides: {depth}x{width}",
               functools.partial(gemm_widths.unaligned_widths_run_on_the_template, depth, width,
                                 interpret=False))
          for depth, width in gemm_widths.WIDTHS),
        Cell("the blockwise FP8 kernel gradient is on par",
             gemm_blockwise.a_blockwise_kernel_gradient_is_on_par, "sm90"),
        Cell("the blockwise FP8 kernel gradient's portable arm sums exactly",
             gemm_blockwise.a_blockwise_kernel_gradients_portable_arm_sums_exactly),
        *(Cell(f"a gathered grouped product is on par{': paired and scattered' if paired else ''}",
               functools.partial(gemm_gather.a_gathered_product_is_on_par, paired=paired, scattered=paired),
               "sm90")
          for paired in (False, True)),
        *(Cell(f"a gathered kernel gradient is its gathered rows' gradient: {splits} splits",
               functools.partial(gemm_gather.a_gathered_kernel_gradient_is_its_gathered_rows_gradient,
                                 splits=splits), "sm90")
          for splits in (1, 2)),
        Cell("a Blackwell's gathered product is its gathered rows' product",
             gemm_gather.a_blackwells_gathered_product_is_its_gathered_rows_product, "sm100"),
        *(Cell(f"CUTLASS runs a mapped projection as exactly as XLA where chosen: "
               f"{'shared' if shared else 'mapped'} matrices",
               functools.partial(grouped_gemm.cutlass_runs_a_mapped_projection_as_exactly_as_xla_where_chosen,
                                 shared))
          for shared in (False, True)),
        *(Cell(f"a ring streams tiles of their own trip counts: {consumers} consumers",
               functools.partial(mosaic_toolkit.a_ring_streams_tiles_of_their_own_trip_counts_on_an_h100,
                                 consumers), "sm90")
          for consumers in (1, 2)),
        *(Cell(f"every warpgroup read lands its products: {path}",
               functools.partial(mosaic_toolkit.every_warpgroup_read_lands_its_products_on_an_h100, path),
               "sm90")
          for path in mosaic_toolkit.READS),
        *(Cell(f"consumers feeding their own ring split a step: {consumers} consumers from {lo}",
               functools.partial(mosaic_toolkit.consumers_feeding_their_own_ring_split_a_step_on_an_h100,
                                 consumers, lo), "sm90")
          for consumers, lo in ((1, 0), (2, 0), (2, 3))),
        Cell("the shared GEMM mainloop runs on Blackwell's tcgen05",
             gemm_tcgen05.the_shared_mainloop_runs_exactly_on_blackwell, "sm100"),
        *(Cell(f"tcgen05's epilogues hold check 1: {case} at {tiles.family} {tiles.block_n}x{tiles.stages}",
               functools.partial(gemm_tcgen05.tcgen05s_epilogues_hold_check_1_on_blackwell, case, tiles),
               "sm100")
          for tiles in (*gemm_tcgen05.TILES, gemm_tcgen05.PAIRED) for case in gemm_tcgen05.EPILOGUES),
        *(Cell(f"the NVFP4 product's packed kernel holds every order: {name} {edge}",
               functools.partial(contracts_gemm.test_an_nvfp4_products_kernels_hold_every_order, edge,
                                 "packed"), name)
          for name in ("sm90", "sm80") for edge in contracts_gemm.EDGES),
        *(Cell(f"the warpgroup scan walks the recurrence: {positions} positions"
               f"{' reversed' if reverse else ''}{' exclusive' if exclusive else ''}",
               functools.partial(warpgroup_scan.the_scan_walks_the_recurrence_on_this_gpu, positions, reverse,
                                 exclusive))
          for positions in warpgroup_scan.POSITIONS for reverse in (False, True)
          for exclusive in (False, True)),
    ])


@pytest.mark.users_process
def test_the_gemm_kernels_run_on_this_gpu_as_a_users_process_runs_them():
    """The GEMM cells whose claim is what a user's process computes, run in one (`users_process`: no matmul
    precision set, XLA's own schedules and autotuning as `lane_environment.USERS_GPU_FLAGS` state them), not
    in the suite's, whose JAX_DEFAULT_MATMUL_PRECISION=highest has gemm.gemm promote each step's sum: the
    gathered product at the default precision, its default form (the tensor cores' running sum) against the
    selector's XLA arm as a user's process compiles it, at the cooperative tile a user can pin and at the
    tile it derives. Its bare running sum read 1.658x a Triton fusion's error under the suite's flags on a
    GH200 (M-EM-19); the default form's contract is the fastest form passing check 1 (kernel-templates.md,
    testing.md), what a user's process gets."""
    run_cells([
        Cell(f"a gathered grouped product is on par in a user's process: "
             f"{'pinned' if pinned else 'derived'}{', paired and scattered' if paired else ''}",
             functools.partial(gemm_gather.a_gathered_product_is_on_par, paired=paired, scattered=paired,
                               pinned=pinned), "sm90")
        for pinned in (True, False) for paired in (False, True)])


def test_the_gemm_matrix_holds_on_this_gpu():
    """GEMM's check-1 matrix (tests/kernel_matrix.py): the template route's training step on par with the
    reference path's (XLA's ragged dot), at each combination of the axes a GPU runs natively
    (`kernel_matrix.native_cells`), each cell on its own GPU."""
    run_cells(kernel_matrix.native_cells("gemm"))


def test_the_fused_gemm_matrix_holds_on_this_gpu():
    """The fused GEMM's check-1 matrix (tests/kernel_matrix.py): a region's training step with its producer
    and consumers fused into the template's stages on par with the same region's stages and product in XLA, at
    each combination of the axes a GPU runs natively (`kernel_matrix.native_cells`), each cell on its own
    GPU."""
    run_cells(kernel_matrix.native_cells("fused_gemm"))


def test_the_nvfp4_matrix_holds_on_this_gpu():
    """The NVFP4 product's check-1 matrix (tests/kernel_matrix.py): its training step with the matrix decoded
    in the mainloop on par with it decoded once a call, at each combination of the axes a GPU runs natively
    (`kernel_matrix.native_cells`), each cell on its own GPU."""
    run_cells(kernel_matrix.native_cells("nvfp4"))


def test_the_mx_matrix_holds_on_this_gpu():
    """The MX product's check-1 matrix (tests/kernel_matrix.py): its forward through the template's block-
    scaled products on par with the decoded operands' product in XLA, at each combination of the axes a GPU
    runs natively (`kernel_matrix.native_cells`), each cell on its own GPU."""
    run_cells(kernel_matrix.native_cells("mxfp8"))


def test_the_blockwise_matrix_holds_on_this_gpu():
    """The blockwise-scaled FP8 product's check-1 matrix (tests/kernel_matrix.py): its forward through the
    template's step-scaled walk on par with the scaled blocks summed in XLA, at each combination of the axes a
    GPU runs natively (`kernel_matrix.native_cells`), each cell on its own GPU."""
    run_cells(kernel_matrix.native_cells("blockwise_fp8"))


def test_the_low_rank_matrix_holds_on_this_gpu():
    """The adapted product's check-1 matrix (tests/kernel_matrix.py): a LoRA-adapted projection's training
    step with its base product and update fused on par with them apart, at each combination of the axes a GPU
    runs natively (`kernel_matrix.native_cells`), each cell on its own GPU."""
    run_cells(kernel_matrix.native_cells("low_rank"))


def test_the_join_matrix_holds_on_this_gpu():
    """The joined projections' check-1 matrix (tests/kernel_matrix.py): a layer's training step with two of
    its projections run as one GEMM on par with the projections apart, at each combination of the axes a GPU
    runs natively (`kernel_matrix.native_cells`), each cell on its own GPU."""
    run_cells(kernel_matrix.native_cells("join"))
