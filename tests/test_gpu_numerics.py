"""A GPU's numerics and its kernels' configs (G14, the owner's rule: no unit tests on GPUs), run in one
process (`gpu_flow`), each cell its own file's property:

- The GPU divides fp32 to the nearest float, as `import dew` sets (test_precision_policy).
- A frozen FP8 site's products take no FP8 GEMM whose sum is narrower than fp32's, on a GPU of FP8
  tensor cores (test_fp8_base).
- The audio encoders compile with their mask built in the same jit, no strided slice on a dot's operands
  (test_audio).
- Every config of the norm and SwiGLU kernels holds the float64 gate (test_norm_swiglu), and every tile
  of the grouped matmuls, the Mosaic ones on an H100 (test_moe_precision).
- A GPU host's CPU runs the seams its placement chooses and a projection without the Triton kernels
  (test_norm_swiglu, test_moe_precision).
- A GPU reports its block, multiprocessors, clock and memory bandwidth (test_lowering).
- A Gemma-sized vocabulary's chunked cross entropy is as exact as the whole logits, its float64 truth at
  the seeds it needs (test_chunked_cross_entropy).
"""

import functools

import jax.numpy as jnp
import pytest
import test_audio as audio
import test_chunked_cross_entropy as chunked_cross_entropy
import test_fp8_base as fp8_base
import test_lowering as lowering
import test_moe_precision as moe_precision
import test_norm_swiglu as norm_swiglu_tests
import test_precision_policy as precision_policy
from gpu_flow import Cell, run_cells, when

from dew.nn.kernels import grouped_matmul, norm_swiglu
from dew.nn.kernels.generation import triton_runs

pytestmark = pytest.mark.gpu(reason="a GPU's own numerics, its kernels' configs and its driver's facts")

DTYPES = (jnp.float32, jnp.bfloat16, jnp.float16)


def test_a_gpus_numerics_and_kernels_hold_on_this_gpu(tmp_path_factory):
    """Every numerics and config cell above on this GPU (`gpu_flow.run_cells`), each one's failure kept and
    all raised together."""
    pallas = "needs a GPU the Pallas kernels compile for"
    run_cells([
        Cell("the GPU divides fp32 to the nearest float",
             precision_policy.the_gpu_divides_fp32_to_the_nearest_float),
        Cell("a frozen FP8 site's products take no FP8 GEMM whose sum is narrower",
             when(fp8_base.FP8_CORES, "this GPU has no FP8 tensor cores", lambda: (
                 fp8_base.a_frozen_sites_products_take_no_fp8_gemm_whose_sum_is_narrower(
                     tmp_path_factory.mktemp("fp8"))))),
        *(Cell(f"the {kind} audio encoder compiles with its mask built in the same jit",
               lambda kind=kind: audio.encoder_compiles_on_gpu_with_the_mask_built_in_the_same_jit(
                   audio._audio(kind)))
          for kind in ("gemma3n", "gemma4")),
        *(Cell(f"every config of {kernel.__name__} holds the float64 gate: {dtype.__name__} {operands}",
               functools.partial(norm_swiglu_tests.every_config_of_the_norm_kernels_holds_the_float64_gate,
                                 kernel, dtype, operands))
          for kernel in (norm_swiglu.RmsForward, norm_swiglu.RmsBackward) for dtype in DTYPES
          for operands in ("full step", "wide rows", "narrow rows")),
        *(Cell(f"every config of {kernel.__name__} holds the float64 gate: {dtype.__name__}",
               functools.partial(norm_swiglu_tests.every_config_of_the_swiglu_kernels_holds_the_float64_gate,
                                 kernel, dtype))
          for kernel in (norm_swiglu.SwigluForward, norm_swiglu.SwigluBackward) for dtype in DTYPES),
        *(Cell(f"every tile of {kernel.__name__} holds the float64 gate",
               when(triton_runs(), pallas, functools.partial(
                   moe_precision.every_tile_of_the_grouped_kernels_holds_the_float64_gate,
                   kernel, width, features)),
               generation)
          for kernel, width, features, generation in (
              (grouped_matmul.Gmm, 192, 320, None), (grouped_matmul.Tgmm, 192, 320, None),
              (grouped_matmul.MosaicGmm, 768, 768, "sm90"), (grouped_matmul.MosaicTgmm, 768, 768, "sm90"))),
        *(Cell(f"a projection on a GPU host's CPU runs without the Triton kernels: {implementation}",
               when(triton_runs(), pallas, functools.partial(
                   moe_precision.a_projection_on_the_cpu_of_a_gpu_host_runs_without_the_triton_kernels,
                   implementation)))
          for implementation in ("auto", "pallas")),
        Cell("a GPU reports its block and multiprocessors whether or not a row names it",
             lowering.a_gpu_reports_its_block_and_multiprocessors_whether_or_not_a_row_names_it),
        Cell("a GPU reports its clock and memory bandwidth from its driver",
             lowering.a_gpu_reports_its_clock_and_memory_bandwidth_from_its_driver),
        *(Cell(f"a Gemma-sized vocabulary is as exact as the whole logits: {chunks} chunks, {softcap}",
               functools.partial(chunked_cross_entropy.a_gemma_sized_vocabulary_is_as_exact_as_the_whole_logits,
                                 chunks, softcap))
          for chunks in (4, 16) for softcap in (None, 30.0)),
    ])


def test_a_gpu_hosts_cpu_runs_the_seams_its_placement_chooses(without_deterministic_ops):
    """The norm seams on a GPU host's CPU, as their placement chooses them, without XLA's deterministic
    ops (`without_deterministic_ops`)."""
    run_cells([
        Cell(f"a GPU host's CPU runs the seams its placement chooses: {implementation}",
             functools.partial(norm_swiglu_tests.a_gpu_hosts_cpu_runs_the_seams_its_placement_chooses,
                               without_deterministic_ops, implementation))
        for implementation in ("auto", "pallas")])
