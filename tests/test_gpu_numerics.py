"""A GPU's numerics and its kernels' configs (G14, the owner's rule: no unit tests on GPUs), run as flows
(`gpu_flow`), one a group of cells (`GROUPS`), each cell its own file's property:

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
- A Dense layer's float64 terms on a GPU host are its inputs' placement's own, CPU or GPU, in either order
  (test_float64_terms).
- A multiply-add marked to round alone rounds each step, as precision's unbarriered transforms take it
  (test_no_contract).
- APG's projection at its range's edges is Diffusers' float64 projection's, as on a CPU (test_guidance).
- A fusion reading a value directly and through a reversed view computes its exact sum (test_view_buffers).
"""

import functools

import jax.numpy as jnp
import pytest
import test_audio as audio
import test_chunked_cross_entropy as chunked_cross_entropy
import test_float64_terms as float64_terms_tests
import test_fp8_base as fp8_base
import test_guidance as guidance_tests
import test_lowering as lowering
import test_moe_precision as moe_precision
import test_no_contract as no_contract
import test_norm_swiglu as norm_swiglu_tests
import test_precision_policy as precision_policy
import test_view_buffers as view_buffers
from gpu_flow import Cell, covering, isolated, run_cells, when

from dew.nn.kernels import grouped_matmul, grouped_triton, norm_swiglu
from dew.nn.kernels.generation import triton_runs

pytestmark = pytest.mark.gpu(reason="a GPU's own numerics, its kernels' configs and its driver's facts")

DTYPES = (jnp.float32, jnp.bfloat16, jnp.float16)


OWN = "the GPU's own arithmetic, facts or placement, which a CPU does not have"
"""Why a numerics cell stays on the GPU: what it holds is the device's own."""

PALLAS = "needs a GPU the Pallas kernels compile for"

ON_THE_HOST = "its programs placed on the host's CPU, whose kernels a prepare would choose as on its GPU"
"""Why a cell placed on a GPU host's CPU compiles at its run (`Cell.at_run`): a bitwise check, no check-1
site, its programs the CPU's, which a GPU tier part's prepare, choosing every kernel as on its node's GPU
(`truths_ahead.prepared`), lowered as Triton kernels on the CPU, where they raised (row 10764)."""


def _flow_scaling():
    """`test_flow_grpo`'s flow-scaling cell, its module imported only where the cell runs (row 8971)."""
    from test_flow_grpo import a_flows_scaling_rounds_a_subnormal_result_once

    a_flows_scaling_rounds_a_subnormal_result_once()


def numerics_cells(tmp_path_factory) -> list[Cell]:
    """The GPU's own arithmetic and driver facts, its encoders' compiles and a Dense layer's float64 terms."""
    return [
        Cell("the GPU divides fp32 to the nearest float",
             precision_policy.the_gpu_divides_fp32_to_the_nearest_float, fault=OWN),
        Cell("a correctly rounded division is IEEE division's at its edges on a GPU",
             precision_policy.divides_as_ieee_division_at_its_edges, fault=OWN),
        Cell("a correctly rounded division is IEEE division's on a GPU dividing approximately",
             precision_policy.divides_as_ieee_division_on_an_approximate_divider, fault=OWN,
             at_run="a fresh process of its own flags"),
        Cell("a power of two's scaling rounds a subnormal result once on a GPU",
             precision_policy.power2_rounds_a_subnormal_result_once, fault=OWN),
        Cell("a flow's scaling rounds a subnormal result once on a GPU", _flow_scaling, fault=OWN),
        Cell("APG holds its projection at its range's edges on a GPU",
             guidance_tests.apg_holds_its_projection_at_its_ranges_edges, fault=OWN),
        Cell("a multiply-add marked to round alone rounds each step on a GPU",
             no_contract.a_marked_multiply_add_rounds_each_step, fault=OWN),
        *(Cell(f"a value read through a reversed view is summed exactly on a GPU: {program}",
               functools.partial(
                   view_buffers.a_value_read_through_a_reversed_view_is_summed_exactly, program),
               fault=OWN)
          for program in view_buffers.PROGRAMS),
        *(Cell(f"a compensated primitive under jit is its eager arithmetic on a GPU: {shape}",
               functools.partial(precision_policy.compensated_primitives_hold_under_jit, shape), fault=OWN)
          for shape in precision_policy.COMPENSATED_SHAPES),
        Cell("a frozen FP8 site's products take no FP8 GEMM whose sum is narrower",
             when(fp8_base.FP8_CORES, "this GPU has no FP8 tensor cores", isolated(tmp_path_factory)(
                 fp8_base.a_frozen_sites_products_take_no_fp8_gemm_whose_sum_is_narrower, "fp8")), fault=OWN),
        *(Cell(f"the {kind} audio encoder compiles with its mask built in the same jit",
               lambda kind=kind: audio.encoder_compiles_on_gpu_with_the_mask_built_in_the_same_jit(
                   audio.audio_on_host(kind)), native={"compile": f"the {kind} encoder's mask in its jit"})
          for kind in ("gemma3n", "gemma4")),
        Cell("a GPU reports its block and multiprocessors whether or not a row names it",
             lowering.a_gpu_reports_its_block_and_multiprocessors_whether_or_not_a_row_names_it, fault=OWN,
             at_run="its GPU's own facts, which a compile-only device does not report"),
        Cell("a GPU reports its clock and memory bandwidth from its driver",
             lowering.a_gpu_reports_its_clock_and_memory_bandwidth_from_its_driver, fault=OWN,
             at_run="its CUDA driver's facts, read on its GPU"),
        # A GPU process walks a truth's terms where its values lie, from either placement in either order
        # (`float64_terms.evaluated_on`: an unprepared site's, a steered walk's), each compiled evaluation
        # keyed by its platform (row 8531).
        Cell("a Dense layer's float64 terms are its placement's own on a GPU host",
             isolated(tmp_path_factory)(
                 float64_terms_tests.a_dense_layers_terms_are_its_placements_own_on_a_gpu_host, "placed"),
             fault=OWN, at_run="fresh processes of each placement"),
    ]


def norm_cells(_) -> list[Cell]:
    """Every config of the norm and SwiGLU kernels."""
    return [
        *(Cell(f"every config of {kernel.__name__} holds the float64 gate: {dtype.__name__} {operands}",
               functools.partial(norm_swiglu_tests.every_config_of_the_norm_kernels_holds_the_float64_gate,
                                 kernel, dtype, operands),
               # Its operands native: a step's rows, rows two tiles wide and rows for the tallest tiles
               # each take launches of their own.
               native={"kernel": kernel.__name__, "dtype": dtype.__name__, "operands": operands})
          for kernel in (norm_swiglu.RmsForward, norm_swiglu.RmsBackward) for dtype in DTYPES
          for operands in ("full step", "wide rows", "narrow rows")),
        *(Cell(f"every config of {kernel.__name__} holds the float64 gate: {dtype.__name__}",
               functools.partial(norm_swiglu_tests.every_config_of_the_swiglu_kernels_holds_the_float64_gate,
                                 kernel, dtype), native={"kernel": kernel.__name__, "dtype": dtype.__name__})
          for kernel in (norm_swiglu.SwigluForward, norm_swiglu.SwigluBackward) for dtype in DTYPES),
        Cell("a norm with Dew's silu after it holds the float64 gate",
             norm_swiglu_tests.a_norm_with_dews_silu_after_it_holds_the_float64_gate),
    ]


def grouped_cells(_) -> list[Cell]:
    """Every tile of the grouped matmuls, and a projection on a GPU host's CPU without their kernels."""
    return [
        *(Cell(f"every tile of {kernel.__name__} holds the float64 gate",
               when(triton_runs(), PALLAS, functools.partial(
                   moe_precision.every_tile_of_the_grouped_kernels_holds_the_float64_gate,
                   kernel, width, features)),
               generation, native={"kernel": kernel.__name__})
          for kernel, width, features, generation in (
              (grouped_triton.Gmm, 192, 320, None), (grouped_triton.Tgmm, 192, 320, None),
              (grouped_matmul.MosaicGmm, 768, 768, "sm90"), (grouped_matmul.MosaicTgmm, 768, 768, "sm90"))),
        *projection_cells(),
    ]


def projection_cells() -> list[Cell]:
    """A projection on a GPU host's CPU, without the grouped matmuls' kernels."""
    return [Cell(f"a projection on a GPU host's CPU runs without the Triton kernels: {implementation}",
                 when(triton_runs(), PALLAS, functools.partial(
                     moe_precision.a_projection_on_the_cpu_of_a_gpu_host_runs_without_the_triton_kernels,
                     implementation)), fault=OWN, at_run=ON_THE_HOST)
            for implementation in ("auto", "pallas")]


def vocabulary_cells(_) -> list[Cell]:
    """A Gemma-sized vocabulary's chunked cross entropy, every chunking in one check."""
    check = chunked_cross_entropy.a_gemma_sized_vocabulary_is_as_exact_as_the_whole_logits
    return [Cell(f"a Gemma-sized vocabulary is as exact as the whole logits at "
                 f"{chunked_cross_entropy.CHUNKS} chunks: {softcap}", functools.partial(check, softcap),
                 native={"logits": "plain" if softcap is None else "softcapped"})
            for softcap in (None, 30.0)]


GROUPS = {"numerics": numerics_cells, "norms": norm_cells, "grouped": grouped_cells,
          "vocabulary": vocabulary_cells}
"""The cells above by group, each group one flow: a tier runs the groups as parts of their own."""


@pytest.mark.parametrize("group", list(GROUPS))
def test_a_gpus_numerics_and_kernels_hold_on_this_gpu(group, tmp_path_factory):
    """Every cell of `group` on this GPU (`gpu_flow.run_cells`), each one's failure kept and all raised
    together."""
    run_cells(covering(GROUPS[group](tmp_path_factory)))


def test_a_gpu_hosts_cpu_runs_the_seams_its_placement_chooses(without_deterministic_ops):
    """The norm seams on a GPU host's CPU, as their placement chooses them, without XLA's deterministic
    ops (`without_deterministic_ops`)."""
    run_cells(seams_cells(without_deterministic_ops))


def seams_cells(without_deterministic_ops) -> list[Cell]:
    """The norm seams on a GPU host's CPU, each implementation a cell."""
    return [Cell(f"a GPU host's CPU runs the seams its placement chooses: {implementation}",
                 functools.partial(norm_swiglu_tests.a_gpu_hosts_cpu_runs_the_seams_its_placement_chooses,
                                   without_deterministic_ops, implementation), at_run=ON_THE_HOST)
            for implementation in ("auto", "pallas")]
