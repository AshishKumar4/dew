"""Attention on the GPU (G14, the owner's rule: no unit tests on GPUs): each attention route whose kernel
runs on a GPU alone, run in one process (`gpu_flow`), each cell its own file's property:

- The frozen Triton template at every config of its forward's and its backward's spaces, bf16, every hook,
  over causal documents under a window and the irregular calls (test_attention_triton); its default form
  finite where the reference is at fp32's edges for the families too slow to interpret
  (test_attention_stability).
- The Mosaic mainloop's default form, and its backward in two launches, finite where the reference is at
  fp32's edges over every family it takes, on an H100 (test_attention_stability).
- A dropping route call on a GPU runs finite (test_attention_dropout).
- A decode step's sinks take their gradient and tangent through the dispatch, plain and softcapped; the
  decode kernel's attention differentiates as the folded attention (test_decode_attention).
- Fused routes, without XLA's deterministic ops, which Dew refuses them under: flash reads a cache call's
  documents as exactly as jax.nn reads its mask (test_decode_attention), and packed windowed bf16
  attention runs its band on par (test_local_attention).
"""

import functools

import pytest
import test_attention_dropout as dropout
import test_attention_stability as stability
import test_attention_triton as triton
import test_decode_attention as decode
import test_local_attention as local
from gpu_flow import Cell, run_cells

pytestmark = pytest.mark.gpu(reason="each route's GPU kernel: Triton, Mosaic, flash and the band")

CASES = ("causal-documents-window", *triton.IRREGULAR)
"""The Triton sweeps' calls (`triton.every_config_on_a_gpu`)."""


def sweep(check, space) -> Cell:
    """`check` at every config of `space` over every call of CASES, as one cell: each config its own
    inner cell, every failure kept (`run_cells`)."""
    return Cell(f"{check.__name__.replace('_', ' ')}: {len(space)} configs by {len(CASES)} calls",
                lambda: run_cells([Cell(f"{case} {config}", functools.partial(check, config, case))
                                   for case in CASES for config in space]))


def test_the_attention_kernels_run_on_this_gpu():
    """Every kernel cell above on this GPU (`gpu_flow.run_cells`), each one's failure kept and all raised
    together; the Mosaic cells on an H100 alone."""
    run_cells([
        sweep(triton.every_config_on_a_gpu, triton.attention_triton.SPACE),
        sweep(triton.every_backward_config_on_a_gpu, triton.attention_triton.BACKWARD_SPACE),
        *(Cell(f"the frozen Triton mainloop's default is finite at fp32's edges: {family}",
               functools.partial(stability.default_is_finite, family, "triton"))
          for family in stability.GPU_TRITON_FAMILIES),
        *(Cell(f"the Mosaic mainloop's default is finite at fp32's edges: {family}",
               functools.partial(stability.default_is_finite, family, "mosaic"), "sm90")
          for family in stability.MOSAIC_FAMILIES),
        *(Cell(f"the Mosaic backward in two launches is finite at fp32's edges: {family}",
               functools.partial(stability.default_is_finite, family, "mosaic two launches"), "sm90")
          for family in stability.HOOKLESS_FAMILIES),
        Cell("a dropping route call on a GPU runs finite",
             dropout.a_dropping_route_call_on_a_gpu_runs_finite),
        *(Cell(f"a decode step's sinks take their gradient and tangent through the dispatch: {name}",
               functools.partial(
                   decode.a_decode_steps_sinks_take_their_gradient_and_tangent_through_the_dispatch, softcap))
          for name, softcap in (("plain", None), ("softcap", decode.SOFTCAP))),
        Cell("the decode kernel's attention differentiates as the folded attention",
             decode.attention_the_decode_kernel_takes_differentiates_as_the_folded_attention),
    ])


def test_the_fused_attention_routes_run_on_this_gpu(without_deterministic_ops):
    """Every fused route's cell above on this GPU, without XLA's deterministic ops
    (`without_deterministic_ops`); the band on a bf16 GPU, sm80 or later."""
    run_cells([
        *(Cell(f"flash reads a cache call's documents as exactly as jax.nn reads its mask: window {window}",
               functools.partial(decode.flash_reads_a_cache_calls_documents_as_exactly_as_jax_nn_reads_its_mask,
                                 without_deterministic_ops, window))
          for window in (None, 4)),
        Cell("packed windowed bf16 attention runs its band on par",
             functools.partial(local.packed_windowed_bf16_attention_runs_its_band_on_par,
                               without_deterministic_ops), "sm80"),
    ])
