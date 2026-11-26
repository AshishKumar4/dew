"""The torchax fallback's GPU flow (G14, the owner's rule: no unit tests on GPUs), apart from the other
flows: torchax renames torch's privateuse1 backend process-wide (the `torchax` mark), so its cells share
no process with the torch-using ones (`gpu_flow`):

- A float32 torchax model's logits and its output tables are as exact as torch's under the GPU's TF32
  default (test_torchax_fallback).
- A tier-3 source loads and runs in a process of the GPU alone, with no CPU backend (test_torchax_fallback).
- A bf16 tier-3 Qwen3-Next steps on a Hopper through Dew's native norm, SiLU-product and delta-rule kernels,
  its loss and gradients as exact as transformers' (test_torchax_fallback, row 13280).
"""

import pytest
import test_torchax_fallback as fallback
from gpu_flow import Cell, run_cells

pytestmark = [pytest.mark.gpu(reason="TF32 is a GPU's default float32 product"), pytest.mark.torchax]

AT_RUN = "torchax models and a tier-3 source, no check-1 site"
"""Why the first two cells compile at their run (`Cell.at_run`)."""
PLACED_AT_LOAD = ("a tier-3 load places the source's tensors on the device as it loads "
                  "(MakeArraysFromHostBufferShards), which a prepare's compile-only device refuses")
"""Why the bf16 Qwen3-Next cell, whose check-1 site follows its load, compiles at its run (`Cell.at_run`)."""


def test_the_torchax_fallback_runs_on_the_gpu(tmp_path):
    """Every torchax cell above on this GPU (`gpu_flow.run_cells`)."""
    run_cells([
        Cell("a float32 torchax model's logits and output tables are as exact as torch's (TF32)",
             lambda: fallback.a_float32_forwards_logits_and_its_output_tables_are_as_exact_as_torchs(
                 tmp_path / "exact"), at_run=AT_RUN),
        Cell("a tier-3 source runs in a process with no CPU backend",
             lambda: fallback.a_tier_three_source_runs_in_a_process_with_no_cpu_backend(tmp_path / "alone"),
             at_run=AT_RUN),
        # On a Hopper, whose measured cells give bf16 rows the fused norm and SiLU product (`NORM_CELLS`).
        Cell("a bf16 tier-3 Qwen3-Next steps through Dew's native kernels as exactly as transformers",
             lambda: fallback.a_bf16_tier_three_qwen3_next_steps_through_dews_native_kernels(
                 tmp_path / "kernels"), "sm90", at_run=PLACED_AT_LOAD),
    ])
