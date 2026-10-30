"""The torchax fallback's GPU flow (G14, the owner's rule: no unit tests on GPUs), apart from the other
flows: torchax renames torch's privateuse1 backend process-wide (the `torchax` mark), so its cells share
no process with the torch-using ones (`gpu_flow`):

- A float32 torchax model's logits and its output tables are as exact as torch's under the GPU's TF32
  default (test_torchax_fallback).
"""

import pytest
import test_torchax_fallback as fallback
from gpu_flow import Cell, run_cells

pytestmark = [pytest.mark.gpu(reason="TF32 is a GPU's default float32 product"), pytest.mark.torchax]


def test_the_torchax_fallback_runs_on_the_gpu(tmp_path):
    """Every torchax cell above on this GPU (`gpu_flow.run_cells`)."""
    run_cells([
        Cell("a float32 torchax model's logits and output tables are as exact as torch's (TF32)",
             lambda: fallback.a_float32_forwards_logits_and_its_output_tables_are_as_exact_as_torchs(
                 tmp_path)),
    ])
