"""GEMM's TPU tier: the flow of GEMM's check-1 kernel matrix (tests/kernel_matrix.py) whose cells run on a
TPU, the TPU's Mosaic launches of the grouped matmul (`mosaic_tpu`), each cell on its own TPU generation."""

import gemm_matrix
import pytest
from gpu_flow import cells_of, run_cells


@pytest.mark.tpu(reason="the TPU's Mosaic launches of the grouped matmul, compiled")
def test_the_gemm_matrix_holds_on_this_tpu():
    """GEMM's check-1 matrix on this TPU: the training step of each case its TPU's launches serve on par
    with the reference path's, each cell on its own generation (the GPU tier runs the rest,
    tests/test_gpu_gemm.py)."""
    run_cells(cells_of(gemm_matrix.a_gemm_training_step_is_on_par))
