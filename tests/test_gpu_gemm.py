"""The GEMM template and the Mosaic toolkit on the GPU (G14, the owner's rule: no unit tests on GPUs), run in
one process (`gpu_flow`), each cell its own file's property:

- A dense product's exact sums survive one row tile through Triton's MMA accumulation (test_gemm_template).
- CUTLASS runs the grouped projection as exactly as XLA wherever it is chosen, over every grouping that
  routes a row (test_grouped_gemm).
- A ring streams tiles of their own trip counts, and consumers feeding their own ring split a step, on an
  H100 (test_mosaic_toolkit).
"""

import functools

import pytest
import test_gemm_template as gemm_template
import test_grouped_gemm as grouped_gemm
import test_mosaic_toolkit as mosaic_toolkit
from gpu_flow import Cell, run_cells

from dew.nn.kernels import generation

pytestmark = pytest.mark.gpu(reason="the GEMM template's and the Mosaic toolkit's kernels, compiled")


def test_the_gemm_kernels_run_on_this_gpu():
    """Every GEMM and toolkit cell above on this GPU (`gpu_flow.run_cells`), each one's failure kept and all
    raised together; the toolkit's on an H100 alone."""
    def on_triton():
        if not generation.triton_runs():
            pytest.skip("needs a GPU the Pallas kernels compile for")
        gemm_template.a_dense_products_exact_sums_survive_one_row_tile()

    run_cells([
        Cell("a dense product's exact sums survive one row tile", on_triton),
        *(Cell(f"CUTLASS runs the projection as exactly as XLA where chosen: {sizes}",
               functools.partial(grouped_gemm.cutlass_runs_the_projection_as_exactly_as_xla_where_chosen,
                                 sizes))
          for sizes in grouped_gemm.SIZES if any(grouped_gemm.SIZES[sizes])),
        *(Cell(f"a ring streams tiles of their own trip counts: {consumers} consumers",
               functools.partial(mosaic_toolkit.a_ring_streams_tiles_of_their_own_trip_counts_on_an_h100,
                                 consumers), "sm90")
          for consumers in (1, 2)),
        *(Cell(f"consumers feeding their own ring split a step: {consumers} consumers from {lo}",
               functools.partial(mosaic_toolkit.consumers_feeding_their_own_ring_split_a_step_on_an_h100,
                                 consumers, lo), "sm90")
          for consumers, lo in ((1, 0), (2, 0), (2, 3))),
    ])
