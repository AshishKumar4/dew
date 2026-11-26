"""A run's systems on the GPU (G14, the owner's rule: no unit tests on GPUs): each system that reads a GPU of
its own, run in one process (`gpu_flow`), each cell its own file's property, each with a directory of its
own:

- A GPU window past CUPTI's activity cap reports no device figure its kernels cannot establish
  (test_profile).
- A fused attention counts the pairs its mask computes, with and without a window; a run prepared on a GPU
  keeps its autotuner's picks where the policy puts them (test_instrumentation).
- A compilation counts the memory kept alive outside its state, in a fresh 2 GiB allocator; two checkouts
  share one compiled kernel (tests/systems_cells.py).

The tuner's and the benchmark tools' cells are tests/test_gpu_tuning.py's.
"""

import functools

import pytest
import systems_cells
import test_instrumentation as instrumentation
import test_profile as profile
from gpu_flow import Cell, isolated, run_cells

pytestmark = pytest.mark.gpu(reason="each system's GPU reads: CUPTI, the allocator, Triton")


@pytest.mark.compiles_at_run(reason="reads this GPU's systems (CUPTI, the allocator, Triton) as a run does: "
                                    "no check-1 site")
def test_a_runs_systems_read_this_gpu(tmp_path_factory):
    """Every systems cell above on this GPU (`gpu_flow.run_cells`), each run as the suite runs a test
    (`gpu_flow.isolated`), each one's failure kept and all raised together."""
    in_own = isolated(tmp_path_factory)

    run_cells([
        Cell("a GPU window past CUPTI's cap reports no device figure its kernels cannot establish",
             in_own(profile.a_gpu_window_past_cuptis_cap_reports_no_device_figure_its_kernels_cannot_establish,
                    "cupti")),
        *(Cell(f"a fused attention counts the pairs its mask computes: window {window}",
               functools.partial(instrumentation.a_fused_attention_counts_the_pairs_its_mask_computes,
                                 window))
          for window in (None, 64)),
        Cell("a run prepared on a GPU keeps its picks where the policy puts them",
             in_own(instrumentation.a_run_prepared_on_a_gpu_keeps_its_picks_where_the_policy_puts_them,
                    "picks")),
        Cell("a compilation counts memory kept alive outside its state",
             in_own(systems_cells.compilation_counts_memory_kept_alive_outside_its_state, "remat")),
        Cell("two checkouts share one compiled kernel",
             in_own(systems_cells.two_checkouts_share_one_compiled_kernel, "checkouts")),
    ])
