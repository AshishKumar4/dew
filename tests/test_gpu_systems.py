"""A run's systems on the GPU (G14, the owner's rule: no unit tests on GPUs): each system that reads a GPU of
its own, run in one process (`gpu_flow`), each cell its own file's property, each with a directory of its
own:

- The tuner measures only on an explicit pass to itself, each launch once; a config failing for more than
  its shared memory stops the measurement; a launch measured beside a profile capture keeps its heuristic
  (test_tuning).
- A GPU window past CUPTI's activity cap reports no device figure its kernels cannot establish
  (test_profile).
- A fused attention counts the pairs its mask computes, with and without a window; a run prepared on a GPU
  keeps its autotuner's picks where the policy puts them (test_instrumentation).
- A compilation counts the memory kept alive outside its state, in a fresh 2 GiB allocator; two checkouts
  share one compiled kernel (tests/systems_cells.py).
- A case's measured peak holds at least the bytes XLA planned (test_benchmark_step).
"""

import functools

import pytest
import systems_cells
import test_benchmark_step as benchmark_step
import test_instrumentation as instrumentation
import test_profile as profile
import test_tuning as tuning
from gpu_flow import Cell, run_cells

pytestmark = pytest.mark.gpu(reason="each system's GPU reads: the tuner, CUPTI, the allocator, Triton")


def test_a_runs_systems_read_this_gpu(tmp_path_factory):
    """Every systems cell above on this GPU (`gpu_flow.run_cells`), each in a directory of its own, each
    one's failure kept and all raised together."""
    def in_own(check, name):
        return lambda: check(tmp_path_factory.mktemp(name))

    run_cells([
        Cell("the tuner measures only on an explicit pass to itself, each launch once",
             in_own(tuning.only_an_explicit_pass_on_a_gpu_to_itself_measures_and_it_measures_each_launch_once,
                    "pass")),
        Cell("a config failing for more than its shared memory stops the measurement",
             tuning.a_config_failing_for_more_than_its_shared_memory_stops_the_measurement),
        Cell("a launch measured beside a profile capture keeps its heuristic",
             in_own(tuning.a_launch_measured_beside_a_profile_capture_keeps_its_heuristic, "capture")),
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
        Cell("a case's measured peak holds at least the bytes XLA planned",
             in_own(benchmark_step.a_cases_measured_peak_holds_at_least_the_bytes_xla_planned, "peak")),
        Cell("two checkouts share one compiled kernel",
             in_own(systems_cells.two_checkouts_share_one_compiled_kernel, "checkouts")),
    ])
