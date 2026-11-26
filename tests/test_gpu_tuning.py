"""The tuner and the benchmark tools on the GPU (G14, the owner's rule: no unit tests on GPUs), run in one
process (`gpu_flow`), each cell its own file's property, each with a directory of its own:

- The tuner measures only on an explicit pass to itself, each launch once; a config failing for more than
  its shared memory stops the measurement; a launch measured beside a profile capture keeps its heuristic; a
  gradient's launch is timed with its backward; a launch's measurement compiles its candidates in the
  process's compile slots (test_tuning).
- A case's measured peak holds at least the bytes XLA planned (test_benchmark_step).
"""

import pytest
import test_benchmark_step as benchmark_step
import test_tuning as tuning
from gpu_flow import Cell, isolated, run_cells

pytestmark = pytest.mark.gpu(reason="the tuner's and the benchmark tools' GPU reads: launches, the allocator")

MEASURED = "the tuner compiles and times each config on its GPU as it measures the launch: no check-1 site"
"""Why a cell measuring a launch in this process has nothing its GPU tier part's prepare compiles ahead
(`gpu_flow.Cell.at_run`)."""
FRESH = "a fresh process measures on its GPU, compiling its programs there: no check-1 site"
"""Why a cell measuring in a process of its own has nothing its prepare compiles ahead."""


def test_the_tuner_and_benchmarks_read_this_gpu(tmp_path_factory):
    """Every cell above on this GPU (`gpu_flow.run_cells`), each run as the suite runs a test
    (`gpu_flow.isolated`), each one's failure kept and all raised together. Each measures on its GPU, so
    its part's prepare, whose device computes nothing, passes it by (`at_run`) where it ran each until it
    computed and named the test unprepared (row 13039)."""
    in_own = isolated(tmp_path_factory)

    run_cells([
        Cell("the tuner measures only on an explicit pass to itself, each launch once",
             in_own(tuning.only_an_explicit_pass_on_a_gpu_to_itself_measures_and_it_measures_each_launch_once,
                    "pass"), at_run=FRESH),
        Cell("a config failing for more than its shared memory stops the measurement",
             tuning.a_config_failing_for_more_than_its_shared_memory_stops_the_measurement, at_run=MEASURED),
        Cell("a launch measured beside a profile capture keeps its heuristic",
             in_own(tuning.a_launch_measured_beside_a_profile_capture_keeps_its_heuristic, "capture"),
             at_run=MEASURED),
        Cell("a pulled launch is timed with its backward",
             in_own(tuning.a_pulled_launch_is_timed_with_its_backward, "pulled"), at_run=MEASURED),
        Cell("a launch's measurement compiles its candidates in the process's compile slots",
             in_own(tuning.a_launchs_measurement_compiles_its_candidates_in_the_processs_compile_slots,
                    "slots"), at_run=FRESH),
        Cell("a case's measured peak holds at least the bytes XLA planned",
             in_own(benchmark_step.a_cases_measured_peak_holds_at_least_the_bytes_xla_planned, "peak"),
             at_run=FRESH),
    ])
