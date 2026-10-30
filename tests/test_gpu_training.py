"""A run's training on the GPU (G14, the owner's rule: no unit tests on GPUs): each property of a training
step that only a GPU shows, run in one process (`gpu_flow`), each cell its own file's property, each with a
directory, rung records and data cache of its own:

- A step compiled ahead for this GPU's generation, saved and loaded, runs in the run's place; a run handed it
  trains from it, refusing another program's; a run on two GPUs traces with the links it was compiled with;
  a live run takes the saved offline split before it loads the executable (test_compile_ahead).
- A recorded plan settles its kernel launches without lowering the step for them; a step guided by its
  profile is the step the run takes, as exact; a split that misses tries the half rung before the batch
  splits further (test_plan).
- A CUDA LM repeats and resumes bit for bit under deterministic ops; a convolution model trains to the bit
  again after another run measured kernels into its cache (test_trainer).
- An accumulating step updates the optimizer in its commit alone (test_training_transactions).
- An LM run measures its kernel launches as set (test_run_option_matrix).
"""

import dataclasses
import inspect
import os

import lane_environment
import jax
import pytest
import test_compile_ahead as compile_ahead
import test_plan as plan
import test_run_option_matrix as run_option_matrix
import test_trainer as trainer
import test_training_transactions as transactions
from gpu_flow import Cell, cells_of, run_cells

pytestmark = pytest.mark.gpu(reason="each property a training step shows on a GPU alone")


def test_a_runs_training_reads_this_gpu(tmp_path_factory):
    """Every training cell above on this GPU (`gpu_flow.run_cells`), each in a directory of its own, with
    its own rung records and data cache as the suite gives each test (tests/conftest.py), each one's
    failure kept and all raised together."""
    def in_own(check, name, *, devices=1, cuda_flags=True, **given):
        """`check` run in a directory of its own, its fixtures by name (`tmp_path`, `monkeypatch`), where
        this process holds `devices` devices, under the cuda lane's flags but its deterministic ops
        where `cuda_flags` is False (`without_deterministic_ops`)."""
        def run():
            if jax.device_count() < devices:
                pytest.skip(f"needs {devices} devices; this process has {jax.device_count()}")
            directory = tmp_path_factory.mktemp(name)
            with pytest.MonkeyPatch.context() as patch:
                patch.setenv("DEW_RUNG_RECORDS", str(directory / "rungs"))
                patch.setenv("DEW_DATA_CACHE", str(directory / "data-cache"))
                if not cuda_flags:
                    patch.setenv("XLA_FLAGS", lane_environment.without_deterministic_ops(
                        os.environ.get("XLA_FLAGS", "")))
                fixtures = {"tmp_path": directory, "monkeypatch": patch}
                taken = inspect.signature(check).parameters
                check(**{name: value for name, value in fixtures.items() if name in taken}, **given)
        return run

    saved, handed, links, split = (
        compile_ahead.a_step_compiled_ahead_and_saved_runs_on_its_gpu_in_the_run_s_place,
        compile_ahead.a_run_handed_its_step_compiled_ahead_trains_from_it_on_its_gpu,
        compile_ahead.a_run_handed_a_step_compiled_ahead_traces_with_the_links_it_was_compiled_with,
        compile_ahead.a_run_takes_the_saved_offline_split_before_loading_its_executable)
    settled, guided, half = (
        plan.a_recorded_plan_settles_its_kernel_launches_without_lowering_the_step_for_them,
        plan.a_step_guided_by_its_profile_is_the_step_the_run_takes_and_trains_as_exactly,
        plan.a_split_that_misses_tries_the_half_rung_before_the_batch_splits_further)
    repeats, conv = (trainer.a_cuda_lm_repeats_and_resumes_bit_exactly_with_deterministic_ops,
                     trainer.a_conv_model_trains_to_the_bit_again_after_another_run_measured_kernels_into_its_cache)
    run_cells([
        Cell("a step compiled ahead and saved runs on its GPU in the run's place", in_own(saved, "saved")),
        Cell("a run handed its step compiled ahead trains from it on its GPU", in_own(handed, "handed")),
        Cell("a run handed a step compiled ahead traces with the links it was compiled with",
             in_own(links, "links", devices=2)),
        Cell("a run takes the saved offline split before loading its executable", in_own(split, "split")),
        Cell("a recorded plan settles its kernel launches without lowering the step for them",
             in_own(settled, "settled")),
        Cell("a step guided by its profile is the step the run takes and trains as exactly",
             in_own(guided, "guided")),
        Cell("a split that misses tries the half rung before the batch splits further", in_own(half, "half")),
        # The test's own grid of dtypes (`cells_of`), each cell in a directory of its own.
        *(dataclasses.replace(cell, run=in_own(cell.run, f"repeats-{index}"))
          for index, cell in enumerate(cells_of(repeats))),
        Cell("a conv model trains to the bit again after another run measured kernels into its cache",
             in_own(conv, "conv")),
        Cell("an accumulating step updates the optimizer in its commit alone",
             in_own(transactions.an_accumulating_step_updates_the_optimizer_in_its_commit_alone, "commit")),
        Cell("an LM run measures its kernel launches as set",
             in_own(run_option_matrix.an_lm_run_measures_its_kernel_launches_as_set, "launches",
                    cuda_flags=False)),
    ])
