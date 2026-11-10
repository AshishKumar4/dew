"""Flows of the suite run again in a process that computes as a user's does
(`lane_environment.users_process`).

The suite's processes sum a CPU's float32 products in float64 (Dew's test setting, docs/design/testing.md).
Every program a test builds outside check 1's arms is the wide one: a training step's shapes and placements,
a plan's footprint, a partitioning, the regions the algebra pass reads. Each flow here is a test of another
file, run whole by pytest in a child process on the CPU with the suite's devices and none of its arithmetic,
so a crash, a refusal or a wrong plan on the path a library process takes fails a train. They are the tests
that first showed each class (rows 5422, 5427, 5432, 5433, 5434, 5438), one a class."""

import json
import os
import sys
from pathlib import Path
from xml.etree import ElementTree

import jax
import pytest
import test_codec_checkpoints
import test_decode_attention
import test_derived_plan
import test_device_image_augmentation
import test_host_placement
import test_jepa
import test_layer_stack
import test_mesh_axes
import test_run_options
from lane_environment import users_process
from process_support import relayed

FLOWS = (
    # A training step on an Explicit data axis, through a video encoder and through a device's own steps.
    test_jepa.test_a_video_objective_steps_on_an_explicit_data_axis,
    test_device_image_augmentation.test_a_device_steps_rows_keep_their_explicit_axis_where_the_run_reads_them,
    # A plan derived for the program a user's process compiles, that program's communication, work and
    # temporaries, and a pool of processes over a host-resident bank.
    test_derived_plan.test_the_first_compile_is_at_the_least_recompute_rung_the_derivation_says_fits,
    test_mesh_axes.test_tensor_parallelism_computes_latent_attentions_down_projections_once,
    test_decode_attention.test_a_grouped_query_keeps_the_tensor_axis_split,
    test_layer_stack.test_a_one_stage_state_holds_a_scanned_stack_as_its_scan_reads_it,
    test_host_placement.test_a_pool_reads_its_own_shards_of_a_host_resident_bank,
    # A command-line run of each kind: the algebra pass reads the programs a library process builds.
    test_run_options.test_a_kinds_run_set_on_the_command_line_pools_logs_scores_and_keeps_as_set,
    # A quantized checkpoint's products against its author's own forward.
    test_codec_checkpoints.test_a_quantized_checkpoint_computes_its_reference,
)
"""The flows, each another file's test function: every case of it runs."""

FLOW_SECONDS = 1200.0
"""What a flow's child process has before it is aborted for its stacks (`process_support.relayed`): below
CI's task deadline (tools/armada/session.py's `DEADLINE`), so a flow that hangs fails here with what it
wrote, not as a task cut short."""


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="a user's CPU path, which the CPU lane runs")
@pytest.mark.parametrize("flow", FLOWS, ids=lambda flow: flow.__name__)
def test_a_flow_holds_in_a_process_that_computes_as_a_users_does(flow, tmp_path):
    """`flow`, a test of the suite, run by pytest in a child process on the CPU with a user's arithmetic:
    every case of it ran and passed, none skipped, each in a process that reports no wide CPU arithmetic, no
    matmul precision of the suite's and the suite's devices (the JUnit property `users_arithmetic`,
    tests/conftest.py)."""
    tests = Path(__file__).resolve().parent
    report = tmp_path / "flow.xml"
    relayed([sys.executable, "-m", "pytest", f"{tests.name}/{flow.__module__}.py::{flow.__name__}", "-q",
             "--no-header", "-p", "no:cacheprovider", f"--junitxml={report}"], users_process(os.environ),
            timeout=FLOW_SECONDS)
    cases = list(ElementTree.parse(report).iter("testcase"))
    ended = {case.get("name"): [child.tag for child in case if child.tag != "properties"] for case in cases}
    computed = [json.loads(node.get("value")) for case in cases for node in case.iter("property")
                if node.get("name") == "users_arithmetic"]
    assert cases and not any(ended.values()), ended
    assert computed == len(cases) * [{"wide_cpu_arithmetic": False, "matmul_precision": None,
                                      "devices": jax.device_count()}]
