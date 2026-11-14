"""Flows of the suite run again in a process that computes as a user's does
(`lane_environment.users_process`).

The suite's processes sum a CPU's float32 products in float64 (Dew's test setting, docs/design/testing.md).
Every program a test builds outside check 1's arms is the wide one: a training step's shapes and placements,
a plan's footprint, a partitioning, the regions the algebra pass reads. Each flow here is a test of another
file, run whole by pytest in a child process on the CPU with the suite's devices and none of its arithmetic,
so a crash or a refusal on the path a library process takes fails a train. Each is the test that first
showed a class the whole suite run without the setting found (row 5438 and its kin, rows 5422 to 5442); a
class whose fix is not in yet adds its flow with it."""

import json
import os
import sys
from pathlib import Path
from xml.etree import ElementTree

import jax
import pytest
import run_kinds
import test_codec_checkpoints
import test_host_placement
import test_run_options
from lane_environment import users_process
from process_support import relayed
from tool_support import load

session = load("session", "tools/armada")
"""CI's pytest session: what a task's pytest never inherits (`UNKEYED`), and its deadline."""

FLOWS = (
    # A command-line run of each kind: the algebra pass reads the programs a library process builds.
    test_run_options.test_a_kinds_run_set_on_the_command_line_pools_logs_scores_and_keeps_as_set,
    # A quantized checkpoint's products against its author's own forward.
    test_codec_checkpoints.test_a_quantized_checkpoint_computes_its_reference,
    # A pool of two processes, whose ranks' output a user's arithmetic makes long (row 5434).
    test_host_placement.test_a_pool_reads_its_own_shards_of_a_host_resident_bank,
)
"""The flows, each another file's test function: every case of it runs, each in a child of its own where the
flow states its cases (`CASES`)."""

CASES = {test_run_options.test_a_kinds_run_set_on_the_command_line_pools_logs_scores_and_keeps_as_set:
         tuple(run_kinds.KINDS)}
"""The cases a flow runs one to a child, by their ids: a command-line run of each kind took 141 s together,
past a group's 120 s (row 5541)."""

TARGETS = [(flow, case) for flow in FLOWS for case in CASES.get(flow, (None,))]
"""Each flow, or each of its cases (`CASES`), as one test."""

NAMES = [flow.__name__ + ("" if case is None else f"[{case}]") for flow, case in TARGETS]
"""Each target's name: its flow's, its case's id after it."""

FLOW_SECONDS = 1200.0
"""What a flow's child process has before it is aborted for its stacks (`process_support.relayed`): below
CI's task deadline (tools/armada/session.py's `DEADLINE`), so a flow that hangs fails here with what it
wrote, not as a task cut short."""


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="a user's CPU path, which the CPU lane runs")
@pytest.mark.parametrize(("flow", "case"), TARGETS, ids=NAMES)
def test_a_flow_holds_in_a_process_that_computes_as_a_users_does(flow, case, tmp_path):
    """`flow`, a test of the suite, run by pytest in a child process on the CPU with a user's arithmetic:
    every case of it ran and passed, none skipped, each in a process that reports no wide CPU arithmetic, no
    matmul precision of the suite's and the suite's devices (the JUnit property `users_arithmetic`,
    tests/conftest.py), and the child ended clean."""
    tests = Path(__file__).resolve().parent
    report = tmp_path / "flow.xml"
    # The flow whole: what a supervisor's environment would have pytest select or load is no part of it.
    environment = {name: value for name, value in users_process(os.environ).items()
                   if name not in session.UNKEYED}
    target = f"{tests.name}/{flow.__module__}.py::{NAMES[TARGETS.index((flow, case))]}"
    ran = relayed([sys.executable, "-m", "pytest", target, "-q", "--no-header", "-p", "no:cacheprovider",
                   f"--junitxml={report}"], environment, timeout=FLOW_SECONDS)
    cases = list(ElementTree.parse(report).iter("testcase"))
    ended = {case.get("name"): [child.tag for child in case if child.tag != "properties"] for case in cases}
    computed = [json.loads(node.get("value")) for case in cases for node in case.iter("property")
                if node.get("name") == "users_arithmetic"]
    assert ran.returncode == 0 and cases and not any(ended.values()), (ran.returncode, ended)
    assert computed == len(cases) * [{"wide_cpu_arithmetic": False, "matmul_precision": None,
                                      "devices": jax.device_count()}]
