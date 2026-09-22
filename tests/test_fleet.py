"""The sandbox fleet runs real programs: how each one ends is its verdict.

Every check runs an actual interpreter under the fleet's limits: a program
that completes, one that fails, one that sleeps past the wall clock, one that
spins past its CPU time, one that allocates past its memory, one that floods
its output, one killed by a signal, and one whose forked child must not
outlive the timeout. The verifiable rewards score completions through the
same fleet. The container runner repeats the boundary checks when a Docker
daemon and the image are present locally.
"""

import shutil
import subprocess
import sys
import time

import pytest

from dew.objectives.rl import (
    CodeReward,
    ContainerRunner,
    MathReward,
    Program,
    SandboxFleet,
    SandboxLimits,
    Verdict,
    code_block,
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the fleet's limits are Linux process limits")

PYTHON = (sys.executable, "-I", "-S")


def program(source, stdin=""):
    return Program({"main.py": source}, (*PYTHON, "main.py"), stdin)


@pytest.fixture
def fleet():
    with SandboxFleet(limits=SandboxLimits(wall_seconds=10, cpu_seconds=5), workers=4) as running:
        yield running


def test_a_program_reads_stdin_and_completes(fleet):
    outcome = fleet.run([program("a, b = map(int, open(0).read().split())\nprint(a * b)", "6 7\n")])[0]
    assert outcome.verdict is Verdict.COMPLETED
    assert outcome.exit_code == 0 and outcome.stdout == "42\n"


def test_exit_status_and_exceptions_are_failures(fleet):
    exited, raised = fleet.run([program("raise SystemExit(3)"), program("1 / 0")])
    assert exited.verdict is Verdict.FAILED and exited.exit_code == 3
    assert raised.verdict is Verdict.FAILED and "ZeroDivisionError" in raised.stderr


def test_the_wall_clock_kills_a_sleeping_program_and_its_children():
    limits = SandboxLimits(wall_seconds=1.5, cpu_seconds=5)
    source = ("import subprocess, sys, time\n"
              "child = subprocess.Popen(['sleep', '30'])\n"
              "print(child.pid, flush=True)\n"
              "time.sleep(30)\n")
    with SandboxFleet(limits=limits, workers=1) as fleet:
        began = time.monotonic()
        outcome = fleet.run([program(source)])[0]
    assert outcome.verdict is Verdict.TIMEOUT and outcome.exit_code is None
    assert time.monotonic() - began < 8
    child = int(outcome.stdout.split()[0])
    # The group kill reaches the grandchild; allow the kernel a moment to reap it.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with open(f"/proc/{child}/stat") as stat:
                if stat.read().split(")")[-1].split()[0] == "Z":
                    break
        except FileNotFoundError:
            break
        time.sleep(.05)
    else:
        pytest.fail("the forked child outlived the timeout")


def test_cpu_time_is_a_timeout_before_the_wall_clock():
    with SandboxFleet(limits=SandboxLimits(wall_seconds=30, cpu_seconds=1), workers=1) as fleet:
        outcome = fleet.run([program("while True:\n    pass")])[0]
    assert outcome.verdict is Verdict.TIMEOUT
    assert outcome.seconds < 15


def test_memory_past_the_limit_fails_the_program():
    with SandboxFleet(limits=SandboxLimits(memory_bytes=256 * 1024 ** 2), workers=1) as fleet:
        outcome = fleet.run([program("block = bytearray(1024 ** 3)")])[0]
    assert outcome.verdict is Verdict.FAILED and "MemoryError" in outcome.stderr


def test_a_signal_is_a_crash_and_a_flood_is_cut_off(fleet):
    crashed, flooded = fleet.run([
        program("import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)"),
        program("import sys\nwhile True:\n    sys.stdout.write('x' * 65536)")])
    assert crashed.verdict is Verdict.CRASHED
    assert flooded.verdict is Verdict.OUTPUT_LIMIT
    assert len(flooded.stdout) <= SandboxLimits().message_bytes + 65536


def test_programs_run_concurrently_and_return_in_order():
    with SandboxFleet(limits=SandboxLimits(wall_seconds=20), workers=4) as fleet:
        began = time.monotonic()
        outcomes = fleet.run(program(f"import time\ntime.sleep(1.5)\nprint({index})") for index in range(4))
        elapsed = time.monotonic() - began
    assert [outcome.stdout for outcome in outcomes] == ["0\n", "1\n", "2\n", "3\n"]
    assert elapsed < 5, "four one-and-a-half-second programs on four workers ran one after another"


def test_a_program_file_cannot_escape_its_directory():
    with pytest.raises(ValueError, match="relative path"):
        Program({"../x.py": ""}, ("python",))


CASES = '[{"stdin": "2 3\\n", "stdout": "5"}, {"stdin": "10 -4\\n", "stdout": "6"}]'


def test_code_reward_scores_the_fraction_of_passing_cases(fleet):
    reward = CodeReward(fleet)
    right = "Here:\n```python\na, b = map(int, input().split())\nprint(a + b)\n```"
    half = "```python\na, b = map(int, input().split())\nprint(5)\n```"
    looping = "```python\nwhile True:\n    pass\n```"
    assert reward("code", right, CASES, "") == 1.0
    assert reward("code", half, CASES, "") == 0.5
    assert reward("code", "the answer is 5", CASES, "") == 0.0
    assert CodeReward(fleet, all_or_nothing=True)("code", half, CASES, "") == 0.0
    with SandboxFleet(limits=SandboxLimits(wall_seconds=1, cpu_seconds=1), workers=2) as tight:
        assert CodeReward(tight)("code", looping, CASES, "") == 0.0


def test_code_block_prefers_the_last_tagged_block():
    text = "```\nuntagged\n```\n```python\nfirst\n```\n```python\nsecond\n```"
    assert code_block(text) == "second\n"
    assert code_block("```\nonly\n```") == "only\n"
    assert code_block("no code") is None


def test_math_reward_compares_rational_answers():
    reward = MathReward()
    assert reward("math", r"so \boxed{\frac{1}{2}}", "0.5", "") == 1.0
    assert reward("math", r"\boxed{1,024}", "1024", "") == 1.0
    assert reward("math", r"\boxed{-3/4}", "-0.75", "") == 1.0
    assert reward("math", r"\boxed{7}", "8", "") == 0.0
    assert reward("math", "the answer is 8", "8", "") == 0.0
    assert MathReward(require_boxed=False)("math", "the answer is 8.", "8", "") == 1.0
    assert reward("math", r"\boxed{x+1}", "x+1", "") == 1.0


IMAGE = "python:3.12-slim"


def _image_present() -> bool:
    if shutil.which("docker") is None:
        return False
    found = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, check=False)
    return found.returncode == 0


@pytest.mark.skipif(not _image_present(), reason=f"needs a Docker daemon with {IMAGE} pulled")
def test_a_container_has_no_network_a_read_only_root_and_a_deadline():
    runner = ContainerRunner(IMAGE)
    network = ("import socket\n"
               "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
               "except OSError:\n    print('offline')\n")
    readonly = ("try:\n    open('/work/escape', 'w')\n"
                "except OSError:\n    print('read-only')\n")
    limits = SandboxLimits(wall_seconds=30, cpu_seconds=5)
    with SandboxFleet(runner, limits=limits, workers=2) as fleet:
        offline, sealed = fleet.run([Program({"main.py": network}, ("python", "main.py")),
                                     Program({"main.py": readonly}, ("python", "main.py"))])
    assert offline.verdict is Verdict.COMPLETED and offline.stdout == "offline\n"
    assert sealed.verdict is Verdict.COMPLETED and sealed.stdout == "read-only\n"
    with SandboxFleet(runner, limits=SandboxLimits(wall_seconds=4, cpu_seconds=30), workers=1) as fleet:
        slept = fleet.run([Program({"main.py": "import time\ntime.sleep(60)"}, ("python", "main.py"))])[0]
    assert slept.verdict is Verdict.TIMEOUT
    leftover = subprocess.run(["docker", "ps", "--filter", "name=dew-fleet-", "-q"],
                              capture_output=True, text=True, check=True).stdout.strip()
    assert leftover == "", "a timed-out container is still running"
