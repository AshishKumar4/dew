"""`dew launch`: the pool it starts, the commands it writes, and how it stops.

Each test runs the launcher as its own process against small Python
programs on this machine, so what is checked is what a shell sees: the exit
code, the output, and which processes are left alive.
"""

import errno
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from lane_environment import outside_any_cluster

REPO_ROOT = Path(__file__).resolve().parents[1]
# The launches below stand in for Slurm, Open MPI or a plain host with
# variables of their own, on a machine in no cluster of its own.
ENV = {**outside_any_cluster(os.environ), "PYTHONPATH": str(REPO_ROOT / "src")}

def test_a_launch_avoids_an_occupied_coordinator_port():
    """A service on the old fixed port must not stop a new pool starting."""
    program = ("import os, socket\n"
               "host, port = os.environ['JAX_COORDINATOR_ADDRESS'].rsplit(':', 1)\n"
               "with socket.socket() as coordinator:\n"
               "    coordinator.bind((host, int(port)))\n"
               "    print('coordinator bound')\n")
    with socket.socket() as occupied:
        try:
            occupied.bind(("localhost", 43217))
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
        done = launched("--processes-per-host", "1",
                        "--", sys.executable, "-c", program, env={**ENV, "JAX_PLATFORMS": "cpu"})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "coordinator bound" in done.stdout


def launcher(*arguments: str, env: dict[str, str] = ENV) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-m", "dew.cli.main", "launch", *arguments],
                            cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def started_ranks(tmp_path: Path, count: int, seconds: float = 60) -> list[int]:
    """The pids the ranks wrote, once all `count` have written one."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        pids = [path.read_text() for path in tmp_path.glob("rank*")]
        if len(pids) == count and all(pids):
            return [int(pid) for pid in pids]
        time.sleep(0.1)
    raise AssertionError(f"{count} ranks did not start")


# Each rank records its pid and then waits the way one blocked in a
# collective does.
WAITING_RANK = ("import os, pathlib, sys, time\n"
                "pathlib.Path(sys.argv[1], 'rank' + os.environ['DEW_PROCESS_ID'])"
                ".write_text(str(os.getpid()))\n"
                "time.sleep(600)\n")


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT])
def test_a_signalled_launch_stops_its_pool_and_reports_the_signal(tmp_path, signum):
    """A scheduler's SIGTERM, a closed terminal's SIGHUP and a Ctrl-C all
    stop every rank, which sit in sessions of their own that no terminal
    signal reaches, and the launch exits 128 plus the signal."""
    launch = launcher("--processes-per-host", "2", "--port", "1", "--",
                      sys.executable, "-c", WAITING_RANK, str(tmp_path))
    ranks = started_ranks(tmp_path, 2)
    launch.send_signal(signum)
    assert launch.wait(timeout=60) == 128 + signum
    deadline = time.monotonic() + 10
    while any(alive(pid) for pid in ranks) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not any(alive(pid) for pid in ranks)


def test_a_ranks_lines_reach_the_launch_while_the_rank_runs():
    """A Python rank's print arrives while the rank runs on. Its stdout is
    the launcher's pipe, which Python fills in blocks, so without unbuffered
    output a run's step lines would reach the log only when the run ends.
    The launch runs without PYTHONUNBUFFERED, as from a plain shell."""
    program = "import time\nprint('ready')\ntime.sleep(600)\n"
    shell = {name: value for name, value in ENV.items() if name != "PYTHONUNBUFFERED"}
    launch = launcher("--port", "1", "--", sys.executable, "-c", program, env=shell)
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: [lines.put(line.decode()) for line in launch.stdout or ()],
                     daemon=True).start()
    seen = []
    deadline = time.monotonic() + 30
    try:
        while "[0] ready" not in seen:
            seen.append(lines.get(timeout=max(0.0, deadline - time.monotonic())).rstrip())
    except queue.Empty:
        pytest.fail(f"the rank's line had not arrived after 30 s; the launch printed {seen}")
    finally:
        launch.send_signal(signal.SIGTERM)
        launch.wait(timeout=60)


def test_a_rank_killed_by_a_signal_exits_the_launch_128_plus_it():
    """The OOM killer's SIGKILL reads as 137, the shell's convention, which
    wrappers test for; Python's -9 would reach the shell as 247."""
    program = "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n"
    launch = launcher("--port", "1", "--", sys.executable, "-c", program)
    launch.communicate(timeout=60)
    assert launch.returncode == 137


def test_a_failing_rank_stops_the_pool_and_is_named_with_its_last_lines(tmp_path):
    """Rank 1 fails while rank 0 waits as one in a collective would. The
    launch stops rank 0, exits with rank 1's code, and ends with rank 1's
    last lines, after rank 0's shutdown output, so the reason is the last
    thing on the screen."""
    program = ("import os, pathlib, sys, time\n"
               "rank = os.environ['DEW_PROCESS_ID']\n"
               "pathlib.Path(sys.argv[1], 'rank' + rank).write_text(str(os.getpid()))\n"
               "if rank == '1':\n"
               "    time.sleep(1)\n"
               "    print('loading shard 7'); print('OSError: shard 7 is gone'); sys.exit(3)\n"
               "time.sleep(600)\n")
    launch = launcher("--processes-per-host", "2", "--port", "1", "--",
                      sys.executable, "-c", program, str(tmp_path))
    try:
        launch.communicate(timeout=60)
    finally:
        launch.kill()
    ranks = [int(path.read_text()) for path in tmp_path.glob("rank*")]
    assert launch.returncode == 3
    assert not any(alive(pid) for pid in ranks)


def launched(*arguments: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "dew.cli.main", "launch", *arguments],
                          cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120)


def fake_srun(tmp_path: Path) -> dict:
    """An allocation's environment whose `srun` records its arguments and
    the variables it would hand its tasks, in place of Slurm's: four GPUs
    and a jax that runs on them. Every source the launcher counts GPUs from
    is set here, so the lane the tests run on decides nothing: the CPU
    lane's JAX_PLATFORMS=cpu would keep a pool off the GPUs, and a GPU
    lane's SLURM_* or visible devices would count its own."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    srun = bin_dir / "srun"
    srun.write_text("#!/bin/sh\n"
                    f"printf '%s\\n' \"$@\" > {tmp_path}/argv\n"
                    f"printf '%s' \"$XLA_FLAGS\" > {tmp_path}/env\n")
    srun.chmod(0o755)
    return {**ENV, "PATH": f"{bin_dir}:{os.environ['PATH']}", "SLURM_JOB_ID": "77",
            "JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}


def test_srun_runs_a_task_per_gpu_and_receives_values_with_commas_whole(tmp_path):
    """jax gives each Slurm task the one GPU at its SLURM_LOCALID, so a node
    runs a task per GPU Slurm left it. Slurm reads --export as a
    comma-separated list, so the variables travel in srun's environment,
    which --export=ALL hands to every task."""
    done = launched("--env", "XLA_FLAGS=--a=1,--b=2", "--", "python", "train.py",
                    env=fake_srun(tmp_path))
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "argv").read_text().split() == [
        "--kill-on-bad-exit=1", "--export=ALL", "--label", "--ntasks-per-node=4", "python", "train.py"]
    assert (tmp_path / "env").read_text() == "--a=1,--b=2"


@pytest.mark.distributed
def test_a_bare_launch_runs_one_process_per_gpu_of_this_machine():
    """`dew launch -- python ...` with no flags: every GPU here gets a
    process of its own, and each joins the pool holding one device."""
    from dew.pool import local_gpu_count

    gpus = local_gpu_count()
    if gpus is None or gpus < 2:
        pytest.skip(f"needs two GPUs; this machine shows {gpus}")
    program = ("from dew.training.runtime import prepare_process\n"
               "prepare_process()\n"
               "import jax\n"
               "print('placed', jax.process_index(), jax.process_count(), "
               "[d.id for d in jax.local_devices()])\n")
    env = {name: value for name, value in ENV.items() if name != "JAX_PLATFORMS"}
    done = launched("--", sys.executable, "-c", program, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    placed = sorted(line.split("placed ", 1)[1] for line in done.stdout.splitlines()
                    if "placed " in line)
    assert placed == [f"{rank} {gpus} [{rank}]" for rank in range(gpus)]
