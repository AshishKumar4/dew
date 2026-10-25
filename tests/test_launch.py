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
from collections.abc import Sequence
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
    # The bind raises where the address is taken, so the child exits 0 only once it bound.
    assert done.returncode == 0, done.stdout + done.stderr


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


def test_a_failing_rank_leaves_none_of_its_own_processes_running(tmp_path):
    """Rank 1 starts a worker of its own, as a data loader does, and fails; the
    worker is in its rank's process group, which the launch stops though the
    rank itself has exited, so nothing it started outlives the launch."""
    program = ("import os, pathlib, subprocess, sys, time\n"
               "rank = os.environ['DEW_PROCESS_ID']\n"
               "pathlib.Path(sys.argv[1], 'rank' + rank).write_text(str(os.getpid()))\n"
               "if rank == '1':\n"
               "    worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
               "    pathlib.Path(sys.argv[1], 'worker').write_text(str(worker.pid))\n"
               "    time.sleep(1)\n"
               "    sys.exit(3)\n"
               "time.sleep(600)\n")
    launch = launcher("--processes-per-host", "2", "--port", "1", "--",
                      sys.executable, "-c", program, str(tmp_path))
    try:
        launch.communicate(timeout=60)
    finally:
        launch.kill()
    assert launch.returncode == 3
    worker = int((tmp_path / "worker").read_text())
    deadline = time.monotonic() + 10
    while running(worker) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not running(worker)


def test_a_pool_that_exits_zero_leaves_none_of_its_ranks_processes_running(tmp_path):
    """Each rank starts a worker of its own, as a data loader or a server does, and exits 0 with the
    worker still running in its process group; the launch ends that group with the pool, so nothing a
    rank started outlives a successful launch either."""
    program = ("import os, pathlib, subprocess, sys\n"
               "rank = os.environ['DEW_PROCESS_ID']\n"
               "worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
               "pathlib.Path(sys.argv[1], 'worker' + rank).write_text(str(worker.pid))\n")
    launch = launcher("--processes-per-host", "2", "--port", "1", "--",
                      sys.executable, "-c", program, str(tmp_path))
    try:
        launch.communicate(timeout=60)
    finally:
        launch.kill()
    assert launch.returncode == 0
    workers = [int((tmp_path / f"worker{rank}").read_text()) for rank in (0, 1)]
    deadline = time.monotonic() + 10
    while any(running(worker) for worker in workers) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not any(running(worker) for worker in workers)


def running(pid: int) -> bool:
    """Whether `pid` runs: an exited process its new parent has not reaped yet (a zombie, state Z in
    /proc) has stopped, though a signal still finds it."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != "Z"


# ssh -tt as the launcher runs it, to a host that is this machine: the remote command runs under a
# terminal of its own whose input is ssh's, and closing the connection, as ssh exiting does, hangs that
# terminal up.
SSH = """import os, pty, select, signal, sys
words, skipped = [], False
for word in sys.argv[1:]:
    if skipped or word == "-o":
        skipped = not skipped
    elif not word.startswith("-"):
        words.append(word)
pid, terminal = pty.fork()
if pid == 0:
    # sshd starts the remote shell with every signal at its default, whatever this process inherited.
    for number in (signal.SIGINT, signal.SIGQUIT):
        signal.signal(number, signal.SIG_DFL)
    os.execvp(SHELL[0], [*SHELL, "-c", " ".join(words[1:])])
reading = True
while True:
    ready = select.select([terminal, *([0] if reading else [])], [], [])[0]
    if 0 in ready:
        typed = os.read(0, 1024)
        reading = bool(typed)
        if typed:
            os.write(terminal, typed)
    if terminal in ready:
        try:
            shown = os.read(terminal, 1024)
        except OSError:
            break
        if not shown:
            break
        os.write(1, shown)
sys.exit(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]))
"""


def ssh_stand_in(tmp_path: Path, shell: Sequence[str] = ("sh",)) -> Path:
    """A directory holding `ssh`, the stand-in `SSH` runs as, to put first on a PATH, whose remote
    commands `shell` runs, as the remote user's login shell does."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    (binaries / "ssh.py").write_text(f"SHELL = {list(shell)!r}\n{SSH}")
    (binaries / "ssh").write_text(f'#!/bin/sh\nexec {sys.executable} {binaries / "ssh.py"} "$@"\n')
    (binaries / "ssh").chmod(0o755)
    return binaries


def test_a_rank_over_ssh_is_stopped_with_sigterm_and_its_connection_waits_for_it(tmp_path):
    """A rank on another host, started over ssh -tt, takes a stopped launch as a
    local rank does, SIGTERM, the notice a training run checkpoints on, not the
    SIGHUP its connection closing would bring, and its connection stays open
    while it writes its checkpoint, so it ends the checkpoint before the launch
    exits 128 plus the signal."""
    binaries = ssh_stand_in(tmp_path)
    program = ("import os, pathlib, signal, sys, time\n"
               "folder, rank = pathlib.Path(sys.argv[1]), os.environ['DEW_PROCESS_ID']\n"
               "def stopped(signum, frame):\n"
               "    (folder / f'signal{rank}').write_text(signal.Signals(signum).name)\n"
               "    time.sleep(1)\n"
               "    (folder / f'checkpoint{rank}').write_text('written')\n"
               "    sys.exit(0)\n"
               "signal.signal(signal.SIGTERM, stopped)\n"
               "signal.signal(signal.SIGHUP, stopped)\n"
               "(folder / f'rank{rank}').write_text(str(os.getpid()))\n"
               "time.sleep(600)\n")
    launch = launcher("--hosts", "node-a,node-b", "--processes-per-host", "1", "--port", "1", "--",
                      sys.executable, "-c", program, str(tmp_path),
                      env={**ENV, "PATH": f"{binaries}:{ENV['PATH']}"})
    started_ranks(tmp_path, 2)
    launch.send_signal(signal.SIGTERM)
    assert launch.wait(timeout=60) == 128 + signal.SIGTERM
    for rank in range(2):
        assert (tmp_path / f"signal{rank}").read_text() == "SIGTERM"
        assert (tmp_path / f"checkpoint{rank}").read_text() == "written"


def test_a_failing_rank_over_ssh_leaves_none_of_its_processes_on_its_host(tmp_path):
    """Rank 1, on another host over ssh, starts a worker of its own and fails: the
    launch exits with its code, and the worker, which no signal from this host
    reaches once the connection closes, is stopped on its host before the
    connection does."""
    binaries = ssh_stand_in(tmp_path)
    program = ("import os, pathlib, subprocess, sys, time\n"
               "rank = os.environ['DEW_PROCESS_ID']\n"
               "pathlib.Path(sys.argv[1], 'rank' + rank).write_text(str(os.getpid()))\n"
               "if rank == '1':\n"
               "    worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
               "    pathlib.Path(sys.argv[1], 'worker').write_text(str(worker.pid))\n"
               "    time.sleep(1)\n"
               "    sys.exit(3)\n"
               "time.sleep(600)\n")
    launch = launcher("--hosts", "node-a,node-b", "--processes-per-host", "1", "--port", "1", "--",
                      sys.executable, "-c", program, str(tmp_path),
                      env={**ENV, "PATH": f"{binaries}:{ENV['PATH']}"})
    try:
        launch.communicate(timeout=60)
    finally:
        launch.kill()
    assert launch.returncode == 3
    assert not running(int((tmp_path / "worker").read_text()))


@pytest.mark.parametrize("shell", ["sh", "bash"])
def test_a_rank_over_ssh_stopped_through_its_terminal_exits_with_its_own_code(tmp_path, shell):
    """The shell behind a rank's terminal waits again on a wait the stop cut
    short, so a rank that takes the stop and exits at once, before the shell
    looks again, still hands back its own code, not the shell's interrupted
    wait's, under either login shell. strace holds the shell after each kill it
    sends, so the rank has exited, and bash has reaped it, by the time the shell
    looks."""
    from dew.cli.launch import STOP_BYTE, remote_script

    binaries = ssh_stand_in(tmp_path, ("strace", "-qq", "-o", "/dev/null", "-e", "trace=kill",
                                       "-e", "inject=kill:delay_exit=300000", shell))
    program = ("import signal, sys, time\n"
               "signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(7))\n"
               "print('up', flush=True)\n"
               "time.sleep(600)\n")
    ssh = subprocess.Popen([str(binaries / "ssh"), "-tt", "node-a",
                            remote_script((sys.executable, "-c", program), {}, None)],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert ssh.stdin is not None and ssh.stdout is not None
    assert ssh.stdout.readline().strip() == "up"
    ssh.stdin.write(STOP_BYTE)
    ssh.stdin.flush()
    assert ssh.wait(timeout=60) == 7


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
