"""`dew launch`: the pool it starts, the commands it writes, and how it stops.

Each test runs the launcher as its own process against small Python
programs on this machine, so what is checked is what a shell sees: the exit
code, the output, and which processes are left alive.
"""

import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from conftest import outside_any_cluster

REPO_ROOT = Path(__file__).resolve().parents[1]
# The launches below stand in for Slurm, Open MPI or a plain host with
# variables of their own, on a machine in no cluster of its own.
ENV = {**outside_any_cluster(os.environ), "PYTHONPATH": str(REPO_ROOT / "src")}


def test_training_does_not_import_the_command_line():
    """The pool's variable names are shared with `prepare_process` through a
    module with no imports, so the training layer never loads the CLI."""
    probe = ("import sys, dew.training.runtime\n"
             "print(sorted(name for name in sys.modules if name.startswith('dew.cli')))\n")
    done = subprocess.run([sys.executable, "-c", probe], env=ENV, capture_output=True,
                          text=True, check=True)
    assert done.stdout.strip() == "[]"


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


def test_a_rank_printing_bytes_that_are_not_utf8_still_finishes():
    """One undecodable byte, then more output than a pipe holds: the relay
    keeps reading, so the rank never blocks on a full pipe, and the byte
    arrives as a replacement character."""
    program = ("import sys\n"
               "sys.stdout.buffer.write(b'\\xff\\xfe\\n')\n"
               "sys.stdout.buffer.write(b'x' * 300_000 + b'\\n')\n")
    launch = launcher("--port", "1", "--", sys.executable, "-c", program)
    try:
        output, _ = launch.communicate(timeout=60)
    finally:
        launch.kill()
    assert launch.returncode == 0
    assert "[0] \ufffd\ufffd" in output.decode()


def test_a_signalled_pool_gets_a_checkpoints_grace_and_a_second_signal_ends_it(tmp_path):
    """A stop signal gives the ranks the time a training run needs to reach
    the step they agree on and checkpoint it; a JAX rank takes SIGTERM as a
    preemption notice and runs on meanwhile. These ranks ignore SIGTERM as
    such a rank does: they still run past the 10 s a failed peer's pool
    gets, which cut every preemption checkpoint off, and a second signal
    ends them at once."""
    program = "import signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n" + WAITING_RANK
    launch = launcher("--processes-per-host", "2", "--port", "1", "--",
                      sys.executable, "-c", program, str(tmp_path))
    ranks = started_ranks(tmp_path, 2)
    launch.send_signal(signal.SIGTERM)
    time.sleep(15)
    assert launch.poll() is None and all(alive(pid) for pid in ranks)
    again = time.monotonic()
    launch.send_signal(signal.SIGTERM)
    assert launch.wait(timeout=30) == 128 + signal.SIGTERM
    assert time.monotonic() - again < 5
    assert not any(alive(pid) for pid in ranks)


def test_a_rank_killed_by_a_signal_exits_the_launch_128_plus_it():
    """The OOM killer's SIGKILL reads as 137, the shell's convention, which
    wrappers test for; Python's -9 would reach the shell as 247."""
    program = "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n"
    launch = launcher("--port", "1", "--", sys.executable, "-c", program)
    launch.communicate(timeout=60)
    assert launch.returncode == 137


@pytest.mark.parametrize("variable", ["A B=1", "X;touch y=1", "1X=1", "=1"])
def test_a_variable_name_the_shell_would_read_as_syntax_is_refused(variable):
    """A name is spliced unquoted into the remote shell line, so one that is
    not an identifier would run as something other than an assignment."""
    from dew.cli.launch import Launch

    with pytest.raises(ValueError, match="variable name"):
        Launch(command=("python",), hosts=("node1",), env=(variable,))
    assert Launch(command=("python",), env=("XLA_FLAGS=a=b,c",)).extra_env() == {
        "XLA_FLAGS": "a=b,c"}


def test_every_repeated_env_flag_reaches_the_ranks():
    """`--env A=1 --env B=2` is how the guide spells two variables; a flag
    that kept only its last value would drop the first without a word."""
    done = subprocess.run(
        [sys.executable, "-m", "dew.cli.main", "launch", "--env", "A=1", "--env", "B=2",
         "--port", "1", "--", sys.executable, "-c",
         "import os; print(os.environ['A'] + os.environ['B'])"],
        cwd=REPO_ROOT, env=ENV, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout
    assert "[0] 12" in done.stdout


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
        output, _ = launch.communicate(timeout=60)
    finally:
        launch.kill()
    ranks = [int(path.read_text()) for path in tmp_path.glob("rank*")]
    lines = output.decode().splitlines()
    assert launch.returncode == 3
    assert "rank 1 on localhost exited 3; stopping the other 1" in lines
    assert lines[-3:] == ["last lines of rank 1:", "[1] loading shard 7",
                          "[1] OSError: shard 7 is gone"]
    assert not any(alive(pid) for pid in ranks)


def test_a_failure_counts_only_the_ranks_it_stops(tmp_path):
    """Rank 0 has already finished when rank 1 fails, and rank 2 still
    runs: the launch stops one other rank, and says one."""
    program = ("import os, sys, time\n"
               "rank = os.environ['DEW_PROCESS_ID']\n"
               "if rank == '0': sys.exit(0)\n"
               "if rank == '1':\n"
               "    time.sleep(2); sys.exit(3)\n"
               "time.sleep(600)\n")
    # Three ranks split no GPU count evenly; these ones use none.
    launch = launcher("--processes-per-host", "3", "--port", "1", "--env", "JAX_PLATFORMS=cpu", "--",
                      sys.executable, "-c", program)
    try:
        output, _ = launch.communicate(timeout=60)
    finally:
        launch.kill()
    lines = output.decode().splitlines()
    assert launch.returncode == 3
    assert "rank 1 on localhost exited 3; stopping the other 1" in lines, lines


@pytest.mark.parametrize(("gpus", "processes", "devices", "expected"), [
    (4, None, None, (4, 1)),
    (8, 2, None, (2, 4)),
    (8, None, 2, (4, 2)),
    (1, None, None, (1, None)),
    (4, 1, None, (1, None)),
    (2, 4, None, (4, None)),
    (0, 4, None, (4, None)),
])
def test_a_pool_splits_the_gpus_of_a_host_between_its_processes(monkeypatch, gpus, processes,
                                                                 devices, expected):
    """Unset, a host runs one process per GPU; a process count alone gets
    an even share each, and more processes than GPUs share them all; a CPU
    pool leaves devices alone. The launcher names GPUs only where it splits
    them: one process a host holds every GPU and names none, and
    prepare_process keeps a cluster's detection from narrowing it."""
    from dew.cli import launch

    monkeypatch.setattr(launch, "gpu_count", lambda host, visible: gpus)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    plan = launch.Launch(command=("python",), processes_per_host=processes,
                         devices_per_process=devices)
    assert plan.layout("localhost") == expected


def test_a_share_of_gpus_that_does_not_divide_is_refused_and_cpu_pools_ignore_gpus(monkeypatch):
    from dew.cli import launch

    monkeypatch.setattr(launch, "gpu_count", lambda host, visible: 8)
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    with pytest.raises(ValueError, match="do not split"):
        launch.Launch(command=("python",), processes_per_host=3).layout("localhost")
    rehearsal = launch.Launch(command=("python",), processes_per_host=4,
                              env=("JAX_PLATFORMS=cpu",))
    assert rehearsal.layout("localhost") == (4, None)


def test_a_hostfile_names_hosts_by_the_first_word_of_each_line(tmp_path):
    """An MPI hostfile's `slots=` and comments are not host names."""
    from dew.cli.launch import Launch

    hostfile = tmp_path / "hosts"
    hostfile.write_text("# the rack\nnode0 slots=8\n\nnode1  # spare\n")
    assert Launch(command=("python",), hostfile=hostfile).host_names() == ("node0", "node1")
    assert Launch(command=("python",), hosts=("a,b", "c")).host_names() == ("a", "b", "c")


def launched(*arguments: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "dew.cli.main", "launch", *arguments],
                          cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120)


SLURM_STEP = {"SLURM_JOB_ID": "77", "SLURM_STEP_NODELIST": "gpu[01-02]", "SLURM_NTASKS": "16",
              "SLURM_PROCID": "3", "SLURM_LOCALID": "3"}


@pytest.mark.parametrize(("variables", "arguments", "expected"), [
    (SLURM_STEP, (), "slurm: process 3 of 16, placed by the cluster"),
    ({"OMPI_MCA_orte_hnp_uri": "1531576320.0;tcp://10.0.0.5:34911", "OMPI_COMM_WORLD_SIZE": "4",
      "OMPI_COMM_WORLD_RANK": "2", "OMPI_COMM_WORLD_LOCAL_RANK": "2"},
     (), "ompi: process 2 of 4, placed by the cluster"),
    ({"SLURM_JOB_ID": "77", "SLURM_NTASKS_PER_NODE": "8"}, (),
     "srun --kill-on-bad-exit=1 --export=ALL --label python train.py"),
    ({"SLURM_JOB_ID": "77", "SLURM_GPUS_PER_NODE": "a100:8"}, (),
     "srun --kill-on-bad-exit=1 --export=ALL --label --ntasks-per-node=8 python train.py"),
    (SLURM_STEP, ("--hosts", "localhost", "--processes-per-host", "1", "--port", "5"),
     "DEW_PROCESS_COUNT=1"),
    ({**SLURM_STEP, "SLURM_NTASKS": "1", "SLURM_PROCID": "0", "SLURM_LOCALID": "0",
      "CUDA_VISIBLE_DEVICES": "0,1"}, (), "pool: 2 processes on localhost, 1 GPU each"),
])
def test_where_the_launch_runs_follows_jax_detection_and_names_win(variables, arguments,
                                                                   expected):
    """A Slurm step or an Open MPI rank is already placed and runs the
    program in place, where jax reads the rank; a Slurm allocation outside a
    step starts srun; named hosts win over any cluster around them; a
    one-task step, as a scheduler's GPU wrapper runs, leaves its GPUs to a
    pool of the launcher's own."""
    env = {name: value for name, value in ENV.items() if name != "JAX_PLATFORMS"}
    done = launched("--dry-run", *arguments, "--", "python", "train.py",
                    env={**env, **variables})
    assert done.returncode == 0, done.stderr
    assert expected in done.stdout


@pytest.mark.parametrize(("variables", "arguments"), [
    ({"SLURM_JOB_ID": "77", "SLURM_NTASKS_PER_NODE": "1", "SLURM_GPUS_PER_NODE": "a100:4"}, ()),
    ({"SLURM_JOB_ID": "77", "SLURM_GPUS_PER_NODE": "4"}, ("--processes-per-host", "2")),
    ({**SLURM_STEP, "SLURM_NTASKS": "2", "SLURM_PROCID": "1", "SLURM_LOCALID": "0",
      "SLURM_STEP_TASKS_PER_NODE": "1(x2)", "SLURM_NODEID": "1", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}, ()),
], ids=["allocation", "processes-per-host", "step"])
def test_a_slurm_placement_that_leaves_gpus_idle_is_refused(variables, arguments):
    """jax gives each Slurm task the one GPU at its SLURM_LOCALID, so a node
    running fewer tasks than it has GPUs trains on that many and leaves the
    rest idle, silently: `sbatch --ntasks-per-node 1 --gpus-per-node 4`
    would use one GPU of four. An allocation the launcher starts srun in,
    and a step already running, are refused, naming the task count."""
    env = {name: value for name, value in ENV.items() if name != "JAX_PLATFORMS"}
    done = launched("--dry-run", *arguments, "--", "python", "train.py", env={**env, **variables})
    assert done.returncode != 0, done.stdout
    assert "--ntasks-per-node=4" in done.stderr, done.stderr


@pytest.mark.parametrize("arguments", [
    ("--processes-per-host", "2", "--devices-per-process", "1"),
    ("--devices-per-process", "2"),
], ids=["processes-and-devices", "devices"])
def test_a_pool_that_asks_for_more_gpus_than_the_host_shows_is_refused(arguments):
    """A host that shows one GPU, as CUDA_VISIBLE_DEVICES says here in place
    of a machine's own: a pool naming two GPUs is refused before any rank
    starts, with both counts. Unchecked, rank 1 took a GPU 1 that did not
    exist (Accuracy's one-A100 lane)."""
    env = {name: value for name, value in ENV.items()
           if not name.startswith(("SLURM_", "OMPI_"))}
    done = launched("--dry-run", *arguments, "--", "python", "train.py",
                    env={**env, "JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": "0"})
    assert done.returncode != 0, done.stdout
    assert "needs 2 GPUs on localhost, which shows 1" in done.stderr, done.stderr
    assert "JAX_COORDINATOR_ADDRESS" not in done.stdout, done.stdout


def failing_nvidia_smi(tmp_path: Path) -> dict:
    """An environment whose PATH finds an nvidia-smi that fails first, as a
    host without the driver's tools, or an ssh command's non-interactive
    PATH, gives it: the launcher cannot count this host's GPUs."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tool = bin_dir / "nvidia-smi"
    tool.write_text("#!/bin/sh\necho 'NVIDIA-SMI has failed' >&2\nexit 9\n")
    tool.chmod(0o755)
    env = {name: value for name, value in ENV.items()
           if not name.startswith(("SLURM_", "OMPI_")) and name != "CUDA_VISIBLE_DEVICES"}
    return {**env, "PATH": f"{bin_dir}:{os.environ['PATH']}", "JAX_PLATFORMS": "cuda"}


def test_a_host_whose_gpus_cannot_be_counted_starts_its_pool_unchecked(tmp_path):
    """A failing nvidia-smi counts nothing, which is not zero GPUs: a pool
    that names two starts as asked, and says the check was skipped. The
    same host with CUDA_VISIBLE_DEVICES=0 shows one GPU without asking
    nvidia-smi, and a pool that names two is refused."""
    env = failing_nvidia_smi(tmp_path)
    arguments = ("--dry-run", "--processes-per-host", "2", "--devices-per-process", "1",
                 "--", "python", "train.py")
    started = launched(*arguments, env=env)
    assert started.returncode == 0, started.stderr
    assert "JAX_LOCAL_DEVICE_IDS=1" in started.stdout, started.stdout
    assert "could not count localhost's GPUs" in started.stderr, started.stderr
    refused = launched(*arguments, env={**env, "CUDA_VISIBLE_DEVICES": "0"})
    assert refused.returncode != 0, refused.stdout
    assert "needs 2 GPUs on localhost, which shows 1" in refused.stderr, refused.stderr


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


def test_srun_refuses_a_per_process_gpu_count(tmp_path):
    """A per-task GPU count would narrow each task's visible GPUs to one that
    jax then numbers by the local rank."""
    done = launched("--devices-per-process", "1", "--", "python", "train.py",
                    env=fake_srun(tmp_path))
    assert done.returncode != 0
    assert "SLURM_LOCALID" in done.stderr
    assert not (tmp_path / "argv").exists()


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
    assert f"pool: {gpus} processes on localhost, 1 GPU each" in done.stdout
    placed = sorted(line.split("placed ", 1)[1] for line in done.stdout.splitlines()
                    if "placed " in line)
    assert placed == [f"{rank} {gpus} [{rank}]" for rank in range(gpus)]

