"""Real processes, a real mesh across them, and runs that die mid-epoch.

Every other test in the suite simulates eight devices inside one process,
which is the one topology a bug on the process axis cannot show up in: a mesh
that stops at the local devices still covers every device, a pipeline that
shards by process still yields the whole dataset, and a checkpoint still has
one writer. These tests spawn processes that join a jax.distributed pool over a
loopback coordinator, and SIGKILL one mid-epoch so a resume has to prove
itself.

The pool runs on CPU with the simulated eight devices split among its
processes, so the global device count is the one the rest of the suite uses.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import multiprocess_worker as worker

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("multiprocess_worker.py")

# The simulated mesh the rest of the suite runs on, split among the processes.
DEVICES = 8
# tests/test_parallelism.py's tolerance for the same step on another topology.
PARITY = {"rtol": 2e-4, "atol": 2e-5}

# The preempted runs: nine steps, a checkpoint every three, killed two steps
# past the first one.
STEPS = 9
SAVE_EVERY = 3
BLOCK_AFTER = 5
RECORDS = worker.BATCH * 16


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def worker_env(devices: int) -> dict:
    """A worker's environment: this worktree's dew, on `devices` CPU devices."""
    return {**os.environ, "JAX_PLATFORMS": "cpu",
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}"}


def spawn(mode, out, processes=1, process_id=0, coordinator=None, **flags):
    """One worker process, started and not waited for.

    Its own session, so killing it takes down anything it spawned with it.
    """
    command = [sys.executable, str(WORKER), mode, "--out", str(out),
               "--processes", str(processes), "--process-id", str(process_id)]
    if coordinator is not None:
        command += ["--coordinator", coordinator]
    for name, value in flags.items():
        flag = "--" + name.replace("_", "-")
        if value is True:
            command.append(flag)
        elif value is not None:
            command += [flag, str(value)]
    return subprocess.Popen(
        command, cwd=REPO_ROOT, env=worker_env(DEVICES // processes),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True)


def terminate(process) -> None:
    """SIGKILL the worker and every process in its session."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=60)


def report_of(process, out: Path, timeout=600) -> dict:
    """What the worker recorded, once it has exited cleanly."""
    try:
        log = process.communicate(timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        terminate(process)
        pytest.fail(f"{out.name} did not finish within {timeout}s")
    assert process.returncode == 0, f"{out.name} exited {process.returncode}\n{log}"
    return json.loads(out.read_text())


def run_worker(mode, out: Path, **flags) -> dict:
    """One worker process, run to completion."""
    return report_of(spawn(mode, out, **flags), out)


def run_pool(mode, directory: Path, processes: int, **flags) -> list[dict]:
    """`processes` workers in one pool, and their reports in process order."""
    directory.mkdir(parents=True, exist_ok=True)
    coordinator = f"127.0.0.1:{free_port()}"
    outs = [directory / f"process{index}.json" for index in range(processes)]
    running = [
        spawn(mode, out, processes=processes, process_id=index, coordinator=coordinator,
              **flags)
        for index, out in enumerate(outs)]
    try:
        return [report_of(process, out) for process, out in zip(running, outs)]
    finally:
        for process in running:
            if process.poll() is None:
                terminate(process)


def dumped_params(out: Path) -> dict:
    """The parameter leaves a worker wrote next to its report."""
    with np.load(out.with_suffix(".npz")) as dump:
        return {name: dump[name] for name in dump.files}


def largest_difference(left: dict, right: dict) -> float:
    assert set(left) == set(right), "the two runs do not even hold the same parameters"
    return max(float(np.max(np.abs(left[name] - right[name]))) for name in left)


def assert_same_parameters(left: dict, right: dict) -> None:
    assert set(left) == set(right), "the two runs do not even hold the same parameters"
    for name in left:
        np.testing.assert_allclose(left[name], right[name], err_msg=name, **PARITY)


def token_corpus(directory: Path, records: int, seq_len: int) -> Path:
    """A tokenized corpus whose windows say which record they are.

    The stream is a ramp, so the window of record i opens with token
    i * seq_len and a process can report the records it read rather than their
    contents.
    """
    directory.mkdir(parents=True, exist_ok=True)
    tokens = np.arange(records * seq_len + 1, dtype=np.uint16)
    (directory / "train.bin").write_bytes(tokens.tobytes())
    (directory / "val.bin").write_bytes(tokens.tobytes())
    (directory / "meta.json").write_text(json.dumps({
        "tokenizer": "byte", "vocab_size": int(tokens.max()) + 1, "dtype": "uint16",
        "train_tokens": len(tokens), "val_tokens": len(tokens)}))
    return directory


@pytest.fixture(scope="module")
def two_processes(tmp_path_factory):
    """The topology two real processes on one 4x2 mesh report."""
    return run_pool("topology", tmp_path_factory.mktemp("topology"), 2, fsdp_size=2)


# --------------------------------------------------------------------------
# The mesh across processes
# --------------------------------------------------------------------------

@pytest.mark.distributed
def test_the_mesh_covers_every_process_in_the_pool(two_processes):
    """A mesh that stopped at the local devices would train two models.

    build_mesh takes jax.devices(), which inside a pool is every device of
    every process, so its axes have to multiply out to the global count and
    its devices have to come from every process. Nothing in a simulated
    single-process run can tell the two apart.
    """
    for index, report in enumerate(two_processes):
        assert report["process_index"] == index
        assert report["process_count"] == 2
        assert report["device_count"] == DEVICES
        assert report["local_device_count"] == DEVICES // 2
        assert report["mesh_shape"] == {"data": DEVICES // 2, "fsdp": 2}
        assert report["mesh_devices"] == DEVICES
        assert report["mesh_process_indices"] == [0, 1]


@pytest.mark.distributed
def test_four_processes_build_the_same_mesh_as_two(tmp_path):
    """Two devices per process, four processes, one mesh over all eight."""
    reports = run_pool("topology", tmp_path, 4, fsdp_size=2)
    for index, report in enumerate(reports):
        assert report["process_index"] == index
        assert report["local_device_count"] == DEVICES // 4
        assert report["mesh_shape"] == {"data": DEVICES // 2, "fsdp": 2}
        assert report["mesh_process_indices"] == [0, 1, 2, 3]


@pytest.mark.distributed
def test_the_global_batch_is_the_union_of_the_process_slices(two_processes):
    """Each process hands shard_batch its own rows and gets the whole batch.

    The rows a process can address have to be the ones it contributed, and the
    two sets have to cover the global batch exactly once. A process that
    assembled the wrong slice would train on duplicated data and never say so.
    """
    slices = [report["local_rows"] for report in two_processes]
    for report in two_processes:
        assert report["batch_shape"] == [worker.BATCH, worker.RES, worker.RES, 3]
        assert report["addressable_shards"] == DEVICES // 2
    assert not set(slices[0]) & set(slices[1]), "both processes hold the same rows"
    assert sorted(slices[0] + slices[1]) == list(range(worker.BATCH))
    assert slices[0] == list(range(worker.BATCH // 2)), "process 0 did not get its own rows"


# --------------------------------------------------------------------------
# The data pipeline across processes
# --------------------------------------------------------------------------

@pytest.mark.distributed
def test_processes_read_disjoint_shards_that_cover_the_corpus(tmp_path):
    """ShardByJaxProcess has to partition the records, not repeat them.

    The union of what the processes read is the dataset, the intersection is
    empty, and no process reads a record twice. Under one process the sharding
    is the identity, so this is the only place the property can fail.
    """
    records, seq_len = 32, 8
    corpus = token_corpus(tmp_path / "corpus", records, seq_len)
    reports = run_pool("data", tmp_path / "out", 2, tokens=corpus, seq_len=seq_len,
                       workers=2)

    shards = [report["records"] for report in reports]
    for shard in shards:
        assert len(set(shard)) == len(shard), "a record arrived twice in one process"
    assert not set(shards[0]) & set(shards[1]), "both processes read the same record"
    assert sorted(shards[0] + shards[1]) == list(range(records))
    for report in reports:
        # The loader reports the whole corpus, which is what a run turns into
        # steps per epoch, and batches the local slice of the global batch.
        assert report["train_len"] == records
        assert report["global_batch_size"] == worker.BATCH
        assert report["local_batch_size"] == worker.BATCH // 2
        assert report["batches"] == records // worker.BATCH


# --------------------------------------------------------------------------
# The same step, on more processes
# --------------------------------------------------------------------------

@pytest.mark.distributed
def test_two_processes_train_the_step_one_process_trains(tmp_path):
    """Splitting the batch across processes must not change the arithmetic.

    Same seed, same global batch, same 4x2 mesh; only the number of processes
    holding it differs, and the reference is this process running the same
    twenty steps on all eight devices itself. Largest observed difference on
    CPU is 6.0e-8 for the losses and 6.3e-8 for the parameters, against a
    tolerance of rtol 2e-4 and atol 2e-5.
    """
    steps = 20
    pool = run_pool("steps", tmp_path / "pool", 2, fsdp_size=2, steps=steps,
                    name="pool", run_dir=tmp_path / "pool-run")

    trainer = worker.build_trainer("single", tmp_path / "single-run", fsdp_size=2)
    state, _, losses = worker.run_losses(trainer, steps, worker.global_images())

    assert len(pool[0]["losses"]) == steps
    assert np.isfinite(pool[0]["losses"]).all(), "the pool diverged"
    assert pool[0]["losses"] == pool[1]["losses"], "the processes disagreed with each other"
    np.testing.assert_allclose(pool[0]["losses"], losses, **PARITY)

    # The pool's arrays really do span processes, so the parity above is not
    # two single-process runs agreeing with each other.
    assert pool[0]["sharding"]["fully_addressable"] == [False]
    assert pool[0]["sharding"]["device_counts"] == [DEVICES]

    assert_same_parameters(dumped_params(tmp_path / "pool" / "process0.json"),
                           worker.params_dict(state.params))


# --------------------------------------------------------------------------
# Checkpoints between topologies
# --------------------------------------------------------------------------

@pytest.mark.distributed
def test_a_checkpoint_written_by_a_pool_restores_in_one_process(tmp_path):
    """Two processes write the shards of one checkpoint; one process reads it.

    A checkpoint is bytes rather than arithmetic, so the restored parameters
    have to be equal, not close. The reading run also uses a different mesh,
    which is what a resume on smaller hardware does.
    """
    pool = run_pool("steps", tmp_path / "pool", 2, fsdp_size=2, steps=4, save=True,
                    name="pool", run_dir=tmp_path / "written")
    written = worker.checkpoint_dir(tmp_path / "written", "pool")
    assert pool[0]["checkpoint_path"] == str(written)

    restored = run_worker("steps", tmp_path / "restored.json", fsdp_size=1, steps=0,
                          load=written, name="restored", run_dir=tmp_path / "restored-run")
    assert restored["restored_step"] == 4
    assert restored["sharding"]["specs"] == ["P()"], "a replicated run sharded something"
    assert largest_difference(dumped_params(tmp_path / "restored.json"),
                              dumped_params(tmp_path / "pool" / "process0.json")) == 0.0


@pytest.mark.distributed
def test_a_checkpoint_written_by_one_process_restores_in_a_pool(tmp_path):
    """The other direction, onto a mesh whose fsdp axis is four wide.

    Every process has to end up with the same parameters the writer had, and
    with arrays that span the pool rather than copies of a local restore.
    """
    single = run_worker("steps", tmp_path / "single.json", fsdp_size=1, steps=4,
                        save=True, name="single", run_dir=tmp_path / "written")
    written = worker.checkpoint_dir(tmp_path / "written", "single")
    assert single["checkpoint_path"] == str(written)

    pool = run_pool("steps", tmp_path / "pool", 2, fsdp_size=4, steps=0, load=written,
                    name="pooled", run_dir=tmp_path / "pool-run")
    expected = dumped_params(tmp_path / "single.json")
    for index, report in enumerate(pool):
        assert report["restored_step"] == 4
        assert report["mesh_shape"] == {"data": 2, "fsdp": 4}
        assert report["sharding"]["fully_addressable"] == [False]
        assert largest_difference(
            dumped_params(tmp_path / "pool" / f"process{index}.json"), expected) == 0.0


# --------------------------------------------------------------------------
# Preemption
# --------------------------------------------------------------------------

def fit_flags(run_dir: Path, **flags) -> dict:
    return {"name": "preempt", "run_dir": run_dir, "fsdp_size": 2, "steps": STEPS,
            "save_every": SAVE_EVERY, "records": RECORDS, **flags}


def committed(checkpoints: Path, step: int) -> Path:
    """The file orbax writes last when a step has landed on disk."""
    return checkpoints / str(step) / "commit_success.txt"


def committed_steps(checkpoints: Path) -> list[int]:
    """The steps that landed, ignoring whatever a killed writer left behind."""
    return sorted(int(step.name) for step in checkpoints.iterdir()
                  if step.name.isdigit() and committed(checkpoints, int(step.name)).exists())


def kill_when_blocked(process, marker: Path, landed: Path, timeout=600) -> int:
    """SIGKILL the run once it is blocked mid-epoch with `landed` on disk.

    Waiting on the checkpoint rather than on a duration is what makes the step
    the run dies past the same everywhere: the source has stopped handing out
    batches, so nothing can advance while this waits.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.exists() and landed.exists():
            terminate(process)
            return int(marker.read_text())
        if process.poll() is not None:
            pytest.fail(f"the run ended before it blocked\n{process.communicate()[0]}")
        time.sleep(0.05)
    terminate(process)
    pytest.fail(f"no checkpoint at {landed} within {timeout}s")


def first_batch_after(position: str) -> list[int]:
    """The records a fresh loader hands out next from a saved position."""
    iterator = iter(worker.indexed_loader(RECORDS))
    iterator.set_state(position.encode())
    return worker.batch_records(next(iterator))


@pytest.fixture(scope="module")
def whole_run(tmp_path_factory):
    """One nine-step run that nothing interrupts, as the yardstick."""
    directory = tmp_path_factory.mktemp("whole")
    out = directory / "whole.json"
    report = run_worker("fit", out, **fit_flags(directory / "run", name="whole"))
    assert report["step"] == STEPS
    return {**report, "params": dumped_params(out)}


@pytest.mark.slow
def test_a_killed_run_resumes_on_the_batch_after_its_checkpoint(tmp_path, whole_run):
    """A preempted run redoes the work it never checkpointed, and no more.

    The killed process trained five steps and had committed three, so the
    resume has to open on the fourth batch, not on the sixth (the two steps
    whose work was never saved) and not on the first. It then has to land on
    the parameters of the run nobody killed: observed difference 0.0 on CPU,
    tolerance rtol 2e-4 and atol 2e-5.
    """
    run_dir = tmp_path / "run"
    checkpoints = worker.checkpoint_dir(run_dir, "preempt")
    marker = tmp_path / "blocked"
    killed = spawn("fit", tmp_path / "killed.json",
                   **fit_flags(run_dir, block_after=BLOCK_AFTER, marker=marker))
    handed = kill_when_blocked(killed, marker, committed(checkpoints, SAVE_EVERY))

    assert killed.returncode == -signal.SIGKILL, "the run was not preempted"
    assert handed == BLOCK_AFTER, "the run did not get past its checkpoint"
    assert committed_steps(checkpoints) == [SAVE_EVERY], "the kill was not mid-epoch"

    resumed = run_worker("fit", tmp_path / "resumed.json",
                         **fit_flags(run_dir, load=checkpoints))
    position = resumed["restored_dataset_state"]
    assert position is not None, "the checkpoint carries no position for the data pipeline"
    assert resumed["restored_step"] == SAVE_EVERY
    assert json.loads(position)["last_seen_indices"] == {
        "0": SAVE_EVERY * worker.BATCH - 1}, "the checkpoint holds the wrong position"
    assert first_batch_after(position) == list(
        range(SAVE_EVERY * worker.BATCH, (SAVE_EVERY + 1) * worker.BATCH))

    assert resumed["step"] == STEPS
    # Same final position means the same batches were consumed overall, in the
    # same order, with the two uncommitted ones redone rather than skipped.
    assert resumed["dataset_state"] == whole_run["dataset_state"]
    assert_same_parameters(dumped_params(tmp_path / "resumed.json"), whole_run["params"])


@pytest.mark.slow
def test_two_preemptions_in_one_epoch_still_land_where_the_whole_run_did(tmp_path, whole_run):
    """Resuming a resume has to work, and the guard has to hold in between.

    A run restarted against its own populated directory without being asked
    to resume refuses instead of overwriting it, which is the difference
    between a preemption costing three steps and costing the run. Two kills
    and two resumes still have to end where the run nobody killed ended.
    """
    run_dir = tmp_path / "run"
    checkpoints = worker.checkpoint_dir(run_dir, "preempt")

    first = spawn("fit", tmp_path / "first.json",
                  **fit_flags(run_dir, block_after=BLOCK_AFTER,
                              marker=tmp_path / "blocked-first"))
    kill_when_blocked(first, tmp_path / "blocked-first",
                      committed(checkpoints, SAVE_EVERY))
    assert first.returncode == -signal.SIGKILL

    clobber = spawn("fit", tmp_path / "clobber.json", **fit_flags(run_dir))
    refusal = clobber.communicate(timeout=600)[0]
    assert clobber.returncode != 0, "a fresh run took over a populated directory"
    assert f"already holds checkpoints up to step {SAVE_EVERY}" in refusal
    assert "Nothing has been deleted" in refusal
    assert committed(checkpoints, SAVE_EVERY).exists(), "the refused run deleted the step"

    # The second window: three more steps, a checkpoint at six, killed again.
    second = spawn("fit", tmp_path / "second.json",
                   **fit_flags(run_dir, load=checkpoints, block_after=SAVE_EVERY,
                               marker=tmp_path / "blocked-second"))
    kill_when_blocked(second, tmp_path / "blocked-second",
                      committed(checkpoints, 2 * SAVE_EVERY))
    assert second.returncode == -signal.SIGKILL
    assert committed_steps(checkpoints)[-1] == 2 * SAVE_EVERY, "the run was not killed mid-epoch"

    third = run_worker("fit", tmp_path / "third.json",
                       **fit_flags(run_dir, load=checkpoints))
    assert third["step"] == STEPS
    assert third["dataset_state"] == whole_run["dataset_state"]
    assert_same_parameters(dumped_params(tmp_path / "third.json"), whole_run["params"])
