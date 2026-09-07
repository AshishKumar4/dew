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
The processes join through `prepare_process` from the environment a launcher
leaves, so the production join and the rendezvous right after it are what
every pool test runs.
"""

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh

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


def spawn(mode, out, processes=1, process_id=0, coordinator=None, devices=None, **flags):
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
        command, cwd=REPO_ROOT, env=worker_env(DEVICES // processes if devices is None else devices),
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


def run_pool(mode, directory: Path, processes: int, *, timeout=600, **flags) -> list[dict]:
    """`processes` workers in one pool, and their reports in process order."""
    directory.mkdir(parents=True, exist_ok=True)
    coordinator = f"127.0.0.1:{free_port()}"
    outs = [directory / f"process{index}.json" for index in range(processes)]
    running = [
        spawn(mode, out, processes=processes, process_id=index, coordinator=coordinator,
              **flags)
        for index, out in enumerate(outs)]
    try:
        return [report_of(process, out, timeout=timeout) for process, out in zip(running, outs)]
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
    i * seq_len and a process can report the records it read by index, with
    no contents.
    """
    directory.mkdir(parents=True, exist_ok=True)
    tokens = np.arange(records * seq_len + 1, dtype=np.uint16)
    (directory / "train.bin").write_bytes(tokens.tobytes())
    (directory / "val.bin").write_bytes(tokens.tobytes())
    (directory / "meta.json").write_text(json.dumps({
        "tokenizer": "byte", "vocab_size": int(tokens.max()) + 1, "dtype": "uint16",
        "train_tokens": len(tokens), "val_tokens": len(tokens)}))
    return directory


def document_corpus(directory: Path, documents: int, length) -> Path:
    """A corpus whose documents each say which document they are.

    Document i is the token value i + 1 repeated `length` times (one length
    for all, or one per document), closed by the eos id 0, so the values in a
    packed window name the documents packed into it and padding, which is
    also 0, names none.
    """
    directory.mkdir(parents=True, exist_ok=True)
    lengths = [length] * documents if isinstance(length, int) else list(length)
    # The dtype has to stay uint16 through the join, or the file holds twice
    # the bytes and reads back as a different corpus entirely.
    stream = np.concatenate([
        np.array([index + 1] * lengths[index] + [0], np.uint16)
        for index in range(documents)])
    assert stream.dtype == np.uint16
    (directory / "train.bin").write_bytes(stream.tobytes())
    (directory / "val.bin").write_bytes(stream.tobytes())
    (directory / "meta.json").write_text(json.dumps({
        "tokenizer": "byte", "vocab_size": documents + 1, "dtype": "uint16",
        "eos_id": 0, "train_tokens": len(stream), "val_tokens": len(stream)}))
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
        assert report["mesh_shape"] == {"data": DEVICES // 2, "expert": 1, "fsdp": 2,
                                        "tensor": 1, "sequence": 1, "stage": 1}
        assert report["mesh_devices"] == DEVICES
        assert report["mesh_process_indices"] == [0, 1]


@pytest.mark.distributed
def test_four_processes_build_the_same_mesh_as_two(tmp_path):
    """Two devices per process, four processes, one mesh over all eight."""
    reports = run_pool("topology", tmp_path, 4, fsdp_size=2)
    for index, report in enumerate(reports):
        assert report["process_index"] == index
        assert report["local_device_count"] == DEVICES // 4
        assert report["mesh_shape"] == {"data": DEVICES // 2, "expert": 1, "fsdp": 2,
                                        "tensor": 1, "sequence": 1, "stage": 1}
        assert report["mesh_process_indices"] == [0, 1, 2, 3]


@pytest.mark.distributed
def test_a_default_run_name_takes_its_timestamp_from_process_zero(two_processes):
    """The date in a default run name is the checkpoint directory's name, and
    every process writes into that directory, so the name comes from process
    zero's clock. The worker sets process 1's clock a year ahead; a name
    built from it would put process 1's shards in a directory of their own."""
    stamps = [report["run_timestamp"] for report in two_processes]
    assert stamps[0] == stamps[1]
    assert int(stamps[0][:4]) == two_processes[0]["own_year"]
    assert int(stamps[1][:4]) != two_processes[1]["own_year"], "the skew did not apply"


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
        # The dataset reports the whole corpus (a run turns that into steps
        # per epoch) and batches the local slice of the global batch.
        assert report["train_len"] == records
        assert report["global_batch_size"] == worker.BATCH
        assert report["local_batch_size"] == worker.BATCH // 2
        assert report["batches"] == records // worker.BATCH


@pytest.mark.distributed
def test_processes_pack_disjoint_documents(tmp_path):
    """The packed loader shards by slicing documents, which is its own rule.

    It cannot use ShardByJaxProcess, because sharding after the packer would
    have every process pack the same documents into the same windows, so it
    strides the document list instead. A process may only see documents from
    its own stride, and never one of another's.
    """
    documents, seq_len = 24, 6
    corpus = document_corpus(tmp_path / "corpus", documents, length=seq_len)
    reports = run_pool("packed", tmp_path / "out", 2, tokens=corpus,
                       seq_len=seq_len, workers=2)

    packed = [report["documents"] for report in reports]
    for index, seen in enumerate(packed):
        assert seen, f"process {index} packed nothing"
        # Document i is value i + 1, and the stride keeps i for process i % 2.
        assert all((value - 1) % 2 == index for value in seen), seen
    assert not set(packed[0]) & set(packed[1]), "both processes packed one document"
    assert set(packed[0] + packed[1]) <= set(range(1, documents + 1))
    for report in reports:
        assert report["local_batch_size"] == worker.BATCH // 2
        assert report["windows"] % (worker.BATCH // 2) == 0, "a partial batch came out"


@pytest.mark.distributed
def test_a_validation_split_packed_unevenly_ends_on_every_process(tmp_path):
    """Every process scores the batch count all of them have.

    The packed split strides its documents over the processes and packs each
    stride on its own. Of these 60 documents, 16 on process 0's stride and
    20 on process 1's fill a window and the rest are one eos each, so the
    strides pack into 18 and 22 windows, 4 and 5 batches of 4. Each batch is
    agreed before it is scored, so both score 4; a process that bounded its
    own pass with an islice would issue a fifth validation collective after
    the other had left the pass, and the pool would sit in it until the
    heartbeat killed both.
    """
    seq_len, val_steps = 8, 5
    lengths = [seq_len if index // 2 < (16, 20)[index % 2] else 0 for index in range(60)]
    corpus = document_corpus(tmp_path / "corpus", 60, lengths)
    reports = run_pool("fit", tmp_path / "out", 2, name="uneven", run_dir=tmp_path / "run",
                       fsdp_size=2, steps=2, records=RECORDS, tokens=corpus,
                       seq_len=seq_len, val_steps=val_steps)

    assert [report["val_available"] for report in reports] == [4, 5]
    for report in reports:
        assert report["val_batches"] == 4
        assert report["step"] == 2


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

    from dew.training.distributed import shard_batch

    trainer = worker.build_trainer("single", tmp_path / "single-run", fsdp=2)
    state, _, _ = trainer.place()
    batch = shard_batch(trainer.device_mesh, {"image": worker.global_images()})
    compiled = trainer.compile(state, batch)
    losses = []
    for _ in range(steps):
        state, loss, _, _, _ = compiled(state, batch)
        losses.append(float(loss))

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


@pytest.mark.distributed
def test_two_processes_run_the_pipeline_one_process_runs(tmp_path):
    """A two-stage pipeline over a pool of two processes, each holding half
    of every stage's devices, against this process running the same twenty
    steps on the eight devices itself, pipelined and whole. Largest observed
    difference on CPU: 4.8e-07 for the losses and 2.9e-06 for the
    parameters, against a tolerance of rtol 2e-4 and atol 2e-5.
    """
    steps = 20
    pool = run_pool("pipeline", tmp_path / "pool", 2, fsdp_size=2, stage_size=2,
                    microbatches=4, steps=steps)

    piped, piped_state = worker.pipeline_losses(
        worker.pipeline_trainer(2, 4, 2), worker.token_batch(), steps)
    whole, whole_state = worker.pipeline_losses(
        worker.pipeline_trainer(1, None, 4), worker.token_batch(), steps)

    assert len(pool[0]["losses"]) == steps
    assert np.isfinite(pool[0]["losses"]).all(), "the pool diverged"
    assert pool[0]["losses"] == pool[1]["losses"], "the processes disagreed with each other"
    assert pool[0]["mesh_shape"]["stage"] == 2
    np.testing.assert_allclose(pool[0]["losses"], piped, rtol=PARITY["rtol"], atol=PARITY["atol"])
    np.testing.assert_allclose(pool[0]["losses"], whole, rtol=PARITY["rtol"], atol=PARITY["atol"])
    assert pool[0]["sharding"]["fully_addressable"] == [False]
    assert_same_parameters(dumped_params(tmp_path / "pool" / "process0.json"),
                           worker.params_dict(whole_state.params))


# --------------------------------------------------------------------------
# Checkpoints between topologies
# --------------------------------------------------------------------------

@pytest.mark.distributed
def test_a_checkpoint_written_by_a_pool_restores_in_one_process(tmp_path):
    """Two processes write the shards of one checkpoint; one process reads it.

    A checkpoint is bytes, not arithmetic, so the restored parameters have
    to be equal, not close. The reading run also uses a different mesh, as a
    resume on smaller hardware does.
    """
    pool = run_pool("steps", tmp_path / "pool", 2, fsdp_size=2, steps=4, save=True,
                    name="pool", run_dir=tmp_path / "written")
    written = worker.checkpoint_dir(tmp_path / "written", "pool")
    assert pool[0]["checkpoint_path"] == str(written)

    restored = run_worker("steps", tmp_path / "restored.json", fsdp_size=1, steps=0,
                          save=True, name="pool", run_dir=tmp_path / "written")
    assert restored["restored_step"] == 4
    assert restored["sharding"]["specs"] == ["P()"], "a replicated run sharded something"
    assert largest_difference(dumped_params(tmp_path / "restored.json"),
                              dumped_params(tmp_path / "pool" / "process0.json")) == 0.0


@pytest.mark.distributed
def test_a_checkpoint_written_by_one_process_restores_in_a_pool(tmp_path):
    """The other direction, onto a mesh whose fsdp axis is four wide.

    Every process has to end up with the same parameters the writer had, and
    with arrays that span the pool, not copies of a local restore.
    """
    single = run_worker("steps", tmp_path / "single.json", fsdp_size=1, steps=4,
                        save=True, name="single", run_dir=tmp_path / "written")
    written = worker.checkpoint_dir(tmp_path / "written", "single")
    assert single["checkpoint_path"] == str(written)

    pool = run_pool("steps", tmp_path / "pool", 2, fsdp_size=4, steps=0, save=True,
                    name="single", run_dir=tmp_path / "written")
    expected = dumped_params(tmp_path / "single.json")
    for index, report in enumerate(pool):
        assert report["restored_step"] == 4
        assert report["mesh_shape"] == {"data": 2, "expert": 1, "fsdp": 4, "tensor": 1, "sequence": 1, "stage": 1}
        assert report["sharding"]["fully_addressable"] == [False]
        assert largest_difference(
            dumped_params(tmp_path / "pool" / f"process{index}.json"), expected) == 0.0


# --------------------------------------------------------------------------
# Resuming a pool
# --------------------------------------------------------------------------

POOL_STEPS = 3


def shard_of(position: str) -> int:
    """The process whose shard a saved grain position belongs to."""
    return int(re.search(r"shard_index=(\d+)", json.loads(position)["sampler"]).group(1))


@pytest.fixture(scope="module")
def pool_checkpoint(tmp_path_factory):
    """A three-step fit on two processes, saved with each one's data position."""
    directory = tmp_path_factory.mktemp("pool-checkpoint")
    reports = run_pool("fit", directory / "out", 2, name="pool", run_dir=directory / "run",
                       fsdp_size=2, steps=POOL_STEPS, records=RECORDS)
    for index, report in enumerate(reports):
        assert report["written_steps"] == [POOL_STEPS]
        assert shard_of(report["dataset_state"]) == index
    return {"checkpoints": worker.checkpoint_dir(directory / "run", "pool"),
            "reports": reports}


@pytest.mark.distributed
def test_a_pool_resumes_every_process_at_its_own_position(tmp_path, pool_checkpoint):
    """A checkpoint hands each process back the position that process wrote.

    The steps-mode checkpoint tests above never call fit, so they save no
    position and this could not show up there: a checkpoint holding one
    position, which orbax writes from process 0, hands process 0's shard to
    process 1, and grain refuses a sampler whose shard_index is not its own.
    Process 1 died at load and process 0 followed in its next collective.
    With one row per process the resumed pool has to land where a pool
    nobody stopped lands, at the same final position with the same
    parameters.
    """
    resumed = run_pool("fit", tmp_path / "resumed", 2, name="pool",
                       run_dir=pool_checkpoint["checkpoints"].parent, fsdp_size=2,
                       steps=2 * POOL_STEPS, records=RECORDS)
    whole = run_pool("fit", tmp_path / "whole", 2, name="whole",
                     run_dir=tmp_path / "whole-run", fsdp_size=2,
                     steps=2 * POOL_STEPS, records=RECORDS)
    for index, report in enumerate(resumed):
        assert report["restored_step"] == POOL_STEPS
        assert report["restored_dataset_state"] == pool_checkpoint["reports"][index]["dataset_state"]
        assert shard_of(report["restored_dataset_state"]) == index
        assert report["step"] == 2 * POOL_STEPS
        assert report["dataset_state"] == whole[index]["dataset_state"]
        assert_same_parameters(dumped_params(tmp_path / "resumed" / f"process{index}.json"),
                               dumped_params(tmp_path / "whole" / f"process{index}.json"))


@pytest.mark.distributed
def test_a_pool_checkpoint_refuses_a_different_process_count(tmp_path, pool_checkpoint):
    """One process cannot take over two processes' positions, and says so.

    Each position is where one shard stopped, and a sampler over one shard
    of one has no such place. Before the positions were per process this
    surfaced as grain's repr comparison of two samplers; now it stops at load
    with both counts in the message.
    """
    refused = spawn("fit", tmp_path / "single.json", fsdp_size=1, steps=2 * POOL_STEPS,
                    records=RECORDS, name="pool", run_dir=pool_checkpoint["checkpoints"].parent)
    log = refused.communicate(timeout=600)[0]
    assert refused.returncode != 0, "one process resumed a two-process position"
    assert "position for each of 2 processes and this run has 1 process" in log
    assert "Sampler in checkpoint" not in log, "grain's repr error is what the user sees"


@pytest.fixture(scope="module")
def elastic_checkpoint(tmp_path_factory):
    """A three-step fit on two processes over `train_stream`, saved."""
    directory = tmp_path_factory.mktemp("elastic-checkpoint")
    reports = run_pool("fit", directory / "out", 2, name="elastic", elastic=True,
                       run_dir=directory / "run", fsdp_size=1, steps=POOL_STEPS,
                       records=RECORDS)
    positions = {report["dataset_state"] for report in reports}
    assert len(positions) == 1, "the processes saved different global positions"
    saved = json.loads(positions.pop())[worker.ENVELOPE]
    assert saved["records"] == POOL_STEPS * worker.BATCH
    return {"run_dir": directory / "run", "reports": reports}


@pytest.mark.distributed
@pytest.mark.parametrize("processes", [1, 4])
def test_a_pool_position_resumes_on_another_process_count(tmp_path, elastic_checkpoint,
                                                          processes):
    """Two processes save where the run's data stopped, and one process or
    four take it over.

    Real processes, because a pool is where the position is gathered onto
    every host, written by orbax from process zero and read back by
    `read_position` against a process count that is not the one that wrote
    it. The resumed pool has to land on the parameters of a pool of its own
    size that nobody stopped: the global batches after the resume are the
    global batches that run trained on, and the loss is a mean over the
    step's rows, so which process holds which row cannot change them.
    """
    directory = tmp_path / "resumed-run"
    shutil.copytree(elastic_checkpoint["run_dir"], directory)
    flags = {"name": "elastic", "elastic": True, "fsdp_size": 1,
             "steps": 2 * POOL_STEPS, "records": RECORDS}

    resumed = run_pool("fit", tmp_path / "resumed", processes, run_dir=directory, **flags)
    whole = run_pool("fit", tmp_path / "whole", processes,
                     run_dir=tmp_path / "whole-run", **flags)

    for index, report in enumerate(resumed):
        assert report["restored_step"] == POOL_STEPS
        assert json.loads(report["restored_dataset_state"])[worker.ENVELOPE]["records"] == (
            POOL_STEPS * worker.BATCH)
        assert report["step"] == 2 * POOL_STEPS
        assert report["dataset_state"] == whole[index]["dataset_state"]
        assert_same_parameters(dumped_params(tmp_path / "resumed" / f"process{index}.json"),
                               dumped_params(tmp_path / "whole" / f"process{index}.json"))


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

    Waiting on the checkpoint, not on a duration, is what makes the step the
    run dies past the same everywhere. The source has stopped handing out
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
    the parameters of the run nobody killed, to rtol 2e-4 and atol 2e-5, with
    0.0 the largest difference observed on CPU.
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

    resumed = run_worker("fit", tmp_path / "resumed.json", **fit_flags(run_dir))
    position = resumed["restored_dataset_state"]
    assert position is not None, "the checkpoint carries no position for the data pipeline"
    assert resumed["restored_step"] == SAVE_EVERY
    assert json.loads(position)["last_seen_indices"] == {
        "0": SAVE_EVERY * worker.BATCH - 1}, "the checkpoint holds the wrong position"
    assert first_batch_after(position) == list(
        range(SAVE_EVERY * worker.BATCH, (SAVE_EVERY + 1) * worker.BATCH))

    assert resumed["step"] == STEPS
    # Same final position means the same batches were consumed overall, in the
    # same order, with the two uncommitted ones redone, not skipped.
    assert resumed["dataset_state"] == whole_run["dataset_state"]
    assert_same_parameters(dumped_params(tmp_path / "resumed.json"), whole_run["params"])


@pytest.mark.slow
def test_two_preemptions_in_one_epoch_still_land_where_the_whole_run_did(tmp_path, whole_run):
    """Resuming a resume has to work: a rerun into the run's own directory
    continues from the last step that landed, so two kills and two resumes
    still end where the run nobody killed ended.
    """
    run_dir = tmp_path / "run"
    checkpoints = worker.checkpoint_dir(run_dir, "preempt")

    first = spawn("fit", tmp_path / "first.json",
                  **fit_flags(run_dir, block_after=BLOCK_AFTER,
                              marker=tmp_path / "blocked-first"))
    kill_when_blocked(first, tmp_path / "blocked-first",
                      committed(checkpoints, SAVE_EVERY))
    assert first.returncode == -signal.SIGKILL

    # The second window: three more steps, a checkpoint at six, killed again.
    second = spawn("fit", tmp_path / "second.json",
                   **fit_flags(run_dir, block_after=SAVE_EVERY,
                               marker=tmp_path / "blocked-second"))
    kill_when_blocked(second, tmp_path / "blocked-second",
                      committed(checkpoints, 2 * SAVE_EVERY))
    assert second.returncode == -signal.SIGKILL
    assert committed_steps(checkpoints)[-1] == 2 * SAVE_EVERY, "the run was not killed mid-epoch"

    third = run_worker("fit", tmp_path / "third.json", **fit_flags(run_dir))
    assert third["step"] == STEPS
    assert third["dataset_state"] == whole_run["dataset_state"]
    assert_same_parameters(dumped_params(tmp_path / "third.json"), whole_run["params"])


# --------------------------------------------------------------------------
# Local checkpoints on a pool
# --------------------------------------------------------------------------

# A pool run with a local checkpoint every two steps beside the persistent
# one every three. Local step eight is the last local write of a full run
# (nine is the final persistent save) and the kill point of the preempted
# run, so both cases look for it.
LOCAL_EVERY = 2
LAST_LOCAL_STEP = 8
LOCAL_KILL_AFTER = LAST_LOCAL_STEP


def local_flags(directory: Path, **flags) -> dict:
    return {"name": "local", "run_dir": directory / "run", "fsdp_size": 2,
            "steps": STEPS, "save_every": SAVE_EVERY, "records": RECORDS,
            "local_dir": directory / "local", "local_every": LOCAL_EVERY, **flags}


def local_committed(directory: Path, process: int, step: int) -> Path:
    return committed(directory / "local" / f"process{process}", step)


@pytest.fixture(scope="module")
def killed_local_pool(tmp_path_factory):
    """The killed pool run: two processes, both SIGKILLed once every process
    has committed local step eight, with persistent steps three and six."""
    directory = tmp_path_factory.mktemp("killed-local")
    markers = [directory / f"blocked{index}" for index in range(2)]
    coordinator = f"127.0.0.1:{free_port()}"
    (directory / "killed").mkdir()
    pool = [spawn("fit", directory / "killed" / f"process{index}.json", processes=2,
                  process_id=index, coordinator=coordinator,
                  **local_flags(directory, block_after=LOCAL_KILL_AFTER, marker=marker))
            for index, marker in enumerate(markers)]
    deadline = time.monotonic() + 600
    landed = [local_committed(directory, index, LOCAL_KILL_AFTER) for index in range(2)]
    landed.append(committed(worker.checkpoint_dir(directory / "run", "local"), 2 * SAVE_EVERY))
    while time.monotonic() < deadline:
        if all(marker.exists() for marker in markers) and all(path.exists() for path in landed):
            break
        for process in pool:
            if process.poll() is not None:
                for other in pool:
                    terminate(other)
                pytest.fail(f"a process ended before the pool blocked\n"
                            f"{process.communicate()[0]}")
        time.sleep(0.05)
    else:
        for process in pool:
            terminate(process)
        pytest.fail(f"the pool did not block with local step {LOCAL_KILL_AFTER} landed")
    for process in pool:
        terminate(process)
    assert all(process.returncode == -signal.SIGKILL for process in pool)
    assert [int(marker.read_text()) for marker in markers] == [LOCAL_KILL_AFTER] * 2
    return directory


def persistent_bytes(checkpoints: Path) -> dict[str, bytes]:
    return {str(path.relative_to(checkpoints)): path.read_bytes()
            for path in sorted(checkpoints.rglob("*")) if path.is_file()}


@pytest.mark.distributed
def test_every_process_writes_its_own_local_checkpoint_every_n_steps(tmp_path):
    """A pool run to the end: every process holds exactly the newest local
    step under a directory of its own, the persistent directory holds the
    persistent cadence and nothing of the local one."""
    directory = tmp_path
    reports = run_pool("fit", directory / "out", 2, **local_flags(directory))
    for index, report in enumerate(reports):
        assert report["step"] == STEPS
        assert report["local_path"] == str(directory / "local" / f"process{index}")
        # Steps 2, 4, 6 and 8 were written; one is kept.
        assert report["local_steps"] == [LAST_LOCAL_STEP]
        assert local_committed(directory, index, LAST_LOCAL_STEP).exists()
        assert report["written_steps"] == [SAVE_EVERY, 2 * SAVE_EVERY, STEPS]
    assert committed_steps(worker.checkpoint_dir(directory / "run", "local")) == [
        SAVE_EVERY, 2 * SAVE_EVERY, STEPS]


@pytest.mark.distributed
def test_a_killed_pool_resumes_from_the_newer_local_checkpoint(tmp_path, killed_local_pool):
    """The resume reads local step eight on every process, not persistent
    step six, opens on the ninth batch, leaves the persistent checkpoints
    byte for byte as they were, and lands on the parameters of the run
    nobody killed."""
    directory = tmp_path / "killed"
    shutil.copytree(killed_local_pool, directory)
    persistent = worker.checkpoint_dir(directory / "run", "local")
    assert committed_steps(persistent) == [SAVE_EVERY, 2 * SAVE_EVERY]
    before = persistent_bytes(persistent)
    whole = run_pool("fit", tmp_path / "whole", 2, **local_flags(tmp_path / "whole-run"))

    resumed = run_pool("fit", tmp_path / "resumed", 2, **local_flags(directory))

    for index, report in enumerate(resumed):
        assert report["restored_step"] == LOCAL_KILL_AFTER
        assert report["restored_from"] == str(directory / "local" / f"process{index}")
        position = report["restored_dataset_state"]
        assert shard_of(position) == index
        # Eight batches of this process's shard, half the batch each, were
        # consumed when local step eight was written; grain strides the
        # sampler's index over the processes, so the last one seen is the
        # 32nd of this process's stride.
        assert json.loads(position)["last_seen_indices"] == {
            "0": (LOCAL_KILL_AFTER * worker.BATCH // 2 - 1) * 2 + index}
        assert report["step"] == STEPS
        assert report["dataset_state"] == whole[index]["dataset_state"]
        assert_same_parameters(dumped_params(tmp_path / "resumed" / f"process{index}.json"),
                               dumped_params(tmp_path / "whole" / f"process{index}.json"))
    # The resume added its final step and touched nothing that was there.
    after = persistent_bytes(persistent)
    assert committed_steps(persistent) == [SAVE_EVERY, 2 * SAVE_EVERY, STEPS]
    assert {name: after[name] for name in before} == before


@pytest.mark.distributed
def test_a_local_checkpoint_refuses_another_process_count(tmp_path, killed_local_pool):
    """One process cannot take over a pool's local checkpoint: it holds the
    shards of two processes' devices, and the refusal names both counts.
    Without the local directory the persistent checkpoint's own count
    refusal stands as before."""
    directory = tmp_path / "killed"
    shutil.copytree(killed_local_pool, directory)
    refused = spawn("fit", tmp_path / "single.json", **local_flags(directory, fsdp_size=1))
    log = refused.communicate(timeout=600)[0]
    assert refused.returncode != 0, "one process resumed a two-process local checkpoint"
    assert "written by 2 processes, and this run has 1 process" in log

    persistent = spawn("fit", tmp_path / "persistent.json", fsdp_size=1, steps=STEPS,
                       records=RECORDS, name="local", run_dir=directory / "run")
    log = persistent.communicate(timeout=600)[0]
    assert persistent.returncode != 0
    assert "position for each of 2 processes and this run has 1 process" in log

# --------------------------------------------------------------------------
# Validation and artifacts in a pool
# --------------------------------------------------------------------------

@pytest.mark.distributed
def test_a_pool_scores_a_diffusion_validation_pass(tmp_path):
    """The default diffusion validation on two ranks.

    Every rank holds one shard of the sampled grid and of the batch the metric
    reads, and numpy cannot read a shard at all: `evaluate` decoding captions
    off the batch's tokens with np.asarray, or the clip metric reading the
    artifact the same way, raises on every rank at once. The artifacts and the
    batch come home through one collective every rank makes, so the score is
    over the whole global batch and both ranks agree on it.
    """
    reports = run_pool("validate", tmp_path, 2, fsdp_size=2, steps=1)

    assert [report["process_count"] for report in reports] == [2, 2]
    assert [report["step"] for report in reports] == [1, 1]
    # The tracker draws on process zero only, and what it drew is the whole
    # grid, not this rank's quarter of it.
    drawn, rest = reports[0]["drawn"], reports[1]["drawn"]
    assert rest == []
    assert [entry["type"] for entry in drawn] == ["ImageGrid"]
    assert drawn[0]["shape"][0] == 4
    assert drawn[0]["captions"] == worker.PROMPTS[:4]
    score = reports[0]["scores"]["val/clip_similarity"]
    assert np.isfinite(score)
    # A metric that reads the whole field sees the global batch, not the rows
    # this rank happens to hold.
    assert reports[0]["scores"]["val/global_mean"] == pytest.approx(
        float(worker.global_images().mean()), rel=1e-6)

    single = run_worker("validate", tmp_path / "single.json", steps=1)
    assert single["scores"]["val/clip_similarity"] == pytest.approx(
        score, rel=PARITY["rtol"], abs=PARITY["atol"])


@pytest.mark.distributed
def test_a_pool_samples_rollouts_with_different_lengths_and_eos(tmp_path):
    """Two ranks own prompts of different lengths and stop at different steps.

    A shared padded shape and fixed decode trip count keep collectives in
    the same order despite different validity masks. Both ranks reach the
    rendezvous after sampling and after a peer rejects invalid input.
    The sampled rows, lengths and likelihoods match a single process over
    the same prompts, and the update that follows moves the same parameters.
    """
    reports = run_pool("rollout", tmp_path, 2, fsdp_size=2, timeout=90)
    single = run_worker("rollout", tmp_path / "single.json", fsdp_size=1)

    assert [report["arrivals"] for report in reports] == [[0, 1], [0, 1]]
    for report in reports:
        assert "vocabulary" in report["invalid_errors"]["token"]
        assert "prompt_length" in report["invalid_errors"]["length"]
        assert "single JAX PRNG key" in report["invalid_errors"]["key"]
    assert all(report["sampled_seconds"] < 300 for report in reports)
    for report in reports:
        assert report["process_count"] == 2
        assert report["sharding"]["fully_addressable"] == [False]
        assert report["step"] == 1
    expected = single["single"]
    pooled = {name: reports[0][name] + reports[1][name]
              for name in ("input_ids", "response_length", "terminated",
                           "response_mask", "old_log_probs", "behavior_log_probs")}
    assert pooled["input_ids"] == expected["input_ids"]
    assert pooled["response_length"] == expected["response_length"]
    assert pooled["terminated"] == expected["terminated"]
    assert pooled["response_mask"] == expected["response_mask"]
    np.testing.assert_allclose(pooled["old_log_probs"], expected["old_log_probs"],
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(pooled["behavior_log_probs"], expected["behavior_log_probs"],
                               rtol=1e-5, atol=1e-6)
    # The ranks stop at different steps and hold different prompt lengths.
    assert len(set(pooled["response_length"])) > 1
    assert reports[0]["response_length"] != reports[1]["response_length"]
    assert sorted(set(reports[0]["prompt_lengths"])) != sorted(set(reports[1]["prompt_lengths"]))
    # Stochastic draws: each rank's rows are drawn with their global row keys,
    # so the pool reproduces the single process token for token.
    for name in ("drawn_tokens", "drawn_lengths"):
        assert reports[0][name] + reports[1][name] == single[name]
    for name in ("drawn_behavior", "drawn_raw"):
        np.testing.assert_allclose(reports[0][name] + reports[1][name], single[name],
                                   rtol=1e-5, atol=1e-6)
    assert any(value < 0 for row in single["drawn_behavior"] for value in row)
    # A task takes a resident global array without fetching it to the host;
    # each rank generates from its own rows exactly as the direct call does.
    for report in reports:
        assert report["resident_addressable"] is False
        assert report["resident_tokens"] == report["direct_tokens"]
        np.testing.assert_allclose(report["resident_behavior"], report["direct_behavior"],
                                   rtol=1e-5, atol=1e-6)
    assert reports[0]["resident_tokens"] + reports[1]["resident_tokens"] == single["resident_tokens"]
    assert_same_parameters(dumped_params(tmp_path / "process0.json"),
                           dumped_params(tmp_path / "single.json"))


@pytest.mark.distributed
def test_a_pool_draws_a_jepa_artifact(tmp_path):
    """A tracker attached to a real pool run gets a complete artifact.

    The render path transfers with numpy on process zero, so a representation
    still sharded across the pool would raise there, and a renderer that
    gathered it from that one process would wedge the pool in a collective the
    others never enter. What process zero draws covers the global batch.
    """
    reports = run_pool("tracked", tmp_path, 2, fsdp_size=2, steps=1)

    assert [report["step"] for report in reports] == [1, 1]
    assert reports[1]["drawn"] == []
    drawn = reports[0]["drawn"]
    assert [entry["type"] for entry in drawn] == ["Representations"]
    assert drawn[0]["rendered"] is True
    assert drawn[0]["features"][0] == worker.BATCH
    assert drawn[0]["std"] > 0.0


@pytest.mark.distributed
def test_evaluation_coordinates_root_consumers_keys_and_host_failures(tmp_path):
    reports = run_pool("evaluation_contract", tmp_path, 2)
    first, second = (report["results"] for report in reports)
    for event in ("normal", "repeat", "untracked", "preview_only", "uneven"):
        assert first[event]["scores"] == second[event]["scores"]
    assert first["normal"]["scores"]["val/mean"] == first["repeat"]["scores"]["val/mean"]
    assert first["normal"]["scores"]["val/mean"] == first["untracked"]["scores"]["val/mean"]
    for report in reports:
        events = report["results"]["normal"]["events"]
        scoring = [key for kind, key in events if kind == "score"]
        preview = [key for kind, key in events if kind == "preview"]
        assert len(scoring) == 2 and scoring[0] != scoring[1]
        assert len(preview) == 1 and preview[0] not in scoring
        assert report["results"]["preview_only"]["events"][0][0] == "preview"
        assert len(report["results"]["preview_only"]["events"]) == 1
        assert report["results"]["preview_only"]["scores"]["evaluation/coordinated_batches"] == 1
        assert report["results"]["preview_only"]["scores"]["evaluation/records"] == 8
        assert report["results"]["uneven"]["scores"]["evaluation/coordinated_batches"] == 1
        assert report["results"]["uneven"]["scores"]["evaluation/uneven_shards"] == 1
        for phase in ("metric", "preview", "finalize", "log", "render", "construct", "next"):
            assert "error" in report["results"][phase], (phase, report)
            if phase != "construct":
                assert phase in report["closed"]
        assert "iterator next failed" in report["results"]["next"]["error"]
        assert report["results"]["empty"] == {"scores": {}, "events": []}
        assert report["results"]["no_consumer"] == {"scores": {}, "events": []}
        assert "no_consumer" not in report["closed"]
        assert report["results"]["normal"]["scores"]["evaluation/records"] == 16
        assert report["results"]["uneven"]["scores"]["evaluation/records"] == 8
        for invalid in ("mismatch", "duplicates"):
            assert "error" in report["results"][invalid]
            assert report["results"][invalid]["events"] == []
        for failure in ("deleted_first", "deleted_later", "deleted_batch", "deleted_preview"):
            assert "deleted" in report["results"][failure]["error"]
            assert failure in report["closed"]
        assert "gather plans differ" in report["results"]["mismatched_plan"]["error"]


@pytest.mark.distributed
@pytest.mark.parametrize("axis", ["stage", "sequence"])
def test_evaluation_counts_rows_once_across_replicated_process_axes(tmp_path, axis):
    reports = run_pool("evaluation_replicas", tmp_path, 2, devices=1, name=axis)
    assert reports[0]["measured"] == reports[1]["measured"]
    for report in reports:
        assert report["measured"]["val/count"] == 3
        assert report["measured"]["evaluation/records"] == 3
        assert report["no_consumer"] == {}
    assert reports[0]["local"] == [0, 1, 2]
    assert reports[1]["local"] is None


@pytest.mark.distributed
def test_builtin_previews_coordinate_nested_setup_generation_and_transfer_failures(tmp_path):
    reports = run_pool("builtin_preview_failures", tmp_path, 2, devices=1)
    for kind in ("lm", "diffusion", "masked"):
        for phase in ("setup", "generation", "preflight"):
            for source in (0, 1):
                case = f"{kind}-{phase}-{source}"
                local, remote = reports[source][case], reports[1 - source][case]
                assert local["closed"] and remote["closed"]
                assert remote["type"] == "RuntimeError"
                assert f"rank {source}" in remote["error"]
                if phase == "setup":
                    assert local["type"] == "AttributeError"
                elif phase == "generation":
                    assert local["type"] == "ValueError" and local["original"]
                else:
                    assert local["type"] == "RuntimeError" and "deleted" in local["error"]
        assert reports[0][f"{kind}-healthy"]["drawn"] == 1
        assert reports[1][f"{kind}-healthy"]["drawn"] == 0
        assert reports[0][f"{kind}-healthy"]["scores"] == reports[1][f"{kind}-healthy"]["scores"]


@pytest.mark.distributed
def test_composite_rejection_and_partial_restart_agree_across_processes(tmp_path):
    pool = run_pool("training_contract", tmp_path / "pool", 2, devices=1,
                    run_dir=str(tmp_path / "pool-state"))
    single = run_worker("training_contract", tmp_path / "single.json", devices=2,
                        run_dir=str(tmp_path / "single-state"))
    for report in [*pool, single]:
        assert report["flags"] == [[True, True], [True, False], [True, True], [True, True]]
        assert report["clocks"] == [4, 3, 1]
        assert report["resumed_equal"]
        assert report["scale"] == 65536.
    for report in pool:
        for want, got in zip(single["state"], report["state"], strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)

