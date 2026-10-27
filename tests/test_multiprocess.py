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

import collections
import json
import os
import re
import select
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import multiprocess_worker as worker
import numpy as np
import pytest
from lane_environment import MESH_DEVICES
from interop_support import assert_same_stored_tensors, written_tensors
from process_support import (
    free_port,
    report_of,
    run_pool,
    run_worker,
    serving,
    spawn,
    stuck,
    terminate,
    worker_env as worker_env,
)
from token_support import Corpus, write_token_corpus

from dew.checkpoints import FROZEN_STORE
from dew.position import ENVELOPE

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh


# The simulated mesh the rest of the suite runs on, split among the processes.
# tests/test_parallelism.py's tolerance for the same step on another topology.
PARITY = {"rtol": 2e-4, "atol": 2e-5}

# The preempted runs: nine steps, a checkpoint every three, killed two steps
# past the first one.
STEPS = 9
SAVE_EVERY = 3
BLOCK_AFTER = 5
RECORDS = worker.BATCH * 16
AVERAGING = Path(__file__).with_name("averaging_worker.py")
"""The worker of the pools that average runs."""


def dumped_params(out: Path, suffix: str = ".npz") -> dict:
    """The parameter leaves a worker wrote next to its report, or, at suffix ".update.npz", the
    per-parameter update its step took."""
    with np.load(out.with_suffix(suffix)) as dump:
        return {name: dump[name] for name in dump.files}


def relative_difference(left: dict, right: dict) -> float:
    """The largest disagreement over the scale of `right`.

    Gradients of different runs are compared this way, so a difference is
    read against the size of the gradient itself and not an absolute bound
    that a rate or a batch size would move.
    """
    scale = max(float(np.max(np.abs(leaf))) for leaf in right.values())
    return largest_difference(left, right) / scale


def largest_difference(left: dict, right: dict) -> float:
    assert set(left) == set(right), "the two runs do not even hold the same parameters"
    return max(float(np.max(np.abs(left[name] - right[name]))) for name in left)


def assert_same_parameters(left: dict, right: dict) -> None:
    assert set(left) == set(right), "the two runs do not even hold the same parameters"
    for name in left:
        np.testing.assert_allclose(left[name], right[name], err_msg=name, **PARITY)


def assert_identical_parameters(left: dict, right: dict) -> None:
    """Two runs of one program over the same processes and mesh: a resume
    restores the state, the key and the data position, so it reduces in the
    order the run it continues did, and lands on its parameters bit for bit."""
    assert set(left) == set(right), "the two runs do not even hold the same parameters"
    for name in left:
        np.testing.assert_array_equal(left[name], right[name], err_msg=name)


def token_corpus(directory: Path, records: int, seq_len: int) -> Corpus:
    """A tokenized corpus whose windows say which record they are.

    The stream is a ramp, so the window of record i opens with token
    i * seq_len and a process can report the records it read by index, with
    no contents.
    """
    tokens = np.arange(records * seq_len + 1, dtype=np.uint16)
    return write_token_corpus(directory, tokens, tokens, vocab_size=int(tokens.max()) + 1)


@pytest.mark.distributed

def test_a_field_no_step_reads_stays_on_the_host_across_processes(tmp_path):
    """Each process places its share of the rows, so a batch as placed is
    bigger than as read; the read mask is found by the placed signature, and
    the step runs on the batch with its unread field left on the host."""
    for report in run_pool("reads", tmp_path / "out", 2):
        # The batch as read has a share of the rows, so its own signature names no mask.
        assert report["read_differs_from_placed"]
        assert report["unused_on_host"] and report["x_placed"]
        # A stand-in of the placed rows, which the executable checks and drops.
        assert report["unused_shape"] == [MESH_DEVICES]
        assert report["sum"] == 3 * MESH_DEVICES

def test_a_real_token_budget_agrees_across_processes_after_logging_ticks(tmp_path):
    corpus = token_corpus(tmp_path / "tokens", records=16, seq_len=4)
    reports = run_pool("token_budget", tmp_path / "pool", 2, devices=1, tokens=corpus.train, seq_len=4,
                       timeout=60)
    assert [report["step"] for report in reports] == [3, 3]
    assert [report["updates"] for report in reports] == [3, 3]
    assert [report["logged"] for report in reports] == [3, 0]


@pytest.mark.distributed
def test_a_schedule_ends_its_stage_at_one_step_on_every_process(tmp_path):
    """The committed tokens a schedule's stage ends on are the pool's, summed over its processes, so
    two processes end the first stage at one step, the one a single process ends it at, and land on
    its parameters."""
    # Fifteen windows of four, whose tokens all stand in the model's vocabulary of 64.
    corpus = token_corpus(tmp_path / "tokens", records=15, seq_len=4)
    flags = {"tokens": corpus.train, "seq_len": 4, "steps": 6}
    pool = run_pool("schedule_fit", tmp_path / "pool", 2, **flags)
    alone = run_pool("schedule_fit", tmp_path / "alone", 1, **flags)
    assert [report["step"] for report in pool + alone] == [6, 6, 6]
    ends = {tuple(report["stage_ends"]) for report in pool + alone}
    assert len(ends) == 1, ends
    (end,) = ends.pop()
    assert 0 < end < 6 * worker.BATCH
    for index in range(2):
        assert_same_parameters(dumped_params(tmp_path / "pool" / f"process{index}.json"),
                               dumped_params(tmp_path / "alone" / "process0.json"))


@pytest.mark.distributed
def test_a_decision_release_partitioned_over_a_pool_publishes_its_readout(tmp_path):
    """A slot release placed over two processes, its readout partitioned across them, publishes every
    tensor it saves: its readout gathered home to process 0 beside its backbone, written to
    safetensors as `SafetensorsReload` writes them and exported as `NCCLPush` sends them, the tensors
    the release saves and exports on one host."""
    from test_slot_sources import slot_release

    from dew.interop import Pretrained

    source, _ = slot_release(tmp_path, "head")
    reports = run_pool("decision_publish", tmp_path / "out", 2, release=source, run_dir=tmp_path / "served")
    assert [report["partitioned"] for report in reports] == [True, True]
    host = Pretrained.load(source, dtype="float32", attention_impl="xla", on_host=True)
    host.save(tmp_path / "saved")
    assert_same_stored_tensors(written_tensors(tmp_path / "served"), written_tensors(tmp_path / "saved"))
    assert_same_stored_tensors(dumped_params(tmp_path / "out" / "process0.json"),
                               {name: np.asarray(value) for name, value in host.export().items()})


def document_corpus(directory: Path, documents: int, length) -> Corpus:
    """A corpus whose documents each say which document they are.

    Document i is the token value i + 1 repeated `length` times (one length
    for all, or one per document), closed by the eos id 0, so the values in a
    packed window name the documents packed into it and padding, which is
    also 0, names none.
    """
    lengths = [length] * documents if isinstance(length, int) else list(length)
    # The dtype has to stay uint16 through the join, or the file holds twice
    # the bytes and reads back as a different corpus entirely.
    stream = np.concatenate([
        np.array([index + 1] * lengths[index] + [0], np.uint16)
        for index in range(documents)])
    assert stream.dtype == np.uint16
    return write_token_corpus(directory, stream, stream, vocab_size=documents + 1, eos_id=0)


ZONES = ("UTC0", "ZZZ+11", "ZZZ-9", "ZZZ+5")
"""Each rank's time zone (TZ, POSIX's offset form, which needs no zone database): its wall clock
eleven, nine or five hours from the first rank's, as processes on hosts set apart read theirs, where
processes on one machine read nearly the same second."""


def zoned(mode, out, *, process_id: int, **flags):
    """`spawn`, the rank's wall clock in its own time zone (`ZONES`)."""
    return spawn(mode, out, process_id=process_id, environment={"TZ": ZONES[process_id]}, **flags)


@pytest.fixture(scope="module")
def two_processes(tmp_path_factory):
    """The topology two real processes on one 4x2 mesh report, each its clock in its own zone."""
    return run_pool("topology", tmp_path_factory.mktemp("topology"), 2, fsdp_size=2, start=zoned)


# --------------------------------------------------------------------------
# The mesh across processes
# --------------------------------------------------------------------------

@pytest.mark.distributed
def test_the_mesh_covers_every_process_in_the_pool(two_processes):
    """A mesh that stopped at the local devices would train two models.

    MeshSpec.build takes jax.devices(), which inside a pool is every device of
    every process, so its axes have to multiply out to the global count and
    its devices have to come from every process. Nothing in a simulated
    single-process run can tell the two apart.
    """
    for index, report in enumerate(two_processes):
        assert report["process_index"] == index
        assert report["process_count"] == 2
        assert report["device_count"] == MESH_DEVICES
        assert report["local_device_count"] == MESH_DEVICES // 2
        assert report["mesh_shape"] == {"data": MESH_DEVICES // 2, "expert": 1, "fsdp": 2,
                                        "tensor": 1, "sequence": 1, "stage": 1}
        assert report["mesh_devices"] == MESH_DEVICES
        assert report["mesh_process_indices"] == [0, 1]


@pytest.mark.distributed
def test_every_process_names_its_run_by_the_first_processs_clock(two_processes):
    """A run's default name holds its start time, and that name is the checkpoint directory every
    process writes into, so each process takes process 0's clock (`run_timestamp`): two processes
    whose own clocks read eleven hours apart, as hosts in other zones read theirs, name one run."""
    first, second = two_processes
    assert first["own_clock"] != second["own_clock"]
    assert first["run_timestamp"] == second["run_timestamp"]


@pytest.mark.distributed
def test_four_processes_build_the_same_mesh_as_two(tmp_path):
    """Two devices per process, four processes, one mesh over all eight."""
    reports = run_pool("topology", tmp_path, 4, fsdp_size=2, start=zoned)
    for index, report in enumerate(reports):
        assert report["process_index"] == index
        assert report["local_device_count"] == MESH_DEVICES // 4
        assert report["mesh_shape"] == {"data": MESH_DEVICES // 2, "expert": 1, "fsdp": 2,
                                        "tensor": 1, "sequence": 1, "stage": 1}
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
        assert report["addressable_shards"] == MESH_DEVICES // 2
    assert not set(slices[0]) & set(slices[1]), "both processes hold the same rows"
    assert sorted(slices[0] + slices[1]) == list(range(worker.BATCH))
    assert slices[0] == list(range(worker.BATCH // 2)), "process 0 did not get its own rows"


@pytest.mark.distributed
def test_process_placement_owns_its_reader_buffer(two_processes):
    for report in two_processes:
        assert not report["aliases_reader"]
        assert report["preserved_rows"] == report["local_rows"]


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
    reports = run_pool("data", tmp_path / "out", 2, tokens=corpus.train, held_tokens=corpus.val,
                       seq_len=seq_len, workers=2)

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
def test_processes_read_disjoint_windows_of_one_packing(tmp_path):
    """The packing is planned over the whole corpus, ahead of the shard, and
    the windows are what the processes divide.

    Sharding the documents instead, as this loader once did, made a window a
    fact about one process's own documents: no two process counts packed the
    same windows and a saved position belonged to the count that wrote it.
    With one plan, a process reads every nth window of it, so the union of
    what the pool reads is the split and the intersection is empty.
    """
    documents, seq_len = 24, 6
    # Each document is six tokens and an eos, so it fills a seven-id window
    # by itself and window i holds document i.
    corpus = document_corpus(tmp_path / "corpus", documents, length=seq_len)
    reports = run_pool("packed", tmp_path / "out", 2, tokens=corpus.train, held_tokens=corpus.val,
                       seq_len=seq_len, workers=2)

    packed = [report["documents"] for report in reports]
    for index, seen in enumerate(packed):
        assert seen, f"process {index} read nothing"
        # Window i is document i + 1 here, and process p reads windows
        # p, p + 2, ... of the one plan.
        assert all((value - 1) % 2 == index for value in seen), seen
    assert not set(packed[0]) & set(packed[1]), "both processes read one window"
    assert set(packed[0] + packed[1]) == set(range(1, documents + 1))
    for report in reports:
        assert report["local_batch_size"] == worker.BATCH // 2
        assert report["windows"] % (worker.BATCH // 2) == 0, "a partial batch came out"


WHOLE_VALIDATION_PARTS = 3
"""The pools the objectives' passes are spread over: one pool ran all of them
in 371 s, the longest test CI had, which no other test could run beside."""


@pytest.mark.distributed
@pytest.mark.parametrize("part", range(WHOLE_VALIDATION_PARTS))
def test_every_objective_scores_every_record_of_a_split_once_on_two_processes(tmp_path, part):
    """`tests/test_whole_validation.py`'s passes on a pool of two: each
    process reads its share of a split of a batch and one record, the share
    that runs out first scoring its last batch's copy with every row a
    repeat, and every objective's loss and `Perplexity` count each record once.
    A split of one record leaves process 1's share empty, and it is scored once.
    Each part is a pool of its own over every third objective, and the first
    also scores `Perplexity`."""
    from test_whole_validation import EVALUATED, assert_every_record_once, assert_the_metric_over_every_record

    reports = run_pool("whole_validation", tmp_path / "out", 2, run_dir=tmp_path / "run", timeout=1800,
                       part=f"{part}/{WHOLE_VALIDATION_PARTS}")
    assert reports[0]["passes"] == reports[1]["passes"]
    assert sorted(reports[0]["passes"]) == EVALUATED[part::WHOLE_VALIDATION_PARTS]
    for name, report in reports[0]["passes"].items():
        assert_every_record_once(name, report)
    if part == 0:
        assert_the_metric_over_every_record(reports[0]["perplexity"])
        assert_the_metric_over_every_record(reports[0]["one"], 1)


@pytest.mark.distributed
def test_a_metric_reads_the_host_fields_of_every_process_share(tmp_path):
    """`tests/test_whole_validation.py`'s metric over host fields on a pool of two: each process
    reads its share of every batch, and the root's metric reads every record's name and its list of tags
    beside the label the objective produced for that record, each record once."""
    from test_whole_validation import RECORDS

    reports = run_pool("host_fields", tmp_path / "out", 2)
    seen = [tuple(read) for read in reports[0]["seen"]]
    assert sorted(seen, key=lambda read: read[2]) == [(f"record {index}", index % 3 + 1, index)
                                                      for index in range(RECORDS)]


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

    from dew.distributed.batches import shard_batch

    trainer = worker.build_trainer("single", tmp_path / "single-run", fsdp=2)
    state, _, _ = trainer.place()
    batch = shard_batch(trainer.device_mesh, {"image": worker.global_images()})
    compiled = trainer.compile(state, batch)
    losses = []
    for _ in range(steps):
        state, loss, _, _, _, _ = compiled(state, batch)
        losses.append(float(loss))

    assert len(pool[0]["losses"]) == steps
    assert np.isfinite(pool[0]["losses"]).all(), "the pool diverged"
    assert pool[0]["losses"] == pool[1]["losses"], "the processes disagreed with each other"
    np.testing.assert_allclose(pool[0]["losses"], losses, **PARITY)

    # The pool's arrays really do span processes, so the parity above is not
    # two single-process runs agreeing with each other.
    assert pool[0]["sharding"]["fully_addressable"] == [False]
    assert pool[0]["sharding"]["device_counts"] == [MESH_DEVICES]

    assert_same_parameters(dumped_params(tmp_path / "pool" / "process0.json"),
                           worker.params_dict(state.variables))


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
                    microbatches=2, steps=steps)

    piped, _piped_state = worker.pipeline_losses(
        worker.pipeline_trainer(2, 2, 2), worker.token_batch(), steps)
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
                           worker.params_dict(whole_state.variables))


# --------------------------------------------------------------------------
# Checkpoints between topologies
# --------------------------------------------------------------------------

@pytest.mark.distributed
def test_a_checkpoint_written_by_a_pool_restores_in_one_process(tmp_path):
    """Two processes write the shards of one checkpoint; one process reads it.

    A checkpoint is bytes, not arithmetic, so the restored parameters have
    to be equal, not close. The reading run also uses a different mesh, as a
    resume on smaller hardware does. The DiT's `constants` (its time
    embedding's frequencies) go to the store once (`FROZEN_STORE`).
    """
    pool = run_pool("steps", tmp_path / "pool", 2, fsdp_size=2, steps=4, save=True,
                    name="pool", run_dir=tmp_path / "written")
    written = worker.checkpoint_dir(tmp_path / "written", "pool")
    assert pool[0]["checkpoint_path"] == str(written)
    assert len(list((written / FROZEN_STORE).iterdir())) == 1

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
        assert report["mesh_shape"] == {
            "data": 2,
            "expert": 1,
            "fsdp": 4,
            "tensor": 1,
            "sequence": 1,
            "stage": 1,
        }
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
def test_a_pools_observer_holds_every_processs_parameters_whole_at_each_observed_step(tmp_path):
    """An observer on two processes of one device each, whose parameters fsdp splits between them:
    every process runs the call and holds each observed step's variables whole, copied home before
    the next step donates them, the same on both, and the last observation is the variables the
    fit returns."""
    reports = run_pool("fit", tmp_path / "out", 2, devices=1, name="observed", run_dir=tmp_path / "run",
                       fsdp_size=2, steps=4, records=RECORDS, observe_every=2)
    assert all(report["split_between_processes"] for report in reports)
    observed = [dict(np.load(tmp_path / "out" / f"process{index}.observed.npz")) for index in range(2)]
    assert sorted({name.partition("[")[0] for name in observed[0]}) == ["2", "4"]
    assert_identical_parameters(*observed)
    # The fit's own dump gathers each global leaf whole (`params_dict`).
    final = dumped_params(tmp_path / "out" / "process0.json")
    assert_identical_parameters({name.removeprefix("4"): leaf for name, leaf in observed[0].items()
                                 if name.startswith("4")}, final)
    assert any(not np.array_equal(observed[0][f"2{name}"], leaf) for name, leaf in final.items())


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
        assert_identical_parameters(dumped_params(tmp_path / "resumed" / f"process{index}.json"),
                                    dumped_params(tmp_path / "whole" / f"process{index}.json"))


@pytest.fixture(scope="module")
def elastic_checkpoint(tmp_path_factory):
    """A three-step fit on two processes over a compiled global order, saved."""
    directory = tmp_path_factory.mktemp("elastic-checkpoint")
    reports = run_pool("fit", directory / "out", 2, name="elastic", elastic=True,
                       run_dir=directory / "run", fsdp_size=1, steps=POOL_STEPS,
                       records=RECORDS)
    positions = {report["dataset_state"] for report in reports}
    assert len(positions) == 1, "the processes saved different global positions"
    saved = json.loads(positions.pop())[ENVELOPE]
    assert saved["records"] == POOL_STEPS * worker.BATCH
    return {"run_dir": directory / "run", "reports": reports, "out": directory / "out"}


@pytest.mark.distributed
@pytest.mark.parametrize("records", ["resident", "http"])
@pytest.mark.parametrize("processes", [1, 4])
def test_a_pool_position_resumes_on_another_process_count(tmp_path, request, records, processes):
    """Two processes save where the run's data stopped, and one process or
    four take it over, over records held in memory or fetched over HTTP by
    every process into one global order.

    Real processes, because a pool is where the position is gathered onto
    every host, written by orbax from process zero and read back by
    `read_position` against a process count that is not the one that wrote
    it. The resumed pool has to land on the parameters of a pool of its own
    size that nobody stopped: the global batches after the resume are the
    global batches that run trained on, and the loss is a mean over the
    step's rows, so which process holds which row cannot change them.
    """
    resident = records == "resident"
    saved = request.getfixturevalue("elastic_checkpoint" if resident else "online_checkpoint")
    directory = tmp_path / "resumed-run"
    shutil.copytree(saved["run_dir"], directory)
    flags = {"name": "elastic", "elastic": True} if resident else {
        "name": "online", "online": request.getfixturevalue("online_server"), "environment": ONLINE}
    flags |= {"fsdp_size": 1, "steps": 2 * POOL_STEPS, "records": RECORDS}

    resumed = run_pool("fit", tmp_path / "resumed", processes, run_dir=directory, **flags)
    whole = run_pool("fit", tmp_path / "whole", processes,
                     run_dir=tmp_path / "whole-run", **flags)

    for index, report in enumerate(resumed):
        assert report["restored_step"] == POOL_STEPS
        assert json.loads(report["restored_dataset_state"])[ENVELOPE]["records"] == (
            POOL_STEPS * worker.BATCH)
        assert report["step"] == 2 * POOL_STEPS
        assert report["dataset_state"] == whole[index]["dataset_state"]
        assert_same_parameters(dumped_params(tmp_path / "resumed" / f"process{index}.json"),
                               dumped_params(tmp_path / "whole" / f"process{index}.json"))


def served(request) -> None:
    """A pool's server: each record's url answered with its image (`worker.served_image`), a pure
    function of the path, so processes that fetch at once read what one would."""
    body = worker.served_image(int(request.path.strip("/")))
    request.send_response(200)
    request.send_header("Content-Length", str(len(body)))
    request.end_headers()
    request.wfile.write(body)


ONLINE = {"NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}
"""The workers reach the parent's server directly, past any proxy their environment names."""


@pytest.mark.distributed
def test_processes_fetch_disjoint_shares_of_one_online_order(tmp_path):
    """Two processes each fetch their own share of one pass of a shuffled catalog over HTTP, to its end:
    their shares cover the corpus once, and the server answers each url once, so neither fetched a
    record of the other's share."""
    answered, lock = collections.Counter(), threading.Lock()

    def counted(request) -> None:
        with lock:
            answered[request.path] += 1
        served(request)

    with serving(counted) as address:
        reports = run_pool("online_data", tmp_path / "out", 2, online=address, records=RECORDS,
                           environment=ONLINE)
    shards = [report["records"] for report in reports]
    assert not set(shards[0]) & set(shards[1]), "both processes read the same record"
    assert sorted(shards[0] + shards[1]) == list(range(RECORDS))
    assert answered == {f"/{index}": 1 for index in range(RECORDS)}


@pytest.mark.distributed
def test_a_pool_whose_processes_fetched_other_records_is_refused_before_it_trains(tmp_path,
                                                                                online_checkpoint):
    """Every process of a memory read fetches the whole catalog itself. A fetch that fails for now on
    one of them (the first GET of a url answered 503) leaves that process holding other accepted
    records, and the pool is refused before any process trains, rather than cutting its shares from
    two different orders."""
    asked, lock = set(), threading.Lock()

    def flaky(request) -> None:
        with lock:
            first = request.path == "/5" and request.path not in asked
            asked.add(request.path)
        if not first:
            served(request)
            return
        request.send_response(503)
        request.send_header("Content-Length", "0")
        request.end_headers()

    with serving(flaky) as address:
        reports = run_pool("fit", tmp_path / "out", 2, name="online", online=address, refusal=True,
                           observe_every=1, run_dir=tmp_path / "run", fsdp_size=1, steps=POOL_STEPS,
                           records=RECORDS, environment=ONLINE)
    # The same pool over a server that answers every url trains (`online_checkpoint`); this one
    # observed no step and saved none.
    assert [(report["refused"], report["observed"], report["written_steps"]) for report in reports] == [
        ("ValueError", [], [])] * 2


@pytest.fixture(scope="module")
def online_server():
    with serving(served) as address:
        yield address


def online_run(directory: Path, address: str, *, shared: bool) -> dict:
    """`elastic_checkpoint`'s fit over the same records fetched from the pool's server, saved: every
    process fetching the whole catalog as a memory read does, one global order, or each its own
    share (`shared`)."""
    reports = run_pool("fit", directory / "out", 2, name="online", online=address, shared=shared or None,
                       run_dir=directory / "run", fsdp_size=1, steps=POOL_STEPS, records=RECORDS,
                       environment=ONLINE)
    return {"run_dir": directory / "run", "out": directory / "out", "reports": reports}


@pytest.fixture(scope="module")
def online_checkpoint(tmp_path_factory, online_server):
    saved = online_run(tmp_path_factory.mktemp("online-checkpoint"), online_server, shared=False)
    positions = {report["dataset_state"] for report in saved["reports"]}
    assert len(positions) == 1, "the processes saved different global positions"
    assert json.loads(positions.pop())[ENVELOPE]["records"] == POOL_STEPS * worker.BATCH
    return saved


@pytest.mark.distributed
def test_an_online_pool_trains_on_what_its_records_hold(online_checkpoint, elastic_checkpoint):
    """Records every process fetches over HTTP train the pool as the same records held in memory do:
    the same global batches in the same steps, to the same parameters bit for bit."""
    for index in range(2):
        assert_identical_parameters(dumped_params(online_checkpoint["out"] / f"process{index}.json"),
                                    dumped_params(elastic_checkpoint["out"] / f"process{index}.json"))


@pytest.mark.distributed
def test_a_pool_fetching_its_shares_online_resumes_where_it_stopped(tmp_path, online_server):
    """A pool whose processes each fetch their own share over HTTP saves each share's position and
    resumes on the same processes where it stopped: the resumed pool lands where the pool nobody
    stopped does, on its parameters bit for bit."""
    saved = online_run(tmp_path / "saved", online_server, shared=True)
    flags = {"name": "online", "online": online_server, "shared": True, "fsdp_size": 1,
             "steps": 2 * POOL_STEPS, "records": RECORDS, "environment": ONLINE}
    resumed = run_pool("fit", tmp_path / "resumed", 2, run_dir=saved["run_dir"], **flags)
    whole = run_pool("fit", tmp_path / "whole", 2, run_dir=tmp_path / "whole-run", **flags)
    for index, report in enumerate(resumed):
        assert report["restored_step"] == POOL_STEPS
        assert report["step"] == 2 * POOL_STEPS
        assert report["dataset_state"] == whole[index]["dataset_state"]
        assert_identical_parameters(dumped_params(tmp_path / "resumed" / f"process{index}.json"),
                                    dumped_params(tmp_path / "whole" / f"process{index}.json"))


PACKED_SEQ_LEN = 15


@pytest.fixture(scope="module")
def packed_checkpoint(tmp_path_factory):
    """A three-step fit on two processes over the packed training stream."""
    directory = tmp_path_factory.mktemp("packed-checkpoint")
    corpus = document_corpus(directory / "corpus", 48, length=PACKED_SEQ_LEN)
    reports = run_pool("packed_fit", directory / "out", 2, name="packed",
                       run_dir=directory / "run", fsdp_size=1, steps=POOL_STEPS,
                       tokens=corpus.train, seq_len=PACKED_SEQ_LEN)
    positions = {report["dataset_state"] for report in reports}
    assert len(positions) == 1, "the processes saved different global positions"
    saved = json.loads(positions.pop())[ENVELOPE]
    assert saved["records"] == POOL_STEPS * worker.BATCH
    # The token store states its documents' lengths, so the run reads it through a plan over the
    # whole store (`PackedSource`), and no Pack stage is left to run per share.
    from dew.data import presets
    from dew.data.analysis import _promoted
    from dew.data.packing import PackedSource
    from dew.data.pipeline import Pack
    from dew.data.sources.tokens import TokenFiles

    train = TokenFiles(corpus.train).pipe(presets.lm, None, seq_len=PACKED_SEQ_LEN, batch=None,
                                          pack="first-fit")
    read = _promoted(train)
    assert isinstance(read.source, PackedSource) and isinstance(read.source.plan.pack, Pack)
    assert not any(isinstance(stage, Pack) for stage in read.stages)
    return {"run_dir": directory / "run", "corpus": corpus, "reports": reports}


@pytest.mark.distributed
@pytest.mark.parametrize("processes", [1, 4])
def test_a_packed_pool_position_resumes_on_another_process_count(tmp_path,
                                                                 packed_checkpoint,
                                                                 processes):
    """The packed corpus is the case a resume used to refuse.

    Its windows are planned over the whole corpus ahead of the shard, so
    global batch k is the same windows at any process count, and the count
    two processes saved is where one process or four carry on. What is
    compared is the windows themselves, step by step, against a run of the
    resumed size that nobody stopped, and the parameters they trained.
    """
    directory = tmp_path / "resumed-run"
    shutil.copytree(packed_checkpoint["run_dir"], directory)
    flags = {"name": "packed", "fsdp_size": 1, "steps": 2 * POOL_STEPS,
             "tokens": packed_checkpoint["corpus"].train, "seq_len": PACKED_SEQ_LEN}

    resumed = run_pool("packed_fit", tmp_path / "resumed", processes,
                       run_dir=directory, **flags)
    whole = run_pool("packed_fit", tmp_path / "whole", processes,
                     run_dir=tmp_path / "whole-run", **flags)

    for index, report in enumerate(resumed):
        assert report["restored_step"] == POOL_STEPS
        assert json.loads(report["restored_dataset_state"])[ENVELOPE]["records"] == (
            POOL_STEPS * worker.BATCH)
        assert report["step"] == 2 * POOL_STEPS
        assert report["windows"] == whole[index]["windows"][POOL_STEPS:], (
            "the resumed run read other windows than the run nobody stopped")
        assert report["dataset_state"] == whole[index]["dataset_state"]
        assert_same_parameters(dumped_params(tmp_path / "resumed" / f"process{index}.json"),
                               dumped_params(tmp_path / "whole" / f"process{index}.json"))


@pytest.mark.distributed
def test_a_packed_pool_reads_the_windows_one_process_reads(tmp_path, packed_checkpoint):
    """The same claim without a checkpoint in it: the union of the windows a
    pool reads at a step is the step one process reads, row for row."""
    flags = {"name": "packed", "fsdp_size": 1, "steps": POOL_STEPS,
             "tokens": packed_checkpoint["corpus"].train, "seq_len": PACKED_SEQ_LEN}
    single = run_worker("packed_fit", tmp_path / "single.json", devices=MESH_DEVICES,
                        run_dir=tmp_path / "single-run", **flags)
    pool = run_pool("packed_fit", tmp_path / "pool", 2,
                    run_dir=tmp_path / "pool-run", **flags)

    for step in range(POOL_STEPS):
        rows = [row for held in zip(*(report["windows"][step] for report in pool), strict=True)
                for row in held]
        assert rows == single["windows"][step], f"step {step} read other windows"


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


def channel(path: Path) -> Path:
    """A FIFO at `path` for a worker to tell how many batches its source handed out once it is
    blocked mid-epoch with its checkpoints landed (multiprocess_worker.blocked); `told` reads it."""
    os.mkfifo(path)
    return path


def told(pool: list[tuple[subprocess.Popen, Path]], timeout: float = 600) -> list[int]:
    """What each worker of `pool` tells its channel, read as it arrives: one blocking wait at a time
    on the channels and on the workers' exits, and a worker that ends before it tells fails the
    test with its output. Every worker is killed, and the channels removed, however it ends. A
    worker that opens its channel first waits there until this opens it to read."""
    readers = [os.open(path, os.O_RDONLY | os.O_NONBLOCK) for _, path in pool]
    waiting = {reader: index for index, reader in enumerate(readers)}
    exits = {os.pidfd_open(process.pid): index for index, (process, _) in enumerate(pool)}
    heard: dict[int, int] = {}
    deadline = time.monotonic() + timeout
    try:
        while waiting:
            left = deadline - time.monotonic()
            # A worker that told is no longer waited on: its exit, once killed, is not a failure.
            unheard = [handle for handle, index in exits.items() if index not in heard]
            # The channels come first, so a worker's word is read before its exit in one wake.
            ready = select.select([*waiting, *unheard], [], [], left)[0] if left > 0 else []
            if not ready:
                pytest.fail(stuck([process for process, _ in pool],
                                  f"blocking with its checkpoints landed within {timeout}s"))
            for handle in ready:
                if handle in waiting:
                    heard[waiting.pop(handle)] = int(os.read(handle, 64))
                elif exits[handle] not in heard:
                    process = pool[exits[handle]][0]
                    pytest.fail(f"a run ended before it blocked\n{process.communicate()[0]}")
    finally:
        for handle in [*exits, *readers]:
            os.close(handle)
        for process, path in pool:
            terminate(process)
            path.unlink()
    return [heard[index] for index in range(len(pool))]


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
    the parameters of the run nobody killed, bit for bit.
    """
    run_dir = tmp_path / "run"
    checkpoints = worker.checkpoint_dir(run_dir, "preempt")
    blocked = channel(tmp_path / "blocked")
    killed = spawn("fit", tmp_path / "killed.json",
                   **fit_flags(run_dir, block_after=BLOCK_AFTER, notify=blocked))
    [handed] = told([(killed, blocked)])

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
    assert_identical_parameters(dumped_params(tmp_path / "resumed.json"), whole_run["params"])


@pytest.mark.slow
def test_two_preemptions_in_one_epoch_still_land_where_the_whole_run_did(tmp_path, whole_run):
    """Resuming a resume has to work: a rerun into the run's own directory
    continues from the last step that landed, so two kills and two resumes
    still end where the run nobody killed ended.
    """
    run_dir = tmp_path / "run"
    checkpoints = worker.checkpoint_dir(run_dir, "preempt")

    blocked = channel(tmp_path / "blocked-first")
    first = spawn("fit", tmp_path / "first.json",
                  **fit_flags(run_dir, block_after=BLOCK_AFTER, notify=blocked))
    told([(first, blocked)])
    assert first.returncode == -signal.SIGKILL
    assert committed(checkpoints, SAVE_EVERY).exists()

    # The second window: three more steps, a checkpoint at six, killed again.
    blocked = channel(tmp_path / "blocked-second")
    second = spawn("fit", tmp_path / "second.json",
                   **fit_flags(run_dir, block_after=SAVE_EVERY, notify=blocked))
    told([(second, blocked)])
    assert second.returncode == -signal.SIGKILL
    assert committed_steps(checkpoints)[-1] == 2 * SAVE_EVERY, "the run was not killed mid-epoch"

    third = run_worker("fit", tmp_path / "third.json", **fit_flags(run_dir))
    assert third["step"] == STEPS
    assert third["dataset_state"] == whole_run["dataset_state"]
    assert_identical_parameters(dumped_params(tmp_path / "third.json"), whole_run["params"])


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
    channels = [channel(directory / f"blocked{index}") for index in range(2)]
    coordinator = f"127.0.0.1:{free_port()}"
    (directory / "killed").mkdir()
    pool = [spawn("fit", directory / "killed" / f"process{index}.json", processes=2,
                  process_id=index, coordinator=coordinator,
                  **local_flags(directory, block_after=LOCAL_KILL_AFTER, notify=path))
            for index, path in enumerate(channels)]
    handed = told(list(zip(pool, channels, strict=True)))
    assert all(process.returncode == -signal.SIGKILL for process in pool)
    assert handed == [LOCAL_KILL_AFTER] * 2
    # Each process waited on its writes before it told: local step eight on each, persistent six.
    assert all(local_committed(directory, index, LOCAL_KILL_AFTER).exists() for index in range(2))
    assert committed(worker.checkpoint_dir(directory / "run", "local"), 2 * SAVE_EVERY).exists()
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
def test_a_local_save_that_fails_in_the_background_ends_the_pool(tmp_path):
    """A process whose local checkpoint fails in orbax's background thread
    (its tmp directory not made, as FileExistsError left one on a CI shard on
    2026-10-05; here a file stands where its local directory was) raises at
    its next save, and the pool ends with it, rather than the two processes
    waiting for ever in barriers the other will not reach: that shard's pool
    hung until its 600 s timeout."""
    directory = tmp_path
    coordinator = f"127.0.0.1:{free_port()}"
    (directory / "out").mkdir()
    pool = [spawn("fit", directory / "out" / f"process{index}.json", processes=2, process_id=index,
                  coordinator=coordinator, **local_flags(directory), fail_local_save=LOCAL_EVERY * 2)
            for index in range(2)]
    outputs = []
    for process in pool:
        try:
            outputs.append(process.communicate(timeout=120)[0])
        except subprocess.TimeoutExpired:
            pytest.fail(stuck(pool, "end after a failed local save"))
    assert pool[1].returncode != 0 and "NotADirectoryError" in outputs[1], outputs[1][-4000:]
    assert pool[0].returncode != 0, outputs[0][-4000:]


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
        assert_identical_parameters(dumped_params(tmp_path / "resumed" / f"process{index}.json"),
                                    dumped_params(tmp_path / "whole" / f"process{index}.json"))
    # The resume added its final step and touched nothing that was there.
    after = persistent_bytes(persistent)
    assert committed_steps(persistent) == [SAVE_EVERY, 2 * SAVE_EVERY, STEPS]
    assert {name: after[name] for name in before} == before


@pytest.mark.distributed
def test_a_host_that_lost_its_local_checkpoint_takes_a_replicas_copy(tmp_path, killed_local_pool):
    """Each process's devices of the killed pool hold every slice of the state (fsdp within a process,
    data across them), so a process whose local directory is gone, as a replaced host's is, takes the
    other's copy of local step eight, sent over the pool's devices: both resume from local step
    eight, not persistent step six, process one at its own data position, and land on the parameters
    of the run nobody killed."""
    directory = tmp_path / "killed"
    shutil.copytree(killed_local_pool, directory)
    shutil.rmtree(directory / "local" / "process1")
    whole = run_pool("fit", tmp_path / "whole", 2, **local_flags(tmp_path / "whole-run"))

    resumed = run_pool("fit", tmp_path / "resumed", 2, **local_flags(directory))

    for index, report in enumerate(resumed):
        assert report["holds_whole_state"]
        assert report["restored_step"] == LOCAL_KILL_AFTER
        assert report["restored_from"] == str(directory / "local" / f"process{index}")
        # The copy holds every process's data position, and process one resumes its own share's.
        assert shard_of(report["restored_dataset_state"]) == index
        assert report["step"] == STEPS
        assert report["dataset_state"] == whole[index]["dataset_state"]
        assert_identical_parameters(dumped_params(tmp_path / "resumed" / f"process{index}.json"),
                                    dumped_params(tmp_path / "whole" / f"process{index}.json"))


@pytest.fixture(scope="module")
def killed_split_pool(tmp_path_factory):
    """The killed pool run of `killed_local_pool` with fsdp over every device of the pool, so each
    process holds half of each sharded variable, and its local checkpoint half the state."""
    directory = tmp_path_factory.mktemp("killed-split")
    channels = [channel(directory / f"blocked{index}") for index in range(2)]
    coordinator = f"127.0.0.1:{free_port()}"
    (directory / "killed").mkdir()
    pool = [spawn("fit", directory / "killed" / f"process{index}.json", processes=2,
                  process_id=index, coordinator=coordinator,
                  **local_flags(directory, fsdp_size=MESH_DEVICES, block_after=LOCAL_KILL_AFTER, notify=path))
            for index, path in enumerate(channels)]
    handed = told(list(zip(pool, channels, strict=True)))
    assert all(process.returncode == -signal.SIGKILL for process in pool)
    assert handed == [LOCAL_KILL_AFTER] * 2
    assert all(local_committed(directory, index, LOCAL_KILL_AFTER).exists() for index in range(2))
    return directory


@pytest.mark.distributed
def test_a_host_whose_slices_no_survivor_holds_resumes_the_pool_from_the_persistent_checkpoint(
        tmp_path, killed_split_pool):
    """Where each process holds half the state, no survivor holds a lost process's slices, so a pool
    missing one process's local step resumes, every process, from persistent step six. The other's
    local step eight, from the run the kill cut short, gives way to the one the resumed run writes,
    and the pool lands on the parameters of the run nobody killed."""
    directory = tmp_path / "killed"
    shutil.copytree(killed_split_pool, directory)
    shutil.rmtree(directory / "local" / "process1")
    flags = {"fsdp_size": MESH_DEVICES}
    whole = run_pool("fit", tmp_path / "whole", 2, **local_flags(tmp_path / "whole-run", **flags))

    resumed = run_pool("fit", tmp_path / "resumed", 2, **local_flags(directory, **flags))

    for index, report in enumerate(resumed):
        assert not report["holds_whole_state"]
        assert report["restored_step"] == 2 * SAVE_EVERY
        assert report["restored_from"] == str(worker.checkpoint_dir(directory / "run", "local"))
        assert report["step"] == STEPS
        assert report["local_steps"] == [LAST_LOCAL_STEP]
        assert report["dataset_state"] == whole[index]["dataset_state"]
        assert_identical_parameters(dumped_params(tmp_path / "resumed" / f"process{index}.json"),
                                    dumped_params(tmp_path / "whole" / f"process{index}.json"))


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
def test_the_readers_of_one_share_score_and_pack_one_synchronous_rollout(tmp_path):
    """`tensor` across two processes, both reading the one share, a stochastic reward drawing
    differently on each: the first reader alone scores, and both hand the step the rows, advantages
    and likelihoods it packed, the reward run once a share."""
    first, second = run_pool("replicated_rollout", tmp_path, 2, timeout=120, devices=1)
    assert first["partition"] == {"count": 1, "readers": 2}
    assert first["rewarded"][0] > 0 and first["rewarded"][1] == 0
    for name, rows in first["handed"].items():
        np.testing.assert_array_equal(rows[1], rows[0], err_msg=name)
    assert second["handed"] == first["handed"]


@pytest.mark.distributed
def test_a_pool_samples_rollouts_with_different_lengths_and_eos(tmp_path):
    """Two ranks own prompts of different lengths and stop at different steps.

    A shared padded shape and fixed decode trip count keep collectives in
    the same order despite different validity masks. Both ranks reach the
    rendezvous after sampling and after a peer rejects invalid input or a
    decoding component only it can describe, and over weights on no mesh
    each rank's input is its own.
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
        assert "memory address" in report["invalid_errors"]["component"]
    # Over weights on no mesh, rank 1's zero prompt length is its own refusal
    # and rank 0 samples its rows without hearing of it.
    assert reports[0]["meshless"] == {"rows": 2 * len(reports[0]["prompt_lengths"])}
    assert "prompt_length" in reports[1]["meshless"]["refused"]
    assert all(report["sampled_seconds"] < 300 for report in reports)
    for report in reports:
        assert report["process_count"] == 2
        assert report["sharding"]["fully_addressable"] == [False]
        assert report["step"] == 1
    def completions(report):
        """Each completion's chain and columns, in rollout order: packing
        places chains per rank, so rows are compared through their chains."""
        index = np.asarray(report["session_index"])
        return [{name: np.asarray(report[name])[index == number]
                 for name in ("input_ids", "response_mask", "old_log_probs", "behavior_log_probs")}
                for number in range(int(index.max()) + 1)]

    pooled = completions(reports[0]) + completions(reports[1])
    expected = completions(single["single"])
    assert len(pooled) == len(expected)
    for mine, theirs in zip(pooled, expected, strict=True):
        np.testing.assert_array_equal(mine["input_ids"], theirs["input_ids"])
        np.testing.assert_array_equal(mine["response_mask"], theirs["response_mask"])
        for name in ("old_log_probs", "behavior_log_probs"):
            np.testing.assert_allclose(mine[name], theirs[name], rtol=1e-5, atol=1e-6)
    # The ranks stop at different steps and hold different prompt lengths.
    drawn = [int(completion["response_mask"].sum()) for completion in pooled]
    assert len(set(drawn)) > 1
    assert reports[0]["response_mask"] != reports[1]["response_mask"]
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
def test_a_pool_agrees_one_validity_schema_when_only_some_ranks_padded(tmp_path):
    """Validity is optional and rank-local, and the pool has to agree on it.

    Process zero's prompts and token rows fill their width, so its host
    producer omits the field; process one's are padded and carry it. Both
    assembly seams have to hand every rank the same tree: generation agrees
    the schema before it digests the request, and placement materializes the
    field before it assembles the batch. The run then has to be the run with
    the mask spelled out on every row: the same draws, and the same gradient.

    The step is plain unclipped SGD, so each parameter's update is the rate
    times its gradient and comparing updates compares gradient magnitudes.
    Adam's first step would divide them away and pass on a wrong gradient.
    Observed on CPU: the largest update is 1.85e-02, the pool sits 1.6e-06 of
    that from the explicit-mask reference (parameters 3.0e-08 apart), and the
    same batch with no validity anywhere, which counts the padded targets,
    sits 3.4e-01 away. That last number is why the comparison means
    something, and the loss shows why it has to be the gradient: masked
    4.419333 against unmasked 4.421612, three digits apart.

    Without the agreement this fails in generation with "generation input
    shapes and sampling must agree across processes", which is also how
    test_a_pool_samples_rollouts_with_different_lengths_and_eos fails, since
    `PolicyCollector` omits validity for a rank whose prompts are whole. With
    generation agreed but placement left alone, a rank dies inside the step
    and the coordination service takes the pool down with it.
    """
    reports = run_pool("mixed_validity", tmp_path / "pool", 2, fsdp_size=2, timeout=180)
    single = run_worker("mixed_validity", tmp_path / "single.json", fsdp_size=2,
                        explicit=True)

    # The producers really did disagree, and the seams really did agree.
    assert reports[0]["local_fields"] == [] and reports[1]["local_fields"] == ["attention_mask"]
    assert [report["agreed_fields"] for report in reports] == [["attention_mask"]] * 2
    assert [report["placed_fields"] for report in reports] == [["attention_mask"]] * 2
    assert reports[0]["agreed_all_true"] is True and reports[1]["agreed_all_true"] is False
    assert reports[0]["batch_sharding"]["fully_addressable"] == [False]

    # The pool is the explicit-mask reference: the same draws and the same step.
    assert reports[0]["tokens"] + reports[1]["tokens"] == single["tokens"]
    np.testing.assert_allclose(reports[0]["raw_log_probs"] + reports[1]["raw_log_probs"],
                               single["raw_log_probs"], rtol=1e-5, atol=1e-6)
    assert reports[0]["loss"] == reports[1]["loss"], "the processes disagreed with each other"
    assert single["loss"] == pytest.approx(reports[0]["loss"], rel=PARITY["rtol"], abs=PARITY["atol"])
    assert_same_parameters(dumped_params(tmp_path / "pool" / "process0.json"),
                           dumped_params(tmp_path / "single.json"))

    # The gradient itself, normalized by its own scale.
    pooled = dumped_params(tmp_path / "pool" / "process0.json", ".update.npz")
    reference = dumped_params(tmp_path / "single.json", ".update.npz")
    scale = max(float(np.max(np.abs(leaf))) for leaf in reference.values())
    assert scale == pytest.approx(single["update_scale"]) and scale > 1e-3
    assert relative_difference(pooled, reference) < 1e-4

    # A step that ignored validity is a different gradient, so the comparison
    # above is sensitive to the term the field controls.
    import jax.numpy as jnp

    from dew.nn.inputs import ModelInputs

    unmasked = worker.mixed_validity_step(
        worker.mixed_validity_trainer(2),
        {"text": ModelInputs(jnp.asarray(worker.token_batch()))})["update"]
    assert relative_difference(unmasked, reference) > 1e-2


@pytest.mark.distributed
def test_a_pool_trains_a_hybrid_split_over_the_sequence(tmp_path):
    """Two processes of one device each train a hybrid decoder's Mamba-2 and attention layers split
    over the sequence, each half of every sequence on another process: the pool fills validity on
    both, so the Mamba-2 layers take their padded-row path under the sequence axis, its halo and state
    crossing the processes (tests/test_mamba2.py holds that path's output and gradients to the unsplit
    layer's by check 1), and the processes report one finite loss and a finite, real update."""
    worker_script = Path(__file__).with_name("sequence_worker.py")
    reports = run_pool("sequence_hybrid", tmp_path / "pool", 2, script=worker_script, devices=1, fsdp_size=1,
                       timeout=180)
    assert [report["sequence_processes"] for report in reports] == [[[0, 1]]] * 2
    assert [report["placed_fields"] for report in reports] == [["attention_mask"]] * 2
    assert reports[0]["loss"] == reports[1]["loss"], "the processes disagreed with each other"
    assert np.isfinite(reports[0]["loss"]) and np.isfinite(reports[0]["update_scale"])
    assert reports[0]["update_scale"] > 0


@pytest.mark.distributed
def test_a_pool_decides_its_frozen_fp8_sites_on_process_zero(tmp_path):
    """The frozen FP8 sites decide which kernels a rank's train state holds
    packed, so process 0 decides them for the pool: on CPUs it refuses for
    want of FP8 tensor cores and its peer hears of it at the agreement,
    deciding nothing itself. A tree that lacks a site on process 1 alone is
    refused there and heard of on process 0, and both pass the next
    agreement."""
    reports = run_pool("fp8_sites", tmp_path / "pool", 2, timeout=180)
    unsupported = [report["unsupported"] for report in reports]
    assert unsupported[0].startswith("ValueError") and "no FP8 tensor cores" in unsupported[0]
    assert unsupported[1].startswith("PeerFailure")
    missing = [report["missing"] for report in reports]
    assert missing[0].startswith("PeerFailure")
    assert missing[1].startswith("ValueError") and "frozen FP8 site narrow" in missing[1]
    assert [report["after"] for report in reports] == [True, True]


@pytest.mark.distributed
def test_the_front_door_answers_the_same_rows_on_a_pool(tmp_path):
    """Identical user code on one device and on a two-process pool: the run's
    weights land on the mesh under the trainer's layout, each rank hands in
    its own prompts, results stay row-sharded, and `host()` gives every rank
    the rows a single process draws for the same prompts."""
    from test_inference import make_lm_run, make_run

    make_run(tmp_path / "diffusion", encoder="char_table", checkpoint="char_table")
    make_lm_run(tmp_path / "lm")
    runs = {"run_dir": str(tmp_path / "diffusion"), "lm_dir": str(tmp_path / "lm")}
    reports = run_pool("inference_pipeline", tmp_path, 2, devices=2, fsdp_size=2, timeout=240, **runs)
    single = run_worker("inference_pipeline", tmp_path / "single.json", fsdp_size=1, devices=1, **runs)

    for report in reports:
        assert report["process_count"] == 2
        assert report["image_spec"] == report["token_spec"] == "P(('data', 'expert', 'fsdp', 'tensor'),)"
        assert report["image_rows"] == report["rows"] == 3
        assert set(report["rejected"]) == {"prompts", "guidance", "steps", "row_count",
                                           "prepared", "prepared_rows", "request_kind", "budget"}
        assert any("fsdp" in spec for spec in report["parameter_specs"])
    assert single["image_rows"] == single["rows"] == 6
    np.testing.assert_allclose(
        reports[0]["images"] + reports[1]["images"], single["images"], atol=2e-5, rtol=2e-5
    )
    assert reports[0]["tokens"] + reports[1]["tokens"] == single["tokens"]
    assert reports[0]["text"] + reports[1]["text"] == single["text"]


@pytest.mark.distributed
def test_a_pool_draws_every_prompts_continuations_on_the_process_that_asked(tmp_path):
    """Three continuations of each prompt over two processes against one
    process drawing the same global batch. Each rank reads back its own
    prompts' continuations in prompt order, the padded prompt row it placed
    for the mesh contributes none of them, and a rank asking for a different
    number of continuations is refused on both ranks rather than issuing a
    decode collective the peer never enters.
    """
    reports = run_pool("continuations", tmp_path, 2, devices=2, fsdp_size=2, timeout=240)
    single = run_worker("continuations", tmp_path / "single.json", fsdp_size=1, devices=1)

    for report in reports:
        assert report["process_count"] == 2
        assert report["spec"] == "P(('data', 'expert', 'fsdp', 'tensor'),)"
        # Four placed prompt rows per process, three of them real, times three
        # continuations: the padded prompt's group stays out of the result.
        assert report["global_rows"] == 24 and report["rows"] == 9
        assert len(report["tokens"]) == 9
        assert len(report["rejected"]) == 1 and "agree across processes" in report["rejected"][0]
    assert single["rows"] == 18 and single["global_rows"] == 18
    assert reports[0]["tokens"] + reports[1]["tokens"] == single["tokens"]
    assert reports[0]["lengths"] + reports[1]["lengths"] == single["lengths"]
    assert reports[0]["terminated"] + reports[1]["terminated"] == single["terminated"]
    np.testing.assert_allclose(reports[0]["behavior"] + reports[1]["behavior"],
                               single["behavior"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(reports[0]["raw"] + reports[1]["raw"], single["raw"], rtol=1e-5, atol=1e-6)
    for report in reports:
        assert report["canvas_rejected"] and report["canvas_rows"] == 4
    assert [report["canvas_alone"] for report in reports] == [1, 2]
    for name in ("canvas_tokens", "canvas_lengths", "canvas_steps"):
        assert reports[0][name] + reports[1][name] == single[name]


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
        # Process 0 holds one batch and process 1 two: process 0 covers the
        # second with its first's rows marked repeats, so the pass reads all 12.
        assert report["results"]["uneven"]["scores"]["evaluation/coordinated_batches"] == 2
        for phase in ("metric", "preview", "finalize", "log", "render", "construct", "next"):
            assert "error" in report["results"][phase], (phase, report)
            if phase != "construct":
                assert phase in report["closed"]
        assert "iterator next failed" in report["results"]["next"]["error"]
        assert report["results"]["empty"] == {"scores": {}, "events": []}
        assert report["results"]["no_consumer"] == {"scores": {}, "events": []}
        assert "no_consumer" not in report["closed"]
        assert report["results"]["normal"]["scores"]["evaluation/records"] == 16
        assert report["results"]["uneven"]["scores"]["evaluation/records"] == 12
        for invalid in ("mismatch", "duplicates"):
            assert "error" in report["results"][invalid]
            assert report["results"][invalid]["events"] == []
        for failure in ("deleted_first", "deleted_later", "deleted_batch", "deleted_preview"):
            assert "deleted" in report["results"][failure]["error"]
            assert failure in report["closed"]
        assert "gather plan differs between the processes" in report["results"]["mismatched_plan"]["error"]


@pytest.mark.distributed
@pytest.mark.parametrize("axis", ["stage", "sequence"])
def test_evaluation_counts_rows_once_across_replicated_process_axes(tmp_path, axis):
    reports = run_pool("evaluation_replicas", tmp_path, 2, devices=1, name=axis)
    assert reports[0]["measured"] == reports[1]["measured"]
    for report in reports:
        assert report["measured"]["val/count"] == 3
        assert report["measured"]["evaluation/records"] == 3
        assert report["no_consumer"] == {}


@pytest.mark.distributed
def test_ranks_whose_clocks_differ_evaluate_at_roots(tmp_path):
    """Rank 1's step and schedule step run ahead of rank 0's. Both report
    root's step and event identity, and a diffusion validation loss, which
    draws its noise from the event key, scores as one process at root's
    clocks scores it, to the pool's reduction order."""
    reports = run_pool("evaluation_clocks", tmp_path, 2, devices=1)
    alone = run_pool("evaluation_clocks", tmp_path / "alone", 1, devices=2)[0]
    assert reports[0] == reports[1], reports
    assert (reports[0]["step"], reports[0]["event"]) == (alone["step"], alone["event"]) == (5, alone["event"])
    assert reports[0]["scores"] == pytest.approx(alone["scores"], rel=1e-5), (reports, alone)


@pytest.mark.distributed
def test_composite_rejection_and_partial_restart_agree_across_processes(tmp_path):
    pool = run_pool("training_contract", tmp_path / "pool", 2, devices=1,
                    run_dir=str(tmp_path / "pool-state"))
    single = run_worker("training_contract", tmp_path / "single.json", devices=2,
                        run_dir=str(tmp_path / "single-state"))
    for report in [*pool, single]:
        assert report["flags"] == [[True, True], [True, False], [True, True], [True, True]]
        # The rejected second attempt drops its window, the first with it: the last two make the update.
        assert report["clocks"] == [4, 2, 1]
        assert report["resumed_equal"]
        assert report["scale"] == 65536.
    for report in pool:
        for want, got in zip(single["state"], report["state"], strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)


@pytest.mark.distributed
def test_a_pool_serves_a_prefix_run_as_one_process_and_refuses_a_short_prompt_together(tmp_path):
    """A prefix run's task over a request two processes each hold a share of, as one global array, answers
    the rows one process answers: each prompt's blocks are made of the process's own rows inside the
    request's agreed check. A prompt too short for the prefix in one rank's share alone is refused on both
    ranks, neither left waiting on its peer's agreement."""
    worker_script = Path(__file__).with_name("prefix_worker.py")
    reports = run_pool("prefix_generation", tmp_path / "pool", 2, script=worker_script, devices=2,
                       fsdp_size=2, timeout=240)
    single = run_worker("prefix_generation", tmp_path / "single.json", script=worker_script, fsdp_size=1,
                        devices=1)

    assert [report["refused"] for report in reports] == [True, True] and single["refused"]
    assert reports[0]["tokens"] + reports[1]["tokens"] == single["tokens"]
    assert reports[0]["lengths"] + reports[1]["lengths"] == single["lengths"]


@pytest.mark.distributed
def test_a_pool_agrees_on_its_decoding_components_or_refuses_the_request(tmp_path):
    """Two processes resolving the same chain answer the same rows as one, and
    a rank whose components differ only in their values is refused on both
    ranks rather than issuing a decode collective its peer never enters.

    The disagreements keep every shape and dtype identical: one rank bans a
    different token, one names a different function, one suppresses a
    different token. A pool comparing only the schema would accept all three
    and run two different policies without saying so.
    """
    reports = run_pool("decoding_components", tmp_path, 2, devices=2, fsdp_size=2, timeout=240)
    single = run_worker("decoding_components", tmp_path / "single.json", fsdp_size=1, devices=1)

    for rank, report in enumerate(reports):
        assert report["rows"] == 3
        assert report["refused"] == ["criterion payload", "transform identity", "transform payload"]
        # A chain both ranks agreed on, undefined on rank 1's rows alone,
        # is refused on both; weights on no mesh answer each rank's own
        # request of its own size as one process answers it.
        assert "without a distribution" in report["undefined"], report["undefined"]
        assert report["independent"] == {str(rank): single["independent"][str(rank)]}
    assert reports[0]["tokens"] + reports[1]["tokens"] == single["tokens"]
    assert reports[0]["lengths"] + reports[1]["lengths"] == single["lengths"]
    np.testing.assert_allclose(reports[0]["behavior"] + reports[1]["behavior"],
                               single["behavior"], rtol=1e-5, atol=1e-6)
    # A search and a speculative block both run ragged rows over the pool: a
    # criterion ends some rows early while their peers keep going, and each
    # rank still answers exactly what one process answers for its own rows.
    for name in ("beam_tokens", "beam_lengths", "draft_tokens", "draft_lengths"):
        assert reports[0][name] + reports[1][name] == single[name], name
    assert len(set(single["draft_lengths"])) > 1, "the rows did not end raggedly"
    assert len(single["beam_tokens"]) == 2 * len(single["draft_tokens"])


@pytest.mark.distributed
def test_masked_task_preserves_pool_rows_and_coordinates_invalid_requests(tmp_path):
    deadline = 180
    reports = run_pool("masked_generation", tmp_path / "pool", 2,
                       devices=2, fsdp_size=2, timeout=deadline)
    out = tmp_path / "single.json"
    single = report_of(spawn("masked_generation", out, devices=1, fsdp_size=1), out, timeout=deadline)
    assert reports[0]["valid_lengths"] == [4, 4, 4]
    assert reports[1]["valid_lengths"] == [2, 3, 1]
    for report in reports:
        assert report["rows"] == 6
        assert report["refused"] == ["token_type_ids", "steps"]
    assert [report["alone_width"] for report in reports] == [3 + 4, 4 + 4]
    for name in ("tokens", "lengths", "terminated", "decoder_steps"):
        assert reports[0][name] + reports[1][name] == single[name], name


@pytest.mark.distributed
def test_a_profile_window_that_fails_on_one_rank_fails_the_pool_together(tmp_path):
    """A capture that cannot start on one rank must not leave its peer's open.

    Rank 0's scheduled directory is a file, so its start fails while rank 1's
    already began. Both have to report the profiling-start failure and reach
    the barrier after it rather than hang in a collective. The capture's
    export reads XProf, the optional profile extra, as installed."""
    pytest.importorskip("xprof.convert.raw_to_tool_data", reason="optional profile extra")
    reports = run_pool("profile_failure", tmp_path / "pool", 2,
                       devices=2, timeout=120, steps=4,
                       profile_dir=str(tmp_path / "pool" / "trace"))
    assert "profiling window start" in reports[0]["failed"]
    assert "NotADirectoryError" in reports[0]["failed"]
    assert "profiling window start" in reports[1]["failed"]
    assert all(report["recovered"] == 2 for report in reports)

@pytest.mark.distributed
def test_a_pool_refuses_what_its_processes_do_not_share_before_any_trains(tmp_path):
    """Each process opens its own stream over its own copy of the corpus, reads
    its batches on its own prefetch thread and lists the checkpoints itself.
    Where one process's corpus holds other records of the same length, under
    its own description or one name both copies share, or another count of
    them, where its first read fails, where its
    stream (one Dew does not know endless) runs out a step before its peers',
    where its batch lacks a field, or where its listing has not yet seen the
    newest checkpoint a resume reads, every process refuses with the cause,
    rather than one raising alone while its peers enter a step or a restore
    it never joins. Streams that run out at the same step on every process
    end the run there."""
    reports = run_pool("pool_refusals", tmp_path / "pool", 2, devices=4, timeout=300,
                       run_dir=str(tmp_path / "runs"))
    for rank, report in enumerate(reports):
        assert report["same"] is None and report["saved"] is None and report["latest"] == 2, report
        assert "training data differs between the processes" in report["content"], report
        assert "training data differs between the processes" in report["named"], report
        # A pool compares what its processes hold, not where each mounted it: no description is named.
        assert "training data differs between the processes" in report["count"], report
        assert "holds no rows" in report["first_read"], report
        assert "training read: the stream ended on 1 of 2 processes" in report["eof"], report
        assert report["ended"] is None, report
        assert "training batch layout differs" in report["layout"], report
        assert "newest checkpoint differs" in report["stale"], report
        if rank == 0:
            assert "training read failed on rank 1" in report["first_read"], report


@pytest.mark.distributed
def test_runs_average_across_a_pool_each_process_its_own_shards(tmp_path):
    """Two runs a pool trained, averaged in the pool onto its mesh with their leaves split over its
    fsdp axis, which no one process holds whole: every process's gathered mean is float64's mean of
    the two runs' weights rounded to fp32, read from their checkpoints in one process."""
    import jax

    from dew.checkpoints import Checkpoints

    corpus = document_corpus(tmp_path / "corpus", 24, length=7)
    # The fsdp axis spans both processes' devices, so a split leaf is half on each.
    reports = run_pool("averaged", tmp_path / "out", 2, script=AVERAGING, run_dir=tmp_path / "runs",
                       fsdp_size=MESH_DEVICES, tokens=corpus.train, seq_len=7)
    assert all(report["split"] for report in reports)
    trees = []
    for run in reports[0]["runs"]:
        with Checkpoints(run) as checkpoints:
            restored, _ = checkpoints.restore()
        # The live weights, as the pool averages them (ema=False).
        trees.append(restored["variables"])
    flat = [{jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf
             in jax.tree_util.tree_leaves_with_path(tree)} for tree in trees]
    for index in range(2):
        dumped = dumped_params(tmp_path / "out" / f"process{index}.json")
        assert set(dumped) == set(flat[0])
        for name, got in dumped.items():
            wide = (flat[0][name].astype(np.float64) + flat[1][name].astype(np.float64)) / 2
            np.testing.assert_array_equal(got, wide.astype(np.float32), err_msg=name)


@pytest.mark.distributed
def test_runs_one_process_refuses_are_refused_by_every_process_of_the_pool(tmp_path):
    """Two runs whose integer counts differ in an entry the first process alone holds, one with its
    token table frozen and kept whole, the other training and splitting it: that process refuses them
    (ValueError), and its peer, whose placement of the table gathers it, hears the refusal where the
    pool agrees on each run (PeerFailure), rather than waiting in the gather alone or averaging."""
    corpus = document_corpus(tmp_path / "corpus", 24, length=7)
    reports = run_pool("averaged_refused", tmp_path / "out", 2, script=AVERAGING, run_dir=tmp_path / "runs",
                       fsdp_size=MESH_DEVICES, tokens=corpus.train, seq_len=7)
    assert all(report["split"] for report in reports)
    assert [report["refused"] for report in reports] == ["ValueError", "PeerFailure"]
