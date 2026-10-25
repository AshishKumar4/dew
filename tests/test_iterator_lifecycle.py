"""Owned iterator shutdown without changing which batches train or resume."""

import contextlib
import gc
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import BATCH, TimingRegression as Regression, batches
from process_support import assert_exits_cleanly
from recording import RecordingTracker

from dew.artifacts import Representations
from dew.data import Loading, Rejected, from_grain, mix
from dew.data.compile import compile
from dew.data.dataset import DataPartition, Loader, mapped
from dew.data.readers import _Host
from dew.telemetry.records import FitEnded
from dew.training import Checkpoints, MeshSpec, Trainer
from dew.training.distributed import DevicePrefetchIterator, batch_signature, shard_batch, step_reads


class Source:
    """A position-bearing stream with observable finalization and read ownership."""

    def __init__(self, end=None, failure=None, close_failure=None):
        self.position = 0
        self.end = end
        self.failure = failure
        self.close_failure = close_failure
        self.closed = threading.Event()
        self.ahead = threading.Event()
        self.owners = []

    def __iter__(self):
        return self

    def __next__(self):
        self.owners.append(threading.get_ident())
        if self.position == self.end:
            if self.failure is not None:
                raise self.failure
            raise StopIteration
        self.position += 1
        if self.position >= 3:
            self.ahead.set()
        return {"x": np.full((jax.device_count(), 3), self.position, np.float32),
                "y": np.full((jax.device_count(), 2), self.position * 2, np.float32)}

    def get_state(self):
        self.owners.append(threading.get_ident())
        return {"position": self.position}

    def set_state(self, state):
        self.owners.append(threading.get_ident())
        self.position = state["position"]

    def close(self):
        self.owners.append(threading.get_ident())
        assert not self.closed.is_set(), "source was finalized twice"
        self.closed.set()
        if self.close_failure is not None:
            raise self.close_failure


class ObservedRows(grain.IterDataset):
    """Grain records with observable opening, checkpoint and read ownership."""

    def __init__(self, end=32, *, block_open=False, block_read_at=None):
        super().__init__()
        self.end, self.block_open, self.block_read_at = end, block_open, block_read_at
        self.entered, self.release = threading.Event(), threading.Event()
        self.opened, self.reads, self.states, self.closed = [], [], [], []

    def __iter__(self):
        return ObservedRead(self)

    def observe(self, parent):
        return ObservedPipeline(parent, self)


class ObservedRead(grain.DatasetIterator):
    def __init__(self, source):
        super().__init__()
        self.source, self.position = source, 0

    def __next__(self):
        if self.position == self.source.block_read_at:
            self.source.entered.set()
            self.source.release.wait()
        if self.position == self.source.end:
            raise StopIteration
        row = {"i": self.position}
        self.position += 1
        return row

    def get_state(self):
        return {"position": self.position}

    def set_state(self, state):
        self.position = state["position"]


class ObservedPipeline(grain.IterDataset):
    """Observe the Grain reader opened for a whole compiled chain."""

    def __init__(self, parent, source):
        super().__init__(parent)
        self.source = source

    def __iter__(self):
        self.source.opened.append(threading.get_ident())
        if self.source.block_open and len(self.source.opened) == 1:
            self.source.entered.set()
            self.source.release.wait()
        return ObservedPipelineRead(iter(self._parent), self.source)


class ObservedPipelineRead(grain.DatasetIterator):
    def __init__(self, parent, source):
        super().__init__(parent)
        self.source = source

    def __next__(self):
        self.source.reads.append(threading.get_ident())
        return next(self._parent)

    def get_state(self):
        self.source.states.append(threading.get_ident())
        return self._parent.get_state()

    def set_state(self, state):
        self._parent.set_state(state)

    def close(self):
        if not self._closed:
            self.source.closed.append(threading.get_ident())
        super().close()


READ = Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1)


@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("operation", ["request_stop", "prefetch_close"])
def test_unused_chain_cancellation_does_not_open_its_grain_reader(host, operation):
    source = ObservedRows()
    chain = (from_grain(source, "unused").repeat(1).batch(jax.device_count())
             .grain(source.observe, identity="observed-reader"))
    data = compile(chain.map(dict) if host else chain, read="streamed", loading=READ)
    reader = data.train(DataPartition())
    try:
        assert source.opened == []
        if operation == "request_stop":
            reader.request_stop()
        else:
            with DevicePrefetchIterator(reader, MeshSpec().build()):
                pass
        assert source.opened == source.reads == source.states == source.closed == []
    finally:
        reader.close()


def test_chain_cancellation_during_opening_keeps_one_reader_on_the_worker():
    source = ObservedRows(block_open=True)
    chain = (from_grain(source, "opening").repeat(1).batch(jax.device_count())
             .grain(source.observe, identity="observed-reader"))
    data = compile(chain, read="streamed", loading=READ)
    stream = DevicePrefetchIterator(data.train(DataPartition()), MeshSpec().build())
    consumer = threading.Thread(target=lambda: list(stream))
    consumer.start()
    try:
        assert source.entered.wait(5)
        with pytest.raises(TimeoutError):
            stream.close(timeout=0.01)
        assert len(source.opened) == 1
        assert source.opened[0] != threading.get_ident()
    finally:
        source.release.set()
        stream.close()
        consumer.join(5)
    assert not consumer.is_alive()
    assert source.closed == source.opened


def reject_thirds(row):
    return Rejected("third") if row["i"] % 3 == 0 else row


@pytest.mark.parametrize("host", [False, True])
def test_chain_statistics_publish_without_reading_the_workers_live_iterator(host):
    batch = jax.device_count()
    kept = [index for index in range(2 * batch) if index % 3]
    attempted = kept[batch - 1] + 1
    source = ObservedRows(end=3 * batch + 1, block_read_at=attempted)
    chain = (from_grain(source, "observed").repeat(1).filter_map(reject_thirds).batch(batch)
             .grain(source.observe, identity="observed-reader"))
    data = compile(chain.map(dict) if host else chain, read="streamed", loading=READ)
    reader = data.train(DataPartition())
    with DevicePrefetchIterator(reader, MeshSpec().build(), depth=1) as stream:
        try:
            np.testing.assert_array_equal(np.asarray(next(stream)["i"]), kept[:batch])
            assert source.entered.wait(5)
            states = tuple(source.states)
            logged = {
                "data/rows/observed": batch,
                "data/rejected/observed/third": attempted - batch,
                "data/truncated/observed": 0.0,
                "data/chunks_per_record/observed": 0.0,
            }
            for _ in range(3):
                assert data.statistics() == logged
            assert tuple(source.states) == states
            published = data.statistics()
        finally:
            source.release.set()
        remaining = [np.asarray(value["i"]).tolist() for value in stream]
    assert published == logged
    assert set(source.states) == set(source.reads) == set(source.opened)
    assert source.opened[0] != threading.get_ident()
    assert np.concatenate([kept[:batch], *remaining]).tolist() == [
        index for index in range(source.end) if index % 3
    ]
    reference = weakref.ref(reader)
    del reader
    gc.collect()
    assert reference() is None
    final = data.statistics()
    assert final["data/rows/observed"] == sum(index % 3 != 0 for index in range(source.end))
    assert final["data/rejected/observed/third"] == sum(index % 3 == 0 for index in range(source.end))


def test_chain_statistics_count_the_records_every_worker_read():
    """Two worker processes each read part of a pass; the published counts are the pass's, every
    worker's records summed, where the first worker's once stood for all of them. A worker's
    counts are those its last row carried, so a rejection after it is not yet among them."""
    from dew.data import Memory
    from dew.data.pipeline import Pipeline

    rows = [{"i": np.int32(index)} for index in range(24)]
    chain = Pipeline(Memory(rows, shard_rows=4), name="rows").repeat(1).filter_map(reject_thirds).batch(2)
    data = compile(chain, read="streamed", loading=Loading(workers=2, threads=1, read_buffer=1,
                                                           worker_buffer=1))
    reader = data.train(DataPartition())
    try:
        read = [int(value) for batch in reader for value in np.asarray(batch["i"])]
    finally:
        reader.close()
    assert sorted(read) == [index for index in range(24) if index % 3]
    published = data.statistics()
    assert published["data/rows/rows"] == len(read)
    assert published["data/rows/rows"] + published["data/rejected/rows/third"] > 16


def test_corpus_counts_hold_through_grains_restore_into_a_smaller_cycle():
    """Grain's elastic restore puts the positions of shards past a smaller cycle aside as future
    states; the counts they carry are still the corpus's, so the published counts do not change."""
    import grain.experimental as gx

    from dew.data.mixing import Accepted, state_counts

    def shard(number: int):
        rows = [{"i": np.int32(10 * number + index)} for index in range(6)]
        return Accepted(grain.MapDataset.source(rows).to_iter_dataset().map(reject_thirds), "rows", "rows")

    def interleave(cycle: int):
        return gx.InterleaveIterDataset([shard(number) for number in range(4)], cycle_length=cycle).__iter__()

    read = interleave(4)
    for _ in range(10):
        next(read)
    counts = {name: counted.metrics(name) for name, counted in state_counts(read.get_state()).items()}
    restored = interleave(2)
    restored.set_shard_states(read.get_shard_states())
    restored_counts = state_counts(restored.get_state())
    assert {name: counted.metrics(name) for name, counted in restored_counts.items()} == counts
    assert counts["rows"]["data/rows/rows"] == 10


def test_chain_statistics_follow_each_corpus_through_direct_reads_and_resume():
    sources = [ObservedRows(end=64) for _ in range(2)]
    chain = mix([from_grain(source, name).filter_map(reject_thirds)
                 for source, name in zip(sources, ("a", "b"), strict=True)]).batch(4)
    data = compile(chain, read="streamed", loading=READ)
    first = data.train(DataPartition())
    try:
        next(first)
        before = data.statistics()
        saved = first.get_state()
        expected = next(first)["i"]
        after = data.statistics()
        def read(counts, name):
            return counts[f"data/rows/{name}"] + counts.get(f"data/rejected/{name}/third", 0)

        assert all(read(after, name) > read(before, name) for name in ("a", "b"))
    finally:
        first.close()
    resumed = compile(chain, read="streamed", loading=READ)
    reader = resumed.train(DataPartition())
    try:
        reader.set_state(saved)
        np.testing.assert_array_equal(next(reader)["i"], expected)
        assert resumed.statistics() == after
    finally:
        reader.close()


def test_a_field_no_compiled_step_reads_stays_on_the_host():
    """Once a step of a batch's signature compiles, the prefetcher places
    only the leaves its executable kept: a field the step never reads stays a
    host array of its own, and the step runs on the batch as placed. A batch
    of a signature no step has compiled for is placed whole. A field held as
    None, as an optional part of `Coordinates` is, is no leaf to place."""
    mesh = MeshSpec().build()
    rows = jax.device_count()
    batch = {"x": np.ones((rows, 3), np.float32), "unused": np.arange(rows), "absent": None}
    # A step's arguments as the trainer's: the moving state, the frozen tree, then the batch.
    step = jax.jit(lambda state, frozen, batch, part: state + frozen + jnp.sum(batch["x"]) + part)
    first = shard_batch(mesh, batch)
    compiled = step.lower(jnp.zeros(()), jnp.zeros(()), first, jnp.int32(0)).compile()
    # The trainer publishes the mask under the placed batch's signature.
    signature = batch_signature(first)
    reads = {signature: step_reads(compiled, signature)}
    wider = {**batch, "x": np.ones((rows, 4), np.float32)}
    with DevicePrefetchIterator(iter([batch, wider]), mesh, reads=reads) as stream:
        placed, whole = next(stream), next(stream)
    assert isinstance(placed["x"], jax.Array) and isinstance(placed["unused"], np.ndarray)
    assert float(compiled(jnp.zeros(()), jnp.zeros(()), placed, jnp.int32(0))) == 3 * rows
    assert isinstance(whole["unused"], jax.Array)


def test_full_queue_close_preserves_consumed_position_and_releases_source():
    source = Source()
    ref = weakref.ref(source)
    with DevicePrefetchIterator(source, MeshSpec().build(), depth=1) as stream:
        np.testing.assert_array_equal(np.asarray(next(stream)["x"]), 1)
        assert source.ahead.wait(5)
        stream.close()
        assert source.closed.is_set()
        assert json.loads(stream.source_state) == {"position": 1}
        assert len(set(source.owners)) == 1
        assert source.owners[0] != threading.get_ident()
        with pytest.raises(StopIteration):
            next(stream)
    del source
    gc.collect()
    assert ref() is None


def test_a_stop_request_racing_the_workers_close_reaches_the_source_it_read():
    """close asks a stoppable source to stop while the worker, the source just drained, closes it
    and lets it go: the request reaches the source close read, and close returns without error.
    The source holds the race open: when close checks it for request_stop, the source lets its
    worker drain and finish, and waits a moment for it."""
    events, drain = [], threading.Event()
    stream = None

    class Draining:
        @property
        def __class__(self):
            # The main thread's checks after the stop was asked for: let the worker finish meanwhile.
            if threading.current_thread() is threading.main_thread() and stream._stop.is_set():
                drain.set()
                for _ in range(100):
                    if stream._iterator is None:
                        break
                    threading.Event().wait(0.01)
            return Draining

        def __iter__(self):
            return self

        def __next__(self):
            drain.wait(10)
            raise StopIteration

        def request_stop(self):
            events.append("stop")

        def close(self):
            events.append("close")

    stream = DevicePrefetchIterator(Draining(), MeshSpec().build(), depth=1)
    consumer = threading.Thread(target=lambda: next(stream, None))
    consumer.start()
    stream.close()
    consumer.join(10)
    assert sorted(events) == ["close", "stop"]


@pytest.mark.parametrize("failure", [None, ValueError("source failed")])
def test_terminal_outcome_finalizes_with_a_full_data_queue(failure):
    source = Source(end=3, failure=failure)
    with DevicePrefetchIterator(source, MeshSpec().build(), depth=2) as stream:
        assert int(np.asarray(next(stream)["x"])[0, 0]) == 1
        # Finalization must not depend on a consumer making room for EOF/error.
        assert source.closed.wait(5)
        assert [int(np.asarray(next(stream)["x"])[0, 0]) for _ in range(2)] == [2, 3]
        if failure is not None:
            with pytest.raises(ValueError) as raised:
                next(stream)
            assert raised.value is failure
        with pytest.raises(StopIteration):
            next(stream)


def test_early_close_discards_only_unconsumed_failure():
    source = Source(end=1, failure=ValueError("speculative read"))
    with DevicePrefetchIterator(source, MeshSpec().build(), depth=2) as stream:
        next(stream)
        assert source.closed.wait(5)
    assert json.loads(stream.source_state) == {"position": 1}


def test_restoration_and_position_capture_share_the_iteration_thread():
    source = Source(end=5)
    with DevicePrefetchIterator(source, MeshSpec().build(), source_state=b'{"position": 3}') as stream:
        assert int(np.asarray(next(stream)["x"])[0, 0]) == 4
        stream.close()
        assert json.loads(stream.source_state) == {"position": 4}
    assert len(set(source.owners)) == 1
    assert source.owners[0] != threading.get_ident()


def test_timeout_does_not_close_a_running_generator_and_can_be_rejoined():
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    def blocked():
        try:
            entered.set()
            release.wait()
            yield {"x": np.ones((jax.device_count(), 2), np.float32)}
        finally:
            closed.set()

    stream = DevicePrefetchIterator(blocked(), MeshSpec().build())
    consumer = threading.Thread(target=lambda: list(stream))
    consumer.start()
    try:
        assert entered.wait(5)
        with pytest.raises(TimeoutError):
            stream.close(timeout=0.01)
        assert not closed.is_set()
        with pytest.raises(StopIteration):
            next(stream)
    finally:
        release.set()
        stream.close()
        consumer.join(5)
    assert not consumer.is_alive()
    assert closed.is_set()


def test_tokenized_stop_interrupts_next_then_finalizes_on_its_owner():
    entered, stopped, closed = threading.Event(), threading.Event(), threading.Event()
    owners = []

    class Blocking:
        def __iter__(self):
            return self

        def __next__(self):
            owners.append(threading.get_ident())
            entered.set()
            stopped.wait()
            raise StopIteration

        def request_stop(self):
            stopped.set()

        def close(self):
            owners.append(threading.get_ident())
            closed.set()

    wrapped = mapped(lambda partition: Blocking(), dict)(DataPartition())
    with DevicePrefetchIterator(wrapped, MeshSpec().build()) as stream:
        consumer = threading.Thread(target=lambda: list(stream))
        consumer.start()
        try:
            assert entered.wait(5)
        finally:
            stream.close()
            consumer.join(5)
        assert not consumer.is_alive()
    assert stopped.is_set() and closed.is_set()
    assert len(set(owners)) == 1 and owners[0] != threading.get_ident()


@pytest.mark.parametrize("body_fails", [False, True])
def test_source_close_failure_preserves_primary_and_is_reported_once(body_fails):
    cleanup = OSError("source close failed")
    primary = RuntimeError("body failed")
    source = Source(close_failure=cleanup)
    with (
        pytest.raises((RuntimeError, OSError)) as raised,
        DevicePrefetchIterator(source, MeshSpec().build()) as stream,
    ):
        next(stream)
        if body_fails:
            raise primary
    assert raised.value is (primary if body_fails else cleanup)
    if body_fails:
        assert any("source close failed" in note for note in primary.__notes__)
    stream.close()


def test_eof_reports_finalization_failure_instead_of_clean_exhaustion():
    cleanup = OSError("cannot finalize source")
    source = Source(end=0, close_failure=cleanup)
    with DevicePrefetchIterator(source, MeshSpec().build()) as stream:
        with pytest.raises(OSError) as raised:
            next(stream)
        assert raised.value is cleanup
        with pytest.raises(StopIteration):
            next(stream)



def test_repeated_bounded_fit_releases_each_source_and_reuses_checkpointer(tmp_path):
    opened = []

    def start():
        source = Source()
        opened.append(source)
        return source

    checkpoints = Checkpoints(str(tmp_path))
    trainer = Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0),
                      checkpoints=checkpoints)
    data = Loader(lambda partition: start(), None, None, 8)
    for target in (1, 3):
        state = trainer.fit(data, steps=target, checkpoint_every=1)
        assert int(state.step) == target
        assert all(source.closed.is_set() for source in opened)
    _, position = checkpoints.restore(share=DataPartition())
    assert json.loads(position) == {"position": 3}
    trainer.fit(data, steps=3)
    assert len(opened) == 2


def test_a_training_stream_that_runs_out_closes_and_ends_the_run_at_its_last_batch(tmp_path):
    """A stream of one batch under a run of two steps ends the run at step 1, as a run of one step
    would end: the stream closed, its final checkpoint written there with the position it stopped at,
    and the run's outcome completed."""
    source = Source(end=1)
    checkpoints = Checkpoints(str(tmp_path))
    tracker = RecordingTracker()
    trainer = Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0),
                      checkpoints=checkpoints, tracker=tracker)
    state = trainer.fit(Loader(lambda partition: source, None, None, 8), steps=2)
    assert source.closed.is_set()
    assert int(state.step) == checkpoints.latest == 1
    assert json.loads(checkpoints.restore(share=DataPartition())[1]) == {"position": 1}
    assert [value.status for _, value in tracker.artifacts if isinstance(value, FitEnded)] == ["completed"]


@pytest.mark.parametrize("steps", [0, 2])
def test_metric_failure_closes_owned_iterators_before_it_escapes(steps):
    train, validation = Source(), Source(end=2)

    class Evaluated(Regression):
        def evaluate(self, params, batch, step):
            return Representations(features=batch["x"], labels=jnp.zeros((8,), jnp.int32))

    class InvalidMetric:
        name, reads = "invalid", Representations

        def __call__(self, artifact, batch):
            return np.linalg.inv(artifact.features)  # Not square: a real metric failure.

        def merge(self, accumulated, contribution):
            return accumulated + contribution

        def finalize(self, accumulated):
            return float(accumulated)

    def training():
        if steps == 0:
            raise AssertionError("zero-step fit opened training")
        return train

    trainer = Trainer(Evaluated(), optax.sgd(0.01), key=jax.random.key(0))
    with pytest.raises(np.linalg.LinAlgError):
        trainer.fit(Loader(lambda partition: training(), lambda partition: validation, None, 8),
                    steps=steps, eval_every=1, metrics=(InvalidMetric(),))
    assert train.closed.is_set() == (steps > 0)
    assert validation.closed.is_set()
    assert validation.owners[-1] == threading.get_ident()


def test_a_refusal_before_the_transfer_finalizes_the_untransferred_source(tmp_path):
    """A stream fit refuses before its prefetch worker owns it is closed by fit. Here the stream
    keeps no position, and the run resumes from a checkpoint that holds none, which any mesh
    refuses: reading it again from the start would repeat its data."""
    closed = threading.Event()

    class Uncheckpointable:
        def __iter__(self):
            return self

        def __next__(self):
            return next(batches())

        def close(self):
            closed.set()

    def fit(steps):
        trainer = Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0),
                          checkpoints=Checkpoints(str(tmp_path)))
        trainer.fit(Loader(lambda partition: Uncheckpointable(), None, None, BATCH), steps=steps,
                    checkpoint_every=1)

    fit(1)
    closed.clear()
    with pytest.raises(ValueError):
        fit(2)
    assert closed.is_set()

def test_sigint_during_fit_restoration_closes_only_after_restoration_returns(tmp_path):
    checkpoints = Checkpoints(str(tmp_path))
    trainer = Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0),
                      checkpoints=checkpoints)
    trainer.fit(Loader(lambda partition: Source(), None, None, 8), steps=1)
    entered, stopped = threading.Event(), threading.Event()

    class Restoring(Source):
        restoring = False

        def set_state(self, state):
            self.restoring = True
            entered.set()
            stopped.wait()
            super().set_state(state)
            self.restoring = False

        def request_stop(self):
            stopped.set()

        def close(self):
            assert not self.restoring, "final close raced state restoration"
            super().close()

    source = Restoring()

    def interrupt():
        if entered.wait(10):
            os.kill(os.getpid(), signal.SIGINT)

    sender = threading.Thread(target=interrupt)
    sender.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            trainer.fit(Loader(lambda partition: source, None, None, 8), steps=2)
        assert source.closed.is_set()
        assert len(set(source.owners)) == 1
        assert source.owners[0] != threading.get_ident()
    finally:
        stopped.set()
        sender.join(10)


@pytest.mark.parametrize("host", [False, True])
def test_wrapped_cancellation_reaches_source_during_finalization(host):
    entered, stopped, closed = threading.Event(), threading.Event(), threading.Event()

    class Finalizing:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def request_stop(self):
            stopped.set()

        def close(self):
            entered.set()
            stopped.wait()
            closed.set()

    def opened(partition):
        return Finalizing()

    wrapped = _Host(opened(DataPartition()), (dict,)) if host else mapped(opened, dict)(DataPartition())
    stream = DevicePrefetchIterator(wrapped, MeshSpec().build())
    consumer = threading.Thread(target=lambda: list(stream))
    consumer.start()
    try:
        assert entered.wait(5)
        stream.close(timeout=1)
        assert closed.is_set()
    finally:
        stopped.set()
        stream.close()
        consumer.join(5)
    assert not consumer.is_alive()


def test_failed_thread_creation_finalizes_the_unstarted_source(monkeypatch):
    owners = []
    stopped = threading.Event()

    class Source:
        def __iter__(self):
            return self

        def __next__(self):
            raise AssertionError("an unstarted producer cannot read")

        def request_stop(self):
            stopped.set()

        def close(self):
            assert stopped.is_set()
            owners.append(threading.get_ident())

    failure = RuntimeError("cannot start new thread")

    def fail_start(thread):
        raise failure

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with (
        pytest.raises(RuntimeError) as caught,
        DevicePrefetchIterator(Source(), MeshSpec().build()) as stream,
    ):
        next(stream)
    assert caught.value is failure
    assert owners == [threading.get_ident()]
    stream.close()


def test_a_program_that_ends_without_closing_its_prefetcher_exits_cleanly():
    """The worker places each batch inside jaxlib. A program that ended
    without `close` left it there when Python finalized, and the process
    aborted after its last line (SIGABRT, "terminate called ..."). The
    worker now stops at exit."""
    program = (
        "import numpy as np\n"
        "from dew.training import MeshSpec\n"
        "from dew.training.distributed import DevicePrefetchIterator\n"
        "batch = {'x': np.ones((2048, 4096), np.float32)}\n"
        "def endless():\n"
        "    while True:\n"
        "        yield batch\n"
        "prefetch = DevicePrefetchIterator(endless(), MeshSpec().build(), depth=1)\n"
        "for _ in range(20):\n"
        "    next(prefetch)\n")
    assert_exits_cleanly(program)


def _running(pid: int) -> bool:
    """Whether process `pid` runs: it exists and has not ended (a zombie has). One that ends between
    the open of its state and the read fails the read with ESRCH, and has ended too."""
    try:
        status = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return status.rsplit(")", 1)[1].split()[0] != "Z"


def test_a_process_that_ends_as_its_state_is_read_is_not_running(monkeypatch):
    """A loading worker that ends after its state file opens and before it is read: the kernel fails
    the read with ESRCH, which says the process has ended, as a missing file does."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    state = Path(f"/proc/{child.pid}/stat")
    read = Path.read_text

    def ended_while_read(path, *args, **kwargs):
        if path != state:
            return read(path, *args, **kwargs)
        with path.open() as opened:
            child.kill()
            child.wait()
            return opened.read()

    monkeypatch.setattr(Path, "read_text", ended_while_read)
    try:
        assert not _running(child.pid)
    finally:
        child.kill()
        child.wait()


def _segments_marked(mark: bytes) -> set[str]:
    """The shared memory segments on this host that begin with `mark`: those a run whose every
    batch begins with it handed its batches over in, whatever else shares the host. Python names a
    segment at random (`psm_...`), so its content, not its name, says whose it is."""
    marked = set()
    for name in os.listdir("/dev/shm"):
        if name.startswith("psm_"):
            with contextlib.suppress(OSError), Path("/dev/shm", name).open("rb") as segment:
                if segment.read(len(mark)) == mark:
                    marked.add(name)
    return marked


@pytest.mark.skipif(not (Path("/proc/self/stat").exists() and Path("/dev/shm").is_dir()),
                    reason="reads process states from /proc and shared memory from /dev/shm")
def test_a_killed_program_leaves_no_loading_worker_running():
    """A program killed mid-read, as a timeout or the kernel kills one, takes its loading workers
    with it, and the shared memory its batches were handed over in. Each worker read until its
    parent asked it to stop, so a killed parent left them reading into a queue no one empties,
    holding their memory until the host was rebooted."""
    program = (
        "import multiprocessing, sys, time\n"
        "import numpy as np\n"
        "from dew.data import Loading, Memory\n"
        "from dew.data.compile import compile\n"
        "from dew.data.dataset import DataPartition\n"
        "from dew.data.pipeline import Pipeline\n"
        "rows = Memory(tuple({'x': np.full(4096, int(sys.argv[1]), np.int64)} for _ in range(64)))\n"
        "loading = Loading(workers=2, threads=1, read_buffer=1, worker_buffer=1)\n"
        "stream = compile(Pipeline(rows).repeat().batch(2), read='memory', loading=loading)"
        ".train(DataPartition())\n"
        "next(stream)\n"
        "print(' '.join(str(child.pid) for child in multiprocessing.active_children()), flush=True)\n"
        "time.sleep(600)\n")
    # Every batch is two records of 4096 copies of this run's own token, 64 KiB, large enough to cross
    # in shared memory (`dew.data.dataset.SHARED_BYTES`), so its segments begin with the token.
    token = secrets.randbits(63)
    mark = np.full(8, token, np.int64).tobytes()
    root = Path(__file__).resolve().parents[1]
    parent = subprocess.Popen(
        [sys.executable, "-c", program, str(token)], cwd=root, stdout=subprocess.PIPE, text=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(root / "src")})
    workers: list[int] = []
    try:
        assert parent.stdout is not None
        workers = [int(pid) for pid in parent.stdout.readline().split()]
        assert len(workers) >= 2, "the program started no loading workers"
        assert _segments_marked(mark), "the program's workers handed over no batch in shared memory"
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=30)
        deadline = time.monotonic() + 60
        while (any(map(_running, workers)) or _segments_marked(mark)) and time.monotonic() < deadline:
            time.sleep(0.5)
        assert [pid for pid in workers if _running(pid)] == []
        assert _segments_marked(mark) == set()
    finally:
        parent.kill()
        for pid in workers:
            if _running(pid):
                os.kill(pid, signal.SIGKILL)
        for name in _segments_marked(mark):
            Path("/dev/shm", name).unlink(missing_ok=True)


def test_a_source_with_no_deadline_of_its_own_closes_when_its_finalization_ends():
    """A source that sets no stop deadline (no `stop_seconds`) owns how long it takes to stop: a
    close waits for its finalization, six seconds here, and the worker is gone after it, where a
    five-second wait of the iterator's own once raised and left the worker holding the source."""
    finished = threading.Event()

    def slow():
        try:
            while True:
                yield {"x": np.ones((jax.device_count(), 2), np.float32)}
        finally:
            time.sleep(6)
            finished.set()

    stream = DevicePrefetchIterator(slow(), MeshSpec().build())
    next(stream)
    stream.close()
    assert finished.is_set()
