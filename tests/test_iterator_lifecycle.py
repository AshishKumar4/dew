"""Owned iterator shutdown without changing which batches train or resume."""

import gc
import json
import os
import signal
import threading
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.artifacts import Representations
from dew.data import Dataset
from dew.data.dataset import tokenized
from dew.training import Checkpoints, Profile, Trainer, build_mesh
from dew.training.distributed import DevicePrefetchIterator
from test_instrumentation import Regression, batches


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


def test_full_queue_close_preserves_consumed_position_and_releases_source():
    source = Source()
    ref = weakref.ref(source)
    with DevicePrefetchIterator(source, build_mesh(), depth=1) as stream:
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


@pytest.mark.parametrize("failure", [None, ValueError("source failed")])
def test_terminal_outcome_finalizes_with_a_full_data_queue(failure):
    source = Source(end=3, failure=failure)
    with DevicePrefetchIterator(source, build_mesh(), depth=2) as stream:
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
    with DevicePrefetchIterator(source, build_mesh(), depth=2) as stream:
        next(stream)
        assert source.closed.wait(5)
    assert json.loads(stream.source_state) == {"position": 1}


def test_restoration_and_position_capture_share_the_iteration_thread():
    source = Source(end=5)
    with DevicePrefetchIterator(source, build_mesh(), source_state=b'{"position": 3}') as stream:
        assert int(np.asarray(next(stream)["x"])[0, 0]) == 4
        stream.close()
        assert json.loads(stream.source_state) == {"position": 4}
    assert len(set(source.owners)) == 1
    assert source.owners[0] != threading.get_ident()


@pytest.mark.parametrize("depth", [0, -1])
def test_unbounded_depth_is_rejected_without_taking_ownership(depth):
    source = Source()
    with pytest.raises(ValueError, match="positive"):
        DevicePrefetchIterator(source, build_mesh(), depth=depth)
    assert not source.owners
    source.close()


def test_timeout_does_not_close_a_running_generator_and_can_be_rejoined():
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    def blocked():
        try:
            entered.set()
            release.wait()
            yield {"x": np.ones((jax.device_count(), 2), np.float32)}
        finally:
            closed.set()

    stream = DevicePrefetchIterator(blocked(), build_mesh())
    consumer = threading.Thread(target=lambda: list(stream))
    consumer.start()
    try:
        assert entered.wait(5)
        with pytest.raises(TimeoutError, match="still alive"):
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

    wrapped = tokenized(Blocking, None)()
    with DevicePrefetchIterator(wrapped, build_mesh()) as stream:
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
    with pytest.raises((RuntimeError, OSError)) as raised:
        with DevicePrefetchIterator(source, build_mesh()) as stream:
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
    with DevicePrefetchIterator(source, build_mesh()) as stream:
        with pytest.raises(OSError) as raised:
            next(stream)
        assert raised.value is cleanup
        with pytest.raises(StopIteration):
            next(stream)


def test_fit_zero_steps_opens_no_training_source_but_still_evaluates():
    validation = Source(end=1)

    def unwanted():
        raise AssertionError("zero-step fit opened training")

    trainer = Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0))
    state = trainer.fit(Dataset(unwanted, lambda: validation, None, 8),
                        steps=0, eval_every=1)
    assert int(state.step) == 0 and validation.closed.is_set()


def test_repeated_bounded_fit_releases_each_source_and_reuses_checkpointer(tmp_path):
    opened = []

    def start():
        source = Source()
        opened.append(source)
        return source

    checkpoints = Checkpoints(str(tmp_path))
    trainer = Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0),
                      checkpoints=checkpoints)
    data = Dataset(start, None, None, 8)
    for target in (1, 3):
        state = trainer.fit(data, steps=target, checkpoint_every=1)
        assert int(state.step) == target
        assert all(source.closed.is_set() for source in opened)
    _, position = checkpoints.restore()
    assert json.loads(position) == {"position": 3}
    trainer.fit(data, steps=3)
    assert len(opened) == 2


def test_unexpected_training_eof_still_closes_and_does_not_save_success(tmp_path):
    source = Source(end=1)
    checkpoints = Checkpoints(str(tmp_path))
    trainer = Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0),
                      checkpoints=checkpoints)
    with pytest.raises(StopIteration):
        trainer.fit(Dataset(lambda: source, None, None, 8), steps=2)
    assert source.closed.is_set()
    assert checkpoints.latest is None


def test_metric_failure_closes_both_iterators_before_it_escapes():
    train, validation = Source(), Source(end=2)

    class Evaluated(Regression):
        def evaluate(self, params, batch, step):
            return Representations(features=batch["x"], labels=jnp.zeros((8,), jnp.int32))

    class InvalidMetric:
        name, reads = "invalid", Representations

        def __call__(self, artifact, batch):
            return np.linalg.inv(artifact.features)  # Not square: a real metric failure.

        def reduce(self, values):
            return np.mean(values)

    trainer = Trainer(Evaluated(), optax.sgd(0.01), key=jax.random.key(0))
    with pytest.raises(np.linalg.LinAlgError):
        trainer.fit(Dataset(lambda: train, lambda: validation, None, 8),
                    steps=2, eval_every=1, metrics=(InvalidMetric(),))
    assert train.closed.is_set() and validation.closed.is_set()
    assert validation.owners[-1] == threading.get_ident()


def test_fit_attempts_every_cleanup_without_masking_tracker_error(tmp_path, monkeypatch):
    primary = LookupError("tracker unavailable")
    source = Source(close_failure=OSError("source finalization failed"))
    stopped, waited = threading.Event(), threading.Event()

    class Tracker:
        def log(self, scalars, step):
            raise primary

    class Waiting(Checkpoints):
        def wait(self):
            super().wait()
            waited.set()
            raise OSError("checkpoint wait failed")

    def stop_trace():
        stopped.set()
        raise OSError("trace stop failed")

    monkeypatch.setattr(jax.profiler, "start_trace", lambda directory: None)
    monkeypatch.setattr(jax.profiler, "stop_trace", stop_trace)
    trainer = Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0),
                      checkpoints=Waiting(str(tmp_path)), tracker=Tracker(),
                      profile=Profile(str(tmp_path / "trace"), steps=5, warmup=0))
    with pytest.raises(LookupError) as raised:
        trainer.fit(Dataset(lambda: source, None, None, 8), steps=2, log_every=1)
    assert raised.value is primary
    assert source.closed.is_set() and stopped.is_set() and waited.is_set()
    notes = "\n".join(primary.__notes__)
    assert all(message in notes for message in (
        "source finalization failed", "trace stop failed", "checkpoint wait failed"))


def test_checkpointability_refusal_finalizes_the_untransferred_source(tmp_path):
    closed = threading.Event()

    class Uncheckpointable:
        def __iter__(self):
            return self

        def __next__(self):
            return next(batches())

        def close(self):
            closed.set()

    trainer = Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0),
                      checkpoints=Checkpoints(str(tmp_path)))
    with pytest.raises(ValueError, match="get_state"):
        trainer.fit(Dataset(Uncheckpointable, None, None, 8), steps=1, checkpoint_every=1)
    assert closed.is_set()

def test_constructor_starts_no_unowned_source_work():
    source = Source()
    with DevicePrefetchIterator(source, build_mesh(), source_state=b'{"position": 2}'):
        assert source.owners == []
    assert source.closed.is_set()
    assert source.position == 0
    assert source.owners[0] != threading.get_ident()


def test_sigint_during_fit_restoration_closes_only_after_restoration_returns(tmp_path):
    checkpoints = Checkpoints(str(tmp_path))
    trainer = Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0),
                      checkpoints=checkpoints)
    trainer.fit(Dataset(Source, None, None, 8), steps=1)
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
            trainer.fit(Dataset(lambda: source, None, None, 8), steps=2)
        assert source.closed.is_set()
        assert len(set(source.owners)) == 1
        assert source.owners[0] != threading.get_ident()
    finally:
        stopped.set()
        sender.join(10)


def test_tokenized_cancellation_reaches_source_during_finalization():
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

    wrapped = tokenized(Finalizing, None)()
    stream = DevicePrefetchIterator(wrapped, build_mesh())
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
    with pytest.raises(RuntimeError) as caught:
        with DevicePrefetchIterator(Source(), build_mesh()) as stream:
            next(stream)
    assert caught.value is failure
    assert owners == [threading.get_ident()]
    stream.close()
