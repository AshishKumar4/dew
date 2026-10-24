"""Scratch: the checkpoint-writer scenarios many times over, beside threads that start and end."""

import threading

import pytest
from test_trainer import (
    test_a_run_that_fails_after_a_failed_write_raises_its_own_error_and_closes_its_checkpoints as after_failed,
)
from test_trainer import test_checkpoints_whose_write_failed_close_and_keep_the_error_their_run_raised as failed
from test_trainer import test_closed_checkpoints_leave_no_writer_running_across_trials as closed


@pytest.fixture
def churn():
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            worker = threading.Thread(target=sum, args=(range(20000),))
            worker.start()
            worker.join()

    runner = threading.Thread(target=loop)
    runner.start()
    yield
    stop.set()
    runner.join()


@pytest.mark.parametrize("round", range(25))
@pytest.mark.parametrize("body_fails", [False, True])
def test_failed(tmp_path, monkeypatch, churn, round, body_fails):
    failed(tmp_path, monkeypatch, body_fails)


@pytest.mark.parametrize("round", range(15))
def test_closed(tmp_path, churn, round):
    closed(tmp_path)


@pytest.mark.parametrize("round", range(15))
def test_after_failed(tmp_path, monkeypatch, churn, round):
    after_failed(tmp_path, monkeypatch)
