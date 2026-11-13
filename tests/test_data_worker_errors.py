"""Errors from real loading workers and transforms prepared before workers start."""

import dataclasses
import faulthandler
import multiprocessing
import os
from itertools import islice

import numpy as np
import pytest

from dew.data import Loading, Memory
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline


@dataclasses.dataclass(frozen=True)
class _WorkerRefusal:
    def __call__(self, row):
        if multiprocessing.parent_process() is not None and int(row["text"][-1]) > 100:
            raise ValueError("plain refusal")
        return {**row, "transform_pid": os.getpid()}


def _chain():
    rows = [{"text": np.arange(5, dtype=np.int32) + index} for index in range(8)]
    rows[-1]["text"][-1] = 205
    return Pipeline(Memory(rows, shard_rows=4)).map(_WorkerRefusal()).batch(4)


@pytest.fixture(autouse=True)
def _bounded_read():
    faulthandler.dump_traceback_later(60, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


@pytest.mark.parametrize("read", ["indexed", "streamed"])
@pytest.mark.parametrize("workers", [1, 2])
def test_a_loading_workers_original_error_reaches_its_reader(read, workers):
    data = compile(_chain(), read=read,
                   loading=Loading(workers=workers, threads=1, read_buffer=2, worker_buffer=1))
    reader = data.train(DataPartition())
    try:
        first = next(reader)
        assert np.all(np.asarray(first["transform_pid"]) != os.getpid())
        with pytest.raises(ValueError) as raised:
            next(reader)
        assert type(raised.value) is ValueError
        assert raised.value.args == ("plain refusal",)
    finally:
        reader.close()


@pytest.mark.parametrize("read", [None, "memory"])
def test_a_memory_reads_map_is_prepared_before_loading_workers(read):
    data = compile(_chain(), read=read,
                   loading=Loading(workers=2, threads=1, read_buffer=2, worker_buffer=1))
    reader = data.train(DataPartition())
    try:
        batches = list(islice(reader, 4))
        assert len(batches) == 4
        transformed = np.concatenate([batch["transform_pid"] for batch in batches])
        np.testing.assert_array_equal(transformed, np.full(16, os.getpid()))
        ends = np.concatenate([batch["text"][:, -1] for batch in batches])
        np.testing.assert_array_equal(ends, np.tile([4, 5, 6, 7, 8, 9, 10, 205], 2))
    finally:
        reader.close()
