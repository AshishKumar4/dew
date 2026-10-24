"""A mixture of a thousand sources at CPU scale: it compiles, reads and resumes with no cost per
source at each step, its weights are each source's share, and a source that runs out follows its
policy. Each test records what it measured (`record_property`) and prints it."""

import gc
import itertools
import os
import resource
import time
import tracemalloc

import numpy as np
import pytest

from dew.data import Files, Loading, Memory
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline, mix

SOURCES = 1000
RECORDS = 8
BATCH = 16
READ = Loading(workers=0, threads=1, read_buffer=8, worker_buffer=1)


class Counted:
    """A source's records, each saying which source and record it is, and how many reads reach it."""

    reads = 0

    def __init__(self, tag: int, records: int = RECORDS):
        self.tag, self.records = tag, records

    def __repr__(self) -> str:
        return f"Counted(tag={self.tag}, records={self.records})"

    def __len__(self) -> int:
        return self.records

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.records:
            raise IndexError(index)
        Counted.reads += 1
        return {"id": np.int64(self.tag * 1000 + index)}


def sources(count: int = SOURCES, *, repeat: bool = True) -> list[Pipeline]:
    chains = [Pipeline(Memory(Counted(tag)), name=f"source-{tag}") for tag in range(count)]
    return [chain.repeat() for chain in chains] if repeat else chains


def ids(stream, batches: int) -> list[int]:
    return [int(value) for batch in itertools.islice(stream, batches) for value in batch["id"]]


def open_files() -> int:
    return len(os.listdir("/proc/self/fd"))


def test_a_thousand_sources_compile_and_step_without_a_cost_per_source(record_property):
    """A step reads the records it batches and nothing of the sources it does not draw: a thousand
    sources cost a step what ten do, and compiling them costs memory linear in their count."""
    measured = {}
    for count in (10, SOURCES):
        gc.collect()
        tracemalloc.start()
        began = time.perf_counter()
        data = compile(mix(sources(count)).batch(BATCH), read="memory", loading=READ)
        built = time.perf_counter() - began
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stream = data.train(DataPartition())
        try:
            ids(stream, 4)
            Counted.reads = 0
            began = time.perf_counter()
            ids(stream, 64)
            stepped = (time.perf_counter() - began) / 64
            reads = Counted.reads
        finally:
            stream.close()
        measured[count] = {"compile_s": built, "compile_peak_mib": peak / 2**20, "step_ms": 1e3 * stepped,
                           "reads_per_step": reads / 64}
        record_property(f"sources_{count}", measured[count])
        print(f"sources={count}", measured[count])
    # A read buffer reads at most its depth ahead, so 64 steps reach no more than 64 batches and it.
    for count in (10, SOURCES):
        assert measured[count]["reads_per_step"] <= BATCH + READ.read_buffer * BATCH / 64
    record_property("rss_mib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    print("rss_mib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def test_a_thousand_file_sources_hold_no_file_open_per_source(tmp_path, record_property):
    """A thousand JSON-lines files read as one mixture hold a bounded number of files open while
    they read, not one each."""
    paths = []
    for tag in range(SOURCES):
        path = tmp_path / f"{tag:04d}.jsonl"
        path.write_text("".join(f'{{"id": {tag * 1000 + index}}}\n' for index in range(RECORDS)))
        paths.append(path)
    before = open_files()
    chains = [Pipeline(Files(str(path)), name=f"file-{tag}").repeat() for tag, path in enumerate(paths)]
    data = compile(mix(chains).batch(BATCH), read="memory", loading=READ)
    stream = data.train(DataPartition())
    try:
        ids(stream, 64)
        held = open_files() - before
    finally:
        stream.close()
    record_property("open_files_while_reading", held)
    print("open_files_while_reading", held)
    assert held < 64


def test_the_weights_are_each_sources_share_of_the_records_read():
    """A thousand sources weighted 1 or 2 each read their share of a whole period of the mix, the
    records in which every proportion comes round once, to the record; weights scaled together
    read the same records."""
    weights = [float(tag % 2 + 1) for tag in range(SOURCES)]

    def read(scale, batches):
        data = compile(mix(sources(), weights=[scale * weight for weight in weights]).batch(BATCH),
                       read="memory", loading=READ)
        stream = data.train(DataPartition())
        try:
            return ids(stream, batches)
        finally:
            stream.close()

    # Grain's mix scales the least weight to 100 records a period.
    period = int(100 * sum(weights))
    whole = read(1.0, period // BATCH)
    assert len(whole) == period
    counts = np.bincount(np.asarray(whole) // 1000, minlength=SOURCES)
    np.testing.assert_array_equal(counts, 100 * np.asarray(weights, np.int64))
    assert read(0.25, 64) == whole[:64 * BATCH]


def test_an_epoch_of_a_thousand_finite_sources_reads_every_record():
    """Sources written without a repeat run out at their own lengths; an epoch reads every record of
    every one of them, the mixture's policy for a source that runs out."""
    lengths = [RECORDS + tag % 5 for tag in range(SOURCES)]
    chains = [Pipeline(Memory(Counted(tag, length)), name=f"source-{tag}")
              for tag, length in enumerate(lengths)]
    data = compile(mix(chains).batch(BATCH), read="memory", loading=READ)
    stream = data.train(DataPartition())
    try:
        read = ids(stream, data.steps_per_epoch)
    finally:
        stream.close()
    every = {tag * 1000 + index for tag, length in enumerate(lengths) for index in range(length)}
    assert every <= set(read)


@pytest.mark.parametrize("stop", [1, 37])
def test_a_thousand_source_mixture_resumes_where_it_stopped(stop):
    """A run stopped after `stop` steps and resumed from its saved position reads what the run that
    never stopped reads next, record for record."""
    data = compile(mix(sources(), weights=[float(tag % 7 + 1) for tag in range(SOURCES)]).batch(BATCH),
                   read="memory", loading=READ)
    whole = data.train(DataPartition())
    try:
        ids(whole, stop)
        state = whole.get_state()
        expected = ids(whole, 20)
    finally:
        whole.close()
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(state)
        assert ids(resumed, 20) == expected
    finally:
        resumed.close()


def test_a_thousand_source_selection_reads_in_order_what_each_position_reads_alone():
    """Read one position after another, a mixture's selection reads runs of positions at once; it
    reads what each position reads alone, Grain's order, and a position reached out of order (a
    resume, a seek) reads the same, before and after a run."""
    from dew.data.mixing import ProportionalSelection, Proportions

    weights = [float(tag % 7 + 1) for tag in range(SOURCES)]
    lengths = [10**6] * SOURCES
    selection = ProportionalSelection(lengths, weights)
    proportions = Proportions.of(weights)
    ordered = [selection[index] for index in range(10_000)]
    assert ordered == [proportions.at(index) for index in range(10_000)]
    for index in (123_456, 7, 9_999, 50_000, 50_001, 3):
        assert selection[index] == proportions.at(index)
