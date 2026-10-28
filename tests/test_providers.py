"""Storage records enter the same compiler in every read mode."""

import dataclasses
import itertools
import json
from pathlib import Path

import grain.python as grain
import numpy as np
import pytest
from data_support import WideDraw

from dew.batch import VALID_ROWS
from dew.data import TFDS, Files, Loading, Memory
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline

READ = Loading(workers=0, threads=1, read_buffer=4, worker_buffer=1)
PREPARED = Path(__file__).parent / "fixtures" / "tfds" / "dew_images" / "1.0.0"


@dataclasses.dataclass(frozen=True)
class Draw(grain.RandomMapTransform):
    def random_map(self, element, rng):
        return {**element, "draw": np.int32(rng.integers(1 << 20))}


def run(train, *, validation=None, batch=4, seed=0, read="indexed", loading=READ, provision=False):
    """`train`'s records as a run reads them: shuffled by `seed`, in windows of 64 where streamed,
    repeated and in batches of `batch`; `validation`'s read once to validate on."""
    order = train.shuffle(seed, window=64 if read == "streamed" else None).repeat()
    held = None if validation is None else validation.batch(batch)
    return compile(order.batch(batch), validation=held, read=read, seed=seed, loading=loading,
                   provision=provision)


def indices(iterator, count):
    return [[int(value) for value in batch["index"]] for batch in itertools.islice(iterator, count)]


@pytest.fixture
def jsonl(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps({"index": index, "value": [index, -index]}) + "\n"
                            for index in range(24)))
    return Files(str(path), format="jsonl", shard_bytes=96)


def test_indexed_jsonl_refuses_to_build_an_index_without_provisioning(jsonl):
    with pytest.raises((ValueError, FileNotFoundError)):
        run(jsonl)
    data = run(jsonl, provision=True)
    stream = data.train(DataPartition())
    try:
        assert sorted(value for batch in indices(stream, 6) for value in batch) == list(range(24))
    finally:
        stream.close()


def test_a_held_out_slice_provisions_the_source_it_reads(jsonl):
    """A held-out slice and the rows after it read their source by index, so provision=True
    builds that source's index rather than refusing the slices, which have none of their own."""
    train, held_out = Pipeline(jsonl).holdout(records=4)
    data = run(train, validation=held_out, provision=True)
    stream, held = data.train(DataPartition()), data.val(DataPartition())
    try:
        assert sorted(value for batch in indices(stream, 5) for value in batch) == list(range(4, 24))
        assert [value for batch in held for value in batch["index"]] == [0, 1, 2, 3]
    finally:
        stream.close()
        held.close()


def test_memory_and_provisioned_indexed_sources_produce_identical_epochs(jsonl):
    jsonl.provision()
    memory, indexed = run(jsonl, seed=7, read="memory"), run(jsonl, seed=7)
    left, right = memory.train(DataPartition()), indexed.train(DataPartition())
    try:
        for _ in range(12):
            actual, expected = next(left), next(right)
            for field in actual:
                np.testing.assert_array_equal(actual[field], expected[field])
    finally:
        left.close()
        right.close()


def test_a_native_prepared_tfds_split_reads_without_provisioning():
    source = TFDS("dew_images", path=str(PREPARED), split="all")
    records = source.records("indexed")
    assert len(records) == 20 and "image" in records[0] and "label" in records[0]
    assert source.metadata()["rows"] == 20


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
@pytest.mark.parametrize("workers", [0, 2])
def test_a_validation_pass_ends_with_each_record_scored_once(jsonl, read, workers):
    if read == "indexed":
        jsonl.provision()
    data = run(jsonl, validation=jsonl, batch=8, read=read,
               loading=Loading(workers=workers, threads=1, read_buffer=4, worker_buffer=1))
    assert data.val is not None
    stream = data.val(DataPartition())
    try:
        batches = list(stream)
    finally:
        stream.close()
    real = np.concatenate([batch["index"][batch.get(VALID_ROWS, np.ones(8, bool))] for batch in batches])
    assert real.tolist() == list(range(24))


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_validation_takes_accepted_rows_before_the_share_cut(jsonl, read):
    if read == "indexed":
        jsonl.provision()
    train = Pipeline(jsonl).shuffle(3, window=8 if read == "streamed" else None).repeat().batch(4)
    held = Pipeline(jsonl).filter(lambda row: int(row["index"]) % 3 == 0).take(5).batch(4)
    data = compile(train, validation=held, read=read, loading=READ, seed=3)
    assert data.val is not None
    for shares in (1, 2):
        marked = []
        for share in range(shares):
            stream = data.val(DataPartition(share, shares))
            try:
                marked.extend(int(value) for batch in stream for value in
                              batch["index"][batch.get(VALID_ROWS, np.ones(4 // shares, bool))])
            finally:
                stream.close()
        assert sorted(marked) == [0, 3, 6, 9, 12]


@pytest.mark.parametrize("read", ["indexed", "streamed"])
@pytest.mark.parametrize("workers", [0, 2])
def test_a_shuffled_stream_restores_its_exact_next_batches(jsonl, read, workers):
    if read == "indexed":
        jsonl.provision()
    data = run(jsonl, read=read, seed=19, loading=Loading(workers=workers, threads=1, read_buffer=4,
                                                          worker_buffer=1))
    stream = data.train(DataPartition())
    try:
        indices(stream, 3)
        state = stream.get_state()
        expected = indices(stream, 7)
    finally:
        stream.close()
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(state)
        assert indices(resumed, 7) == expected
    finally:
        resumed.close()


def test_a_per_share_position_refuses_another_worker_topology(jsonl):
    data = run(jsonl, read="streamed")
    stream = data.train(DataPartition())
    try:
        next(stream)
        state = stream.get_state()
    finally:
        stream.close()
    other = run(jsonl, read="streamed", loading=Loading(workers=2, threads=1, read_buffer=4,
                                                        worker_buffer=1)).train(DataPartition())
    try:
        with pytest.raises(ValueError):
            other.set_state(state)
    finally:
        other.close()


@pytest.mark.parametrize("draw, seed", [(Draw(), 11), (WideDraw(), 7)])
def test_per_record_rng_draws_agree_with_plain_grain(draw, seed):
    """A random step draws each record as Grain's `random_map` does, from its seed and the record's
    index, under the seed Dew keys the step by: the run's seed, the corpus's place in its mixtures
    (none here) and the random steps before it (none), drawn as a SeedSequence
    (`dew.data.lowering._Build.stage_seed`), so a read or a wrapper does not move it as Grain's
    default seed would."""
    rows = [{"index": np.int32(index)} for index in range(24)]
    data = compile(Pipeline(Memory(rows)).shuffle(seed).repeat().random_map(draw).batch(4),
                   seed=seed, read="indexed", loading=READ)
    keyed = int(np.random.SeedSequence((seed, 0)).generate_state(1, np.uint32)[0])
    oracle = iter(grain.MapDataset.source(rows).shuffle(seed).repeat().random_map(
        draw, seed=keyed).to_iter_dataset(
        grain.ReadOptions(num_threads=1, prefetch_buffer_size=4)).batch(4))
    actual = data.train(DataPartition())
    try:
        for _ in range(12):
            expected, batch = next(oracle), next(actual)
            for field in expected:
                np.testing.assert_array_equal(batch[field], expected[field])
    finally:
        actual.close()
        oracle.close()


def test_record_holdouts_are_disjoint_from_every_training_epoch():
    rows = [{"index": np.int32(index)} for index in range(24)]
    train, held_out = Memory(rows).holdout(records=8)
    data = run(train, validation=held_out, read="memory")
    assert data.val is not None
    training, held = data.train(DataPartition()), data.val(DataPartition())
    try:
        assert indices(held, 8) == [[0, 1, 2, 3], [4, 5, 6, 7]]
        epochs = [indices(training, 4) for _ in range(3)]
        assert all(sorted(value for batch in epoch for value in batch) == list(range(8, 24))
                   for epoch in epochs)
    finally:
        training.close()
        held.close()
