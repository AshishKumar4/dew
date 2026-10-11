"""Storage records enter the same compiler in every read mode."""

import dataclasses
import itertools
import json
from pathlib import Path

import grain.python as grain
import numpy as np
import pytest

from dew.config import RunConfig, TrainerConfig
from dew.data import TFDS, DataPartition, Files, HoldOut, Loading, Memory, Pipeline, Records, compile, load
from dew.objectives.base import VALID_ROWS

READ = Loading(workers=0, threads=1, read_buffer=4, worker_buffer=1)
PREPARED = Path(__file__).parent / "fixtures" / "tfds" / "dew_images" / "1.0.0"


@dataclasses.dataclass(frozen=True)
class Draw(grain.RandomMapTransform):
    def random_map(self, element, rng):
        return {**element, "draw": np.int32(rng.integers(1 << 20))}


def indices(iterator, count):
    return [[int(value) for value in batch["index"]] for batch in itertools.islice(iterator, count)]


@pytest.fixture
def jsonl(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps({"index": index, "value": [index, -index]}) + "\n"
                            for index in range(24)))
    return Files(str(path), format="jsonl", shard_bytes=96)


@pytest.mark.parametrize("uri", ["tfds://dew_images", "hf://datasets/acme/pets", "data/rows.jsonl"])
def test_load_requires_an_alias_or_a_typed_recipe(uri):
    with pytest.raises(ValueError, match="recipe or alias"):
        load(uri, batch=4)


def test_recipe_sources_round_trip_in_a_run_record(tmp_path):
    recipe = Records(source=Files(str(tmp_path / "rows.parquet"), format="parquet"),
                     validation=Files(str(tmp_path / "held.parquet"), format="parquet"))
    run = RunConfig(data=recipe, loading=READ, trainer=TrainerConfig(steps=1))
    assert RunConfig.from_dict(json.loads(json.dumps(run.to_dict()))) == run


def test_indexed_jsonl_refuses_to_build_an_index_without_provisioning(jsonl):
    with pytest.raises((ValueError, FileNotFoundError), match=r"index|provision"):
        load(Records(source=jsonl), batch=4, read="indexed", loading=READ)
    data = load(Records(source=jsonl), batch=4, read="indexed", provision=True, loading=READ)
    stream = data.train(DataPartition())
    try:
        assert sorted(value for batch in indices(stream, 6) for value in batch) == list(range(24))
    finally:
        stream.close()


def test_memory_and_provisioned_indexed_sources_produce_identical_epochs(jsonl):
    jsonl.provision()
    memory = load(Records(source=jsonl), batch=4, seed=7, read="memory", loading=READ)
    indexed = load(Records(source=jsonl), batch=4, seed=7, read="indexed", loading=READ)
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
    data = load(Records(source=jsonl, validation=jsonl), batch=8, read=read,
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
    data = load(Records(source=jsonl), batch=4, read=read, seed=19,
                loading=Loading(workers=workers, threads=1, read_buffer=4, worker_buffer=1))
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
    data = load(Records(source=jsonl), batch=4, read="streamed", loading=READ)
    stream = data.train(DataPartition())
    try:
        next(stream)
        state = stream.get_state()
    finally:
        stream.close()
    other = load(Records(source=jsonl), batch=4, read="streamed",
                 loading=Loading(workers=2, threads=1, read_buffer=4, worker_buffer=1)).train(DataPartition())
    try:
        with pytest.raises(ValueError, match=r"worker|topology|plan"):
            other.set_state(state)
    finally:
        other.close()


def test_per_record_rng_draws_agree_with_plain_grain():
    rows = [{"index": np.int32(index)} for index in range(24)]
    data = compile(Pipeline(Memory(rows)).shuffle(11).repeat().random_map(Draw()).batch(4),
                   seed=11, read="indexed", loading=READ)
    oracle = iter(grain.MapDataset.source(rows).seed(11).shuffle(11).repeat().random_map(
        Draw()).to_iter_dataset(
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
    data = load(Records(source=rows, validation=HoldOut(records=8)), batch=4,
                read="memory", loading=READ)
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
