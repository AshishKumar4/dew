"""Record boundaries through packing, batching, encodings and threaded orders."""

import dataclasses
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import grain.python as grain
import numpy as np
import pytest

from dew.data import Loading, Memory, preferences, presets, prompts, text
from dew.data.compile import compile, iterate
from dew.data.dataset import DataPartition
from dew.data.ordering import BlockShuffle
from dew.objectives.base import VALID_ROWS

READ = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
@pytest.mark.parametrize("policy", ["first-fit", "best-fit"])
@pytest.mark.parametrize("planned", [False, True])
def test_pack_refuses_implicit_document_chunks_in_every_read(read, policy, planned):
    """Both eager records and certified lengths refuse an unsplit long document."""
    documents = [np.arange(40, dtype=np.int32), np.arange(3, dtype=np.int32)]
    stored = (Memory({"text": documents, "length": [len(row) for row in documents]}) if planned
              else Memory([{"text": row} for row in documents]))
    with pytest.raises(ValueError):
        data = compile(stored.pack(16, policy=policy).batch(1), read=read, loading=READ)
        reader = data.train(DataPartition())
        try:
            next(reader)
        finally:
            reader.close()


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_explicit_overlapping_chunks_keep_all_document_targets(read):
    """Explicit chunking keeps each next-token pair exactly once through packing."""
    stored = Memory([{"text": np.arange(40, dtype=np.int32)}])
    rows = iterate(stored.flat_map(text.chunk(16, overlap=1)).pack(16), read=read, loading=READ)
    pairs = []
    for row in rows:
        ids, segments = row["text"], row["text_segment_ids"]
        scored = (segments[:-1] > 0) & (segments[:-1] == segments[1:])
        pairs.extend(zip(ids[:-1][scored].tolist(), ids[1:][scored].tolist(), strict=True))
    assert sorted(pairs) == [(token, token + 1) for token in range(39)]


@dataclasses.dataclass(frozen=True)
class InspectObjects:
    def __call__(self, batch):
        expected = batch["id"].tolist()
        assert batch["caption"].shape == (len(expected),)
        assert batch["caption"].tolist() == [f"image {index}" for index in expected]
        assert batch["messages"] == [[{"role": "user", "content": str(index)}] * (index + 1)
                                     for index in expected]
        assert batch["tools"] == [{"names": [f"tool {index}"]} for index in expected]
        assert isinstance(batch["objects"], list)
        for held, index in zip(batch["objects"], expected, strict=True):
            np.testing.assert_array_equal(held, np.array([{"index": index}], dtype=object))
        np.testing.assert_array_equal(batch["nested"]["pixels"],
                                      np.asarray([[index, index + 1] for index in expected], np.float32))
        np.testing.assert_array_equal(batch["image"],
                                      np.stack([np.full((2, 2, 3), index, np.uint8) for index in expected]))
        mask = {VALID_ROWS: batch[VALID_ROWS]} if VALID_ROWS in batch else {}
        return {"id": batch["id"], "pixels": batch["nested"]["pixels"], "image": batch["image"], **mask}


def object_records(count):
    return [{"id": np.int32(index), "caption": f"image {index}",
             "image": np.full((2, 2, 3), index, np.uint8),
             "messages": [{"role": "user", "content": str(index)}] * (index + 1),
             "tools": {"names": [f"tool {index}"]},
             "objects": np.array([{"index": index}], dtype=object),
             "nested": {"pixels": np.asarray([index, index + 1], np.float32)}} for index in range(count)]


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
@pytest.mark.parametrize("preset", [False, True])
def test_batch_keeps_ragged_object_columns_for_host_steps_and_resumes(read, preset):
    """Host work sees whole object records beside stacked numeric arrays."""
    stored = Memory(object_records(4)).repeat()
    chain = stored.pipe(presets.images, 2, batch=2) if preset else stored.batch(2)
    data = compile(chain.map(InspectObjects()), read=read, loading=READ)
    reader = data.train(DataPartition())
    try:
        first = next(reader)
        saved = reader.get_state()
        expected = next(reader)
        np.testing.assert_array_equal(first["id"], [0, 1])
        np.testing.assert_array_equal(expected["id"], [2, 3])
    finally:
        reader.close()
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(saved)
        restored = next(resumed)
        for name, values in expected.items():
            np.testing.assert_array_equal(restored[name], values)
    finally:
        resumed.close()


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_validation_shares_and_padding_preserve_whole_object_records(read):
    """Finite shares pad the final batch without cutting into its object records."""
    chain = Memory(object_records(5)).batch(4, drop_remainder=False).map(InspectObjects())
    data = compile(chain, validation=chain, read=read, loading=READ)
    for count in (1, 2, 4):
        seen = []
        for share in range(count):
            reader = data.val(DataPartition(share, count))
            try:
                for batch in reader:
                    mask = batch.get(VALID_ROWS, np.ones(len(batch["id"]), bool))
                    seen.extend(batch["id"][mask].tolist())
            finally:
                reader.close()
        assert sorted(seen) == list(range(5))


def test_ramped_batches_keep_object_columns_and_restore_their_row_order():
    """A batch ramp slices and joins records, not the contents of object columns."""
    from dew.data.dataset import RampedStream, Stage

    data = compile(Memory(object_records(8)).repeat().batch(4), read="indexed", loading=READ)
    stages = (Stage(batch=2, records=0), Stage(batch=4, records=4))
    reader = RampedStream(data.train(DataPartition()), stages, DataPartition())
    try:
        first = InspectObjects()(next(reader))
        second = InspectObjects()(next(reader))
        saved = reader.get_state()
        expected = InspectObjects()(next(reader))
        np.testing.assert_array_equal(first["id"], [0, 1])
        np.testing.assert_array_equal(second["id"], [2, 3])
        np.testing.assert_array_equal(expected["id"], [4, 5, 6, 7])
    finally:
        reader.close()
    resumed = RampedStream(data.train(DataPartition()), stages, DataPartition())
    try:
        resumed.set_state(saved)
        actual = InspectObjects()(next(resumed))
        for name, values in expected.items():
            np.testing.assert_array_equal(actual[name], values)
    finally:
        resumed.close()


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_batch_refuses_ragged_numeric_arrays_beside_object_columns(read):
    rows = [{"text": np.arange(length, dtype=np.int32), "messages": ["turn"] * length}
            for length in (2, 3)]
    with pytest.raises(ValueError):
        reader = compile(Memory(rows).batch(2), read=read, loading=READ).train(DataPartition())
        try:
            next(reader)
        finally:
            reader.close()


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_prompt_encoding_passes_unread_catalog_fields_through(read):
    """Prompt padding and reward encoding leave catalog and routing fields intact."""
    originals = [{"id": index, "prompt": [index + 1, index + 2], "data_source": "rule",
                  "reward_model": {"ground_truth": str(index)}, "extra_info": {"split": "train"},
                  "ability": "math", "agent_name": "solver", "catalog": {"tags": [str(index)]}}
                 for index in range(3)]
    rows = list(iterate(Memory(originals).filter_map(prompts.encode("byte", max_prompt_len=4)),
                        read=read, loading=READ))
    assert len(rows) == len(originals)
    for row, original in zip(rows, originals, strict=True):
        for name in ("id", "ability", "agent_name", "catalog"):
            assert row[name] == original[name]
        assert "reward_model" not in row
        np.testing.assert_array_equal(row["prompt"], [0, 0, *original["prompt"]])
        assert int(row["prompt_length"]) == 2
        for name, expected in (("data_source", "rule"), ("ground_truth", str(original["id"])),
                               ("extra_info", '{"split": "train"}')):
            assert bytes(row[name][row[name] != 0].astype(np.uint8)).decode() == expected


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_preference_pair_passes_unread_catalog_fields_through(read):
    """Pair padding consumes only the completion ids and their masks."""
    originals = [{"id": index, "chosen": [1, 2, index + 3], "rejected": [1, index + 4],
                  "chosen_mask": [0, 1, 1], "rejected_mask": [0, 1],
                  "catalog": {"tags": [str(index)]}} for index in range(3)]
    rows = list(iterate(Memory(originals).map(preferences.pair(4)), read=read, loading=READ))
    assert len(rows) == len(originals)
    for row, original in zip(rows, originals, strict=True):
        assert row["id"] == original["id"] and row["catalog"] == original["catalog"]
        assert not {"chosen", "rejected", "chosen_mask", "rejected_mask"} & row.keys()
        np.testing.assert_array_equal(row["input_ids"],
                                      [[*original["chosen"], 0], [*original["rejected"], 0, 0]])
        np.testing.assert_array_equal(row["completion_mask"], [[0, 1, 1, 0], [0, 1, 0, 0]])


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
@pytest.mark.parametrize("row", [
    {"id": 7}, {"id": 7, "prompt": ""},
    {"id": 7, "chosen": [1, 2]},
    {"id": 7, "chosen": [1, 2], "rejected": [3], "chosen_mask": [1]},
])
def test_encodings_still_validate_the_fields_they_read(read, row):
    stored = Memory([row])
    chain = (stored.map(preferences.pair(4)) if "chosen" in row
             else stored.filter_map(prompts.encode("byte", max_prompt_len=4)))
    with pytest.raises(ValueError):
        list(iterate(chain, read=read, loading=READ))


def yield_reader_threads(frame, event, argument):
    """Open scheduling points throughout real indexed reads, without replacing their code."""
    time.sleep(0)
    return yield_reader_threads


def test_threaded_block_shuffle_matches_serial_epochs_and_pickles():
    """Concurrent reads across three tiny epochs keep each deterministic shard order."""
    total, epochs = 6, 3
    rows = grain.MapDataset.source([{"id": np.int32(index)} for index in range(total)]).repeat(epochs)
    indices = np.random.default_rng(7).integers(0, total * epochs, 500).tolist()
    serial = BlockShuffle(rows, (1, 2, 3), 2, 11)
    expected = [serial[index]["id"] for index in indices]
    blocked = BlockShuffle(rows, (1, 2, 3), 2, 11)
    previous = sys.getswitchinterval()
    previous_trace = threading.gettrace()
    sys.setswitchinterval(1e-6)
    threading.settrace(yield_reader_threads)
    try:
        with ThreadPoolExecutor(max_workers=32) as readers:
            actual = [row["id"] for row in readers.map(blocked.__getitem__, indices)]
    finally:
        sys.setswitchinterval(previous)
        threading.settrace(previous_trace)
    np.testing.assert_array_equal(actual, expected)
    restored = pickle.loads(pickle.dumps(blocked))
    np.testing.assert_array_equal([restored[index]["id"] for index in indices], expected)
