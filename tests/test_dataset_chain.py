"""The chain surface: stores carry the verbs, mix reads members of every kind, holdout splits by identity."""

import itertools
import json

import grain.python as grain
import numpy as np
import pytest

from dew.data import Dataset, Files, Loading, Memory, from_grain, mix
from dew.data.compile import compile, iterate
from dew.data.dataset import DataPartition

READ = Loading(workers=0, threads=1, read_buffer=2, worker_buffer=1)


def rows(origin: str, count: int = 64) -> list[dict]:
    return [{"origin": origin, "i": index} for index in range(count)]


def test_a_mix_reads_memory_grain_and_files_members_at_their_shares(tmp_path):
    """Three kinds of store in one mix, each its own share of every step."""
    path = tmp_path / "tickets.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows("files")))
    memory, counter = Memory(rows("memory")), from_grain(grain.MapDataset.source(rows("grain")), "counter")
    members = [memory.shuffle(1).repeat(), counter.shuffle(2).repeat(), Files(path.as_uri()).named("tickets")
               .shuffle(3).repeat()]
    assert all(isinstance(member, Dataset) for member in (memory, counter, *members))
    data = compile(mix(members, weights=[0.5, 0.25, 0.25]).batch(8), read="indexed", provision=True,
                   loading=READ)
    stream = data.train(DataPartition())
    try:
        origins = np.concatenate([batch["origin"] for batch in itertools.islice(stream, 32)])
    finally:
        stream.close()
    shares = {origin: float(np.mean(origins == origin)) for origin in ("memory", "grain", "files")}
    assert shares == pytest.approx({"memory": 0.5, "grain": 0.25, "files": 0.25}, abs=0.05)


def test_holdout_puts_each_record_in_one_part_by_its_identity_the_same_in_every_read():
    """A 3-way holdout: the parts are disjoint and together every record, each about its share,
    the same whether the store is read whole or streamed, with the steps replayed on each."""
    corpus = Memory(rows("a", 4000)).map(lambda row: {**row, "seen": True})
    train, val, test = corpus.holdout(val=0.05, test=0.1)

    def keys(part, read):
        return {(row["origin"], int(row["i"])) for row in iterate(part, read=read, loading=READ)
                if row["seen"]}

    parts = [keys(part, "memory") for part in (train, val, test)]
    assert [keys(part, "streamed") for part in (train, val, test)] == parts
    assert sum(map(len, parts)) == len(set().union(*parts)) == 4000
    assert [len(part) / 4000 for part in parts] == pytest.approx([0.85, 0.05, 0.1], abs=0.02)
    first, held = Memory(rows("c", 10)).holdout(records=3)
    assert [int(row["i"]) for row in iterate(held, read="memory")] == [0, 1, 2]
    assert [int(row["i"]) for row in iterate(first, read="memory")] == list(range(3, 10))


def drop_thirds(row: dict) -> dict:
    """Drop every third row, as a fetch drops a dead url."""
    from dew.data import Rejected

    return Rejected("third") if row["i"] % 3 == 0 else row


drop_thirds.__annotations__["return"] = "dict | Rejected"


def test_dropped_records_are_the_same_whole_sharded_and_resumed():
    """A mix whose members drop records drops the same ones however it is read: one reader and two
    shares each see only the kept records, and over an epoch all of them; a resumed reader continues
    exactly where it stopped, its drop counts part of the saved position."""
    members = [Memory(rows(origin, 30)).shuffle(index).repeat().filter_map(drop_thirds)
               for index, origin in enumerate(("a", "b"))]
    data = compile(mix(members).batch(8), read="indexed", loading=READ)
    kept = {(origin, i) for origin in ("a", "b") for i in range(30) if i % 3}

    def keys(batch):
        return set(zip(batch["origin"].tolist(), batch["i"].tolist(), strict=True))

    whole = data.train(DataPartition())
    halves = [data.train(DataPartition(index, 2)) for index in range(2)]
    try:
        seen = set().union(*(keys(next(whole)) for _ in range(10)))
        shared = set().union(*(keys(next(half)) for half in halves for _ in range(10)))
        assert seen == shared == kept
        state = whole.get_state()
        rest = [keys(next(whole)) for _ in range(3)]
        share_state = halves[1].get_state()
        share_rest = [keys(next(halves[1])) for _ in range(3)]
    finally:
        for reader in (whole, *halves):
            reader.close()
    for partition, saved, expected in ((DataPartition(), state, rest),
                                       (DataPartition(1, 2), share_state, share_rest)):
        resumed = data.train(partition)
        try:
            resumed.set_state(saved)
            assert [keys(next(resumed)) for _ in range(3)] == expected
        finally:
            resumed.close()
    assert b'"third"' in state
