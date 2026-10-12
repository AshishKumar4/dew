"""The chain surface: stores carry the verbs, mix reads members of every kind, holdout splits by identity."""

import itertools
import json

import grain.python as grain
import numpy as np
import pytest

from dew.data import Dataset, Files, Loading, Memory, from_grain, from_iter, mix, text
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
    # With a key, a document's other versions land in its part: the split follows "i" alone.
    revised = Memory([{**row, "origin": "a2"} for row in rows("a", 4000)])
    revised = revised.map(lambda row: {**row, "seen": True})
    by_key = [{i for _, i in keys(part, "memory")} for part in corpus.holdout(0.1, key="i")]
    assert by_key == [{i for _, i in keys(part, "memory")} for part in revised.holdout(0.1, key="i")]
    first, held = Memory(rows("c", 10)).holdout(records=3)
    assert [int(row["i"]) for row in iterate(held, read="memory")] == [0, 1, 2]
    assert [int(row["i"]) for row in iterate(first, read="memory")] == list(range(3, 10))


def drop_thirds(row: dict) -> dict:
    """Drop every third row, as a fetch drops a dead url."""
    from dew.data import Rejected

    return Rejected("third") if row["i"] % 3 == 0 else row


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


def tokenized_in_batches(columns: dict[str, list]) -> dict[str, list]:
    """A preprocessing batch: one call for several records, as a batched tokenizer runs, which
    keeps the even ones."""
    kept = [i for i in columns["i"] if i % 2 == 0]
    return {"i": kept, "doubled": [2 * i for i in kept]}


def test_a_batched_step_runs_on_groups_and_the_chain_goes_on_by_record_through_a_resume():
    """A step over groups of five records, which returns fewer, then the training batch: each kept
    record once, the batched work done, and a resumed reader continues exactly. A batched map must
    keep its row count."""
    with pytest.raises(ValueError, match="keeps the 5 rows"):
        next(compile(Memory(rows("a", 10)).map(tokenized_in_batches, batched=True, batch_size=5).batch(2),
                     read="memory", loading=READ).train(DataPartition()))
    chain = (Memory(rows("a", 30)).shuffle(4).repeat()
             .flat_map(tokenized_in_batches, batched=True, batch_size=5).batch(5))
    data = compile(chain, read="indexed", loading=READ)
    reader = data.train(DataPartition())
    try:
        first = [next(reader) for _ in range(3)]
        state = reader.get_state()
        rest = [next(reader)["i"].tolist() for _ in range(3)]
    finally:
        reader.close()
    assert sorted(np.concatenate([batch["i"] for batch in first]).tolist()) == list(range(0, 30, 2))
    assert all(np.array_equal(batch["doubled"], 2 * batch["i"]) for batch in first)
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(state)
        assert [next(resumed)["i"].tolist() for _ in range(3)] == rest
    finally:
        resumed.close()


def test_stats_counts_a_fresh_pass_and_mix_weighs_tokens():
    """stats reads a fresh reader and reports its drops by reason; a mix by tokens gives each member
    its share of target tokens, not of records."""
    counts = Memory(rows("a", 40)).filter_map(drop_thirds).stats(10)
    assert any(key.endswith("/third") and value > 0 for key, value in counts.items())
    short = Memory([{"text": np.arange(3, dtype=np.int32), "origin": "short"} for _ in range(64)])
    long = Memory([{"text": np.arange(9, dtype=np.int32), "origin": "long"} for _ in range(64)])
    mixed = mix([short.shuffle(0).repeat(), long.shuffle(1).repeat()], by="tokens")
    data = compile(mixed.select_columns("origin").batch(8), read="indexed", loading=READ)
    reader = data.train(DataPartition())
    try:
        origins = np.concatenate([next(reader)["origin"] for _ in range(40)])
    finally:
        reader.close()
    # Three-token records carry two targets, nine-token records eight: equal targets, four short to one long.
    assert np.mean(origins == "short") == pytest.approx(0.8, abs=0.05)


class Countdown:
    """A resumable iterator, as a streaming library keeps its own position."""

    def __init__(self, start: int = 0):
        self.next = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.next >= 40:
            raise StopIteration
        self.next += 1
        return {"i": self.next - 1, "origin": "iter"}


def test_an_iterator_resumes_through_its_own_state_or_by_replaying():
    """A stateful iterator resumes through state= and load_state=, and one without them by
    replaying the records it already gave; both continue exactly where they stopped."""
    for source in (from_iter(Countdown, name="countdown", state=lambda it: it.next,
                             load_state=lambda it, saved: setattr(it, "next", saved)),
                   from_iter(Countdown, name="replayed")):
        reader = compile(source.repeat().batch(4), read="streamed", loading=READ).train(DataPartition())
        try:
            first = [next(reader)["i"].tolist() for _ in range(3)]
            state = reader.get_state()
            rest = [next(reader)["i"].tolist() for _ in range(3)]
        finally:
            reader.close()
        assert [i for batch in first for i in batch] == list(range(12))
        resumed = compile(source.repeat().batch(4), read="streamed", loading=READ).train(DataPartition())
        try:
            resumed.set_state(state)
            assert [next(resumed)["i"].tolist() for _ in range(3)] == rest
        finally:
            resumed.close()


def test_chunks_cut_long_documents_keeping_every_target():
    pieces = list(Memory([{"text": np.arange(10)}]).flat_map(text.chunk(4, overlap=1)).iterate())
    assert [piece["text"].tolist() for piece in pieces] == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]]
