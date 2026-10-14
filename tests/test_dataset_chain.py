"""The chain surface: stores carry the verbs, mix reads members of every kind, holdout splits by identity."""

import dataclasses
import hashlib
import itertools
import json
from dataclasses import dataclass

import grain.python as grain
import numpy as np
import pytest

from dew.data import Dataset, Files, Loading, Memory, from_grain, from_hf, from_iter, mix, text
from dew.data.cache import Cache
from dew.data.compile import compile, iterate
from dew.data.dataset import DataPartition
from dew.data.pipeline import Configured, ObjectiveData, Rejected
from dew.data.plan import Manifest, StepIdentity

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


@pytest.mark.parametrize("read", ["memory", "streamed"])
@pytest.mark.parametrize("key", [None, "identity"])
@pytest.mark.parametrize("nested", ["mapping", "arrays", "object_array"])
def test_holdout_ignores_mapping_order_at_every_depth(read, key, nested):
    original = []
    reordered = []
    for index in range(64):
        first = {"z": np.arange(3, dtype=np.int32), "a": index}
        second = {"a": index, "z": np.arange(3, dtype=np.int32)}
        if nested == "arrays":
            first, second = [first, b"document"], [second, b"document"]
        elif nested == "object_array":
            first, second = np.asarray([first], dtype=object), np.asarray([second], dtype=object)
        original.append({"identity": first, "id": index})
        reordered.append({"id": index, "identity": second})

    def parts(documents):
        return [{int(row["id"]) for row in iterate(part, read=read, loading=READ)}
                for part in Memory(documents).holdout(0.5, key=key)]

    first, second = parts(original), parts(reordered)
    assert first[0].isdisjoint(first[1])
    assert first[0] | first[1] == set(range(64))
    assert first == second


def test_canonical_holdout_moves_unsorted_splits_but_keeps_sorted_and_scalar_hashes():
    from dew.data.sources.codec import encode

    original = [{"z": index, "a": {"z": "document", "a": index}} for index in range(200)]
    ordered = [{"a": {"a": index, "z": "document"}, "z": index} for index in range(200)]

    def legacy(documents, key=None):
        return {row["z"] for row in documents
                if int.from_bytes(hashlib.sha256(encode(row if key is None else row[key])).digest()[:8],
                                  "big") < 2**63}

    def held(documents, key=None):
        _, validation = Memory(documents).holdout(0.5, key=key)
        return {int(row["z"]) for row in iterate(validation, read="memory", loading=READ)}

    assert held(original) == held(ordered) == legacy(ordered)
    assert held(original) != legacy(original)
    assert held(original, "z") == legacy(original, "z")


def add_draw(row, rng):
    draw = np.int64(rng.integers(1, 2**16))
    return {**row, "value": row["value"] + draw, "draw": draw}


def double(row):
    return {**row, "value": 2 * row["value"]}


@dataclass(frozen=True)
class Draw(grain.RandomMapTransform):
    def random_map(self, row, rng):
        return add_draw(row, rng)


def draw_step(metadata):
    return Draw()


@dataclass(frozen=True)
class ObjectiveDraw(grain.RandomMapTransform):
    def random_map(self, row, rng):
        return {**row, "objective_input": row["value"], "objective_draw": rng.integers(1, 2**16)}


def objective_draw(metadata):
    return ObjectiveDraw()


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
@pytest.mark.parametrize("preparation", ["empty", "random", "configured", "configured_user"])
def test_objective_preparation_keeps_the_written_order_of_random_mixture_steps(read, preparation):
    """Objective work reads the rows each child's random map and later map made."""
    sources = [Memory([{**row, "value": row["i"]} for row in rows(origin, 16)]) for origin in ("a", "b")]
    members = [(source.map(Configured(draw_step, needs=(), scope="run", random=True))
                if preparation == "configured_user" else source.random_map(add_draw)).map(double)
               for source in sources]
    chain = mix(members).batch(8)
    steps = ()
    if preparation == "random":
        steps = (ObjectiveDraw(),)
    elif preparation == "configured":
        steps = (Configured(objective_draw, needs=(), scope="run", random=True),)
    expected = compile(chain, read=read, loading=READ, seed=37).train(DataPartition())
    trained = compile(chain, objective=ObjectiveData(train=steps), read=read,
                      loading=READ, seed=37).train(DataPartition())
    try:
        batches = list(itertools.islice(trained, 4))
        reference = list(itertools.islice(expected, 4))
        assert len(batches) == len(reference) == 4
        for batch, written in zip(batches, reference, strict=True):
            for field in ("i", "value", "draw"):
                np.testing.assert_array_equal(batch[field], written[field])
            np.testing.assert_array_equal(batch["value"], 2 * (batch["i"] + batch["draw"]))
            if steps:
                np.testing.assert_array_equal(batch["objective_input"], batch["value"])
    finally:
        expected.close()
        trained.close()


@dataclass(frozen=True)
class FetchLike:
    """Unsealed payloads, dropping unavailable rows as a network fetch does."""

    volatile = True
    rejects = True
    identity = "fetch-like@1"

    def __call__(self, row):
        return (Rejected("unavailable") if row["i"] % 3 == 0 else
                {**row, "payload": np.full(4, row["i"], dtype=np.uint8)})


@dataclass(frozen=True)
class ConfigureFetch:
    identity = "fetch-like-factory@1"

    def __call__(self, metadata):
        return FetchLike()


def fetched_generation(source, directory):
    cache = Cache((directory / "fetched").as_uri())
    fetch = FetchLike()
    manifest = Manifest(source.fingerprint(), (StepIdentity.of(fetch),))
    stored = cache.mapped(grain.MapDataset.source(source.records("indexed")), manifest, source,
                          transform=lambda index, row: (fetch(row),))
    try:
        for index in range(len(stored)):
            stored[index]
        stored.finish()
    finally:
        stored.close()
    return cache.publish(manifest, source)


@pytest.mark.parametrize("configured", [False, True])
def test_eager_fetch_keeps_a_memory_plan_volatile_until_its_generation_is_sealed(tmp_path, configured):
    """An eager fetch still needs one reader per share; a sealed generation does not."""
    source = Memory(rows("fetched", 16))
    chain = (source.map(Configured(ConfigureFetch(), needs=())) if configured
             else source.filter_map(FetchLike()))
    data = compile(chain.batch(2), read="memory", loading=READ)
    assert data.plan.volatile
    assert data.records == 10
    reader = data.train(DataPartition())
    try:
        batch = next(reader)
        np.testing.assert_array_equal(batch["i"], [1, 2])
        np.testing.assert_array_equal(batch["payload"], np.repeat(batch["i"][:, None], 4, axis=1))
    finally:
        reader.close()
    with pytest.raises(ValueError):
        data.train(DataPartition(readers=2))
    generation = fetched_generation(source, tmp_path)
    sealed = compile(chain.cache(generation).batch(2), read="memory", loading=READ)
    assert not sealed.plan.volatile
    readers = [sealed.train(DataPartition(readers=2, reader=index)) for index in range(2)]
    try:
        first, second = (next(reader) for reader in readers)
        for field in ("i", "payload"):
            np.testing.assert_array_equal(first[field], second[field])
    finally:
        for reader in readers:
            reader.close()


@pytest.mark.parametrize(("selection", "read"), [("fraction", "memory"), ("fraction", "streamed"),
                                               ("records", "memory"), ("records", "indexed"),
                                               ("random", "memory"), ("random", "indexed")])
def test_a_sealed_generation_keeps_its_holdout_selection_in_training_and_validation(tmp_path,
                                                                                 selection, read):
    """Generation promotion splits its accepted rows, not the original inputs or the full generation."""
    from dew.config.data import RecordsData, loader
    from dew.data.holdout import HoldOut

    source = Memory(rows("fetched", 64))
    generation = fetched_generation(source, tmp_path)
    if selection == "random":
        holdout = HoldOut(records=5, by="random", seed=37)
        train, held, _ = RecordsData(generation.source(), validation=holdout).pipelines()
        data = loader(RecordsData(source, validation=holdout), cache=generation,
                      batch=2, read=read, loading=READ)
    else:
        cached = source.filter_map(FetchLike()).cache(generation)
        train, held = (cached.holdout(0.25, key="i") if selection == "fraction"
                       else cached.holdout(records=5))
        data = compile(train.batch(2), validation=held.batch(2, drop_remainder=False),
                       read=read, loading=READ)
    accepted = {index for index in range(64) if index % 3}
    parts = [{int(row["i"]) for row in iterate(part, read=read, loading=READ)} for part in (train, held)]
    assert all(parts)
    assert parts[0].isdisjoint(parts[1])
    assert parts[0] | parts[1] == accepted
    if selection == "records":
        assert parts[1] == set(sorted(accepted)[:5])
    if selection == "random":
        assert len(parts[1]) == 5
    training, validation = data.train(DataPartition()), data.val(DataPartition())
    try:
        trained = {int(index) for batch in itertools.islice(training, len(accepted)) for index in batch["i"]}
        scored = {int(index) for batch in validation for index in batch["i"]}
        assert [trained, scored] == parts
    finally:
        training.close()
        validation.close()


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
        return set(zip(batch["origin"], batch["i"].tolist(), strict=True))

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
    with pytest.raises(ValueError):
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
    # The same step over groups of another size reads other records: that position is not its.
    regrouped = (Memory(rows("a", 30)).shuffle(4).repeat()
                 .flat_map(tokenized_in_batches, batched=True, batch_size=10).batch(5))
    other = compile(regrouped, read="indexed", loading=READ).train(DataPartition())
    try:
        with pytest.raises(ValueError):
            other.set_state(state)
    finally:
        other.close()


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
    # The state is a NumPy array, as a library's own state often holds, through the JSON checkpoint.
    for source in (from_iter(Countdown, name="countdown", state=lambda it: np.array([it.next]),
                             load_state=lambda it, saved: setattr(it, "next", int(saved[0]))),
                   from_iter(Countdown, name="replayed", replay=True)):
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


def test_an_iterator_without_a_position_or_a_declared_replay_does_not_resume():
    """A stateless iterator that declares no replay reads and reports how far
    it read, but refuses to resume, where one that declares either resumes
    (`test_an_iterator_resumes_through_its_own_state_or_by_replaying`)."""
    source = from_iter(Countdown, name="undeclared")
    reader = compile(source.repeat().batch(4), read="streamed", loading=READ).train(DataPartition())
    try:
        next(reader)
        state = reader.get_state()
    finally:
        reader.close()
    resumed = compile(source.repeat().batch(4), read="streamed", loading=READ).train(DataPartition())
    try:
        with pytest.raises(ValueError):
            resumed.set_state(state)
            next(resumed)
    finally:
        resumed.close()


def test_an_iterator_resuming_through_its_state_cannot_also_declare_a_replay():
    from_iter(Countdown, name="stateful", state=lambda it: 0, load_state=lambda it, saved: None)
    from_iter(Countdown, name="replayed", replay=True)
    with pytest.raises(ValueError):
        from_iter(Countdown, name="both", state=lambda it: 0, load_state=lambda it, saved: None, replay=True)


def test_a_hugging_face_stream_resumes_exactly_and_its_own_shuffle_is_refused():
    """An IterableDataset reads as one stream and resumes from its own state_dict, each
    opening at its own position; a stream that shuffles through Hugging Face's buffer, whose
    rows that state leaves out, is refused, also under a later map, and so is one with no name
    to tell it from another stream."""
    datasets = pytest.importorskip("datasets")
    stream = datasets.IterableDataset.from_generator(lambda: ({"i": i} for i in range(40)))
    source = from_hf(stream, name="counting@1")
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
    assert [row["i"] for row in source.iterate()] == list(range(40))
    with pytest.raises(ValueError):
        from_hf(stream.shuffle(seed=0, buffer_size=8).map(lambda row: row), name="shuffled@1")
    with pytest.raises(ValueError):
        from_hf(stream)


def hf_rows(shards: list[int]):
    """The records of a Hugging Face stream, written across several shards."""
    for shard in shards:
        yield from ({"i": 12 * shard + index} for index in range(12))


def hf_file_rows(shards: list[str]):
    """A Hugging Face stream over JSONL files that can go offline."""
    for shard in shards:
        with open(shard) as source:
            yield from map(json.loads, source)


@pytest.mark.parametrize("shard_count", [1, 3, 4])
@pytest.mark.parametrize("workers", [0, 2], ids=["workers0", "workers2"])
def test_hugging_face_stream_shares_are_disjoint_and_resume_exactly(shard_count, workers):
    """Physical shards and undersharded streams cover every record once across two shares,
    with independent openings and exact per-share resumes, also through real workers."""
    datasets = pytest.importorskip("datasets")
    stream = datasets.IterableDataset.from_generator(hf_rows, gen_kwargs={"shards": list(range(shard_count))})
    source = from_hf(stream, name=f"sharded-{shard_count}@1")
    chain = (source.map(lambda row: {**row, "twice": 2 * row["i"]}, identity="twice@1")
             .interleave(2).repeat(1).batch(2))
    data = compile(chain, read="streamed", loading=dataclasses.replace(READ, workers=workers))
    shares = []
    for index in range(2):
        partition = DataPartition(index, 2)
        reader = data.train(partition)
        try:
            first = [next(reader) for _ in range(3)]
            state = reader.get_state()
            rest = list(reader)
        finally:
            reader.close()
        batches = [*first, *rest]
        assert all(np.array_equal(batch["twice"], 2 * batch["i"]) for batch in batches)
        shares.append([int(value) for batch in batches for value in batch["i"]])
        resumed = data.train(partition)
        try:
            resumed.set_state(state)
            actual = list(resumed)
            assert len(actual) == len(rest)
            assert all(np.array_equal(left[field], right[field])
                       for left, right in zip(actual, rest, strict=True) for field in ("i", "twice"))
        finally:
            resumed.close()
        other = data.train(DataPartition(1 - index, 2))
        try:
            with pytest.raises(ValueError):
                other.set_state(state)
        finally:
            other.close()
    assert set(shares[0]).isdisjoint(shares[1])
    assert sorted([*shares[0], *shares[1]]) == list(range(12 * shard_count))
    whole = [row["i"] for row in iterate(source.chain(), read="streamed", loading=READ)]
    assert whole == list(range(12 * shard_count))


@pytest.mark.parametrize("shares", [1, 2])
def test_a_hugging_face_stream_cache_publishes_only_its_store_shards(tmp_path, shares):
    """A native-shard fill publishes; a row-wise node split refuses without reopening the store."""
    from dew.data.cache import Cache
    from dew.data.plan import Manifest
    from dew.data.shards import streamed

    datasets = pytest.importorskip("datasets")
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps({"i": index}) + "\n" for index in range(12)))
    stream = datasets.IterableDataset.from_generator(hf_file_rows, gen_kwargs={"shards": [str(path)]})
    source = from_hf(stream, name="cached-counting@1")
    cache, key = Cache((tmp_path / "cache").as_uri()), Manifest(source.fingerprint())
    manifest = source.manifest()
    cached = dataclasses.replace(manifest, reader=cache.streamed(manifest.reader, key, manifest))
    actual = []
    for index in range(shares):
        reader = iter(streamed(cached, partition=DataPartition(index, shares), loading=READ))
        try:
            actual.extend(row["i"] for row in reader)
        finally:
            reader.close()
    assert sorted(actual) == list(range(12))
    if shares > 1:
        path.rename(tmp_path / "offline.jsonl")
        with pytest.raises(ValueError):
            cache.publish(key, source)
        assert cache.generation(key) is None
    else:
        generation = cache.publish(key, source)
        assert [generation[index]["i"] for index in range(len(generation))] == list(range(12))


def test_chunks_cut_long_documents_keeping_every_target():
    pieces = list(Memory([{"text": np.arange(10)}]).flat_map(text.chunk(4, overlap=1)).iterate())
    assert [piece["text"].tolist() for piece in pieces] == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]]


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_diagnostic_records_unstack_nested_numeric_and_object_fields(read):
    records = [{"id": index, "nested": {"x": np.arange(3) + index,
                                        "deep": {"even": index % 2 == 0}},
                "objects": np.array(["label", {"turns": ["question", "answer"]}], dtype=object),
                "caption": f"record {index}"} for index in range(3)]
    chain = dataclasses.replace(Memory(records).read(read), loading=READ)
    actual = list(chain.iterate())
    assert len(actual) == len(records)
    for row, original in zip(actual, records, strict=True):
        assert row["id"] == original["id"]
        assert row["caption"] == original["caption"]
        np.testing.assert_array_equal(row["nested"]["x"], original["nested"]["x"])
        assert row["nested"]["deep"] == original["nested"]["deep"]
        np.testing.assert_array_equal(row["objects"], original["objects"])


@pytest.mark.parametrize("read", [None, "memory", "indexed", "streamed"])
def test_stats_bounds_preprocessing_and_counts_only_the_inspected_records(read):
    processed = []

    def prepare(row):
        processed.append(row["i"])
        return {**row, "twice": 2 * row["i"]}

    chain = dataclasses.replace(Memory(rows("a", 40)).named("sample"), mode=read, loading=READ)
    counts = chain.map(prepare, identity="stats-prepare@1").filter_map(drop_thirds).stats(10)
    assert counts["data/accepted/sample"] == 10
    assert counts["data/attempted/sample"] == 15
    assert counts["data/rejected/sample/third"] == 5
    assert processed == list(range(15))


@pytest.mark.parametrize("records", [0, 100])
def test_stats_stops_at_a_finite_sources_end(records):
    processed = []

    def prepare(row):
        processed.append(row["i"])
        return row

    chain = dataclasses.replace(Memory(rows("a", 7)).named("sample"), loading=READ)
    counts = chain.map(prepare, identity="stats-finite@1").filter_map(drop_thirds).stats(records)
    expected = 0 if records == 0 else 7
    assert processed == list(range(expected))
    assert counts.get("data/attempted/sample", 0) == expected
    assert counts.get("data/accepted/sample", 0) == (0 if records == 0 else 4)


@pytest.mark.parametrize("read", [None, "memory", "indexed", "streamed"])
def test_stats_counts_one_to_one_chains_lazily(read):
    processed = []

    def prepare(row):
        processed.append(row["i"])
        return {**row, "twice": 2 * row["i"]}

    chain = dataclasses.replace(Memory(rows("a", 7)).named("sample"), mode=read, loading=READ)
    counts = chain.map(prepare, identity="stats-one-to-one@1").stats(3)
    assert counts["data/attempted/sample"] == counts["data/accepted/sample"] == 3
    assert processed == [0, 1, 2]


def expand_record(row):
    if row["i"] % 3:
        yield {**row, "piece": 0}
        yield {**row, "piece": 1}


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_flat_map_reads_plain_record_functions_through_iteration_and_resume(read):
    chain = Memory(rows("a", 7)).flat_map(expand_record, identity="expand-record@1")
    chain = dataclasses.replace(chain, mode=read, loading=READ)
    expected = [(index, piece) for index in range(7) if index % 3 for piece in range(2)]
    assert [(int(row["i"]), int(row["piece"])) for row in chain.iterate()] == expected
    data = compile(chain.repeat(1).batch(2), loading=READ)
    reader = data.train(DataPartition())
    try:
        first = next(reader)
        saved = reader.get_state()
        rest = list(reader)
    finally:
        reader.close()
    assert list(zip(first["i"].tolist(), first["piece"].tolist(), strict=True)) == expected[:2]
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(saved)
        actual = list(resumed)
    finally:
        resumed.close()
    assert len(actual) == len(rest)
    for left, right in zip(actual, rest, strict=True):
        for name in ("i", "piece"):
            np.testing.assert_array_equal(left[name], right[name])


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_finite_diagnostics_draw_after_rejection_and_shuffle_as_training_does(read):
    chain = Memory([{**row, "value": row["i"]} for row in rows("a", 7)])
    chain = chain.shuffle(13, **({"window": 3} if read == "streamed" else {}))
    chain = chain.filter_map(drop_thirds).random_map(add_draw)
    chain = dataclasses.replace(chain, mode=read, loading=READ)
    actual = list(chain.iterate())
    reader = compile(chain.repeat(1).batch(1, drop_remainder=False), loading=READ).train(DataPartition())
    try:
        reference = list(reader)
    finally:
        reader.close()
    assert len(actual) == len(reference) == 4
    for row, batch in zip(actual, reference, strict=True):
        for name in ("i", "draw", "value"):
            np.testing.assert_array_equal(row[name], batch[name][0])
        assert row["value"] == row["i"] + row["draw"]


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_an_augmented_image_preset_can_be_inspected_without_repeating(read):
    from dew.data import presets
    from dew.data.compile import fitted

    records = [{"image": np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3) + index,
                "id": index} for index in range(4)]
    chain = Memory(records).read(read).pipe(presets.images, 4, batch=2, augment=True)
    chain = dataclasses.replace(chain, loading=READ)
    inspected = list(chain.iterate())
    data, _ = fitted(chain, None, objective=ObjectiveData())
    reader = data.train(DataPartition())
    try:
        training = [next(reader) for _ in range(2)]
    finally:
        reader.close()
    assert len(inspected) == 2
    for actual, batch in zip(inspected, training, strict=True):
        expected = batch
        for step in data.device:
            expected = step(expected)
        np.testing.assert_array_equal(actual["id"], expected["id"])
        np.testing.assert_array_equal(actual["image"], expected["image"])


@pytest.mark.parametrize("consumer", ["iterate", "training"])
def test_host_maps_refuse_undeclared_rejections_before_the_next_stage(consumer):
    seen = []

    def reject(batch):
        return Rejected("bad_batch")

    def following(batch):
        seen.append(batch)
        return batch

    chain = Memory(rows("a", 4)).read("indexed").repeat(1).batch(2).map(reject, following)
    chain = dataclasses.replace(chain, loading=READ)
    reader = (chain.iterate() if consumer == "iterate"
              else compile(chain, loading=READ).train(DataPartition()))
    try:
        with pytest.raises(ValueError):
            next(reader)
    finally:
        reader.close()
    assert seen == []


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_column_verbs_preserve_simultaneous_renames_and_selection(read):
    records = [{"a": index, "b": 10 + index, "unused": -index} for index in range(3)]
    chain = (Memory(records).read(read).rename_columns({"a": "b", "b": "a"})
             .remove_columns("unused", "absent").select_columns("b", "a"))
    chain = dataclasses.replace(chain, loading=READ)
    actual = list(chain.iterate())
    assert [{name: int(value) for name, value in row.items()} for row in actual] == [
        {"b": index, "a": 10 + index} for index in range(3)]
