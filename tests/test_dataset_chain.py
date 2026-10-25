"""The chain surface: stores carry the verbs, mix reads members of every kind, holdout splits by identity."""

import dataclasses
import hashlib
import itertools
import json
from dataclasses import dataclass

import grain.python as grain
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew.data import Dataset, Files, Loading, Memory, from_grain, from_hf, from_iter, mix, text
from dew.data.cache import Cache
from dew.data.compile import compile, iterate
from dew.data.dataset import DataPartition, GlobalStream
from dew.data.pipeline import Configured, ObjectiveData, Rejected
from dew.data.plan import Manifest

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
    # A mix places each member's records at its exact share of every prefix of the order, 2:1:1 here.
    counts = {origin: int(np.sum(origins == origin)) for origin in ("memory", "grain", "files")}
    assert counts == {"memory": 128, "grain": 64, "files": 64}


@pytest.fixture
def grain_rows():
    return grain.MapDataset.source(rows("filtered", 16))


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("partition", [DataPartition(), DataPartition(1, 2)])
def test_filtered_grain_roots_refuse_global_record_count_batches(grain_rows, mixed, partition):
    filtered = grain_rows.filter(lambda row: row["i"] % 3 != 0)
    member = from_grain(filtered.shuffle(1), "filtered@1").repeat()
    chain = mix([member, member]) if mixed else member
    data = compile(chain.batch(4), read="indexed", loading=READ)
    assert data.plan is not None and data.plan.lowering == "global"
    reader = data.train(partition)
    assert isinstance(reader, GlobalStream)
    try:
        with pytest.raises(ValueError):
            list(itertools.islice(reader, 8))
    finally:
        reader.close()


@pytest.mark.parametrize("mixed", [False, True])
def test_filtered_memory_roots_compact_before_global_counts_and_share_cuts(grain_rows, mixed):
    filtered = grain_rows.filter(lambda row: row["i"] % 3 != 0)
    member = Memory(filtered, identity="filtered@1").repeat()
    chain = mix([member, member]) if mixed else member
    data = compile(chain.batch(4), read="indexed", loading=READ)
    whole, resumed = data.train(DataPartition()), data.train(DataPartition())
    assert isinstance(whole, GlobalStream) and isinstance(resumed, GlobalStream)
    shares = [data.train(DataPartition(index, 2)) for index in range(2)]
    try:
        first = next(whole)["i"]
        saved = whole.get_state()
        expected = next(whole)["i"]
        resumed.set_state(saved)
        np.testing.assert_array_equal(next(resumed)["i"], expected)
        interleaved = np.stack([next(reader)["i"] for reader in shares], axis=-1).reshape(-1)
        np.testing.assert_array_equal(interleaved, first)
        assert np.all(first % 3 != 0) and np.all(expected % 3 != 0)
    finally:
        for reader in (whole, resumed, *shares):
            assert isinstance(reader, GlobalStream)
            reader.close()


HOLDOUT_SHARES = (0.85, 0.05, 0.1)
"""The 3-way holdout's requested shares: train, validation and test."""

HOLDOUT_RECORDS = 500
"""The records of each drawn corpus the holdout's shares are read over: as many as hold a part
favoured by 0.02 of the records over check 1's PARITY of an independent draw's error (5,000 records read
whole in one corpus held it to 0.02 by hand)."""

HOLDOUT_SEEDS = 256
"""The holdout-shares site's declared draws (`seeds_for`, the drawn split against itself)."""


def test_a_holdouts_parts_take_their_shares_as_independently_drawn_parts_would():
    """Over drawn corpora (their identities a seed's own), a 3-way holdout's parts take shares on par
    with each record's part drawn independently from the requested shares (check 1), both measured
    from the requested shares themselves: a record's identity hashes it into its part as a uniform
    draw would, so no part is favoured nor its count spread wider."""

    def split(seed):
        corpus = Memory(rows(f"s{int(seed)}", HOLDOUT_RECORDS))
        parts = corpus.holdout(val=HOLDOUT_SHARES[1], test=HOLDOUT_SHARES[2])
        counts = [sum(1 for _ in iterate(part, read="memory", loading=READ)) for part in parts]
        return {"shares": np.asarray(counts, np.float64) / HOLDOUT_RECORDS}

    def drawn(seed):
        counts = np.random.default_rng([0x5A11, int(seed)]).multinomial(HOLDOUT_RECORDS, HOLDOUT_SHARES)
        return {"shares": counts / HOLDOUT_RECORDS}

    assert_as_exact(split, drawn, lambda seed: (np.int64(seed),), "holdout shares", seeds=HOLDOUT_SEEDS,
                    truth_fn=lambda seed: {"shares": np.asarray(HOLDOUT_SHARES, np.float64)})


def test_holdout_puts_each_record_in_one_part_by_its_identity_the_same_in_every_read():
    """A 3-way holdout: the parts are disjoint and together every record, the same whether the store
    is read whole or streamed, with the steps replayed on each (their shares,
    `test_a_holdouts_parts_take_their_shares_as_independently_drawn_parts_would`)."""
    corpus = Memory(rows("a", 400)).map(lambda row: {**row, "seen": True})
    train, val, test = corpus.holdout(val=0.05, test=0.1)

    def keys(part, read):
        return {(row["origin"], int(row["i"])) for row in iterate(part, read=read, loading=READ)
                if row["seen"]}

    parts = [keys(part, "memory") for part in (train, val, test)]
    assert [keys(part, "streamed") for part in (train, val, test)] == parts
    assert sum(map(len, parts)) == len(set().union(*parts)) == 400
    # With a key, a document's other versions land in its part: the split follows "i" alone.
    revised = Memory([{**row, "origin": "a2"} for row in rows("a", 400)])
    revised = revised.map(lambda row: {**row, "seen": True})
    by_key = [{i for _, i in keys(part, "memory")} for part in corpus.holdout(0.1, key="i")]
    assert by_key == [{i for _, i in keys(part, "memory")} for part in revised.holdout(0.1, key="i")]
    first, held = Memory(rows("c", 10)).holdout(records=3)
    assert [int(row["i"]) for row in iterate(held, read="memory")] == [0, 1, 2]
    assert [int(row["i"]) for row in iterate(first, read="memory")] == list(range(3, 10))


@dataclasses.dataclass(frozen=True)
class SplitMemory(Memory):
    """Records in memory with named splits of their own beside them, as a Hub repository holds."""

    splits: tuple[tuple[str, tuple], ...] = ()

    def with_split(self, name: str) -> Memory:
        return Memory(dict(self.splits)[name], identity=f"{self.identity}:{name}")


def test_holdout_returns_each_of_the_named_splits_a_store_holds_after_its_own():
    """`holdout(split=("validation", "test"))` returns the store's default records, then each
    named split in order, with the steps replayed on each."""
    store = SplitMemory(rows("train", 3), identity="splits", splits=(
        ("validation", tuple(rows("validation", 2))), ("test", tuple(rows("test", 1)))))
    parts = store.map(lambda row: {**row, "seen": True}).holdout(split=("validation", "test"))
    assert [[(row["origin"], int(row["i"])) for row in iterate(part, read="memory") if row["seen"]]
            for part in parts] == [[("train", 0), ("train", 1), ("train", 2)],
                                   [("validation", 0), ("validation", 1)], [("test", 0)]]
    assert len(store.holdout(split="test")) == 2
    with pytest.raises(ValueError):
        store.holdout(split=("test", "test"))


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


def fetched_generation(chain, directory):
    """The fetch outcomes of `chain`, sealed as the generation of its prefix: the keyed
    manifest publication records, filled once through the cache."""
    from dew.data.analysis import _cache_manifest
    from dew.data.compile import _prepared

    (prepared,) = _prepared((chain,), "indexed", provision=False)
    source = prepared.source
    cache = Cache((directory / "fetched").as_uri())
    fetch = FetchLike()
    manifest = _cache_manifest(source, prepared.stages, "indexed")
    stored = cache.mapped(grain.MapDataset.source(source.records("indexed")), manifest,
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
    generation = fetched_generation(chain, tmp_path)
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
                                               ("records", "memory"), ("records", "indexed")])
def test_a_sealed_generation_keeps_its_holdout_selection_in_training_and_validation(tmp_path,
                                                                                 selection, read):
    """A holdout of a chain whose steps a published generation stands for splits the store's inputs,
    as it does uncached, and reads the generation's rows of the inputs each part keeps: the first
    five inputs held out are the accepted ones among them, not the generation's first five rows."""
    source = Memory(rows("fetched", 64))
    generation = fetched_generation(source.filter_map(FetchLike()), tmp_path)
    cached = source.filter_map(FetchLike()).cache(generation)
    train, held = cached.holdout(0.25, key="i") if selection == "fraction" else cached.holdout(records=5)
    data = compile(train.batch(2), validation=held.batch(2, drop_remainder=False), read=read, loading=READ)
    accepted = {index for index in range(64) if index % 3}
    parts = [{int(row["i"]) for row in iterate(part, read=read, loading=READ)} for part in (train, held)]
    assert all(parts)
    assert parts[0].isdisjoint(parts[1])
    assert parts[0] | parts[1] == accepted
    if selection == "records":
        assert parts[1] == {index for index in range(5) if index % 3}
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
    # Three-token records carry two targets, nine-token records eight. The member furthest below its
    # share of the targets goes next, ties to the first: one short record, then a long one and four
    # short ones in turn, so 1 + 63 * 4 + 3 of the 320 records are short.
    assert int(np.sum(origins == "short")) == 256


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


def test_a_replayed_stream_checks_the_model_inputs_its_batch_holds():
    """A replayed stream (`replayed`) is guarded at its position by a digest of the batch there,
    which reads a Dew dataclass's fields as the codec does: a stream whose model inputs there
    changed, and nothing else, is refused, and one that replays them resumes after them."""
    from dew.data.dataset import Replayed
    from dew.nn.inputs import ModelInputs

    def opened(offset):
        def batches():
            return iter([{"inputs": ModelInputs(tokens=np.arange(4, dtype=np.int32) + offset + index,
                                                conditioning={"mask": np.ones(4, bool)})}
                         for index in range(3)])
        return batches

    stopped = Replayed(opened(0))
    next(stopped), next(stopped)
    saved = stopped.get_state()
    with pytest.raises(ValueError):
        Replayed(opened(10)).set_state(saved)
    resumed = Replayed(opened(0))
    resumed.set_state(saved)
    np.testing.assert_array_equal(next(resumed)["inputs"].tokens, np.arange(4) + 2)


FIRST, SECOND = float("nan"), float("nan")
"""Two keys of one repr that a mapping holds apart."""


@pytest.mark.parametrize(("saved", "replayed", "same"), [
    ({1: np.zeros(2)}, {"1": np.zeros(2)}, False),
    ({FIRST: 1, SECOND: 2}, {FIRST: 5, SECOND: 2}, False),
    ({"phase": 1 + 2j}, {"phase": 9 + 8j}, False),
    ({"pixels": np.arange(6.).reshape(2, 3)},
     {"pixels": np.asfortranarray(np.arange(6.).reshape(2, 3))}, True),
], ids=["key-type", "same-repr-keys", "complex", "storage-order"])
def test_a_replayed_stream_reads_its_batch_by_content(saved, replayed, same):
    """A replayed stream's digest reads what its batch holds: a mapping's every key as itself, a
    complex scalar by value and an array by its elements, in whichever order it is stored. A
    stream whose batch changed at the saved position is refused; one that holds the same content
    resumes."""
    from dew.data.dataset import Replayed

    stopped = Replayed(lambda: iter([saved, saved]))
    next(stopped)
    state = stopped.get_state()
    resumed = Replayed(lambda: iter([replayed, replayed]))
    if same:
        resumed.set_state(state)
        assert next(resumed).keys() == replayed.keys()
    else:
        with pytest.raises(ValueError):
            resumed.set_state(state)


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
    import datasets

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
    assert [row["i"] for row in source] == list(range(40))
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
    import datasets

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
    whole = [row["i"] for row in iterate(source._chain(), read="streamed", loading=READ)]
    assert whole == list(range(12 * shard_count))


@pytest.mark.parametrize("shares", [1, 2])
def test_a_hugging_face_stream_cache_publishes_only_its_store_shards(tmp_path, shares):
    """A native-shard fill publishes; a row-wise node split refuses without reopening the store."""
    import datasets

    from dew.data.cache import Cache
    from dew.data.shards import streamed

    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps({"i": index}) + "\n" for index in range(12)))
    stream = datasets.IterableDataset.from_generator(hf_file_rows, gen_kwargs={"shards": [str(path)]})
    source = from_hf(stream, name="cached-counting@1")
    cache, key = Cache((tmp_path / "cache").as_uri()), Manifest(source.fingerprint())
    manifest = source.manifest()
    cached = dataclasses.replace(manifest, reader=cache.streamed(manifest.reader, key))
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


@pytest.mark.parametrize("form", ["NFKC", "NFC"])
def test_text_normalize_reads_composed_and_compatibility_characters_alike(form):
    """text.Normalized puts a record's text into one Unicode form, so a decomposed accent reads as
    its composed letter and, under NFKC, a ligature and a full-width digit as their plain ones;
    the record's other fields pass through."""
    records = [{"text": "cafe\u0301 \ufb01ne \uff11", "id": 7}, {"text": b"plain", "id": 8}]
    normalized = list(Memory(records).map(text.Normalized(form)))
    assert [row["text"] for row in normalized] == [
        "caf\u00e9 fine 1" if form == "NFKC" else "caf\u00e9 \ufb01ne \uff11", "plain"]
    assert [int(row["id"]) for row in normalized] == [7, 8]


def test_chunks_cut_long_documents_keeping_every_target():
    pieces = list(Memory([{"text": np.arange(10)}]).flat_map(text.Chunks(4, overlap=1)))
    assert [piece["text"].tolist() for piece in pieces] == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]]


def test_a_mix_counts_each_corpus_under_a_name_of_its_own(tmp_path):
    """Two unnamed stores share a file name and a third is named as one of theirs would be
    numbered: each corpus keeps its own counts, the two numbered past the name the third holds."""
    for part in ("first", "second"):
        (tmp_path / part).mkdir()
        (tmp_path / part / "train.jsonl").write_text("".join(f'{{"id": {index}}}\n' for index in range(8)))
    corpora = [Files(str(tmp_path / part / "train.jsonl")) for part in ("first", "second")]
    named = Memory([{"id": index} for index in range(8)]).named("train-1")
    counts = mix([*corpora, named]).stats(records=6)
    assert {name for name in counts if name.startswith("data/rows/")} == {
        "data/rows/train-1", "data/rows/train-2", "data/rows/train-3"}
    assert sum(counts[f"data/rows/train-{place}"] for place in (1, 2, 3)) == 6


@pytest.mark.parametrize("read", ["indexed", "streamed"])
def test_a_mixtures_grain_step_drops_rows_it_rejects_and_counts_them(read):
    """A Grain step after a mix that rejects rows (`Rejected`) has them dropped before the batch
    and counted under the mixture, beside its corpora's own counts."""
    from dew.data.pipeline import Rejected

    def odd_rejected(rows):
        return rows.map(lambda row: Rejected("odd") if int(row["i"]) % 2 else row)

    mixed = mix([Memory(rows("a", 8)).named("a"), Memory(rows("b", 8)).named("b")])
    chain = mixed.grain(odd_rejected, identity="odd-rejected@1").batch(2)
    data = compile(chain, read=read, loading=READ)
    stream = data.train(DataPartition())
    try:
        batches = [next(stream) for _ in range(3)]
    finally:
        stream.close()
    assert all(int(value) % 2 == 0 for batch in batches for value in batch["i"])
    assert data.statistics()["data/rejected/mixture/odd"] > 0


@dataclasses.dataclass(frozen=True)
class Tagged:
    """Marks a record as read from the second corpus."""

    def __call__(self, row):
        return {**row, "tagged": True}


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_stats_reports_the_records_a_second_tokens_and_padding_of_packed_rows(read):
    """Token documents of differing lengths packed into rows of 16: stats reports the tokens
    the rows hold, all the documents' (one of them id 0, a token like any other), the share of
    the rows' places that pad them, and the records it read a second; rows with no token ids
    report none of those two."""
    lengths = [5, 7, 3, 9, 11, 2, 6, 8]
    documents = Memory([{"text": np.arange(length, dtype=np.int32)} for length in lengths]).named("docs")
    counts = documents.pack(16).read(read).loading(READ).stats(records=100)
    packed = counts["data/rows/docs"]
    assert counts["data/tokens"] == sum(lengths)
    assert counts["data/padding"] == (16 * packed - sum(lengths)) / (16 * packed)
    assert 0 < counts["data/records_per_second"] < float("inf")
    plain = Memory(rows("a", 4)).named("plain").read(read).loading(READ).stats(records=4)
    assert "data/tokens" not in plain and "data/padding" not in plain
    # Rows stored packed, their documents and validity beside them: a place of segment 0 or
    # marked invalid pads its row.
    stored = Memory({"text": [np.arange(9, dtype=np.int32)] * 2,
                     "text_segment_ids": [np.asarray([1, 1, 1, 2, 2, 2, 2, 2, 0], np.int32)] * 2,
                     "text_attention_mask": [np.asarray([1] * 8 + [0], bool),
                                             np.asarray([1] * 7 + [0, 0], bool)]}).named("stored")
    held = stored.read(read).loading(READ).stats(records=2)
    assert held["data/tokens"] == 8 + 7
    assert held["data/padding"] == (1 + 2) / 18


@dataclasses.dataclass(frozen=True)
class Twice:
    """Each record of a group twice, as columns."""

    def __call__(self, columns):
        return {name: [value for value in values for _ in range(2)] for name, values in columns.items()}


@dataclasses.dataclass(frozen=True)
class Keep:
    """Keeps every record, though it may drop one."""

    rejects = True

    def __call__(self, row):
        return row


def test_show_prints_the_chain_as_written_then_the_records_and_leaves_a_reader_where_it_was(capsys):
    """`show` prints the store and each verb as the chain wrote it, a preset's expanded steps and
    a verb that drops or expands records under its own name, then the first records as a reader
    makes them; a reader open beside it reads on as one never shown would."""
    from dew.data import presets

    pictures = Memory([{"image": np.full((8, 8, 3), index, np.uint8), "label": index % 2}
                       for index in range(6)])
    chain = pictures.pipe(presets.images, 4, batch=2, augment=True).loading(READ)
    data = compile(chain, read="indexed", loading=READ)
    reader, untouched = data.train(DataPartition()), data.train(DataPartition())
    try:
        next(reader)
        chain.show(count=1)
        rest, expected = next(reader), (next(untouched), next(untouched))[1]
        for name in expected:
            np.testing.assert_array_equal(np.asarray(rest[name]), np.asarray(expected[name]))
    finally:
        reader.close()
        untouched.close()
    lines = capsys.readouterr().out.splitlines()
    written = [line.strip() for line in lines[:6]]
    assert written[0].startswith("Memory")
    assert [line.split("(")[0] for line in written[1:]] == [
        ".filter_map", ".map", ".random_map", ".batch", ".map"]
    assert "DecodeImage" in written[1] and "Resize" in written[2] and "AugmentationKey" in written[3]
    assert written[4] == ".batch(2)" and "DeviceAugment" in written[5] and "device" in written[5]
    assert "uint8[2, 4, 4, 3]" in "".join(lines[6:])

    grouped = Memory(rows("a", 4)).named("a").flat_map(Twice(), batched=True, batch_size=2)
    grouped.filter_map(Keep()).loading(READ).show(count=1)
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].strip() == ".flat_map(Twice(), batched=True, batch_size=2)"
    assert lines[2].strip().startswith(".filter_map(Keep")


def test_a_runs_truncated_share_is_of_the_rows_it_delivered():
    """A memory read runs a corpus's steps before its repeat and reads the rest as it goes: every
    row it delivers was cut short after the repeat, so its truncated share is one, whatever the
    rows the memory read held before."""
    from dew.data.steps.text import LengthPolicy

    rows = Memory([{"text": np.arange(5, dtype=np.int32)} for _ in range(4)])
    chain = (rows.repeat(1).map(LengthPolicy(2)).filter(lambda row: True, identity="all@1")
             .take(1).batch(1))
    data = compile(chain, read="memory", loading=Loading(workers=0, threads=1, read_buffer=1))
    stream = data.train(DataPartition())
    try:
        assert np.asarray(next(stream)["text"]).tolist() == [[0, 1]]
    finally:
        stream.close()
    counts = data.statistics()
    assert counts["data/rows/memory"] == 1
    assert counts["data/truncated/memory"] == 1.0


def test_a_character_filter_and_caption_dropout_keep_and_empty_their_records():
    """text.MinChars keeps the texts of at least its count of characters, and text.Dropped empties
    a kept record's caption by the record's own draw: none at p=0, every one at p=1, and at any p
    the same captions on every pass."""
    records = [{"text": "x" * size, "caption": f"a photo {size}", "id": size} for size in range(12)]

    def captions(p):
        chain = Memory(records).filter(text.MinChars(5)).random_map(text.Dropped(p=p))
        return {int(row["id"]): str(row["caption"]) for row in chain}

    assert captions(0.0) == {size: f"a photo {size}" for size in range(5, 12)}
    assert captions(1.0) == dict.fromkeys(range(5, 12), "")
    halved = captions(0.5)
    assert halved == captions(0.5)
    assert all(caption in ("", f"a photo {size}") for size, caption in halved.items())


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_diagnostic_records_unstack_nested_numeric_and_object_fields(read):
    records = [{"id": index, "nested": {"x": np.arange(3) + index,
                                        "deep": {"even": index % 2 == 0}},
                "objects": np.array(["label", {"turns": ["question", "answer"]}], dtype=object),
                "caption": f"record {index}"} for index in range(3)]
    chain = Memory(records).read(read).loading(READ)
    actual = list(chain)
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

    chain = Memory(rows("a", 40)).named("sample").read(read).loading(READ)
    counts = chain.map(prepare, identity="stats-prepare@1").filter_map(drop_thirds).stats(10)
    assert counts["data/rows/sample"] == 10
    assert counts["data/rows/sample"] + counts["data/rejected/sample/third"] == 15
    assert counts["data/rejected/sample/third"] == 5
    assert processed == list(range(15))


@pytest.mark.parametrize("records", [0, 100])
def test_stats_stops_at_a_finite_sources_end(records):
    processed = []

    def prepare(row):
        processed.append(row["i"])
        return row

    chain = Memory(rows("a", 7)).named("sample").loading(READ)
    counts = chain.map(prepare, identity="stats-finite@1").filter_map(drop_thirds).stats(records)
    expected = 0 if records == 0 else 7
    assert processed == list(range(expected))
    assert counts.get("data/rows/sample", 0) + counts.get("data/rejected/sample/third", 0) == expected
    assert counts.get("data/rows/sample", 0) == (0 if records == 0 else 4)


@pytest.mark.parametrize("read", [None, "memory", "indexed", "streamed"])
def test_stats_counts_one_to_one_chains_lazily(read):
    processed = []

    def prepare(row):
        processed.append(row["i"])
        return {**row, "twice": 2 * row["i"]}

    chain = Memory(rows("a", 7)).named("sample").read(read).loading(READ)
    counts = chain.map(prepare, identity="stats-one-to-one@1").stats(3)
    assert counts["data/rows/sample"] == 3
    assert processed == [0, 1, 2]


def expand_record(row):
    if row["i"] % 3:
        yield {**row, "piece": 0}
        yield {**row, "piece": 1}


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_flat_map_reads_plain_record_functions_through_iteration_and_resume(read):
    chain = Memory(rows("a", 7)).flat_map(expand_record, identity="expand-record@1")
    chain = chain.read(read).loading(READ)
    expected = [(index, piece) for index in range(7) if index % 3 for piece in range(2)]
    assert [(int(row["i"]), int(row["piece"])) for row in chain] == expected
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
    chain = chain.read(read).loading(READ)
    actual = list(chain)
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


def keep_heads(row, rng) -> dict | Rejected:
    """The row with its draw, or `Rejected("tails")` when the draw lands below a half."""
    draw = rng.random()
    return {**row, "draw": np.float64(draw)} if draw >= 0.5 else Rejected("tails")


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_a_random_step_that_drops_records_draws_alike_on_every_read_and_counts_its_drops(read):
    """A plain `fn(record, rng)` that may reject goes through random_filter_map: each read keeps
    the records training keeps, with training's draws, and counts the rest under the step's
    reason, where a random step that drops once had a verb only as a Grain transform."""
    chain = Memory(rows("a", 24)).named("coins").random_filter_map(keep_heads)
    chain = chain.read(read).loading(READ)
    kept = [(int(row["i"]), float(row["draw"])) for row in chain]
    reader = compile(chain.repeat(1).batch(1, drop_remainder=False), loading=READ).train(DataPartition())
    try:
        trained = [(int(batch["i"][0]), float(batch["draw"][0])) for batch in reader]
    finally:
        reader.close()
    assert kept == trained
    assert 0 < len(kept) < 24 and all(draw >= 0.5 for _, draw in kept)
    assert chain.stats(records=24)["data/rejected/coins/tails"] == 24 - len(kept)


def may_drop(row, rng) -> dict | Rejected:
    """Draws as `keep_heads` does and keeps every record, though it says it may drop one."""
    return {**row, "draw": np.float64(rng.random())}


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_a_random_step_that_may_drop_draws_as_one_that_cannot(read):
    """A draw is keyed by the record's place in the run's order: a random step that may reject
    draws what random_map draws, in training's endless reader and in an inspection alike, where
    the share a dropping step reads once sat between the step and its seed and moved its draws."""

    def draws(kind: str) -> tuple[list[float], list[float]]:
        chain = getattr(Memory(rows("a", 8)), kind)(may_drop)
        reader = compile(chain.batch(1), read=read, loading=READ).train(DataPartition())
        try:
            trained = [float(batch["draw"][0]) for batch in itertools.islice(reader, 8)]
        finally:
            reader.close()
        inspected = iter(chain.read(read).loading(READ))
        return trained, [float(row["draw"]) for row in inspected]

    expected, inspected = draws("random_map")
    assert inspected == expected
    assert draws("random_filter_map") == (expected, expected)


class Coin(grain.RandomMapTransform):
    """Keeps a record, with its draw, on heads; it says it drops records."""

    rejects = True

    def random_map(self, element, rng) -> dict | Rejected:
        return keep_heads(element, rng)


def test_an_objective_s_random_step_that_drops_records_runs_in_its_training_branch():
    """An objective's training step that draws and may reject runs as random_filter_map does,
    where it once reached filter_map, which now refuses a step that draws."""
    records = Memory(rows("a", 24)).repeat(1).batch(1, drop_remainder=False)
    data = compile(records, read="memory", loading=READ, objective=ObjectiveData(train=(Coin(),)))
    reader = data.train(DataPartition())
    try:
        draws = [float(batch["draw"][0]) for batch in reader]
    finally:
        reader.close()
    assert 0 < len(draws) < 24 and all(draw >= 0.5 for draw in draws)


def test_filter_map_refuses_a_step_that_draws():
    """A Grain RandomMapTransform draws from a generator, so filter_map, whose steps read the
    record alone, refuses it rather than call it without one."""
    from dew.data import video

    with pytest.raises(TypeError):
        Memory(rows("a", 2)).filter_map(video.PickClip(1.0))


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_an_augmented_image_preset_can_be_inspected_without_repeating(read):
    from dew.data import presets
    from dew.data.compile import fitted

    records = [{"image": np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3) + index,
                "id": index} for index in range(4)]
    chain = Memory(records).read(read).pipe(presets.images, 4, batch=2, augment=True)
    chain = chain.loading(READ)
    inspected = list(chain)
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
    chain = chain.loading(READ)
    reader = (iter(chain) if consumer == "iterate"
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
    chain = chain.loading(READ)
    actual = list(chain)
    assert [{name: int(value) for name, value in row.items()} for row in actual] == [
        {"b": index, "a": 10 + index} for index in range(3)]


def _stamped(facts):
    """A run-scoped step that writes the run's row count into each record."""
    count = facts["rows"]
    return lambda row: {**row, "rows": np.int32(count)}


@pytest.mark.parametrize("sizes", [(1, 2, 3), (3, 1, 2), (1, 2, 1)])
def test_a_fact_two_corpora_state_differently_reaches_no_run_step_whatever_another_states(sizes):
    """A run-scoped step reads only facts every corpus shares: corpora of 1, 2 and 3 records
    share no row count, in any order, as two of them do not, and a third that agrees with
    the first does not make the count shared again."""
    members = [Memory([{"x": np.int32(index)}] * size, identity=f"corpus {index}")
               for index, size in enumerate(sizes)]
    with pytest.raises(ValueError):
        compile(mix(members).map(Configured(_stamped, needs=("rows",), scope="run")).batch(1),
                read="indexed", loading=Loading(workers=0))


class _Dropping(grain.MapTransform):
    """Keeps even ids; says it drops the rest."""

    rejects = True

    def map(self, element):
        from dew.data.pipeline import Rejected

        return element if int(element["id"]) % 2 == 0 else Rejected("odd")


class _Doubled:
    """Each record twice."""

    def flat_map(self, element):
        return [element, element]


class _Noised(grain.RandomMapTransform):
    """Adds a draw from the record's generator."""

    def random_map(self, element, rng):
        return {**element, "noise": int(rng.integers(0, 1000))}


@pytest.mark.parametrize("read", ["memory", "indexed"])
def test_an_objectives_record_steps_read_as_the_verbs_that_name_what_they_are(read):
    """An objective's record steps, a dropping one, an expanding one and a drawing one, train on what
    the verbs for what they are, `filter_map`, `flat_map` and `random_map`, written where the objective
    reads the records, make of them: before their batch, a pass of 8 kept and doubled in 4 batches."""
    from dew.data.pipeline import ObjectiveData, Pipeline

    rows = Pipeline(Memory([{"id": index} for index in range(16)]))
    steps = (_Dropping(), _Doubled(), _Noised())
    written = rows.filter_map(steps[0]).flat_map(steps[1]).random_map(steps[2]).batch(4)
    expected = compile(written, read=read, loading=READ, seed=5).train(DataPartition())
    trained = compile(rows.batch(4), objective=ObjectiveData(train=steps), read=read, loading=READ,
                      seed=5).train(DataPartition())
    try:
        for batch, reference in zip(itertools.islice(trained, 4), itertools.islice(expected, 4), strict=True):
            for field in ("id", "noise"):
                np.testing.assert_array_equal(batch[field], reference[field])
    finally:
        expected.close()
        trained.close()


def test_the_defaults_a_verb_and_its_stage_take_are_the_same_value():
    """A verb written without its settings and the stage it makes written without them are one:
    the same stage, so the same plan."""
    from dew.data.pipeline import Interleave, Pack, Pipeline, Prefetch, Windows, Workers

    rows = Pipeline(Memory([{"text": np.arange(4)}]))
    assert rows.interleave().stages[-1] == Interleave()
    assert rows.pack(8).stages[-1] == Pack(8)
    assert rows.windows(8).stages[-1] == Windows(8)
    assert rows.prefetch().stages[-1] == Prefetch()
    assert rows.workers(2).stages[-1] == Workers(2)
