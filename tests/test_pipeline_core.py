"""The pipeline's observable order, branch scope and exact positions."""

import dataclasses
import itertools
import json
from uuid import uuid4

import grain.python as grain
import numpy as np
import pytest

from dew.data.cache import Cache
from dew.data.compile import compile, iterate, publish
from dew.data.dataset import DataPartition, Loading
from dew.data.mix import SearchBudget, TokenDeficit, token_weights
from dew.data.pipeline import Artifact, Pipeline, Rejected
from dew.data.plan import Manifest, Plan, StepIdentity, fingerprint
from dew.data.sources import Memory
from dew.objectives.base import VALID_ROWS

READ = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)


@dataclasses.dataclass(frozen=True)
class Draw(grain.RandomMapTransform):
    def random_map(self, element, rng):
        return {**element, "draw": np.int64(rng.integers(0, 2**32))}


@dataclasses.dataclass(frozen=True)
class Odd(grain.MapTransform):
    rejects = True

    def map(self, element):
        return element if int(element["id"]) % 2 else Rejected("even")


@dataclasses.dataclass(frozen=True)
class Shift:
    by: int

    def __call__(self, batch):
        return {**batch, "id": batch["id"] + self.by}


@dataclasses.dataclass(frozen=True)
class Targets:
    def __call__(self, row):
        return int(row["targets"])


def source(count=24):
    return Memory(tuple({"id": np.int32(index)} for index in range(count)))


def take(reader, count):
    return [{name: np.asarray(value).tolist() for name, value in row.items()}
            for row in itertools.islice(reader, count)]


def test_memory_rejections_are_dense_before_global_order():
    stored = source()
    data = compile(Pipeline(stored).map(Odd()).shuffle(3).repeat().batch(4),
                   read="memory", loading=READ, seed=3)
    rows = tuple(value for value in stored.values if int(value["id"]) % 2)
    reader = data.train(DataPartition())
    old = iter(grain.MapDataset.source(rows).seed(3).shuffle(3).repeat()
               .to_iter_dataset(grain.ReadOptions(num_threads=1, prefetch_buffer_size=0)).batch(4))
    try:
        assert data.records == 12
        assert take(reader, 4) == take(old, 4)
        assert "dew_global_position" in json.loads(reader.get_state())
    finally:
        reader.close()
        old.close()


def test_global_position_resumes_across_share_counts_with_rng():
    data = compile(Pipeline(source()).shuffle(7).repeat().random_map(Draw()).batch(4), loading=READ, seed=7)
    reader = data.train(DataPartition())
    take(reader, 3)
    saved = reader.get_state()
    expected = take(reader, 5)
    reader.close()
    shares = []
    for index in range(2):
        reader = data.train(DataPartition(index, 2))
        reader.set_state(saved)
        shares.append(take(reader, 5))
        reader.close()
    joined = [{key: [value for pair in zip(left[key], right[key], strict=True) for value in pair]
               for key in left} for left, right in zip(*shares, strict=True)]
    assert joined == expected


def test_host_steps_are_branch_scoped_and_common_steps_run_on_both():
    prefix = Pipeline(source(8))
    data = compile(prefix.batch(4).map_batch(Shift(10)), validation=prefix.batch(4),
                   host=(Shift(100),), loading=READ)
    train = data.train(DataPartition())
    held = data.val(DataPartition())
    try:
        assert take(train, 1)[0]["id"] == [110, 111, 112, 113]
        assert take(held, 1)[0]["id"] == [100, 101, 102, 103]
    finally:
        train.close()
        held.close()


def test_rejection_iterator_resumes_exactly_and_refuses_topology_change():
    data = compile(Pipeline(source()).shuffle(2).repeat().map(Odd()).batch(3), loading=READ, seed=2)
    reader = data.train(DataPartition())
    take(reader, 3)
    saved = reader.get_state()
    rest = take(reader, 4)
    reader.close()
    resumed = data.train(DataPartition())
    resumed.set_state(saved)
    assert take(resumed, 4) == rest
    resumed.close()
    changed = data.train(DataPartition(0, 3))
    with pytest.raises(ValueError, match="topology"):
        changed.set_state(saved)
    changed.close()


def test_cache_refuses_random_prefix_before_opening_the_artifact():
    pipeline = Pipeline(source()).random_map(Draw()).then(Artifact(object())).batch(4)
    with pytest.raises(ValueError, match=r"cache prefix.*Draw"):
        compile(pipeline, loading=READ)


def test_map_batch_before_batch_and_random_validation_are_named_refusals():
    with pytest.raises(ValueError, match=r"map_batch.*batch"):
        compile(Pipeline(source()).map_batch(Shift(1)).batch(4), loading=READ)
    with pytest.raises(ValueError, match=r"validation.*Draw.*canonical"):
        compile(Pipeline(source()).batch(4),
                validation=Pipeline(source()).shuffle(0).random_map(Draw()).batch(4), loading=READ)


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("read", ["memory", "indexed"])
def test_worker_boundary_keeps_downstream_record_steps_and_exact_positions(workers, read):
    pipeline = Pipeline(source()).repeat().workers(workers).map(Shift(100)).limit(12).batch(4)
    data = compile(pipeline, read=read, loading=READ)
    for share in range(2):
        reader = data.train(DataPartition(share, 2))
        first = next(reader)
        state = reader.get_state()
        expected = take(reader, 8)
        reader.close()
        ids = [*first["id"].tolist(), *(value for row in expected for value in row["id"])]
        assert ids == [100 + share + 2 * index for index in range(12)]
        resumed = data.train(DataPartition(share, 2))
        resumed.set_state(state)
        assert take(resumed, 8) == expected
        resumed.close()


def test_worker_processes_read_a_global_order_in_the_batches_one_process_reads():
    """Each worker reads whole batches of an indexed order by index, and the
    reader interleaves them back into the order one process reads."""
    pipeline = Pipeline(source()).shuffle(5).repeat().batch(4)

    def batches(workers):
        reader = compile(pipeline, seed=5, loading=dataclasses.replace(READ, workers=workers)).train(
            DataPartition())
        try:
            return take(reader, 9)
        finally:
            reader.close()

    assert batches(2) == batches(0)


def test_token_deficit_mix_counts_targets_and_restores_delivered_counts():
    corpora = [grain.MapDataset.source([{"id": tag, "targets": targets}]).repeat().to_iter_dataset()
               for tag, targets in [(0, 2), (1, 7)]]
    mix = TokenDeficit(corpora, [1, 3], tokens=Targets(), names=["short", "long"])
    reader = iter(mix)
    delivered = [0, 0]
    for row in itertools.islice(reader, 100):
        delivered[row["id"]] += row["targets"]
        assert abs(delivered[0] - sum(delivered) / 4) <= 7
    saved = reader.get_state()
    rest = take(reader, 20)
    resumed = iter(mix)
    resumed.set_state(saved)
    assert take(resumed, 20) == rest
    reader.close()
    resumed.close()


def test_zero_target_corpus_exhausts_named_rejection_budget_without_starvation():
    parents = [grain.MapDataset.source([{"targets": count}]).repeat().to_iter_dataset()
               for count in (0, 4)]
    reader = iter(TokenDeficit(parents, [1, 1], tokens=Targets(), names=["all-zero", "useful"],
                              budget=SearchBudget(consecutive=5, seconds=None)))
    with pytest.raises(ValueError, match=r"all-zero.*search budget exhausted.*zero_targets"):
        next(reader)
    counts = reader.get_state()["parents"][0]["counts"]
    assert counts["attempted"] == 5
    assert counts["rejected"] == {"zero_targets": 5}
    reader.close()


def test_authoritative_zero_target_count_is_refused_before_division():
    with pytest.raises(ValueError, match=r"empty-targets.*zero valid target"):
        token_weights(["empty-targets", "useful"], [1, 1], [3, 2], [0, 10])


def test_dependency_change_preserves_global_resume_only_when_derived_order_matches():
    manifest = Manifest({"name": "test"}, revision="before")
    plan = Plan("global", manifest, {"seed": 0}, {"accepted_ids": fingerprint([1, 3, 5])})
    changed = dataclasses.replace(plan, manifest=dataclasses.replace(manifest, revision="after"))
    assert changed.check(plan.record()) == ("revision",)
    changed = dataclasses.replace(changed, derived={"accepted_ids": fingerprint([1, 3, 7])})
    with pytest.raises(ValueError, match="accepted_ids"):
        changed.check(plan.record())


def test_lambda_identity_requires_explicit_non_exact_resume():
    identity = StepIdentity.of(lambda row: row)
    plan = Plan("global", Manifest({}, (identity,)), {"seed": 0})
    with pytest.raises(ValueError, match="non-exact"):
        plan.check(plan.record())
    assert plan.check(plan.record(), resume="non-exact") == ()


def test_cold_warm_and_resumed_cache_keep_per_epoch_randomness(tmp_path):
    stored = source(8)
    cache = Cache(str(tmp_path / "cache"), shard_bytes=2048, spool_bytes=2 * 1024**2)
    prefix = Pipeline(stored).cache(cache)

    def build(prefix):
        return compile(prefix.shuffle(5).repeat().random_map(Draw()).batch(4),
                       read="memory", seed=5, loading=READ)

    cold = build(prefix)
    reader = cold.train(DataPartition())
    first_epoch = take(reader, 2)
    saved = reader.get_state()
    second_epoch = take(reader, 2)
    reader.close()

    def by_id(epoch):
        return {identifier: draw for batch in epoch for identifier, draw in
                zip(batch["id"], batch["draw"], strict=True)}

    assert by_id(first_epoch) != by_id(second_epoch)
    manifest = Manifest.of(stored.snapshot(), (), fill={"read": "memory", "row_ids": "indexed"})
    generation = cache.publish(manifest, stored)
    resumed = build(prefix).train(DataPartition())
    resumed.set_state(saved)
    assert take(resumed, 2) == second_epoch
    resumed.close()
    warm = build(Pipeline(stored).cache(generation)).train(DataPartition())
    warm_first, warm_second = take(warm, 2), take(warm, 2)
    assert by_id(warm_first) != by_id(warm_second)
    assert warm_first == first_epoch
    assert warm_second == second_epoch
    warm.close()


def test_causal_windows_keep_each_boundary_target_and_restore_carry():
    from dew.data.pipeline import Windows
    from dew.data.steps.windows import CausalWindows

    rows = grain.MapDataset.source([{"text": np.arange(4, dtype=np.int32)},
                                    {"text": np.arange(4, 11, dtype=np.int32)}]).to_iter_dataset()
    windows = CausalWindows(rows, Windows(3))
    reader = iter(windows)
    assert next(reader)["text"].tolist() == [0, 1, 2, 3]
    saved = reader.get_state()
    rest = take(reader, 10)
    assert rest == [{"text": [3, 4, 5, 6]}, {"text": [6, 7, 8, 9]}]
    resumed = iter(windows)
    resumed.set_state(saved)
    assert take(resumed, 10) == rest
    reader.close()
    resumed.close()


def test_validation_takes_accepted_rows_before_the_share_cut():
    prefix = Pipeline(source(11)).map(Odd()).take(3).batch(4)
    data = compile(Pipeline(source()).batch(4), validation=prefix, loading=READ)
    for share in range(2):
        held = data.val(DataPartition(share, 2))
        rows = list(held)
        assert len(rows) == 1
        valid = rows[0].get(VALID_ROWS, np.ones(len(rows[0]["id"]), bool))
        assert rows[0]["id"][valid].tolist() == ([1, 5] if share == 0 else [3])
        held.close()


@dataclasses.dataclass(frozen=True)
class PhysicalMemory(Memory):
    physical: tuple[int, ...] = (4, 6, 10)

    def locality(self):
        return self.physical, False


def test_block_shuffle_is_a_complete_bounded_physical_order_and_resumes():
    stored = PhysicalMemory(source(20).values)
    configured = Pipeline(stored).shuffle(11, block_shards=2).repeat().batch(4)
    data = compile(configured, read="memory", loading=READ)
    reader = data.train(DataPartition())
    first = take(reader, 5)
    ids = [value for batch in first for value in batch["id"]]
    assert sorted(ids) == list(range(20))
    state = reader.get_state()
    rest = take(reader, 4)
    reader.close()
    resumed = data.train(DataPartition())
    resumed.set_state(state)
    assert take(resumed, 4) == rest
    resumed.close()
    from dew.data.ordering import BlockShuffle

    blocked = BlockShuffle(grain.MapDataset.source(stored.values), stored.physical, 2, 11)
    order, ends = blocked._epoch(0)
    assert len(order) == 3
    start = 0
    original_ends = np.cumsum(stored.physical)
    for end in ends:
        selected = {int(np.searchsorted(original_ends, blocked.index(index), side="right"))
                    for index in range(start, int(end))}
        assert len(selected) <= 2
        start = int(end)
    changed = compile(Pipeline(stored).shuffle(11, block_shards=3).repeat().batch(4),
                      read="memory", loading=READ).train(DataPartition())
    with pytest.raises(ValueError, match="block_shards"):
        changed.set_state(state)
    changed.close()


def test_token_mixer_probes_an_unselected_tiny_share_before_emitting_a_row():
    parents = [grain.MapDataset.source([{"targets": 3}]).repeat().to_iter_dataset(),
               grain.MapDataset.source([{"targets": 0}]).repeat().to_iter_dataset()]
    reader = iter(TokenDeficit(parents, [1, 1e-9], tokens=Targets(), names=["normal", "broken"],
                              budget=SearchBudget(consecutive=3, seconds=None)))
    with pytest.raises(ValueError, match=r"broken.*budget exhausted"):
        next(reader)
    reader.close()


@dataclasses.dataclass(frozen=True)
class Twice:
    def flat_map(self, element):
        return [element, {**element, "copy": np.int32(1)}]


@dataclasses.dataclass(frozen=True)
class Placed:
    def __call__(self, index, element):
        return {**element, "position": np.int64(index)}


@pytest.mark.parametrize("expansion", [False, True])
def test_map_with_index_reads_the_source_position_in_every_read(expansion):
    """After a filter each row keeps its place in its source, so a rejected
    row leaves a gap, and after a flat_map each output shares its input's.
    Memory, indexed and streamed reads, and the same steps filled into a cache
    under each read, agree on it across shard boundaries."""
    stored = Memory(source(10).values, shard_rows=3)

    def steps(pipeline):
        pipeline = (pipeline.flat_map(Twice(), identity="twice@1") if expansion
                    else pipeline.filter(lambda row: int(row["id"]) % 2 == 1, identity="odd@1"))
        return pipeline.map_with_index(Placed(), identity="placed@1")

    expected = ([(index, index) for index in range(10) for _ in range(2)] if expansion
                else [(index, index) for index in range(1, 10, 2)])
    for read in ("memory", "indexed", "streamed"):
        cached = steps(Pipeline(stored)).cache(Cache(f"memory://{uuid4().hex}"))
        for pipeline in (steps(Pipeline(stored)), cached):
            rows = iterate(pipeline, read=read, loading=READ)
            assert [(int(row["id"]), int(row["position"])) for row in rows] == expected, (read, pipeline)


def test_map_with_index_is_refused_where_a_row_has_no_source_position():
    mixed = Pipeline(source()).mix(Pipeline(source()), weights=(1, 1))
    with pytest.raises(ValueError, match="map_with_index in mixture follows mix"):
        compile(mixed.map_with_index(Placed()).batch(4), loading=READ)
    windows = Pipeline(Memory([{"text": np.arange(9, dtype=np.int32)}])).windows(4)
    with pytest.raises(ValueError, match="follows Windows"):
        compile(windows.map_with_index(Placed()).batch(1), read="memory", loading=READ)


@pytest.mark.parametrize("read", ["memory", "indexed"])
def test_a_token_split_after_a_shuffle_splits_the_shuffled_order(read):
    """The held quarter is the first quarter of the shuffled order's tokens
    and the trained branch the rest: disjoint, together the corpus, and
    counted as the records an epoch trains on."""
    documents = [{"text": np.arange(100 * index, 100 * index + size, dtype=np.int32)}
                 for index, size in enumerate((5, 3, 8, 2, 6, 4, 7, 1))]
    shuffled = [int(token) for document in grain.MapDataset.source(documents).seed(0).shuffle(0)
                for token in document["text"]]

    def branch(*, held):
        return Pipeline(Memory(documents)).shuffle(0).token_split(0.25, held=held)

    def rows(pipeline):
        return [[int(token) for token in row["text"]] for row in iterate(pipeline, read=read, loading=READ)]

    held, train = rows(branch(held=True)), rows(branch(held=False))
    assert [token for row in held for token in row] == shuffled[:9]
    assert [token for row in train for token in row] == shuffled[9:]
    assert compile(branch(held=False).batch(1), read=read, loading=READ).records == len(train)


def test_a_stage_without_a_lowering_is_refused_at_compile_time():
    @dataclasses.dataclass(frozen=True)
    class Foreign:
        pass

    with pytest.raises(ValueError, match=r"Foreign in .* is not a pipeline stage"):
        compile(Pipeline(source()).then(Foreign()).batch(4), loading=READ)


def test_shares_agree_on_one_identity_and_resume_only_their_own_rows():
    """A per-share order is agreed across the pool without the share index,
    which only a share's saved position keeps."""
    thirds = Pipeline(source()).filter(lambda row: int(row["id"]) % 3 > 0, identity="thirds@1")
    data = compile(thirds.batch(4), loading=READ)
    first, second = data.train(DataPartition(0, 2)), data.train(DataPartition(1, 2))
    try:
        assert first.identity() == second.identity()
        next(first)
        with pytest.raises(ValueError, match="topology changed"):
            second.set_state(first.get_state())
    finally:
        first.close()
        second.close()


def test_a_per_share_reader_states_its_loadings_stop_budget_without_opening():
    """A per-share reader wraps a Grain iterator, which reports no stop budget, so the reader
    states its Loading's own, without opening the iterator to ask."""
    loading = Loading(workers=8, threads=1, read_buffer=1)
    thirds = Pipeline(source()).filter(lambda row: int(row["id"]) % 3 > 0, identity="thirds@1")
    data = compile(thirds.batch(4), loading=loading)
    opened = []
    reader = data.train(DataPartition(0, 2))
    reader._open = lambda: opened.append(True)
    assert reader.stop_seconds == loading.stop_seconds > 5.0
    assert opened == []


def test_a_memory_read_counts_a_missing_row_alike_with_or_without_a_cache():
    """A row the source holds as None is left out and counted missing,
    whether the memory read fills a cache with it or not."""
    stored = Memory([{"id": np.int32(0)}, None, {"id": np.int32(2)}])
    for pipeline in (Pipeline(stored), Pipeline(stored).cache(Cache(f"memory://{uuid4().hex}"))):
        data = compile(pipeline.batch(1), read="memory", loading=READ)
        assert data.records == 2
        assert [count.rejected["missing"] for count in data.counts.values()] == [1]


def test_a_cached_indexed_root_is_cut_into_disjoint_shares_that_repeat():
    """Each share of a cached indexed read takes its own rows of the source,
    and a training read with no repeat stage goes on past one pass."""
    cached = Pipeline(source(8)).map(Shift(1), identity="shift@1").cache(Cache(f"memory://{uuid4().hex}"))
    data = compile(cached.batch(2), loading=READ)
    shares = [data.train(DataPartition(index, 2)) for index in range(2)]
    try:
        read = [[value for batch in take(reader, 8) for value in batch["id"]] for reader in shares]
    finally:
        for reader in shares:
            reader.close()
    assert not set(read[0]) & set(read[1])
    assert sorted(set(read[0]) | set(read[1])) == list(range(1, 9))
    assert all(sorted(rows) == sorted(list(set(rows)) * 2) for rows in read)


def test_a_mixture_cut_per_share_reads_each_corpus_in_disjoint_shares():
    """A filter after a mixture cuts its order per share, and then every
    corpus it mixes is cut per share too: two shares never read one row."""
    other = Memory(tuple({"id": np.int32(100 + index)} for index in range(16)))
    mixed = Pipeline(source(16)).mix(Pipeline(other), weights=(1, 1))
    thirds = mixed.filter(lambda row: int(row["id"]) % 3 > 0, identity="thirds@1")
    data = compile(thirds.batch(2), loading=READ)
    shares = [data.train(DataPartition(index, 2)) for index in range(2)]
    try:
        read = [[value for batch in take(reader, 4) for value in batch["id"]] for reader in shares]
    finally:
        for reader in shares:
            reader.close()
    assert not set(read[0]) & set(read[1])


def test_a_per_share_position_resumes_against_the_same_rows_cached_elsewhere():
    """The saved plan names a cache by the rows it serves, so a per-share
    position saved with the cache in one place resumes with it in another."""
    def reader():
        cached = Pipeline(source(8)).map(Shift(1), identity="shift@1").cache(Cache(f"memory://{uuid4().hex}"))
        return compile(cached.batch(2), loading=READ).train(DataPartition())

    saved = reader()
    try:
        take(saved, 1)
        state = saved.get_state()
        expected = take(saved, 2)
    finally:
        saved.close()
    resumed = reader()
    try:
        resumed.set_state(state)
        assert take(resumed, 2) == expected
    finally:
        resumed.close()


def test_a_limit_counts_the_whole_order_and_a_prefetch_never_moves_the_share_cut():
    """24 records with prefetch(4).take(12) train on the same 12 records on one
    share or two."""
    data = compile(Pipeline(source()).prefetch(4).take(12).batch(4), loading=READ)

    def read(count):
        readers = [data.train(DataPartition(index, count)) for index in range(count)]
        try:
            return sorted(value for reader in readers for batch in take(reader, 10) for value in batch["id"])
        finally:
            for reader in readers:
                reader.close()

    assert read(1) == read(2) and len(read(1)) == 12


def test_a_token_split_is_part_of_the_order_a_resume_checks():
    """Four 4-token documents split at a quarter train on documents 1-3 and
    at a half on 2-3, so a position saved under one split is refused under
    the other."""
    documents = [{"text": np.arange(4 * index, 4 * index + 4, dtype=np.int32)} for index in range(4)]

    def reader(fraction):
        split = Pipeline(Memory(documents)).token_split(fraction, held=False).batch(1)
        return compile(split, read="indexed", loading=READ).train(DataPartition())

    quarter = reader(0.25)
    try:
        next(quarter)
        state = quarter.get_state()
    finally:
        quarter.close()
    half = reader(0.5)
    try:
        with pytest.raises(ValueError, match="order changed"):
            half.set_state(state)
    finally:
        half.close()


def test_a_recipe_reads_a_published_packing_before_its_default_order(tmp_path):
    """A typed recipe given a generation with a published packing plan reads
    that plan, and shuffles its windows, instead of packing again online."""
    from dew.data import Text, load
    from dew.data.steps.text import Chunks, Tokenize

    stored = Memory(tuple({"text": "x" * size} for size in (3, 7, 18, 2, 4, 12, 8, 1, 15, 6)))
    published = publish(Pipeline(stored).map(Tokenize()).flat_map(Chunks(9)).pack(9, bins=3),
                        Cache(str(tmp_path / "cache")), read="indexed", loading=READ)
    data = load(Text(source=stored, seq_len=8, pack="first-fit", packing_bins=3), batch=2, cache=published,
                read="indexed", loading=READ)
    assert data.plan is not None
    assert "packing" in data.plan.manifest.source["sources"][0]


@dataclasses.dataclass(frozen=True)
class Tagged:
    def __call__(self, element):
        return {**element, "tag": element["id"] * 10}


def test_record_steps_before_an_interleave_run_inside_each_shard():
    """The steps written before an interleave run on each shard's rows and the
    interleave flattens them where it stands; anything but a record step
    there is refused."""
    stored = Memory(source(10).values, shard_rows=3)
    before = Pipeline(stored).map(Tagged(), identity="tagged@1").interleave(2)
    after = Pipeline(stored).interleave(2).map(Tagged(), identity="tagged@1")
    rows = [[(int(row["id"]), int(row["tag"])) for row in iterate(pipeline, read="streamed", loading=READ)]
            for pipeline in (before, after)]
    assert rows[0] == rows[1] and sorted(rows[0]) == [(index, 10 * index) for index in range(10)]
    with pytest.raises(ValueError, match="only record steps before it"):
        iterate(Pipeline(stored).shuffle(0, window=4).interleave(2), read="streamed", loading=READ)


@dataclasses.dataclass(frozen=True)
class Laid:
    def __call__(self, element):
        return {"image": np.resize(np.asarray(element["text"]), (2, 4))}


def test_a_scored_branch_cut_per_share_splits_its_whole_order():
    """A validation branch is read in one canonical pass, so a token split in
    it is whole even where a filter cuts its order per share: it holds out the
    first half of the documents' tokens."""
    documents = Memory(tuple({"text": np.arange(4, dtype=np.int32) + 4 * index} for index in range(4)))
    held = Pipeline(documents).token_split(0.5, held=True).filter(lambda row: True, identity="all@1")
    data = compile(Pipeline(source()).batch(2), validation=held.batch(1), read="indexed", loading=READ)
    assert data.val is not None
    reader = data.val(DataPartition())
    try:
        tokens = [value for batch in reader for row in np.asarray(batch["text"]) for value in row.tolist()]
    finally:
        reader.close()
    assert tokens == list(range(8))


def test_a_streamed_repeat_repeats_where_it_is_written_past_a_limit_or_a_repeat():
    """A streamed root repeats in place of a written repeat only when nothing
    before it counts the rows: two rows taken and repeated three times read
    six, and eight rows repeated twice and then three times read 48, streamed
    as indexed."""
    def ids(pipeline, read):
        reader = compile(pipeline.batch(1), read=read, loading=READ).train(DataPartition())
        try:
            return [row["id"] for row in take(reader, 64)]
        finally:
            reader.close()

    limited = Pipeline(source(8)).take(2).repeat(3)
    streamed = ids(limited, "streamed")
    assert streamed == streamed[:2] * 3 and len(ids(limited, "indexed")) == 6
    twice = Pipeline(source(8)).repeat(2).repeat(3)
    assert len(ids(twice, "streamed")) == len(ids(twice, "indexed")) == 48


def test_a_nested_mixture_counts_its_epoch_from_each_part():
    """A mixture of a corpus and a mixture ends its epoch when the first of its
    parts comes round, the inner mixture counted the same way: the inner one
    of 8 and 4 rows at even weights in 16, the outer of 12 and 16 in 32."""
    inner = Pipeline(source(8)).mix(Pipeline(source(4)), weights=(1, 1))
    nested = Pipeline(source(12)).mix(inner, weights=(1, 1))
    assert compile(nested.batch(2), loading=READ).records == 32


def test_a_streamed_training_read_with_a_random_step_and_no_repeat_never_runs_out():
    """A training order that writes no repeat repeats for ever, as its plan
    says, whether or not a random step reads its rows."""
    reader = compile(Pipeline(source(8)).random_map(Draw()).batch(4), read="streamed",
                     loading=READ).train(DataPartition())
    try:
        assert reader.endless and len(take(reader, 6)) == 6
    finally:
        reader.close()


def test_the_saved_order_names_the_stages_a_memory_read_ran_as_an_indexed_read_does():
    """The canonical order a resume checks is the pipeline as written,
    whichever read runs its packing. A memory read packs before it orders,
    and its order still names the pack."""
    documents = [{"text": np.arange(size, dtype=np.int32)} for size in (8, 1, 8)]
    packed = Pipeline(Memory(documents)).pack(8, bins=1).shuffle(0, window=2).repeat().batch(1)
    orders = [compile(packed, read=read, loading=READ).plan.order["stages"] for read in ("memory", "indexed")]
    assert orders[0] == orders[1] and orders[0][0].startswith("Pack(")


def test_an_indexed_pack_over_a_token_store_reads_its_planned_global_windows(tmp_path):
    """A token store with its document index built packs by the plan of its
    documents' lengths: the indexed order is global, and its windows are the
    ones a memory read packs from the same documents."""
    from dew.data.sources.files import TokenFiles

    documents = [np.append(np.full(size, index + 1), 0).astype(np.int32)
                 for index, size in enumerate((5, 3, 9, 2, 7, 4, 6, 1))]
    (tmp_path / "train.bin").write_bytes(np.concatenate(documents).astype(np.uint16).tobytes())
    (tmp_path / "meta.json").write_text(json.dumps({
        "tokenizer": "byte", "dtype": "uint16", "vocab_size": 256, "eos_id": 0,
        "train_tokens": sum(len(document) for document in documents)}))
    store = TokenFiles(str(tmp_path)).provision()

    def windows(pipeline, read):
        rows = iterate(pipeline.pack(8, bins=2), read=read, loading=READ)
        return [{name: np.asarray(value).tolist() for name, value in row.items() if name[:5] != "_dew_"}
                for row in rows]

    lowering = compile(Pipeline(store).pack(8, bins=2).batch(2), read="indexed", loading=READ).plan.lowering
    assert lowering == "global"
    memory = Pipeline(Memory([{"text": document} for document in documents]))
    assert windows(Pipeline(store), "indexed") == windows(memory, "memory")


def test_a_memory_read_counts_valid_targets_where_it_packs():
    """Windows are kept by their own targets, however a later step lays them
    out: the one-token document's window has none and is left out."""
    documents = [{"text": np.arange(size, dtype=np.int32)} for size in (8, 1, 8)]
    laid = Pipeline(Memory(documents)).pack(8, bins=1).map(Laid(), identity="laid@1").batch(1)
    assert compile(laid, read="memory", loading=READ).records == 2
