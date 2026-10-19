"""The pipeline's observable order, branch scope and exact positions."""

import dataclasses
import gc
import importlib
import itertools
import json
import multiprocessing
import os
import weakref
from uuid import uuid4

import grain.python as grain
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dew import position
from dew.data.cache import Cache
from dew.data.compile import compile, iterate, validation
from dew.data.dataset import DataPartition, Loading
from dew.data.mixing import SearchBudget, TokenDeficit, token_weights
from dew.data.pipeline import Configured, Pipeline, Rejected, mix
from dew.data.plan import Manifest, Plan, StepIdentity, fingerprint
from dew.data.sources import Files, Memory
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


@dataclasses.dataclass(frozen=True)
class Counted:
    """A token row as its corpus and the targets it holds, which batch whatever the row's length."""

    def __call__(self, row):
        return {"source": row["source"], "targets": np.int32(len(row["text"]) - 1)}


def source(count=24):
    return Memory(tuple({"id": np.int32(index)} for index in range(count)))


def take(reader, count):
    return [{name: np.asarray(value).tolist() for name, value in row.items()}
            for row in itertools.islice(reader, count)]


@dataclasses.dataclass(frozen=True)
class Logged(Memory):
    """Records in memory that log each read a run takes of them."""

    reads: list = dataclasses.field(default_factory=list, compare=False)

    def records(self, read):
        self.reads.append(read)
        return super().records(read)


def test_a_memory_compile_reads_each_source_once():
    """Compiling a memory read pins each source's facts and materializes its steps from one read of
    its records, whatever else states its facts by index."""
    reads = []
    stored = Logged(source(8).values, reads=reads)
    data = compile(Pipeline(stored).map(Tagged()).shuffle(0).repeat().batch(2), read="memory", loading=READ)
    reader = data.train(DataPartition())
    try:
        assert sorted(value for batch in take(reader, 4) for value in batch["tag"]) == list(range(0, 80, 10))
    finally:
        reader.close()
    assert reads.count("memory") == 1


class Decoded(list):
    """The rows one read of a source made, which a test can watch be let go."""


@dataclasses.dataclass(frozen=True)
class Decoding(Memory):
    """Records in memory each read builds anew, as a file source decodes its rows, keeping a weak
    reference to each memory read it made."""

    reads: list = dataclasses.field(default_factory=list, compare=False)

    def records(self, read):
        rows = Decoded(super().records(read))
        if read == "memory":
            self.reads.append(weakref.ref(rows))
        return rows


@pytest.mark.parametrize("configured", [False, True])
def test_a_memory_loader_holds_the_rows_it_made_and_lets_the_read_it_made_them_from_go(configured):
    """A memory compile reads its source once, and once it has materialized every branch from that
    read it lets it go: the loader holds the rows its steps made, and reads them, a step configured
    from the source's facts, which preparation runs over the read first, too, and a validation
    branch's."""
    reads = []
    stored = Decoding(source(8).values, reads=reads)
    steps = (Configured(Shifting(0), needs=()), Tagged()) if configured else (Tagged(),)
    data = compile(Pipeline(stored).map(*steps).shuffle(0).repeat().batch(2), read="memory", loading=READ)
    held = validation(Pipeline(stored).map(*steps).batch(2), batch=2, read="memory", loading=READ)
    gc.collect()
    assert len(reads) == 2 and all(read() is None for read in reads)
    reader, checked = data.train(DataPartition()), held(DataPartition())
    try:
        assert sorted(value for batch in take(reader, 4) for value in batch["tag"]) == list(range(0, 80, 10))
        assert sorted(value for batch in checked for value in batch["tag"]) == list(range(0, 80, 10))
    finally:
        reader.close()
        checked.close()


def test_memory_rejections_are_dense_before_global_order():
    stored = source()
    data = compile(Pipeline(stored).filter_map(Odd()).shuffle(3).repeat().batch(4),
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


@pytest.mark.parametrize("read", ["memory", "indexed"])
@pytest.mark.parametrize("after", ["take", "window shuffle", "random map", "finite repeat"])
def test_a_global_order_resumes_through_the_stages_after_its_batch(read, after):
    """A global position counts the records of the batches the run read, in the order its batches
    come in after every stage written after the batch. A run resumed after two batches reads on as
    the uninterrupted run does through a limit, a window shuffle, a random step or a finite repeat
    after the batch, and a stage that keeps whole rows reads on alike from two shares."""
    rows = Pipeline(source(16))
    chain = {"take": rows.batch(2).take(5),
             "window shuffle": rows.repeat(1).batch(2).shuffle(7, window=3),
             "random map": rows.repeat().batch(2).random_map(Draw()),
             "finite repeat": rows.take(6).batch(2).repeat(2)}[after]
    data = compile(chain, read=read, loading=READ, seed=3)
    whole = data.train(DataPartition())
    try:
        expected = take(whole, 8)
    finally:
        whole.close()
    stopped = data.train(DataPartition())
    try:
        head = take(stopped, 2)
        saved = stopped.get_state()
    finally:
        stopped.close()
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(saved)
        assert head + take(resumed, 6) == expected
    finally:
        resumed.close()
    if after == "random map":
        return
    shares = []
    for index in range(2):
        reader = data.train(DataPartition(index, 2))
        try:
            reader.set_state(saved)
            shares.append(take(reader, 6))
        finally:
            reader.close()
    joined = [{"id": [value for pair in zip(left["id"], right["id"], strict=True) for value in pair]}
              for left, right in zip(*shares, strict=True)]
    assert head + joined == expected


def test_a_random_step_draws_alike_in_every_read_of_one_order():
    """A random step's draw is keyed by its record's place in the order and by the step's place in
    what the run wrote, never by how the read lowers the steps before it: a memory read, which runs
    the map before it over the whole source, an indexed read and a streamed read of one share draw
    the same values, in a mixture as alone, and a second random step draws apart from the first."""
    rows = Pipeline(source(8)).map(Tagged())
    alone = rows.random_map(Draw()).repeat(1).batch(4)
    mixed = mix([rows.random_map(Draw()).repeat(1)]).batch(4)

    def draws(chain, read):
        data = compile(chain, read=read, loading=READ, seed=5)
        stream = data.train(DataPartition())
        try:
            return [(int(index), int(draw)) for batch in itertools.islice(stream, 2)
                    for index, draw in zip(batch["id"], batch["draw"], strict=True)]
        finally:
            stream.close()

    for chain in (alone, mixed):
        assert draws(chain, "memory") == draws(chain, "indexed") == draws(chain, "streamed")
    again = draws(rows.random_map(Draw()).random_map(Draw()).repeat(1).batch(4), "indexed")
    assert [draw for _, draw in again] != [draw for _, draw in draws(alone, "indexed")]


@dataclasses.dataclass(frozen=True)
class Member:
    """A record marked with the mixture member it came from."""

    index: int

    def __call__(self, element):
        return {**element, "member": np.int32(self.index)}


def test_a_mixture_members_draws_stay_as_they_were_when_another_member_joins():
    """A random step in a mixture member draws by the name the member goes by, not by its place among
    the members: a member joining before it leaves its draws as they were, and two members of one
    name draw apart."""
    def member(name, index):
        return Pipeline(source(8), name=name).map(Member(index)).random_map(Draw()).repeat(1)

    def draws(members):
        data = compile(mix(members).batch(4), read="indexed", loading=READ, seed=5)
        stream = data.train(DataPartition())
        try:
            return {(int(index), int(of)): int(draw) for batch in stream
                    for index, of, draw in zip(batch["id"], batch["member"], batch["draw"], strict=True)}
        finally:
            stream.close()

    alone = draws([member("kept", 1)])
    joined = draws([member("joining", 0), member("kept", 1)])
    assert {key: draw for key, draw in joined.items() if key[1] == 1} == alone
    twins = draws([member("twin", 0), member("twin", 1)])
    assert [twins[index, 0] for index in range(8)] != [twins[index, 1] for index in range(8)]


def test_an_unnamed_member_draws_alike_wherever_a_host_mounts_it_and_when_a_named_one_joins(tmp_path):
    """A member given no name goes by its place among the members given none, never by where its
    source sits: one dataset mounted at two paths draws alike in a mixture, whose plans agree, and a
    named member joining the mixture leaves its draws as they were."""
    def mounted(directory, joining=False):
        directory.mkdir(exist_ok=True)
        (directory / "rows.jsonl").write_text("".join(json.dumps({"id": i}) + "\n" for i in range(8)))
        chain = Pipeline(Files(str(directory / "rows.jsonl"))).map(Member(0)).random_map(Draw())
        named = Pipeline(source(8), name="joining").map(Member(1)).random_map(Draw()).repeat(1)
        joined = [named] if joining else []
        return compile(mix([*joined, chain.repeat(1)]).batch(4), read="memory", loading=READ, seed=5)

    here, there = mounted(tmp_path / "here"), mounted(tmp_path / "there")
    assert here.plan.identity() == there.plan.identity()
    draws = []
    for data in (here, there, mounted(tmp_path / "here", joining=True)):
        stream = data.train(DataPartition())
        try:
            draws.append({int(index): int(draw) for batch in stream
                          for index, of, draw in zip(batch["id"], batch["member"], batch["draw"], strict=True)
                          if of == 0})
        finally:
            stream.close()
    assert draws[0] == draws[1]
    # The mixture may end once a member runs out; every row it read draws as it did alone.
    assert draws[2] and all(draws[0][index] == draw for index, draw in draws[2].items())


def test_a_member_named_as_a_place_keeps_its_draws_when_an_unnamed_one_joins():
    """A member's name and an unnamed member's place are kept apart, whatever the name spells: a
    member named like a place keeps its draws when an unnamed member joins ahead of it, and an
    unnamed member keeps its own when that named member joins ahead of it."""
    def member(name, index):
        return Pipeline(source(8), name=name).map(Member(index)).random_map(Draw()).repeat(1)

    def draws(members, of):
        data = compile(mix(members).batch(4), read="indexed", loading=READ, seed=5)
        stream = data.train(DataPartition())
        try:
            return {int(index): int(draw) for batch in stream
                    for index, at, draw in zip(batch["id"], batch["member"], batch["draw"], strict=True)
                    if at == of}
        finally:
            stream.close()

    for name in ("place:0", "#0", "0"):
        alone = draws([member(name, 1)], 1)
        joined = draws([member(None, 0), member(name, 1)], 1)
        assert joined and all(alone[index] == draw for index, draw in joined.items())
        unnamed = draws([member(None, 0)], 0)
        ahead = draws([member(name, 1), member(None, 0)], 0)
        assert ahead and all(unnamed[index] == draw for index, draw in ahead.items())


def member_rows(members, by, saved=None, batches=6):
    """The (member, id) rows a streamed mix of `members` delivers over `batches` batches, and its
    position after the first two, resumed from `saved` when given."""
    data = compile(mix(members, by=by).batch(2), read="streamed", loading=READ, seed=5)
    stream = data.train(DataPartition())
    try:
        if saved is not None:
            stream.set_state(saved)
        rows, position = [], None
        for step in range(batches):
            if step == 2 and saved is None:
                position = stream.get_state()
            batch = next(stream)
            rows += [(int(of), int(index)) for of, index in zip(batch["source"], batch["id"], strict=True)]
        return rows, position
    finally:
        stream.close()


@pytest.mark.parametrize("by", ["records", "tokens"])
@pytest.mark.parametrize("edit", ["joined", "left"])
def test_a_streamed_mix_resumes_its_members_where_they_were_when_one_joins_or_leaves(by, edit):
    """A streamed mix keeps each member's position under its key: resumed with a member joined, the
    members it had read on from where they were and the new one from its first row; resumed with a
    member gone, the others read on from where they were."""
    def member(name, source):
        return Pipeline(roled(source, 6, 2, count=8), name=name).repeat()

    before = [member("a", 0), member("b", 1)] + ([] if edit == "joined" else [member("c", 2)])
    after = [member("a", 0), member("b", 1)] + ([member("c", 2)] if edit == "joined" else [])
    whole, saved = member_rows(before, by)
    read = 2 * 2  # the rows of the two batches read before the position was saved
    resumed, _ = member_rows(after, by, saved=saved)

    def rows_of(rows, source):
        return [index for of, index in rows if of == source]

    for source in (0, 1):
        ahead = rows_of(whole[read:], source)
        again = rows_of(resumed, source)
        assert again and again[:len(ahead)] == ahead[:len(again)]
    if edit == "joined":
        assert rows_of(resumed, 2)[:1] == [0]


def test_a_streamed_mix_refuses_a_member_that_changed_what_it_reads():
    """A member kept across a resume must read and run what it did: one whose steps changed is
    refused by name, as its saved position counts other rows."""
    _, saved = member_rows([Pipeline(roled(0, 6, 2), name="a").repeat(),
                            Pipeline(roled(1, 6, 2), name="b").repeat()], "records")
    changed = [Pipeline(roled(0, 6, 2), name="a").map(Relabelled()).repeat(),
               Pipeline(roled(1, 6, 2), name="b").repeat()]
    with pytest.raises(ValueError, match="member"):
        member_rows(changed, "records", saved=saved)


def test_an_indexed_mix_still_refuses_a_member_joining():
    """An indexed mix reads by one global index over all its members, which a member joining moves:
    its resume refuses the edit."""
    def mixed(members):
        return compile(mix(members).batch(2), read="indexed", loading=READ, seed=5)

    members = [Pipeline(roled(0, 6, 2), name="a").repeat(), Pipeline(roled(1, 6, 2), name="b").repeat()]
    stream = mixed(members).train(DataPartition())
    try:
        next(stream)
        saved = stream.get_state()
    finally:
        stream.close()
    resumed = mixed([*members, Pipeline(roled(2, 6, 2), name="c").repeat()]).train(DataPartition())
    try:
        with pytest.raises(ValueError):
            resumed.set_state(saved)
    finally:
        resumed.close()


def test_a_position_is_refused_where_a_mixture_member_was_renamed():
    """A mixture's plan names its members, whose draws their names key: a position saved under one
    name is refused by the mixture that renamed it, which would draw other rows."""
    def mixed(name):
        members = [Pipeline(source(8), name=name).random_map(Draw()).repeat(),
                   Pipeline(source(4)).random_map(Draw()).repeat()]
        return compile(mix(members).batch(4), read="indexed", loading=READ, seed=5)

    stream = mixed("first").train(DataPartition())
    try:
        next(stream)
        saved = stream.get_state()
    finally:
        stream.close()
    resumed = mixed("second").train(DataPartition())
    try:
        with pytest.raises(ValueError):
            resumed.set_state(saved)
    finally:
        resumed.close()


def drawn_natively(rows):
    """A Grain step Dew does not name, whose random step draws by the default seed of the rows it
    is handed."""
    return rows.random_map(Draw())


@pytest.mark.parametrize("read", ["indexed", "streamed"])
def test_a_native_step_draws_by_the_runs_seed_and_resumes(read):
    """A native step (`Pipeline.grain`) is handed rows seeded by the run, as Grain's own pipelines
    seed theirs, so a random step it adds without a seed draws: by the run's seed, apart from a
    second native step's draws, and on in a resumed run as in the uninterrupted one."""
    rows = Pipeline(source(8)).grain(drawn_natively)

    def draws(chain, seed):
        data = compile(chain.repeat().batch(2), read=read, loading=READ, seed=seed)
        whole = data.train(DataPartition())
        try:
            expected = take(whole, 4)
        finally:
            whole.close()
        stopped = data.train(DataPartition())
        try:
            head = take(stopped, 2)
            saved = stopped.get_state()
        finally:
            stopped.close()
        resumed = data.train(DataPartition())
        try:
            resumed.set_state(saved)
            assert head + take(resumed, 2) == expected
        finally:
            resumed.close()
        return [draw for batch in expected for draw in batch["draw"]]

    assert draws(rows, 5) != draws(rows, 6)
    assert draws(rows.grain(drawn_natively), 5) != draws(rows, 5)


@pytest.mark.parametrize("order", ["global", "per-share"])
def test_a_position_saved_under_another_plan_version_is_refused(order):
    """A plan's version names how Dew makes rows from it, which its fields do not say: a position
    an earlier version saved, whose random steps drew by another key, is refused rather than
    resumed into other draws, in a global order and in one read per share; one this version saved
    resumes."""
    rows = Pipeline(source(8)).random_map(Draw())
    chain = rows if order == "global" else rows.grain(drawn_natively)
    data = compile(chain.repeat().batch(2), read="indexed", loading=READ, seed=7)
    assert data.plan is not None and data.plan.lowering == order
    stopped = data.train(DataPartition())
    try:
        take(stopped, 2)
        saved = stopped.get_state()
        expected = take(stopped, 2)
    finally:
        stopped.close()
    if order == "global":
        place = position.read(saved)
        assert place.plan is not None
        earlier = position.encode(dataclasses.replace(place, plan={**place.plan, "version": 1}))
    else:
        state = json.loads(saved)
        earlier = json.dumps({**state, "dew_data_plan": {**state["dew_data_plan"], "version": 1}}).encode()
    resumed = data.train(DataPartition())
    try:
        with pytest.raises(ValueError):
            resumed.set_state(earlier)
        resumed.set_state(saved)
        assert take(resumed, 2) == expected
    finally:
        resumed.close()


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_host_steps_are_branch_scoped_and_common_steps_run_on_both(read):
    prefix = Pipeline(source(8))
    data = compile(prefix.batch(4).map(Shift(10)), validation=prefix.batch(4),
                   host=(Shift(100),), read=read, loading=READ)
    train = data.train(DataPartition())
    held = data.val(DataPartition())
    try:
        assert take(train, 1)[0]["id"] == [110, 111, 112, 113]
        saved = train.get_state()
        expected = take(train, 2)
        if read != "streamed":
            assert train.records == 12
            assert train.order == json.loads(saved)["dew_global_position"]["order"]
        assert take(held, 1)[0]["id"] == [100, 101, 102, 103]
    finally:
        train.close()
        held.close()
    train.close()
    with pytest.raises(StopIteration):
        next(train)
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(saved)
        assert take(resumed, 2) == expected
    finally:
        resumed.close()


def test_rejection_iterator_resumes_exactly_and_refuses_topology_change():
    data = compile(Pipeline(source()).shuffle(2).repeat().filter_map(Odd()).batch(3), loading=READ, seed=2)
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
    with pytest.raises(ValueError):
        changed.set_state(saved)
    changed.close()


def test_random_validation_is_refused():
    compile(Pipeline(source()).batch(4), validation=Pipeline(source()).shuffle(0).batch(4), loading=READ)
    with pytest.raises(ValueError):
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


def test_automatic_loading_sizes_an_indexed_read_buffer_by_the_batches_it_holds(monkeypatch):
    """An indexed order's read buffer holds whole batches, so automatic
    loading sizes it by a batch: 2 MiB holds four batches of eight 64 KiB
    rows, where sized by a row it would hold sixteen."""
    import grain.experimental as gx

    chosen = []
    pick = gx.pick_performance_config

    def recorded(*args, **kwargs):
        config = pick(*args, **kwargs)
        chosen.append(config.read_options.prefetch_buffer_size)
        return config

    monkeypatch.setattr(gx, "pick_performance_config", recorded)
    rows = Memory(tuple({"x": np.zeros(16384, np.float32)} for _ in range(64)))
    compile(Pipeline(rows).repeat().batch(8), loading=Loading(workers="auto", ram_budget_mb=2, max_workers=1))
    assert chosen == [4]


def test_a_validation_pass_starts_at_most_one_worker_per_batch(monkeypatch):
    """A held pass of two batches starts two worker processes though training
    reads with four: a third and a fourth would read nothing, yet each would
    import the program and keep its pipes until the reader is collected."""
    started = []
    start = multiprocessing.context.SpawnProcess.start

    def counted(process):
        started.append(process)
        start(process)

    monkeypatch.setattr(multiprocessing.context.SpawnProcess, "start", counted)
    prefix = Pipeline(source(8))
    data = compile(prefix.repeat().batch(4), validation=prefix.batch(4),
                   loading=Loading(workers=4, threads=1, read_buffer=1, worker_buffer=1))
    reader = data.val(DataPartition())
    try:
        assert [row["id"].tolist() for row in reader] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    finally:
        reader.close()
    assert len(started) == 2


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


@pytest.mark.parametrize("sliced", [False, True])
def test_a_token_mixture_weighs_the_rows_each_corpus_delivers(sliced):
    """Mixed by tokens, two corpora deliver equal valid targets. A corpus whose order reads every
    row its source's counts describe keeps one global order, weighed by those counts; one whose
    order takes a slice of its rows delivers targets those counts do not describe, so its mixture
    counts targets as they arrive."""
    from dew.data.steps.text import CountTargets

    def corpus(name, lengths):
        rows = Memory([{"text": np.arange(size, dtype=np.int32), "source": np.int32(name == "long")}
                       for size in lengths])
        return Pipeline(rows, name=name).filter_map(CountTargets())

    long = corpus("long", (2, 10, 10, 10))
    long = long.slice(slice(1)) if sliced else long
    mixed = mix([long.repeat(), corpus("short", (2, 2)).repeat()], by="tokens")
    data = compile(mixed.map(Counted()).batch(8), read="memory", loading=READ)
    stream = data.train(DataPartition())
    targets = [0, 0]
    try:
        for batch in itertools.islice(stream, 8):
            for source, count in zip(batch["source"], batch["targets"], strict=True):
                targets[int(source)] += int(count)
    finally:
        stream.close()
    assert abs(targets[0] - targets[1]) <= 9
    assert data.plan is not None and data.plan.lowering == ("per-share" if sliced else "global")


@dataclasses.dataclass(frozen=True)
class Stated(Memory):
    """Token rows in memory that state their rows and valid targets, as a published store does."""

    def metadata(self):
        rows = self.records("indexed")
        return {"rows": len(rows), "valid_targets": sum(len(row["text"]) - 1 for row in rows)}


class Doubled:
    """A step over a token row that makes another row of it, as tokenizing or chunking does, so a
    store's stated counts describe rows the corpus no longer holds."""

    def __call__(self, row):
        return {**row, "text": np.concatenate([row["text"], row["text"]])}


def test_a_memory_read_weighs_a_token_mixture_by_the_targets_of_the_rows_it_holds():
    """A memory read that runs steps over its stores holds every row a mixture by tokens reads, so it
    counts their valid targets by the mixture's rule, not by the stores' stated counts of rows the
    steps changed: the order stays global, an epoch of it has a record count, and the corpora
    deliver equal targets by what their rows hold."""
    def corpus(name, lengths):
        return Stated([{"text": np.arange(size, dtype=np.int32), "source": np.int32(name == "long")}
                       for size in lengths])

    parts = [Pipeline(corpus("long", (2, 10, 10, 10)), name="long").map(Doubled()),
             Pipeline(corpus("short", (2, 2)), name="short")]
    mixed = mix([part.repeat() for part in parts], by="tokens")
    data = compile(mixed.map(Counted()).batch(8), read="memory", loading=READ)
    assert data.plan is not None and data.plan.lowering == "global" and data.steps_per_epoch is not None
    stream = data.train(DataPartition())
    targets = [0, 0]
    try:
        for batch in itertools.islice(stream, 8):
            for source, count in zip(batch["source"], batch["targets"], strict=True):
                targets[int(source)] += int(count)
    finally:
        stream.close()
    assert abs(targets[0] - targets[1]) <= 19


@dataclasses.dataclass(frozen=True)
class Halved:
    """A random step that keeps the first half of a token row, as a random crop keeps a part, so the
    rows a corpus delivers hold fewer targets than the rows its read holds."""

    def __call__(self, row, rng):
        return {**row, "text": row["text"][: len(row["text"]) // 2]}


def test_a_token_mixture_weighs_a_corpus_by_the_targets_its_rows_deliver():
    """A corpus whose rows a step still to run changes, here a random crop to half, delivers other
    targets than the rows its read holds and its store states: the mixture by tokens counts them as
    they arrive, so both corpora deliver equal targets, to within one row's."""
    def corpus(name, lengths):
        return Stated([{"text": np.arange(size, dtype=np.int32), "source": np.int32(name == "long")}
                       for size in lengths])

    parts = [Pipeline(corpus("long", (20, 20, 20, 20)), name="long").random_map(Halved()),
             Pipeline(corpus("short", (4, 4)), name="short")]
    mixed = mix([part.repeat() for part in parts], by="tokens")
    data = compile(mixed.map(Counted()).batch(8), read="memory", loading=READ)
    stream = data.train(DataPartition())
    targets = [0, 0]
    try:
        for batch in itertools.islice(stream, 16):
            for source, count in zip(batch["source"], batch["targets"], strict=True):
                targets[int(source)] += int(count)
    finally:
        stream.close()
    assert abs(targets[0] - targets[1]) <= 9


def test_a_token_mixture_counts_the_targets_a_corpus_delivers_after_its_repeat_changes_them():
    """A corpus whose rows a random crop changes after its repeat delivers other targets than its
    store states: the mixture by tokens weighs it by the targets its rows deliver, so both corpora
    deliver equal targets, to within one row's."""
    def corpus(name, lengths):
        return Stated([{"text": np.arange(size, dtype=np.int32), "source": np.int32(name == "long")}
                       for size in lengths])

    long = Pipeline(corpus("long", (20, 20, 20, 20)), name="long").repeat().random_map(Halved())
    mixed = mix([long, Pipeline(corpus("short", (4, 4)), name="short").repeat()], by="tokens")
    data = compile(mixed.map(Counted()).batch(8), read="memory", loading=READ)
    stream = data.train(DataPartition())
    targets = [0, 0]
    try:
        for batch in itertools.islice(stream, 16):
            for source, count in zip(batch["source"], batch["targets"], strict=True):
                targets[int(source)] += int(count)
    finally:
        stream.close()
    assert abs(targets[0] - targets[1]) <= 9


@dataclasses.dataclass(frozen=True)
class Cropped:
    """A step that keeps the first half of a token row and its roles, as a crop keeps a part."""

    def __call__(self, row):
        half = len(row["text"]) // 2
        return {**row, "text": row["text"][:half], "text_roles": row["text_roles"][:half]}


@dataclasses.dataclass(frozen=True)
class CroppedTokens:
    """`Cropped` over a row whose tokens a projection named `tokens`."""

    def __call__(self, row):
        half = len(row["tokens"]) // 2
        return {**row, "tokens": row["tokens"][:half], "text_roles": row["text_roles"][:half]}


@dataclasses.dataclass(frozen=True)
class Relabelled:
    """A step that changes only a token row's id, leaving its tokens and roles as they were."""

    def __call__(self, row):
        return {**row, "id": row["id"] + 100}


@dataclasses.dataclass(frozen=True)
class Scored:
    """A delivered row as its corpus and the targets the loss scores of it, by `role`'s rule."""

    role: int | None

    def __call__(self, row):
        from dew.data.steps.text import ValidTargets

        # A row whose roles a projection renamed reads them under their new name.
        scored = {"text": row["text"], "text_roles": row.get("text_roles", row.get("roles"))}
        return {"source": row["source"],
                "targets": np.int32(ValidTargets(role=self.role, measured=False)(scored))}


@dataclasses.dataclass(frozen=True)
class Copied:
    """A step that copies a token row's arrays, values unchanged, as a format conversion does."""

    def __call__(self, row):
        return {name: np.array(value) for name, value in row.items()}


@dataclasses.dataclass(frozen=True)
class Encoded(grain.MapTransform):
    """A step that stores a record as the artifact codec's bytes, which `Restored` reads back."""

    def map(self, element):
        from dew.data.sources.codec import encode

        return encode(element)


@dataclasses.dataclass(frozen=True)
class Restored(grain.MapTransform):
    """A step that reads back the record `Encoded` stored."""

    def map(self, element):
        from dew.data.sources.codec import decode

        return decode(element)


def roled(source, length, scored, count=4):
    """`count` token rows of `length`, the first `scored` of each row's targets in role 1."""
    roles = np.asarray([1] * (scored + 1) + [0] * (length - scored - 1), np.int32)
    return Memory(tuple({"text": np.arange(length, dtype=np.int32), "text_roles": roles,
                         "id": np.int32(index), "source": np.int32(source)} for index in range(count)))


def delivered(corpora, role, read, batches=24):
    """The targets each of two corpora mixed by tokens delivers, as the loss scores them by `role`.
    A mix by tokens keeps each corpus within one of its own rows of its share, so the two differ by
    at most one row of each."""
    mixed = mix([corpus.repeat() for corpus in corpora], by="tokens")
    data = compile(mixed.map(Scored(role)).batch(8), read=read, loading=READ)
    stream = data.train(DataPartition())
    targets = [0, 0]
    try:
        for batch in itertools.islice(stream, batches):
            for source, count in zip(batch["source"], batch["targets"], strict=True):
                targets[int(source)] += int(count)
    finally:
        stream.close()
    return targets


@pytest.mark.parametrize("role", [None, 1])
@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_a_token_mixture_weighs_a_cropped_corpus_by_the_targets_left_after_the_crop(read, role):
    """Targets counted on a row (`CountTargets`) a crop then halves are counted again by the rule
    that counted them, its role included: the mixture weighs the cropped corpus by what its rows
    deliver, so both corpora deliver equal targets, to within one row of each (`delivered`)."""
    from dew.data.steps.text import CountTargets

    long = Pipeline(roled(0, 24, 22), name="long").filter_map(CountTargets(role=role)).map(Cropped())
    short = Pipeline(roled(1, 6, 4), name="short").filter_map(CountTargets(role=role))
    left, right = delivered([long, short], role, read)
    rows = (Cropped()(roled(0, 24, 22).records("memory")[0]), roled(1, 6, 4).records("memory")[0])
    assert abs(left - right) <= sum(int(Scored(role)(row)["targets"]) for row in rows)


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_a_token_mixture_keeps_a_rows_count_through_a_step_over_its_other_fields(read):
    """A step that changes only a row's other fields keeps the targets its count's own rule counted,
    a role's here, which no other rule gives: two corpora whose rows score one and three role-1
    targets deliver equal scored targets, to within one row of each (`delivered`)."""
    from dew.data.steps.text import CountTargets

    one = Pipeline(roled(0, 8, 1), name="one").filter_map(CountTargets(role=1)).map(Relabelled())
    three = Pipeline(roled(1, 8, 3), name="three").filter_map(CountTargets(role=1))
    left, right = delivered([one, three], 1, read)
    assert abs(left - right) <= 1 + 3


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
@pytest.mark.parametrize("after", ["renamed-roles", "renamed-roles-then-copied",
                                   "renamed-text-then-cropped", "selected-then-copied",
                                   "counted-again-then-copied", "encoded-and-decoded"])
def test_a_rows_count_and_its_rule_ride_through_the_steps_that_leave_its_targets(read, after):
    """A row's count, by its role's rule, holds through steps that change no target: its roles
    renamed, and copied after, its fields narrowed and then copied, the bytes of an encoding and
    back; a crop of tokens a projection renamed counts them again under their new name; and a
    second count, by the plain rule, replaces the first one's rule with its own, which a later copy
    keeps.
    Two corpora whose rows score one and three role-1 targets, or seven plain ones each, deliver
    equal targets by the rule last counted, to within one row of each (`delivered`)."""
    from dew.data.steps.text import CountTargets

    def corpus(source, scored):
        chain = Pipeline(roled(source, 8, scored), name=str(source)).filter_map(CountTargets(role=1))
        if after == "renamed-roles":
            return chain.rename_columns({"text_roles": "roles"})
        if after == "renamed-roles-then-copied":
            return chain.rename_columns({"text_roles": "roles"}).map(Copied())
        if after == "renamed-text-then-cropped":
            return (chain.rename_columns({"text": "tokens"}).map(CroppedTokens())
                    .rename_columns({"tokens": "text"}))
        if after == "selected-then-copied":
            return chain.select_columns("text", "text_roles", "source").map(Copied())
        if after == "counted-again-then-copied":
            return chain.filter_map(CountTargets()).map(Copied())
        return chain.map(Encoded()).map(Restored())

    role = None if after == "counted-again-then-copied" else 1
    # A crop halves a row's six role-1 targets to three and leaves the other's two as they were.
    scored = (6, 2) if after == "renamed-text-then-cropped" else (1, 3)
    left, right = delivered([corpus(0, scored[0]), corpus(1, scored[1])], role, read)
    bound = {None: 7 + 7, 1: 3 + 2 if after == "renamed-text-then-cropped" else 1 + 3}[role]
    assert abs(left - right) <= bound


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_a_count_follows_its_tokens_through_a_rename_that_reuses_their_name(read):
    """A rename that moves the counted tokens away and names another field as they were, and then
    back again, leaves the count on the tokens: the padding the other field holds is no target."""
    from dew.data.steps.text import CountTargets

    def corpus(source):
        rows = Memory(tuple({"text": np.asarray([1, 2, 3, 4], np.int32), "backup": np.zeros(4, np.int32),
                             "source": np.int32(source)} for _ in range(4)))
        return (Pipeline(rows, name=str(source)).filter_map(CountTargets(pad_id=0))
                .rename_columns({"text": "tokens", "backup": "text"})
                .rename_columns({"text": "backup", "tokens": "text"}))

    mixed = mix([corpus(0).repeat(), corpus(1).repeat()], by="tokens")
    stream = compile(mixed.batch(4), read=read, loading=READ).train(DataPartition())
    try:
        assert np.asarray(next(stream)["text"]).tolist()[0] == [1, 2, 3, 4]
    finally:
        stream.close()


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_tokenized_rows_pack_by_concatenation(read):
    """Rows a tokenizer made, which carry Dew's own record of their lengths, pack by concatenation
    into windows of the packed field alone."""
    from dew.data.text import Tokenize

    rows = Memory(tuple({"text": "abcdefgh"[: 3 + index % 5]} for index in range(12)))
    chain = Pipeline(rows).map(Tokenize()).pack(4, policy="concat").batch(2)
    stream = compile(chain, read=read, loading=READ).train(DataPartition())
    try:
        batch = next(stream)
    finally:
        stream.close()
    assert np.asarray(batch["text"]).shape == (2, 4)


def test_a_counted_corpus_projected_after_its_count_keeps_it_in_memory():
    """A corpus counted and then renamed or narrowed to other fields keeps its rows' counts, which
    no field it still holds could give again."""
    from dew.data.steps.text import CountTargets

    for projected in (lambda chain: chain.rename_columns({"text": "tokens"}),
                      lambda chain: chain.select_columns("id", "source")):
        chains = [projected(Pipeline(roled(source, 8, 3), name=str(source)).filter_map(CountTargets(role=1)))
                  for source in (0, 1)]
        data = compile(mix([chain.repeat() for chain in chains], by="tokens").batch(4), read="memory",
                       loading=READ)
        stream = data.train(DataPartition())
        try:
            assert len(next(stream)["id"]) == 4
        finally:
            stream.close()


@pytest.mark.parametrize("held", [False, True])
def test_a_token_mixture_weighs_a_random_holdout_by_the_rows_it_keeps(held):
    """Mixed by tokens, two corpora whose stores state their counts deliver equal valid targets,
    in one global order weighed by those counts. A random holdout keeps some of its store's rows,
    so the store's counts describe rows it does not read: its mixture counts targets as they
    arrive instead."""
    from dew.config.data import RecordsData
    from dew.data.holdout import HoldOut

    def corpus(name, lengths):
        return Stated([{"text": np.arange(size, dtype=np.int32), "source": np.int32(name == "long")}
                       for size in lengths])

    long = corpus("long", (2, 10, 10, 10))
    trained = (RecordsData(long, validation=HoldOut(records=3, by="random", seed=2)).pipelines()[0] if held
               else Pipeline(long))
    parts = [dataclasses.replace(trained, name="long"), Pipeline(corpus("short", (2, 2)), name="short")]
    mixed = mix([part.repeat() for part in parts], by="tokens")
    data = compile(mixed.map(Counted()).batch(8), read="indexed", loading=READ)
    stream = data.train(DataPartition())
    targets = [0, 0]
    try:
        for batch in itertools.islice(stream, 8):
            for source, count in zip(batch["source"], batch["targets"], strict=True):
                targets[int(source)] += int(count)
    finally:
        stream.close()
    assert abs(targets[0] - targets[1]) <= 9
    assert data.plan is not None and data.plan.lowering == ("per-share" if held else "global")


def test_zero_target_corpus_exhausts_named_rejection_budget_without_starvation():
    parents = [grain.MapDataset.source([{"targets": count}]).repeat().to_iter_dataset()
               for count in (0, 4)]
    reader = iter(TokenDeficit(parents, [1, 1], tokens=Targets(), names=["all-zero", "useful"],
                              budget=SearchBudget(consecutive=5)))
    with pytest.raises(ValueError):
        next(reader)
    counts = reader.get_state()["parents"]["0:all-zero"]["counts"]
    assert counts["attempted"] == 5
    assert counts["rejected"] == {"zero_targets": 5}
    reader.close()


def test_authoritative_zero_target_count_is_refused_before_division():
    token_weights(["one-target", "useful"], [1, 1], [3, 2], [1, 10])
    with pytest.raises(ValueError):
        token_weights(["empty-targets", "useful"], [1, 1], [3, 2], [0, 10])


def test_dependency_change_preserves_global_resume_only_when_derived_order_matches():
    manifest = Manifest({"name": "test"}, revision="before")
    plan = Plan("global", manifest, {"seed": 0}, {"accepted_ids": fingerprint([1, 3, 5])})
    changed = dataclasses.replace(plan, manifest=dataclasses.replace(manifest, revision="after"))
    assert changed.check(plan.record()) == ("revision",)
    changed = dataclasses.replace(changed, derived={"accepted_ids": fingerprint([1, 3, 7])})
    with pytest.raises(ValueError):
        changed.check(plan.record())


def test_an_exact_resume_refuses_a_step_that_makes_other_values():
    """Shift(1) and Shift(2) keep every row in its place, so the order and its
    derived facts agree; the step's identity is what says the rows differ."""
    def reader(by):
        shifted = Pipeline(source(8)).map(Shift(by)).shuffle(0).repeat().batch(2)
        return compile(shifted, loading=READ).train(DataPartition())

    first = reader(1)
    try:
        next(first)
        state = first.get_state()
    finally:
        first.close()
    same = reader(1)
    try:
        same.set_state(state)
    finally:
        same.close()
    other = reader(2)
    try:
        with pytest.raises(ValueError):
            other.set_state(state)
    finally:
        other.close()


SHIFTED = """import dataclasses


@dataclasses.dataclass(frozen=True)
class Shifted:
    by: int
    rejects = False

    def __call__(self, row):
        return {**row, "id": row["id"] + self.by}
"""


GIVEN = """import dataclasses
from {shifted} import Shifted


@dataclasses.dataclass(frozen=True)
class Given:
    by: int
    version: int = 1

    def __call__(self, metadata):
        return Shifted(self.by)
"""


@pytest.mark.parametrize("read", ["indexed", "memory"])
def test_an_edit_to_a_configured_factorys_module_alone_is_environment(tmp_path, monkeypatch, read):
    """A configured step's factory comes from a module of its own. An edit to
    that file that leaves the factory alone changes only the environment, so
    an exact resume reads on, where the plan names the step and where a
    memory read's rows name the steps that made them; the same factory
    configured otherwise, to make the same step, is refused."""
    shifted = tmp_path / f"shifted_{uuid4().hex}.py"
    shifted.write_text(SHIFTED)
    factory = GIVEN.format(shifted=shifted.stem)
    module = tmp_path / f"given_{uuid4().hex}.py"
    module.write_text(factory)
    monkeypatch.syspath_prepend(str(tmp_path))
    given = importlib.import_module(module.stem).Given

    def reader(version):
        configured = Configured(given(1, version), needs=())
        pipeline = Pipeline(source(8)).map(configured).shuffle(0).repeat().batch(2)
        return compile(pipeline, read=read, loading=READ).train(DataPartition())

    first = reader(1)
    try:
        next(first)
        state = first.get_state()
        expected = take(first, 2)
    finally:
        first.close()
    module.write_text(factory + "# an edit that changes no code\n")
    edited = reader(1)
    try:
        edited.set_state(state)
        assert edited.manifest_changes == ("modules",)
        assert take(edited, 2) == expected
    finally:
        edited.close()
    other = reader(2)
    try:
        with pytest.raises(ValueError):
            other.set_state(state)
    finally:
        other.close()


def test_an_exact_resume_refuses_a_rewritten_file_whose_token_split_keeps_its_spans(tmp_path):
    """A Parquet file rewritten with as many documents of the same lengths and
    no length column leaves the token split's spans and every order fact
    alike; the file's pinned version in the source snapshot refuses it."""
    path = tmp_path / "documents.parquet"

    def write(start):
        staged = tmp_path / "staged.parquet"
        documents = [list(range(start + 4 * index, start + 4 * index + 4)) for index in range(4)]
        pq.write_table(pa.table({"text": documents}), staged)
        os.replace(staged, path)

    def reader():
        split = Pipeline(Files(str(path))).token_split(0.25, held=False).batch(1)
        return compile(split, read="indexed", loading=READ).train(DataPartition())

    write(0)
    first = reader()
    try:
        next(first)
        state = first.get_state()
    finally:
        first.close()
    same = reader()
    try:
        same.set_state(state)
    finally:
        same.close()
    write(100)
    rewritten = reader()
    try:
        with pytest.raises(ValueError):
            rewritten.set_state(state)
    finally:
        rewritten.close()


def test_lambda_identity_requires_explicit_non_exact_resume():
    identity = StepIdentity.of(lambda row: row)
    plan = Plan("global", Manifest({}, (identity,)), {"seed": 0})
    with pytest.raises(ValueError):
        plan.check(plan.record())
    assert plan.check(plan.record(), resume="non-exact") == ()


@dataclasses.dataclass(frozen=True)
class Shifting:
    by: int

    def __call__(self, metadata):
        return Shift(self.by)


@pytest.mark.parametrize("read", ["indexed", "memory"])
def test_a_lambda_step_needs_a_non_exact_resume_whether_or_not_memory_ran_it(read):
    """A lambda has no stable identity, so a plan that runs one is not exact
    and resumes only with resume='non-exact'. Read into memory beside a
    configured step, the lambda is one of the steps the memory read already
    ran into the rows it holds, which the plan counts as it counts its own."""
    def compiled(resume):
        lambdas = Pipeline(source(8)).map(lambda row: row)
        return compile(lambdas.map(Configured(Shifting(1), needs=())).batch(2),
                       read=read, loading=READ, resume=resume)

    data = compiled("exact")
    assert data.plan.record()["exact"] is False
    first = data.train(DataPartition())
    try:
        next(first)
        state = first.get_state()
    finally:
        first.close()
    again = compiled("exact").train(DataPartition())
    try:
        with pytest.raises(ValueError):
            again.set_state(state)
    finally:
        again.close()
    resumed = compiled("non-exact").train(DataPartition())
    try:
        resumed.set_state(state)
    finally:
        resumed.close()


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
    manifest = Manifest.of(stored.fingerprint(), (), fill={"read": "memory", "row_ids": "indexed"})
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


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_causal_windows_reject_per_document_field_drift_in_every_read(read):
    stored = Memory([
        {"text": np.asarray([1, 2], np.int32), "roles": np.asarray([10], np.int8)},
        {"text": np.asarray([3], np.int32), "roles": np.asarray([20, 30], np.int8)},
    ])
    windows = Pipeline(stored).windows(2, fields=("text", "roles"))
    with pytest.raises(ValueError):
        list(iterate(windows, read=read, loading=READ))


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
@pytest.mark.parametrize("filtered", [False, True])
def test_chunk_statistics_count_fanout_per_input_document_and_resume(read, filtered):
    from dew.data.text import Chunks

    stored = Memory([
        {"text": np.arange(1, 8, dtype=np.int32)},
        {"text": np.arange(11, 15, dtype=np.int32)},
    ])
    pipeline = Pipeline(stored, name="chunks").flat_map(Chunks(3, overlap=1))
    if filtered:
        pipeline = pipeline.filter(lambda row: len(row["text"]) == 3, identity="full-chunk@1")
    data = compile(pipeline.repeat(1).batch(1), read=read, loading=READ)
    reader = data.train(DataPartition())
    try:
        first = take(reader, 1)
        saved = reader.get_state()
        rest = take(reader, 10)
        metrics = dict(data.statistics())
        expected = [
            [1, 2, 3], [3, 4, 5], [5, 6, 7], [11, 12, 13], [13, 14],
        ]
        assert [row["text"][0] for row in first + rest] == (expected[:-1] if filtered else expected)
        assert metrics["data/chunks_per_record/chunks"] == 2.5
    finally:
        reader.close()
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(saved)
        assert take(resumed, 10) == rest
        assert data.statistics()["data/chunks_per_record/chunks"] == 2.5
    finally:
        resumed.close()


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
@pytest.mark.parametrize("stride", [1, 2, 3, 7])
def test_aligned_causal_windows_share_spans_and_resume_across_document_edges(read, stride):
    text = np.arange(1, 15, dtype=np.int32)
    roles = text + 100
    cuts = (0, 2, 3, 3, 8, 14)
    stored = Memory([{"text": text[start:stop], "roles": roles[start:stop]}
                     for start, stop in itertools.pairwise(cuts)])
    pipeline = Pipeline(stored).windows(2, stride=stride, fields=("text", "roles")).repeat(1).batch(1)
    data = compile(pipeline, read=read, loading=READ)
    reader = data.train(DataPartition())
    try:
        first = take(reader, 1)
        saved = reader.get_state()
        rest = take(reader, 20)
        assert first + rest == [{"text": [text[start:start + 3].tolist()],
                                 "roles": [roles[start:start + 3].tolist()]}
                                for start in range(0, len(text) - 2, stride)]
    finally:
        reader.close()
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(saved)
        assert take(resumed, 20) == rest
    finally:
        resumed.close()


def test_validation_takes_accepted_rows_before_the_share_cut():
    prefix = Pipeline(source(11)).filter_map(Odd()).take(3).batch(4)
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
    with pytest.raises(ValueError):
        changed.set_state(state)
    changed.close()


@pytest.mark.parametrize("tiny_targets", [0, 1], ids=["broken", "working"])
def test_token_mixer_probes_an_unselected_tiny_share_before_emitting_a_row(tiny_targets):
    """A share weighted too small to be chosen is still probed first: one
    whose rows hold no targets exhausts its budget before the mixture
    emits a row, and one whose rows hold one lets it."""
    parents = [grain.MapDataset.source([{"targets": 3}]).repeat().to_iter_dataset(),
               grain.MapDataset.source([{"targets": tiny_targets}]).repeat().to_iter_dataset()]
    reader = iter(TokenDeficit(parents, [1, 1e-9], tokens=Targets(), names=["normal", "tiny"],
                              budget=SearchBudget(consecutive=3)))
    try:
        if tiny_targets:
            next(reader)
            return
        with pytest.raises(ValueError):
            next(reader)
    finally:
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


@pytest.mark.parametrize("after", ["mix", "windows"])
def test_map_with_index_is_refused_where_a_row_has_no_source_position(after):
    """A mixture's rows and windows' rows have no one place in a source: each
    compiles, and map_with_index after it is refused."""
    if after == "mix":
        rows, read, batch = mix([Pipeline(source()), Pipeline(source())], weights=(1, 1)), None, 4
    else:
        rows, read, batch = Pipeline(Memory([{"text": np.arange(9, dtype=np.int32)}])).windows(4), "memory", 1
    compile(rows.batch(batch), read=read, loading=READ)
    with pytest.raises(ValueError):
        compile(rows.map_with_index(Placed()).batch(batch), read=read, loading=READ)


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

    compile(Pipeline(source()).batch(4), loading=READ)
    with pytest.raises(ValueError):
        compile(Pipeline(source()).then(Foreign()).batch(4), loading=READ)


def test_shares_agree_on_one_identity_and_resume_only_their_own_rows():
    """A per-share order is agreed across the pool without the share index,
    which only a share's saved position keeps. (Held in memory, the order is
    global and its position fits every share.)"""
    thirds = Pipeline(source()).filter(lambda row: int(row["id"]) % 3 > 0, identity="thirds@1")
    data = compile(thirds.batch(4), loading=READ, read="indexed")
    first, second = data.train(DataPartition(0, 2)), data.train(DataPartition(1, 2))
    again = data.train(DataPartition(0, 2))
    try:
        assert first.identity() == second.identity()
        next(first)
        again.set_state(first.get_state())
        with pytest.raises(ValueError):
            second.set_state(first.get_state())
    finally:
        first.close()
        second.close()
        again.close()


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


def test_a_plan_identity_is_its_rows_on_any_host_and_share():
    """A pool agrees on the plan's identity: processes that read other shares
    of the same rows with other code agree, and processes whose source holds
    other rows do not."""
    manifest = Manifest({"rows": "one"}, revision="before")
    plan = Plan("per-share", manifest, {"seed": 0}, topology={"shares": 2, "share": 0})
    elsewhere = dataclasses.replace(
        plan,
        manifest=dataclasses.replace(manifest, revision="after", platform="another host"),
        topology={"shares": 2, "share": 1},
    )
    other = dataclasses.replace(plan, manifest=dataclasses.replace(manifest, source={"rows": "two"}))
    assert elsewhere.identity() == plan.identity() != other.identity()


def test_a_memory_read_counts_a_missing_row_alike_with_or_without_a_cache():
    """A row the source holds as None is left out and counted missing,
    whether the memory read fills a cache with it or not."""
    stored = Memory([{"id": np.int32(0)}, None, {"id": np.int32(2)}])
    for pipeline in (Pipeline(stored), Pipeline(stored).cache(Cache(f"memory://{uuid4().hex}"))):
        data = compile(pipeline.batch(1), read="memory", loading=READ)
        assert data.records == 2
        assert [count.rejected["missing"] for count in data.counts.values()] == [1]


def test_an_indexed_read_batches_the_records_memory_holds_in_full_batches():
    """A source held in memory may hold a missing record. An indexed order
    batches the records present, so every batch is full and the order counts
    three records."""
    stored = Memory([{"id": np.int32(0)}, None, {"id": np.int32(2)}, {"id": np.int32(3)}])
    data = compile(Pipeline(stored).batch(2), read="indexed", loading=READ)
    reader = data.train(DataPartition())
    try:
        batches = take(reader, 3)
    finally:
        reader.close()
    assert data.records == 3
    assert all(len(batch["id"]) == 2 for batch in batches)
    assert sorted(value for batch in batches for value in batch["id"]) == [0, 0, 2, 2, 3, 3]


def _numbered(index: int, row: dict) -> dict:
    return {**row, "at": np.int32(index)}


def test_map_with_index_reads_each_row_at_its_source_position_in_every_epoch():
    """Grain counts an index on across the epochs a repeat reads, but a row's
    source position is its place in the source: four rows read twice over
    are numbered by their ids both times, held whole or as the records
    memory holds."""
    whole = Pipeline(source(4))
    missing = Pipeline(Memory([{"id": np.int32(0)}, None, {"id": np.int32(2)}, {"id": np.int32(3)}]))
    for pipeline, rows in ((whole, 4), (missing, 3)):
        numbered = pipeline.map_with_index(_numbered, identity="at@1")
        data = compile(numbered.batch(2), read="indexed", loading=READ)
        reader = data.train(DataPartition())
        try:
            batches = take(reader, rows)
        finally:
            reader.close()
        assert all(batch["at"] == batch["id"] for batch in batches)


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
    mixed = mix([Pipeline(source(16)), Pipeline(other)], weights=(1, 1))
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
    again = reader(0.25)
    try:
        again.set_state(state)
    finally:
        again.close()
    half = reader(0.5)
    try:
        with pytest.raises(ValueError):
            half.set_state(state)
    finally:
        half.close()


@pytest.mark.parametrize("grown", ["in the source", "by a step"])
def test_a_resume_is_refused_where_the_same_token_split_keeps_other_documents(grown):
    """Four 4-token documents split at a quarter train on documents 1-3; with
    the first grown to eight tokens the same split trains on its last three
    tokens and on documents 1-3, so a resume after one row would read
    document 1 again. The first document grows in the source, which keeps its
    name and row count, or in a step without a stable identity, which a
    non-exact resume cannot check; the split's spans in the derived order
    refuse both."""
    def reader(first):
        sizes = (first if grown == "in the source" else 4, 4, 4, 4)
        documents = [{"text": np.arange(4 * index, 4 * index + size, dtype=np.int32)}
                     for index, size in enumerate(sizes)]
        pipeline = Pipeline(Memory(documents, identity="docs"))
        if grown == "by a step":
            pipeline = pipeline.map(
                lambda row: {"text": np.resize(row["text"], first) if row["text"][0] == 0 else row["text"]},
            )
        split = pipeline.token_split(0.25, held=False).batch(1)
        resume = "exact" if grown == "in the source" else "non-exact"
        return compile(split, read="indexed", loading=READ, resume=resume).train(DataPartition())

    four = reader(4)
    try:
        assert next(four)["text"].tolist() == [[4, 5, 6, 7]]
        state = four.get_state()
    finally:
        four.close()
    again = reader(4)
    try:
        again.set_state(state)
    finally:
        again.close()
    eight = reader(8)
    try:
        with pytest.raises(ValueError):
            eight.set_state(state)
    finally:
        eight.close()


def test_a_token_split_on_a_mixture_within_a_mixture_records_its_spans():
    """A split written on a mixture of three corpora, itself mixed with a
    fourth, splits the inner mixture's order as the inner mixture's own
    analysis builds it. Its spans are part of the derived order, and an inner
    document grown from four tokens to eight changes them."""
    def spans(first):
        def corpus(start, sizes):
            return Pipeline(Memory([{"text": np.arange(start + 10 * index, start + 10 * index + size,
                                                       dtype=np.int32)}
                                    for index, size in enumerate(sizes)]))

        inner = mix([corpus(0, (first, 4)), corpus(100, (4, 4)), corpus(200, (4, 4))])
        mixed = mix([inner.token_split(0.25, held=False), corpus(300, (4, 4))])
        return compile(mixed.batch(1), read="indexed", loading=READ).plan.derived["token_spans"]

    assert spans(4) != spans(8)


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
    with pytest.raises(ValueError):
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


def test_a_scored_memory_branch_that_shuffles_before_its_split_holds_out_half_its_tokens():
    """A memory read of a scored branch is analyzed as scored too, so a token
    split after a shuffle, in an order a filter cuts per share, holds out the
    first half of the shuffled documents' tokens."""
    documents = Memory(tuple({"text": np.arange(4, dtype=np.int32) + 4 * index} for index in range(4)))
    split = Pipeline(documents).shuffle(0).token_split(0.5, held=True)
    held = split.filter(lambda row: True, identity="all@1")
    data = compile(Pipeline(source()).batch(2), validation=held.batch(1), read="memory", loading=READ)
    assert data.val is not None
    reader = data.val(DataPartition())
    try:
        tokens = [value for batch in reader for row in np.asarray(batch["text"]) for value in row.tolist()]
    finally:
        reader.close()
    assert len(tokens) == len(set(tokens)) == 8


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
    inner = mix([Pipeline(source(8)), Pipeline(source(4))], weights=(1, 1))
    nested = mix([Pipeline(source(12)), inner], weights=(1, 1))
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
    from dew.data.sources.tokens import TokenFiles
    from dew.data.text import Chunks

    documents = [np.append(np.full(size, index + 1), 0).astype(np.int32)
                 for index, size in enumerate((5, 3, 9, 2, 7, 4, 6, 1))]
    (tmp_path / "train.bin").write_bytes(np.concatenate(documents).astype(np.uint16).tobytes())
    (tmp_path / "meta.json").write_text(json.dumps({
        "tokenizer": "byte", "dtype": "uint16", "vocab_size": 256, "eos_id": 0,
        "train_tokens": sum(len(document) for document in documents)}))
    store = TokenFiles(str(tmp_path)).provision()

    def windows(pipeline, read):
        rows = iterate(pipeline.flat_map(Chunks(8)).pack(8, bins=2), read=read, loading=READ)
        return [{name: np.asarray(value).tolist() for name, value in row.items() if name[:5] != "_dew_"}
                for row in rows]

    # The 10-token document is cut where the chunk step says, which the plan of lengths cuts alike.
    lowering = compile(Pipeline(store).flat_map(Chunks(8)).pack(8, bins=2).batch(2), read="indexed",
                       loading=READ).plan.lowering
    assert lowering == "global"
    memory = Pipeline(Memory([{"text": document} for document in documents]))
    assert windows(Pipeline(store), "indexed") == windows(memory, "memory")


def test_a_memory_read_counts_valid_targets_where_it_packs():
    """Windows are kept by their own targets, however a later step lays them
    out: the one-token document's window has none and is left out."""
    documents = [{"text": np.arange(size, dtype=np.int32)} for size in (8, 1, 8)]
    laid = Pipeline(Memory(documents)).pack(8, bins=1).map(Laid(), identity="laid@1").batch(1)
    assert compile(laid, read="memory", loading=READ).records == 2


def test_each_pass_declares_the_most_documents_its_own_windows_hold():
    """Training windows of one document each and validation windows of four, packed with their
    roles: each pass declares the most its own windows hold (`Coordinates.max_segments`), not the
    other's nor its window's length."""
    long = Memory(tuple({"text": np.arange(8, dtype=np.int32)} for _ in range(4)))
    short = Memory(tuple({"text": np.arange(2, dtype=np.int32), "text_roles": np.ones(2, np.int32)}
                         for _ in range(8)))
    data = compile(Pipeline(long).pack(9).batch(2), read="memory", loading=READ,
                   validation=Pipeline(short).pack(9, fields=("text", "text_roles")).batch(2))
    for reader, most in ((data.train, 1), (data.val, 4)):
        assert reader is not None
        opened = reader(DataPartition())
        try:
            laid = next(opened)["text_coordinates"]
        finally:
            opened.close()
        assert int(laid.segments.max()) == laid.max_segments == most


def test_a_corpus_packed_where_the_read_cannot_see_declares_its_windows_length():
    """A mixture of windows a pack makes and windows a corpus holds already packed: the read cannot
    see how many documents the second's windows hold, so the batches declare a window's length
    rather than the first's most, which the second's windows pass."""
    packed = Memory(tuple({"text": np.arange(8, dtype=np.int32)} for _ in range(4))).named("packed").pack(9)
    ready = Memory(tuple({"text": np.arange(9, dtype=np.int32),
                          "text_segment_ids": np.asarray([1, 1, 2, 2, 3, 3, 4, 4, 0], np.int32),
                          "text_positions": np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0], np.int32)}
                         for _ in range(4))).named("ready")
    opened = compile(mix([packed, ready]).batch(4), read="memory", loading=READ).train(DataPartition())
    try:
        laid = [batch["text_coordinates"] for batch in itertools.islice(opened, 2)]
    finally:
        opened.close()
    assert max(int(coordinates.segments.max()) for coordinates in laid) == 4
    assert {coordinates.max_segments for coordinates in laid} == {9}


def test_a_window_holding_more_documents_than_its_read_declares_is_refused():
    """A sequence kernel sized by the declared documents would run the rest into the next row's."""
    from dew.data.steps.generic import Coordinated

    window = {"text": np.zeros((1, 6), np.int32),
              "text_segment_ids": np.asarray([[1, 1, 2, 3, 0, 0]], np.int32),
              "text_positions": np.asarray([[0, 1, 0, 0, 0, 0]], np.int32)}
    assert Coordinated(("text",), 3).map(window)["text_coordinates"].max_segments == 3
    with pytest.raises(ValueError):
        Coordinated(("text",), 2).map(window)


def test_a_worker_batch_shares_its_large_columns_and_pickles_the_small_ones():
    """A loading worker stacks a batch's large numeric column straight into shared memory, which
    the reader maps without a copy, and keeps its captions and small columns as plain arrays,
    which cross pickled: a shared segment costs the reader more to map, close and unlink than a
    few KiB cost to copy."""
    from grain._src.python.ipc.shared_memory_array import SharedMemoryArray

    from dew.data.dataset import record_batch

    rows = [{"image": np.full((64, 64, 3), index, np.uint8), "caption": f"a photo {index}",
             "label": np.int32(index)} for index in range(32)]
    batch = record_batch(rows, shared=True)
    try:
        assert isinstance(batch["image"], SharedMemoryArray)
        assert type(batch["caption"]) is np.ndarray and type(batch["label"]) is np.ndarray
        assert batch["caption"].tolist() == [f"a photo {index}" for index in range(32)]
        assert batch["label"].tolist() == list(range(32))
        assert batch["image"][:, 0, 0, 0].tolist() == list(range(32))
    finally:
        for column in batch.values():
            if isinstance(column, SharedMemoryArray):
                column.unlink_on_del()


@pytest.mark.parametrize("read", ["memory", "streamed"])
def test_a_rebatched_loader_declares_the_rows_its_batches_hold(read):
    """A loader's batch is the rows of the batches it yields, the last rebatch's size, which the
    trainer's preflight and microbatch checks read; it once declared the first batch's."""
    rows = Memory([{"x": np.int32(index)} for index in range(32)])
    data = compile(Pipeline(rows).repeat().batch(8).rebatch(4), read=read, loading=READ)
    stream = data.train(DataPartition())
    try:
        assert [len(next(stream)["x"]) for _ in range(3)] == [data.batch] * 3 == [4] * 3
    finally:
        stream.close()


@pytest.mark.parametrize("read", ["memory", "streamed"])
def test_a_validation_pass_holds_every_record_whatever_size_training_rebatches_to(read):
    """A validation pass batches by its own size, so training rebatched to four rows leaves each
    eight-row held-out batch whole, every held-out record scored once."""
    rows = Memory([{"x": np.int32(index)} for index in range(32)])
    held = Memory([{"x": np.int32(100 + index)} for index in range(16)])
    data = compile(Pipeline(rows).repeat().batch(8).rebatch(4), validation=Pipeline(held).batch(8), read=read,
                   loading=READ)
    assert data.val is not None
    scored = []
    for batch in data.val(DataPartition()):
        valid = np.asarray(batch.get(VALID_ROWS, np.ones(len(batch["x"]), bool)))
        scored += [int(value) for value, kept in zip(np.asarray(batch["x"]), valid, strict=True) if kept]
    assert sorted(scored) == list(range(100, 116))


def test_a_step_drops_records_only_when_it_says_so():
    """A step keeps every record unless it declares `rejects`, whatever its annotations read: an
    unannotated indexed step and an annotated one compile to one global order with an epoch
    count, a step that declares `rejects` drops what it rejects, and one that rejects without
    declaring it is refused, where an unannotated step was once taken to drop records."""

    def unannotated(index, row):
        return {**row, "at": np.int32(index)}

    def annotated(index: int, row: dict) -> dict:
        return {**row, "at": np.int32(index)}

    @dataclasses.dataclass(frozen=True)
    class EvenOnly:
        rejects = True

        def __call__(self, index: int, row: dict) -> dict | Rejected:
            return row if index % 2 == 0 else Rejected("odd")

    def undeclared(index, row):
        return row if index % 2 == 0 else Rejected("odd")

    rows = Pipeline(source(8))
    for step in (unannotated, annotated):
        data = compile(rows.map_with_index(step).batch(4), read="indexed", loading=READ)
        assert data.records == 8
    kept = compile(rows.map_with_index(EvenOnly()).repeat(1).batch(2), read="indexed", loading=READ)
    stream = kept.train(DataPartition())
    try:
        assert sorted(int(value) for batch in stream for value in batch["id"]) == [0, 2, 4, 6]
    finally:
        stream.close()
    refused = compile(rows.map_with_index(undeclared).repeat(1).batch(2), read="indexed", loading=READ)
    stream = refused.train(DataPartition())
    try:
        with pytest.raises(ValueError):
            list(stream)
    finally:
        stream.close()


@pytest.mark.parametrize("by", ["records", "tokens"])
@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_a_mixture_of_one_dataset_reads_that_dataset(by, read):
    """A mixture whose members a run picks may come down to one, which reads as that member
    alone, by records or by tokens, in every read."""
    from dew.data.steps.text import CountTargets

    rows = Memory([{"text": np.full(3, index, np.int32), "id": np.int32(index)} for index in range(6)])
    member = Pipeline(rows, name="only").filter_map(CountTargets())
    data = compile(mix([member.repeat(1)], by=by).batch(2), read=read, loading=READ)
    stream = data.train(DataPartition())
    try:
        assert sorted(int(value) for batch in stream for value in batch["id"]) == list(range(6))
    finally:
        stream.close()


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_a_corpus_smaller_than_a_batch_fills_one_by_its_repeats(read):
    """A corpus of two records repeated fills a batch of eight, as Grain's repeat and batch do,
    and a run counted in steps reads it; a run counted in epochs is refused, since one pass
    over the two records fills no batch. Repeated after the batch, a pass holds no batch to
    replay, so the read is refused rather than replaying nothing forever, whatever limit
    follows, in every read; eight records make a batch either way."""
    for count, repeated in ((2, "before"), (2, "after"), (8, "after"), (2, "after, then take"),
                            (8, "after, then take")):
        rows = Pipeline(Memory([{"id": np.int32(index)} for index in range(count)]))
        chain = rows.repeat().batch(8) if repeated == "before" else rows.batch(8).repeat()
        data = compile(chain.take(1) if repeated.endswith("take") else chain, read=read, loading=READ)
        if count < 8 and repeated.startswith("after"):
            with pytest.raises(ValueError):
                refused = data.train(DataPartition())
                try:
                    next(refused)
                finally:
                    refused.close()
            continue
        stream = data.train(DataPartition())
        try:
            assert sorted(int(value) for value in next(stream)["id"]) == sorted(
                index % count for index in range(8))
        finally:
            stream.close()
        if count < 8:
            with pytest.raises(ValueError):
                data.epoch_steps(1)


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
@pytest.mark.parametrize("batched", ["before", "after"])
def test_an_endless_repeat_resumed_at_the_end_of_a_pass_reads_on(read, batched):
    """A position saved just as a pass of an endless repeat ends restores to that end, and the
    read goes on into the next pass as the uninterrupted one does, rather than taking the
    restored end for a pass that read nothing; a repeat after the batch reads its passes as a
    stream in every read."""
    rows = Pipeline(Memory([{"id": np.int32(index)} for index in range(2)]))
    data = compile(rows.repeat().batch(2) if batched == "before" else rows.batch(2).repeat(), read=read,
                   loading=READ)

    def ids(stream, count: int) -> list[int]:
        return [int(value) for batch in itertools.islice(stream, count) for value in batch["id"]]

    whole = data.train(DataPartition())
    try:
        expected = ids(whole, 3)
    finally:
        whole.close()
    stopped = data.train(DataPartition())
    try:
        head = ids(stopped, 1)
        state = stopped.get_state()
    finally:
        stopped.close()
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(state)
        assert head + ids(resumed, 2) == expected
    finally:
        resumed.close()


def test_a_random_step_after_an_endless_repeat_draws_as_after_a_finite_one():
    """A streamed repeat read endlessly guards its passes without changing what a random step
    after it draws: the passes a finite repeat reads draw the same values."""
    rows = Pipeline(source(2)).take(2)

    def draws(chain) -> list[int]:
        data = compile(chain.random_map(Draw()).batch(2), read="streamed", loading=READ)
        stream = data.train(DataPartition())
        try:
            return [int(value) for batch in itertools.islice(stream, 2) for value in batch["draw"]]
        finally:
            stream.close()

    assert draws(rows.repeat()) == draws(rows.repeat(2))


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_a_repeat_after_the_batch_replays_its_first_pass_draws_included(read):
    """A repeat after the batch repeats the batches the first pass made, a random step's draws
    and all, in every read, and a resume reads on as the uninterrupted run does."""
    data = compile(Pipeline(source(2)).random_map(Draw()).batch(2).repeat(), read=read, loading=READ)

    def batches(stream, count: int) -> list[tuple[list[int], list[int]]]:
        return [(np.asarray(batch["id"]).tolist(), np.asarray(batch["draw"]).tolist())
                for batch in itertools.islice(stream, count)]

    whole = data.train(DataPartition())
    try:
        expected = batches(whole, 3)
    finally:
        whole.close()
    assert expected[0] == expected[1] == expected[2]
    stopped = data.train(DataPartition())
    try:
        head = batches(stopped, 1)
        state = stopped.get_state()
    finally:
        stopped.close()
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(state)
        assert head + batches(resumed, 2) == expected
    finally:
        resumed.close()
