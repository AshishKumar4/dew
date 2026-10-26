"""The pipeline's observable order, branch scope and exact positions."""

import dataclasses
import gc
import importlib
import itertools
import json
import multiprocessing
import os
import threading
import weakref
from uuid import uuid4

import grain.python as grain
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from data_support import WideDraw, listed
from reader_support import assert_resumes_where_it_stopped
from token_support import write_token_corpus

from dew import position
from dew.batch import VALID_ROWS
from dew.data.cache import Cache
from dew.data.compile import compile, iterate, validation
from dew.data.dataset import GRAIN_KILL_SECONDS, DataPartition, Loading
from dew.data.mixing import TokenDeficit, token_weights
from dew.data.pipeline import Configured, Pipeline, Rejected, mix
from dew.data.plan import Manifest, Plan, StepIdentity, fingerprint
from dew.data.sources import Files, Memory

READ = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)


@dataclasses.dataclass(frozen=True)
class RowDraws(grain.RandomMapTransform):
    """A draw for each row of a batch."""

    def random_map(self, element, rng):
        return {**element, "draw": rng.integers(0, 2**32, size=len(element["id"]))}


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
class Kept:
    """Keeps the record whose id is `id` and no other."""

    id: int

    def __call__(self, row):
        return int(row["id"]) == self.id


@dataclasses.dataclass(frozen=True)
class Counted:
    """A token row as its corpus and the targets it holds, which batch whatever the row's length."""

    def __call__(self, row):
        return {"source": row["source"], "targets": np.int32(len(row["text"]) - 1)}


def source(count=24):
    return Memory(tuple({"id": np.int32(index)} for index in range(count)))


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
        tags = sorted(value for batch in listed(reader, 4) for value in batch["tag"])
        assert tags == list(range(0, 80, 10))
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
        tags = sorted(value for batch in listed(reader, 4) for value in batch["tag"])
        assert tags == list(range(0, 80, 10))
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
        assert listed(reader, 4) == listed(old, 4)
        assert "dew_global_position" in json.loads(reader.get_state())
    finally:
        reader.close()
        old.close()


def test_global_position_resumes_across_share_counts_with_rng():
    chain = Pipeline(source()).shuffle(7).repeat().random_map(WideDraw()).batch(4)
    data = compile(chain, loading=READ, seed=7)
    reader = data.train(DataPartition())
    listed(reader, 3)
    saved = reader.get_state()
    expected = listed(reader, 5)
    reader.close()
    shares = []
    for index in range(2):
        reader = data.train(DataPartition(index, 2))
        reader.set_state(saved)
        shares.append(listed(reader, 5))
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
    after the batch. Two shares resumed there read on the batches' rows the run reads, each as it
    reads uninterrupted, every leaf included; a random step after the batch draws for each share's
    part of a batch, so one share's draws are not another's."""
    rows = Pipeline(source(16))
    chain = {"take": rows.batch(2).take(5),
             "window shuffle": rows.repeat(1).batch(2).shuffle(7, window=3),
             "random map": rows.repeat().batch(2).random_map(RowDraws()),
             "finite repeat": rows.take(6).batch(2).repeat(2)}[after]
    data = compile(chain, read=read, loading=READ, seed=3)
    expected, saved = assert_resumes_where_it_stopped(lambda: data.train(DataPartition()), listed, 2, 6)
    head = expected[:2]
    shares, uninterrupted = [], []
    for index in range(2):
        reader = data.train(DataPartition(index, 2))
        try:
            reader.set_state(saved)
            shares.append(listed(reader, 6))
        finally:
            reader.close()
        reader = data.train(DataPartition(index, 2))
        try:
            uninterrupted.append(listed(reader, 8)[2:])
        finally:
            reader.close()
    assert shares == uninterrupted
    joined = [{key: [value for pair in zip(left[key], right[key], strict=True) for value in pair]
               for key in left} for left, right in zip(*shares, strict=True)]
    if after == "random map":
        assert all(left["draw"] != right["draw"] for left, right in zip(*shares, strict=True))
        assert [batch["id"] for batch in head + joined] == [batch["id"] for batch in expected]
    else:
        assert head + joined == expected


def test_a_random_step_draws_alike_in_every_read_of_one_order():
    """A random step's draw is keyed by its record's place in the order and by the step's place in
    what the run wrote, never by how the read lowers the steps before it: a memory read, which runs
    the map before it over the whole source, an indexed read and a streamed read of one share draw
    the same values, in a mixture as alone, and a second random step draws apart from the first."""
    rows = Pipeline(source(8)).map(Tagged())
    alone = rows.random_map(WideDraw()).repeat(1).batch(4)
    mixed = mix([rows.random_map(WideDraw()).repeat(1)]).batch(4)

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
    again = draws(rows.random_map(WideDraw()).random_map(WideDraw()).repeat(1).batch(4), "indexed")
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
        return Pipeline(source(8), name=name).map(Member(index)).random_map(WideDraw()).repeat(1)

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
        chain = Pipeline(Files(str(directory / "rows.jsonl"))).map(Member(0)).random_map(WideDraw())
        named = Pipeline(source(8), name="joining").map(Member(1)).random_map(WideDraw()).repeat(1)
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
        return Pipeline(source(8), name=name).map(Member(index)).random_map(WideDraw()).repeat(1)

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


def member_rows(members, by="records", saved=None, batches=6, above=lambda chain: chain, host=(),
                weights=None):
    """The (member, id) rows a streamed mix of `members` delivers over `batches` batches, and its
    position after the first two, resumed from `saved` when given."""
    chain = above(mix(members, weights, by=by)).batch(2)
    data = compile(chain, read="streamed", loading=READ, seed=5, host=host)
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


def corpus(name, source, **options):
    return Pipeline(roled(source, 6, 2, count=8), name=name, **options).repeat()


@pytest.mark.parametrize("by", ["records", "tokens"])
@pytest.mark.parametrize("edit", ["joined-last", "joined-first", "left-last", "left-first"])
def test_a_streamed_mix_resumes_its_members_where_they_were_when_one_joins_or_leaves(by, edit):
    """A streamed mix keeps each member's position under its key, wherever the member sits among
    the others: resumed with a member joined, ahead of the others or after them, the members it had
    read on from where they were and the new one from its first row; resumed with a member gone,
    ahead of the others or after them, the others read on from where they were."""
    kept = [corpus("a", 0), corpus("b", 1)]
    other = [corpus("c", 2)]
    place = (lambda members: other + members) if edit.endswith("first") else (lambda members: members + other)
    before, after = (kept, place(kept)) if edit.startswith("joined") else (place(kept), kept)
    whole, saved = member_rows(before, by)
    resumed, _ = member_rows(after, by, saved=saved)

    def rows_of(rows, source):
        return [index for of, index in rows if of == source]

    read = 2 * 2  # the rows of the two batches read before the position was saved
    for source in (0, 1):
        ahead = rows_of(whole[read:], source)
        again = rows_of(resumed, source)
        # Each kept member reads on as it read uninterrupted, as far as both read.
        assert again and again[:len(ahead)] == ahead[:len(again)]
    if edit.startswith("joined"):
        assert rows_of(resumed, 2)[:1] == [0]


@pytest.mark.parametrize("edit", ["member-steps", "mixture-limit", "host-steps", "mixture-stages"])
def test_a_streamed_mix_refuses_a_resume_whose_kept_rows_changed(edit):
    """A resume takes members joining, leaving or reweighted, and refuses one where what a kept
    member reads or makes changed: its own steps, a limit over the whole mix, the compile's host
    steps, or the mixture's own stages."""
    members = [corpus("a", 0), corpus("b", 1)]
    _, saved = member_rows(members, host=(Shift(1),))
    relabelled = Pipeline(roled(0, 6, 2, count=8), name="a").map(Relabelled()).repeat()
    changed = {
        "member-steps": {"members": [relabelled, corpus("b", 1)], "host": (Shift(1),)},
        "mixture-limit": {"members": members, "weights": [1, 2], "host": (Shift(1),),
                          "above": lambda chain: chain.take(10_000)},
        "host-steps": {"members": members, "weights": [1, 2], "host": (Shift(2),)},
        "mixture-stages": {"members": members, "weights": [1, 2], "host": (Shift(1),),
                           "above": lambda chain: chain.map(Relabelled())},
    }[edit]
    with pytest.raises(ValueError):
        member_rows(saved=saved, **changed)


def test_a_streamed_mix_refuses_a_kept_nested_member_whose_order_changed():
    """A member that is itself a mix keeps its order whole: one whose limit moved from an inner
    member to the inner mix, reading the same sources through the same steps, is refused when the
    outer weights change too, as its saved position counts other rows."""
    def leaf(name, source):
        return Pipeline(roled(source, 6, 2, count=8), name=name)

    # The same steps either way, a limit of five; only where it sits differs.
    limited_inside = mix([leaf("a", 0).take(5), leaf("b", 1)]).named("inner")
    limited_outside = mix([leaf("a", 0), leaf("b", 1)]).take(5).named("inner")
    _, saved = member_rows([limited_inside, corpus("c", 2)])
    with pytest.raises(ValueError):
        member_rows([limited_outside, corpus("c", 2)], weights=[1, 2], saved=saved)


@pytest.mark.parametrize("read", ["streamed", "memory"])
def test_two_members_of_one_source_keep_their_own_caches_when_one_leaves(read):
    """Two members may read one source, one of them through a cache: each member's digest holds its
    own caches, so the one that stays resumes when the other leaves."""
    shared = roled(0, 6, 2, count=8)
    cached = Pipeline(shared, name="cached").cache(Cache(f"memory://{uuid4().hex}")).repeat()
    plain = Pipeline(shared, name="plain").repeat()

    def rows(members, saved=None):
        data = compile(mix(members).batch(2), read=read, loading=READ, seed=5)
        stream = data.train(DataPartition())
        try:
            if saved is not None:
                stream.set_state(saved)
            next(stream)
            return stream.get_state()
        finally:
            stream.close()

    saved = rows([cached, plain])
    if read == "memory":
        # A memory read is a global order, which keeps no member by key: the edit is refused.
        with pytest.raises(ValueError):
            rows([cached], saved)
    else:
        rows([cached], saved)


def test_a_memory_mix_with_a_member_that_accepts_no_row_compiles_and_reads_the_others():
    """A memory read drops a member every row of which its steps reject: the mix compiles and reads
    the other member's rows."""
    empty = Pipeline(roled(0, 6, 2), name="empty").filter(lambda row: False, identity="none@1").repeat()
    full = corpus("full", 1)
    data = compile(mix([empty, full]).workers(1).batch(1), read="memory", loading=READ, seed=5)
    stream = data.train(DataPartition())
    try:
        assert int(next(stream)["source"][0]) == 1
    finally:
        stream.close()


def test_a_streamed_mix_beneath_a_shuffle_refuses_a_member_joining():
    """A shuffle above a mix holds the mix's rows in its window across a saved position, which a
    member joining would change beneath it: that resume is refused rather than replaying rows."""
    def shuffled(chain):
        return chain.shuffle(42, window=10)

    _, saved = member_rows([corpus("a", 0)], above=shuffled)
    with pytest.raises(ValueError):
        member_rows([corpus("a", 0), corpus("b", 1)], above=shuffled, saved=saved)


def test_a_member_joining_a_resumed_mix_is_read_ahead_before_the_mix_gives_a_row():
    """A member that joins a resumed mix has its first row read ahead before the mix gives one, as
    every member has at the start, and resumes beside the others however small its weight."""
    _, saved = member_rows([corpus("a", 0)])
    rows, _ = member_rows([corpus("a", 0), corpus("b", 1)], saved=saved, weights=[1, 1e-9], batches=1)
    assert rows


@pytest.mark.parametrize("by", ["records", "tokens"])
def test_a_streamed_mix_resumed_while_a_member_holds_a_row_read_ahead_reads_on_as_uninterrupted(by):
    """A streamed mix reads every member's first row ahead before it gives a row, so a position it
    saves after its first row falls while the other member holds its first row unread. Resumed
    there, by records or by tokens, the mix gives that row and reads on record for record as the
    uninterrupted one does."""
    members = [corpus("a", 0), corpus("b", 1)]

    def rows(saved=None):
        data = compile(mix(members, by=by).batch(1), read="streamed", loading=READ, seed=5)
        stream = data.train(DataPartition())
        try:
            first = None
            if saved is None:
                first = int(next(stream)["source"][0])
                saved = stream.get_state()
            else:
                stream.set_state(saved)
            read = [(int(batch["source"][0]), int(batch["id"][0])) for batch in itertools.islice(stream, 6)]
            return read, saved, first
        finally:
            stream.close()

    whole, saved, first = rows()
    resumed, _, _ = rows(saved)
    assert (1 - first, 0) in whole
    assert resumed == whole


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
        members = [Pipeline(source(8), name=name).random_map(WideDraw()).repeat(),
                   Pipeline(source(4)).random_map(WideDraw()).repeat()]
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
    return rows.random_map(WideDraw())


@pytest.mark.parametrize("read", ["indexed", "streamed"])
def test_a_native_step_draws_by_the_runs_seed_and_resumes(read):
    """A native step (`Pipeline.grain`) is handed rows seeded by the run, as Grain's own pipelines
    seed theirs, so a random step it adds without a seed draws: by the run's seed, apart from a
    second native step's draws, and on in a resumed run as in the uninterrupted one."""
    rows = Pipeline(source(8)).grain(drawn_natively)

    def draws(chain, seed):
        data = compile(chain.repeat().batch(2), read=read, loading=READ, seed=seed)
        expected, _ = assert_resumes_where_it_stopped(lambda: data.train(DataPartition()), listed, 2, 2)
        return [draw for batch in expected for draw in batch["draw"]]

    assert draws(rows, 5) != draws(rows, 6)
    assert draws(rows.grain(drawn_natively), 5) != draws(rows, 5)


@pytest.mark.parametrize("order", ["global", "per-share"])
def test_a_position_saved_under_another_plan_version_is_refused(order):
    """A plan's version names how Dew makes rows from it, which its fields do not say: a position
    an earlier version saved, whose random steps drew by another key, is refused rather than
    resumed into other draws, in a global order and in one read per share; one this version saved
    resumes."""
    rows = Pipeline(source(8)).random_map(WideDraw())
    chain = rows if order == "global" else rows.grain(drawn_natively)
    data = compile(chain.repeat().batch(2), read="indexed", loading=READ, seed=7)
    assert data.plan is not None and data.plan.lowering == order
    stopped = data.train(DataPartition())
    try:
        listed(stopped, 2)
        saved = stopped.get_state()
        expected = listed(stopped, 2)
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
        assert listed(resumed, 2) == expected
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
        assert listed(train, 1)[0]["id"] == [110, 111, 112, 113]
        saved = train.get_state()
        expected = listed(train, 2)
        if read != "streamed":
            assert train.records == 12
            assert train.order == json.loads(saved)["dew_global_position"]["order"]
        assert listed(held, 1)[0]["id"] == [100, 101, 102, 103]
    finally:
        train.close()
        held.close()
    train.close()
    with pytest.raises(StopIteration):
        next(train)
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(saved)
        assert listed(resumed, 2) == expected
    finally:
        resumed.close()


def test_rejection_iterator_resumes_exactly_and_refuses_topology_change():
    data = compile(Pipeline(source()).shuffle(2).repeat().filter_map(Odd()).batch(3), loading=READ, seed=2)
    reader = data.train(DataPartition())
    listed(reader, 3)
    saved = reader.get_state()
    rest = listed(reader, 4)
    reader.close()
    resumed = data.train(DataPartition())
    resumed.set_state(saved)
    assert listed(resumed, 4) == rest
    resumed.close()
    changed = data.train(DataPartition(0, 3))
    with pytest.raises(ValueError):
        changed.set_state(saved)
    changed.close()


def test_random_validation_is_refused():
    compile(Pipeline(source()).batch(4), validation=Pipeline(source()).shuffle(0).batch(4), loading=READ)
    with pytest.raises(ValueError):
        compile(Pipeline(source()).batch(4),
                validation=Pipeline(source()).shuffle(0).random_map(WideDraw()).batch(4), loading=READ)


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("read", ["memory", "indexed"])
def test_worker_boundary_keeps_downstream_record_steps_and_exact_positions(workers, read):
    pipeline = Pipeline(source()).repeat().workers(workers).map(Shift(100)).take(12).batch(4)
    data = compile(pipeline, read=read, loading=READ)
    for share in range(2):
        reader = data.train(DataPartition(share, 2))
        first = next(reader)
        state = reader.get_state()
        expected = listed(reader, 8)
        reader.close()
        ids = [*first["id"].tolist(), *(value for row in expected for value in row["id"])]
        assert ids == [100 + share + 2 * index for index in range(12)]
        resumed = data.train(DataPartition(share, 2))
        resumed.set_state(state)
        assert listed(resumed, 8) == expected
        resumed.close()


def operators(dataset) -> int:
    """The Grain datasets `dataset` reads through, itself included."""
    seen, pending = set(), [dataset]
    while pending:
        node = pending.pop()
        if id(node) not in seen:
            seen.add(id(node))
            pending.extend(node.parents)
    return len(seen)


@pytest.mark.parametrize("read", ["indexed", "streamed"])
def test_adjacent_record_steps_read_through_one_grain_operator(read):
    """Adjacent record steps lower as one Grain operator, as tf.data fuses maps and filters: a draw,
    a map, a filter and two maps read through as many Grain datasets as the filter alone, and make
    the records the steps make one after another."""
    built = []

    def watched(dataset):
        built.append(dataset)
        return dataset

    def read_through(chain):
        built.clear()
        rows = list(chain.grain(watched, identity="watched@1").read(read).loading(READ))
        return operators(built[0]), rows

    alone, kept = read_through(Pipeline(source(8)).filter(odd_id))
    steps = Pipeline(source(8)).random_map(WideDraw()).map(Shift(1)).filter(odd_id).map(Shift(10))
    fused, made = read_through(steps.map(Tagged()))
    assert fused == alone
    assert [int(row["id"]) for row in kept] == [1, 3, 5, 7]
    assert [(int(row["id"]), int(row["tag"])) for row in made] == [(id, id * 10) for id in (11, 13, 15, 17)]


def odd_id(row) -> bool:
    return bool(int(row["id"]) % 2)


@dataclasses.dataclass(frozen=True)
class Thread:
    """The record with the thread that ran this step on it, under `field`."""

    field: str

    def __call__(self, row):
        return {**row, self.field: threading.get_ident()}


def test_a_step_after_a_filter_runs_on_the_thread_that_ran_the_steps_before_it():
    """A step written after a filter runs where the steps before it ran, on the read's own threads,
    where it once ran alone in the reading process after the filter's records were counted out."""
    chain = Pipeline(source(16)).map(Thread("before")).filter(odd_id).map(Thread("after"))
    rows = list(chain.read("indexed").loading(Loading(workers=0, threads=4, read_buffer=4)))
    assert [int(row["id"]) for row in rows] == list(range(1, 16, 2))
    assert all(row["after"] == row["before"] != threading.get_ident() for row in rows)


def test_worker_processes_read_a_global_order_in_the_batches_one_process_reads():
    """Each worker reads whole batches of an indexed order by index, and the
    reader interleaves them back into the order one process reads."""
    pipeline = Pipeline(source()).shuffle(5).repeat().batch(4)

    def batches(workers):
        reader = compile(pipeline, seed=5, loading=dataclasses.replace(READ, workers=workers)).train(
            DataPartition())
        try:
            return listed(reader, 9)
        finally:
            reader.close()

    assert batches(2) == batches(0)


@pytest.mark.parametrize("bound", ["cpus", "memory"])
def test_automatic_loading_starts_no_more_workers_than_the_hosts_cpus_and_memory_hold(monkeypatch, bound):
    """Automatic loading sizes its workers from the CPUs this process may run on and the memory it
    may take, together. On two CPUs of a 64-core machine it reads in at most one worker, the other
    CPU left to the reading process; in 512 MiB, at most two, each a process of its own that
    imports the program again. Grain's pick read every core the machine has and counted a worker's
    memory as one batch, so it started 64 workers in both."""
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(2 if bound == "cpus" else 16)))
    budget = {"cpus": 4096, "memory": 512}[bound]
    loading = Loading(workers="auto", ram_budget_mb=budget)
    data = compile(Pipeline(source(64)).repeat().batch(8), loading=loading)
    reader = data.train(DataPartition())
    try:
        workers = reader.stop_seconds / GRAIN_KILL_SECONDS - 1
    finally:
        reader.close()
    assert workers <= {"cpus": 1, "memory": 2}[bound]


def test_automatic_loading_holds_its_reads_and_each_workers_batches_within_half_the_memory():
    """Each worker holds its `worker_buffer` batches and its `read_buffer` reads in a process of its
    own, within half the memory the process may take; a read is a batch where the order is read by
    batch, a record where it streams. Eight 64 KiB rows make a 512 KiB batch: 2 MiB holds four reads
    of a batch and no worker, and half a host's memory that holds two and a half workers holds two,
    reading by the batch or by the record. Workers take each CPU but one, at most `max_workers`."""
    from dew.data.dataset import WORKER_PROCESS_BYTES, Host

    rows = grain.MapDataset.source([{"x": np.zeros((8, 16384), np.float32)}] * 6).to_iter_dataset()
    auto = Loading(workers="auto", worker_buffer=2)
    small = dataclasses.replace(auto, ram_budget_mb=2).resolved(rows, reads=1, host=Host(8, 2**40))
    assert (small.workers, small.read_buffer) == (0, 4)
    held = WORKER_PROCESS_BYTES + 2 * 2**19 + 16 * 2**19
    roomy = auto.resolved(rows, reads=1, host=Host(8, 2 * (2 * held + held // 2)))
    assert (roomy.workers, roomy.read_buffer) == (2, 16)
    streamed = auto.resolved(rows, reads=8, host=Host(8, 2 * (2 * held + held // 2)))
    assert streamed.workers == 2 and streamed.read_buffer == 16
    assert auto.resolved(rows, reads=1, host=Host(3, 2**40)).workers == 2
    assert auto.resolved(rows, reads=1, host=Host(1, 2**40)).workers == 0
    assert dataclasses.replace(auto, max_workers=1).resolved(rows, reads=1, host=Host(8, 2**40)).workers == 1


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
    mix = TokenDeficit(corpora, (1, 3), ("short", "long"), tokens=Targets(), names=["short", "long"],
                       labels=["short", "long"])
    reader = iter(mix)
    delivered = [0, 0]
    for row in itertools.islice(reader, 100):
        delivered[row["id"]] += row["targets"]
        # The short corpus within 7 targets of its quarter of all delivered, in whole targets.
        assert abs(4 * int(delivered[0]) - int(sum(delivered))) <= 4 * 7
    saved = reader.get_state()
    rest = listed(reader, 20)
    resumed = iter(mix)
    resumed.set_state(saved)
    assert listed(resumed, 20) == rest
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
    long = long.take(1) if sliced else long
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
class Blocked:
    """A step that adds a prefix run's blocks to a token row, as a user's own map does, leaving the row's
    count as it was: the row's first `prefix` tokens are one block."""

    prefix: int

    def __call__(self, row):
        from dew.nn.visibility import prefix_blocks

        places = np.arange(len(row["text"]), dtype=np.int32)
        blocks = prefix_blocks(np.ones(len(places), np.int32), places, self.prefix, xp=np)
        return {**row, "text_block_groups": blocks}


@dataclasses.dataclass(frozen=True)
class Scored:
    """A delivered row as its corpus and the targets the loss scores of it, by `role`'s rule."""

    role: int | None

    def __call__(self, row):
        from dew.data.steps.text import ValidTargets

        # A row whose roles a projection renamed reads them under their new name; a packed row's
        # segments end its documents' targets at their boundaries, and its blocks leave out the
        # targets they read whole.
        scored = {"text": row["text"], "text_roles": row.get("text_roles", row.get("roles")),
                  **{name: row[name] for name in ("text_segment_ids", "text_block_groups") if name in row}}
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
    assert abs(int(left) - int(right)) <= sum(int(Scored(role)(row)["targets"]) for row in rows)


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_a_token_mixture_keeps_a_rows_count_through_a_step_over_its_other_fields(read):
    """A step that changes only a row's other fields keeps the targets its count's own rule counted,
    a role's here, which no other rule gives: two corpora whose rows score one and three role-1
    targets deliver equal scored targets, to within one row of each (`delivered`)."""
    from dew.data.steps.text import CountTargets

    one = Pipeline(roled(0, 8, 1), name="one").filter_map(CountTargets(role=1)).map(Relabelled())
    three = Pipeline(roled(1, 8, 3), name="three").filter_map(CountTargets(role=1))
    left, right = delivered([one, three], 1, read)
    assert abs(int(left) - int(right)) <= 1 + 3


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
    assert abs(int(left) - int(right)) <= bound


@pytest.mark.parametrize("read", ["memory", "indexed", "streamed"])
def test_a_map_that_adds_blocks_to_a_counted_row_counts_its_targets_again(read):
    """Blocks a user's map adds to a counted row (`text_block_groups`, a prefix run's) leave out the targets
    they read whole, so the row's count is counted again where the blocks arrive (`recounted`). The mixture
    weighs the corpus by what its rows deliver: the blocked corpus's rows score one role-1 target each, a
    six-token block holding the other five, so the two corpora deliver equal targets, to within one row of
    each (`delivered`)."""
    from dew.data.steps.text import CountTargets

    blocked = Pipeline(roled(0, 8, 6), name="blocked").filter_map(CountTargets(role=1)).map(Blocked(6))
    plain = Pipeline(roled(1, 8, 3), name="plain").filter_map(CountTargets(role=1))
    left, right = delivered([blocked, plain], 1, read)
    assert abs(int(left) - int(right)) <= 1 + 3


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


def test_a_token_mixture_weighs_its_corpora_by_their_stores_counts():
    """Mixed by tokens, two corpora whose stores state their counts deliver equal valid targets,
    in one global order weighed by those counts."""
    def corpus(name, lengths):
        return Stated([{"text": np.arange(size, dtype=np.int32), "source": np.int32(name == "long")}
                       for size in lengths])

    parts = [Pipeline(corpus("long", (2, 10, 10, 10)), name="long"),
             Pipeline(corpus("short", (2, 2)), name="short")]
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
    assert data.plan is not None and data.plan.lowering == "global"


class Cursor(grain.IterDataset):
    """A Grain dataset whose position names its own fields as it likes: here the place it has read
    to, under `counts`."""

    def __init__(self, rows):
        super().__init__()
        self.rows = rows

    def __iter__(self):
        return CursorRead(self.rows)


class CursorRead(grain.DatasetIterator):
    def __init__(self, rows):
        super().__init__()
        self.rows, self.at = rows, 0

    def __next__(self):
        if self.at == len(self.rows):
            raise StopIteration
        self.at += 1
        return self.rows[self.at - 1]

    def get_state(self):
        return {"corpus": "cursor", "counts": self.at}

    def set_state(self, state):
        self.at = state["counts"]


def test_a_repeat_rewinds_a_native_grain_position_as_the_dataset_wrote_it():
    """A repeat that rewinds a pass keeps its corpus's counts, and leaves the rest of the position
    as the iterators wrote it, a Grain dataset's own fields whatever their names."""
    from dew.data import from_grain

    rows = [{"id": np.int32(index)} for index in range(2)]
    data = compile(from_grain(Cursor(rows), "native").take(2).repeat().batch(1), read="streamed",
                   loading=READ)
    stream = data.train(DataPartition())
    try:
        assert [int(next(stream)["id"][0]) for _ in range(5)] == [0, 1, 0, 1, 0]
    finally:
        stream.close()


def test_a_read_finds_a_record_after_any_run_of_rejections():
    """A training read reads on through however many rejected records come before one its filter
    keeps: ten thousand and more in a row are no proof the corpus holds none."""
    rows = Memory([{"id": index} for index in range(10_001)])
    reader = compile(Pipeline(rows).filter(Kept(10_000)).batch(1), read="indexed",
                     loading=Loading(workers=0, threads=1, read_buffer=1)).train(DataPartition())
    try:
        assert next(reader)["id"].tolist() == [10_000]
    finally:
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


@dataclasses.dataclass(frozen=True)
class Offset:
    """Adds each row its entry of a table, which the step holds whole."""

    table: np.ndarray

    def __call__(self, batch):
        return {**batch, "id": batch["id"] + self.table[batch["id"] % len(self.table)]}


def test_an_exact_resume_refuses_a_table_that_differs_where_its_repr_does_not_show():
    """Two steps whose 1024-entry tables differ at entry 512 alone, which their reprs elide, make
    other values: an exact resume under one refuses a position saved under the other, and one
    under the same table restores it."""
    table = np.zeros(1024, np.int64)
    edited = table.copy()
    edited[512] = 1
    assert repr(Offset(table)) == repr(Offset(edited))

    def reader(values):
        return compile(Pipeline(source(8)).map(Offset(values)).shuffle(0).repeat().batch(2),
                       loading=READ).train(DataPartition())

    first = reader(table)
    try:
        next(first)
        state = first.get_state()
    finally:
        first.close()
    same = reader(table.copy())
    try:
        same.set_state(state)
    finally:
        same.close()
    other = reader(edited)
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
        expected = listed(first, 2)
    finally:
        first.close()
    module.write_text(factory + "# an edit that changes no code\n")
    edited = reader(1)
    try:
        edited.set_state(state)
        assert edited.manifest_changes == ("modules",)
        assert listed(edited, 2) == expected
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
        split = Pipeline(Files(str(path))).holdout(tokens=0.25)[0].batch(1)
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
        return compile(prefix.shuffle(5).repeat().random_map(WideDraw()).batch(4),
                       read="memory", seed=5, loading=READ)

    cold = build(prefix)
    reader = cold.train(DataPartition())
    first_epoch = listed(reader, 2)
    saved = reader.get_state()
    second_epoch = listed(reader, 2)
    reader.close()

    def by_id(epoch):
        return {identifier: draw for batch in epoch for identifier, draw in
                zip(batch["id"], batch["draw"], strict=True)}

    assert by_id(first_epoch) != by_id(second_epoch)
    manifest = Manifest.of(stored.fingerprint(), (), fill={"read": "memory", "row_ids": "indexed"})
    generation = cache.publish(manifest, stored)
    resumed = build(prefix).train(DataPartition())
    resumed.set_state(saved)
    assert listed(resumed, 2) == second_epoch
    resumed.close()
    warm = build(Pipeline(stored).cache(generation)).train(DataPartition())
    warm_first, warm_second = listed(warm, 2), listed(warm, 2)
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
    rest = listed(reader, 10)
    assert rest == [{"text": [3, 4, 5, 6]}, {"text": [6, 7, 8, 9]}]
    resumed = iter(windows)
    resumed.set_state(saved)
    assert listed(resumed, 10) == rest
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
        first = listed(reader, 1)
        saved = reader.get_state()
        rest = listed(reader, 10)
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
        assert listed(resumed, 10) == rest
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
        first = listed(reader, 1)
        saved = reader.get_state()
        rest = listed(reader, 20)
        assert first + rest == [{"text": [text[start:start + 3].tolist()],
                                 "roles": [roles[start:start + 3].tolist()]}
                                for start in range(0, len(text) - 2, stride)]
    finally:
        reader.close()
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(saved)
        assert listed(resumed, 20) == rest
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
    first = listed(reader, 5)
    ids = [value for batch in first for value in batch["id"]]
    assert sorted(ids) == list(range(20))
    state = reader.get_state()
    rest = listed(reader, 4)
    reader.close()
    resumed = data.train(DataPartition())
    resumed.set_state(state)
    assert listed(resumed, 4) == rest
    resumed.close()
    from dew.data.ordering import BlockShuffle

    blocked = BlockShuffle(grain.MapDataset.source(stored.values), stored.physical, 2, 11)
    order, blocks = blocked._epoch(0)
    assert len(order) == 3
    start = 0
    original_ends = np.cumsum(stored.physical)
    for end in blocks.ends:
        selected = {int(np.searchsorted(original_ends, blocked.index(index), side="right"))
                    for index in range(start, int(end))}
        assert len(selected) <= 2
        start = int(end)
    changed = compile(Pipeline(stored).shuffle(11, block_shards=3).repeat().batch(4),
                      read="memory", loading=READ).train(DataPartition())
    with pytest.raises(ValueError):
        changed.set_state(state)
    changed.close()


def test_token_mixer_probes_an_unselected_tiny_share_before_emitting_a_row():
    """A share weighted too small to be chosen is still read ahead before the mixture emits a row,
    and its first row lets it."""
    parents = [grain.MapDataset.source([{"targets": 3}]).repeat().to_iter_dataset(),
               grain.MapDataset.source([{"targets": 1}]).repeat().to_iter_dataset()]
    reader = iter(TokenDeficit(parents, (1, 1e-9), ("normal", "tiny"), tokens=Targets(),
                               names=["normal", "tiny"], labels=["normal", "tiny"]))
    try:
        assert next(reader) == {"targets": 3}
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
        return Pipeline(Memory(documents)).shuffle(0).holdout(tokens=0.25)[held]

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
        compile(Pipeline(source())._then(Foreign()).batch(4), loading=READ)


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
        batches = listed(reader, 3)
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
            batches = listed(reader, rows)
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
        read = [[value for batch in listed(reader, 8) for value in batch["id"]] for reader in shares]
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
        read = [[value for batch in listed(reader, 4) for value in batch["id"]] for reader in shares]
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
        listed(saved, 1)
        state = saved.get_state()
        expected = listed(saved, 2)
    finally:
        saved.close()
    resumed = reader()
    try:
        resumed.set_state(state)
        assert listed(resumed, 2) == expected
    finally:
        resumed.close()


def test_a_limit_counts_the_whole_order_and_a_prefetch_never_moves_the_share_cut():
    """24 records with prefetch(4).take(12) train on the same 12 records on one
    share or two."""
    data = compile(Pipeline(source()).prefetch(4).take(12).batch(4), loading=READ)

    def read(count):
        readers = [data.train(DataPartition(index, count)) for index in range(count)]
        try:
            return sorted(value for reader in readers
                          for batch in listed(reader, 10) for value in batch["id"])
        finally:
            for reader in readers:
                reader.close()

    assert read(1) == read(2) and len(read(1)) == 12


@pytest.mark.parametrize("read", ["memory", "indexed"])
def test_a_token_holdout_cuts_the_per_token_fields_it_names_together(read):
    """A token holdout over a field of another name and its roles keeps the first 3/8 of the tokens
    for the held part and the rest for training, cutting the one document it splits in both fields
    alike: the parts together are the corpus, each role beside its token."""
    documents = [{"ids": np.arange(6 * index, 6 * index + 6, dtype=np.int32),
                  "ids_roles": np.arange(6 * index, 6 * index + 6, dtype=np.int32) % 2} for index in range(4)]
    train, held = Memory(documents).holdout(tokens=0.375, fields=("ids", "ids_roles"))

    def rows(part):
        return [{name: np.asarray(row[name]).tolist() for name in ("ids", "ids_roles")}
                for row in part.read(read).loading(READ)]

    trained, kept = rows(train), rows(held)
    tokens = [token for part in (kept, trained) for row in part for token in row["ids"]]
    assert tokens == list(range(24))
    assert [len(row["ids"]) for row in kept] == [6, 3]
    assert all(row["ids_roles"] == [token % 2 for token in row["ids"]] for row in (*kept, *trained))


def test_a_token_split_is_part_of_the_order_a_resume_checks():
    """Four 4-token documents split at a quarter train on documents 1-3 and
    at a half on 2-3, so a position saved under one split is refused under
    the other."""
    documents = [{"text": np.arange(4 * index, 4 * index + 4, dtype=np.int32)} for index in range(4)]

    def reader(fraction):
        split = Pipeline(Memory(documents)).holdout(tokens=fraction)[0].batch(1)
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
        split = pipeline.holdout(tokens=0.25)[0].batch(1)
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
        mixed = mix([inner.holdout(tokens=0.25)[0], corpus(300, (4, 4))])
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
    held = Pipeline(documents).holdout(tokens=0.5)[1].filter(lambda row: True, identity="all@1")
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
    split = Pipeline(documents).shuffle(0).holdout(tokens=0.5)[1]
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
            return [row["id"] for row in listed(reader, 64)]
        finally:
            reader.close()

    limited = Pipeline(source(8)).take(2).repeat(3)
    streamed = ids(limited, "streamed")
    assert streamed == streamed[:2] * 3 and len(ids(limited, "indexed")) == 6
    twice = Pipeline(source(8)).repeat(2).repeat(3)
    assert len(ids(twice, "streamed")) == len(ids(twice, "indexed")) == 48


def test_a_streamed_training_read_with_a_random_step_and_no_repeat_never_runs_out():
    """A training order that writes no repeat repeats for ever, as its plan
    says, whether or not a random step reads its rows."""
    reader = compile(Pipeline(source(8)).random_map(WideDraw()).batch(4), read="streamed",
                     loading=READ).train(DataPartition())
    try:
        assert reader.endless and len(listed(reader, 6)) == 6
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
    corpus = write_token_corpus(tmp_path, np.concatenate(documents).astype(np.uint16), eos_id=0)
    store = TokenFiles(corpus.train).provision()

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
    and a run counted in steps reads it, as one counted in epochs does where the read counts its
    records, an epoch's one step holding records of the next. Repeated after the batch, a pass
    holds no batch to replay, so the read is refused rather than replaying nothing forever, whatever limit
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
        if count < 8 and data.records is not None:
            assert data.epoch_steps(1) == 1


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

    assert_resumes_where_it_stopped(lambda: data.train(DataPartition()), ids, 1, 2)


def test_a_random_step_after_an_endless_repeat_draws_as_after_a_finite_one():
    """A streamed repeat read endlessly guards its passes without changing what a random step
    after it draws: the passes a finite repeat reads draw the same values."""
    rows = Pipeline(source(2)).take(2)

    def draws(chain) -> list[int]:
        data = compile(chain.random_map(WideDraw()).batch(2), read="streamed", loading=READ)
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
    data = compile(Pipeline(source(2)).random_map(WideDraw()).batch(2).repeat(), read=read, loading=READ)

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
