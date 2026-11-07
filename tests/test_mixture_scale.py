"""A mixture of a thousand sources at CPU scale: it compiles, reads and resumes with no cost per
source at each step, its weights are each source's share, and a source that runs out follows its
policy. Each test records what it measured (`record_property`) and prints it."""

import gc
import itertools
import os
import resource
import sys
import threading
import time
import tracemalloc

import numpy as np
import pytest

from dew.data import Files, Loading, Memory
from dew.data.compile import compile
from dew.data.dataset import DataPartition
from dew.data.pipeline import Pipeline, mix

SOURCES = 1000
RECORDS = 8
BATCH = 16
DOCUMENTS = 128
"""Each packed source's documents (`packed`): more than five times the most the packed tests read of one
source (24, a thousand sources mixed by records), so no step reopens a source, and few enough that hashing
every source's records to name it costs compiling a thousand of them 8 s, not 61 (row 4247)."""
SOURCE_TOKENS = 100_000
"""The tokens each packed source's documents count from, past any of another's (`Documents`)."""
READ = Loading(workers=0, threads=1, read_buffer=8, worker_buffer=1)


class Counted:
    """A source's records, each saying which source and record it is, and how many reads reach it."""

    reads = 0

    def __init__(self, tag: int, records: int = RECORDS):
        self.tag, self.records = tag, records

    def __repr__(self) -> str:
        return f"Counted(tag={self.tag}, records={self.records})"

    def __len__(self) -> int:
        return self.records

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.records:
            raise IndexError(index)
        Counted.reads += 1
        return {"id": np.int64(self.tag * 1000 + index)}


def sources(count: int = SOURCES, *, repeat: bool = True) -> list[Pipeline]:
    chains = [Pipeline(Memory(Counted(tag)), name=f"source-{tag}") for tag in range(count)]
    return [chain.repeat() for chain in chains] if repeat else chains


def ids(stream, batches: int) -> list[int]:
    return [int(value) for batch in itertools.islice(stream, batches) for value in batch["id"]]


def open_files() -> int:
    return len(os.listdir("/proc/self/fd"))


class Documents:
    """A source's token documents, each 3 to 12 tokens of its own, saying which source, document and
    token it is, and how many reads reach them: a packer of 16 tokens a row lays some whole and leaves
    others to the next row."""

    reads = 0

    def __init__(self, tag: int, records: int = RECORDS):
        self.tag, self.records = tag, records

    def __repr__(self) -> str:
        return f"Documents(tag={self.tag}, records={self.records})"

    def __len__(self) -> int:
        return self.records

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.records:
            raise IndexError(index)
        Documents.reads += 1
        length = 3 + (self.tag * 7 + index * 5) % 10
        return {"text": np.arange(length, dtype=np.int32) + self.tag * SOURCE_TOKENS + index * 20}


def packed(count: int, *, by: str = "records", examples: int | None = 10**6):
    """A mix of `count` sources of documents, seeded by records or each next from the source furthest below
    its share of tokens (`by`), packed into rows of 16 tokens by the default packer (first fit), read as a
    stream under the default loading and bounded in examples, which counts each member's examples where its
    rows are delivered (`MEMBERS`): a user's packed mixture, whose packer asks the mix's position before
    every document."""
    # Each source a shard of more documents than the steps a test reads take of it, as a corpus's are: a
    # shard opened again costs its open once (Grain registers each dataset it reads for its statistics,
    # a few blocks it keeps), which the steps between opens do not.
    chains = [Pipeline(Memory(Documents(tag, DOCUMENTS)), name=f"source-{tag}").repeat()
              for tag in range(count)]
    mixed = mix(chains, seed=0) if by == "records" else mix(chains, by="tokens")
    return compile(mixed.pack(16, fields=("text",)).batch(BATCH), read="streamed", examples=examples)


def asked(stream, batches: int) -> int:
    """The positions `stream` asks while it gives `batches`: the calls of any read's `get_state` and any
    count's `state`, on every thread the read runs on."""
    calls: list[None] = []

    def counted(frame, event, argument):
        if event == "call" and frame.f_code.co_name in ("get_state", "state"):
            calls.append(None)

    threading.setprofile_all_threads(counted)
    try:
        texts(stream, batches)
    finally:
        threading.setprofile_all_threads(None)
    return len(calls)


def texts(stream, batches: int) -> list[list[int]]:
    return [np.asarray(batch["text"]).tolist() for batch in itertools.islice(stream, batches)]


def drawn(stream, count: int) -> None:
    """`stream` read until it has given a row of each of its `count` sources, each opened as the mix first
    draws from it: what a step costs once a run has read from every source."""
    left = set(range(count))
    while left:
        for batch in texts(stream, 8):
            left -= {value // SOURCE_TOKENS for row in batch for value in row if value}


@pytest.mark.parametrize("by", ["records", "tokens"])
def test_a_thousand_sources_compile_and_step_without_a_cost_per_source(by, record_property):
    """A user's packed mixture, the default packer and loading (`packed`), steps once it has read from every
    source at no cost per source: a thousand sources' step reads the records it packs and nothing of the
    sources it does not draw, asks the positions the packer asks before each document in the calls ten
    do (a mix's position captured copy-on-write, `SharedMapping`, each capture asking again only what the
    document moved), holds the threads ten do (its members read under their share of the loading, one
    after another on the thread that draws from them), and keeps no block a step, and compiling them
    costs memory linear in their count (row 4079: each document's capture asked and copied every
    member's position and counts, and each member's shard read on sixteen threads of its own)."""
    measured = {}
    for count in (10, SOURCES):
        gc.collect()
        tracemalloc.start()
        began = time.perf_counter()
        data = packed(count, by=by)
        built = time.perf_counter() - began
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stream = data.train(DataPartition())
        try:
            drawn(stream, count)
            Documents.reads = 0
            began = time.perf_counter()
            asks = asked(stream, 64)
            stepped = (time.perf_counter() - began) / 64
            reads = Documents.reads
            # What steps keep, counted between two counts past the first window: a buffer it sized once
            # is in both, and whatever a step keeps grows with the steps between them.
            blocks = []
            for _ in range(2):
                texts(stream, 64)
                gc.collect()
                blocks.append(sys.getallocatedblocks())
            kept = (blocks[1] - blocks[0]) / 64
            threads = threading.active_count()
        finally:
            stream.close()
        measured[count] = {"compile_s": built, "compile_peak_mib": peak / 2**20, "step_ms": 1e3 * stepped,
                           "reads_per_step": reads / 64, "positions_asked_per_step": asks / 64,
                           "threads": threads, "blocks_kept_per_step": kept}
        record_property(f"sources_{count}", measured[count])
        print(f"sources={count}", measured[count])
    few, many = measured[10], measured[SOURCES]
    assert many["reads_per_step"] < 2 * few["reads_per_step"], measured
    assert many["positions_asked_per_step"] < 2 * few["positions_asked_per_step"], measured
    assert many["threads"] <= few["threads"], measured
    assert many["blocks_kept_per_step"] < 1, measured
    record_property("rss_mib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    print("rss_mib", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def test_a_thousand_file_sources_hold_no_file_open_per_source(tmp_path, record_property):
    """A thousand JSON-lines files read as one mixture hold a bounded number of files open while
    they read, not one each."""
    paths = []
    for tag in range(SOURCES):
        path = tmp_path / f"{tag:04d}.jsonl"
        path.write_text("".join(f'{{"id": {tag * 1000 + index}}}\n' for index in range(RECORDS)))
        paths.append(path)
    before = open_files()
    chains = [Pipeline(Files(str(path)), name=f"file-{tag}").repeat() for tag, path in enumerate(paths)]
    data = compile(mix(chains).batch(BATCH), read="memory", loading=READ)
    stream = data.train(DataPartition())
    try:
        ids(stream, 64)
        held = open_files() - before
    finally:
        stream.close()
    record_property("open_files_while_reading", held)
    print("open_files_while_reading", held)
    assert held < 64


def test_the_weights_are_each_sources_share_of_the_records_read():
    """A thousand sources weighted 1 or 2 each read their share of a whole period of the mix, the
    records in which every proportion comes round once, to the record; weights scaled together
    read the same records."""
    weights = [float(tag % 2 + 1) for tag in range(SOURCES)]

    def read(scale, batches):
        data = compile(mix(sources(), weights=[scale * weight for weight in weights]).batch(BATCH),
                       read="memory", loading=READ)
        stream = data.train(DataPartition())
        try:
            return ids(stream, batches)
        finally:
            stream.close()

    # Grain's mix scales the least weight to 100 records a period.
    period = int(100 * sum(weights))
    whole = read(1.0, period // BATCH)
    assert len(whole) == period
    counts = np.bincount(np.asarray(whole) // 1000, minlength=SOURCES)
    np.testing.assert_array_equal(counts, 100 * np.asarray(weights, np.int64))
    assert read(0.25, 64) == whole[:64 * BATCH]


def test_an_epoch_of_a_thousand_finite_sources_reads_every_record():
    """Sources written without a repeat run out at their own lengths; an epoch reads every record of
    every one of them, the mixture's policy for a source that runs out."""
    lengths = [RECORDS + tag % 5 for tag in range(SOURCES)]
    chains = [Pipeline(Memory(Counted(tag, length)), name=f"source-{tag}")
              for tag, length in enumerate(lengths)]
    data = compile(mix(chains).batch(BATCH), read="memory", loading=READ)
    stream = data.train(DataPartition())
    try:
        read = ids(stream, data.steps_per_epoch)
    finally:
        stream.close()
    every = {tag * 1000 + index for tag, length in enumerate(lengths) for index in range(length)}
    assert every <= set(read)


@pytest.mark.parametrize("stop", [1, 37])
def test_a_thousand_source_mixture_resumes_where_it_stopped(stop):
    """A run stopped after `stop` steps and resumed from its saved position reads what the run that
    never stopped reads next, record for record."""
    data = compile(mix(sources(), weights=[float(tag % 7 + 1) for tag in range(SOURCES)]).batch(BATCH),
                   read="memory", loading=READ)
    whole = data.train(DataPartition())
    try:
        ids(whole, stop)
        state = whole.get_state()
        expected = ids(whole, 20)
    finally:
        whole.close()
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(state)
        assert ids(resumed, 20) == expected
    finally:
        resumed.close()


def test_a_thousand_source_selection_reads_in_order_what_each_position_reads_alone():
    """Read one position after another, a mixture's selection reads runs of positions at once; it
    reads what each position reads alone, Grain's order, and a position reached out of order (a
    resume, a seek) reads the same, before and after a run."""
    from dew.data.mixing import ProportionalSelection, Proportions

    weights = [float(tag % 7 + 1) for tag in range(SOURCES)]
    lengths = [10**6] * SOURCES
    selection = ProportionalSelection(lengths, weights)
    proportions = Proportions.of(weights)
    ordered = [selection[index] for index in range(10_000)]
    assert ordered == [proportions.at(index) for index in range(10_000)]
    for index in (123_456, 7, 9_999, 50_000, 50_001, 3):
        assert selection[index] == proportions.at(index)


def test_a_mixtures_rejection_merges_its_members_counts_only_where_a_log_or_refusal_names_them():
    """A mixture's counter past the mix counts what it rejects of each member's rows under that member's
    counts, and merges the members' counts by reason only where a log or a refusal names the reasons: once
    it has warned of its rejections, a rejection costs what it does at ten members, however many the mix
    holds (row 3531: 5.7 us at 10 members, 156 us at 300)."""
    from collections import Counter

    from dew.data.mixing import CorpusCounts, _KeepAccepted
    from dew.data.pipeline import Rejected

    class Merged(Counter):
        """A member's rejections, which count each merge of them into the mix's."""

        merges = 0

        def __add__(self, other):
            Merged.merges += 1
            return super().__add__(other)

        def __radd__(self, other):
            Merged.merges += 1
            return Counter.__add__(other, self)

    members = {f"member-{at}": CorpusCounts(f"member-{at}", f"m{at}", rejected=Merged())
               for at in range(SOURCES)}
    counter = _KeepAccepted(CorpusCounts("mix", "mix"), None, None, members)
    for at in range(200):
        counter(Rejected("short", member=f"member-{at % SOURCES}"))
    assert counter.warned
    Merged.merges = 0
    for at in range(300):
        counter(Rejected("short", member=f"member-{at % SOURCES}"))
    assert Merged.merges == 0


@pytest.mark.parametrize("by", ["records", "tokens"])
@pytest.mark.parametrize("stop", [1, 3, 7])
def test_a_packed_mixture_resumes_at_the_document_that_did_not_fit(stop, by):
    """A packed mix's rows leave documents that do not fit to the next row, which its position begins at,
    the mix's and its counters' positions as they stood before that document: a run of a mix by records
    or by tokens stopped after `stop` steps and resumed reads what the run that never stopped reads next,
    row for row, and counts what it counts (row 4079: positions captured copy-on-write, never the live
    counts, nor a token mix's live deliveries)."""
    data = packed(SOURCES, by=by)
    whole = data.train(DataPartition())
    try:
        texts(whole, stop)
        state = whole.get_state()
        expected = texts(whole, 6)
        counted = data.statistics()
    finally:
        whole.close()
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(state)
        assert texts(resumed, 6) == expected
        assert data.statistics() == counted
    finally:
        resumed.close()


class Listed:
    """A parent read in order on the caller's thread alone, counting its own reads: no prefetch reads
    ahead of what asks it."""

    def __new__(cls, count: int):
        import grain.python as pygrain

        class Iterator(pygrain.DatasetIterator):
            def __init__(self):
                super().__init__()
                self.next, self.reads = 0, 0

            def __next__(self):
                if self.next >= count:
                    raise StopIteration
                self.next, self.reads = self.next + 1, self.reads + 1
                return {"id": self.next - 1}

            def get_state(self):
                return {"next": self.next}

            def set_state(self, state):
                self.next = state["next"]

        class Dataset(pygrain.IterDataset):
            def __iter__(self):
                self.last = Iterator()
                return self.last

        return Dataset()


@pytest.mark.parametrize("stop", [5, 31, 32, 100])
def test_a_streamed_window_shuffle_reads_each_window_as_the_one_before_is_given(stop):
    """A streamed read's window shuffle (`WindowShuffle`) gives Grain's order and saves Grain's position
    (`gx.WindowShuffleIterDataset`, the reference arm), so a saved run reads and resumes as it did, stopped
    anywhere in a window or at its end; and past its first window it reads each next window as the one
    before is given, one record a record given, where Grain read a window whole as its first record was
    asked (row 4740: a 16k-record window 9.8 s, the step that asked it stalled)."""
    import grain.experimental as gx

    from dew.data.ordering import WindowShuffle

    window = 32
    listed = Listed(1000)
    ours = iter(WindowShuffle(listed, window=window, seed=3))
    theirs = iter(gx.WindowShuffleIterDataset(Listed(1000), window_size=window, seed=3))
    parent = listed.last
    reads = []
    for _ in range(stop):
        before = parent.reads
        assert next(ours) == next(theirs)
        reads.append(parent.reads - before)
    assert ours.get_state() == theirs.get_state()
    assert reads[0] == window and max(reads[window:], default=0) <= 1, reads
    resumed = iter(WindowShuffle(Listed(1000), window=window, seed=3))
    resumed.set_state(ours.get_state())
    assert list(itertools.islice(resumed, 3 * window)) == list(itertools.islice(theirs, 3 * window))
