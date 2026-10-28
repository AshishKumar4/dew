"""Weighted corpora, elastic packed windows and the batch ramp.

The three answer the same question: which records does global step k read?
A mixture answers it with grain's proportional interleave over corpora that
are shuffled and cycled before they are mixed, the packed loader answers it
with a packing plan over the whole corpus, and a ramp answers it with a
step-indexed batch cut out of that same order. So what is asserted here is
which records a step holds, and that the answer does not change when a run
is killed and resumed, when it is resumed on a different number of
processes, or when its batch grows.

A pool of processes is its shares here: each reader opens the stream for
the `DataPartition` a process of that pool would read, which is the whole
of what a loader learns about the pool. `tests/test_multiprocess.py` runs
the same claims in real processes with a real checkpoint between them.
"""

import collections
import dataclasses
import itertools
import json
from pathlib import Path

import grain.python as pygrain
import jax
import numpy as np
import optax
import pytest
from affine_run import FEATURES, Affine
from recording import RecordingTracker
from token_support import write_token_corpus

from dew import position
from dew.batch import VALID_ROWS, freeze, thaw
from dew.data import Files, Loading, Memory, phases, presets, schedule
from dew.data.compile import compile, held, publish, validation
from dew.data.dataset import CAPTION, DataPartition, GlobalStream, Loader, PhasedStream, Ramp, Stage, ramped
from dew.data.images import EncodeCaptions
from dew.data.pipeline import DeviceStep, Pipeline, mix
from dew.data.plan import Manifest, Plan
from dew.data.sources import TokenFiles
from dew.objectives.base import Aux, Objective, Ratio
from dew.training import Checkpoints, Layout, MeshSpec, Trainer

READ = Loading(workers=0, threads=1, read_buffer=8, worker_buffer=1)


class Indexed:
    """Records that say which corpus and which record they are."""

    def __init__(self, tag: int, records: int):
        self.tag, self.records = tag, records

    def __repr__(self) -> str:
        return f"Indexed(tag={self.tag}, records={self.records})"

    def __len__(self) -> int:
        return self.records

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.records:
            raise IndexError(index)
        return {"id": np.int32(self.tag * 1000 + index)}


def ids(batches) -> list[list[int]]:
    """The record ids of each batch, in row order."""
    return [[int(value) for value in batch["id"]] for batch in batches]


def taken(stream, batches: int) -> list[list[int]]:
    return ids(itertools.islice(stream, batches))


def pooled(build, processes: int, batches: int, state: bytes | None = None,
           rows=ids) -> list:
    """The global batches a pool of `processes` reads, each row where the
    share that holds it put it: share p owns rows p, p + n, ... of a step,
    which is how a compiled global order is sliced. `build` opens the stream
    a share reads."""
    shards = []
    for index in range(processes):
        stream = build(DataPartition(index, processes))
        if state is not None:
            stream.set_state(state)
        shards.append(rows(itertools.islice(stream, batches)))
    return [[row for held in zip(*step, strict=True) for row in held] for step in zip(*shards, strict=True)]


# --------------------------------------------------------------------------
# Weighted mixtures
# --------------------------------------------------------------------------

def two_corpora(weights=(0.75, 0.25), records=(40, 8), seed=0) -> Pipeline:
    """Wiki and code records, each reshuffled every epoch and cycled, mixed at `weights`."""
    wiki, code = (Pipeline(Memory(Indexed(tag, count)), name=name).shuffle(seed).repeat()
                  for name, tag, count in (("wiki", 1, records[0]), ("code", 2, records[1])))
    return mix([wiki, code], weights=weights)


def mixed(corpora, batch=4, seed=0):
    return compile(corpora.batch(batch), read="memory", seed=seed, loading=READ).train


def test_every_batch_of_a_mixture_holds_each_corpus_share_of_it():
    """The interleave is proportional per prefix, so the weights hold inside
    a batch and not only over the run: a step of four at 3:1 is three wiki
    records and one code record, every step."""
    stream = mixed(two_corpora())(DataPartition())

    batches = taken(stream, 8)

    for step in batches:
        assert sum(value // 1000 == 2 for value in step) == 1, step
        assert sum(value // 1000 == 1 for value in step) == 3, step


def test_a_small_corpus_of_a_mixture_comes_round_while_a_large_one_is_read_once():
    """Weights are shares of a step, not of the data: the corpora are cycled
    before they are mixed, so eight code records at a quarter of the batch
    are read again long before the forty wiki records are through."""
    stream = mixed(two_corpora())(DataPartition())

    seen = [value for step in taken(stream, 10) for value in step]

    code = [value for value in seen if value // 1000 == 2]
    wiki = [value for value in seen if value // 1000 == 1]
    assert len(code) == 10 and len(set(code)) == 8, "the small corpus never repeated"
    assert len(wiki) == 30 and len(set(wiki)) == 30, "the large corpus repeated a record"


def test_a_mixture_is_the_same_order_from_the_same_seed_and_another_from_another():
    first = taken(mixed(two_corpora())(DataPartition()), 5)

    assert taken(mixed(two_corpora())(DataPartition()), 5) == first
    assert taken(mixed(two_corpora(seed=1), seed=1)(DataPartition()), 5) != first


def test_a_mixtures_position_resumes_on_another_process_count():
    """The mixing runs ahead of the shard, so global batch k is the same
    records at any process count and the count two processes saved is where
    one process or four carry on."""
    # The global batch is eight whatever the count; each share reads its
    # part of it.
    build = mixed(two_corpora(), batch=8)
    whole = pooled(build, 1, 6)

    assert pooled(build, 2, 6) == whole
    assert pooled(build, 4, 6) == whole

    stopped = build(DataPartition(0, 2))
    taken(stopped, 2)
    state = stopped.get_state()

    assert json.loads(state)["dew_global_position"]["records"] == 16
    for processes in (1, 4):
        assert pooled(build, processes, 4, state) == whole[2:]


@dataclasses.dataclass(frozen=True)
class Halved:
    """A host step that cuts each batch to a shape of its own, as a stream
    that pads each batch only to its longest record does."""

    single_reader = True

    def __call__(self, batch):
        return {"id": batch["id"][:2]}


def test_a_cut_stream_maps_every_batch_and_is_read_by_one_process():
    """The training stream and the validation pass alike."""
    held = Pipeline(Memory(Indexed(3, 6)), name="held").batch(4).map(Halved())
    data = compile(two_corpora().batch(4).map(Halved()), validation=held, read="memory", loading=READ)
    assert [len(batch) for batch in taken(data.train(DataPartition()), 3)] == [2, 2, 2]
    for reader in (data.train, data.val):
        with pytest.raises(ValueError):
            reader(DataPartition(0, 2))


def test_a_validation_pass_over_a_mixture_reads_each_record_at_most_once():
    """A mixed pass keeps the weights and each split's own order, and stops
    before any corpus comes round, so a score is a score of the same records
    every time."""
    wiki, code = (Pipeline(Memory(Indexed(tag, count))) for tag, count in ((1, 40), (2, 8)))
    held = validation(mix([wiki, code], weights=(0.75, 0.25)), batch=4, loading=READ)(DataPartition())
    try:
        read = [int(value) for batch in held
                for value in batch["id"][batch.get(VALID_ROWS, np.ones(4, bool))]]
    finally:
        held.close()

    assert len(read) == 32, "a pass runs until the first corpus would repeat"
    assert len(set(read)) == len(read), "a record was scored twice"
    assert read[:8] == [1000, 1001, 1002, 2000, 1003, 1004, 1005, 2001]


@pytest.mark.parametrize("count, weights", [(0, ()), (2, (1.0,)), (2, (1.0, 0.0)), (2, (1.0, -1.0))])
def test_a_mixture_that_is_not_one_is_refused(count, weights):
    """A mixture is one or more datasets with one finite, positive weight each."""
    corpora = [Pipeline(Memory(Indexed(tag, 8))) for tag in range(1, count + 1)]
    with pytest.raises(ValueError):
        mix(corpora, weights=weights)


# --------------------------------------------------------------------------
# Mixtures of stores of rows
# --------------------------------------------------------------------------


def rows_dir(root: Path, name: str, first: int, count: int) -> str:
    """A directory of one jsonl file, which `load_dataset` reads as a split."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "rows.jsonl").write_text(
        "\n".join(json.dumps({"index": first + row}) for row in range(count)))
    return str(directory)


def just_index(record, rng):
    return {"id": np.int32(record["index"])}


@pytest.fixture(scope="module")
def two_splits(tmp_path_factory) -> tuple[str, str]:
    root = tmp_path_factory.mktemp("mixture")
    return (rows_dir(root, "big", 1000, 40), rows_dir(root, "small", 2000, 8))


def row_corpus(root: str) -> Pipeline:
    """The rows of `root`'s jsonl file, each its index as an `id`, under `root`'s name."""
    rows = Pipeline(Files(str(Path(root) / "rows.jsonl"), format="jsonl"))
    return dataclasses.replace(rows.map(lambda row: {"id": np.int32(row["index"])}), name=root)


def row_mix(sources: dict[str, float], *, read: str = "memory", seed: int = 0) -> Loader:
    """`sources`' rows mixed at their weights as a run trains on them: each corpus shuffled (in a
    window of 64 where it streams) and repeated, in batches of four."""
    window = 64 if read == "streamed" else None
    corpora = [row_corpus(root).shuffle(seed, window=window).repeat() for root in sorted(sources)]
    weights = [sources[root] for root in sorted(sources)]
    return compile(mix(corpora, weights=weights).batch(4), read=read, loading=READ, seed=seed)


def test_a_weighted_mixture_of_two_datasets_holds_each_share_of_every_batch(two_splits):
    big, small = two_splits
    data = row_mix({big: 0.75, small: 0.25})
    # An epoch is the fewest records in which the big split's 40 are read, three in every four: 53,
    # in 14 steps of four, the last leading into the next epoch.
    assert data.records == 53 and data.batch == 4 and data.steps_per_epoch == 14
    stream = data.train(DataPartition())
    try:
        batches = taken(stream, 6)
    finally:
        stream.close()
    assert all(sum(value >= 2000 for value in step) == 1 for step in batches)


def test_a_mixed_validation_pass_ends_before_any_source_repeats(two_splits):
    big, small = two_splits
    passed = held(mix([row_corpus(big), row_corpus(small)], weights=[0.75, 0.25]).batch(4)
                  .read("memory").loading(READ))
    first, second = passed(DataPartition()), passed(DataPartition())
    try:
        actual, again = ids(first), ids(second)
    finally:
        first.close()
        second.close()
    assert len(actual) == 8 and actual == again
    scored = [value for batch in actual for value in batch]
    assert len(scored) == len(set(scored)) == 32
    assert all(sum(value >= 2000 for value in batch) == 1 for batch in actual)


def test_a_mixture_written_in_either_order_is_the_same_run(two_splits):
    big, small = two_splits
    first, other = row_mix({big: 0.75, small: 0.25}), row_mix({small: 0.25, big: 0.75})
    left, right = first.train(DataPartition()), other.train(DataPartition())
    try:
        assert taken(left, 4) == taken(right, 4)
        assert left.get_state() == right.get_state()
        assert taken(left, 3) == taken(right, 3)
    finally:
        left.close()
        right.close()


def test_sources_from_different_stores_mix_before_the_share_cut(two_splits):
    big, _ = two_splits
    files = Pipeline(Files(str(Path(big) / "rows.jsonl"), format="jsonl")).map(
        lambda row: {"id": np.int32(row["index"])})
    memory = Pipeline(Memory([{"id": np.int32(2000 + index)} for index in range(8)]))
    data = compile(mix([files.shuffle(0).repeat(), memory.shuffle(0).repeat()], weights=(3, 1)).batch(4),
                   read="memory", loading=READ)
    stream = data.train(DataPartition())
    try:
        assert all(sum(value >= 2000 for value in batch) == 1 for batch in taken(stream, 8))
    finally:
        stream.close()


def test_a_streamed_mixture_keeps_accepted_row_shares(two_splits):
    big, small = two_splits
    data = row_mix({big: 0.75, small: 0.25}, read="streamed")
    stream = data.train(DataPartition())
    try:
        assert all(sum(value >= 2000 for value in batch) == 1 for batch in taken(stream, 8))
    finally:
        stream.close()


def document_dir(root: Path, documents, eos: int = 0, tokenizer: str = "byte") -> str:
    """A token store whose stream is `documents`, each closed by eos."""
    stream = np.concatenate([np.asarray([*list(document), eos], np.uint16)
                             for document in documents])
    return write_token_corpus(root, stream, vocab_size=256, eos_id=eos, tokenizer=tokenizer).train


def packed_corpus(root: Path, documents: int = 40, length: int = 3) -> str:
    # Document i is the value i + 1 repeated, so a window's values name the
    # documents packed into it and the padding names none.
    return document_dir(root, [[index + 1] * length for index in range(documents)])


def packed(url: str) -> Pipeline:
    """A token store's documents packed first-fit, two rows open, into rows of eight inputs."""
    return TokenFiles(url).pipe(presets.lm, None, seq_len=8, batch=None, pack="first-fit", bins=2)


def ordered(corpus: Pipeline, seed: int = 0) -> Pipeline:
    """Packed rows in the order a run trains on them: shuffled and repeated."""
    return corpus.shuffle(seed).repeat()


def trained(chain: Pipeline, batch: int = 4) -> Loader:
    return compile(chain.batch(batch), read="memory", loading=READ)


def mixed_packed(stores: dict[str, float]) -> Pipeline:
    """Token stores' packed rows mixed by their tokens at their weights, each in a run's order."""
    return mix([ordered(packed(url)) for url in stores], weights=list(stores.values()), by="tokens")


def windows_of(batches) -> list[list[tuple[int, ...]]]:
    """Each batch as its rows' token windows."""
    return [[tuple(int(value) for value in row) for row in batch["text"]]
            for batch in batches]


def packed_rows(stream, batches: int) -> list[list[tuple[int, ...]]]:
    return windows_of(itertools.islice(stream, batches))


def test_a_packed_step_is_the_same_windows_at_every_process_count(tmp_path):
    """The packing is planned over the whole corpus ahead of the shard, so
    global batch k holds the same windows in the same places however many
    processes read it. Packed behind the shard, as the loader once was, each
    process packed its own documents and no two counts agreed."""
    open_stream = trained(ordered(packed(packed_corpus(tmp_path / "corpus")))).train

    whole = pooled(open_stream, 1, 5, rows=windows_of)

    assert whole[0], "the chain produced no windows"
    assert pooled(open_stream, 2, 5, rows=windows_of) == whole
    assert pooled(open_stream, 4, 5, rows=windows_of) == whole


def test_a_packed_position_written_by_two_processes_resumes_on_one(tmp_path):
    """The packed stream's position is a global window count, so the two
    processes that stopped at step two hand one process, or four, the steps
    the run would have taken next."""
    open_stream = trained(ordered(packed(packed_corpus(tmp_path / "corpus")))).train
    whole = pooled(open_stream, 1, 6, rows=windows_of)

    stopped = open_stream(DataPartition(1, 2))
    packed_rows(stopped, 2)
    state = stopped.get_state()

    assert json.loads(state)["dew_global_position"]["records"] == 8
    for processes in (1, 2, 4):
        assert pooled(open_stream, processes, 4, state, rows=windows_of) == whole[2:]


def test_a_packed_pass_covers_every_document_once_at_every_process_count(tmp_path):
    """A validation pass over the plan is the same windows in the same order
    for every count, and the processes of a pool cover the split between
    them without overlapping."""
    pass_over = held(packed(packed_corpus(tmp_path / "corpus", documents=24)).batch(4).read("memory")
                     .loading(READ))
    whole = [row for batch in pass_over(DataPartition()) for row in windows_of([batch])[0]]

    for processes in (2, 4):
        shards = [[row for batch in pass_over(DataPartition(index, processes))
                   for row in windows_of([batch])[0]] for index in range(processes)]
        assert sum(len(shard) for shard in shards) == len(whole)
        assert sorted(row for shard in shards for row in shard) == sorted(whole)


def weighted_packed(tmp_path, weights, *, tokenizer: str = "byte"):
    """Two corpora told apart by their values: 1..40 in the first, 101..124
    in the second, each document one value repeated three times."""
    first = document_dir(tmp_path / "first", [[index + 1] * 3 for index in range(40)])
    second = document_dir(tmp_path / "second", [[index + 101] * 3 for index in range(24)],
                          tokenizer=tokenizer)
    return {first: weights[0], second: weights[1]}, first, second


def grains_corpora(weights) -> pygrain.MapDataset:
    """Grain's own weighted mix (`MapDataset.mix`) of one repeated corpus a weight, each position the
    number of the corpus Grain's mix reads there: the reference a mix's member order is held to."""
    return pygrain.MapDataset.mix([pygrain.MapDataset.range(1).map(lambda _, tag=tag: tag).repeat()
                                   for tag in range(len(weights))], weights)


def corpus_of(window: tuple[int, ...]) -> int:
    values = {value for value in window if value}
    corpora = {0 if value < 100 else 1 for value in values}
    assert len(corpora) == 1, f"window {window} mixes corpora"
    return corpora.pop()


def test_weighted_packed_corpora_fill_every_step_at_their_shares(tmp_path):
    """MaxText's weighted `grain_train_files`, over packed windows: every
    window is one corpus's documents, and the stream's windows come from the
    corpora Grain's own `MapDataset.mix` reads at each position, which holds
    every prefix to each corpus's share of it within one. At 3:1 a batch of
    four is three and one."""
    weights = (3.0, 1.0)
    stores, _, _ = weighted_packed(tmp_path, weights)
    rows = [row for batch in packed_rows(trained(mixed_packed(stores)).train(DataPartition()), 12)
            for row in batch]
    drawn = [corpus_of(row) for row in rows]
    grains = grains_corpora(weights)
    assert drawn == [grains[position] for position in range(len(drawn))]
    assert all(drawn[step * 4:step * 4 + 4].count(1) == 1 for step in range(12))


def test_weighted_packed_corpora_resume_on_another_process_count(tmp_path):
    stores, _, _ = weighted_packed(tmp_path, (1.0, 1.0))
    open_stream = trained(mixed_packed(stores)).train
    whole = pooled(open_stream, 1, 6, rows=windows_of)

    stopped = open_stream(DataPartition(0, 2))
    packed_rows(stopped, 2)
    state = stopped.get_state()
    for processes in (1, 4):
        assert pooled(open_stream, processes, 4, state, rows=windows_of) == whole[2:]


class Asked(pygrain.IterDataset):
    """Corpus `tag` as its stages read it, counting in `gave` the documents it
    gives and in `asked` how often a reader asks its position."""

    def __init__(self, parent: pygrain.IterDataset, tag: int, gave: collections.Counter,
                 asked: collections.Counter):
        super().__init__(parent)
        self.documents, self.tag, self.gave, self.asked = parent, tag, gave, asked

    def __iter__(self) -> pygrain.DatasetIterator:
        return _Asked(self.documents.__iter__(), self)


class _Asked(pygrain.DatasetIterator):
    def __init__(self, parent: pygrain.DatasetIterator, corpus: Asked):
        super().__init__(parent)
        self.documents, self.corpus = parent, corpus

    def __next__(self):
        self.corpus.gave[self.corpus.tag] += 1
        return next(self.documents)

    def get_state(self):
        self.corpus.asked[self.corpus.tag] += 1
        return self.documents.get_state()

    def set_state(self, state):
        self.documents.set_state(state)


def test_a_packed_stream_asks_each_corpus_its_position_only_after_the_corpus_reads():
    """A packer notes the stream's position before each document, and a
    mixture's position is every corpus's own. Only the corpus that gave the
    document moved, so each corpus is asked about once per document it gives
    (and twice as the stream opens), not once per document of the mixture;
    the stream still resumes where it stopped."""
    gave, asked = collections.Counter(), collections.Counter()

    def corpus(tag: int) -> Pipeline:
        def counted(dataset: pygrain.MapDataset | pygrain.IterDataset) -> Asked:
            assert isinstance(dataset, pygrain.IterDataset)
            return Asked(dataset, tag, gave, asked)

        documents = Memory([{"text": np.full(3, tag * 100 + index, np.int32)} for index in range(40)])
        return Pipeline(documents).read("streamed").shuffle(tag, window=8).repeat().grain(
            counted, identity=f"asked-{tag}")

    run = compile(mix([corpus(tag) for tag in range(1, 7)], weights=[1] * 6).pack(8, bins=2).batch(2),
                  read="streamed", loading=READ)
    stream = run.train(DataPartition())
    try:
        packed_rows(stream, 8)
        state = stream.get_state()
        rest = packed_rows(stream, 4)
    finally:
        stream.close()
    assert set(gave) == set(range(1, 7))
    assert all(asked[tag] <= gave[tag] + 2 for tag in gave)
    resumed = run.train(DataPartition())
    try:
        resumed.set_state(state)
        assert packed_rows(resumed, 4) == rest
    finally:
        resumed.close()


def test_a_weighted_packed_pass_counts_and_scores_every_corpus(tmp_path):
    """The pass is the fewest windows in which every corpus has been read once, and
    the validation pass reads each held-out window at most once."""
    stores, first, second = weighted_packed(tmp_path, (1.0, 1.0))
    loaded = trained(mixed_packed(stores), batch=2)
    alone = [trained(ordered(packed(url)), batch=2).records for url in (first, second)]
    # At even weights the mix reads its corpora in turn from the first, the larger, whose last
    # window is the mix's 2 * 20 - 1th.
    assert alone[0] == max(alone) and loaded.records == 2 * max(alone) - 1
    passed = held(mix([packed(url) for url in stores], weights=list(stores.values()), by="tokens").batch(2)
                  .read("memory").loading(READ))
    scored = [row for batch in passed(DataPartition()) for row in windows_of([batch])[0]]
    assert len(scored) == len(set(scored))
    assert {corpus_of(row) for row in scored} == {0, 1}


def test_corpora_from_different_tokenizers_are_not_mixed(tmp_path):
    stores, _, _ = weighted_packed(tmp_path, (1.0, 1.0), tokenizer="gpt2")
    with pytest.raises(ValueError):
        trained(mixed_packed(stores))


# --------------------------------------------------------------------------
# Mixture phases
# --------------------------------------------------------------------------

def packed_stage(stores: str | dict[str, float]) -> Pipeline:
    """One token store's packed rows in a run's order, or a mix of several's (`{url: weight}`)."""
    return ordered(packed(stores)) if isinstance(stores, str) else mixed_packed(stores)


def phased_packed(*entries: tuple[str | dict[str, float], int | None]) -> Loader:
    """Phases each of `packed_stage`'s rows until its step, in batches of four."""
    return trained(phases(*((packed_stage(stores), until) for stores, until in entries)))


def test_phases_switch_the_mixture_at_a_step(tmp_path):
    """Rigel's curriculum: three steps of the first corpus alone, then both
    at equal shares, every step of the second phase half of each."""
    _, first, second = weighted_packed(tmp_path, (1.0, 1.0))
    both = {first: 1.0, second: 1.0}

    steps = packed_rows(phased_packed((first, 3), (both, None)).train(DataPartition()), 7)

    assert all(corpus_of(row) == 0 for step in steps[:3] for row in step)
    assert all([corpus_of(row) for row in step].count(1) == 2 for step in steps[3:])


def test_every_phase_declares_the_most_documents_a_window_of_any_phase_holds(tmp_path):
    """Packed phases step one program: a phase whose windows hold one document each and one whose
    windows hold four both declare four (`Coordinates.max_segments`), not their own most nor the
    window's length."""
    whole = document_dir(tmp_path / "whole", [[index + 1] * 7 for index in range(8)])
    short = document_dir(tmp_path / "short", [[index + 101] for index in range(32)])
    stream = phased_packed((whole, 2), (short, None)).train(DataPartition())
    batches = list(itertools.islice(stream, 4))
    assert [int(batch["text_coordinates"].segments.max()) for batch in batches] == [1, 1, 4, 4]
    assert {batch["text_coordinates"].max_segments for batch in batches} == {4}


def test_a_run_of_one_mixture_resumes_into_phases_that_begin_with_it(tmp_path):
    """Resuming onto a changed mixture is refused, but resuming onto a phase
    list whose first phase is the run's mixture, with a boundary it has not
    passed, is the same run with a switch ahead of it, at any process count."""
    _, first, second = weighted_packed(tmp_path, (1.0, 1.0))
    both = {first: 1.0, second: 1.0}
    stopped = trained(ordered(packed(first))).train(DataPartition())
    packed_rows(stopped, 2)
    state = stopped.get_state()
    open_stream = phased_packed((first, 4), (both, None)).train
    whole = pooled(open_stream, 1, 7, rows=windows_of)

    for processes in (1, 2):
        assert pooled(open_stream, processes, 5, state, rows=windows_of) == whole[2:]
    with pytest.raises(ValueError):
        phased_packed((both, 4), (first, None)).train(DataPartition()).set_state(state)
    with pytest.raises(ValueError):
        phased_packed((first, 1), (both, None)).train(DataPartition()).set_state(state)


def test_a_phased_run_resumes_past_a_switch_and_refuses_a_changed_history(tmp_path):
    """Two steps into the second phase, the position names the first phase
    and where it ended. The same list, or one with a phase appended, resumes
    on the next step; a list that moved the finished boundary or changed the
    finished mixture, or a run of one order, is refused."""
    _, first, second = weighted_packed(tmp_path, (1.0, 1.0))
    both = {first: 1.0, second: 1.0}
    stream = phased_packed((first, 3), (both, None)).train(DataPartition())
    packed_rows(stream, 5)
    state = stream.get_state()
    rest = packed_rows(stream, 3)

    saved = json.loads(state)["dew_global_position"]
    assert saved["records"] == 20 and saved["completed"][0][1] == 12
    for entries in (((first, 3), (both, None)), ((first, 3), (both, 9), (second, None))):
        resumed = phased_packed(*entries).train(DataPartition())
        resumed.set_state(state)
        assert packed_rows(resumed, 3) == rest
    for changed in (((first, 2), (both, None)), ((second, 3), (both, None))):
        with pytest.raises(ValueError):
            phased_packed(*changed).train(DataPartition()).set_state(state)
    with pytest.raises(ValueError):
        trained(mixed_packed(both)).train(DataPartition()).set_state(state)


def test_phases_are_a_list_of_ends_and_refuse_a_ramp(tmp_path):
    _, first, second = weighted_packed(tmp_path, (1.0, 1.0))
    with pytest.raises(ValueError):
        phased_packed((first, 3), (second, 3), (first, None)).train(DataPartition())
    with pytest.raises(ValueError):
        phased_packed((first, 3), (second, 5)).train(DataPartition())
    with pytest.raises(TypeError):
        ramped(phased_packed((first, 3), (second, None)), Ramp(start=2, increment=2,
                                                               samples=8)).train(DataPartition())


def scheduled_packed(*entries: tuple[str | dict[str, float], float | None]) -> Loader:
    """`phased_packed`'s stages as a schedule, each for its committed tokens (`schedule`)."""
    return trained(schedule([(packed_stage(stores), tokens) for stores, tokens in entries]))


def test_a_schedule_reads_a_stage_until_the_run_ends_it_then_reads_as_its_phases_do(tmp_path):
    """A stage of a schedule reads on until the run ends it, where its commits reached its tokens
    (`PhasedStream.ending`). Ended after three steps, it reads what phases switching at that step
    read, the first stage's corpus continuing in the second where the stage left it, not where the
    rows read past the end did, on one process or two; its position holds where the stage ended, and a
    position past that resumes there."""
    _, first, second = weighted_packed(tmp_path, (1.0, 1.0))
    both = {first: 1.0, second: 1.0}
    whole = pooled(phased_packed((first, 3), (both, None)).train, 1, 7, rows=windows_of)
    def open_stream(partition: DataPartition) -> PhasedStream:
        stream = scheduled_packed((first, 1000), (both, None)).train(partition)
        assert isinstance(stream, PhasedStream)
        return stream

    stream = open_stream(DataPartition())
    assert packed_rows(stream, 3) == whole[:3]
    state = stream.get_state()
    # Read ahead of the end, as a prefetch worker reads: the first corpus alone, never trained on.
    assert all(corpus_of(row) == 0 for step in packed_rows(stream, 2) for row in step)

    ended = open_stream(DataPartition()).ending(state)
    assert position.read(ended).completed[0][1] == 12
    for processes in (1, 2):
        assert pooled(open_stream, processes, 4, ended, rows=windows_of) == whole[3:7]
    resumed = open_stream(DataPartition())
    resumed.set_state(ended)
    packed_rows(resumed, 2)
    past = resumed.get_state()
    again = open_stream(DataPartition())
    again.set_state(past)
    assert packed_rows(again, 2) == whole[5:7]


def test_a_schedule_counts_the_tokens_of_every_stage_but_its_last(tmp_path):
    """Every stage but the last commits a count of tokens, at least one; the last runs on, or ends
    the data at its own. Only a stage on the token clock, and not the last, ends where the run
    stands, and phases on the step clock end at their steps."""
    _, first, second = weighted_packed(tmp_path, (1.0, 1.0))
    for stages in ((), ((ordered(packed(first)), None), (ordered(packed(second)), 10)),
                   ((ordered(packed(first)), 0.5), (ordered(packed(second)), None))):
        with pytest.raises(ValueError):
            schedule(stages)
    def opened(loader: Loader) -> PhasedStream:
        stream = loader.train(DataPartition())
        assert isinstance(stream, PhasedStream)
        return stream

    stream = opened(scheduled_packed((first, 10), (second, None)))
    packed_rows(stream, 1)
    ended = stream.ending(stream.get_state())
    with pytest.raises(ValueError):
        opened(scheduled_packed((first, 10), (second, None))).ending(ended)
    phased = opened(phased_packed((first, 3), (second, None)))
    packed_rows(phased, 1)
    with pytest.raises(ValueError):
        phased.ending(phased.get_state())


# --------------------------------------------------------------------------
# The batch ramp: MaxText's schedule
# --------------------------------------------------------------------------

def ramp_batches(ramp: Ramp, final: int, steps: int, records: int = 0) -> list[int]:
    """The global batch this ramp reads at each of `steps` steps, from a
    position of `records` records read."""
    stages, reads = ramp.stages(final), []
    for _ in range(steps):
        stage = stages[max(index for index, stage in enumerate(stages)
                           if stage.records <= records)]
        reads.append(stage.batch)
        records += stage.batch
    return reads


@pytest.mark.parametrize("ramp, final, message", [
    (Ramp(start=4, increment=2, samples=48), 4, "nothing to ramp"),
    (Ramp(start=4, increment=2, samples=48), 3, "nothing to ramp"),
    (Ramp(start=4, increment=3, samples=48), 8, "whole increments"),
    (Ramp(start=0, increment=2, samples=48), 8, "counts records"),
    (Ramp(start=4, increment=2, samples=0), 8, "counts records"),
])
def test_a_schedule_that_cannot_be_run_is_refused(ramp, final, message):
    with pytest.raises(ValueError):
        ramp.stages(final)


def test_a_ramp_stage_that_does_not_split_into_the_shares_is_refused():
    """A stage of six records a step has no equal share for each of four
    readers; the stream refuses it at the step that reads it."""
    stream = ramped(indexed_data(64, 8), Ramp(start=6, increment=2, samples=48)).train(
        DataPartition(0, 4))
    with pytest.raises(ValueError):
        next(stream)
    stream.close()


# --------------------------------------------------------------------------
# The batch ramp over a real stream
# --------------------------------------------------------------------------

def test_epochs_under_a_ramp_read_that_many_passes_of_records():
    """Two epochs of 64 records ramping from 8 to 16 over the first 32 read
    exactly 128 records: four steps of 8, then six of 16."""
    ramp = Ramp(start=8, increment=8, samples=32)
    data = ramped(Loader(train=lambda partition: iter(()), val=None, records=64, batch=16), ramp)
    batches = [stage.batch for stage in ramp.stages(16)]
    assert (data.epoch_steps(1), data.epoch_steps(2)) == (6, 10) and batches == [8, 16]


@pytest.mark.parametrize(("records", "steps"), [(5, 3), (16, 8), (17, 9), (40, 14), (41, 15)])
def test_an_epoch_under_a_ramp_counts_the_step_that_completes_it(records, steps):
    """An order read on past its epoch, its batch ramping from 2 to 4 over the first 16 records,
    counts the steps that read every record of an epoch: inside the ramp's first stage, at its end,
    past it, and in the final batch, the last step leading into the next epoch."""
    ramp = Ramp(start=2, increment=2, samples=16)
    data = ramped(Loader(train=lambda partition: iter(()), val=None, records=records, batch=4, tail="read"),
                  ramp)
    assert data.epoch_steps(1) == steps


@pytest.mark.parametrize(("records", "keep", "real"), [(6, False, [4]), (7, True, [4, 3]), (5, True, [4, 1])])
def test_a_ramp_over_an_order_read_once_ends_where_its_records_do(records, keep, real):
    """An order read once under a ramp of four-record steps takes the steps its tail policy counts and
    no more: six records in batches of six that drop a short one take one whole step, the other two
    dropped; seven or five in batches that keep one take the steps that hold them all, the last
    holding the order's last records real and the rest filled, never a step of padding alone."""
    order = Pipeline(Memory(Indexed(1, records))).repeat(1).batch(6, drop_remainder=not keep)
    data = ramped(compile(order, loading=READ), Ramp(start=4, increment=2, samples=100))
    assert data.epoch_steps(1) == len(real)
    stream = data.train(DataPartition())
    try:
        steps = list(stream)
    finally:
        stream.close()
    held = [np.asarray(step[VALID_ROWS], bool) if VALID_ROWS in step else np.ones(4, bool) for step in steps]
    assert [int(np.count_nonzero(valid)) for valid in held] == real
    ids = sorted(int(value) for step, valid in zip(steps, held, strict=True)
                 for value in np.asarray(step["id"])[valid])
    assert ids == [1000 + index for index in range(sum(real))]


@pytest.mark.parametrize("keep", [False, True])
def test_a_ramp_over_batches_repeated_a_counted_number_of_times_reads_every_pass(keep):
    """Twelve records in batches of six, the batches repeated twice, under a ramp of four-record
    steps: the ramp reads both passes, the six steps two epochs count, every record twice."""
    order = Pipeline(Memory(Indexed(1, 12))).batch(6, drop_remainder=not keep).repeat(2)
    data = ramped(compile(order, loading=READ), Ramp(start=4, increment=2, samples=100))
    assert (data.epoch_steps(1), data.epoch_steps(2)) == (3, 6)
    stream = data.train(DataPartition())
    try:
        steps = list(stream)
    finally:
        stream.close()
    assert len(steps) == 6
    ids = sorted(int(value) for step in steps for value in np.asarray(step["id"]))
    assert ids == sorted([1000 + index for index in range(12)] * 2)


def indexed_data(records: int, batch: int, seed: int = 0) -> Loader:
    """A dataset whose records say which they are, compiled to the global
    order every dataset reads through."""
    return compile(Pipeline(Memory(Indexed(1, records))).shuffle(seed).repeat().batch(batch), seed=seed,
                   loading=READ)


def stitched_data(records: int, batch: int, seed: int) -> Loader:
    """The order `indexed_data` compiles, read straight off plain Grain at
    `batch` records a step, behind a position that counts records and so
    carries on at another batch."""

    def train(partition: DataPartition) -> GlobalStream:
        def open_at(offset: int):
            order = pygrain.MapDataset.source(Indexed(1, records)).seed(seed).shuffle(seed).repeat(None)
            mine = order[offset + partition.index::partition.count]
            return iter(mine.to_iter_dataset(pygrain.ReadOptions(1, 8)).batch(partition.rows(batch)))

        order = {"records": records, "reshuffled from seed": seed}
        return GlobalStream(open_at, batch, Plan("global", Manifest({}), order), READ)

    return Loader(train=train, val=None, records=records, batch=batch)


def test_a_ramped_stream_reads_the_records_the_flat_stream_reads():
    """The ramp cuts the same order into different steps, so the records it
    hands over are the flat stream's records in the flat stream's order: a
    stage change neither skips one nor reads one twice."""
    flat = [value for step in taken(indexed_data(64, 4).train(DataPartition()), 24) for value in step]

    stream = ramped(indexed_data(64, 4), Ramp(start=2, increment=1, samples=6)).train(DataPartition())
    ramp = taken(stream, 24)

    # Two steps of two, one of three, then the run's own four: the schedule
    # `test_the_ramp_reads_the_batch_maxtext_reads_at_every_step` pins.
    assert [len(step) for step in ramp] == [2, 2, 3] + [4] * 21
    assert [value for step in ramp for value in step] == flat[:sum(
        len(step) for step in ramp)]


def batch_shuffled_data(records: int, batch: int) -> Loader:
    """A dataset whose records say which they are, batched in order and its batches shuffled after
    the batch, so a record count inside a batch names rows of a batch the order moved."""
    return compile(Pipeline(Memory(Indexed(1, records))).repeat().batch(batch).shuffle(7, window=3),
                   seed=0, loading=READ)


@pytest.mark.parametrize("data", [indexed_data, batch_shuffled_data])
def test_a_ramped_stream_resumes_on_the_records_it_had_not_read(data):
    """The rolling buffer holds records the run has not trained on, so they
    are not in the position: a resume reads them, and reads them once, in
    the batches the order put them in, a shuffle after the batch's too."""
    schedule = Ramp(start=2, increment=1, samples=6)
    stream = ramped(data(64, 4), schedule).train(DataPartition())
    taken(stream, 4)
    state = stream.get_state()
    rest = taken(stream, 6)

    resumed = ramped(data(64, 4), schedule).train(DataPartition())
    resumed.set_state(state)

    assert json.loads(state)["dew_global_position"]["records"] == 11
    assert taken(resumed, 6) == rest


def test_a_ramped_stream_put_back_mid_read_forgets_its_buffer():
    """A restore into a stream that is already reading, as the prefetch
    worker restores the stream it opened, hands over the records at the
    saved count and none it had buffered past it."""
    schedule = Ramp(start=2, increment=1, samples=6)
    stream = ramped(indexed_data(64, 4), schedule).train(DataPartition())
    taken(stream, 1)
    state = stream.get_state()
    rest = taken(stream, 6)

    stream.set_state(state)

    assert taken(stream, 6) == rest


@pytest.mark.parametrize("data", [indexed_data, batch_shuffled_data])
def test_a_ramped_steps_position_counts_every_process(data):
    """Each process cuts its own share of a stage's batch, and the position
    is the records all of them handed over: two processes three and four
    steps into a ramp of two, then four, have read six and eight records,
    and the pool's steps are the single process's steps with the shares
    interleaved, resumed inside a batch or after one."""
    schedule = Ramp(start=2, increment=2, samples=8)

    build = ramped(data(64, 4), schedule).train

    whole = pooled(build, 1, 6)
    assert [len(step) for step in whole] == [2, 2, 2, 2, 4, 4]
    assert pooled(build, 2, 6) == whole
    for steps in (3, 4):
        stopped = build(DataPartition(1, 2))
        taken(stopped, steps)
        assert json.loads(stopped.get_state())["dew_global_position"]["records"] == 2 * steps
        assert pooled(build, 2, 6 - steps, stopped.get_state()) == whole[steps:]


@pytest.mark.parametrize("data", [indexed_data, batch_shuffled_data])
def test_a_ramped_run_resumed_by_more_processes_inside_a_batch_reads_on_in_order(data):
    """One process stops a ramp of two, then four, after its first step, two records into a batch
    of four. Four processes resume it in the four-record stage: the records after the count go to
    them in turn from the first on, so each takes the place the count leaves it, and the pool's
    steps are the one process's."""
    build = ramped(data(64, 4), Ramp(start=2, increment=2, samples=2)).train
    whole = pooled(build, 1, 5)
    stopped = build(DataPartition())
    taken(stopped, 1)
    assert [len(step) for step in whole] == [2, 4, 4, 4, 4]
    assert json.loads(stopped.get_state())["dew_global_position"]["records"] == 2
    assert pooled(build, 4, 4, stopped.get_state()) == whole[1:]


def test_a_ramp_reads_through_a_captioned_datasets_tokenized_stream():
    """An image or video dataset opens `tokenized` over its global stream, so
    the ramp has to cut the batches that stage hands over and save the
    position it forwards; a ramp that only knew the bare stream refused
    every captioned dataset by the wrapper's name."""

    class Captioned(Indexed):
        def __getitem__(self, index: int) -> dict:
            return {**super().__getitem__(index), CAPTION: f"caption {index}"}

    def lengths(captions):
        return {"length": np.asarray([len(caption) for caption in captions], np.int32)}

    def captioned(batch: int) -> Loader:
        return compile(Pipeline(Memory(Captioned(1, 64))).shuffle(0).repeat().batch(batch).map(
                           EncodeCaptions(lengths)), read="memory", seed=0, loading=READ)

    flat = [value for step in taken(captioned(4).train(DataPartition()), 12) for value in step]
    schedule = Ramp(start=2, increment=1, samples=6)
    stream = ramped(captioned(4), schedule).train(DataPartition())
    steps = list(itertools.islice(stream, 6))
    state = stream.get_state()
    rest = taken(stream, 3)

    assert [len(step["id"]) for step in steps] == [2, 2, 3, 4, 4, 4]
    assert [int(value) for step in steps for value in step["id"]] == flat[:19]
    assert all(len(step["length"]) == len(step["id"]) for step in steps), (
        "the tokenized field was cut to another step than the ids")
    resumed = ramped(captioned(4), schedule).train(DataPartition())
    resumed.set_state(state)
    assert json.loads(state)["dew_global_position"]["records"] == 19
    assert taken(resumed, 3) == rest


class OwnBatches:
    """A stream that cut its batches itself, with no position at all."""

    def __iter__(self):
        return self

    def __next__(self):
        return {"id": np.zeros((4,), np.int32)}


class ShardOffset(OwnBatches):
    """The same, reporting where its own shard stopped, as grain does."""

    def get_state(self):
        return {"next_index": 3}

    def set_state(self, state):
        pass


@pytest.mark.parametrize("stream", [OwnBatches, ShardOffset])
def test_only_a_stream_whose_position_is_a_global_count_can_ramp(stream):
    data = Loader(train=lambda partition: stream(), val=None, records=64, batch=8)
    with pytest.raises(TypeError):
        ramped(data, Ramp(start=4, increment=4, samples=16)).train(DataPartition())


# --------------------------------------------------------------------------
# The batch ramp through the trainer
# --------------------------------------------------------------------------

class Regression(Objective):
    """Squared error of an affine map on records that differ per index, so
    which records a step read shows up in the parameters."""

    def __init__(self):
        self.model = Affine()

    def fresh_variables(self, key, held):
        return self.model.init(key, np.zeros((1, FEATURES), np.float32))

    def errors(self, variables, batch):
        """Each record's mean squared error."""
        import jax.numpy as jnp

        features = jnp.stack([jnp.sin(batch["id"].astype(jnp.float32) * (index + 1))
                              for index in range(FEATURES)], axis=-1)
        target = jnp.stack([features[:, 0] * 2, features[:, 1] - 1], axis=-1)
        return jnp.mean((self.model.apply(variables, features) - target) ** 2, axis=-1)

    def loss(self, variables, batch, step):
        import jax.numpy as jnp

        return jnp.mean(self.errors(variables, batch)), Aux({})


class Composite(Regression):
    """`Regression` whose statistics are a pair it reduces itself, which an accumulation window keeps
    by replaying the window's batches."""

    def loss(self, variables, batch, step):
        import jax.numpy as jnp

        errors = self.errors(variables, batch)
        return (jnp.sum(errors), jnp.asarray(errors.shape[0], errors.dtype)), Aux({})

    def reduce_loss(self, stats):
        total, count = stats
        return total / count, count > 0


class Pooled(Regression):
    """`Regression` as its records' summed error and their count, which an accumulation window pools by
    the records each batch read."""

    def loss(self, variables, batch, step):
        import jax.numpy as jnp

        errors = self.errors(variables, batch)
        return Ratio(jnp.sum(errors), jnp.asarray(errors.shape[0], errors.dtype)), Aux({})


def regression_trainer(directory: Path) -> Trainer:
    return Trainer(
        Regression(), optax.sgd(0.5), key=jax.random.key(0),
        layout=Layout(min_shard=1, tolerance=1.0),
        checkpoints=Checkpoints(str(directory), keep=8))


def parameters(state) -> dict:
    return {"/".join(str(part) for part in path): np.asarray(leaf)
            for path, leaf in jax.tree_util.tree_flatten_with_path(state.variables)[0]}


def assert_same(left: dict, right: dict, why: str) -> None:
    assert set(left) == set(right)
    for name in left:
        np.testing.assert_array_equal(left[name], right[name], err_msg=f"{why}: {name}")


def endless_chains(name: str) -> tuple[Pipeline, Pipeline]:
    """A chain a fit reads without being given a repeat, and the chain it reads: phases or a mix whose
    corpora repeat themselves read as written; phases of passes batched on their own, repeated after
    each phase's batch as the chain's own repeat at its end is; and phases or a mix whose repeats are
    bounded, repeated before the batch."""
    first = Pipeline(Memory(Indexed(1, 64))).shuffle(0)
    second = Pipeline(Memory(Indexed(2, 64))).shuffle(1)
    if name == "phases":
        chain = phases((first.repeat(), 3), (second.repeat(), None)).batch(8)
        return chain, chain
    if name == "mix":
        chain = mix([first.repeat(), second.repeat()]).batch(8)
        return chain, chain
    if name == "batched-phases":
        chain = phases((first.batch(8), 3), (second.batch(8), None))
        return chain, chain.repeat()
    if name == "bounded-phases":
        # The phases' shared take bounds the one that repeats itself: the chain repeats after it.
        bounded = phases((first.repeat(), 2), (second, None)).take(16)
        return bounded.batch(8), bounded.repeat().batch(8)
    members = [first.repeat().take(12), second.repeat().take(12)]
    return mix(members).batch(8), mix(members).repeat().batch(8)


@pytest.mark.parametrize("name", ["phases", "mix", "batched-phases", "bounded-phases", "bounded-mix"])
def test_a_fit_repeats_a_chain_only_where_its_read_can_end(tmp_path, name):
    """A fit reads a dataset without a repeat endlessly by adding one where its read can end
    (`analysis.reach`): phases whose corpora repeat themselves, and a mix of members that all do, are
    read as written, while phases of batched passes and a mix of bounded repeats repeat as before,
    landing where the chain written with that repeat lands."""
    chain, read = endless_chains(name)
    fitted = regression_trainer(tmp_path / "fitted").fit(chain, steps=5)
    written = regression_trainer(tmp_path / "written").fit(compile(read, seed=0), steps=5)
    assert_same(parameters(fitted), parameters(written), f"{name} as the fit reads it")


class TokenCounted(Regression):
    """`Regression` whose records each commit `id % 3 + 1` real tokens, so the steps of a run commit
    unevenly."""

    counts_tokens = True

    def scored_tokens(self, batch):
        return np.asarray(batch["id"]) % 3 + 1


def token_trainer(directory: Path, tracker: RecordingTracker | None = None,
                  accumulation: int = 1) -> Trainer:
    return Trainer(TokenCounted(), optax.sgd(0.5), key=jax.random.key(0), accumulation=accumulation,
                   layout=Layout(min_shard=1, tolerance=1.0),
                   checkpoints=Checkpoints(str(directory), keep=8), tracker=tracker)


def test_a_cooldown_branch_ends_a_schedules_stage_where_a_run_decayed_there_does(tmp_path):
    """A cooldown branch from before a schedule's first stage ends trains on across its end, the stage
    ending at the step whose commit brings the branch's own committed tokens to its 50, as a run's
    does: the branch scores, bit for bit, what a run whose rate decays from the same step over the
    same steps scores, and the run it left trains on as one with no cooldown."""
    from dew.training.cooldown import Cooldowns

    first = Pipeline(Memory(Indexed(1, 64))).shuffle(0).repeat()
    second = Pipeline(Memory(Indexed(2, 64))).shuffle(1).repeat()
    data = schedule([(first, 50), (second, None)]).batch(8)
    held_out = Pipeline(Memory(Indexed(3, 16))).batch(8)

    def run(optimizer, steps: int, tracker=None, cooldowns=None):
        trainer = Trainer(TokenCounted(), optimizer, key=jax.random.key(0), tracker=tracker,
                          layout=Layout(min_shard=1, tolerance=1.0))
        return trainer.fit(data, steps=steps, log_every=1, eval_every=steps, validation=held_out,
                           cooldowns=cooldowns)

    peak = 0.5
    constant = optax.inject_hyperparams(optax.sgd)(learning_rate=peak)
    tracker = RecordingTracker()
    branched = run(constant, 8, tracker, Cooldowns(at=(2,), length=4, shape="linear"))
    cooled = {step: scalars["cooldown_val/loss"] for step, scalars in tracker.scalars
              if "cooldown_val/loss" in scalars}
    decaying = optax.inject_hyperparams(optax.sgd)(learning_rate=optax.join_schedules(
        [optax.constant_schedule(peak), optax.linear_schedule(peak, 0.0, 4)], [2]))
    decayed = RecordingTracker()
    run(decaying, 6, decayed)
    committed = list(itertools.accumulate(scalars["train/tokens"] for _, scalars in decayed.scalars
                                          if "train/tokens" in scalars))
    assert committed[1] < 50 <= committed[-1], "the branch's steps cross the stage's end"
    finals = [scalars["val/loss"] for _, scalars in decayed.scalars if "val/loss" in scalars]
    assert cooled == {2: finals[-1]}
    assert_same(parameters(branched), parameters(run(constant, 8)), "the run a branch left")


def test_a_cooldown_branch_past_a_schedules_last_tokens_records_no_score(tmp_path):
    """A schedule whose last stage ends its data at its tokens ends a cooldown branch's data there too:
    a branch longer than the tokens left reads to that end and records no score, as a branch whose data
    runs out does, and the run ends at its own end as one with no cooldown."""
    from dew.training.cooldown import Cooldowns

    first = Pipeline(Memory(Indexed(1, 64))).shuffle(0).repeat()
    second = Pipeline(Memory(Indexed(2, 64))).shuffle(1).repeat()
    data = schedule([(first, 30), (second, 30)]).batch(8)
    held_out = Pipeline(Memory(Indexed(3, 16))).batch(8)

    def run(tracker=None, cooldowns=None):
        trainer = Trainer(TokenCounted(), optax.inject_hyperparams(optax.sgd)(learning_rate=0.5),
                          key=jax.random.key(0), tracker=tracker, layout=Layout(min_shard=1, tolerance=1.0))
        return trainer.fit(data, log_every=1, eval_every=1, validation=held_out, cooldowns=cooldowns)

    tracker = RecordingTracker()
    branched = run(tracker, Cooldowns(at=(1,), length=12, shape="linear"))
    assert not any("cooldown_val/loss" in scalars for _, scalars in tracker.scalars)
    assert_same(parameters(branched), parameters(run()), "the run a branch left")


@pytest.mark.parametrize("accumulation", [1, 2])
def test_a_schedule_ends_its_stage_at_the_step_whose_commit_reaches_its_tokens(tmp_path, accumulation):
    """A run on a schedule ends its first stage at the step whose commit brings the run's committed
    tokens to the stage's 50: it lands where phases switching at that step land, its parameters bit
    for bit, the read ahead of the end untrained, and its saved position holds where the stage
    ended. A run stopped before the end, or at the step that reached it, resumes to the same
    parameters."""
    first = Pipeline(Memory(Indexed(1, 64))).shuffle(0).repeat()
    second = Pipeline(Memory(Indexed(2, 64))).shuffle(1).repeat()
    data = schedule([(first, 50), (second, None)]).batch(8)
    tracker = RecordingTracker()
    scheduled = token_trainer(tmp_path / "scheduled", tracker, accumulation)
    whole = parameters(scheduled.fit(data, steps=8, log_every=1))
    committed = list(itertools.accumulate(scalars["train/tokens"] for _, scalars in tracker.scalars
                                          if "train/tokens" in scalars))
    step = next(index + 1 for index, tokens in enumerate(committed) if tokens >= 50)
    assert 1 < step < 8
    phased = phases((first, step), (second, None)).batch(8)
    switched = token_trainer(tmp_path / "phased", accumulation=accumulation).fit(phased, steps=8)
    assert_same(whole, parameters(switched), "the schedule against phases switching where its commits did")
    assert scheduled.checkpoints is not None
    _, saved = scheduled.checkpoints.restore(share=DataPartition())
    assert saved is not None and position.read(saved).completed[0][1] == step * 8
    for stopped in (2, step):
        token_trainer(tmp_path / f"stopped-{stopped}", accumulation=accumulation).fit(data, steps=stopped)
        resumed = token_trainer(tmp_path / f"stopped-{stopped}", accumulation=accumulation).fit(data, steps=8)
        assert_same(whole, parameters(resumed), f"resumed from step {stopped}")
    # The last stage's own tokens end the data: the run's last step is the one whose commit reaches
    # them, saved as a token budget's last step is, where phases running on to that step land.
    last = next(index + 1 for index, tokens in enumerate(committed) if tokens >= committed[step - 1] + 30)
    ended = token_trainer(tmp_path / "ended", accumulation=accumulation)
    state = ended.fit(schedule([(first, 50), (second, committed[step - 1] + 30 - 50)]).batch(8))
    assert int(state.step) == last and ended.checkpoints is not None and ended.checkpoints.latest == last
    ran_on = token_trainer(tmp_path / "ran-on", accumulation=accumulation).fit(phased, steps=last)
    assert_same(parameters(state), parameters(ran_on),
                "a schedule whose data ends against phases running on to its last step")


def test_a_schedules_completed_stage_cannot_move_under_a_resumed_run_and_its_later_stages_can(tmp_path):
    """A run that completed its first stage at 50 committed tokens saves that threshold beside the
    record count it ended at: resumed into a schedule whose first stage ends at 500, it is refused,
    where it once read the stage after it at once, and resumed into one whose stage past the one it
    stands in changed, it reads on (row 1871, DATA-04)."""
    first = Pipeline(Memory(Indexed(1, 64))).shuffle(0).repeat()
    second = Pipeline(Memory(Indexed(2, 64))).shuffle(1).repeat()
    third = Pipeline(Memory(Indexed(3, 64))).shuffle(2).repeat()
    trained = token_trainer(tmp_path / "run")
    trained.fit(schedule([(first, 50), (second, 10_000), (third, None)]).batch(8), steps=8)
    assert trained.checkpoints is not None
    _, saved = trained.checkpoints.restore(share=DataPartition())
    assert saved is not None and position.read(saved).thresholds == (50,)
    with pytest.raises(ValueError):
        token_trainer(tmp_path / "run").fit(schedule([(first, 500), (second, 10_000), (third, None)]).batch(8),
                                           steps=10)
    fourth = Pipeline(Memory(Indexed(4, 64))).shuffle(3).repeat()
    state = token_trainer(tmp_path / "run").fit(
        schedule([(first, 50), (second, 10_000), (fourth, None)]).batch(8), steps=10)
    assert int(state.step) == 10


def test_a_configured_run_on_a_schedule_ends_where_its_data_does(tmp_path):
    """A run configured with no length, on a schedule whose last stage ends its data at its tokens, is
    bounded by them as a fit of the schedule is (`budget_tokens`): `RunConfig.train` trains to the step
    whose commit reaches them, where `Trainer.fit` on the same schedule ends."""
    from dew.config import RunConfig, TrainerConfig
    from dew.optim import OptimConfig

    first = Pipeline(Memory(Indexed(1, 64))).shuffle(0).repeat()
    second = Pipeline(Memory(Indexed(2, 64))).shuffle(1).repeat()
    data = schedule([(first, 50), (second, 30)]).batch(8)
    fitted = token_trainer(tmp_path / "fitted").fit(data)
    config = RunConfig(optim=OptimConfig(optimizer="adamw", learning_rate=1e-3), trainer=TrainerConfig(
        checkpoint_dir=str(tmp_path / "configured"), steps=None, epochs=None, tokens=None, log_every=1,
        eval_every=None, checkpoint_every=None))
    configured = config.train(TokenCounted(), data, name="scheduled")
    assert 1 < int(configured.step) == int(fitted.step)


# Every stage's batch is a multiple of the eight simulated devices, because a
# batch's rows are sharded over them.
RAMP = Ramp(start=8, increment=8, samples=64)
FINAL = 24


def test_a_ramped_run_lands_where_the_fixed_batch_runs_stitched_together_land(tmp_path):
    """Three stages of a ramp against three runs at those batches, each
    carrying on where the last left the data. One optimizer update a step in
    both, the loss a mean over the records the step read, and the parameters
    the same parameters.

    The stitched runs are the definition of what a ramp should do, and they
    are only comparable because a saved data position is a record count over
    an order the batch does not enter: run two opens where run one stopped
    while reading twice as many records a step.
    """
    records, seed = 4096, 3
    stages = RAMP.stages(FINAL)
    lengths = [(stages[1].records - stages[0].records) // stages[0].batch,
               (stages[2].records - stages[1].records) // stages[1].batch, 3]

    ramped_run = regression_trainer(tmp_path / "ramped").fit(
        ramped(indexed_data(records, FINAL, seed), RAMP),
        steps=sum(lengths), log_every=100, checkpoint_every=None)

    stitched = [regression_trainer(tmp_path / "stitched").fit(
                    stitched_data(records, batch, seed), steps=sum(lengths[:stage + 1]),
                    log_every=100, checkpoint_every=None)
                for stage, batch in enumerate([8, 16, 24])][-1]

    assert [stage.batch for stage in stages] == [8, 16, 24]
    assert lengths == [4, 2, 3]
    assert int(ramped_run.step) == int(stitched.step) == sum(lengths)
    assert int(ramped_run.updates) == sum(lengths), "one update a step, at every stage"
    assert_same(parameters(ramped_run), parameters(stitched),
                "the ramp trained something other than the stitched runs")


def test_a_ramped_run_killed_mid_ramp_lands_where_the_run_nobody_stopped_lands(tmp_path):
    """The ramp's position is the data position, so a restart continues the
    schedule where it was: same stage, same records, same parameters."""
    records, seed, steps = 4096, 5, 8

    whole = regression_trainer(tmp_path / "whole").fit(
        ramped(indexed_data(records, FINAL, seed), RAMP), steps=steps, log_every=100,
        checkpoint_every=None)

    # Killed inside the second stage, two steps past its last checkpoint.
    regression_trainer(tmp_path / "resumed").fit(
        ramped(indexed_data(records, FINAL, seed), RAMP), steps=5, log_every=100,
        checkpoint_every=3)
    resumed = regression_trainer(tmp_path / "resumed").fit(
        ramped(indexed_data(records, FINAL, seed), RAMP), steps=steps, log_every=100,
        checkpoint_every=3)

    assert int(resumed.step) == steps
    assert_same(parameters(whole), parameters(resumed),
                "the resumed ramp did not land on the uninterrupted run")


def test_a_ramp_stage_begins_where_an_update_does_and_an_update_pools_one_stage(tmp_path):
    """A batch ramp changes stage where an update begins, as a batch schedule applies per optimizer
    step: the stage of sixteen, due at step three, begins at step four, where a window of two steps
    does, so each update pools two batches of eight. Four such steps land where two batches of
    sixteen, the same records in the same order, land."""
    records, seed, layout = 4096, 7, Layout(min_shard=1, tolerance=1.0)
    ramp = Ramp(start=8, increment=8, samples=24)
    assert ramp.stages(16) == (Stage(batch=8, records=0), Stage(batch=16, records=24))
    assert ramp.stages(16, window=2) == (Stage(batch=8, records=0), Stage(batch=16, records=32))
    assert ramp.steps_for(64, 16, window=2) == 6
    state = Trainer(Pooled(), optax.sgd(0.5), key=jax.random.key(0), accumulation=2, layout=layout).fit(
        ramped(indexed_data(records, 16, seed), ramp), steps=4, log_every=100)
    batched = Trainer(Pooled(), optax.sgd(0.5), key=jax.random.key(0), layout=layout).fit(
        stitched_data(records, 16, seed), steps=2, log_every=100)

    assert int(state.step) == 4
    assert int(state.updates) == int(batched.updates) == 2
    pooled, whole = parameters(state), parameters(batched)
    assert set(pooled) == set(whole)
    for name, value in whole.items():
        np.testing.assert_allclose(pooled[name], value, rtol=1e-5, atol=1e-6, err_msg=name)


def test_a_configured_run_counts_its_ramped_epochs_in_stages_aligned_to_its_updates():
    """A config's run length and its epoch intervals count a ramp's steps as the trainer's stream cuts
    them, each stage beginning where an update of its accumulation does: an epoch of 64 records ramped
    from 8 to 16 is six steps read one at a time, three of 8 and three of 16, the last leading into
    the next epoch, and six read two to an update, four of 8 and two of 16."""
    from dew.config import TrainerConfig

    data = ramped(indexed_data(64, 16), Ramp(start=8, increment=8, samples=24))
    assert TrainerConfig(epochs=1).total_steps(data) == 6
    config = TrainerConfig(epochs=1, accumulation=2)
    assert config.total_steps(data) == config.eval_interval(data) == config.checkpoint_interval(data) == 6
    with pytest.raises(ValueError):
        Ramp(start=8, increment=8, samples=24).stages(16, window=0)


class Rejecting(Pooled):
    """`Pooled` whose loss is not finite on a batch holding the record `rejected`, as a dynamic loss
    scale rejects an overflowing step."""

    def __init__(self, rejected: int, frozen: bool = False):
        super().__init__()
        self.rejected, self.frozen = rejected, frozen

    def fresh_variables(self, key, held):
        # A frozen bias a host-master run holds resident on the accelerator, apart from the moving weights.
        variables = super().fresh_variables(key, held)
        return freeze(variables, lambda path: "kernel" in str(path)) if self.frozen else variables

    def loss(self, variables, batch, step):
        import jax.numpy as jnp

        stats, aux = super().loss(thaw(variables), batch, step)
        poisoned = jnp.any(batch["id"] == self.rejected)
        return Ratio(jnp.where(poisoned, jnp.inf, stats.total), stats.mass), aux


@pytest.mark.parametrize("route", ["resident", "host", "host-frozen", "resumed"])
def test_a_window_a_rejection_shifted_is_dropped_where_a_ramp_stage_begins(route, tmp_path):
    """A rejected step drops its window, so the next begins a step later than the ramp's stages,
    which begin where unrejected windows do: the stage's first step then drops the window its last
    one left part filled, and no update pools two stages' batches. Rejecting step 0 of a ramp of two
    eight-record steps, then sixteen, two to an update: step 1 alone in its window is dropped at step
    2, which with step 3 makes the run's one update, whether the variables live on the devices or the
    host, the host's beside frozen weights the accelerator holds, and in a run resumed from a
    checkpoint written at the stage's start."""
    records, seed = 4096, 7
    ramp = Ramp(start=8, increment=8, samples=16)
    first = next(iter(ramped(indexed_data(records, 16, seed), ramp).train(DataPartition(window=2))))
    layout = Layout(host=("variables",) if route.startswith("host") else (), min_shard=1, tolerance=1.0)

    def trainer():
        checkpoints = Checkpoints(str(tmp_path / "run"), keep=8) if route == "resumed" else None
        return Trainer(Rejecting(int(first["id"][0]), frozen=route == "host-frozen"), optax.sgd(0.5),
                       key=jax.random.key(0), accumulation=2, dynamic_scale=True, layout=layout,
                       checkpoints=checkpoints)

    if route == "resumed":
        trainer().fit(ramped(indexed_data(records, 16, seed), ramp), steps=2, log_every=100,
                      checkpoint_every=2)
    state = trainer().fit(ramped(indexed_data(records, 16, seed), ramp), steps=4, log_every=100)
    assert (int(state.step), int(state.microstep), int(state.updates)) == (4, 2, 1)


def test_each_aligned_stage_begins_from_its_own_schedule_and_one_left_no_step_is_dropped():
    """Every stage's start moves to the update boundary at or after its own, not after the moved
    start before it, so no stage drifts past its end; a stage whose start and the next's meet
    there runs no step and is gone. A window is at least one step."""
    ramp = Ramp(start=4, increment=4, samples=12)
    # A step each of 4, 8 and 12 records, from steps 0, 1 and 2; 16 from step 3.
    assert ramp.stages(16) == (Stage(batch=4, records=0), Stage(batch=8, records=4),
                               Stage(batch=12, records=12), Stage(batch=16, records=24))
    assert ramp.stages(16, window=2) == (Stage(batch=4, records=0), Stage(batch=12, records=8),
                                         Stage(batch=16, records=32))
    assert ramp.stages(16, window=4) == (Stage(batch=4, records=0), Stage(batch=16, records=16))
    with pytest.raises(ValueError):
        DataPartition(window=0)


def test_a_window_that_replays_its_batches_pools_a_ramps_stages_as_batches_of_their_records(tmp_path):
    """A loss of composite statistics replays its window's batches, whose shape a ramp changes at each
    stage; each stage begins where a window does, and the window's replay slots are the new stage's
    from there. Four steps of eight and then sixteen, two to a window, land where a batch of sixteen
    and then one of thirty-two, the same records in the same order, land. A run resuming the ramp
    where its stage of sixteen begins trains on, the window its checkpoint held, of eight-record
    slots, replaced; one handed a window part filled with batches of thirty-two is refused."""
    records, seed, layout = 4096, 7, Layout(min_shard=1, tolerance=1.0)
    ramp = Ramp(start=8, increment=8, samples=16)

    def trainer(checkpoints: Checkpoints | None = None, accumulation: int = 2) -> Trainer:
        return Trainer(Composite(), optax.sgd(0.5), key=jax.random.key(0), accumulation=accumulation,
                       layout=layout, checkpoints=checkpoints)

    state = trainer().fit(ramped(indexed_data(records, 16, seed), ramp), steps=4, log_every=100)
    stitched = [trainer(Checkpoints(str(tmp_path / "stitched"), keep=8), accumulation=1).fit(
        stitched_data(records, batch, seed), steps=stage + 1, log_every=100, checkpoint_every=None)
        for stage, batch in enumerate([16, 32])][-1]
    assert int(state.step) == 4 and int(state.updates) == int(stitched.updates) == 2
    pooled, batched = parameters(state), parameters(stitched)
    assert set(pooled) == set(batched)
    for name, value in batched.items():
        np.testing.assert_allclose(pooled[name], value, rtol=1e-5, atol=1e-6, err_msg=name)

    directory = str(tmp_path / "ramp")
    staged = trainer(Checkpoints(directory, keep=8)).fit(ramped(indexed_data(256, 16), ramp), steps=2,
                                                         log_every=100)
    assert staged.accumulation is not None and jax.tree.leaves(staged.accumulation.batches)[0].shape[1] == 8
    resumed = trainer(Checkpoints(directory, keep=8)).fit(ramped(indexed_data(256, 16), ramp), steps=4,
                                                          log_every=100)
    assert (int(resumed.step), int(resumed.updates)) == (4, 2)
    part_filled = trainer().fit(indexed_data(256, 32), steps=1, log_every=100)
    with pytest.raises(ValueError):
        trainer().fit(ramped(indexed_data(256, 16), ramp), steps=4, log_every=100, state=part_filled)


def test_a_ramp_stage_due_inside_a_window_that_replays_its_batches_begins_at_the_next_update():
    """A stage of one step would begin inside a window of two, whose replay slots hold one shape of
    batch: it begins at the next update instead, and the run trains through both stages."""
    ramp = Ramp(start=8, increment=8, samples=8)
    assert ramp.stages(16, window=2) == (Stage(batch=8, records=0), Stage(batch=16, records=16))
    trainer = Trainer(Composite(), optax.sgd(0.5), key=jax.random.key(0), accumulation=2,
                      layout=Layout(min_shard=1, tolerance=1.0), checkpoints=None)
    state = trainer.fit(ramped(indexed_data(256, 16), ramp), steps=4, log_every=100)
    assert (int(state.step), int(state.updates)) == (4, 2)


@pytest.mark.mesh
def test_a_ramp_stage_the_mesh_cannot_hold_is_refused_before_the_run_reads(tmp_path):
    """Eight devices hold whole rows, so a stage of four records has nowhere
    to put them; the run says so before it trains rather than an hour in,
    when the stage arrives."""
    trainer = Trainer(Regression(), optax.sgd(0.5), key=jax.random.key(0),
                      layout=Layout(min_shard=1, tolerance=1.0))

    with pytest.raises(ValueError):
        trainer.fit(ramped(indexed_data(256, 12), Ramp(start=4, increment=4, samples=16)),
                    steps=2, log_every=100)


@pytest.mark.mesh
def test_a_ramp_stage_the_pipeline_cannot_cut_into_microbatches_is_refused():
    """A pipelined step cuts each device's rows into microbatches, which the
    decoder checks when it traces, so a stage a later compile would refuse
    is refused before the run reads: two stages beside four row shards and
    six microbatches hold batches of twenty-four, six rows a device, and
    stages of eight and sixteen are not ones."""
    trainer = Trainer(Regression(), optax.sgd(0.5), key=jax.random.key(0),
                      mesh=MeshSpec(stage=2), microbatches=6,
                      layout=Layout(min_shard=1, tolerance=1.0))

    with pytest.raises(ValueError):
        trainer.fit(ramped(indexed_data(256, 24), Ramp(start=8, increment=8, samples=16)),
                    steps=2, log_every=100)


@pytest.mark.mesh
def test_a_batch_the_pipeline_cannot_cut_into_microbatches_is_refused_before_it_compiles():
    """Eight rows over data2 x fsdp2 hold two a device, which four
    microbatches do not divide: a device that holds none of a microbatch's
    rows would compute another's again. The run is refused as it starts,
    before anything compiles, naming a batch and a microbatch count that
    fit."""
    from dew.mesh import LayoutRefused

    trainer = Trainer(Regression(), optax.sgd(0.5), key=jax.random.key(0),
                      mesh=MeshSpec(fsdp=2, stage=2), microbatches=4,
                      layout=Layout(min_shard=1, tolerance=1.0))

    with pytest.raises(LayoutRefused):
        trainer.fit(indexed_data(256, 8), steps=2, log_every=100)
    assert trainer.executable is None


@pytest.mark.mesh
def test_a_batch_the_row_shards_cannot_split_is_refused_before_it_is_placed():
    """Six rows over data 8 leave two devices without a row. The run is
    refused before a batch is placed or anything compiles, naming the row
    shards and a batch that splits over them, where placing the batch raised
    jax's own error about a sharding that does not divide a dimension."""
    trainer = Trainer(Regression(), optax.sgd(0.5), key=jax.random.key(0),
                      layout=Layout(min_shard=1, tolerance=1.0))

    with pytest.raises(ValueError):
        trainer.fit(indexed_data(256, 6), steps=2, log_every=100)
    assert trainer.executable is None


def test_the_log_tick_reports_the_records_a_ramped_interval_read():
    """`samples_per_sec` is the records the interval read over its wall time,
    summed per step, so a ramped interval is not reported as if every step
    had read the batch the ramp ends at."""
    logged = []

    class Tracker:
        def log(self, scalars, step):
            logged.append((step, dict(scalars)))

        def artifact(self, value, step):
            pass

        def close(self):
            pass

    trainer = Trainer(
        Regression(), optax.sgd(0.1), key=jax.random.key(0),
        layout=Layout(min_shard=1, tolerance=1.0), tracker=Tracker())
    trainer.fit(ramped(indexed_data(1024, FINAL), RAMP), steps=6, log_every=6)

    step, scalars = logged[0]
    read = sum(ramp_batches(RAMP, FINAL, 6))
    assert step == 6 and read == 4 * 8 + 2 * 16
    assert scalars["train/samples_per_sec"] == pytest.approx(
        read / (scalars["train/step_time_ms"] * 6 / 1000), rel=1e-6)


def test_a_corpus_that_recurs_across_phases_continues_its_order_and_repeats_nothing():
    """The reviewer's three-corpus case at Rigel-like shares: a corpus read
    in phase 0 and again in phases 1 and 2, and alone in phase 3, picks up
    where the earlier phases left it, so no record of it repeats before its
    epoch ends, and each phase still reads its own mixture at its own shares."""
    from dew.data.pipeline import Phases
    corpora = {name: Indexed(tag, 1000) for name, tag in (("web", 1), ("code", 2), ("math", 3))}
    phases = [({"web": 71, "code": 20, "math": 9}, 10), ({"web": 15, "code": 85}, 20),
              ({"web": 57, "code": 18, "math": 25}, 30), ({"code": 1.0}, None)]
    pipelines = []
    for weights, _ in phases:
        children = tuple(Pipeline(Memory(corpora[name]), name=name).shuffle(0).repeat() for name in weights)
        pipelines.append(children[0] if len(children) == 1 else
                         mix(children, weights=tuple(weights.values())))
    dataset = compile(Pipeline(Phases(tuple(pipelines), tuple(until for _, until in phases))).batch(8),
                      read="memory", seed=0, loading=READ)
    steps = taken(dataset.train(DataPartition()), 40)
    by_phase = [[value for step in steps[start:end] for value in step]
                for start, end in ((0, 10), (10, 20), (20, 30), (30, 40))]
    assert all(value // 1000 == 2 for value in by_phase[3])
    for tag in (1, 2, 3):
        read = [value for phase in by_phase for value in phase if value // 1000 == tag]
        assert len(read) == len(set(read)), f"corpus {tag} repeated a record within its epoch"
    assert abs(sum(value // 1000 == 2 for value in by_phase[1]) - 68) <= 1


def test_corpora_recurring_across_phases_continue_where_grains_weighted_mix_left_each():
    """Shares of 1, 10 and 15, whose scaling to whole proportions Grain truncates: after two steps
    of eight records the mix has read 1, 6 and 9 records of its corpora, so the next phase starts
    each where that left it, and across the switch every corpus reads its own order, nothing
    repeated and nothing skipped."""
    from dew.data.pipeline import Phases

    def corpus(tag):
        return Pipeline(Memory(Indexed(tag, 200)), name=f"corpus-{tag}").shuffle(tag).repeat()

    def mixed():
        return mix(tuple(corpus(tag) for tag in (1, 2, 3)), weights=(1, 10, 15))

    dataset = compile(Pipeline(Phases((mixed(), mixed()), (2, None))).batch(8), read="memory", seed=0,
                      loading=READ)
    read = [value for step in taken(dataset.train(DataPartition()), 6) for value in step]
    for tag in (1, 2, 3):
        alone = compile(corpus(tag).batch(1), read="memory", seed=0, loading=READ)
        mine = [value for value in read if value // 1000 == tag]
        assert mine == [step[0] for step in taken(alone.train(DataPartition()), len(mine))]


@pytest.mark.parametrize("weights, lengths", [((1, 10, 15), (300, 40, 900)), ((0.3, 0.7), (300, 300)),
                                              ((71, 20, 9), (50, 800, 7)), ((1.0,), (12,))])
def test_dews_weighted_selection_reads_the_records_grains_mix_reads(weights, lengths):
    """The selection Dew mixes indexed corpora by and counts a phase's prefix by reads each position
    of the mixture from the corpus and record Grain's own weighted mix reads there, until the corpus
    Grain's mix stops at would read a record twice; a corpus of no records is refused, as Grain
    refuses it."""
    from dew.data.mixing import ProportionalSelection

    parents = [pygrain.MapDataset.range(1000 * (index + 1), 1000 * (index + 1) + length)
               for index, length in enumerate(lengths)]
    ours = pygrain.MapDataset.select_from_datasets(parents, ProportionalSelection(lengths, weights))
    grains = pygrain.MapDataset.mix(parents, weights)
    assert len(ours) == len(grains) and list(ours) == list(grains)
    with pytest.raises(ValueError):
        ProportionalSelection((*lengths[:-1], 0), weights)


@pytest.mark.parametrize("weights, sizes", [((1, 10, 15), (50, 200, 400)), ((1.0, 1.0), (300, 40)),
                                            ((0.3, 0.7), (1000, 1000))])
def test_an_epoch_of_repeated_corpora_ends_where_grains_mix_has_read_every_one_once(weights, sizes):
    """A mix of repeated corpora counts an epoch as the positions of Grain's own mix by which every
    corpus has been read once, its slowest at its share: no fewer, no more."""
    corpora = [Pipeline(Memory(Indexed(tag, size)), name=f"corpus-{tag}").repeat()
               for tag, size in enumerate(sizes, start=1)]
    data = compile(mix(corpora, weights=weights).batch(1), read="memory", loading=READ)
    grains = grains_corpora(weights)
    reads, covered = [0] * len(sizes), 0
    while any(read < size for read, size in zip(reads, sizes, strict=True)):
        reads[grains[covered]] += 1
        covered += 1
    assert data.records == covered


def test_an_epoch_of_a_mix_of_corpora_written_without_repeats_reads_every_corpus_whole():
    """A run's mix of corpora written without repeats reads, in an epoch's steps, every record of the
    corpus it ends on once and every record of the others: the slow corpus's tail is read, not
    skipped by the mix starting over at its shortest corpus's end, and the epoch's last step, which
    leads into the next, holds the record that completes it."""
    from dew.data.compile import fitted
    from dew.data.pipeline import ObjectiveData

    corpora = [Pipeline(Memory(Indexed(tag, size)), name=f"corpus-{tag}")
               for tag, size in ((1, 40), (2, 200))]
    data, _ = fitted(mix(corpora, weights=(1, 10)).batch(4).read("memory").workers(0), None,
                     objective=ObjectiveData())
    values = [value for step in taken(data.train(DataPartition()), data.steps_per_epoch) for value in step]
    slow = sorted(value for value in values if value // 1000 == 1)
    assert slow == [1000 + index for index in range(40)]
    assert {value for value in values if value // 1000 == 2} == {2000 + index for index in range(200)}


def test_one_corpus_starts_past_its_offset_as_a_mixed_one_does():
    """`corpora_dataset` owns how a list of corpora streams, the offset a
    phase gives a recurring corpus included, for one corpus as for several."""
    data = compile(Pipeline(Memory(Indexed(1, 40)), name="wiki").shuffle(0).repeat().batch(4),
                   read="memory", seed=0, loading=READ)
    stream = data.train(DataPartition())
    try:
        taken(stream, 2)
        saved = stream.get_state()
        whole = taken(stream, 2)
    finally:
        stream.close()
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(saved)
        later = taken(resumed, 2)
    finally:
        resumed.close()
    assert later == whole


def _plus_one(batch):
    return {**batch, "x": batch["x"] + 1}


def _missing_field(row):
    return {"x": row["absent"]}


@pytest.mark.parametrize("read", ["indexed", "streamed"])
def test_a_mixed_corpus_whose_first_record_fails_raises_its_own_error(read):
    """A mixture reads each corpus's first record before it mixes: one whose step fails there
    raises its own error, a KeyError here, with its own traceback, where it was once rewritten
    into a ValueError naming the corpus."""
    broken = Pipeline(Memory([{"x": np.int32(0)}] * 4, identity="broken")).map(_missing_field)
    plain = Pipeline(Memory([{"x": np.int32(1)}] * 4, identity="plain"))
    stream = compile(mix([broken.repeat(), plain.repeat()]).batch(2), read=read, loading=READ).train(
        DataPartition())
    try:
        with pytest.raises(KeyError):
            next(stream)
    finally:
        stream.close()


def test_a_mixture_refuses_a_batched_member_rather_than_drop_its_stages():
    """A mixture reads its members' records and batches once, after mix: a member batched
    before it, with a step after its batch, is refused, where the mixture once read the
    member's records and the step never ran."""
    first = Memory([{"x": np.int32(0)}], identity="first").batch(1).map(_plus_one)
    second = Memory([{"x": np.int32(0)}], identity="second")
    with pytest.raises(ValueError):
        compile(mix([first, second]).batch(1), read="indexed", loading=READ)


class _DevicePlusOne(DeviceStep):
    def __call__(self, batch):
        return {**batch, "x": batch["x"] + 1}


def test_phases_that_end_in_other_device_steps_are_refused():
    """A run applies one set of device steps to every batch it reads, so a phase whose own
    device steps differ from the first phase's is refused, where it once ran the first's."""
    from dew.data.pipeline import Phases

    def phase(name: str) -> Pipeline:
        return Pipeline(Memory([{"x": np.int32(0)}] * 4, identity=name)).repeat().batch(1)

    with pytest.raises(ValueError):
        compile(Pipeline(Phases((phase("plain"), phase("stepped").map(_DevicePlusOne())), (1, None))),
                read="memory", loading=READ)


def test_a_token_mixture_of_published_generations_weighs_the_rows_and_targets_each_states(tmp_path):
    """Published generations mix by tokens from the rows and valid targets each states, where
    the weighting once looked for a windows count no generation states."""
    from dew.data.cache import Cache
    from dew.data.steps.text import CountTargets
    from dew.data.text import Tokenize

    def published(name: str, documents: int) -> Pipeline:
        texts = Memory([{"text": f"{name} {index:03d}"[-8:]} for index in range(documents)], identity=name)
        generation = publish(Pipeline(texts).map(Tokenize()).filter_map(CountTargets()),
                             Cache(str(tmp_path / name)), read="indexed", loading=READ)
        return Pipeline(generation.source(), name=name)

    data = compile(mix([published("short", 4), published("long", 8)], by="tokens").repeat().batch(2),
                   read="indexed", loading=READ)
    stream = data.train(DataPartition())
    try:
        assert len(list(itertools.islice(stream, 4))) == 4
    finally:
        stream.close()


def _token_rows(name: str, tag: int, tokens: int, count: int = 200) -> Pipeline:
    """`count` rows of `tokens` tokens each, so `tokens - 1` targets a row, whose ids name the member."""
    from dew.data.steps.text import CountTargets

    rows = [{"text": np.full(tokens, 3, np.int32), "id": np.int32(tag * 1000 + index)}
            for index in range(count)]
    return Pipeline(Memory(rows), name=name).filter_map(CountTargets())


def test_a_token_mixture_recurring_in_a_later_phase_continues_each_member_where_it_stopped():
    """Members weighted alike by tokens, one target a row against nine, read nine rows of the
    first for each of the second; a later phase that mixes them again reads each member on from
    the rows the first phase took, so no row repeats and none is skipped."""
    from dew.data.pipeline import Phases

    def members():
        return mix([_token_rows("short", 1, 2).repeat(), _token_rows("long", 2, 10).repeat()], by="tokens")

    data = compile(Pipeline(Phases((members(), members()), (10, None))).batch(1), read="memory",
                   loading=READ)
    read = [int(batch["id"][0]) for batch in itertools.islice(data.train(DataPartition()), 40)]
    assert sum(value // 1000 == 1 for value in read[:10]) == 9
    for tag in (1, 2):
        member = [value % 1000 for value in read if value // 1000 == tag]
        assert member == list(range(len(member)))


@pytest.mark.parametrize(("by", "read"), [("tokens", "indexed"), ("tokens", "streamed"),
                                          ("records", "streamed")])
def test_a_finite_mixture_resumed_after_a_member_ends_reads_what_the_uninterrupted_one_reads(by, read):
    """A mixture of finite members resumed at any step reads the rest of what the run that was
    never stopped reads, the row a member had already started included, where a resume once
    advanced every member again and ended on one that was done."""
    first, second = _token_rows("a", 1, 3, count=1), _token_rows("b", 2, 8, count=3)
    chain = mix([first.repeat(1), second.repeat(1)], weights=[1, 3], by=by).batch(1)
    data = compile(chain, read=read, loading=READ)

    def ids(stream) -> list[int]:
        return [int(batch["id"][0]) for batch in stream]

    whole = data.train(DataPartition())
    try:
        expected = ids(whole)
    finally:
        whole.close()
    assert len(expected) > 1
    for step in range(1, len(expected)):
        stopped = data.train(DataPartition())
        try:
            assert ids(itertools.islice(stopped, step)) == expected[:step]
            state = stopped.get_state()
        finally:
            stopped.close()
        resumed = data.train(DataPartition())
        try:
            resumed.set_state(state)
            assert ids(resumed) == expected[step:]
        finally:
            resumed.close()


