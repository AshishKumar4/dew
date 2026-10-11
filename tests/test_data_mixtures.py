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

import dataclasses
import itertools
import json
from pathlib import Path

import jax
import numpy as np
import optax
import pytest
from flax import linen as nn
from token_support import write_token_corpus

from dew.data import Corpus, DataPartition, DataPhase, Files, Loading, Memory, Pipeline, Ramp, Text, compile
from dew.data.dataset import (
    CAPTION,
    Dataset,
    LengthGroups,
    mixed_records,
    mixed_stream,
    mixture,
    ramped,
    train_stream,
)
from dew.data.recipes import load
from dew.data.steps.image import EncodeCaptions
from dew.objectives.base import Aux, Objective
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
    which is how `_batches` slices the order. `build` opens the stream a
    share reads."""
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

def two_corpora(weights=(0.75, 0.25), records=(40, 8)) -> list[Corpus]:
    return [Corpus("wiki", Indexed(1, records[0]), weights[0]),
            Corpus("code", Indexed(2, records[1]), weights[1])]


def mixed(corpora, batch=4, seed=0):
    return mixed_stream(corpora, [], batch=batch, seed=seed, loading=READ)


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
    assert taken(mixed(two_corpora(), seed=1)(DataPartition()), 5) != first


def test_a_mixture_resumes_on_the_batch_after_the_one_it_saved():
    """A kill at step k and a restart reads batches k + 1 on, records and
    corpora alike, out of one saved record count."""
    stream = mixed(two_corpora())(DataPartition())
    taken(stream, 3)
    state = stream.get_state()
    rest = taken(stream, 4)

    resumed = mixed(two_corpora())(DataPartition())
    resumed.set_state(state)

    assert taken(resumed, 4) == rest
    assert json.loads(state)["dew_global_position"]["records"] == 12


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


def test_a_mixtures_pass_is_the_records_every_corpus_has_been_read_in():
    """One pass over a mixture has to be a pass over its largest corpus:
    forty records at three quarters of a step take fifty-four. Corpora of
    equal size at equal weights come to the records of both, which is what
    a pass counts everywhere else."""
    assert mixed_records(two_corpora()) == 54
    assert mixed_records(two_corpora(weights=(0.5, 0.5), records=(8, 8))) == 16
    assert mixed_records([Corpus("a", Indexed(1, 10), 3), Corpus("b", Indexed(2, 30), 9)]) == 40


def length(record) -> int:
    """A record's length for `LengthGroups`, read off its id."""
    return int(record["id"]) % 7


def grouped(corpora, batch=4, window=3, cut=None):
    return mixed_stream(corpora, [], batch=batch, seed=0, loading=READ, groups=LengthGroups(length, window),
                        cut=cut)


def test_length_groups_read_each_window_as_batches_of_like_length():
    """Every window of three batches holds the records the plain order puts
    there, sorted into batches whose lengths do not interleave, and the
    batches come in an order of the window's own."""
    plain = taken(mixed(two_corpora())(DataPartition()), 6)
    read = taken(grouped(two_corpora())(DataPartition()), 6)
    for first in (0, 3):
        theirs, ours = plain[first:first + 3], read[first:first + 3]
        assert sorted(itertools.chain(*theirs)) == sorted(itertools.chain(*ours))
        spans = sorted((min(record % 7 for record in batch), max(record % 7 for record in batch))
                       for batch in ours)
        assert all(below[1] <= above[0] for below, above in itertools.pairwise(spans))
    assert read != plain
    assert taken(grouped(two_corpora())(DataPartition()), 6) == read


def test_a_grouped_stream_resumes_mid_window_and_refuses_another_grouping():
    stream = grouped(two_corpora())(DataPartition())
    taken(stream, 2)
    state = stream.get_state()
    rest = taken(stream, 4)
    resumed = grouped(two_corpora())(DataPartition())
    resumed.set_state(state)
    assert taken(resumed, 4) == rest
    with pytest.raises(ValueError, match="grouped by length"):
        mixed(two_corpora())(DataPartition()).set_state(state)


def test_a_cut_stream_maps_every_batch_and_is_read_by_one_process():
    def halved(batch):
        return {"id": batch["id"][:2]}

    stream = grouped(two_corpora(), cut=halved)
    assert [len(batch) for batch in taken(stream(DataPartition()), 3)] == [2, 2, 2]
    with pytest.raises(ValueError, match="one process"):
        stream(DataPartition(0, 2))


def test_a_validation_pass_over_a_mixture_reads_each_record_at_most_once():
    """A mixed pass keeps the weights and each split's own order, and stops
    before any corpus comes round, so a score is a score of the same records
    every time."""
    ordered = mixture(two_corpora(), None)

    read = [ordered[index]["id"].item() for index in range(len(ordered))]

    assert len(read) == 32, "a pass runs until the first corpus would repeat"
    assert len(set(read)) == len(read), "a record was scored twice"
    assert read[:8] == [1000, 1001, 1002, 2000, 1003, 1004, 1005, 2001]


@pytest.mark.parametrize("corpora, message", [
    ([Corpus("only", Indexed(1, 8), 1.0)], "two or more corpora"),
    (two_corpora(weights=(1.0, 0.0)), "positive share"),
    (two_corpora(weights=(1.0, -1.0)), "positive share"),
])
def test_a_mixture_that_is_not_one_is_refused(corpora, message):
    with pytest.raises(ValueError, match=message):
        mixture(corpora, 0)


# --------------------------------------------------------------------------
# The mixture through dew.data.load
# --------------------------------------------------------------------------

datasets = pytest.importorskip("datasets", reason="needs the streaming extra")


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



@dataclasses.dataclass(frozen=True)
class RowMix:
    sources: dict[str, float]
    held: bool = False

    def pipelines(self):
        parents = tuple(dataclasses.replace(
            Pipeline(Files(str(Path(root) / "rows.jsonl"), format="jsonl")).map(
                lambda row: {"id": np.int32(row["index"])}), name=root)
            for root in sorted(self.sources))
        weights = tuple(self.sources[root] for root in sorted(self.sources))
        mixed = parents[0] if len(parents) == 1 else parents[0].mix(*parents[1:], weights=weights)
        return mixed, mixed if self.held else None, ()


def test_load_reads_a_weighted_mixture_of_two_datasets(two_splits):
    big, small = two_splits
    data = load(RowMix({big: 0.75, small: 0.25}), batch=4, read="memory", loading=READ)
    assert data.records == 54 and data.batch == 4 and data.steps_per_epoch == 13
    stream = data.train(DataPartition())
    try:
        batches = taken(stream, 6)
    finally:
        stream.close()
    assert all(sum(value >= 2000 for value in step) == 1 for step in batches)


def test_a_mixed_validation_pass_ends_before_any_source_repeats(two_splits):
    big, small = two_splits
    data = load(RowMix({big: 0.75, small: 0.25}, held=True), batch=4, read="memory", loading=READ)
    assert data.val is not None
    first, second = data.val(DataPartition()), data.val(DataPartition())
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
    first = load(RowMix({big: 0.75, small: 0.25}), batch=4, read="memory", loading=READ)
    other = load(RowMix({small: 0.25, big: 0.75}), batch=4, read="memory", loading=READ)
    left, right = first.train(DataPartition()), other.train(DataPartition())
    try:
        assert taken(left, 4) == taken(right, 4)
        assert left.get_state() == right.get_state()
        assert taken(left, 3) == taken(right, 3)
    finally:
        left.close()
        right.close()


def test_a_single_source_is_not_wrapped_in_a_mixture(two_splits):
    big, _ = two_splits
    data = load(RowMix({big: 1.0}), batch=4, read="memory", loading=READ)
    assert data.records == 40
    stream = data.train(DataPartition())
    try:
        assert sorted(value for batch in taken(stream, 10) for value in batch) == list(range(1000, 1040))
    finally:
        stream.close()


def test_sources_from_different_stores_mix_before_the_share_cut(two_splits):
    big, _ = two_splits
    files = Pipeline(Files(str(Path(big) / "rows.jsonl"), format="jsonl")).map(
        lambda row: {"id": np.int32(row["index"])})
    memory = Pipeline(Memory([{"id": np.int32(2000 + index)} for index in range(8)]))
    data = compile(files.shuffle(0).repeat().mix(memory.shuffle(0).repeat(), weights=(3, 1)).batch(4),
                   read="memory", loading=READ)
    stream = data.train(DataPartition())
    try:
        assert all(sum(value >= 2000 for value in batch) == 1 for batch in taken(stream, 8))
    finally:
        stream.close()


def test_a_streamed_mixture_keeps_accepted_row_shares(two_splits):
    big, small = two_splits
    data = load(RowMix({big: 0.75, small: 0.25}), batch=4, read="streamed", loading=READ)
    stream = data.train(DataPartition())
    try:
        assert all(sum(value >= 2000 for value in batch) == 1 for batch in taken(stream, 8))
    finally:
        stream.close()


def document_dir(root: Path, documents, eos: int = 0) -> str:
    """A token directory whose stream is `documents`, each closed by eos."""
    stream = np.concatenate([np.asarray([*list(document), eos], np.uint16)
                             for document in documents])
    return str(write_token_corpus(root, stream, stream, vocab_size=256, eos_id=eos))


def packed_corpus(root: Path, documents: int = 40, length: int = 3) -> str:
    # Document i is the value i + 1 repeated, so a window's values name the
    # documents packed into it and the padding names none.
    return document_dir(root, [[index + 1] * length for index in range(documents)])


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
    corpus = packed_corpus(tmp_path / "corpus")
    open_stream = load(Text(source=corpus, pack="first-fit", seq_len=8, packing_bins=2), batch=4,
        read="memory", loading=READ).train

    whole = pooled(open_stream, 1, 5, rows=windows_of)

    assert whole[0], "the loader produced no windows"
    assert pooled(open_stream, 2, 5, rows=windows_of) == whole
    assert pooled(open_stream, 4, 5, rows=windows_of) == whole


def test_a_packed_position_written_by_two_processes_resumes_on_one(tmp_path):
    """The packed stream's position is a global window count, so the two
    processes that stopped at step two hand one process, or four, the steps
    the run would have taken next."""
    corpus = packed_corpus(tmp_path / "corpus")
    open_stream = load(Text(source=corpus, pack="first-fit", seq_len=8, packing_bins=2), batch=4,
        read="memory", loading=READ).train
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
    corpus = packed_corpus(tmp_path / "corpus", documents=24)
    pass_over = load(Text(source=corpus, pack="first-fit", seq_len=8, packing_bins=2), batch=4,
        read="memory", loading=READ).val
    assert pass_over is not None
    whole = [row for batch in pass_over(DataPartition()) for row in windows_of([batch])[0]]

    for processes in (2, 4):
        shards = [[row for batch in pass_over(DataPartition(index, processes))
                   for row in windows_of([batch])[0]] for index in range(processes)]
        assert sum(len(shard) for shard in shards) == len(whole)
        assert sorted(row for shard in shards for row in shard) == sorted(whole)


def weighted_packed(tmp_path, weights, **overrides):
    """Two corpora told apart by their values: 1..40 in the first, 101..124
    in the second, each document one value repeated three times."""
    first = document_dir(tmp_path / "first", [[index + 1] * 3 for index in range(40)])
    second = document_dir(tmp_path / "second", [[index + 101] * 3 for index in range(24)])
    spec = Text(source={first: weights[0], second: weights[1]}, pack="first-fit", seq_len=8, packing_bins=2)
    return spec, first, second


def corpus_of(window: tuple[int, ...]) -> int:
    values = {value for value in window if value}
    corpora = {0 if value < 100 else 1 for value in values}
    assert len(corpora) == 1, f"window {window} mixes corpora"
    return corpora.pop()


def test_weighted_packed_corpora_fill_every_step_at_their_shares(tmp_path):
    """MaxText's weighted `grain_train_files`, over packed windows: every
    window is one corpus's documents, and every prefix of the stream holds
    each corpus's share of its windows to within one, the contract grain's
    `MapDataset.mix` keeps. At 3:1 a batch of four is three and one."""
    spec, _, _ = weighted_packed(tmp_path, (3.0, 1.0))
    rows = [row for batch in packed_rows(load(spec, batch=4, read="memory",
        loading=READ).train(DataPartition()), 12) for row in batch]
    drawn = [corpus_of(row) for row in rows]
    for prefix in range(1, len(drawn) + 1):
        assert abs(drawn[:prefix].count(1) - prefix / 4) <= 1
    assert all(drawn[step * 4:step * 4 + 4].count(1) == 1 for step in range(12))


def test_weighted_packed_corpora_resume_on_another_process_count(tmp_path):
    spec, _, _ = weighted_packed(tmp_path, (1.0, 1.0))
    open_stream = load(spec, batch=4, read="memory", loading=READ).train
    whole = pooled(open_stream, 1, 6, rows=windows_of)

    stopped = open_stream(DataPartition(0, 2))
    packed_rows(stopped, 2)
    state = stopped.get_state()
    for processes in (1, 4):
        assert pooled(open_stream, processes, 4, state, rows=windows_of) == whole[2:]


def test_a_weighted_packed_pass_counts_and_scores_every_corpus(tmp_path):
    """The pass is the windows in which every corpus has been read once, and
    the validation pass reads each held-out window at most once."""
    spec, first, second = weighted_packed(tmp_path, (1.0, 1.0))
    loaded = load(spec, batch=2, read="memory", loading=READ)
    alone = [load(Text(source=path, pack="first-fit", seq_len=8, packing_bins=2), batch=2, read="memory",
        loading=READ).records for path in (first, second)]
    assert loaded.records == 2 * max(alone)
    scored = [row for batch in loaded.val(DataPartition()) for row in windows_of([batch])[0]]
    assert len(scored) == len(set(scored))
    assert {corpus_of(row) for row in scored} == {0, 1}


def test_corpora_from_different_tokenizers_are_not_mixed(tmp_path):
    spec, _, second = weighted_packed(tmp_path, (1.0, 1.0))
    meta = json.loads((Path(second) / "meta.json").read_text())
    (Path(second) / "meta.json").write_text(json.dumps({**meta, "tokenizer": "gpt2"}))
    with pytest.raises(ValueError, match="mixed corpora must share the tokenizer contract"):
        load(spec, batch=4, read="memory", loading=READ)


# --------------------------------------------------------------------------
# Mixture phases
# --------------------------------------------------------------------------

def phased_packed(first: str, second: str, *phases: tuple[object, int | None]) -> Text:
    return Text(source="", pack="first-fit", phases=tuple(DataPhase(path, until) for path,
        until in phases), seq_len=8, packing_bins=2)


def test_phases_switch_the_mixture_at_a_step(tmp_path):
    """Rigel's curriculum: three steps of the first corpus alone, then both
    at equal shares, every step of the second phase half of each."""
    _, first, second = weighted_packed(tmp_path, (1.0, 1.0))
    both = {first: 1.0, second: 1.0}
    spec = phased_packed(first, second, (first, 3), (both, None))

    steps = packed_rows(load(spec, batch=4, read="memory", loading=READ).train(DataPartition()), 7)

    assert all(corpus_of(row) == 0 for step in steps[:3] for row in step)
    assert all([corpus_of(row) for row in step].count(1) == 2 for step in steps[3:])


def test_mixed_counts_is_grains_own_selection():
    """The closed form counts what `MapDataset.mix` hands out, prefix by prefix."""
    from dew.data.dataset import mixed_counts
    corpora = [Corpus(name, Indexed(tag, 50), weight)
               for name, tag, weight in (("a", 1, 71), ("b", 2, 20), ("c", 3, 9))]
    drawn = [int(mixture(corpora, 0)[index]["id"]) // 1000 for index in range(400)]
    for prefix in (1, 7, 100, 399, 400):
        assert mixed_counts(corpora, prefix) == tuple(drawn[:prefix].count(tag) for tag in (1, 2, 3))


def test_a_run_of_one_mixture_resumes_into_phases_that_begin_with_it(tmp_path):
    """Resuming onto a changed mixture is refused, but resuming onto a phase
    list whose first phase is the run's mixture, with a boundary it has not
    passed, is the same run with a switch ahead of it, at any process count."""
    _spec, first, second = weighted_packed(tmp_path, (1.0, 1.0))
    both = {first: 1.0, second: 1.0}
    plain = Text(source=first, pack="first-fit", seq_len=8, packing_bins=2)
    stopped = load(plain, batch=4, read="memory", loading=READ).train(DataPartition())
    packed_rows(stopped, 2)
    state = stopped.get_state()
    phases = phased_packed(first, second, (first, 4), (both, None))
    open_stream = load(phases, batch=4, read="memory", loading=READ).train
    whole = pooled(open_stream, 1, 7, rows=windows_of)

    for processes in (1, 2):
        assert pooled(open_stream, processes, 5, state, rows=windows_of) == whole[2:]
    with pytest.raises(ValueError, match="phase 0 reads something else"):
        load(phased_packed(first, second, (both, 4), (first, None)), batch=4, read="memory",
            loading=READ).train(DataPartition()).set_state(
            state
        )
    with pytest.raises(ValueError, match="past this run's end of phase 0"):
        load(phased_packed(first, second, (first, 1), (both, None)), batch=4, read="memory",
            loading=READ).train(DataPartition()).set_state(
            state
        )


def test_a_phased_run_resumes_past_a_switch_and_refuses_a_changed_history(tmp_path):
    """Two steps into the second phase, the position names the first phase
    and where it ended. The same list, or one with a phase appended, resumes
    on the next step; a list that moved the finished boundary or changed the
    finished mixture, or a run of one order, is refused."""
    _, first, second = weighted_packed(tmp_path, (1.0, 1.0))
    both = {first: 1.0, second: 1.0}
    spec = phased_packed(first, second, (first, 3), (both, None))
    stream = load(spec, batch=4, read="memory", loading=READ).train(DataPartition())
    packed_rows(stream, 5)
    state = stream.get_state()
    rest = packed_rows(stream, 3)

    saved = json.loads(state)["dew_global_position"]
    assert saved["records"] == 20 and saved["completed"][0][1] == 12
    for resumed_spec in (spec, phased_packed(first, second, (first, 3), (both, 9), (second, None))):
        resumed = load(resumed_spec, batch=4, read="memory", loading=READ).train(DataPartition())
        resumed.set_state(state)
        assert packed_rows(resumed, 3) == rest
    for changed in (phased_packed(first, second, (first, 2), (both, None)),
                    phased_packed(first, second, (second, 3), (both, None))):
        with pytest.raises(ValueError, match="saved data plan order changed"):
            load(changed, batch=4, read="memory", loading=READ).train(DataPartition()).set_state(state)
    plain = Text(source=both, pack="first-fit", seq_len=8, packing_bins=2)
    with pytest.raises(ValueError, match=r"phase|stable identity"):
        load(plain, batch=4, read="memory", loading=READ).train(DataPartition()).set_state(state)


def test_phases_are_a_list_of_ends_and_refuse_a_ramp(tmp_path):
    _, first, second = weighted_packed(tmp_path, (1.0, 1.0))
    with pytest.raises(ValueError, match="phase step boundaries must increase"):
        load(phased_packed(first, second, (first, 3), (second, 3), (first, None)), batch=4,
            read="memory", loading=READ).train(
            DataPartition()
        )
    with pytest.raises(ValueError, match="last boundary is None"):
        load(phased_packed(first, second, (first, 3), (second, 5)), batch=4, read="memory",
            loading=READ).train(DataPartition())
    spec = phased_packed(first, second, (first, 3), (second, None))
    with pytest.raises(TypeError, match="ramp a run of one order"):
        ramped(load(spec, batch=4, read="memory", loading=READ), Ramp(start=2, increment=2,
            samples=8)).train(DataPartition())


# --------------------------------------------------------------------------
# The batch ramp: MaxText's schedule
# --------------------------------------------------------------------------

MAXTEXT_RAMP = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "ramp" / "maxtext.json").read_text())


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


@pytest.mark.parametrize("case", MAXTEXT_RAMP["cases"], ids=lambda case: str(case["config"]))
def test_the_ramp_reads_the_batch_maxtext_reads_at_every_step(case):
    """MaxText's own `RampupBatchManager` at a pinned commit, as its
    `RampUpDataLoader` drives it (tools/ramp_reference.py): the batch read
    at each of 128 steps, through every stage and past the ramp's end, and
    of a run resumed inside the second stage, which MaxText rebuilds by
    replaying its updates and Dew reads off the records already read. Dew
    counts the global batch where MaxText counts a batch per device, and
    computes a stage's length as one integer ratio where MaxText divides
    twice in floating point; the schedule is the same schedule."""
    config = case["config"]
    devices = config["num_target_devices"]
    ramp = Ramp(start=devices * config["per_device_batch_size_start"],
                increment=devices * config["per_device_batch_size_increment"],
                samples=config["global_rampup_samples"])
    final = devices * config["per_device_batch_size"]
    steps, resume = MAXTEXT_RAMP["steps"], case["resume_step"]

    assert ramp_batches(ramp, final, steps) == case["batches"]
    assert ramp_batches(ramp, final, steps - resume, sum(case["batches"][:resume])) == case["resumed"]


def test_the_ramp_ends_on_the_runs_own_batch_and_stays_there():
    stages = Ramp(start=4, increment=2, samples=48).stages(8)

    assert [stage.batch for stage in stages] == [4, 6, 8]
    # Six steps of four and four of six: 48 records over each increment,
    # rounded up to a whole step.
    assert [stage.records for stage in stages] == [0, 24, 48]
    assert ramp_batches(Ramp(start=4, increment=2, samples=48), 8, 12)[-1] == 8


@pytest.mark.parametrize("ramp, final, message", [
    (Ramp(start=4, increment=2, samples=48), 4, "nothing to ramp"),
    (Ramp(start=4, increment=2, samples=48), 3, "nothing to ramp"),
    (Ramp(start=4, increment=3, samples=48), 8, "whole increments"),
    (Ramp(start=0, increment=2, samples=48), 8, "counts records"),
    (Ramp(start=4, increment=2, samples=0), 8, "counts records"),
])
def test_a_schedule_that_cannot_be_run_is_refused(ramp, final, message):
    with pytest.raises(ValueError, match=message):
        ramp.stages(final)


def test_a_ramp_stage_that_does_not_split_into_the_shares_is_refused():
    """A stage of six records a step has no equal share for each of four
    readers; the stream refuses it at the step that reads it."""
    stream = ramped(indexed_data(64, 8), Ramp(start=6, increment=2, samples=48)).train(
        DataPartition(0, 4))
    with pytest.raises(ValueError, match="batch 6 does not split into 4 equal shares"):
        next(stream)
    stream.close()


def test_a_pass_over_the_data_is_more_steps_under_a_ramp():
    """A ramp reads fewer records a step early, so the same records are more
    steps: sixteen at four, then eight at eight, is twenty-four steps where
    a flat batch of eight is twelve."""
    ramp = Ramp(start=4, increment=4, samples=64)
    flat = Dataset(train=lambda partition: iter(()), val=None, records=96, batch=8)

    assert flat.steps_per_epoch == 12
    assert ramped(flat, ramp).steps_per_epoch == 20
    assert ramp.steps_for(64, 8) == 16, "the ramp itself is sixteen steps of four"


# --------------------------------------------------------------------------
# The batch ramp over a real stream
# --------------------------------------------------------------------------

def indexed_data(records: int, batch: int, seed: int = 0) -> Dataset:
    """A dataset whose records say which they are, read through the stream
    every dataset reads through."""
    source = Indexed(1, records)
    return Dataset(train=train_stream(source, [], batch=batch, seed=seed, loading=READ),
                   val=None, records=records, batch=batch)


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


def test_a_ramped_stream_resumes_on_the_records_it_had_not_read():
    """The rolling buffer holds records the run has not trained on, so they
    are not in the position: a resume reads them, and reads them once."""
    schedule = Ramp(start=2, increment=1, samples=6)
    stream = ramped(indexed_data(64, 4), schedule).train(DataPartition())
    taken(stream, 4)
    state = stream.get_state()
    rest = taken(stream, 6)

    resumed = ramped(indexed_data(64, 4), schedule).train(DataPartition())
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


def test_a_ramped_steps_position_counts_every_process():
    """Each process cuts its own share of a stage's batch, and the position
    is the records all of them handed over: two processes four steps into
    a ramp of two, then four, have read eight records, and the pool's steps
    are the single process's steps with the shares interleaved."""
    schedule = Ramp(start=2, increment=2, samples=8)

    data = Dataset(train=train_stream(Indexed(1, 64), [], batch=4, seed=0, loading=READ),
                   val=None, records=64, batch=4)
    build = ramped(data, schedule).train

    whole = pooled(build, 1, 6)
    stopped = build(DataPartition(1, 2))
    taken(stopped, 4)

    assert [len(step) for step in whole] == [2, 2, 2, 2, 4, 4]
    assert json.loads(stopped.get_state())["dew_global_position"]["records"] == 8
    assert pooled(build, 2, 6) == whole
    assert pooled(build, 2, 2, stopped.get_state()) == whole[4:]


def test_a_position_no_step_of_this_ramp_ends_on_is_refused():
    stream = ramped(indexed_data(64, 4), Ramp(start=2, increment=1, samples=6)).train(DataPartition())
    taken(stream, 3)
    state = stream.get_state()

    # Seven records in, which is a step boundary of the ramp that wrote it
    # (2 + 2 + 3) and lands inside a step of the one that reads it.
    other = ramped(indexed_data(64, 4), Ramp(start=2, increment=2, samples=6)).train(DataPartition())
    with pytest.raises(ValueError, match="ramped differently"):
        other.set_state(state)


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

    def captioned(batch: int) -> Dataset:
        return compile(Pipeline(Memory(Captioned(1, 64))).shuffle(0).repeat().batch(batch).map_batch(
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
    data = Dataset(train=lambda partition: stream(), val=None, records=64, batch=8)
    with pytest.raises(TypeError, match=stream.__name__):
        ramped(data, Ramp(start=4, increment=4, samples=16)).train(DataPartition())


# --------------------------------------------------------------------------
# The batch ramp through the trainer
# --------------------------------------------------------------------------

FEATURES = 3


class Affine(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(2)(x)


class Regression(Objective):
    """Squared error of an affine map on records that differ per index, so
    which records a step read shows up in the parameters."""

    def __init__(self):
        self.model = Affine()

    def init(self, key, variables=None):
        return self.model.init(key, np.zeros((1, FEATURES), np.float32))

    def loss(self, variables, batch, step):
        import jax.numpy as jnp

        features = jnp.stack([jnp.sin(batch["id"].astype(jnp.float32) * (index + 1))
                              for index in range(FEATURES)], axis=-1)
        target = jnp.stack([features[:, 0] * 2, features[:, 1] - 1], axis=-1)
        return jnp.mean((self.model.apply(variables, features) - target) ** 2), Aux({})


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
                    indexed_data(records, batch, seed), steps=sum(lengths[:stage + 1]),
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


def test_a_ramp_and_an_accumulation_window_are_refused_together():
    trainer = Trainer(
        Regression(), optax.sgd(0.5), key=jax.random.key(0), accumulation=2,
        layout=Layout(min_shard=1, tolerance=1.0))

    with pytest.raises(ValueError, match="ramp the batch or accumulate"):
        trainer.fit(ramped(indexed_data(256, 16), Ramp(start=8, increment=8, samples=16)),
                    steps=2, log_every=100)


@pytest.mark.mesh
def test_a_ramp_stage_the_mesh_cannot_hold_is_refused_before_the_run_reads(tmp_path):
    """Eight devices hold whole rows, so a stage of four records has nowhere
    to put them; the run says so before it trains rather than an hour in,
    when the stage arrives."""
    trainer = Trainer(Regression(), optax.sgd(0.5), key=jax.random.key(0),
                      layout=Layout(min_shard=1, tolerance=1.0))

    with pytest.raises(ValueError, match="batch 12 must divide across 8 row shards"):
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
                      mesh=MeshSpec(stage=2, microbatches=6),
                      layout=Layout(min_shard=1, tolerance=1.0))

    with pytest.raises(ValueError, match=r"\[8, 16\].*multiple of 24"):
        trainer.fit(ramped(indexed_data(256, 24), Ramp(start=8, increment=8, samples=16)),
                    steps=2, log_every=100)


@pytest.mark.mesh
def test_a_batch_the_pipeline_cannot_cut_into_microbatches_is_refused_before_it_compiles():
    """Eight rows over data2 x fsdp2 hold two a device, which four
    microbatches do not divide: a device that holds none of a microbatch's
    rows would compute another's again. The run is refused as it starts,
    before anything compiles, naming a batch and a microbatch count that
    fit."""
    from dew.nn.sharding import LayoutRefused

    trainer = Trainer(Regression(), optax.sgd(0.5), key=jax.random.key(0),
                      mesh=MeshSpec(fsdp=2, stage=2, microbatches=4),
                      layout=Layout(min_shard=1, tolerance=1.0))

    with pytest.raises(LayoutRefused, match=r"multiple of 16 rows, 16 the nearest above 8 or "
                                            r"microbatches=2, the most that divide the 2 rows"):
        trainer.fit(indexed_data(256, 8), steps=2, log_every=100)
    assert trainer.executable is None


@pytest.mark.mesh
def test_a_batch_the_row_shards_cannot_split_is_refused_before_it_is_placed():
    """Six rows over data 8 leave two devices without a row. The run is
    refused before a batch is placed or anything compiles, naming the row
    shards and a batch that splits over them, where placing the batch raised
    jax's own error about a sharding that does not divide a dimension."""
    from dew.nn.sharding import LayoutRefused

    trainer = Trainer(Regression(), optax.sgd(0.5), key=jax.random.key(0),
                      layout=Layout(min_shard=1, tolerance=1.0))

    with pytest.raises(ValueError, match="batch 6 must divide across 8 row shards"):
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
    web, code, math = (Corpus(name, Indexed(tag, 1000), 1.0)
                       for name, tag in (("web", 1), ("code", 2), ("math", 3)))
    phases = [([dataclasses.replace(web, weight=71), dataclasses.replace(code, weight=20),
                dataclasses.replace(math, weight=9)], 10),
              ([dataclasses.replace(web, weight=15), dataclasses.replace(code, weight=85)], 20),
              ([dataclasses.replace(web, weight=57), dataclasses.replace(code, weight=18),
                dataclasses.replace(math, weight=25)], 30),
              ([code], None)]
    pipelines = []
    for corpora, _ in phases:
        children = tuple(Pipeline(Memory(corpus.source), name=corpus.name).shuffle(0).repeat()
                         for corpus in corpora)
        pipelines.append(children[0] if len(children) == 1 else
                         children[0].mix(*children[1:], weights=tuple(corpus.weight for corpus in corpora)))
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


def test_a_checkpoint_at_a_phase_boundary_names_the_finished_phase(tmp_path):
    """Saved exactly at the switch, nothing of the next phase has been read:
    a resume may change that phase or extend the finished one."""
    _, first, second = weighted_packed(tmp_path, (1.0, 1.0))
    both = {first: 1.0, second: 1.0}
    stream = load(phased_packed(first, second, (first, 3), (both, None)), batch=4, read="memory",
        loading=READ).train(DataPartition())
    packed_rows(stream, 3)
    state = stream.get_state()
    assert not json.loads(state)["dew_global_position"].get("completed")
    for changed in (phased_packed(first, second, (first, 3), (second, None)),
                    phased_packed(first, second, (first, 5), (both, None))):
        load(changed, batch=4, read="memory", loading=READ).train(DataPartition()).set_state(state)


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
