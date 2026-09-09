"""Weighted corpora, elastic packed windows and the batch ramp.

The three answer the same question: which records does global step k read?
A mixture answers it with grain's proportional interleave over corpora that
are shuffled and cycled before they are mixed, the packed loader answers it
with a packing plan over the whole corpus, and a ramp answers it with a
step-indexed batch cut out of that same order. So what is asserted here is
which records a step holds, and that the answer does not change when a run
is killed and resumed, when it is resumed on a different number of
processes, or when its batch grows.

The process counts are faked here, by pointing `jax.process_index` and
`jax.process_count` at the shard a real pool would read; the loader takes
both from JAX on every open, so a fake count exercises the real slicing.
`tests/test_multiprocess.py` runs the same claims in real processes with a
real checkpoint between them.
"""

import itertools
import json
from pathlib import Path

import jax
import numpy as np
import optax
import pytest
from flax import linen as nn

import dew.data
from dew.data import Corpus, Loading, PackedTokens, Ramp, ramped
from dew.data.dataset import (CAPTION, Dataset, mixed_records, mixed_stream, mixture, tokenized,
                              train_stream)
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


def as_processes(monkeypatch, count: int, index: int) -> None:
    """Read as process `index` of `count`, the way a pool of that size does."""
    monkeypatch.setattr(jax, "process_count", lambda: count)
    monkeypatch.setattr(jax, "process_index", lambda: index)


def pooled(monkeypatch, build, processes: int, batches: int, state: bytes | None = None,
           rows=ids) -> list:
    """The global batches a pool of `processes` reads, each row where the
    process that holds it put it: process p owns rows p, p + n, ... of a
    step, which is how `_batches` slices the order. `build` opens the stream
    that process reads, and takes its own share off the faked count."""
    shards = []
    for index in range(processes):
        as_processes(monkeypatch, processes, index)
        stream = build()
        if state is not None:
            stream.set_state(state)
        shards.append(rows(itertools.islice(stream, batches)))
    return [[row for held in zip(*step) for row in held] for step in zip(*shards)]


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
    stream = mixed(two_corpora())()

    batches = taken(stream, 8)

    for step in batches:
        assert sum(value // 1000 == 2 for value in step) == 1, step
        assert sum(value // 1000 == 1 for value in step) == 3, step


def test_a_small_corpus_of_a_mixture_comes_round_while_a_large_one_is_read_once():
    """Weights are shares of a step, not of the data: the corpora are cycled
    before they are mixed, so eight code records at a quarter of the batch
    are read again long before the forty wiki records are through."""
    stream = mixed(two_corpora())()

    seen = [value for step in taken(stream, 10) for value in step]

    code = [value for value in seen if value // 1000 == 2]
    wiki = [value for value in seen if value // 1000 == 1]
    assert len(code) == 10 and len(set(code)) == 8, "the small corpus never repeated"
    assert len(wiki) == 30 and len(set(wiki)) == 30, "the large corpus repeated a record"


def test_a_mixture_is_the_same_order_from_the_same_seed_and_another_from_another():
    first = taken(mixed(two_corpora())(), 5)

    assert taken(mixed(two_corpora())(), 5) == first
    assert taken(mixed(two_corpora(), seed=1)(), 5) != first


def test_a_mixture_resumes_on_the_batch_after_the_one_it_saved():
    """A kill at step k and a restart reads batches k + 1 on, records and
    corpora alike, out of one saved record count."""
    stream = mixed(two_corpora())()
    taken(stream, 3)
    state = stream.get_state()
    rest = taken(stream, 4)

    resumed = mixed(two_corpora())()
    resumed.set_state(state)

    assert taken(resumed, 4) == rest
    assert json.loads(state)["dew_global_position"]["records"] == 12


def test_a_mixtures_position_resumes_on_another_process_count(monkeypatch):
    """The mixing runs ahead of the shard, so global batch k is the same
    records at any process count and the count two processes saved is where
    one process or four carry on."""
    corpora = two_corpora()
    # The global batch is eight whatever the count; each process reads its
    # own share of it, as `local_batch` divides it.
    build = lambda: mixed(corpora, batch=8 // jax.process_count())()  # noqa: E731
    whole = pooled(monkeypatch, build, 1, 6)

    assert pooled(monkeypatch, build, 2, 6) == whole
    assert pooled(monkeypatch, build, 4, 6) == whole

    as_processes(monkeypatch, 2, 0)
    stopped = build()
    taken(stopped, 2)
    state = stopped.get_state()

    assert json.loads(state)["dew_global_position"]["records"] == 16
    for processes in (1, 4):
        assert pooled(monkeypatch, build, processes, 4, state) == whole[2:]


def test_a_mixtures_pass_is_the_records_every_corpus_has_been_read_in():
    """One pass over a mixture has to be a pass over its largest corpus:
    forty records at three quarters of a step take fifty-four. Corpora of
    equal size at equal weights come to the records of both, which is what
    a pass counts everywhere else."""
    assert mixed_records(two_corpora()) == 54
    assert mixed_records(two_corpora(weights=(0.5, 0.5), records=(8, 8))) == 16
    assert mixed_records([Corpus("a", Indexed(1, 10), 3), Corpus("b", Indexed(2, 30), 9)]) == 40


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


def test_load_reads_a_weighted_mixture_of_two_datasets(two_splits):
    big, small = two_splits
    data = dew.data.load({f"hf/{big}": 0.75, f"hf/{small}": 0.25}, batch=4,
                         preprocess=just_index, loading=READ)

    assert data.records == 54 and data.batch == 4
    assert data.steps_per_epoch == 13
    stream = data.train()
    try:
        batches = taken(stream, 6)
    finally:
        stream.close()
    for step in batches:
        assert sum(value >= 2000 for value in step) == 1, step


def test_load_scores_a_mixture_on_the_same_held_out_records_every_pass(two_splits):
    """A mixed validation pass keeps the weights and stops before the small
    split comes round: eight small records at a quarter of a step are
    thirty-two records, eight batches of four, each record once, and the
    second pass is the first pass again."""
    big, small = two_splits
    data = dew.data.load({f"hf/{big}": 0.75, f"hf/{small}": 0.25}, batch=4,
                         val_split="train", preprocess=just_index, loading=READ)

    assert data.val is not None
    first, second = ids(data.val()), ids(data.val())

    assert len(first) == 8, "a pass runs until the first split would repeat"
    scored = [value for step in first for value in step]
    assert len(set(scored)) == 32, "a record was scored twice"
    assert all(sum(value >= 2000 for value in step) == 1 for step in first), first
    assert second == first


def test_a_mixture_written_in_either_order_is_the_same_run(two_splits):
    """The interleave depends on the order the corpora are mixed in, and the
    loader sorts them by name, so a mapping is not two runs."""
    big, small = two_splits
    first = dew.data.load({f"hf/{big}": 0.75, f"hf/{small}": 0.25}, batch=4,
                          preprocess=just_index, loading=READ)
    other = dew.data.load({f"hf/{small}": 0.25, f"hf/{big}": 0.75}, batch=4,
                          preprocess=just_index, loading=READ)

    left, right = first.train(), other.train()
    try:
        assert left.get_state() == right.get_state()
        assert taken(left, 3) == taken(right, 3)
    finally:
        left.close()
        right.close()


def test_a_single_source_load_is_unchanged_by_the_mixture_route(two_splits):
    """One name reads the corpus it always read, under the same order, so a
    checkpoint written before the mixtures landed still resumes."""
    big, _ = two_splits
    data = dew.data.load(f"hf/{big}", batch=4, preprocess=just_index, loading=READ)
    stream = data.train()
    try:
        order = json.loads(stream.get_state())["dew_global_position"]["order"]
        assert data.records == 40
        assert order.endswith("40 records reshuffled from seed 0")
        assert not order.startswith("mixture"), "one corpus is described as one corpus"
    finally:
        stream.close()


def test_a_mixture_across_providers_is_refused(two_splits):
    big, _ = two_splits
    with pytest.raises(ValueError, match="one provider"):
        dew.data.load({f"hf/{big}": 0.5, "tfds/dew_images": 0.5}, batch=4,
                      preprocess=just_index, loading=READ)


def test_a_mixture_of_streamed_splits_is_refused(two_splits):
    big, small = two_splits
    with pytest.raises(TypeError, match="read at random"):
        dew.data.load({f"hf/{big}": 0.5, f"hf/{small}": 0.5}, batch=4, streaming=True,
                      preprocess=just_index, loading=READ)


def test_a_mixture_computes_its_own_pass_and_takes_no_record_count(two_splits):
    big, small = two_splits
    with pytest.raises(ValueError, match="no records="):
        dew.data.load({f"hf/{big}": 0.5, f"hf/{small}": 0.5}, batch=4, records=48,
                      preprocess=just_index, loading=READ)


# --------------------------------------------------------------------------
# Packed windows across a changed process count
# --------------------------------------------------------------------------

def document_dir(root: Path, documents, eos: int = 0) -> str:
    """A token directory whose stream is `documents`, each closed by eos."""
    root.mkdir(parents=True, exist_ok=True)
    stream = np.concatenate([np.asarray(list(document) + [eos], np.uint16)
                             for document in documents])
    (root / "train.bin").write_bytes(stream.tobytes())
    (root / "val.bin").write_bytes(stream.tobytes())
    (root / "meta.json").write_text(json.dumps(
        {"tokenizer": "byte", "vocab_size": int(stream.max()) + 1, "dtype": "uint16",
         "eos_id": eos, "train_tokens": len(stream), "val_tokens": len(stream)}))
    return str(root)


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


def test_a_packed_step_is_the_same_windows_at_every_process_count(monkeypatch, tmp_path):
    """The packing is planned over the whole corpus ahead of the shard, so
    global batch k holds the same windows in the same places however many
    processes read it. Packed behind the shard, as the loader once was, each
    process packed its own documents and no two counts agreed."""
    corpus = packed_corpus(tmp_path / "corpus")
    open_stream = lambda: PackedTokens(  # noqa: E731  one line, read once
        path=corpus, seq_len=8, val_batches=None, packing_bins=2,
        loading=READ).load(batch=4).train()

    whole = pooled(monkeypatch, open_stream, 1, 5, rows=windows_of)

    assert whole[0], "the loader produced no windows"
    assert pooled(monkeypatch, open_stream, 2, 5, rows=windows_of) == whole
    assert pooled(monkeypatch, open_stream, 4, 5, rows=windows_of) == whole


def test_a_packed_position_written_by_two_processes_resumes_on_one(monkeypatch, tmp_path):
    """The packed stream's position is a global window count, so the two
    processes that stopped at step two hand one process, or four, the steps
    the run would have taken next."""
    corpus = packed_corpus(tmp_path / "corpus")
    open_stream = lambda: PackedTokens(  # noqa: E731  one line, read once
        path=corpus, seq_len=8, val_batches=None, packing_bins=2,
        loading=READ).load(batch=4).train()
    whole = pooled(monkeypatch, open_stream, 1, 6, rows=windows_of)

    as_processes(monkeypatch, 2, 1)
    stopped = open_stream()
    packed_rows(stopped, 2)
    state = stopped.get_state()

    assert json.loads(state)["dew_global_position"]["records"] == 8
    for processes in (1, 2, 4):
        assert pooled(monkeypatch, open_stream, processes, 4, state,
                      rows=windows_of) == whole[2:]


def test_a_packed_pass_covers_every_document_once_at_every_process_count(monkeypatch,
                                                                         tmp_path):
    """A validation pass over the plan is the same windows in the same order
    for every count, and the processes of a pool cover the split between
    them without overlapping."""
    corpus = packed_corpus(tmp_path / "corpus", documents=24)
    pass_over = lambda: PackedTokens(  # noqa: E731  one line, read once
        path=corpus, seq_len=8, val_batches=None, packing_bins=2,
        loading=READ).load(batch=4).val()

    as_processes(monkeypatch, 1, 0)
    whole = [row for batch in pass_over() for row in windows_of([batch])[0]]

    for processes in (2, 4):
        shards = []
        for index in range(processes):
            as_processes(monkeypatch, processes, index)
            shards.append([row for batch in pass_over() for row in windows_of([batch])[0]])
        assert sum(len(shard) for shard in shards) == len(whole)
        assert sorted(row for shard in shards for row in shard) == sorted(whole)


# --------------------------------------------------------------------------
# The batch ramp: MaxText's schedule
# --------------------------------------------------------------------------

def maxtext_batches(start: int, increment: int, samples: int, final: int,
                    steps: int) -> list[int]:
    """The global batch MaxText's `RampupBatchManager` reads at each of
    `steps` steps, as `utils/rampup_batch.py:53-101` computes it: the samples
    since the last increment are accumulated, and the batch grows once they
    reach `global_rampup_samples` over the number of increments."""
    increments = (final - start) // increment
    per_increment = samples / increments
    batch, accumulated, reads = start, 0, []
    for _ in range(steps):
        reads.append(batch)
        accumulated += batch
        if accumulated >= per_increment:
            batch = min(batch + increment, final)
            accumulated = 0
    return reads


def ramp_batches(ramp: Ramp, final: int, steps: int) -> list[int]:
    """The global batch this ramp reads at each of `steps` steps."""
    stages, records, reads = ramp.stages(final), 0, []
    for _ in range(steps):
        stage = stages[max(index for index, stage in enumerate(stages)
                           if stage.records <= records)]
        reads.append(stage.batch)
        records += stage.batch
    return reads


@pytest.mark.parametrize("start, increment, samples, final", [
    (4, 2, 500, 8),
    (8, 8, 64, 32),
    (2, 1, 7, 5),
    (16, 16, 1024, 64),
])
def test_the_ramp_reads_the_batch_maxtext_reads_at_every_step(start, increment,
                                                              samples, final):
    """Dew counts the global batch where MaxText counts a batch per device,
    and computes a stage's length as one integer ratio where MaxText divides
    twice in floating point; the schedule is the same schedule."""
    ramp = Ramp(start=start, increment=increment, samples=samples)

    assert ramp_batches(ramp, final, 40) == maxtext_batches(
        start, increment, samples, final, 40)


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


def test_a_ramp_stage_that_does_not_split_over_the_processes_is_refused(monkeypatch):
    as_processes(monkeypatch, 4, 0)
    with pytest.raises(ValueError, match="does not split over 4"):
        Ramp(start=6, increment=2, samples=48).stages(8)


def test_a_pass_over_the_data_is_more_steps_under_a_ramp():
    """A ramp reads fewer records a step early, so the same records are more
    steps: sixteen at four, then eight at eight, is twenty-four steps where
    a flat batch of eight is twelve."""
    ramp = Ramp(start=4, increment=4, samples=64)
    flat = Dataset(train=lambda: iter(()), val=None, records=96, batch=8)

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


def test_a_ramped_stream_reads_the_records_the_flat_stream_reads(monkeypatch):
    """The ramp cuts the same order into different steps, so the records it
    hands over are the flat stream's records in the flat stream's order: a
    stage change neither skips one nor reads one twice."""
    as_processes(monkeypatch, 1, 0)
    flat = [value for step in taken(indexed_data(64, 4).train(), 24) for value in step]

    stream = ramped(indexed_data(64, 4), Ramp(start=2, increment=1, samples=6)).train()
    ramp = taken(stream, 24)

    # Two steps of two, one of three, then the run's own four: the schedule
    # `test_the_ramp_reads_the_batch_maxtext_reads_at_every_step` pins.
    assert [len(step) for step in ramp] == [2, 2, 3] + [4] * 21
    assert [value for step in ramp for value in step] == flat[:sum(
        len(step) for step in ramp)]


def test_a_ramped_stream_resumes_on_the_records_it_had_not_read(monkeypatch):
    """The rolling buffer holds records the run has not trained on, so they
    are not in the position: a resume reads them, and reads them once."""
    as_processes(monkeypatch, 1, 0)
    schedule = Ramp(start=2, increment=1, samples=6)
    stream = ramped(indexed_data(64, 4), schedule).train()
    taken(stream, 4)
    state = stream.get_state()
    rest = taken(stream, 6)

    resumed = ramped(indexed_data(64, 4), schedule).train()
    resumed.set_state(state)

    assert json.loads(state)["dew_global_position"]["records"] == 11
    assert taken(resumed, 6) == rest


def test_a_ramped_stream_put_back_mid_read_forgets_its_buffer(monkeypatch):
    """A restore into a stream that is already reading, as the prefetch
    worker restores the stream it opened, hands over the records at the
    saved count and none it had buffered past it."""
    as_processes(monkeypatch, 1, 0)
    schedule = Ramp(start=2, increment=1, samples=6)
    stream = ramped(indexed_data(64, 4), schedule).train()
    taken(stream, 1)
    state = stream.get_state()
    rest = taken(stream, 6)

    stream.set_state(state)

    assert taken(stream, 6) == rest


def test_a_ramped_steps_position_counts_every_process(monkeypatch):
    """Each process cuts its own share of a stage's batch, and the position
    is the records all of them handed over: two processes four steps into
    a ramp of two, then four, have read eight records, and the pool's steps
    are the single process's steps with the shares interleaved."""
    schedule = Ramp(start=2, increment=2, samples=8)

    def build():
        rows = 4 // jax.process_count()
        data = Dataset(train=train_stream(Indexed(1, 64), [], batch=rows, seed=0, loading=READ),
                       val=None, records=64, batch=4)
        return ramped(data, schedule).train()

    whole = pooled(monkeypatch, build, 1, 6)
    as_processes(monkeypatch, 2, 1)
    stopped = build()
    taken(stopped, 4)

    assert [len(step) for step in whole] == [2, 2, 2, 2, 4, 4]
    assert json.loads(stopped.get_state())["dew_global_position"]["records"] == 8
    assert pooled(monkeypatch, build, 2, 6) == whole
    assert pooled(monkeypatch, build, 2, 2, stopped.get_state()) == whole[4:]


def test_a_position_no_step_of_this_ramp_ends_on_is_refused(monkeypatch):
    as_processes(monkeypatch, 1, 0)
    stream = ramped(indexed_data(64, 4), Ramp(start=2, increment=1, samples=6)).train()
    taken(stream, 3)
    state = stream.get_state()

    # Seven records in, which is a step boundary of the ramp that wrote it
    # (2 + 2 + 3) and lands inside a step of the one that reads it.
    other = ramped(indexed_data(64, 4), Ramp(start=2, increment=2, samples=6)).train()
    with pytest.raises(ValueError, match="ramped differently"):
        other.set_state(state)


def test_a_ramp_reads_through_a_captioned_datasets_tokenized_stream(monkeypatch):
    """An image or video dataset opens `tokenized` over its global stream, so
    the ramp has to cut the batches that stage hands over and save the
    position it forwards; a ramp that only knew the bare stream refused
    every captioned dataset by the wrapper's name."""
    as_processes(monkeypatch, 1, 0)

    class Captioned(Indexed):
        def __getitem__(self, index: int) -> dict:
            return {**super().__getitem__(index), CAPTION: f"caption {index}"}

    def lengths(captions):
        return {"length": np.asarray([len(caption) for caption in captions], np.int32)}

    def captioned(batch: int) -> Dataset:
        return Dataset(train=tokenized(train_stream(Captioned(1, 64), [], batch=batch,
                                                    seed=0, loading=READ), lengths),
                       val=None, records=64, batch=batch)

    flat = [value for step in taken(captioned(4).train(), 12) for value in step]
    schedule = Ramp(start=2, increment=1, samples=6)
    stream = ramped(captioned(4), schedule).train()
    steps = list(itertools.islice(stream, 6))
    state = stream.get_state()
    rest = taken(stream, 3)

    assert [len(step["id"]) for step in steps] == [2, 2, 3, 4, 4, 4]
    assert [int(value) for step in steps for value in step["id"]] == flat[:19]
    assert all(len(step["length"]) == len(step["id"]) for step in steps), (
        "the tokenized field was cut to another step than the ids")
    resumed = ramped(captioned(4), schedule).train()
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
    data = Dataset(train=stream, val=None, records=64, batch=8)
    with pytest.raises(TypeError, match=stream.__name__):
        ramped(data, Ramp(start=4, increment=4, samples=16)).train()


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

    def loss(self, params, batch, step):
        import jax.numpy as jnp

        features = jnp.stack([jnp.sin(batch["id"].astype(jnp.float32) * (index + 1))
                              for index in range(FEATURES)], axis=-1)
        target = jnp.stack([features[:, 0] * 2, features[:, 1] - 1], axis=-1)
        return jnp.mean((self.model.apply(params, features) - target) ** 2), Aux({})


def regression_trainer(directory: Path) -> Trainer:
    return Trainer(
        Regression(), optax.sgd(0.5), key=jax.random.key(0),
        layout=Layout(min_shard=1, tolerance=1.0),
        checkpoints=Checkpoints(str(directory), keep=8))


def parameters(state) -> dict:
    return {"/".join(str(part) for part in path): np.asarray(leaf)
            for path, leaf in jax.tree_util.tree_flatten_with_path(state.params)[0]}


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


def test_a_ramp_stage_the_mesh_cannot_hold_is_refused_before_the_run_reads(tmp_path):
    """Eight devices hold whole rows, so a stage of four records has nowhere
    to put them; the run says so before it trains rather than an hour in,
    when the stage arrives."""
    trainer = Trainer(Regression(), optax.sgd(0.5), key=jax.random.key(0),
                      layout=Layout(min_shard=1, tolerance=1.0))

    with pytest.raises(ValueError, match=r"\[4, 12\].*multiple of 8"):
        trainer.fit(ramped(indexed_data(256, 12), Ramp(start=4, increment=4, samples=16)),
                    steps=2, log_every=100)


def test_a_ramp_stage_the_pipeline_cannot_cut_into_microbatches_is_refused():
    """A pipelined step cuts its batch into microbatches, which the decoder
    checks when it traces, so a stage a later compile would refuse is
    refused before the run reads: two stages over four devices and six
    microbatches hold batches of twelve, and stages of eight and sixteen
    are not ones."""
    trainer = Trainer(Regression(), optax.sgd(0.5), key=jax.random.key(0),
                      mesh=MeshSpec(stage=2, microbatches=6),
                      layout=Layout(min_shard=1, tolerance=1.0))

    with pytest.raises(ValueError, match=r"\[8, 16\].*multiple of 12"):
        trainer.fit(ramped(indexed_data(256, 24), Ramp(start=8, increment=8, samples=16)),
                    steps=2, log_every=100)


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
