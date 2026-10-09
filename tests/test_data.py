"""Data layer tests: the registry, the Dataset contract, lazy imports, the
video specs and the determinism of what a record becomes.

`import dew.data` and the grain paths must not need the optional extras,
which is the point of several of these tests. Anything that genuinely
requires an optional dependency skips.

Record factories, holdout hashes and batch shares are Dew policies; Grain oracles iteration/resume.
"""

import atexit
import dataclasses
import inspect
import itertools
import json
import os
import sys
import threading
import time

import cv2
import grain.python as pygrain
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dew.data
from dew.data import (
    Checkpointable,
    DataPartition,
    Dataset,
    DatasetSpec,
    ImageDataset,
    Loading,
    LocalVideos,
    images,
    video,
)
from dew.data.dataset import Forwarding, GlobalStream, _batches, hold_out, train_stream, validation_pass
from dew.data.images import ImageTransform, decode_image
from dew.data.sources import av_utils
from dew.data.sources.av_utils import choose_clip_start
from dew.data.sources.hf import HFDatasetSource
from dew.objectives.base import VALID_ROWS
from dew.position import ENVELOPE
from dew.registry import datasets

WORKERS = {"loading": Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1)}


# ---------------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------------

def test_an_unknown_dataset_is_refused():
    with pytest.raises(KeyError, match="no dataset named 'flowers'"):
        datasets["flowers"]


def test_a_spec_field_the_dataset_has_no_declaration_for_is_refused():
    """A misspelled knob built a dataset other than the one asked for."""
    with pytest.raises(ValueError, match=r"unknown fields \['image_scale'\]"):
        datasets.build("tfds_images", image_scale=64)
    assert datasets.build("tfds_images", image_size=64).image_size == 64


def test_arrayrecord_datasets_require_an_explicit_path():
    """The default was one developer's bucket mount, and an unset path reached
    os.path.join(None, ...) inside the source."""
    with pytest.raises(ValueError, match="path="):
        datasets["array_record_images"](shards=("cc12m",)).load(batch=8)


def test_an_unknown_augmentation_is_refused():
    with pytest.raises(ValueError, match="not one of none, flip_only, flip_jitter"):
        images.image_augmentations("jitter")


def test_reading_a_hub_dataset_names_the_streaming_extra(monkeypatch):
    """Naming one works anywhere; the first record is what needs HF datasets."""
    source = HFDatasetSource(name="acme/pets")
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError, match=r"dewml\[streaming\]"):
        len(source)


def test_a_hub_dataset_spec_without_a_name_says_so():
    with pytest.raises(ValueError, match="name="):
        dew.data.HFImages().load(batch=4)


def test_the_streaming_spec_needs_sources_before_it_needs_the_streaming_stack():
    """Asking for nothing must fail on the spec, not on the missing dependency."""
    with pytest.raises(ValueError, match="sources="):
        dew.data.OnlineImages().load(batch=4)


# ---------------------------------------------------------------------------------
# What every dataset kind declares
# ---------------------------------------------------------------------------------

def _from_defaults(name):
    """The registered kind, built from its own defaults. A field with no
    default is one the kind cannot guess (the chat and prompt tokenizers),
    so the test names one rather than skipping the kind."""
    kind = datasets[name]
    required = {field.name: "byte" for field in dataclasses.fields(kind)
                if field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING}
    return kind(**required)


@pytest.mark.parametrize("name", sorted(datasets))
def test_every_dataset_kind_carries_a_seed_and_a_loading(name):
    """Both belong to reading any dataset, so they are declared once on
    DatasetSpec; a kind that redeclares either can drift from it, which is
    how the streamed spec lost its seed."""
    spec = _from_defaults(name)

    assert spec.seed == 0
    assert isinstance(spec.loading, Loading)
    assert dataclasses.replace(spec, seed=7, loading=Loading(workers=0)).seed == 7


@pytest.mark.parametrize("name", sorted(datasets))
def test_every_dataset_kind_takes_the_base_load_signature(name):
    """A recipe holds a DatasetSpec, not the kind it named, so `load(batch=,
    tokenize=)` has to bind on all of them."""
    spec = _from_defaults(name)

    inspect.signature(type(spec).load).bind(spec, batch=8, tokenize=None)


def test_a_dataset_that_writes_no_captions_refuses_a_caption_reader(tmp_path):
    """tokenize= is on every load so one caller can load any spec. A token
    corpus has no captions, and dropping the reader silently would train a
    conditional run on nothing."""
    (tmp_path / "train.bin").write_bytes(np.arange(64, dtype=np.uint16).tobytes())
    (tmp_path / "val.bin").write_bytes(np.arange(64, dtype=np.uint16).tobytes())
    spec = dew.data.TokenWindows(path=str(tmp_path), seq_len=8, **WORKERS)

    with pytest.raises(TypeError, match="TokenWindows writes no captions"):
        spec.load(batch=2, tokenize=keep_captions)
    assert spec.load(batch=2).records == 7


def test_a_run_config_loads_its_dataset_through_the_base_spec(tmp_path):
    """`RunConfig.data` is typed as the base, so the config layer reaches
    `load` without knowing which kind it holds."""
    from dew.config import RunConfig

    (tmp_path / "train.bin").write_bytes(np.arange(64, dtype=np.uint16).tobytes())
    (tmp_path / "val.bin").write_bytes(np.arange(64, dtype=np.uint16).tobytes())
    config = RunConfig(data=dew.data.TokenWindows(path=str(tmp_path), seq_len=8, **WORKERS))

    spec: DatasetSpec = config.data
    data = spec.load(batch=2)

    assert data.batch == 2
    assert next(data.train(DataPartition()))["text"].shape == (2, 9)


# ---------------------------------------------------------------------------------
# The Dataset contract, on a spec of indexed records
# ---------------------------------------------------------------------------------

class _Indexed:
    """Minimal random-access source; stands in for arrayrecord/video sources."""

    def __init__(self, length):
        self.records = [{"index": i} for i in range(length)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


@dataclasses.dataclass(frozen=True)
class Indexed(DatasetSpec):
    """Records that carry their own index, untouched, so a test can read
    which records a batch holds. The plumbing is the one every spec uses."""

    length: int = 32
    val_batches: int | None = None
    count: int | None = None
    seed: int = 0
    loading: Loading = dataclasses.field(default_factory=lambda: Loading(workers=0))

    def source(self):
        return _Indexed(self.length)

    def load(self, *, batch):
        source = self.source()
        records = len(source) if self.count is None else self.count
        train, val = hold_out(source, records, (self.val_batches or 0) * batch, "Indexed")
        knobs = {"batch": batch, "seed": self.seed, "loading": self.loading}
        return Dataset(train=train_stream(train, [], **knobs),
                       val=None if val is None else validation_pass(val, [], **knobs),
                       records=len(train), batch=batch)


def _indices(iterator, num_batches):
    return [[int(i) for i in batch["index"]]
            for batch in itertools.islice(iterator, num_batches)]


def _bounded(iterator, limit):
    """At most `limit` batches, and whether the stream ended inside them.

    Bounded on purpose: an endless stream then fails a count instead of
    hanging the suite.
    """
    taken = list(itertools.islice(iterator, limit))
    return taken, next(iterator, None) is None


def test_a_dataset_without_a_held_out_split_has_no_validation_pass():
    data = Indexed().load(batch=8)
    assert data.records == 32 and data.batch == 8 and data.steps_per_epoch == 4
    assert data.val is None


def test_the_validation_split_is_ordered_and_disjoint_from_train():
    data = Indexed(val_batches=1).load(batch=8)

    assert data.records == 24  # the held-out records leave the train stream
    assert data.steps_per_epoch == 3

    # Validation walks its own records in canonical order, not the shuffled
    # train sampler's, and repeats identically.
    val_batches = _indices(data.val(DataPartition()), 2)
    assert val_batches == [list(range(8))]
    assert _indices(data.val(DataPartition()), 2) == val_batches

    train_indices = [i for batch in _indices(data.train(DataPartition()), 3) for i in batch]
    assert set(train_indices).isdisjoint(range(8))
    assert train_indices != sorted(train_indices)  # the train sampler still shuffles


@pytest.mark.parametrize("workers", [0, 2])
def test_a_validation_pass_reads_every_held_out_record_once(workers):
    """A pass is the split, once, in record order, and then it ends.

    grain's DataLoader applies its operations inside the worker processes, so
    each worker had to fill a whole batch out of its own slice of the split,
    and the unbounded num_epochs a run leaves at None let it read that slice
    again to do so.
    """
    data = Indexed(val_batches=3, loading=Loading(workers=workers)).load(batch=8)

    batches, ended = _bounded(data.val(DataPartition()), 12)
    assert [[int(i) for i in b["index"]] for b in batches] == [
        list(range(8)), list(range(8, 16)), list(range(16, 24))]
    assert ended
    train = [index for batch in _indices(data.train(DataPartition()), 2) for index in batch]
    assert set(train).isdisjoint(range(24))


def test_a_validation_split_cannot_swallow_every_record():
    """One record and a held-out one would leave nothing to train on."""
    with pytest.raises(ValueError, match="leaves nothing"):
        Indexed(val_batches=4).load(batch=8)
    with pytest.raises(ValueError, match="leaves nothing"):
        Indexed(length=1, val_batches=1).load(batch=1)


# ---------------------------------------------------------------------------------
# Validation from a split of the dataset's own
# ---------------------------------------------------------------------------------

def _labels(iterator, batches=1):
    return [int(label) for batch in itertools.islice(iterator, batches)
            for label in batch["label"]]


def test_a_named_validation_split_is_its_own_records_and_costs_training_none():
    """Held out of the head, validation was training records the run then
    never saw. A dataset that has a validation split of its own should be
    scored on it and trained on everything."""
    spec = Augmenting(length=16, splits={"test": 8}, image_size=8, augmentation="none",
                      val_split="test", val_batches=1, **WORKERS)

    data = spec.load(batch=4, tokenize=keep_captions)

    assert data.records == 16, "a named split holds nothing out of training"
    assert data.val is not None
    val = _labels(data.val(DataPartition()))
    train = _labels(data.train(DataPartition()), batches=4)
    assert val == [1000, 1001, 1002, 1003]
    assert set(val).isdisjoint(train)
    assert sorted(train) == list(range(16)), "training reads every record"


def test_val_batches_bounds_a_named_split_and_none_scores_all_of_it():
    """The two readings of val_batches: a bound on a split of its own, a
    hold-out count without one."""
    fields = dict(length=16, splits={"test": 8}, image_size=8, augmentation="none",
                  val_split="test", **WORKERS)

    bounded_pass = Augmenting(val_batches=1, **fields).load(batch=4)
    whole_pass = Augmenting(val_batches=None, **fields).load(batch=4)

    assert bounded_pass.val is not None and whole_pass.val is not None
    assert len(_labels(bounded_pass.val(DataPartition()), batches=8)) == 4
    assert _labels(whole_pass.val(DataPartition()), batches=8) == list(range(1000, 1008))


def test_without_a_named_split_the_head_hold_out_is_unchanged():
    """The default path has to stay exactly what it was: the same records,
    the same pixels, the same training count."""
    spec = Augmenting(length=16, image_size=8, augmentation="none", val_batches=1,
                      **WORKERS)

    data = spec.load(batch=4, tokenize=keep_captions)

    assert data.records == 12 and data.steps_per_epoch == 3
    held = next(data.val(DataPartition()))
    assert _labels(iter([held])) == [0, 1, 2, 3]
    assert set(_labels(data.train(DataPartition()), batches=3)).isdisjoint(range(4))
    expected = ImageTransform(spec).random_map(_Images(16)[0], np.random.default_rng(0))
    assert held["image"][0].tobytes() == expected["image"].tobytes()


def test_a_dataset_of_one_pile_refuses_a_split_it_cannot_name(tmp_path):
    """An arrayrecord spec reads its shards as one pile; scoring a split it
    cannot name would read the training records again and call it validation."""
    from tools.prepare_images import prepare

    prepare(Augmenting(length=8, image_size=8, augmentation="none", val_batches=None,
                       **WORKERS), str(tmp_path), shards=1, source={"dataset": "a"})
    spec = images.ArrayRecordImages(path=str(tmp_path), image_size=8, val_batches=1,
                                    **WORKERS)

    assert spec.load(batch=2).records == 6, "the head hold-out still works"
    with pytest.raises(ValueError, match="val_split='test' names nothing"):
        dataclasses.replace(spec, val_split="test").load(batch=2)


class _Endless:
    def __getitem__(self, index):
        return {"index": index, "image": np.zeros((4, 4, 3), np.uint8)}


@dataclasses.dataclass(frozen=True)
class Unsized(ImageDataset):
    def source(self, split=None):
        return _Endless()

    def record(self, element, rng):
        return element["image"], "", element["index"]


def test_a_source_without_a_length_needs_an_explicit_count():
    """The factory guessed a million records for such a source, so the
    sampler drew indices past the data and the run reported the guess."""
    with pytest.raises(ValueError, match="count="):
        Unsized(**WORKERS).load(batch=8)

    data = Unsized(count=16, val_batches=None, image_size=4, **WORKERS).load(batch=8)
    assert data.records == 16
    assert sorted(int(i) for batch in itertools.islice(data.train(DataPartition()), 2)
                  for i in batch["label"]) == list(range(16))


def test_a_count_past_the_end_of_the_source_is_refused(tmp_path):
    """A count above the source became the sampler's record count, and the
    first index past the end raised inside a worker."""
    with pytest.raises(ValueError, match="count 33 is more than the 32 records"):
        Augmenting(length=32, count=33, **WORKERS).load(batch=8)
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "b.mp4").write_bytes(b"")
    with pytest.raises(ValueError, match="count 3 is more than the 2 records"):
        LocalVideos(path=str(tmp_path), count=3, **WORKERS).load(batch=1)


def test_a_count_uses_the_head_of_the_source():
    data = Indexed(count=16).load(batch=8)
    assert data.records == 16
    assert sorted(i for batch in _indices(data.train(DataPartition()), 2) for i in batch) == list(range(16))


def test_the_training_stream_repeats_instead_of_ending():
    """The trainer keeps asking for batches long after one pass over the
    records, and the next pass is the same records in another order."""
    data = Indexed(val_batches=2).load(batch=8)

    epoch = data.steps_per_epoch  # the sixteen training records, in two batches
    batches, ended = _bounded(data.train(DataPartition()), 3 * epoch)
    first_epoch = [int(i) for batch in batches[:epoch] for i in batch["index"]]
    later = [int(i) for batch in batches[epoch:] for i in batch["index"]]

    assert not ended and len(batches) == 3 * epoch
    assert sorted(first_epoch) == list(range(16, 32))
    assert set(later) == set(first_epoch), "the stream reads the same records again"


def test_a_dataset_of_one_record_yields_batches_of_it():
    data = Indexed(length=1).load(batch=1)
    assert _indices(data.train(DataPartition()), 2) == [[0], [0]]


def test_the_training_iterator_carries_its_position():
    """A checkpoint records the iterator's state and a restored run resumes
    on the batch after it."""
    data = Indexed().load(batch=8)
    first = data.train(DataPartition())
    # The trainer reads the protocol statically, and a caption stage that only
    # forwarded get_state through __getattr__ failed it while hasattr passed.
    assert isinstance(first, Checkpointable)
    seen = _indices(first, 2)
    state = first.get_state()
    rest = _indices(first, 2)

    resumed = data.train(DataPartition())
    resumed.set_state(state)
    assert _indices(resumed, 2) == rest
    assert sorted(i for batch in seen + rest for i in batch) == list(range(32))


# ---------------------------------------------------------------------------------
# The batch is stacked in the worker that read the records
# ---------------------------------------------------------------------------------

def _order(source, seed=None):
    """The order `_batches` slices: the corpus, or it reshuffled endlessly."""
    records = pygrain.MapDataset.source(source).seed(0 if seed is None else seed)
    return records if seed is None else records.shuffle(seed).repeat(None)


def _by_hand(order, *, batch, count, offset=0):
    """`count` batches stacked the way the training process stacked them
    before the workers did: this process's slice of `order`, read record by
    record in order, `batch` records to a batch.

    Read off the `MapDataset` and not off `_batches`, so it is a reference
    for what `_batches` returns rather than a restatement of it.
    """
    mine = order[offset::1]
    rows = [mine[index] for index in range(count * batch)]
    return [{field: [row[field] for row in rows[start:start + batch]]
             for field in rows[start]}
            for start in range(0, count * batch, batch)]


def _fields(iterator, count):
    """`count` batches as plain lists, so a comparison covers every field of
    every record rather than one of them."""
    return [{field: np.asarray(rows).tolist() for field, rows in batch.items()}
            for batch in itertools.islice(iterator, count)]


@pytest.mark.parametrize("workers", [0, pytest.param(1, marks=pytest.mark.slow),
                                     pytest.param(2, marks=pytest.mark.slow),
                                     pytest.param(3, marks=pytest.mark.slow)])
def test_which_records_a_batch_holds_does_not_depend_on_who_stacked_it(workers):
    """A worker stacks the batch it read, so the records it is handed have to
    be a whole batch. Grain gives worker w of W every Wth element of the
    order, which is a batch's worth of every Wth record until the order is
    permuted; unpermuted, three workers at a batch of four would have stacked
    records 0, 3, 6, 9 into batch 0."""
    source = _Indexed(30)
    stream = _batches(_order(source, seed=5), rows=4, partition=DataPartition(), loading=Loading(
        workers=workers, threads=2, read_buffer=4, worker_buffer=2))

    read = _fields(stream, 12)
    stream.close()

    assert read == _by_hand(_order(source, seed=5), batch=4, count=12)
    assert len({tuple(batch["index"]) for batch in read}) == 12, (
        "twelve batches of four is more than one pass over thirty records, "
        "so a reshuffled epoch is inside this comparison")


@pytest.mark.parametrize("workers", [0, pytest.param(2, marks=pytest.mark.slow)])
def test_a_pass_over_a_split_the_workers_do_not_divide_reads_every_record(workers):
    """Thirty records at a batch of four is seven whole batches and two
    records over. Two workers take a batch each in turn, so the last round is
    one worker's alone: every batch arrives in its place, and the eighth
    holds the two records over, filled out with repeats of them that
    `VALID_ROWS` marks."""
    passes = validation_pass(_Indexed(30), [], batch=4, seed=0, loading=Loading(
        workers=workers, threads=2, read_buffer=4, worker_buffer=2))

    batches, ended = _bounded(passes(DataPartition()), 12)

    assert [[int(index) for index in batch["index"]] for batch in batches] == [
        *(list(range(start, start + 4)) for start in range(0, 28, 4)), [28, 29, 28, 29]]
    assert [np.asarray(batch.get(VALID_ROWS, np.ones(4, bool))).tolist() for batch in batches][-2:] == [
        [True] * 4, [True, True, False, False]]
    assert ended


@pytest.mark.parametrize("workers", [0, pytest.param(2, marks=pytest.mark.slow)])
def test_an_offset_stays_a_bound_on_the_slice_this_process_reads(workers):
    """A resume opens the stream `offset` records in, and the permutation the
    workers read through sits behind that slice, so the first batch is still
    the first four records the interrupted run had not reached."""
    stream = _batches(_order(_Indexed(30)), rows=4, partition=DataPartition(), offset=8, loading=Loading(
        workers=workers, threads=2, read_buffer=4, worker_buffer=2))

    first = _fields(stream, 1)
    stream.close()

    assert first == [{"index": [8, 9, 10, 11]}]


# ---------------------------------------------------------------------------------
# The global batch over the shares of a partition
# ---------------------------------------------------------------------------------

def test_a_global_batch_that_does_not_split_into_the_shares_is_refused():
    """Integer division hid the remainder: 65 over eight shares trained on
    64 records a step while the run reported 65, and 7 gave every share a
    batch of nothing."""
    data = Indexed(length=256)
    for batch in (65, 7):
        with pytest.raises(ValueError, match=rf"batch {batch} does not split into 8 equal shares"):
            DataPartition(0, 8).rows(batch)
        with pytest.raises(ValueError, match="8 equal shares"):
            data.load(batch=batch).train(DataPartition(3, 8))
    assert DataPartition(0, 8).rows(64) == 8


def test_each_share_reads_its_own_slice_of_the_validation_split():
    """Share p of n validates records p, p + n, ... of the split, in whole
    batches of the share's size."""
    data = Indexed(val_batches=2).load(batch=8)

    assert _indices(data.val(DataPartition(1, 2)), 3) == [[1, 3, 5, 7], [9, 11, 13, 15]]


def test_the_readers_of_one_share_read_the_same_records():
    """Processes a sequence or a pipeline spans between them hold the same
    rows, so they read one share, and read it alike: that is what makes
    their copies of the batch agree."""
    data = Indexed(length=32).load(batch=8)
    first = data.train(DataPartition(1, 2, readers=2))
    second = data.train(DataPartition(1, 2, readers=2))

    assert _indices(first, 3) == _indices(second, 3)
    first.close()
    second.close()


def _steps(processes, count, *, position=None, length=32, seed=0, batch=8):
    """`(global batches, saved positions)` for `processes` readers, one share each.

    A global batch is the step every share contributes its rows to, so the
    records are collected across the shares; which device holds which row
    is the mesh's business and which records a step trains on is the
    dataset's. Each share is opened in turn, resumed from `position` when
    one is given, and asked where it stopped.
    """
    read, saved = [], []
    for index in range(processes):
        stream = Indexed(length=length, seed=seed).load(batch=batch).train
        iterator = stream(DataPartition(index, processes))
        if position is not None:
            iterator.set_state(position)
        read.append(_indices(iterator, count))
        saved.append(iterator.get_state())
        iterator.close()
    return [sorted(index for rows in read for index in rows[step])
            for step in range(count)], saved


@pytest.mark.parametrize("processes", [1, 2, 4])
def test_a_training_step_reads_the_same_records_at_every_share_count(processes):
    """Step k is records [k * batch, (k + 1) * batch) of one order whatever
    the share count is, because the iterator owns the sharding and the
    batching together. Shuffling the corpus per shard first, as an index
    sampler does, gave each count a different order, which no encoding of a
    position could have translated."""
    alone, _ = _steps(1, 4)
    together, _ = _steps(processes, 4)

    assert together == alone
    assert sorted(index for step in alone for index in step) == list(range(32)), (
        "four steps of eight is one pass over the corpus, each record once")


def test_every_share_saves_the_same_global_position():
    """A position is a record count over the whole run's order, so every
    share reports the same bytes; that is what lets a checkpoint written by
    two processes be handed to one or to four."""
    _, saved = _steps(4, 3)

    assert len(set(saved)) == 1
    assert json.loads(saved[0])[ENVELOPE]["records"] == 3 * 8


@pytest.mark.parametrize("processes", [1, 4])
def test_a_position_saved_by_two_shares_resumes_on_another_count(processes):
    """The steps after a resume are the steps the run that was never stopped
    would have trained on, whether the resume has half the shares or twice
    them."""
    whole, _ = _steps(2, 6)
    _, saved = _steps(2, 3)

    resumed, _ = _steps(processes, 3, position=saved[0])

    assert resumed == whole[3:]


def test_a_position_over_another_order_is_refused():
    """A record count is a place in one order and another place in another, so
    a corpus, record count or seed the position was not written over is
    refused instead of resumed at the same offset into different data."""
    _, saved = _steps(2, 3)

    for other in ({"length": 64}, {"seed": 1}):
        stream = Indexed(**other).load(batch=8).train(DataPartition())
        with pytest.raises(ValueError, match="records into"):
            stream.set_state(saved[0])
        stream.close()


def test_a_shard_offset_cannot_resume_a_global_stream():
    """A stream whose windows come out of one shard reports where that shard
    stopped. Handing such a position to a stream that reads its records
    globally would resume it somewhere else, so it is refused by name."""
    stream = Indexed().load(batch=8).train(DataPartition())

    with pytest.raises(ValueError, match="one process's own offset into its shard"):
        stream.set_state(json.dumps({"last_seen_indices": {"0": 15}}).encode())
    stream.close()


class _Corpus:
    """Records of one named corpus, described by that name as a source with
    an identity is (`describe`)."""

    def __init__(self, name, length=16):
        self.name, self.length = name, length

    def __repr__(self):
        return f"_Corpus({self.name!r})"

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {"index": np.int32(index)}


def test_a_held_out_slice_keeps_the_identity_of_the_corpus_it_cuts():
    """Two corpora of one type and length, each with a slice held out, were
    described the same, so a position saved over one resumed the other."""
    def stream(name):
        train, _ = hold_out(_Corpus(name), 16, 4, "corpus")
        return train_stream(train, [], batch=4, seed=0, loading=Loading(threads=1, read_buffer=1))(
            DataPartition())

    saved = stream("flowers")
    next(saved)
    state = saved.get_state()
    saved.close()

    other = stream("faces")
    with pytest.raises(ValueError, match="_Corpus\\('flowers'\\)"):
        other.set_state(state)
    resumed = stream("flowers")
    resumed.set_state(state)
    resumed.close()


def test_records_listed_in_memory_are_described_by_their_count_not_their_contents():
    """A list has a repr, and it is every record: a saved position held it all."""
    rows = [{"x": np.full(64, index, np.float32)} for index in range(512)]
    stream = Dataset.from_records(rows, batch=8).train(DataPartition())
    next(stream)

    assert len(stream.get_state()) < 512
    stream.close()


def test_a_filtering_grain_dataset_is_refused_for_a_global_position():
    """A filter yields fewer records than the indices it reads, so a record
    count no longer says where a resumed run starts: one resumed on records
    it had already read. Grain's own ElasticIterator refuses it the same way."""
    filtered = pygrain.MapDataset.source(_Points(16)).filter(lambda point: point["index"] % 3)

    shuffled = filtered.seed(0).shuffle()
    mixed = pygrain.MapDataset.mix([_grain_points(), filtered.repeat(2)])
    for pipeline in (filtered, shuffled, mixed):
        with pytest.raises(ValueError, match="filter"):
            Dataset.from_grain(pipeline, batch=4, **WORKERS)
    with pytest.raises(ValueError, match="filter"):
        Dataset.from_grain(_grain_points(), validation=filtered, batch=4, **WORKERS)


# ---------------------------------------------------------------------------------
# Clip sampling uses a local RNG
# ---------------------------------------------------------------------------------

def test_clip_start_is_reproducible_without_touching_the_global_rng():
    """Clip starts come from the generator passed in; the global RNG is untouched."""
    np.random.seed(1234)
    global_state = np.random.get_state()[1].copy()

    starts = [choose_clip_start(100, 16, 3, np.random.default_rng(7)) for _ in range(3)]

    assert len(set(starts)) == 1
    assert np.array_equal(global_state, np.random.get_state()[1])


def test_clip_start_stays_inside_the_padded_window():
    starts = {choose_clip_start(100, 16, 3, np.random.default_rng(seed))
              for seed in range(64)}
    assert min(starts) >= 3
    assert max(starts) <= 100 - 16 - 3
    assert len(starts) > 1  # different seeds really do move the window


def test_clip_start_falls_back_to_the_only_valid_offset():
    assert choose_clip_start(22, 16, 3, np.random.default_rng(0)) == 3


# ---------------------------------------------------------------------------------
# Video datasets
# ---------------------------------------------------------------------------------

def _voxceleb_tree(root, split="train", identities=("id00012", "id00015")):
    """Build <root>/<split>/<identity>/<clip>/<utterance>.mp4, plus some noise."""
    clips = []
    for identity in identities:
        for clip in ("_raOc3-IRsw", "21Uxsk56VDQ"):
            path = root / split / identity / clip / "00001.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not really an mp4")
            clips.append(path)
    (root / split / "filelist.txt").write_text("ignored\n")
    return clips


def test_local_videos_lists_every_file_under_the_directory(tmp_path):
    clips = _voxceleb_tree(tmp_path)
    (tmp_path / "extra.webm").write_bytes(b"")

    records = LocalVideos(path=str(tmp_path), caption="a clip").source()

    assert [r["video_path"] for r in records] == sorted(
        [str(c) for c in clips] + [str(tmp_path / "extra.webm")]
    )
    assert {r["caption"] for r in records} == {"a clip"}
    with pytest.raises(ValueError, match="path="):
        LocalVideos().source()


def test_video_records_flow_through_the_audio_video_transform(tmp_path, monkeypatch):
    """End to end with the reader stubbed and the audio model's extractor
    built here, so its weights are the only thing not real. The extractor
    takes the padded waveform whole and normalises it as one clip, which is
    what the audio condition receives; handed the rows instead, wav2vec2's
    extractor read each frame's 640 samples as a clip of its own and the
    record kept the first, a frame of padding."""
    from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor

    _voxceleb_tree(tmp_path)
    spec = LocalVideos(path=str(tmp_path), caption="a video of a speaker",
                       frame_size=32, frames=4, audio_padding=1)
    records = spec.source()
    frame_samples = 640
    seen = {}

    def fake_read_av_random_clip(video_path, *, num_frames, audio_padding, seed, sample_rate=16000,
                                 fps=25.0):
        seen.update(video_path=video_path, num_frames=num_frames,
                    audio_padding=audio_padding, seed=seed)
        padded = num_frames + 2 * audio_padding
        audio = np.linspace(-0.5, 0.5, padded * int(sample_rate / fps), dtype=np.float32)
        # Native resolution differs from frame_size, so the resize must happen
        return np.zeros((num_frames, 96, 48, 3), np.uint8), audio.reshape(padded, -1)

    monkeypatch.setattr(av_utils, "read_av_random_clip", fake_read_av_random_clip)
    monkeypatch.setattr(AutoFeatureExtractor, "from_pretrained",
                        lambda name: Wav2Vec2FeatureExtractor(sampling_rate=16000))

    batch = video.AudioVideoTransform(spec).random_map(records[0], np.random.default_rng(0))

    assert seen["video_path"] == records[0]["video_path"]
    assert seen["num_frames"] == 4 and seen["audio_padding"] == 1
    assert seen["seed"] == int(np.random.default_rng(0).integers(0, 2**32 - 1))
    assert batch["video"].shape == (4, 32, 32, 3)
    assert batch["caption"] == "a video of a speaker"
    waveform = batch["audio"]["full_audio"]
    assert waveform.shape == (6, frame_samples) and waveform[0, 0] == -0.5
    flat = waveform.reshape(-1)
    normalised = (flat - flat.mean()) / np.sqrt(flat.var() + 1e-7)
    assert np.allclose(batch["audio"]["input_values"], normalised, atol=1e-5)


def test_a_clip_is_decoded_at_the_rate_its_audio_model_reads(tmp_path, monkeypatch):
    """The extractor is told the waveform is at its own rate, so the clip has
    to be decoded at that rate: a 48 kHz model's record holds 1920 samples a
    frame at 25 fps, where a 16 kHz read would hand it 640 and the extractor
    would take a third of a second of sound for a whole one."""
    from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor

    _voxceleb_tree(tmp_path)
    spec = LocalVideos(path=str(tmp_path), frame_size=32, frames=4, audio_padding=1)

    def fake_read_av_random_clip(video_path, *, num_frames, audio_padding, seed, sample_rate=16000,
                                 fps=25.0):
        rows = num_frames + 2 * audio_padding
        return (np.zeros((num_frames, 32, 32, 3), np.uint8),
                np.zeros((rows, int(sample_rate / fps)), np.float32))

    monkeypatch.setattr(av_utils, "read_av_random_clip", fake_read_av_random_clip)
    monkeypatch.setattr(AutoFeatureExtractor, "from_pretrained",
                        lambda name: Wav2Vec2FeatureExtractor(sampling_rate=48000))

    batch = video.AudioVideoTransform(spec).random_map(spec.source()[0], np.random.default_rng(0))

    assert batch["audio"]["full_audio"].shape == (6, 48000 // 25)


def keep_captions(captions):
    """A caption reader that hands the words back, so a test reads what the
    dataset wrote before a run's encoder tokenizes it."""
    return {"caption": np.asarray(captions)}


CAPTION_TEMPLATES = ("a photo of a {}", "a photo of a {} flower", "This is a photo of a {}")


# ---------------------------------------------------------------------------------
# Determinism across worker counts, and a restart mid-epoch
# ---------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Augmenting(ImageDataset):
    """Deterministic images and class captions, addressed by index, through
    the real image transform: resize, flip, jitter and the prompt template
    all draw from grain's per-record rng, the draw a worker count could move.
    The captions are read back as text, so one stays comparable after a trip
    through a worker process."""

    length: int = 16
    splits: dict[str, int] = dataclasses.field(default_factory=dict)
    """Records each named split holds, for a spec that has more than one."""

    def source(self, split=None):
        return _Images(self.length if split is None else self.splits[split],
                       first=0 if split is None else 1000)

    def record(self, element, rng):
        template = CAPTION_TEMPLATES[int(rng.integers(len(CAPTION_TEMPLATES)))]
        name = ["rose", "tulip", "lotus", "orchid", "marigold"][element["index"] % 5]
        return element["image"], template.format(name), element["index"]


class _Images:
    """`length` images, each one its index's own; `first` shifts the indices
    so two of these hold no record in common."""

    def __init__(self, length, first=0):
        self.length = length
        self.first = first

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index += self.first
        rng = np.random.RandomState(index)
        return {"index": index, "image": rng.randint(0, 256, (12, 12, 3), np.uint8)}


def _rows(batch):
    """(index, pixels, caption) per record, so a comparison covers all three."""
    return [(int(batch["label"][row]), batch["image"][row].tobytes(),
             str(batch["caption"][row]))
            for row in range(len(batch["label"]))]


def _augmented(worker_count, length=16, batch=4, seed=3):
    """{record index: (pixels, caption)} for one epoch at this worker count."""
    data = Augmenting(length=length, image_size=8, seed=seed, val_batches=None,
                      loading=Loading(workers=worker_count, threads=1, read_buffer=1,
                                      worker_buffer=1)).load(batch=batch, tokenize=keep_captions)
    return {index: (pixels, caption)
            for b in itertools.islice(data.train(DataPartition()), data.steps_per_epoch)
            for index, pixels, caption in _rows(b)}


@pytest.mark.slow
@pytest.mark.parametrize("worker_count", [1, 2, 4])
def test_a_records_pixels_and_caption_do_not_depend_on_worker_count(
        worker_count):
    """The flip, the colour jitter and the prompt template all come from the
    per-record rng, which grain keys by record index, so the number of workers
    that produced a batch cannot change what is in it."""
    serial = _augmented(0)
    parallel = _augmented(worker_count)

    assert sorted(serial) == list(range(16))
    assert serial == parallel


def test_the_close_budget_is_the_sources_worker_count():
    """A grain pipeline joins its workers one by one, so the prefetch close
    waits what the stream's Loading says, through every wrapper over it."""
    loading = Loading(workers=8, threads=1, read_buffer=1)
    stream = Augmenting(length=16, image_size=8, seed=3, val_batches=None,
                        loading=loading).load(batch=4, tokenize=keep_captions).train(DataPartition())
    assert isinstance(stream, Forwarding)
    assert stream.stop_seconds == loading.stop_seconds > 5.0
    stream.close()
    assert stream.stop_seconds is None


_LINGERING: list[float] = []


def _linger(seconds: float) -> None:
    """Make this process take `seconds` longer to exit, once."""
    if not _LINGERING:
        _LINGERING.append(seconds)
        atexit.register(time.sleep, seconds)


@dataclasses.dataclass(frozen=True)
class Lingering(Augmenting):
    """Augmenting's records from worker processes that each take `seconds`
    to exit once stopped, as a worker finishing a slow batch does."""

    seconds: float = 2.0

    def record(self, element, rng):
        _linger(self.seconds)
        return super().record(element, rng)


@pytest.mark.slow
def test_a_stream_whose_workers_are_slow_to_stop_closes_within_grains_bound(caplog):
    """grain stops a stream's worker processes one after another, each
    finishing the batch in its hands before it exits, and kills a worker
    that has not exited within 25 s. Four workers taking 2 s each keep a
    close over 8 s: slow, but bounded by grain itself, so the close must not
    call it a hang, and says what it waits for. sft_gemma4's four workers
    took 5.1 to 7.4 s to stop, past a budget of 2 s and 1 s a worker."""
    from dew.training import MeshSpec
    from dew.training.distributed import DevicePrefetchIterator

    loading = Loading(workers=4, threads=1, read_buffer=1, worker_buffer=1)
    data = Lingering(length=64, image_size=8, seed=3, val_batches=None, loading=loading)
    stream = data.load(batch=4).train(DataPartition())
    prefetch = DevicePrefetchIterator(stream, MeshSpec().build(jax.devices()[:1]))
    # grain interleaves its workers' batches in turn: one batch from each
    # means every worker has read a record and lingers.
    for _ in range(loading.workers):
        next(prefetch)
    began = time.perf_counter()
    prefetch.close()
    assert time.perf_counter() - began > loading.workers * data.seconds
    assert "waiting for 4 grain workers to stop" in caplog.text


def test_a_stop_closes_grain_when_no_thread_can_announce_it(monkeypatch):
    """At interpreter shutdown no thread starts, so neither does the timer
    that announces a long stop; the stop closes grain's pipeline all the
    same, or its workers would outlive the run."""
    def refused(timer):
        raise RuntimeError("can't create new thread at interpreter shutdown")

    monkeypatch.setattr(threading.Timer, "start", refused)
    closed = []

    class Reads:
        def __next__(self):
            return {"text": np.zeros((1, 2), np.int32)}

        def close(self):
            closed.append(True)

    stream = GlobalStream(lambda offset: Reads(), 1, "one order", Loading(workers=2))
    next(stream)
    stream.close()
    assert closed == [True]


def test_augmentation_really_moves_the_pixels():
    """Guards the test above: identical records at every worker count would
    also be true of a pipeline that augmented nothing."""
    records = _augmented(0)
    source = _Images(16)

    assert any(records[i][0] != cv2.resize(source[i]["image"], (8, 8),
                                           interpolation=cv2.INTER_AREA).tobytes()
               for i in records)
    assert len({caption for _, caption in records.values()}) > 1


def test_augment_image_is_deterministic_for_one_seed():
    """A record's augmentation is keyed by its rng alone: the same seed
    reproduces it exactly, a different seed moves the pixels."""
    augment = images.image_augmentations("flip_jitter")
    image = _Images(1)[0]["image"]

    first = images.augment_image(augment, image, np.random.default_rng(7))
    again = images.augment_image(augment, image, np.random.default_rng(7))
    other = images.augment_image(augment, image, np.random.default_rng(8))

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_flip_only_reorders_pixels_without_changing_them():
    """flip_only may mirror an image but never touches its values, so the
    sorted pixels are identical with and without it."""
    augment = images.image_augmentations("flip_only")
    image = _Images(1)[0]["image"]

    flipped = images.augment_image(augment, image, np.random.default_rng(0))

    assert np.array_equal(np.sort(flipped.ravel()), np.sort(image.ravel()))
    assert flipped.shape == image.shape


def test_the_jitter_stays_inside_the_brightness_envelope():
    """On a constant grey image the output ratio is bounded by the widest
    brightness*contrast product the draws allow, plus one uint8 of rounding;
    a factor outside [0.8, 1.2]x[0.95, 1.05] would show here."""
    augment = images.image_augmentations("flip_jitter")
    grey = np.full((8, 8, 3), 100, np.uint8)
    low, high = 0.8 * 0.95, 1.2 * 1.05

    for seed in range(64):
        out = images.augment_image(augment, grey, np.random.default_rng(seed))
        ratio = out.astype(np.float64) / 100
        assert ratio.min() >= low - 0.01 and ratio.max() <= high + 0.01, seed


@pytest.mark.parametrize("worker_count", [0, pytest.param(2, marks=pytest.mark.slow)])
def test_an_interrupted_epoch_resumes_on_exactly_the_records_it_had_not_seen(
        worker_count):
    """The trainer saves the iterator's position in its checkpoint, so a
    restored run owes the epoch its unseen records, no more and no fewer.

    The loader is built again from a source object of its own, as a resumed
    process has: the position is a record count into a named order, and a
    source described by its address names a different order in every process.

    Eight training records at a batch of four is two whole batches, and the
    iterator batches ahead of the worker slice so that holds at every worker
    count.
    """
    def loader():
        return Augmenting(image_size=8, seed=3, val_batches=2,
                          loading=Loading(workers=worker_count, threads=1, read_buffer=1,
                                          worker_buffer=1)).load(
        batch=4, tokenize=keep_captions)

    interrupted = loader().train(DataPartition())
    seen = _rows(next(interrupted))
    state = interrupted.get_state()
    rest = [row for batch in itertools.islice(interrupted, 3) for row in _rows(batch)]

    restored = loader().train(DataPartition())
    restored.set_state(state)
    resumed = [row for batch in itertools.islice(restored, 3) for row in _rows(batch)]

    assert "object at 0x" not in json.loads(state)[ENVELOPE]["order"], (
        "a source described by its address can only be restored in the process "
        "that saved it")
    assert resumed == rest, "a resumed epoch owes the same records, augmented alike"
    assert sorted(index for index, _, _ in seen + rest[:4]) == list(range(8, 16))


_DEFAULT_VALIDATED_PARTITION = DataPartition()


def _validated(length, val_batches, batch, partition=_DEFAULT_VALIDATED_PARTITION, **read):
    """{record index: (pixels, caption)} for one share's validation pass."""
    data = Augmenting(length=length, image_size=8, seed=3, val_batches=val_batches,
                      loading=Loading(workers=0, worker_buffer=1, **read)).load(
        batch=batch, tokenize=keep_captions)
    return {index: (pixels, caption)
            for b in data.val(partition) for index, pixels, caption in _rows(b)}


def test_validation_pixels_do_not_depend_on_the_read_thread_count():
    """The validation pass transforms its records inside grain's prefetch
    threads, so a record's augmentation has to come only from its per-record
    rng: keyed by the thread's call order instead, one record's seed would
    reach another record's pixels. Every draw here is record-keyed, and the
    captions come from the same rng directly."""
    serial = _validated(512, 64, 4, threads=1, read_buffer=1)
    threaded = _validated(512, 64, 4, threads=32, read_buffer=128)

    assert sorted(serial) == list(range(256))
    assert threaded == serial


@pytest.mark.parametrize("shares", [2, 8])
def test_a_validation_record_does_not_depend_on_the_share_count(shares):
    """Share p of n validates records p, p + n, ... of the split. The rng
    behind a record's flip, jitter and caption has to be keyed by its place in
    the split, not in that slice, or the same seed validates one record with
    one augmentation on a single host and another on a pod: keyed by the
    slice, every record but the first differed at two processes."""
    alone = _validated(64, 4, 8, threads=1, read_buffer=1)

    together = {}
    for index in range(shares):
        together.update(_validated(64, 4, 8, DataPartition(index, shares), threads=1, read_buffer=1))

    assert sorted(alone) == list(range(32))
    assert together == alone


# ---------------------------------------------------------------------------------
# Preparing a dataset at training resolution
# ---------------------------------------------------------------------------------

def test_a_prepared_set_is_the_deterministic_half_of_the_transform(tmp_path):
    """prepare_images.py writes exactly what ImageTransform with
    augmentation='none' produces: the same decode and resize, keyed captions
    and labels, so a prepared record read back through ArrayRecordImages is
    indistinguishable from the live transform's output."""
    from tools.prepare_images import prepare

    spec = Augmenting(length=16, image_size=8, augmentation="none",
                      val_batches=None, **WORKERS)
    manifest = prepare(spec, str(tmp_path), shards=2,
                       source={"dataset": "augmenting"})
    assert manifest["records"] == 16 and manifest["image_size"] == 8
    assert len(manifest["shard_sizes"]) == 2

    data = images.ArrayRecordImages(path=str(tmp_path), image_size=8,
                                    augmentation="none", val_batches=None,
                                    **WORKERS).load(batch=4, tokenize=keep_captions)
    seen = {}
    for batch in itertools.islice(data.train(DataPartition()), data.steps_per_epoch):
        for index, pixels, caption in _rows(batch):
            seen[index] = (pixels, caption)
    assert sorted(seen) == list(range(16))

    live = _Images(16)
    for index in range(16):
        image, caption, _ = spec.record(live[index], np.random.default_rng(index))
        expected = images.resize_image(np.ascontiguousarray(image), 8)
        pixels, stored_caption = seen[index]
        assert pixels == expected.tobytes(), f"record {index}'s pixels differ"
        assert np.asarray(image).shape == (12, 12, 3)
        assert stored_caption == caption, f"record {index}'s caption differed"
        assert np.frombuffer(pixels, np.uint8).shape == (8 * 8 * 3,)


def test_resize_image_returns_an_image_already_at_size():
    """A prepared record arrives at training size already; resizing it must
    not spend a cv2.copy on every record of every batch."""
    image = np.zeros((8, 8, 3), np.uint8)
    assert images.resize_image(image, 8) is image


# ---------------------------------------------------------------------------------
# Failure paths: a record that cannot be read stops the run
# ---------------------------------------------------------------------------------

class _Raising:
    """Raises on record `bad`, or on every record when `bad` is None."""

    def __init__(self, length, bad):
        self.length = length
        self.bad = bad

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.bad is None or index == self.bad:
            raise RuntimeError(f"record {index} is unreadable")
        return {"index": index}


@dataclasses.dataclass(frozen=True)
class Raising(Indexed):
    bad: int | None = None

    def source(self):
        return _Raising(self.length, self.bad)


@pytest.mark.parametrize("worker_count", [0, pytest.param(2, marks=pytest.mark.slow)])
def test_a_record_that_cannot_be_read_stops_the_stream(worker_count):
    """The source's own error has to reach the trainer. A pipeline that caught
    it would train on whatever it substituted, and a worker's exception is the
    easiest one to lose."""
    data = Raising(length=8, bad=3, loading=Loading(workers=worker_count)).load(batch=2)

    delivered = []
    with pytest.raises(RuntimeError, match="record 3 is unreadable"):
        for batch in data.train(DataPartition()):
            delivered.extend(int(i) for i in batch["index"])

    assert 3 not in delivered
    assert all(index in range(8) for index in delivered), "no fabricated records"


def test_a_source_that_fails_on_every_record_raises_instead_of_an_empty_batch():
    data = Raising(length=8, bad=None).load(batch=2)

    with pytest.raises(RuntimeError, match="is unreadable"):
        next(data.train(DataPartition()))


# ---------------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------------

def test_decoding_gives_a_grayscale_record_three_channels():
    """A single-channel record has to come out the same shape as every other
    one, or the batch it lands in cannot be stacked."""
    gray = np.tile(np.arange(0, 128, 8, dtype=np.uint8), (16, 1))
    encoded = cv2.imencode(".png", gray)[1].tobytes()

    out = decode_image(encoded)

    assert out.shape == (16, 16, 3)
    np.testing.assert_array_equal(out[..., 0], gray)
    np.testing.assert_array_equal(out[..., 0], out[..., 2])


@pytest.mark.parametrize("at_least", [None, 8])
def test_decoding_composites_transparency_onto_white(at_least):
    """c * a / 255 + 255 - a per channel, rounded: a transparent pixel is
    white and an opaque one keeps its colour, at any requested scale."""
    bgra = np.random.RandomState(0).randint(0, 256, (32, 32, 4), np.uint8)
    bgra[0, 0, 3], bgra[0, 1, 3] = 0, 255
    encoded = cv2.imencode(".png", bgra)[1].tobytes()
    alpha = bgra[..., 3:].astype(np.float64)
    expected = np.rint(bgra[..., 2::-1] * alpha / 255 + 255 - alpha).astype(np.uint8)

    out = decode_image(encoded, at_least=at_least)

    np.testing.assert_array_equal(out, expected)


def test_a_sixteen_bit_png_decodes_to_its_high_byte():
    """Every record is uint8 whatever depth it was stored at, or a 16-bit
    one lands in the batch at 256 times the scale of the rest."""
    bgr = np.random.RandomState(0).randint(0, 65536, (4, 5, 3)).astype(np.uint16)
    encoded = cv2.imencode(".png", bgr)[1].tobytes()

    out = decode_image(encoded)

    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, (bgr[..., ::-1] >> 8).astype(np.uint8))


def test_a_truncated_image_raises_rather_than_becoming_an_array():
    """cv2.imdecode hands back None for a half-written jpeg, and None resized
    to the training size would be a black record; the decoder raises a
    ValueError ("cv2 could not decode ...") before the colour conversion sees it."""
    # Large enough that the half kept still holds the whole header.
    whole = np.random.RandomState(0).randint(0, 256, (64, 64, 3), np.uint8)
    encoded, buffer = cv2.imencode(".jpg", whole)
    assert encoded
    truncated = buffer.tobytes()[:len(buffer) // 2]

    with pytest.raises(ValueError, match="could not decode"):
        decode_image(truncated)


def test_the_image_transform_resizes_augments_and_captions_one_record():
    """What every image dataset hands the loader: the resized uint8 image,
    the caption text and, for a class-labelled record, its index."""
    spec = Augmenting(image_size=8, augmentation="none")
    element = _Images(16)[5]

    out = ImageTransform(spec).random_map(element, np.random.default_rng(0))

    np.testing.assert_array_equal(
        out["image"], cv2.resize(element["image"], (8, 8), interpolation=cv2.INTER_AREA))
    assert out["caption"] in {template.format("rose") for template in CAPTION_TEMPLATES}
    assert out["label"] == 5 and out["label"].dtype == np.int32


def test_resizing_interpolates_up_and_averages_down():
    """`resize_image` interpolates by direction: area down, cubic up. Picking
    by the target size alone (cubic above 256, area below) sends a small
    source going up to 128 into nearest-neighbour blocks and keeps every
    third pixel of a fine pattern in a large one going down to 300."""
    board = np.zeros((4, 4, 3), np.uint8)
    board[::2, ::2] = board[1::2, 1::2] = 255
    up = images.resize_image(board, 8)
    assert ((up > 0) & (up < 255)).any(), "cubic blends between the squares"

    fine = np.zeros((900, 900, 3), np.uint8)
    fine[::2, ::2] = fine[1::2, 1::2] = 255
    down = images.resize_image(fine, 300)
    assert down.min() >= 100 and down.max() <= 160, "area averages the squares it covers"


# ---------------------------------------------------------------------------------
# Whose tokenizer: the run's condition, not the dataset
# ---------------------------------------------------------------------------------

def test_a_batch_carries_captions_and_each_encoder_tokenizes_them_its_own_way():
    """The dataset writes the words; how many ids they become is the run's
    condition. The same dataset, read once per encoder, gives CLIP's context
    and the char table's eight, and neither encoder is named in the data
    pipeline."""
    from dew.inputs import CharTable, Condition, Field, InputSpec

    spec = Augmenting(length=8, image_size=8, val_batches=None, **WORKERS)
    captions = next(spec.load(batch=4, tokenize=keep_captions).train(DataPartition()))["caption"]
    assert captions.shape == (4,)

    def tokens(encoder):
        inputs = InputSpec(Field("image", (8, 8, 3)),
                           {"textcontext": Condition(encoder, field="text")})
        batch = next(spec.load(batch=4, tokenize=inputs.tokenize).train(DataPartition()))
        assert "caption" not in batch, "strings cannot ride a batch onto a device"
        return batch["text"]["input_ids"].shape

    wide = CharTable.from_pretrained(tokens=77)
    narrow = CharTable.from_pretrained(tokens=8)
    assert tokens(wide) == (4, 77)
    assert tokens(narrow) == (4, 8)


def test_a_run_with_no_condition_leaves_no_captions_in_the_batch():
    """An unconditional run reads nothing out of the captions, and the words
    stop at the loader: a string array cannot be placed on a device."""
    from dew.inputs import Field, InputSpec

    spec = Augmenting(length=8, image_size=8, val_batches=None, **WORKERS)
    inputs = InputSpec(Field("image", (8, 8, 3)))

    batch = next(spec.load(batch=4, tokenize=inputs.tokenize).train(DataPartition()))

    assert sorted(batch) == ["image", "label"]


def test_a_jpeg_decodes_at_the_coarsest_scale_that_still_covers_the_target():
    """The DCT reduction is the largest of 8, 4, 2 that keeps both sides at
    or above the target, so the resize after it only ever shrinks; a target
    the image cannot cover at any reduction decodes at full size."""
    rows = np.random.RandomState(0).randint(0, 256, (400, 600, 3), np.uint8)
    encoded = cv2.imencode(".jpg", rows)[1].tobytes()

    assert decode_image(encoded, at_least=50).shape[:2] == (50, 75)
    assert decode_image(encoded, at_least=64).shape[:2] == (100, 150)
    assert decode_image(encoded, at_least=128).shape[:2] == (200, 300)
    assert decode_image(encoded, at_least=256).shape[:2] == (400, 600)
    assert decode_image(encoded).shape[:2] == (400, 600)


def test_a_reduced_decode_keeps_the_orientation_the_pixels_are_stored_in():
    """A JPEG's EXIF rotation is ignored at full size and at a reduced
    scale alike, as PIL's `Image.open` ignores it."""
    import io

    import PIL.Image

    image = PIL.Image.fromarray(np.random.RandomState(0).randint(0, 256, (32, 64, 3), np.uint8))
    exif = image.getexif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes())
    encoded = buffer.getvalue()

    with PIL.Image.open(io.BytesIO(encoded)) as stored:
        width, height = stored.size
    assert decode_image(encoded).shape[:2] == (height, width)
    assert decode_image(encoded, at_least=16).shape[:2] == (height // 2, width // 2)


# ---------------------------------------------------------------------------------
# A run over grain datasets the caller built
# ---------------------------------------------------------------------------------

class _Points:
    """`length` points whose target is twice the input, addressed by index."""

    def __init__(self, length):
        self.length = length

    def __repr__(self):
        return f"_Points({self.length})"

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = np.full((3,), index, np.float32) / self.length
        return {"x": x, "y": 2 * x[:2], "index": np.int32(index)}


def _grain_points(length=16):
    """The pipeline a caller writes themselves: a source, shuffled."""
    return pygrain.MapDataset.source(_Points(length)).seed(0).shuffle(0)


def test_from_grain_batches_a_map_dataset_per_process_and_counts_its_records():
    data = Dataset.from_grain(_grain_points(), batch=4, **WORKERS)

    assert data.records == 16 and data.batch == 4 and data.steps_per_epoch == 4
    batch = next(data.train(DataPartition()))
    assert batch["x"].shape == (4, 3) and batch["index"].shape == (4,)


def test_from_grain_repeats_a_map_dataset_so_a_run_outlasts_the_corpus():
    """`Dataset.train` is endless; a caller's finite pipeline would otherwise
    stop the run one pass in."""
    stream = Dataset.from_grain(_grain_points(8), batch=4, **WORKERS).train(DataPartition())

    assert len(_indices(stream, 6)) == 6


def test_a_map_dataset_streams_and_resumes_as_grains_own_pipeline_reads_it():
    """A caller's shuffled `MapDataset` reads, batch for batch, as grain's
    own `repeat(None).batch(4)` of it, across the end of a pass, and a
    stream resumed from a saved position reads on where grain's batches do."""
    pipeline = _grain_points(10)
    reference = pipeline.repeat(None).batch(4)
    expected = [[int(i) for i in reference[index]["index"]] for index in range(6)]
    data = Dataset.from_grain(pipeline, batch=4, **WORKERS)

    stream = data.train(DataPartition())
    assert _indices(stream, 3) == expected[:3]
    state = stream.get_state()
    resumed = data.train(DataPartition())
    resumed.set_state(state)
    assert _indices(resumed, 3) == expected[3:]


def test_from_grain_takes_a_streamed_pipeline_and_carries_grains_own_state():
    """An IterDataset has no index, so the caller builds the one a share
    reads, it is batched where it is and it reports the position grain keeps
    for it."""
    def rows(partition):
        return pygrain.MapDataset.source(_Points(8))[partition.index::partition.count].repeat(
            None).to_iter_dataset()

    data = Dataset.from_grain(rows, batch=4, records=8, **WORKERS)

    stream = data.train(DataPartition())
    assert isinstance(stream, Checkpointable)
    first = _indices(stream, 1)
    state = stream.get_state()
    rest = _indices(stream, 1)

    resumed = data.train(DataPartition())
    resumed.set_state(state)
    assert _indices(resumed, 1) == rest != first


def test_from_grain_scores_a_validation_pass_that_ends():
    data = Dataset.from_grain(_grain_points(16), batch=4,
                              validation=pygrain.MapDataset.source(_Points(8)),
                              **WORKERS)

    assert data.val is not None
    assert _indices(data.val(DataPartition()), 5) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_a_run_over_a_grain_dataset_trains_and_resumes_where_it_stopped(tmp_path):
    """The seam is only worth having if the trainer treats it as any other
    dataset: a run over it checkpoints its iterator position, and a resume
    reads the batches after the checkpoint rather than replaying them."""
    import optax
    from flax import linen as nn

    from dew.objectives.base import Aux, Objective
    from dew.training import Checkpoints, Layout, Trainer

    class Regression(Objective):
        """Squared error of an affine map against twice its input."""

        def __init__(self):
            self.model = nn.Dense(2)

        def init(self, key, variables=None):
            return self.model.init(key, jnp.zeros((1, 3)))

        def loss(self, variables, batch, step):
            return jnp.mean((self.model.apply(variables, batch["x"]) - batch["y"]) ** 2), Aux({})

    def run(steps, directory=None):
        trainer = Trainer(
            Regression(), optax.sgd(0.1), key=jax.random.key(0),
            layout=Layout(min_shard=1, tolerance=1.0),
            checkpoints=None if directory is None else Checkpoints(str(directory)))
        # The simulated mesh is eight devices wide, so a step is eight records.
        return trainer.fit(Dataset.from_grain(_grain_points(), batch=8, **WORKERS),
                           steps=steps, log_every=1,
                           checkpoint_every=None if directory is None else 2)

    run(2, tmp_path / "run")
    resumed = run(4, tmp_path / "run")
    whole = run(4)

    for expected, actual in zip(jax.tree.leaves(whole.variables),
                                jax.tree.leaves(resumed.variables), strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6)


# ---------------------------------------------------------------------------------
# Records held in memory, and where a default run reads
# ---------------------------------------------------------------------------------

def _pid_of(index):
    """One record that says which process read it."""
    return {"index": np.int32(index), "pid": np.int32(os.getpid())}


def test_a_default_loading_reads_in_the_training_process():
    """Grain's default is to read in the process that trains. A default that
    started 32 worker processes took 17.8 s and 7.07 GiB to the first batch
    of a 100-record pipeline, and a script without a __main__ guard re-ran
    itself in every worker."""
    data = Dataset.from_grain(pygrain.MapDataset.range(16).map(_pid_of), batch=4)

    stream = data.train(DataPartition())
    try:
        batch = next(stream)
    finally:
        stream.close()
    assert set(batch["pid"].tolist()) == {os.getpid()}


def _columns(length=12):
    index = np.arange(length, dtype=np.int32)
    return {"index": index, "x": np.stack([index, -index], axis=1).astype(np.float32)}


def test_records_in_memory_are_reshuffled_every_epoch_and_each_read_once_per_epoch():
    stream = Dataset.from_records(_columns(), batch=4, seed=3).train(DataPartition())

    epochs = [list(itertools.chain.from_iterable(_indices(stream, 3))) for _ in range(3)]
    for epoch in epochs:
        assert sorted(epoch) == list(range(12)), "each record once per epoch"
    assert len({tuple(epoch) for epoch in epochs}) == 3, "a fresh order every epoch"
    assert epochs[0] != list(range(12)), "the first epoch is shuffled too"


def test_records_in_memory_follow_their_seed():
    def order(seed):
        return _indices(Dataset.from_records(_columns(), batch=4, seed=seed).train(DataPartition()), 6)

    assert order(0) == order(0)
    assert order(0) != order(1)


def test_columns_rows_and_a_torch_dataset_are_the_same_records():
    import torch

    columns = _columns()
    rows = [{"index": columns["index"][i], "x": columns["x"][i]} for i in range(12)]
    tensors = torch.utils.data.TensorDataset(torch.arange(12), torch.from_numpy(columns["x"]))

    first, *others = (next(data.train(DataPartition())) for data in (
        Dataset.from_records(columns, batch=4, seed=0), Dataset.from_records(rows, batch=4, seed=0),
        Dataset.from_torch(tensors, batch=4, fields=("index", "x"), seed=0)))

    assert first["x"].shape == (4, 2) and first["x"].dtype == np.float32
    for other in others:
        for name in ("index", "x"):
            assert other[name].dtype == first[name].dtype
            np.testing.assert_array_equal(first[name], other[name])


def test_a_torchvision_style_dataset_reads_its_tuples_as_named_fields():
    """`(PIL image, label)` is torchvision's sample; a DataLoader's own
    sampler and collate_fn would be dropped by reading its dataset, so it is
    refused rather than unwrapped."""
    import torch
    from PIL import Image

    class Pictures(torch.utils.data.Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, index):
            return Image.fromarray(np.full((5, 5, 3), index, np.uint8)), index

    batch = next(Dataset.from_torch(Pictures(), batch=4, fields=("image", "label"),
                                    validation=Pictures()).val(DataPartition()))

    assert batch["image"].shape == (4, 5, 5, 3) and batch["image"].dtype == np.uint8
    np.testing.assert_array_equal(batch["image"][:, 0, 0, 0], batch["label"])
    with pytest.raises(TypeError, match="fields="):
        Dataset.from_torch(Pictures(), batch=4)
    with pytest.raises(TypeError, match=r"loader\.dataset"):
        Dataset.from_torch(torch.utils.data.DataLoader(Pictures()), batch=4)


def test_columns_of_different_lengths_are_refused_by_name():
    with pytest.raises(ValueError, match="'x' holds 3 records and 'index' holds 4"):
        Dataset.from_records({"index": np.arange(4), "x": np.zeros((3, 2))}, batch=2)


def test_an_in_memory_stream_resumes_on_the_records_it_had_not_read():
    data = Dataset.from_records(_columns(), batch=4, seed=0)
    stream = data.train(DataPartition())
    _indices(stream, 2)
    state = stream.get_state()
    rest = _indices(stream, 4)

    resumed = data.train(DataPartition())
    resumed.set_state(state)
    assert _indices(resumed, 4) == rest


def test_process_shares_of_in_memory_records_split_every_global_batch():
    data = Dataset.from_records(_columns(), batch=4, seed=0)
    whole = _indices(data.train(DataPartition()), 3)
    halves = [_indices(data.train(DataPartition(index=index, count=2)), 3) for index in (0, 1)]

    for step, rows in enumerate(whole):
        first, second = halves[0][step], halves[1][step]
        assert not set(first) & set(second), "the shares are disjoint"
        assert sorted(first + second) == sorted(rows), "together they are the global batch"


def test_held_out_records_are_one_ordered_pass():
    data = Dataset.from_records(_columns(), batch=4, validation=_columns(8))

    assert data.records == 12 and data.steps_per_epoch == 3
    assert data.val is not None
    assert _indices(data.val(DataPartition()), 5) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_held_out_records_too_few_for_one_batch_are_one_filled_batch():
    """Three records at a batch of four are one batch, its fourth row a repeat."""
    batches = list(Dataset.from_records(_columns(), batch=4, validation=_columns(3)).val(DataPartition()))

    assert len(batches) == 1
    np.testing.assert_array_equal(batches[0][VALID_ROWS], [True, True, True, False])


def test_a_validation_split_needs_no_training_records():
    """A second held-out split, beside a dataset's own, is the same ordered
    pass over its records, a short last batch filled and marked."""
    batches = list(Dataset.validation(_columns(5), batch=4)(DataPartition()))

    marked = [np.asarray(batch.get(VALID_ROWS, np.ones(4, bool))).tolist() for batch in batches]
    assert marked == [[True] * 4, [True, False, False, False]]
    assert _indices(Dataset.validation(_columns(5), batch=4)(DataPartition()), 1) == [[0, 1, 2, 3]]


def test_records_whose_field_lengths_differ_are_refused_with_the_remedy():
    """Token ids of varying length are the usual cause, and grain's own
    message names the batch structure rather than what to change."""
    rows = [{"text": np.arange(length, dtype=np.int32)} for length in (3, 5, 4, 6)]
    stream = Dataset.from_records(rows, batch=2).train(DataPartition())

    with pytest.raises(ValueError, match="cut or pad a variable-length field") as refused:
        next(stream)
    assert "same structure" in str(refused.value.__cause__)


def test_training_records_too_few_for_one_batch_are_refused():
    """An endless stream would fill a batch by repeating records inside it,
    and an epoch would be zero steps long."""
    with pytest.raises(ValueError, match="3 training records, fewer than one batch of 4"):
        Dataset.from_records(_columns(3), batch=4)


def test_a_run_over_records_in_memory_checkpoints_and_resumes_where_it_stopped(tmp_path):
    """The in-memory route exists so a first run can checkpoint. A resumed run
    reads the records after the checkpoint and ends where an uninterrupted
    run ends."""
    import optax
    from flax import linen as nn

    from dew.objectives.base import Aux, Objective
    from dew.training import Checkpoints, Layout, Trainer

    class Regression(Objective):
        def __init__(self):
            self.model = nn.Dense(1)

        def init(self, key, variables=None):
            return self.model.init(key, jnp.zeros((1, 2)))

        def loss(self, variables, batch, step):
            target = batch["index"][:, None].astype(jnp.float32)
            return jnp.mean((self.model.apply(variables, batch["x"]) - target) ** 2), Aux({})

    def run(steps, directory=None):
        trainer = Trainer(
            Regression(), optax.sgd(0.01), key=jax.random.key(0),
            layout=Layout(min_shard=1, tolerance=1.0),
            checkpoints=None if directory is None else Checkpoints(str(directory)))
        return trainer.fit(Dataset.from_records(_columns(16), batch=8, seed=0),
                           steps=steps, log_every=1,
                           checkpoint_every=None if directory is None else 2)

    run(2, tmp_path / "run")
    resumed = run(5, tmp_path / "run")
    whole = run(5)

    for expected, actual in zip(jax.tree.leaves(whole.variables),
                                jax.tree.leaves(resumed.variables), strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6)


def test_a_fit_over_a_dataset_that_held_validation_out_says_so_once(capsys):
    import optax
    from flax import linen as nn

    from dew.objectives.base import Aux, Objective
    from dew.training import Layout, Trainer

    class Regression(Objective):
        def __init__(self):
            self.model = nn.Dense(1)

        def init(self, key, variables=None):
            return self.model.init(key, jnp.zeros((1, 2)))

        def loss(self, variables, batch, step):
            return jnp.mean(self.model.apply(variables, batch["x"]) ** 2), Aux({})

    trainer = Trainer(Regression(), optax.sgd(0.01), key=jax.random.key(0),
                      layout=Layout(min_shard=1, tolerance=1.0))
    data = Dataset.from_records(_columns(16), batch=8)

    trainer.fit(dataclasses.replace(data, held_out=24), steps=2, log_every=1)
    noted = capsys.readouterr().out
    trainer.fit(data, steps=2, log_every=1)

    assert noted.count("validation: 24 records held out of train") == 1
    assert "held out" not in capsys.readouterr().out
