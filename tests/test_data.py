"""Data layer tests: the registry, the Dataset contract, lazy imports, the
video specs and the determinism of what a record becomes.

`import dew.data` and the grain paths must not need the optional extras,
which is the point of several of these tests. Anything that genuinely
requires an optional dependency skips.
"""

import dataclasses
import inspect
import itertools
import json
import os
import sys

import cv2
import grain.python as pygrain
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dew.data
from dew.data import (
    Checkpointable,
    Dataset,
    DatasetSpec,
    HFDatasetSource,
    ImageDataset,
    Loading,
    LocalVideos,
    VoxCeleb2,
    images,
    local_batch,
    video,
)
from dew.data.dataset import Forwarding, _batches, hold_out, train_stream, validation_pass
from dew.data.images import ImageTransform, decode_image
from dew.data.sources import av_utils
from dew.data.sources.av_utils import choose_clip_start
from dew.position import ENVELOPE
from dew.registry import datasets

WORKERS = dict(loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))


# ---------------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------------

def test_an_unknown_dataset_is_refused():
    with pytest.raises(KeyError, match="no dataset named 'flowers'"):
        datasets["flowers"]


def test_a_spec_field_the_dataset_has_no_declaration_for_is_refused():
    """A misspelled knob built a dataset other than the one asked for."""
    with pytest.raises(ValueError, match=r"no field for \['image_scale'\]"):
        datasets.build("oxford_flowers102", image_scale=64)
    assert datasets.build("oxford_flowers102", image_size=64).image_size == 64


@pytest.mark.parametrize("name", ["cc12m", "combined_30m"])
def test_arrayrecord_datasets_require_an_explicit_path(name):
    """The default was one developer's bucket mount, and an unset path reached
    os.path.join(None, ...) inside the source."""
    with pytest.raises(ValueError, match="path="):
        datasets[name]().load(batch=8)


def test_an_unknown_augmentation_is_refused():
    with pytest.raises(ValueError, match="not one of none, flip_only, flip_jitter"):
        images.image_augmentations("jitter")


def test_reading_a_hub_dataset_names_the_streaming_extra(monkeypatch):
    """Naming one works anywhere; the first record is what needs HF datasets."""
    source = HFDatasetSource(name="acme/pets")
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError, match=r"dew-ml\[streaming\]"):
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
    assert next(data.train())["text"].shape == (2, 9)


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
    loading: Loading = Loading(workers=0)

    def source(self):
        return _Indexed(self.length)

    def load(self, *, batch):
        source = self.source()
        records = len(source) if self.count is None else self.count
        train, val = hold_out(source, records, (self.val_batches or 0) * batch, "Indexed")
        knobs = dict(batch=local_batch(batch), seed=self.seed,
                     loading=self.loading)
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
    val_batches = _indices(data.val(), 2)
    assert val_batches == [list(range(8))]
    assert _indices(data.val(), 2) == val_batches

    train_indices = [i for batch in _indices(data.train(), 3) for i in batch]
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

    batches, ended = _bounded(data.val(), 12)
    assert [[int(i) for i in b["index"]] for b in batches] == [
        list(range(8)), list(range(8, 16)), list(range(16, 24))]
    assert ended
    train = [index for batch in _indices(data.train(), 2) for index in batch]
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
    val = _labels(data.val())
    train = _labels(data.train(), batches=4)
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
    assert len(_labels(bounded_pass.val(), batches=8)) == 4
    assert _labels(whole_pass.val(), batches=8) == list(range(1000, 1008))


def test_without_a_named_split_the_head_hold_out_is_unchanged():
    """The default path has to stay exactly what it was: the same records,
    the same pixels, the same training count."""
    spec = Augmenting(length=16, image_size=8, augmentation="none", val_batches=1,
                      **WORKERS)

    data = spec.load(batch=4, tokenize=keep_captions)

    assert data.records == 12 and data.steps_per_epoch == 3
    held = next(data.val())
    assert _labels(iter([held])) == [0, 1, 2, 3]
    assert set(_labels(data.train(), batches=3)).isdisjoint(range(4))
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
    assert sorted(int(i) for batch in itertools.islice(data.train(), 2)
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
    assert sorted(i for batch in _indices(data.train(), 2) for i in batch) == list(range(16))


def test_the_training_stream_repeats_instead_of_ending():
    """The trainer keeps asking for batches long after one pass over the
    records, and the next pass is the same records in another order."""
    data = Indexed(val_batches=2).load(batch=8)

    epoch = data.steps_per_epoch  # the sixteen training records, in two batches
    batches, ended = _bounded(data.train(), 3 * epoch)
    first_epoch = [int(i) for batch in batches[:epoch] for i in batch["index"]]
    later = [int(i) for batch in batches[epoch:] for i in batch["index"]]

    assert not ended and len(batches) == 3 * epoch
    assert sorted(first_epoch) == list(range(16, 32))
    assert set(later) == set(first_epoch), "the stream reads the same records again"


def test_a_dataset_of_one_record_yields_batches_of_it():
    data = Indexed(length=1).load(batch=1)
    assert _indices(data.train(), 2) == [[0], [0]]


def test_the_training_iterator_carries_its_position():
    """A checkpoint records the iterator's state and a restored run resumes
    on the batch after it."""
    data = Indexed().load(batch=8)
    first = data.train()
    # The trainer reads the protocol statically, and a caption stage that only
    # forwarded get_state through __getattr__ failed it while hasattr passed.
    assert isinstance(first, Checkpointable)
    seen = _indices(first, 2)
    state = first.get_state()
    rest = _indices(first, 2)

    resumed = data.train()
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
    stream = _batches(_order(source, seed=5), batch=4, loading=Loading(
        workers=workers, threads=2, read_buffer=4, worker_buffer=2))

    read = _fields(stream, 12)
    stream.close()

    assert read == _by_hand(_order(source, seed=5), batch=4, count=12)
    assert len({tuple(batch["index"]) for batch in read}) == 12, (
        "twelve batches of four is more than one pass over thirty records, "
        "so a reshuffled epoch is inside this comparison")


@pytest.mark.parametrize("workers", [0, pytest.param(2, marks=pytest.mark.slow)])
def test_a_pass_over_a_split_the_workers_do_not_divide_is_still_whole_batches(workers):
    """Thirty records at a batch of four is seven whole batches and two
    records over. Two workers take a batch each in turn, so the seventh batch
    is one worker's alone and the last round is half empty: it still arrives,
    in its place, and what is dropped is the two records over."""
    passes = validation_pass(_Indexed(30), [], batch=4, seed=0, loading=Loading(
        workers=workers, threads=2, read_buffer=4, worker_buffer=2))

    batches, ended = _bounded(passes(), 12)

    assert [[int(index) for index in batch["index"]] for batch in batches] == [
        list(range(start, start + 4)) for start in range(0, 28, 4)]
    assert ended


@pytest.mark.parametrize("workers", [0, pytest.param(2, marks=pytest.mark.slow)])
def test_an_offset_stays_a_bound_on_the_slice_this_process_reads(workers):
    """A resume opens the stream `offset` records in, and the permutation the
    workers read through sits behind that slice, so the first batch is still
    the first four records the interrupted run had not reached."""
    stream = _batches(_order(_Indexed(30)), batch=4, offset=8, loading=Loading(
        workers=workers, threads=2, read_buffer=4, worker_buffer=2))

    first = _fields(stream, 1)
    stream.close()

    assert first == [{"index": [8, 9, 10, 11]}]


# ---------------------------------------------------------------------------------
# The global batch over JAX processes
# ---------------------------------------------------------------------------------

def test_a_global_batch_that_does_not_split_over_the_processes_is_refused(monkeypatch):
    """Integer division hid the remainder: 65 over eight processes trained on
    64 records a step while the run reported 65, and 7 gave every process a
    batch of nothing."""
    monkeypatch.setattr(jax, "process_count", lambda: 8)

    for batch in (65, 7):
        with pytest.raises(ValueError, match=rf"batch {batch} does not split over 8 JAX processes"):
            local_batch(batch)
        with pytest.raises(ValueError, match="8 JAX processes"):
            Indexed(length=256).load(batch=batch)
    assert local_batch(64) == 8
    assert Indexed(length=256).load(batch=64).batch == 64


def test_each_process_reads_its_own_slice_of_the_validation_split(monkeypatch):
    """Process p of n validates records p, p + n, ... of the split, in whole
    batches of the per-process size."""
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    monkeypatch.setattr(jax, "process_index", lambda: 1)

    data = Indexed(val_batches=2).load(batch=8)

    assert _indices(data.val(), 3) == [[1, 3, 5, 7], [9, 11, 13, 15]]


def _steps(monkeypatch, processes, count, *, position=None, length=32, seed=0, batch=8):
    """`(global batches, saved positions)` for `processes` simulated processes.

    A global batch is the step every process contributes its rows to, so the
    records are collected across the processes; which process holds which row
    is the mesh's business and which records a step trains on is the
    dataset's. Each process is opened in turn, resumed from `position` when
    one is given, and asked where it stopped.
    """
    monkeypatch.setattr(jax, "process_count", lambda: processes)
    read, saved = [], []
    for index in range(processes):
        monkeypatch.setattr(jax, "process_index", lambda index=index: index)
        stream = Indexed(length=length, seed=seed).load(batch=batch).train
        iterator = stream()
        if position is not None:
            iterator.set_state(position)
        read.append(_indices(iterator, count))
        saved.append(iterator.get_state())
        iterator.close()
    return [sorted(index for rows in read for index in rows[step])
            for step in range(count)], saved


@pytest.mark.parametrize("processes", [1, 2, 4])
def test_a_training_step_reads_the_same_records_at_every_process_count(
        monkeypatch, processes):
    """Step k is records [k * batch, (k + 1) * batch) of one order whatever
    the process count is, because the iterator owns the sharding and the
    batching together. Shuffling the corpus per shard first, as an index
    sampler does, gave each count a different order, which no encoding of a
    position could have translated."""
    alone, _ = _steps(monkeypatch, 1, 4)
    together, _ = _steps(monkeypatch, processes, 4)

    assert together == alone
    assert sorted(index for step in alone for index in step) == list(range(32)), (
        "four steps of eight is one pass over the corpus, each record once")


def test_every_process_saves_the_same_global_position(monkeypatch):
    """A position is a record count over the whole run's order, so every
    process reports the same bytes; that is what lets a checkpoint written by
    two processes be handed to one or to four."""
    _, saved = _steps(monkeypatch, 4, 3)

    assert len(set(saved)) == 1
    assert json.loads(saved[0])[ENVELOPE]["records"] == 3 * 8


@pytest.mark.parametrize("processes", [1, 4])
def test_a_position_saved_by_two_processes_resumes_on_another_count(
        monkeypatch, processes):
    """The steps after a resume are the steps the run that was never stopped
    would have trained on, whether the resume has half the processes or twice
    them."""
    whole, _ = _steps(monkeypatch, 2, 6)
    _, saved = _steps(monkeypatch, 2, 3)

    resumed, _ = _steps(monkeypatch, processes, 3, position=saved[0])

    assert resumed == whole[3:]


def test_a_position_over_another_order_is_refused(monkeypatch):
    """A record count is a place in one order and another place in another, so
    a corpus, record count or seed the position was not written over is
    refused instead of resumed at the same offset into different data."""
    _, saved = _steps(monkeypatch, 2, 3)

    for other in (dict(length=64), dict(seed=1)):
        stream = Indexed(**other).load(batch=8).train()
        with pytest.raises(ValueError, match="records into"):
            stream.set_state(saved[0])
        stream.close()


def test_a_shard_offset_cannot_resume_a_global_stream():
    """A stream whose windows come out of one shard reports where that shard
    stopped. Handing such a position to a stream that reads its records
    globally would resume it somewhere else, so it is refused by name."""
    stream = Indexed().load(batch=8).train()

    with pytest.raises(ValueError, match="one process's own offset into its shard"):
        stream.set_state(json.dumps({"last_seen_indices": {"0": 15}}).encode())
    stream.close()


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


def test_voxceleb2_scans_the_tree_recursively(tmp_path):
    clips = _voxceleb_tree(tmp_path)
    records = VoxCeleb2(path=str(tmp_path)).source()

    assert [record["video_path"] for record in records] == sorted(str(c) for c in clips)


def test_voxceleb2_renders_captions_from_the_template(tmp_path):
    _voxceleb_tree(tmp_path)

    templated = VoxCeleb2(path=str(tmp_path), prompt_template="a video of {identity} speaking")
    captions = {record["caption"] for record in templated.source()}
    assert captions == {"a video of id00012 speaking", "a video of id00015 speaking"}

    plain = VoxCeleb2(path=str(tmp_path)).source()
    assert {record["caption"] for record in plain} == {"a video of a person speaking"}

    # A placeholder the source does not fill is a misspelling, not a caption.
    with pytest.raises(ValueError, match=r"may use \{identity\}"):
        VoxCeleb2(path=str(tmp_path), prompt_template="a video of {speaker}").source()


def test_voxceleb2_reads_the_requested_split(tmp_path):
    _voxceleb_tree(tmp_path, split="train")
    _voxceleb_tree(tmp_path, split="test", identities=("id00017",))

    train = VoxCeleb2(path=str(tmp_path), split="train").source()
    test = VoxCeleb2(path=str(tmp_path), split="test").source()
    assert len(train) == 4
    assert len(test) == 2
    assert all(os.sep + "train" + os.sep in record["video_path"] for record in train)
    assert all(os.sep + "test" + os.sep in record["video_path"] for record in test)
    assert {record["video_path"] for record in train}.isdisjoint({record["video_path"] for record in test})
    assert {record["video_path"].split(os.sep)[-3] for record in test} == {"id00017"}


def test_voxceleb2_reports_missing_roots_clearly(tmp_path):
    with pytest.raises(ValueError, match="dataset root"):
        VoxCeleb2().source()
    with pytest.raises(ValueError, match="split 'train' not found"):
        VoxCeleb2(path=str(tmp_path)).source()


def test_local_videos_lists_every_file_under_the_directory(tmp_path):
    clips = _voxceleb_tree(tmp_path)
    (tmp_path / "extra.webm").write_bytes(b"")

    records = LocalVideos(path=str(tmp_path), caption="a clip").source()

    assert [r["video_path"] for r in records] == sorted([str(c) for c in clips] + [str(tmp_path / "extra.webm")])
    assert {r["caption"] for r in records} == {"a clip"}
    with pytest.raises(ValueError, match="path="):
        LocalVideos().source()


def test_voxceleb2_records_flow_through_the_audio_video_transform(tmp_path, monkeypatch):
    """End to end with the reader stubbed and the audio model's extractor
    built here, so its weights are the only thing not real. The extractor
    takes the padded waveform whole and normalises it as one clip, which is
    what the audio condition receives; handed the rows instead, wav2vec2's
    extractor read each frame's 640 samples as a clip of its own and the
    record kept the first, a frame of padding."""
    from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor

    _voxceleb_tree(tmp_path)
    spec = VoxCeleb2(path=str(tmp_path), prompt_template="a video of {identity}",
                     frame_size=32, frames=4, audio_padding=1)
    records = spec.source()
    frame_samples = 640
    seen = {}

    def fake_read_av_random_clip(video_path, *, num_frames, audio_padding, seed):
        seen.update(video_path=video_path, num_frames=num_frames,
                    audio_padding=audio_padding, seed=seed)
        padded = num_frames + 2 * audio_padding
        audio = np.linspace(-0.5, 0.5, padded * frame_samples, dtype=np.float32)
        # Native resolution differs from frame_size, so the resize must happen
        return np.zeros((num_frames, 96, 48, 3), np.uint8), audio.reshape(padded, frame_samples)

    monkeypatch.setattr(av_utils, "read_av_random_clip", fake_read_av_random_clip)
    monkeypatch.setattr(AutoFeatureExtractor, "from_pretrained",
                        lambda name: Wav2Vec2FeatureExtractor(sampling_rate=16000))

    batch = video.AudioVideoTransform(spec).random_map(records[0], np.random.default_rng(0))

    assert seen["video_path"] == records[0]["video_path"]
    assert seen["num_frames"] == 4 and seen["audio_padding"] == 1
    assert seen["seed"] == int(np.random.default_rng(0).integers(0, 2**32 - 1))
    assert batch["video"].shape == (4, 32, 32, 3)
    assert batch["caption"] == "a video of id00012"
    waveform = batch["audio"]["full_audio"]
    assert waveform.shape == (6, frame_samples) and waveform[0, 0] == -0.5
    flat = waveform.reshape(-1)
    normalised = (flat - flat.mean()) / np.sqrt(flat.var() + 1e-7)
    assert np.allclose(batch["audio"]["input_values"], normalised, atol=1e-5)


def keep_captions(captions):
    """A caption reader that hands the words back, so a test reads what the
    dataset wrote before a run's encoder tokenizes it."""
    return {"caption": np.asarray(captions)}


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
        template = images.PROMPT_TEMPLATES[int(rng.integers(len(images.PROMPT_TEMPLATES)))]
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
            for b in itertools.islice(data.train(), data.steps_per_epoch)
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
                        loading=loading).load(batch=4, tokenize=keep_captions).train()
    assert isinstance(stream, Forwarding)
    assert stream.stop_seconds == loading.stop_seconds > 5.0
    stream.close()
    assert stream.stop_seconds is None


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

    interrupted = loader().train()
    seen = _rows(next(interrupted))
    state = interrupted.get_state()
    rest = [row for batch in itertools.islice(interrupted, 3) for row in _rows(batch)]

    restored = loader().train()
    restored.set_state(state)
    resumed = [row for batch in itertools.islice(restored, 3) for row in _rows(batch)]

    assert "object at 0x" not in json.loads(state)[ENVELOPE]["order"], (
        "a source described by its address can only be restored in the process "
        "that saved it")
    assert resumed == rest, "a resumed epoch owes the same records, augmented alike"
    assert sorted(index for index, _, _ in seen + rest[:4]) == list(range(8, 16))


def _validated(length, val_batches, batch, **read):
    """{record index: (pixels, caption)} for one validation pass."""
    data = Augmenting(length=length, image_size=8, seed=3, val_batches=val_batches,
                      loading=Loading(workers=0, worker_buffer=1, **read)).load(
        batch=batch, tokenize=keep_captions)
    return {index: (pixels, caption)
            for b in data.val() for index, pixels, caption in _rows(b)}


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


@pytest.mark.parametrize("process_count", [2, 8])
def test_a_validation_record_does_not_depend_on_the_process_count(
        monkeypatch, process_count):
    """Process p of n validates records p, p + n, ... of the split. The rng
    behind a record's flip, jitter and caption has to be keyed by its place in
    the split, not in that slice, or the same seed validates one record with
    one augmentation on a single host and another on a pod: keyed by the
    slice, every record but the first differed at two processes."""
    alone = _validated(64, 4, 8, threads=1, read_buffer=1)

    together = {}
    monkeypatch.setattr(jax, "process_count", lambda: process_count)
    for index in range(process_count):
        monkeypatch.setattr(jax, "process_index", lambda index=index: index)
        together.update(_validated(64, 4, 8, threads=1, read_buffer=1))

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
    for batch in itertools.islice(data.train(), data.steps_per_epoch):
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
        for batch in data.train():
            delivered.extend(int(i) for i in batch["index"])

    assert 3 not in delivered
    assert all(index in range(8) for index in delivered), "no fabricated records"


def test_a_source_that_fails_on_every_record_raises_instead_of_an_empty_batch():
    data = Raising(length=8, bad=None).load(batch=2)

    with pytest.raises(RuntimeError, match="is unreadable"):
        next(data.train())


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


def test_decoding_drops_alpha_and_hands_back_rgb():
    """Records arrive as BGR(A) from cv2; the model is trained on RGB."""
    bgra = np.dstack([np.full((16, 16), 10, np.uint8), np.full((16, 16), 20, np.uint8),
                      np.full((16, 16), 30, np.uint8), np.full((16, 16), 40, np.uint8)])
    encoded = cv2.imencode(".png", bgra)[1].tobytes()

    out = decode_image(encoded)

    assert out.shape == (16, 16, 3)
    np.testing.assert_array_equal(out[0, 0], [30, 20, 10])


def test_a_truncated_image_raises_rather_than_becoming_an_array():
    """cv2.imdecode hands back None for a half-written jpeg, and None resized
    to the training size would be a black record; the decoder raises a
    ValueError ("cv2 could not decode ...") before the colour conversion sees it."""
    whole = np.random.RandomState(0).randint(0, 256, (16, 16, 3), np.uint8)
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
    assert out["caption"] in {template.format("rose") for template in images.PROMPT_TEMPLATES}
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
    assert 100 <= down.min() and down.max() <= 160, "area averages the squares it covers"


@pytest.mark.network
def test_an_overlong_caption_is_truncated_to_the_text_context():
    """The tokenizer pads and truncates to CLIP's context, so one enormous
    caption cannot change the batch's shape."""
    tokenizer = dew.data.AutoTextTokenizer(tensor_type="np")
    context = tokenizer.tokenizer.model_max_length

    out = tokenizer(["short", " ".join(["word"] * 500)])

    assert out["input_ids"].shape == (2, context)
    assert int(out["attention_mask"][1].sum()) == context
    assert int(out["attention_mask"][0].sum()) < context


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
    captions = next(spec.load(batch=4, tokenize=keep_captions).train())["caption"]
    assert captions.shape == (4,)

    def tokens(encoder):
        inputs = InputSpec(Field("image", (8, 8, 3)),
                           {"textcontext": Condition(encoder, field="text")})
        batch = next(spec.load(batch=4, tokenize=inputs.tokenize).train())
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

    batch = next(spec.load(batch=4, tokenize=inputs.tokenize).train())

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
    batch = next(data.train())
    assert batch["x"].shape == (4, 3) and batch["index"].shape == (4,)


def test_from_grain_repeats_a_map_dataset_so_a_run_outlasts_the_corpus():
    """`Dataset.train` is endless; a caller's finite pipeline would otherwise
    stop the run one pass in."""
    stream = Dataset.from_grain(_grain_points(8), batch=4, **WORKERS).train()

    assert len(_indices(stream, 6)) == 6


def test_from_grain_takes_a_streamed_pipeline_and_carries_grains_own_state():
    """An IterDataset has no index, so it is batched where it is and reports
    the position grain keeps for it."""
    rows = pygrain.MapDataset.source(_Points(8)).repeat(None).to_iter_dataset()
    data = Dataset.from_grain(rows, batch=4, records=8, **WORKERS)

    stream = data.train()
    assert isinstance(stream, Checkpointable)
    first = _indices(stream, 1)
    state = stream.get_state()
    rest = _indices(stream, 1)

    resumed = data.train()
    resumed.set_state(state)
    assert _indices(resumed, 1) == rest != first


def test_from_grain_scores_a_validation_pass_that_ends():
    data = Dataset.from_grain(_grain_points(16), batch=4,
                              validation=pygrain.MapDataset.source(_Points(8)),
                              **WORKERS)

    assert data.val is not None
    assert _indices(data.val(), 5) == [[0, 1, 2, 3], [4, 5, 6, 7]]


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

        def loss(self, params, batch, step):
            return jnp.mean((self.model.apply(params, batch["x"]) - batch["y"]) ** 2), Aux({})

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

    for expected, actual in zip(jax.tree.leaves(whole.params),
                                jax.tree.leaves(resumed.params), strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6)
