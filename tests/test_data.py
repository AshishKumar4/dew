"""Data layer tests: the registry, the Dataset contract, lazy imports, the
video specs and the determinism of what a record becomes.

`import dew.data` and the grain paths must not need the optional extras,
which is the point of several of these tests. Anything that genuinely
requires an optional dependency skips.

Record factories, holdout hashes and batch shares are Dew policies; Grain oracles iteration/resume.
"""

import atexit
import dataclasses
import itertools
import json
import subprocess
import time

import cv2
import grain.python as pygrain
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from data_support import CAPTION_TEMPLATES, FLIP_JITTER, keep_captions
from reader_support import assert_resumes_where_it_stopped

from dew.config.data import Augment, HoldOut, ImageData, RecordsData, VideoData, loader
from dew.data import Loading, Memory, images
from dew.data.compile import compile, validation
from dew.data.dataset import Checkpointable, DataPartition, Records as IndexedRows, SourceSlice, describe
from dew.data.image_codec import decode_image
from dew.data.images import EncodeCaptions
from dew.data.pipeline import Pipeline
from dew.data.sinks import export
from dew.data.sources import Torch, source as stored_source
from dew.data.steps.generic import Formatting
from dew.objectives.base import VALID_ROWS
from dew.position import ENVELOPE
from dew.registry import datasets

WORKERS = {"loading": Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1)}


# ---------------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------------


def test_an_unknown_dataset_is_refused():
    with pytest.raises(KeyError):
        datasets["flowers"]


def test_an_unknown_recipe_field_is_refused_before_opening_data():
    with pytest.raises(ValueError):
        datasets.build("images", source=[], image_scale=64)


def test_a_recipe_and_an_explicit_pipeline_read_the_same_records():
    rows = [{"index": np.int32(index)} for index in range(16)]
    recipe = loader(RecordsData(source=rows), batch=4, seed=9, read="memory", **WORKERS)
    direct = compile(Pipeline(Memory(rows)).shuffle(9).repeat().batch(4), seed=9, read="memory", **WORKERS)
    left, right = recipe.train(DataPartition()), direct.train(DataPartition())
    try:
        assert _indices(left, 8) == _indices(right, 8)
    finally:
        left.close()
        right.close()


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
class Indexed:
    """Records that carry their own index, untouched, so a test can read
    which records a batch holds. The plumbing is the one every spec uses."""

    length: int = 32
    val_batches: int | None = None
    batch: int = 8
    count: int | None = None
    seed: int = 0
    loading: Loading = dataclasses.field(default_factory=lambda: Loading(workers=0))

    def source(self):
        return _Indexed(self.length)

    def pipelines(self):
        source = self.source()
        records = len(source) if self.count is None else self.count
        held = (self.val_batches or 0) * self.batch
        val = Pipeline(Memory(SourceSlice(source, 0, held))) if held else None
        return Pipeline(Memory(SourceSlice(source, held, records))), val, ()


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


@pytest.mark.parametrize("workers", [0, 2])
def test_a_validation_pass_reads_every_held_out_record_once(workers):
    """A pass is the split, once, in record order, and then it ends.

    grain's DataLoader applies its operations inside the worker processes, so
    each worker had to fill a whole batch out of its own slice of the split,
    and the unbounded num_epochs a run leaves at None let it read that slice
    again to do so.
    """
    data = loader(Indexed(val_batches=3, loading=Loading(workers=workers), batch=8), batch=8,
        read="indexed", seed=0, loading=Loading(workers=workers))

    batches, ended = _bounded(data.val(DataPartition()), 12)
    assert [[int(i) for i in b["index"]] for b in batches] == [
        list(range(8)), list(range(8, 16)), list(range(16, 24))]
    assert ended
    train = [index for batch in _indices(data.train(DataPartition()), 2) for index in batch]
    assert set(train).isdisjoint(range(24))


def test_a_validation_split_cannot_swallow_every_record():
    """One record and a held-out one would leave nothing to train on."""
    with pytest.raises(ValueError):
        loader(Indexed(val_batches=4, batch=8), batch=8, read="indexed", seed=0, loading=Loading(workers=0))
    with pytest.raises(ValueError):
        loader(Indexed(length=1, val_batches=1, batch=1), batch=1, read="indexed", seed=0,
            loading=Loading(workers=0))


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
    spec = Augmenting(length=16, splits={"test": 8}, image_size=8, augment=None,
                      val_split="test", val_batches=1, **WORKERS)

    data = loader(dataclasses.replace(spec, batch=4), batch=4, read="indexed", seed=spec.seed,
        loading=spec.loading, host=(EncodeCaptions(keep_captions),))

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
    fields = dict(length=16, splits={"test": 8}, image_size=8, augment=None,
                  val_split="test", **WORKERS)

    bounded_pass = loader(Augmenting(val_batches=1, **fields, batch=4), batch=4, read="indexed", seed=0,
        **WORKERS)
    whole_pass = loader(Augmenting(val_batches=None, **fields, batch=4), batch=4, read="indexed", seed=0,
        **WORKERS)

    assert bounded_pass.val is not None and whole_pass.val is not None
    assert len(_labels(bounded_pass.val(DataPartition()), batches=8)) == 4
    assert _labels(whole_pass.val(DataPartition()), batches=8) == list(range(1000, 1008))


def test_without_a_named_split_the_head_hold_out_is_unchanged():
    """The default path has to stay exactly what it was: the same records,
    the same pixels, the same training count."""
    spec = Augmenting(length=16, image_size=8, augment=None, val_batches=1,
                      **WORKERS)

    data = loader(dataclasses.replace(spec, batch=4), batch=4, read="indexed", seed=spec.seed,
        loading=spec.loading, host=(EncodeCaptions(keep_captions),))

    assert data.records == 12 and data.steps_per_epoch == 3
    held = next(data.val(DataPartition()))
    assert _labels(iter([held])) == [0, 1, 2, 3]
    assert set(_labels(data.train(DataPartition()), batches=3)).isdisjoint(range(4))
    expected = cv2.resize(_Images(16)[0]["image"], (8, 8), interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(held["image"][0], expected)


def test_a_dataset_of_one_pile_refuses_a_split_it_cannot_name(tmp_path):
    """An arrayrecord spec reads its shards as one pile; scoring a split it
    cannot name would read the training records again and call it validation."""
    rows = [{"image": images.resize_image(_Images(8)[index]["image"], 8), "label": index,
             "caption": f"record {index}"} for index in range(8)]
    export(rows, str(tmp_path), format="array_record")
    from dew.config.data import Split

    spec = ImageData(source=str(tmp_path), image_size=8, augment=None, validation=HoldOut(records=2))
    assert loader(spec, batch=2, read="memory", **WORKERS).records == 6
    with pytest.raises(ValueError):
        loader(dataclasses.replace(spec, validation=Split("test")), batch=2, read="memory", **WORKERS)


class _Endless:
    def __getitem__(self, index):
        return {"index": index, "image": np.zeros((4, 4, 3), np.uint8)}


@dataclasses.dataclass(frozen=True)
class Unsized:
    count: int | None = None
    image_size: int = 4
    val_batches: int | None = None
    seed: int = 0
    loading: Loading = dataclasses.field(default_factory=lambda: Loading(workers=0))
    batch: int = 8

    def pipelines(self):
        if self.count is None:
            raise ValueError("a source without an authoritative length needs count=")
        source = _Endless()
        rows = [{"image": source[index]["image"], "label": source[index]["index"]}
                for index in range(self.count)]
        return ImageData(source=rows, image_size=self.image_size, augment=None, validation=None).pipelines()


def test_a_source_without_a_length_needs_an_explicit_count():
    """The factory guessed a million records for such a source, so the
    sampler drew indices past the data and the run reported the guess."""
    with pytest.raises(ValueError):
        loader(Unsized(**WORKERS, batch=8), batch=8, read="indexed", seed=0, **WORKERS)

    data = loader(Unsized(count=16, val_batches=None, image_size=4, **WORKERS, batch=8), batch=8,
        read="indexed", seed=0, **WORKERS)
    assert data.records == 16
    assert sorted(int(i) for batch in itertools.islice(data.train(DataPartition()), 2)
                  for i in batch["label"]) == list(range(16))


def test_a_count_past_the_end_of_the_source_is_refused():
    """A count above the source became the sampler's record count, and the
    first index past the end raised inside a worker."""
    with pytest.raises(ValueError):
        loader(Augmenting(length=32, count=33, **WORKERS, batch=8), batch=8, read="indexed", seed=0,
               **WORKERS)


def test_a_count_uses_the_head_of_the_source():
    data = loader(Indexed(count=16, batch=8), batch=8, read="indexed", seed=0, loading=Loading(workers=0))
    assert data.records == 16
    assert sorted(i for batch in _indices(data.train(DataPartition()), 2) for i in batch) == list(range(16))


def test_the_training_stream_repeats_instead_of_ending():
    """The trainer keeps asking for batches long after one pass over the
    records, and the next pass is the same records in another order."""
    data = loader(Indexed(val_batches=2, batch=8), batch=8, read="indexed", seed=0,
                  loading=Loading(workers=0))

    epoch = data.steps_per_epoch  # the sixteen training records, in two batches
    batches, ended = _bounded(data.train(DataPartition()), 3 * epoch)
    first_epoch = [int(i) for batch in batches[:epoch] for i in batch["index"]]
    later = [int(i) for batch in batches[epoch:] for i in batch["index"]]

    assert not ended and len(batches) == 3 * epoch
    assert sorted(first_epoch) == list(range(16, 32))
    assert set(later) == set(first_epoch), "the stream reads the same records again"


# ---------------------------------------------------------------------------------
# The batch is stacked in the worker that read the records
# ---------------------------------------------------------------------------------

def _by_hand(source, *, seed, batch, count):
    """`count` batches stacked the way the training process stacked them
    before the workers did: `source` reshuffled endlessly from `seed`, read
    record by record in order, `batch` records to a batch.

    Read off a plain Grain `MapDataset` and not off the compiled stream, so
    it is a reference for what the stream returns rather than a restatement
    of it.
    """
    order = pygrain.MapDataset.source(source).seed(seed).shuffle(seed).repeat(None)
    rows = [order[index] for index in range(count * batch)]
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
    stream = compile(Pipeline(Memory(source)).shuffle(5).repeat().batch(4), seed=5, loading=Loading(
        workers=workers, threads=2, read_buffer=4, worker_buffer=2)).train(DataPartition())

    read = _fields(stream, 12)
    stream.close()

    assert read == _by_hand(source, seed=5, batch=4, count=12)
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
    passes = validation(Pipeline(Memory(_Indexed(30))), batch=4, loading=Loading(
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
    data = compile(Pipeline(Memory(_Indexed(30))).repeat().batch(4), loading=Loading(
        workers=workers, threads=2, read_buffer=4, worker_buffer=2))
    interrupted = data.train(DataPartition())
    _fields(interrupted, 2)
    offset = interrupted.get_state()
    interrupted.close()

    stream = data.train(DataPartition())
    stream.set_state(offset)
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
        with pytest.raises(ValueError):
            DataPartition(0, 8).rows(batch)
        with pytest.raises(ValueError):
            loader(dataclasses.replace(data, batch=batch), batch=batch, read="indexed", seed=data.seed,
                loading=data.loading).train(DataPartition(3, 8))
    assert DataPartition(0, 8).rows(64) == 8


def test_each_share_reads_its_own_slice_of_the_validation_split():
    """Share p of n validates records p, p + n, ... of the split, in whole
    batches of the share's size."""
    data = loader(Indexed(val_batches=2, batch=8), batch=8, read="indexed", seed=0,
                  loading=Loading(workers=0))

    assert _indices(data.val(DataPartition(1, 2)), 3) == [[1, 3, 5, 7], [9, 11, 13, 15]]


def test_the_readers_of_one_share_read_the_same_records():
    """Processes a sequence or a pipeline spans between them hold the same
    rows, so they read one share, and read it alike: that is what makes
    their copies of the batch agree."""
    data = loader(Indexed(length=32, batch=8), batch=8, read="indexed", seed=0, loading=Loading(workers=0))
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
        stream = loader(Indexed(length=length, seed=seed, batch=batch), batch=batch, read="indexed",
            seed=seed, loading=Loading(workers=0)).train
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
        stream = loader(Indexed(**other, batch=8), batch=8, read="indexed", seed=other.get("seed", 0),
                      loading=Loading(workers=0)).train(DataPartition())
        with pytest.raises(ValueError):
            stream.set_state(saved[0])
        stream.close()


def test_a_shard_offset_cannot_resume_a_global_stream():
    """A stream whose windows come out of one shard reports where that shard
    stopped. Handing such a position to a stream that reads its records
    globally would resume it somewhere else, so it is refused by name."""
    stream = loader(Indexed(batch=8), batch=8, read="indexed", seed=0,
        loading=Loading(workers=0)).train(DataPartition())

    with pytest.raises(ValueError):
        stream.set_state(json.dumps({"last_seen_indices": {"0": 15}}).encode())
    stream.close()


def test_a_held_out_slice_keeps_the_identity_of_the_corpus_it_cuts():
    """Two corpora of one type and length, each with a slice held out, were
    described the same, so a position saved over one resumed the other."""
    rows = [{"index": np.int32(index)} for index in range(16)]

    def stream(name):
        recipe = RecordsData(source=Memory(rows, identity=name), validation=HoldOut(records=4))
        return loader(recipe, batch=4, read="indexed", loading=Loading(threads=1, read_buffer=1)).train(
            DataPartition())

    saved = stream("flowers")
    next(saved)
    state = saved.get_state()
    saved.close()

    other = stream("faces")
    with pytest.raises(ValueError):
        other.set_state(state)
    resumed = stream("flowers")
    resumed.set_state(state)
    resumed.close()


def test_an_indexed_filter_resumes_the_exact_accepted_record_order():
    pipeline = Pipeline(Memory(_Points(16))).shuffle(0).repeat().filter(
        lambda row: int(row["index"]) % 3 != 0, identity="exclude-multiples-of-three").batch(4)
    data = compile(pipeline, read="indexed", **WORKERS)
    stream = data.train(DataPartition())
    try:
        _indices(stream, 2)
        state = stream.get_state()
        expected = _indices(stream, 5)
    finally:
        stream.close()
    resumed = data.train(DataPartition())
    try:
        resumed.set_state(state)
        assert _indices(resumed, 5) == expected
    finally:
        resumed.close()
    assert all(index % 3 != 0 for batch in expected for index in batch)


# ---------------------------------------------------------------------------------
# Video datasets
# ---------------------------------------------------------------------------------

def _voxceleb_tree(root, split="train", identities=("id00012", "id00015")):
    """Build <root>/<split>/<identity>/<clip>/<utterance>.avi, 2 seconds of 16 kHz sound under 25 fps
    frames, plus a file that is no clip. AVI with PCM, where the sound's seek is sample-exact."""
    from moviepy.config import FFMPEG_BINARY

    (root / "frames.rgb").write_bytes(np.full((50, 32, 48, 3), 90, np.uint8).tobytes())
    (root / "sound.pcm").write_bytes(np.full(32000, 3000, np.int16).tobytes())
    clips = []
    for identity in identities:
        for clip in ("_raOc3-IRsw", "21Uxsk56VDQ"):
            path = root / split / identity / clip / "00001.avi"
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([FFMPEG_BINARY, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", "48x32", "-r", "25", "-i", str(root / "frames.rgb"), "-f", "s16le",
                            "-ar", "16000", "-ac", "1", "-i", str(root / "sound.pcm"), "-c:v", "ffv1",
                            "-c:a", "pcm_s16le", str(path)], check=True)
            clips.append(path)
    (root / split / "filelist.txt").write_text("ignored\n")
    return clips


def test_local_videos_lists_every_file_under_the_directory(tmp_path):
    clips = _voxceleb_tree(tmp_path)
    (tmp_path / "extra.webm").write_bytes(b"")

    from dew.config.data import _VideoRow
    from dew.data import Files

    patterns = (str(tmp_path / "**" / "*.avi"), str(tmp_path / "**" / "*.webm"))
    rows = Files(patterns, format="files").records("memory")
    records = [_VideoRow(caption="a clip")(row) for row in rows]

    assert sorted(r["mp4"] for r in records) == sorted(
        [c.as_uri() for c in clips] + [(tmp_path / "extra.webm").as_uri()]
    )
    assert {r["caption"] for r in records} == {"a clip"}
    with pytest.raises((ValueError, FileNotFoundError)):
        stored_source(str(tmp_path / "missing"), format="files").records("memory")


@pytest.mark.parametrize("rate", [16000, 48000])
def test_a_video_recipe_reads_frames_and_the_sound_around_them_at_its_audio_model_s_rate(
        tmp_path, monkeypatch, rate):
    """Every clip under the directory, and nothing else, gives one record of `frames` frames fitted
    into `frame_size` squares and the sound under them with `audio_padding` frames' worth either
    side, at the rate the audio model's extractor reads: a 48 kHz model's record holds 1920 samples
    a frame where a 16 kHz one holds 640. The audio model featurizes the sound when a run encodes
    it, so the record holds the waveform."""
    from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor

    clips = _voxceleb_tree(tmp_path)
    monkeypatch.setattr(AutoFeatureExtractor, "from_pretrained",
                        lambda name: Wav2Vec2FeatureExtractor(sampling_rate=rate))
    spec = VideoData(source=str(tmp_path / "train"), caption="a video of a speaker", frame_size=16, frames=4,
                 audio_padding=1)

    train, _, _ = spec.pipelines()
    records = list(train.repeat(1))

    assert len(records) == len(clips)
    for record in records:
        assert set(record) == {"video", "audio", "audio_valid", "caption"}
        assert record["video"].shape == (4, 16, 16, 3) and record["video"].dtype == np.uint8
        assert (record["video"] == 90).all()
        assert record["audio"].shape == (6 * rate // 25,) and record["audio"].dtype == np.float32
        # Two seconds of sound under every clip: each window's 0.24 seconds are all the track's.
        assert record["audio_valid"].dtype == bool and int(record["audio_valid"].sum()) == 6 * rate // 25
        assert np.allclose(record["audio"], 3000 / 32768, atol=1e-3)
        assert record["caption"] == "a video of a speaker"


def test_a_video_recipe_reads_the_same_batches_at_any_worker_count_and_after_a_resume(tmp_path, monkeypatch):
    """Clips whose frames and sound count up, so every drawn window shows: two workers give the
    batches the main process gives, and a reader resumed from a saved position continues with the
    batches an uninterrupted one reads, windows included."""
    from test_data_av import _mux
    from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor

    (tmp_path / "train").mkdir()
    for index in range(6):
        _mux(tmp_path / "train" / f"{index}.avi")
    monkeypatch.setattr(AutoFeatureExtractor, "from_pretrained",
                        lambda name: Wav2Vec2FeatureExtractor(sampling_rate=16000))
    spec = VideoData(source=str(tmp_path / "train"), frames=4, frame_size=16, audio_padding=1)

    def read(workers, count, state=None):
        data = loader(spec, batch=2, read="memory", seed=7,
                      loading=Loading(workers=workers, threads=1, read_buffer=2, worker_buffer=1))
        reader = data.train(DataPartition())
        try:
            if state is not None:
                reader.set_state(state)
            return [next(reader) for _ in range(count)], reader.get_state()
        finally:
            reader.close()

    single, _ = read(0, 4)
    workers, _ = read(2, 4)
    for one, many in zip(single, workers, strict=True):
        for field in ("video", "audio", "audio_valid"):
            np.testing.assert_array_equal(one[field], many[field])
    assert any(not np.array_equal(batch["video"][0], batch["video"][1]) for batch in single)
    _, saved = read(0, 2)
    resumed, _ = read(0, 2, saved)
    for expected, found in zip(single[2:], resumed, strict=True):
        np.testing.assert_array_equal(expected["audio"], found["audio"])


# ---------------------------------------------------------------------------------
# Determinism across worker counts, and a restart mid-epoch
# ---------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Augmenting:
    """Deterministic images and class captions, addressed by index, through
    the real image transform: resize, flip, jitter and the prompt template
    all draw from grain's per-record rng, the draw a worker count could move.
    The captions are read back as text, so one stays comparable after a trip
    through a worker process."""

    length: int = 16
    image_size: int = 128
    augment: Augment | None = FLIP_JITTER
    augmentation_size: int | None = None
    val_batches: int | None = 4
    val_split: str | None = None
    count: int | None = None
    seed: int = 0
    loading: Loading = dataclasses.field(default_factory=lambda: Loading(workers=0))
    batch: int = 4
    splits: dict[str, int] = dataclasses.field(default_factory=dict)
    """Records each named split holds, for a spec that has more than one."""

    def source(self, split=None):
        return _Images(self.length if split is None else self.splits[split],
                       first=0 if split is None else 1000)

    def prepared(self, rows, *, training=True):
        mode = self.augment if training else None
        classes = tuple(["rose", "tulip", "lotus", "orchid", "marigold"] *
                        ((1000 + max((self.length, *self.splits.values()))) // 5 + 1))
        return ImageData(source=Memory(rows), image_size=self.image_size,
                         staging_size=self.augmentation_size, caption_templates=CAPTION_TEMPLATES,
                         classes=classes, augment=mode, validation=None).pipelines()[0]

    def pipelines(self):
        source = self.source()
        count = len(source) if self.count is None else self.count
        if count > len(source):
            raise ValueError(f"count {count} is more than the {len(source)} records")
        held_out = 0 if self.val_split else (self.val_batches or 0) * self.batch
        train = SourceSlice(source, held_out, count)
        held = SourceSlice(source, 0, held_out) if held_out else None
        if self.val_split is not None:
            held = self.source(self.val_split)
            if self.val_batches is not None:
                held = held[:self.val_batches * self.batch] if isinstance(held,
                    list) else [held[index] for index in
                        range(min(len(held), self.val_batches * self.batch))]
        training = self.prepared(train)
        validation = None if held is None else self.prepared(held, training=False)
        return training, validation, (EncodeCaptions(),)


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
        return {"index": index, "label": index, "image": rng.randint(0, 256, (12, 12, 3), np.uint8)}


def _rows(batch):
    """(index, pixels, caption) per record, so a comparison covers all three."""
    return [(int(batch["label"][row]), batch["image"][row].tobytes(),
             str(batch["caption"][row]))
            for row in range(len(batch["label"]))]


def _augmented(worker_count, length=16, batch=4, seed=3):
    """{record index: (pixels, caption)} for one epoch at this worker count."""
    data = loader(Augmenting(length=length, image_size=8, seed=seed, val_batches=None,
                      loading=Loading(workers=worker_count, threads=1, read_buffer=1,
                                      worker_buffer=1), batch=batch), batch=batch, read="indexed",
                                          seed=seed, loading=Loading(workers=worker_count, threads=1,
                                              read_buffer=1,
                                      worker_buffer=1), host=(EncodeCaptions(keep_captions),))
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

    def prepared(self, rows, *, training=True):
        prepared = super().prepared(rows, training=training)
        prefix = Pipeline(prepared.source).map(_LingeringRecord(self.seconds)).stages
        return dataclasses.replace(prepared, stages=(*prefix, *prepared.stages))


@dataclasses.dataclass(frozen=True)
class _LingeringRecord(pygrain.MapTransform):
    seconds: float

    def map(self, element):
        _linger(self.seconds)
        return element


@pytest.mark.slow
def test_a_stream_whose_workers_are_slow_to_stop_closes_within_grains_bound():
    """grain stops a stream's worker processes one after another, each
    finishing the batch in its hands before it exits, and kills a worker
    that has not exited within 25 s. Four workers taking 2 s each keep a
    close over 8 s: slow, but bounded by grain itself, so the close must not
    call it a hang (a TimeoutError). sft_gemma4's four workers took 5.1 to
    7.4 s to stop, past a budget of 2 s and 1 s a worker."""
    from dew.training import MeshSpec
    from dew.training.distributed import DevicePrefetchIterator

    loading = Loading(workers=4, threads=1, read_buffer=1, worker_buffer=1)
    data = Lingering(length=64, image_size=8, seed=3, val_batches=None, loading=loading)
    stream = loader(dataclasses.replace(data, batch=4), batch=4, read="indexed", seed=data.seed,
        loading=data.loading).train(DataPartition())
    prefetch = DevicePrefetchIterator(stream, MeshSpec().build(jax.devices()[:1]))
    # grain interleaves its workers' batches in turn: one batch from each
    # means every worker has read a record and lingers.
    for _ in range(loading.workers):
        next(prefetch)
    started = time.monotonic()
    prefetch.close()
    # A stop past the 5 s a stop takes quietly (`_QUIET_STOP_SECONDS`), and still not a hang.
    assert time.monotonic() - started > 5


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
    def opened():
        return loader(Augmenting(image_size=8, seed=3, val_batches=2,
                          loading=Loading(workers=worker_count, threads=1, read_buffer=1,
                                          worker_buffer=1), batch=4), batch=4, read="indexed", seed=3,
                                              loading=Loading(workers=worker_count, threads=1, read_buffer=1,
                                          worker_buffer=1), host=(EncodeCaptions(keep_captions),))

    interrupted = opened().train(DataPartition())
    seen = _rows(next(interrupted))
    state = interrupted.get_state()
    rest = [row for batch in itertools.islice(interrupted, 3) for row in _rows(batch)]

    restored = opened().train(DataPartition())
    restored.set_state(state)
    resumed = [row for batch in itertools.islice(restored, 3) for row in _rows(batch)]


    assert resumed == rest, "a resumed epoch owes the same records, augmented alike"
    assert sorted(index for index, _, _ in seen + rest[:4]) == list(range(8, 16))


_DEFAULT_VALIDATED_PARTITION = DataPartition()


def _validated(length, val_batches, batch, partition=_DEFAULT_VALIDATED_PARTITION, **read):
    """{record index: (pixels, caption)} for one share's validation pass."""
    data = loader(Augmenting(length=length, image_size=8, seed=3, val_batches=val_batches,
                      loading=Loading(workers=0, worker_buffer=1, **read), batch=batch), batch=batch,
                          read="indexed", seed=3, loading=Loading(workers=0, worker_buffer=1, **read),
                              host=(EncodeCaptions(keep_captions),))
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
    """Exported pixels, labels and captions equal the configured record stage."""
    spec = Augmenting(length=16, image_size=8, augment=None, val_batches=None, **WORKERS)
    live = _Images(16)
    rows = [{name: column[index] for name, column in batch.items()}
            for batch in spec.prepared(live, training=False).batch(4)
            for index in range(4)]
    manifest = export(rows, str(tmp_path), format="array_record", rows_per_shard=8,
                      metadata={"image_size": 8})
    assert manifest["rows"] == 16 and manifest["metadata"]["image_size"] == 8
    assert len(manifest["files"]) == 2
    data = loader(ImageData(source=str(tmp_path), image_size=8, augment=None, validation=None), batch=4,
                  read="memory", **WORKERS, host=(EncodeCaptions(keep_captions),))
    seen = {}
    stream = data.train(DataPartition())
    try:
        for batch in itertools.islice(stream, data.steps_per_epoch):
            for index, pixels, caption in _rows(batch):
                seen[index] = (pixels, caption)
    finally:
        stream.close()
    assert sorted(seen) == list(range(16))
    for index in range(16):
        pixels, caption = seen[index]
        assert pixels == rows[index]["image"].tobytes()
        assert caption == rows[index]["caption"]
        assert np.frombuffer(pixels, np.uint8).shape == (8 * 8 * 3,)


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
class _Lazy:
    """A source whose records are read only as the stream reads them.

    `Memory` fingerprints every record when a pipeline compiles, so a record
    that cannot be read would fail there and never reach the stream.
    """

    values: IndexedRows

    def describe(self):
        return describe(self.values)

    def label(self):
        return "lazy"

    def fingerprint(self):
        return {"source": describe(self.values), "records": len(self.values)}

    def supports(self, read):
        return read != "streamed"

    def records(self, read):
        return self.values

    def metadata(self):
        return {"rows": len(self.values)}

    def lengths(self):
        return None

    def manifest(self):
        raise ValueError("a lazy source is indexed")


@dataclasses.dataclass(frozen=True)
class Raising(Indexed):
    bad: int | None = None

    def pipelines(self):
        return Pipeline(_Lazy(_Raising(self.length, self.bad))), None, ()


@pytest.mark.parametrize("worker_count", [0, pytest.param(2, marks=pytest.mark.slow)])
def test_a_record_that_cannot_be_read_stops_the_stream(worker_count):
    """The source's own error has to reach the trainer. A pipeline that caught
    it would train on whatever it substituted, and a worker's exception is the
    easiest one to lose."""
    data = loader(Raising(length=8, bad=3, loading=Loading(workers=worker_count), batch=2), batch=2,
        read="indexed", seed=0, loading=Loading(workers=worker_count))

    delivered = []
    with pytest.raises(RuntimeError):
        for batch in data.train(DataPartition()):
            delivered.extend(int(i) for i in batch["index"])

    assert 3 not in delivered
    assert all(index in range(8) for index in delivered), "no fabricated records"


def test_a_source_that_fails_on_every_record_raises_instead_of_an_empty_batch():
    data = loader(Raising(length=8, bad=None, batch=2), batch=2, read="indexed", seed=0,
        loading=Loading(workers=0))

    with pytest.raises(RuntimeError):
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

    with pytest.raises(ValueError):
        decode_image(truncated)


def test_the_image_transform_resizes_augments_and_captions_one_record():
    """What every image dataset hands the loader: the resized uint8 image,
    the caption text and, for a class-labelled record, its index."""
    spec = Augmenting(image_size=8, augment=None)
    element = _Images(16)[5]

    out = next(iter(spec.prepared([element]).batch(1)))

    np.testing.assert_array_equal(
        out["image"][0], cv2.resize(element["image"], (8, 8), interpolation=cv2.INTER_AREA))
    assert out["caption"][0] in {template.format("rose") for template in CAPTION_TEMPLATES}
    np.testing.assert_array_equal(out["label"], [5])


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


def test_a_map_dataset_streams_and_resumes_as_grains_own_pipeline_reads_it():
    """A caller's shuffled `MapDataset` reads, batch for batch, as grain's
    own `repeat(None).batch(4)` of it, across the end of a pass, and a
    stream resumed from a saved position reads on where grain's batches do."""
    pipeline = _grain_points(10)
    reference = pipeline.repeat(None).batch(4)
    expected = [[int(i) for i in reference[index]["index"]] for index in range(6)]
    data = compile(Pipeline(Memory(_Points(10))).shuffle(0).repeat().batch(4), read="indexed", **WORKERS)

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
    data = compile(Pipeline(Memory(_Points(8))).repeat().batch(4), read="streamed", **WORKERS)

    stream = data.train(DataPartition())
    assert isinstance(stream, Checkpointable)
    first = _indices(stream, 1)
    state = stream.get_state()
    rest = _indices(stream, 1)

    resumed = data.train(DataPartition())
    resumed.set_state(state)
    assert _indices(resumed, 1) == rest != first


def test_from_grain_scores_a_validation_pass_that_ends():
    data = compile(Pipeline(Memory(_grain_points(16))).repeat().batch(4),
        validation=Pipeline(Memory(pygrain.MapDataset.source(_Points(8)))).batch(4), read="indexed",
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

        def fresh_variables(self, key, held):
            return self.model.init(key, jnp.zeros((1, 3)))

        def loss(self, variables, batch, step):
            return jnp.mean((self.model.apply(variables, batch["x"]) - batch["y"]) ** 2), Aux({})

    def run(steps, directory=None):
        trainer = Trainer(
            Regression(), optax.sgd(0.1), key=jax.random.key(0),
            layout=Layout(min_shard=1, tolerance=1.0),
            checkpoints=None if directory is None else Checkpoints(str(directory)))
        # The simulated mesh is eight devices wide, so a step is eight records.
        return trainer.fit(compile(Pipeline(Memory(_grain_points())).repeat().batch(8), read="indexed",
            **WORKERS),
                           steps=steps, log_every=1,
                           checkpoint_every=None if directory is None else 2)

    run(2, tmp_path / "run")
    resumed = run(4, tmp_path / "run")
    whole = run(4)

    for expected, actual in zip(jax.tree.leaves(whole.variables),
                                jax.tree.leaves(resumed.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


# ---------------------------------------------------------------------------------
# Records held in memory, and where a default run reads
# ---------------------------------------------------------------------------------

def _columns(length=12):
    index = np.arange(length, dtype=np.int32)
    return {"index": index, "x": np.stack([index, -index], axis=1).astype(np.float32)}


def test_records_in_memory_are_reshuffled_every_epoch_and_each_read_once_per_epoch():
    stream = loader(RecordsData(source=_columns()), batch=4, seed=3, read="memory").train(DataPartition())

    epochs = [list(itertools.chain.from_iterable(_indices(stream, 3))) for _ in range(3)]
    for epoch in epochs:
        assert sorted(epoch) == list(range(12)), "each record once per epoch"
    assert len({tuple(epoch) for epoch in epochs}) == 3, "a fresh order every epoch"
    assert epochs[0] != list(range(12)), "the first epoch is shuffled too"


def test_records_in_memory_follow_their_seed():
    def order(seed):
        return _indices(loader(RecordsData(source=_columns()), batch=4, seed=seed,
            read="memory").train(DataPartition()), 6)

    assert order(0) == order(0)
    assert order(0) != order(1)


def test_columns_rows_and_a_torch_dataset_are_the_same_records():
    import torch

    columns = _columns()
    rows = [{"index": columns["index"][i], "x": columns["x"][i]} for i in range(12)]
    tensors = torch.utils.data.TensorDataset(torch.arange(12), torch.from_numpy(columns["x"]))

    first, *others = (next(data.train(DataPartition())) for data in (
        loader(RecordsData(source=columns), batch=4, seed=0, read="memory"), loader(RecordsData(source=rows),
            batch=4, seed=0, read="memory"),
        loader(RecordsData(source=Torch(tensors, fields=("index", "x"), identity="columns@1")),
            batch=4, seed=0, read="memory",
            loading=Loading(workers=0))))

    assert first["x"].shape == (4, 2) and first["x"].dtype == np.float32
    for other in others:
        for name in ("index", "x"):
            assert other[name].dtype == first[name].dtype
            np.testing.assert_array_equal(first[name], other[name])


def test_a_torch_dataset_streams_the_records_it_indexes_and_resumes():
    """A map-style torch dataset streams as consecutive ranges of the records it indexes, as a
    memory source does: a streamed pass holds every record once, and a streamed run resumed after
    two batches reads the batches the uninterrupted one reads."""
    import torch

    tensors = torch.utils.data.TensorDataset(torch.arange(12))
    source = Torch(tensors, fields=("index",), identity="indices@1")
    reading = Loading(workers=0, threads=1, read_buffer=0, worker_buffer=1)
    from dew.data.compile import validation
    from dew.data.pipeline import Pipeline

    passed = validation(Pipeline(source).batch(4), batch=4, read="streamed", loading=reading)(DataPartition())
    try:
        assert sorted(int(value) for batch in passed for value in batch["index"]) == list(range(12))
    finally:
        passed.close()

    def opened():
        return loader(RecordsData(source=source), batch=4, seed=0, read="streamed", loading=reading).train(
            DataPartition())

    def indices(reader, count):
        return [np.asarray(next(reader)["index"]).tolist() for _ in range(count)]

    assert_resumes_where_it_stopped(opened, indices, 2, 4)


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

    batch = next(loader(RecordsData(
        source=Torch(Pictures(), fields=("image", "label"), identity="pictures@1"),
        validation=Torch(Pictures(), fields=("image", "label"), identity="pictures@1")),
            batch=4, read="memory",
            loading=Loading(workers=0)).val(DataPartition()))

    assert batch["image"].shape == (4, 5, 5, 3) and batch["image"].dtype == np.uint8
    np.testing.assert_array_equal(batch["image"][:, 0, 0, 0], batch["label"])
    with pytest.raises(TypeError):
        loader(RecordsData(source=Torch(Pictures())), batch=4, read="memory", loading=Loading(workers=0))
    with pytest.raises(TypeError):
        loader(RecordsData(source=Torch(torch.utils.data.DataLoader(Pictures()))), batch=4, read="memory",
            loading=Loading(workers=0))


def test_columns_of_different_lengths_are_refused_by_name():
    with pytest.raises(ValueError):
        loader(RecordsData(source={"index": np.arange(4), "x": np.zeros((3, 2))}), batch=2, read="memory")


def test_an_in_memory_stream_resumes_on_the_records_it_had_not_read():
    data = loader(RecordsData(source=_columns()), batch=4, seed=0, read="memory")
    stream = data.train(DataPartition())
    _indices(stream, 2)
    state = stream.get_state()
    rest = _indices(stream, 4)

    resumed = data.train(DataPartition())
    resumed.set_state(state)
    assert _indices(resumed, 4) == rest


def test_process_shares_of_in_memory_records_split_every_global_batch():
    data = loader(RecordsData(source=_columns()), batch=4, seed=0, read="memory")
    whole = _indices(data.train(DataPartition()), 3)
    halves = [_indices(data.train(DataPartition(index=index, count=2)), 3) for index in (0, 1)]

    for step, rows in enumerate(whole):
        first, second = halves[0][step], halves[1][step]
        assert not set(first) & set(second), "the shares are disjoint"
        assert sorted(first + second) == sorted(rows), "together they are the global batch"


def test_held_out_records_are_one_ordered_pass():
    data = loader(RecordsData(source=_columns(), validation=_columns(8)), batch=4, read="memory")

    assert data.records == 12 and data.steps_per_epoch == 3
    assert data.val is not None
    assert _indices(data.val(DataPartition()), 5) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_held_out_records_too_few_for_one_batch_are_one_filled_batch():
    """Three records at a batch of four are one batch, its fourth row a repeat."""
    batches = list(loader(RecordsData(source=_columns(), validation=_columns(3)), batch=4,
        read="memory").val(DataPartition()))

    assert len(batches) == 1
    np.testing.assert_array_equal(batches[0][VALID_ROWS], [True, True, True, False])


def test_a_validation_split_needs_no_training_records():
    """A second held-out split, beside a dataset's own, is the same ordered
    pass over its records, a short last batch filled and marked."""
    batches = list(validation(Pipeline(Memory(_columns(5))).map(Formatting()).batch(4), batch=4,
        read="memory")(DataPartition()))

    marked = [np.asarray(batch.get(VALID_ROWS, np.ones(4, bool))).tolist() for batch in batches]
    assert marked == [[True] * 4, [True, False, False, False]]
    assert _indices(validation(Pipeline(Memory(_columns(5))).map(Formatting()).batch(4), batch=4,
        read="memory")(DataPartition()), 1) == [[0, 1, 2, 3]]


def test_records_whose_field_lengths_differ_are_refused_with_the_remedy():
    """Token ids of varying length are the usual cause, and grain's own
    message names the batch structure rather than what to change. Automatic
    loading samples a batch to size its buffers, so the loader refuses them;
    with loading set by hand, the first batch does."""
    rows = [{"text": np.arange(length, dtype=np.int32)} for length in (3, 5, 4, 6)]
    with pytest.raises(ValueError):
        loader(RecordsData(source=rows), batch=2, read="memory")
    loaded = loader(RecordsData(source=rows), batch=2, read="memory", loading=Loading(workers=0))
    stream = loaded.train(DataPartition())
    with pytest.raises(ValueError):
        next(stream)
    stream.close()


def test_an_epoch_of_records_too_few_for_one_batch_is_refused():
    """Three records repeat to fill a batch of four, as a run counted in steps reads them, but an
    epoch of them is zero steps long, so a run counted in epochs is refused."""
    data = loader(RecordsData(source=_columns(3)), batch=4, read="memory")
    stream = data.train(DataPartition())
    try:
        assert len(next(stream)["index"]) == 4
    finally:
        stream.close()
    with pytest.raises(ValueError):
        data.epoch_steps(1)


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

        def fresh_variables(self, key, held):
            return self.model.init(key, jnp.zeros((1, 2)))

        def loss(self, variables, batch, step):
            target = batch["index"][:, None].astype(jnp.float32)
            return jnp.mean((self.model.apply(variables, batch["x"]) - target) ** 2), Aux({})

    def run(steps, directory=None):
        trainer = Trainer(
            Regression(), optax.sgd(0.01), key=jax.random.key(0),
            layout=Layout(min_shard=1, tolerance=1.0),
            checkpoints=None if directory is None else Checkpoints(str(directory)))
        return trainer.fit(loader(RecordsData(source=_columns(16)), batch=8, seed=0, read="memory"),
                           steps=steps, log_every=1,
                           checkpoint_every=None if directory is None else 2)

    run(2, tmp_path / "run")
    resumed = run(5, tmp_path / "run")
    whole = run(5)

    for expected, actual in zip(jax.tree.leaves(whole.variables),
                                jax.tree.leaves(resumed.variables), strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
