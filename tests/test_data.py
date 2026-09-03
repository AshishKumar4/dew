"""Data layer tests: registries, loader contracts, lazy imports, AV decoding.

The shared environment has no HF `datasets`, decord, pyav, moviepy or
video_reader, which is the point of several of these tests: the grain paths and
`import dew.data` must not need them. Anything that genuinely requires an
optional dependency skips.
"""

import os
import importlib.util
import inspect
import itertools
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import grain.python as pygrain
import jax
import numpy as np
import pytest

import dew.data
from dew.data import dataloaders
from dew.data.dataloaders import get_dataset_online, get_media_dataset_grain
from dew.data.registry import mediaDatasetMap
from dew.data.sources.audio_utils import _read_wav_mono, read_audio_ffmpeg
from dew.data.sources.av_utils import choose_clip_start
from dew.data.sources.base import DataAugmenter, DataSource, MediaDataset
from dew.data.sources.hf import HFDatasetSource
from dew.data.sources.images import (
    CombinedImageGCSSource, ImageGCSSource, ImageTFDSAugmenter, augment_image,
    gcs_filters, image_augmentations, image_augmenter, labelizer_oxford_flowers102,
)
from dew.data.sources import videos as videos_module
from dew.data.sources.videos import AudioVideoAugmenter, VideoLocalSource
from dew.data.sources.voxceleb2 import VoxCeleb2Source

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("source_cls", [ImageGCSSource, CombinedImageGCSSource])
def test_gcs_sources_require_an_explicit_dataset_path(source_cls):
    """The default was one developer's bucket mount, and an unset path reached
    os.path.join(None, ...) instead of saying anything."""
    parameter = inspect.signature(source_cls.get_source).parameters["path_override"]
    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(ValueError, match="explicit dataset path"):
        source_cls().get_source(None)


# ---------------------------------------------------------------------------------
# The filter seam: only the GCS augmenter has one, and it is concrete
# ---------------------------------------------------------------------------------

def test_filtering_is_not_part_of_the_augmenter_contract():
    """No pipeline applies a filter, so the abstract seam (and the two stubs
    that returned None instead of a transform) is gone."""
    assert not hasattr(DataAugmenter, "create_filter")
    assert not hasattr(ImageTFDSAugmenter, "create_filter")
    assert not hasattr(AudioVideoAugmenter, "create_filter")


def test_gcs_filter_returns_a_usable_filter_transform():
    filter_transform = gcs_filters(image_scale=256)
    assert isinstance(filter_transform, type)
    assert issubclass(filter_transform, pygrain.FilterTransform)
    # grain drops elements via `filter`, so that is the method that must exist
    assert callable(filter_transform.filter)


# ---------------------------------------------------------------------------------
# Lazy imports: the data layer must not drag in HF datasets / opencv / decord
# ---------------------------------------------------------------------------------

def test_importing_dew_data_pulls_in_no_heavy_dependencies():
    """`import dew.data` must not reach online_loader, which imports HF
    `datasets` at module scope and would take the whole package down with it.

    The hub source has the same duty: naming a hub dataset resolves without
    the streaming extra, only reading one needs it.
    """
    probe = (
        "import sys, dew.data;"
        "heavy = [m for m in ('datasets', 'cv2', 'decord', 'jax',"
        " 'dew.data.online_loader', 'dew.data.dataloaders')"
        " if m in sys.modules];"
        "assert not heavy, heavy;"
        "assert dew.data.MediaDataset.__name__ == 'MediaDataset';"
        "dew.data.HFDatasetSource(name='acme/pets');"
        "assert 'datasets' not in sys.modules"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    result = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT,
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr


def test_lazy_exports_resolve_and_unknown_names_raise_attribute_error():
    assert dew.data.get_media_dataset_grain is get_media_dataset_grain
    assert dew.data.VoxCeleb2Source is VoxCeleb2Source
    assert dew.data.HFDatasetSource is HFDatasetSource
    # An export left behind after a deletion raises only on first use, so
    # every advertised name is resolved here.
    for name in dir(dew.data):
        getattr(dew.data, name)
    with pytest.raises(AttributeError, match="has no attribute"):
        dew.data.get_dataset_from_thin_air


def test_reading_a_hub_dataset_names_the_streaming_extra(monkeypatch):
    """Naming one works anywhere; the first record is what needs HF datasets."""
    source = HFDatasetSource(name="acme/pets")
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError, match=r"dew-ml\[streaming\]"):
        len(source)


def test_a_hub_dataset_name_without_a_dataset_says_so():
    with pytest.raises(ValueError, match="hf:<dataset>:<split>"):
        get_media_dataset_grain("hf:", worker_count=0)


def test_online_dataset_factories_defer_the_streaming_import():
    """The streaming stack needs HF datasets; asking for a bad dataset name must
    fail on the name, not on the missing dependency."""
    already_imported = "dew.data.online_loader" in sys.modules
    with pytest.raises(ValueError, match="not found in onlineDatasetMap"):
        get_dataset_online("no_such_dataset")
    if not already_imported:
        assert "dew.data.online_loader" not in sys.modules


def test_online_streaming_loader_no_longer_accepts_the_unapplied_pre_map_args():
    try:
        from dew.data.online_loader import OnlineStreamingDataLoader
    except ImportError as exc:  # HF datasets missing: nothing to check here
        pytest.skip(f"the streaming loader needs HF datasets ({exc})")

    parameters = inspect.signature(OnlineStreamingDataLoader.__init__).parameters
    assert "pre_map_maker" not in parameters and "pre_map_def" not in parameters


# ---------------------------------------------------------------------------------
# Media grain API: explicit source, and a validation split of its own
# ---------------------------------------------------------------------------------

class _ListSource(DataSource):
    """Minimal random-access source; stands in for arrayrecord/video sources."""

    def __init__(self, length):
        self.records = [{"index": i} for i in range(length)]

    def get_source(self, path_override):
        assert path_override, "the loader must pass the caller's dataset_source through"
        return self.records


class _PassthroughAugmenter(DataAugmenter):
    """Returns records untouched, so tests can read the sampled indices."""

    def create_transform(self, **kwargs):
        self.transform_kwargs = kwargs

        class Passthrough(pygrain.MapTransform):
            def map(self, element):
                return element

        return Passthrough


@pytest.fixture
def fake_media_dataset(monkeypatch):
    """Register a dependency-free media dataset named "fake"."""
    dataset = MediaDataset(source=_ListSource(32), augmenter=_PassthroughAugmenter(),
                           media_type="video")
    monkeypatch.setitem(mediaDatasetMap, "fake", dataset)
    return dataset


def _indices(loader, num_batches):
    return [[int(i) for i in batch["index"]]
            for batch in itertools.islice(iter(loader), num_batches)]


def test_media_dataset_grain_requires_an_explicit_source(fake_media_dataset):
    """dataset_source=None raises instead of reaching os.path.join(None, ...)."""
    with pytest.raises(ValueError, match="dataset_source"):
        get_media_dataset_grain("fake")
    with pytest.raises(ValueError, match="not found in mediaDatasetMap"):
        get_media_dataset_grain("not_a_dataset", dataset_source="/tmp")


def test_media_dataset_grain_without_val_count_has_no_validation_loader(fake_media_dataset):
    data = get_media_dataset_grain("fake", dataset_source="/tmp", batch_size=8,
                                   worker_count=0, num_epochs=1)
    assert data["train_len"] == 32 and data["media_type"] == "video"
    assert "val" not in data and "val_len" not in data


def test_media_dataset_validation_split_is_ordered_and_disjoint_from_train(fake_media_dataset):
    data = get_media_dataset_grain("fake", dataset_source="/tmp", batch_size=8,
                                   val_count=8, val_batch_size=4, worker_count=0,
                                   num_epochs=1, seed=0)

    assert data["val_len"] == 8
    assert data["train_len"] == 24  # the held-out records leave the train stream

    # Validation walks its own records in canonical order, not the shuffled
    # train sampler's, and repeats identically.
    val_batches = _indices(data["val"](), 2)
    assert val_batches == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert _indices(data["val"](), 2) == val_batches

    train_indices = [i for batch in _indices(data["train"](), 3) for i in batch]
    assert set(train_indices).isdisjoint(range(8))
    assert train_indices != sorted(train_indices)  # the train sampler still shuffles


@pytest.mark.parametrize("workers", [0, 2])
def test_a_media_validation_pass_reads_every_held_out_record_once(fake_media_dataset, workers):
    """A pass is the split, once, in record order, and then it ends.

    grain's DataLoader applies its operations inside the worker processes, so
    each worker had to fill a whole batch out of its own slice of the split,
    and the unbounded num_epochs a run leaves at None let it read that slice
    again to do so.
    """
    data = get_media_dataset_grain("fake", dataset_source="/tmp", batch_size=8,
                                   val_count=24, val_batch_size=8,
                                   worker_count=workers, seed=0)

    assert _indices(data["val"](), 12) == [
        list(range(8)), list(range(8, 16)), list(range(16, 24))]
    train = [index for batch in _indices(data["train"](), 2) for index in batch]
    assert set(train).isdisjoint(range(24))


def test_media_dataset_validation_split_rejects_impossible_sizes(fake_media_dataset):
    with pytest.raises(ValueError, match="val_count"):
        get_media_dataset_grain("fake", dataset_source="/tmp", val_count=32)


def test_a_source_without_a_length_needs_an_explicit_count(monkeypatch):
    """The factory guessed a million records for such a source, so the
    sampler drew indices past the data and train_len reported the guess."""
    class Endless:
        def __getitem__(self, index):
            return {"index": index}

    class UnsizedSource(DataSource):
        def get_source(self, path_override):
            return Endless()

    monkeypatch.setitem(mediaDatasetMap, "unsized", MediaDataset(
        source=UnsizedSource(), augmenter=_PassthroughAugmenter(), media_type="video"))

    with pytest.raises(ValueError, match="count="):
        get_media_dataset_grain("unsized", dataset_source="/tmp", worker_count=0)

    data = get_media_dataset_grain("unsized", dataset_source="/tmp", batch_size=8,
                                   count=16, worker_count=0, num_epochs=1)
    assert data["train_len"] == 16
    assert sorted(i for batch in _indices(data["train"](), 3) for i in batch) == list(range(16))


@pytest.mark.parametrize("factory", ["media", "legacy"])
def test_a_count_past_the_end_of_the_source_is_refused(
        factory, fake_media_dataset, fake_legacy_dataset):
    """A count above the source became the sampler's record count, and the
    first index past the end raised inside a worker."""
    with pytest.raises(ValueError, match="count 33"):
        if factory == "media":
            get_media_dataset_grain("fake", dataset_source="/tmp", count=33,
                                    worker_count=0)
        else:
            dataloaders.get_dataset_grain("fake", dataset_source="/tmp", count=33,
                                          worker_count=0)


def test_media_dataset_grain_passes_media_scale_to_the_video_transform(fake_media_dataset):
    get_media_dataset_grain("fake", dataset_source="/tmp", media_scale=64,
                            sequence_length=4, worker_count=0)
    kwargs = fake_media_dataset.augmenter.transform_kwargs
    assert kwargs["frame_size"] == 64 and kwargs["sequence_length"] == 4


def test_media_dataset_grain_keeps_video_arguments_out_of_an_image_transform(monkeypatch):
    """An image augmenter takes an image_scale and nothing else; a clip length
    reached create_transform and raised TypeError on every image dataset."""
    dataset = MediaDataset(source=_ListSource(32), augmenter=_PassthroughAugmenter(),
                           media_type="image")
    monkeypatch.setitem(mediaDatasetMap, "fake_image", dataset)

    get_media_dataset_grain("fake_image", dataset_source="/tmp", media_scale=64,
                            sequence_length=4, worker_count=0)

    kwargs = dataset.augmenter.transform_kwargs
    assert kwargs["image_scale"] == 64 and "sequence_length" not in kwargs


def test_legacy_grain_loader_defaults_validation_to_the_local_batch(fake_legacy_dataset):
    data = dataloaders.get_dataset_grain(
        "fake", dataset_source="/tmp", batch_size=8, val_count=16,
        worker_count=0, val_worker_count=0, num_epochs=1, seed=0)

    assert _indices(data["val"](), 3) == [
        list(range(8)), list(range(8, 16))]


@pytest.fixture
def fake_legacy_dataset(monkeypatch):
    """Register a dependency-free legacy image dataset named "fake"."""
    source = _ListSource(32)
    augmenter = _PassthroughAugmenter()
    monkeypatch.setitem(dataloaders.datasetMap, "fake", {
        "source": source.get_source,
        "augmenter": lambda image_scale, method: augmenter.create_transform(),
    })
    return source


def test_legacy_grain_loader_holds_the_validation_records_out_of_training(fake_legacy_dataset):
    """The validation records are held out of the training set, so FID and CLIP
    are not measured on records the model trained on."""
    data = dataloaders.get_dataset_grain(
        "fake", dataset_source="/tmp", batch_size=8, val_batch_size=4, val_count=8,
        worker_count=0, val_worker_count=0, num_epochs=1, seed=0)

    assert data["val_len"] == 8
    assert data["train_len"] == 24

    val_indices = [i for batch in _indices(data["val"](), 2) for i in batch]
    train_indices = [i for batch in _indices(data["train"](), 3) for i in batch]

    assert val_indices == list(range(8))  # canonical order, its own pass
    assert set(train_indices).isdisjoint(val_indices)
    assert len(train_indices) == 24 and train_indices != sorted(train_indices)


@pytest.mark.parametrize("val_workers", [0, 2])
def test_a_validation_pass_reads_every_held_out_record_once(fake_legacy_dataset, val_workers):
    """A pass is the split, once, in record order, and then it ends.

    The legacy loader defaults to eight validation workers, and grain's
    DataLoader batches inside them, so each worker filled a batch out of its
    own eighth of the split by reading that eighth again. On 512 held-out
    flowers the first batch held 64 records four times over, 44 distinct
    labels where the records carry 96, and the pass never ended.
    """
    data = dataloaders.get_dataset_grain(
        "fake", dataset_source="/tmp", batch_size=8, val_batch_size=8,
        val_count=24, worker_count=0, val_worker_count=val_workers, seed=0)

    assert _indices(data["val"](), 12) == [
        list(range(8)), list(range(8, 16)), list(range(16, 24))]
    train = [index for batch in _indices(data["train"](), 2) for index in batch]
    assert set(train).isdisjoint(range(24))


def test_a_bounded_validation_pass_at_the_default_worker_count_still_yields_batches(
        fake_legacy_dataset):
    """The run's own numbers, with the epochs bounded.

    load_data holds out whole batches and leaves val_worker_count where it
    is, so a worker owns a slice smaller than one batch. grain's DataLoader
    batched inside the workers and dropped what it could not fill, which at
    these numbers is every batch, while val_len went on reporting the
    records. Nothing raised, so the epoch scored nothing.
    """
    workers = inspect.signature(
        dataloaders.get_dataset_grain).parameters["val_worker_count"].default
    assert workers == 8, "the default this test is about"

    data = dataloaders.get_dataset_grain(
        "fake", dataset_source="/tmp", batch_size=8, val_batch_size=8,
        val_count=16, worker_count=0, val_worker_count=workers, num_epochs=1,
        seed=0)

    assert data["val_len"] == 16
    assert _indices(data["val"](), 6) == [list(range(8)), list(range(8, 16))]


def test_legacy_grain_loader_without_val_count_keeps_validating_on_every_record(fake_legacy_dataset):
    data = dataloaders.get_dataset_grain("fake", dataset_source="/tmp", batch_size=8,
                                         worker_count=0, val_worker_count=0, num_epochs=1)
    assert data["train_len"] == 32 and data["val_len"] == 32


def test_legacy_grain_loader_rejects_impossible_val_counts(fake_legacy_dataset):
    with pytest.raises(ValueError, match="val_count"):
        dataloaders.get_dataset_grain("fake", dataset_source="/tmp", val_count=32)


def test_a_run_config_holds_out_a_validation_split_on_both_grain_paths(monkeypatch):
    """Recipe runs get the disjoint split without asking for it."""
    from dew.config import DataConfig

    captured = {}
    for factory in ("get_dataset_grain", "get_media_dataset_grain"):
        monkeypatch.setattr(dataloaders, factory,
                            lambda name, **kwargs: captured.setdefault(name, kwargs))

    dataloaders.load_data(DataConfig(dataset="oxford_flowers102", loader="grain"))
    dataloaders.load_data(DataConfig(dataset="voxceleb2", loader="grain"))

    defaults = DataConfig()
    expected = defaults.val_steps_per_epoch * defaults.batch_size
    assert captured["oxford_flowers102"]["val_count"] == expected
    assert captured["voxceleb2"]["val_count"] == expected


# ---------------------------------------------------------------------------------
# The global batch over JAX processes
# ---------------------------------------------------------------------------------

def test_a_global_batch_that_does_not_split_over_the_processes_is_refused(
        tmp_path, monkeypatch):
    """Integer division hid the remainder: 65 over eight processes trained on
    64 records a step while global_batch_size reported 65, and 7 gave every
    process a batch of nothing."""
    tokens = np.arange(1, 8 * 64 + 2, dtype=np.uint16)
    for name in ("train.bin", "val.bin"):
        (tmp_path / name).write_bytes(tokens.tobytes())
    monkeypatch.setattr(jax, "process_count", lambda: 8)

    def loader(batch_size):
        return dataloaders.get_token_dataset_grain(
            str(tmp_path / "train.bin"), str(tmp_path / "val.bin"),
            batch_size=batch_size, seq_len=8, worker_count=0)

    for batch_size in (65, 7):
        with pytest.raises(ValueError, match=r"batch_size.*8 JAX processes"):
            loader(batch_size)
    data = loader(64)
    assert data["local_batch_size"] == 8 and data["global_batch_size"] == 64


# ---------------------------------------------------------------------------------
# WAV decoding
# ---------------------------------------------------------------------------------

SAMPLE_RATE = 16000
NUM_SAMPLES = 1000


def _write_wav(path, samples, sample_rate=SAMPLE_RATE, channels=1, sample_width=2):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


def _tone(num_samples=NUM_SAMPLES):
    return (np.sin(np.linspace(0, 40, num_samples)) * 20000).astype("<i2")


def test_wav_decode_excludes_the_riff_header(tmp_path):
    """A flat int16 read of the file counts the 44-byte header as 22 samples."""
    samples = _tone()
    path = tmp_path / "tone.wav"
    _write_wav(path, samples)

    audio = _read_wav_mono(str(path))

    assert len(audio) == NUM_SAMPLES
    assert len(np.fromfile(path, np.int16)) == NUM_SAMPLES + 22  # what the flat read gives
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, samples / 32768.0, atol=1e-7)


def test_wav_decode_averages_stereo_to_mono(tmp_path):
    left = _tone()
    right = -left
    interleaved = np.empty(2 * NUM_SAMPLES, dtype="<i2")
    interleaved[0::2], interleaved[1::2] = left, right
    path = tmp_path / "stereo.wav"
    _write_wav(path, interleaved, channels=2)

    audio = _read_wav_mono(str(path))

    assert len(audio) == NUM_SAMPLES
    np.testing.assert_allclose(audio, np.zeros(NUM_SAMPLES), atol=1e-7)


def test_wav_decode_rejects_unsupported_sample_widths(tmp_path):
    path = tmp_path / "24bit.wav"
    _write_wav(path, np.zeros(3 * NUM_SAMPLES, dtype=np.uint8), sample_width=3)
    with pytest.raises(ValueError, match="sample width"):
        _read_wav_mono(str(path))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg binary")
def test_read_audio_ffmpeg_returns_the_exact_sample_count(tmp_path):
    samples = _tone()
    path = tmp_path / "tone.wav"
    _write_wav(path, samples)

    audio, sample_rate = read_audio_ffmpeg(str(path), target_sr=SAMPLE_RATE)

    assert sample_rate == SAMPLE_RATE
    assert audio.dtype == np.float32
    assert len(audio) == NUM_SAMPLES
    np.testing.assert_allclose(audio, samples / 32768.0, atol=1e-4)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg binary")
def test_read_audio_ffmpeg_honours_start_and_duration(tmp_path):
    _write_wav(tmp_path / "tone.wav", _tone())

    audio, _ = read_audio_ffmpeg(str(tmp_path / "tone.wav"), start_time=0.01,
                                 duration=0.02, target_sr=SAMPLE_RATE)

    assert len(audio) == int(0.02 * SAMPLE_RATE)


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
# VoxCeleb2 source
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


def test_voxceleb2_source_scans_the_tree_recursively(tmp_path):
    clips = _voxceleb_tree(tmp_path)
    records = VoxCeleb2Source().get_source(str(tmp_path))

    assert [record["video_path"] for record in records] == sorted(str(c) for c in clips)


def test_voxceleb2_source_renders_captions_from_the_template(tmp_path):
    _voxceleb_tree(tmp_path)

    templated = VoxCeleb2Source(prompt_template="a video of {identity} speaking")
    captions = {record["caption"] for record in templated.get_source(str(tmp_path))}
    assert captions == {"a video of id00012 speaking", "a video of id00015 speaking"}

    plain = VoxCeleb2Source().get_source(str(tmp_path))
    assert {record["caption"] for record in plain} == {VoxCeleb2Source.DEFAULT_PROMPT_TEMPLATE}

    # A template with a placeholder we do not supply must not explode
    odd = VoxCeleb2Source(prompt_template="a video of {speaker}")
    assert {record["caption"] for record in odd.get_source(str(tmp_path))} == {"a video of {speaker}"}


def test_voxceleb2_source_reads_the_requested_split(tmp_path):
    _voxceleb_tree(tmp_path, split="train")
    _voxceleb_tree(tmp_path, split="test", identities=("id00017",))

    assert len(VoxCeleb2Source(split="train").get_source(str(tmp_path))) == 4
    assert len(VoxCeleb2Source(split="test").get_source(str(tmp_path))) == 2


def test_voxceleb2_source_reports_missing_roots_clearly(tmp_path):
    with pytest.raises(ValueError, match="dataset root directory"):
        VoxCeleb2Source().get_source(None)
    with pytest.raises(ValueError, match="split 'train' not found"):
        VoxCeleb2Source().get_source(str(tmp_path))


def test_voxceleb2_is_registered_on_the_media_pipeline():
    dataset = mediaDatasetMap["voxceleb2"]
    assert isinstance(dataset.source, VoxCeleb2Source)
    assert isinstance(dataset.augmenter, AudioVideoAugmenter)
    assert dataset.media_type == "video"


def test_voxceleb2_records_flow_through_the_audio_video_transform(tmp_path, monkeypatch):
    """End to end with the AV reader and audio model stubbed out: the parts that
    need decord/pyav/wav2vec2 weights are exactly the parts we fake."""
    _voxceleb_tree(tmp_path)
    records = VoxCeleb2Source(prompt_template="a video of {identity}").get_source(str(tmp_path))

    frame_samples = 640
    seen = {}

    def fake_read_av_random_clip(video_path, num_frames, audio_frame_padding, random_seed):
        seen.update(video_path=video_path, num_frames=num_frames,
                    audio_frame_padding=audio_frame_padding, random_seed=random_seed)
        padded = num_frames + 2 * audio_frame_padding
        full_audio = np.zeros((padded, frame_samples), dtype=np.float32)
        framewise = np.zeros((1, num_frames, 1, frame_samples), dtype=np.float32)
        # Native resolution differs from frame_size, so the resize must happen
        frames = np.zeros((num_frames, 96, 48, 3), dtype=np.uint8)
        return framewise, full_audio, frames

    class FakeAudioProcessor:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, audio):
            return {"input_values": np.asarray(audio)[None, ...]}

    monkeypatch.setattr(videos_module, "read_av_random_clip", fake_read_av_random_clip)
    monkeypatch.setattr(videos_module, "AutoAudioProcessor", FakeAudioProcessor)

    transform = AudioVideoAugmenter().create_transform(
        frame_size=32, sequence_length=4, audio_frame_padding=1)()
    batch = transform.random_map(records[0], np.random.default_rng(0))

    assert seen["video_path"] == records[0]["video_path"]
    assert seen["num_frames"] == 4 and seen["audio_frame_padding"] == 1
    assert batch["video"].shape == (4, 32, 32, 3)          # resized to frame_size
    assert batch["caption"] == "a video of id00012"        # template survives
    assert batch["audio"]["full_audio"].shape == (6, frame_samples)
    assert batch["audio"]["framewise_audio"].shape == (1, 4, 1, frame_samples)
    assert batch["audio"]["input_values"].shape == (6, frame_samples)


# ---------------------------------------------------------------------------------
# Local video source
# ---------------------------------------------------------------------------------

def test_video_local_source_lists_and_caches_paths(tmp_path):
    """Constructing with a directory raised AttributeError: load_paths read
    self.directory before __init__ ever set it."""
    clips = _voxceleb_tree(tmp_path)
    cache_dir = tmp_path / "cache"

    source = VideoLocalSource(directory=str(tmp_path), cache_dir=str(cache_dir))
    records = source.get_source()

    assert [record["video_path"] for record in records] == sorted(str(c) for c in clips)

    # The cache key must survive a new process: hash() of a str is salted per run
    cached = list(cache_dir.iterdir())
    assert len(cached) == 1
    reloaded = VideoLocalSource(directory=str(tmp_path), cache_dir=str(cache_dir))
    assert reloaded.get_source() == records


def test_video_local_source_without_a_directory_says_so():
    with pytest.raises(ValueError, match="no directory to read"):
        VideoLocalSource().get_source()


# ---------------------------------------------------------------------------------
# Moved scripts
# ---------------------------------------------------------------------------------

def test_av_benchmark_script_imports_against_the_real_av_utils():
    """The script imports only names av_utils defines."""
    script = REPO_ROOT / "tools" / "av_benchmark.py"
    spec = importlib.util.spec_from_file_location("av_benchmark_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.benchmark_av_reading)
    assert callable(module.read_av_improved)
    assert not (REPO_ROOT / "src" / "dew" / "data" / "sources" / "av_example.py").exists()


# ---------------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------------

class _StubTokenizer:
    def __init__(self, tensor_type="np"):
        pass

    def __call__(self, captions):
        n = len(captions)
        return {"input_ids": np.zeros((n, 4), np.int32), "attention_mask": np.ones((n, 4), np.int32)}


def test_image_collate_resizes_mixed_shapes_to_the_largest(monkeypatch):
    """cv2.resize takes (width, height); passing (height, width) transposed every
    non-square image, np.stack failed, and the except branch fed zero images on."""
    monkeypatch.setattr(dataloaders, "AutoTextTokenizer", _StubTokenizer)
    collate = dataloaders.generate_collate_fn("image")
    batch = [
        {"image": np.full((16, 24, 3), 200, np.uint8), "caption": "wide"},
        {"image": np.full((20, 16, 3), 100, np.uint8), "caption": "tall"},
    ]
    out = collate(batch)
    assert out["image"].shape == (2, 20, 24, 3)
    assert out["image"][0].min() == 200 and out["image"][1].min() == 100
    assert out["text"]["input_ids"].shape == (2, 4)


def test_image_collate_raises_on_a_malformed_sample(monkeypatch):
    """The whole-batch try/except returned zeros captioned "Error processing
    image" for any failure, and a batch of zeros trains as data."""
    monkeypatch.setattr(dataloaders, "AutoTextTokenizer", _StubTokenizer)
    collate = dataloaders.generate_collate_fn("image")
    batch = [
        {"image": np.full((16, 16, 3), 200, np.uint8), "caption": "fine"},
        {"image": "not an array", "caption": "broken"},
    ]

    with pytest.raises(AttributeError):
        collate(batch)


def test_collate_raises_on_a_sample_without_a_caption(monkeypatch):
    """A sample without a caption raises instead of collating as the empty string."""
    monkeypatch.setattr(dataloaders, "AutoTextTokenizer", _StubTokenizer)
    image_collate = dataloaders.generate_collate_fn("image")
    video_collate = dataloaders.generate_collate_fn("video")

    with pytest.raises(KeyError, match="caption"):
        image_collate([{"image": np.zeros((8, 8, 3), np.uint8)}])
    with pytest.raises(KeyError, match="caption"):
        video_collate([{"video": np.zeros((2, 8, 8, 3), np.uint8)}])


def test_video_collate_raises_on_a_malformed_sample(monkeypatch):
    monkeypatch.setattr(dataloaders, "AutoTextTokenizer", _StubTokenizer)
    collate = dataloaders.generate_collate_fn("video")
    batch = [
        {"video": np.zeros((2, 8, 8, 3), np.uint8), "caption": "fine"},
        {"video": None, "caption": "broken"},
    ]

    with pytest.raises(AttributeError):
        collate(batch)


# ---------------------------------------------------------------------------------
# Stream termination: a validation pass ends, a training stream does not
# ---------------------------------------------------------------------------------

# Both grain factories build their validation sampler with the num_epochs a run
# passes, which is None, so today a pass never ends and its records come round
# again. wave/fix-val-split owns that sampler; the tests carrying this reason
# state the contract its fix has to meet, and they fail until it lands.
ENDLESS_VAL = "an endless validation pass; wave/fix-val-split owns the sampler"


def _bounded(loader, limit):
    """At most `limit` batches, and whether the stream ended inside them.

    Bounded on purpose: an endless stream then fails a count instead of
    hanging the suite.
    """
    iterator = iter(loader)
    taken = list(itertools.islice(iterator, limit))
    return taken, next(iterator, None) is None


def test_a_media_validation_pass_ends_when_the_split_runs_out(fake_media_dataset):
    """Sixteen held-out records at batch eight are two batches, then the end."""
    data = get_media_dataset_grain("fake", dataset_source="/tmp", batch_size=8,
                                   val_count=16, worker_count=0)

    batches, ended = _bounded(data["val"](), 3)
    indices = [int(i) for batch in batches for i in batch["index"]]

    assert len(batches) == 2 and ended, ENDLESS_VAL
    assert indices == list(range(16))


def test_a_legacy_validation_pass_ends_when_the_split_runs_out(fake_legacy_dataset):
    data = dataloaders.get_dataset_grain(
        "fake", dataset_source="/tmp", batch_size=8, val_count=16,
        worker_count=0, val_worker_count=0)

    batches, ended = _bounded(data["val"](), 3)
    indices = [int(i) for batch in batches for i in batch["index"]]

    assert len(batches) == 2 and ended, ENDLESS_VAL
    assert indices == list(range(16))


@pytest.mark.parametrize("worker_count", [0, 1, pytest.param(2, marks=pytest.mark.slow)])
def test_a_validation_pass_covers_the_held_out_split_once_at_any_worker_count(
        fake_media_dataset, worker_count):
    """Every held-out record reaches the metrics once, and no record the model
    trains on does.

    Which batch a record lands in is worker_count's business, since each worker
    batches its own slice of the sampler's indices. The set is not.
    """
    data = get_media_dataset_grain("fake", dataset_source="/tmp", batch_size=8,
                                   val_count=16, worker_count=worker_count)

    batches, _ = _bounded(data["val"](), 2)
    indices = [int(i) for batch in batches for i in batch["index"]]
    train, _ = _bounded(data["train"](), 2)

    assert sorted(indices) == list(range(16))
    assert {int(i) for batch in train for i in batch["index"]}.isdisjoint(indices)


@pytest.mark.parametrize("factory", ["media", "legacy"])
def test_the_training_stream_repeats_instead_of_ending(
        factory, fake_media_dataset, fake_legacy_dataset):
    """num_epochs is None in a run, and the trainer keeps asking for batches
    long after one pass over the records."""
    if factory == "media":
        data = get_media_dataset_grain("fake", dataset_source="/tmp", batch_size=8,
                                       val_count=16, worker_count=0)
    else:
        data = dataloaders.get_dataset_grain(
            "fake", dataset_source="/tmp", batch_size=8, val_count=16,
            worker_count=0, val_worker_count=0)

    epoch = data["train_len"] // 8  # the sixteen training records, in two batches
    batches, ended = _bounded(data["train"](), 3 * epoch)
    first_epoch = [int(i) for batch in batches[:epoch] for i in batch["index"]]
    later = [int(i) for batch in batches[epoch:] for i in batch["index"]]

    assert not ended and len(batches) == 3 * epoch
    assert sorted(first_epoch) == list(range(16, 32))
    assert set(later) == set(first_epoch), "the stream reads the same records again"


# ---------------------------------------------------------------------------------
# Determinism across worker counts, and a restart mid-epoch
# ---------------------------------------------------------------------------------

class _ImageSource(DataSource):
    """Deterministic images and class labels, addressed by index."""

    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.RandomState(index)
        return {"index": index,
                "image": rng.randint(0, 256, (12, 12, 3), np.uint8),
                "label": index % 5}

    def get_source(self, path_override):
        return self


class _AugmentingAugmenter(DataAugmenter):
    """The real augmentation pipeline and the real labelizer, no tokenizer.

    Those two are what a worker count could move: both draw from grain's
    per-record rng. Tokenizing a caption is deterministic and needs the hub.
    """

    def __init__(self, label_path):
        self.label_path = label_path

    def create_transform(self, image_scale=8, method=None):
        augments = image_augmentations()
        labelizer = labelizer_oxford_flowers102(self.label_path)

        class Augmenting(pygrain.RandomMapTransform):
            def random_map(self, element, rng):
                return {"index": element["index"],
                        "image": augment_image(augments, element["image"], rng),
                        "caption": labelizer(element, rng)}

        return Augmenting


def _register_augmenting(monkeypatch, tmp_path, length):
    """Register "augmenting", a media dataset of `length` records whose
    records really are augmented and captioned."""
    labels = tmp_path / "label.labels.txt"
    labels.write_text("\n".join(["rose", "tulip", "lotus", "orchid", "marigold"]))
    dataset = MediaDataset(source=_ImageSource(length),
                           augmenter=_AugmentingAugmenter(str(labels)),
                           media_type="image")
    monkeypatch.setitem(mediaDatasetMap, "augmenting", dataset)
    return dataset


@pytest.fixture
def augmenting_media_dataset(monkeypatch, tmp_path):
    return _register_augmenting(monkeypatch, tmp_path, 16)


def _rows(batch):
    """(index, pixels, caption) per record, so a comparison covers all three."""
    return [(int(batch["index"][row]), batch["image"][row].tobytes(),
             str(batch["caption"][row]))
            for row in range(len(batch["index"]))]


def _augmented(worker_count, batch=4, seed=3):
    """{record index: (pixels, caption)} for one epoch at this worker count."""
    data = get_media_dataset_grain("augmenting", dataset_source="/tmp",
                                   batch_size=batch, worker_count=worker_count,
                                   num_epochs=1, seed=seed)
    records = {}
    for b in data["train"]():
        for row in range(len(b["index"])):
            records[int(b["index"][row])] = (b["image"][row].tobytes(),
                                             str(b["caption"][row]))
    return records


@pytest.mark.slow
@pytest.mark.parametrize("worker_count", [1, 2, 4])
def test_a_records_pixels_and_caption_do_not_depend_on_worker_count(
        augmenting_media_dataset, worker_count):
    """The flip, the colour jitter and the prompt template all come from the
    per-record rng, which grain keys by record index, so the number of workers
    that produced a batch cannot change what is in it."""
    serial = _augmented(0)
    parallel = _augmented(worker_count)

    assert sorted(serial) == list(range(16))
    assert serial == parallel


def test_augmentation_really_moves_the_pixels(augmenting_media_dataset):
    """Guards the test above: identical records at every worker count would
    also be true of a pipeline that augmented nothing."""
    records = _augmented(0)
    source = _ImageSource(16)

    assert any(records[i][0] != source[i]["image"].tobytes() for i in records)
    assert len({caption for _, caption in records.values()}) > 1


@pytest.mark.parametrize("worker_count", [0, pytest.param(2, marks=pytest.mark.slow)])
def test_an_interrupted_epoch_resumes_on_exactly_the_records_it_had_not_seen(
        augmenting_media_dataset, monkeypatch, worker_count, tmp_path):
    """The trainer saves the iterator's position in its checkpoint, so a
    restored run owes the epoch its unseen records, no more and no fewer.

    The loader is built again from a source object of its own, which is what a
    resumed process has: grain validates a saved position against
    `repr(source)` and will not restore one it cannot match.

    Eight training records over two workers is two whole batches, since each
    worker batches its own slice and drops what is left over.
    """
    def loader():
        labels = tmp_path / "label.labels.txt"
        dataset = MediaDataset(source=_ImageSource(16),
                               augmenter=_AugmentingAugmenter(str(labels)),
                               media_type="image")
        monkeypatch.setitem(mediaDatasetMap, "augmenting", dataset)
        return get_media_dataset_grain("augmenting", dataset_source="/tmp",
                                       batch_size=4, worker_count=worker_count,
                                       num_epochs=1, seed=3, val_count=8)

    interrupted = iter(loader()["train"]())
    seen = _rows(next(interrupted))
    state = interrupted.get_state()
    rest, ended = _bounded(interrupted, 20)
    unseen = [row for batch in rest for row in _rows(batch)]

    restored = iter(loader()["train"]())
    restored.set_state(state)
    after, ended_again = _bounded(restored, 20)
    resumed = [row for batch in after for row in _rows(batch)]

    assert "object at 0x" not in json.loads(state)["data_source"], (
        "a source described by its address can only be restored in the process "
        "that saved it")
    assert ended and ended_again
    assert resumed == unseen, "a resumed epoch owes the same records, augmented alike"
    assert sorted(index for index, _, _ in seen + resumed) == list(range(8, 16))


def _validated(val_count, batch, **read):
    """{record index: (pixels, caption)} for one validation pass."""
    data = get_media_dataset_grain("augmenting", dataset_source="/tmp",
                                   batch_size=batch, worker_count=0, num_epochs=1,
                                   seed=3, val_count=val_count, **read)
    records = {}
    for b in data["val"]():
        for row in range(len(b["index"])):
            records[int(b["index"][row])] = (b["image"][row].tobytes(),
                                             str(b["caption"][row]))
    return records


def test_validation_pixels_do_not_depend_on_the_read_thread_count(monkeypatch, tmp_path):
    """The validation pass transforms its records inside grain's prefetch
    threads, and albumentations keeps the generators a call draws from on the
    pipeline itself, so a pipeline shared by those threads had one record's
    seed applied to another record's pixels: 75 to 82 of these 256 records
    differed between a 32-thread pass and a serial one, pass to pass, before
    each thread got a copy of its own. The captions come from the per-record
    rng directly and never moved."""
    _register_augmenting(monkeypatch, tmp_path, 512)

    serial = _validated(256, 4, read_thread_count=1, read_buffer_size=1)
    threaded = _validated(256, 4, read_thread_count=32, read_buffer_size=128)

    assert sorted(serial) == list(range(256))
    assert threaded == serial


@pytest.mark.parametrize("process_count", [2, 8])
def test_a_validation_record_does_not_depend_on_the_process_count(
        monkeypatch, tmp_path, process_count):
    """Process p of n validates records p, p + n, ... of the split. The rng
    behind a record's flip, jitter and caption has to be keyed by its place in
    the split, not in that slice, or the same seed validates one record with
    one augmentation on a single host and another on a pod: keyed by the
    slice, every record but the first differed at two processes."""
    _register_augmenting(monkeypatch, tmp_path, 64)
    alone = _validated(32, 8, read_thread_count=1, read_buffer_size=1)

    together = {}
    monkeypatch.setattr(jax, "process_count", lambda: process_count)
    for index in range(process_count):
        monkeypatch.setattr(jax, "process_index", lambda index=index: index)
        together.update(_validated(32, 8, read_thread_count=1, read_buffer_size=1))

    assert sorted(alone) == list(range(32))
    assert together == alone


# ---------------------------------------------------------------------------------
# Failure paths: a record that cannot be read stops the run
# ---------------------------------------------------------------------------------

class _RaisingSource(DataSource):
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

    def get_source(self, path_override):
        return self


def _raising_loader(monkeypatch, bad, worker_count, length=8, batch=2):
    dataset = MediaDataset(source=_RaisingSource(length, bad),
                           augmenter=_PassthroughAugmenter(), media_type="image")
    monkeypatch.setitem(mediaDatasetMap, "raising", dataset)
    return get_media_dataset_grain("raising", dataset_source="/tmp", batch_size=batch,
                                   worker_count=worker_count, num_epochs=1)


@pytest.mark.parametrize("worker_count", [0, pytest.param(2, marks=pytest.mark.slow)])
def test_a_record_that_cannot_be_read_stops_the_stream(monkeypatch, worker_count):
    """The source's own error has to reach the trainer. A pipeline that caught
    it would train on whatever it substituted, and a worker's exception is the
    easiest one to lose."""
    data = _raising_loader(monkeypatch, bad=3, worker_count=worker_count)

    delivered = []
    with pytest.raises(RuntimeError, match="record 3 is unreadable"):
        for batch in data["train"]():
            delivered.extend(int(i) for i in batch["index"])

    assert 3 not in delivered
    assert all(index in range(8) for index in delivered), "no fabricated records"


def test_a_source_that_fails_on_every_record_raises_instead_of_an_empty_batch(
        monkeypatch):
    data = _raising_loader(monkeypatch, bad=None, worker_count=0)

    iterator = iter(data["train"]())
    with pytest.raises(RuntimeError, match="is unreadable"):
        next(iterator)


# ---------------------------------------------------------------------------------
# Adversarial shapes
# ---------------------------------------------------------------------------------

def test_a_dataset_of_one_record_yields_one_batch(monkeypatch):
    dataset = MediaDataset(source=_ListSource(1), augmenter=_PassthroughAugmenter(),
                           media_type="image")
    monkeypatch.setitem(mediaDatasetMap, "single", dataset)
    data = get_media_dataset_grain("single", dataset_source="/tmp", batch_size=1,
                                   worker_count=0, num_epochs=1)

    batches, ended = _bounded(data["train"](), 2)

    assert [[int(i) for i in b["index"]] for b in batches] == [[0]] and ended


def test_a_validation_split_cannot_swallow_the_only_record(monkeypatch):
    """One record and a held-out one would leave nothing to train on."""
    dataset = MediaDataset(source=_ListSource(1), augmenter=_PassthroughAugmenter(),
                           media_type="image")
    monkeypatch.setitem(mediaDatasetMap, "single", dataset)

    with pytest.raises(ValueError, match="val_count"):
        get_media_dataset_grain("single", dataset_source="/tmp", batch_size=1,
                                worker_count=0, val_count=1)


def test_the_image_augmenter_gives_a_grayscale_record_three_channels():
    """A single-channel record has to come out the same shape as every other
    one, or the batch it lands in cannot be stacked."""
    gray = np.tile(np.arange(0, 128, 8, dtype=np.uint8), (16, 1))

    out = image_augmenter(gray, 16, cv2.INTER_NEAREST)

    assert out.shape == (16, 16, 3)
    np.testing.assert_array_equal(out[..., 0], gray)
    np.testing.assert_array_equal(out[..., 0], out[..., 2])


def test_the_image_augmenter_drops_alpha_and_hands_back_rgb():
    """Records arrive as BGR(A) from cv2; the model is trained on RGB."""
    bgra = np.dstack([np.full((16, 16), 10, np.uint8), np.full((16, 16), 20, np.uint8),
                      np.full((16, 16), 30, np.uint8), np.full((16, 16), 40, np.uint8)])

    out = image_augmenter(bgra, 16, cv2.INTER_NEAREST)

    assert out.shape == (16, 16, 3)
    np.testing.assert_array_equal(out[0, 0], [30, 20, 10])


def test_a_truncated_image_raises_rather_than_becoming_an_array():
    """cv2.imdecode hands back None for a half-written jpeg, and None resized
    to the training size would be a black record."""
    whole = np.random.RandomState(0).randint(0, 256, (16, 16, 3), np.uint8)
    encoded, buffer = cv2.imencode(".jpg", whole)
    assert encoded
    truncated = buffer.tobytes()[:len(buffer) // 2]

    decoded = cv2.imdecode(np.frombuffer(truncated, np.uint8), cv2.IMREAD_UNCHANGED)

    assert decoded is None
    with pytest.raises(cv2.error):
        image_augmenter(decoded, 16)


def test_image_collate_stacks_a_batch_of_one(monkeypatch):
    monkeypatch.setattr(dataloaders, "AutoTextTokenizer", _StubTokenizer)
    collate = dataloaders.generate_collate_fn("image")

    out = collate([{"image": np.full((8, 8, 3), 7, np.uint8), "caption": "alone"}])

    assert out["image"].shape == (1, 8, 8, 3)
    assert out["text"]["input_ids"].shape == (1, 4)


def test_image_collate_raises_when_a_grayscale_record_meets_colour_ones(monkeypatch):
    """Resizing to the largest shape cannot rescue a record with no channels,
    and a batch that quietly dropped it would train on the wrong captions."""
    monkeypatch.setattr(dataloaders, "AutoTextTokenizer", _StubTokenizer)
    collate = dataloaders.generate_collate_fn("image")
    batch = [
        {"image": np.full((8, 8, 3), 7, np.uint8), "caption": "colour"},
        {"image": np.full((8, 8), 7, np.uint8), "caption": "gray"},
    ]

    with pytest.raises(ValueError, match="same shape"):
        collate(batch)


@pytest.mark.network
def test_an_overlong_caption_is_truncated_to_the_text_context():
    """The tokenizer pads and truncates to CLIP's context, so one enormous
    caption cannot widen a batch or make it ragged."""
    from dew.inputs.processors import AutoTextTokenizer

    tokenizer = AutoTextTokenizer(tensor_type="np")
    context = tokenizer.tokenizer.model_max_length

    out = tokenizer(["x" * 10000, "short"])

    assert out["input_ids"].shape == (2, context)
    assert int(out["attention_mask"][0].sum()) == context
    assert int(out["attention_mask"][1].sum()) < context
