"""Data layer tests: the registry, the Dataset contract, lazy imports, AV decoding.

The shared environment has no HF `datasets`, decord, pyav, moviepy or
video_reader, which is the point of several of these tests: the grain paths and
`import dew.data` must not need them. Anything that genuinely requires an
optional dependency skips.
"""

import dataclasses
import hashlib
import importlib.util
import itertools
import json
import os
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
from dew.data import (Dataset, DatasetSpec, HFDatasetSource, ImageDataset, LocalVideos,
                      OxfordFlowers, TokenWindows, VoxCeleb2, local_batch)
from dew.data import Loading, images, online_loader, video
from dew.data.dataset import hold_out, train_stream, validation_pass
from dew.data.images import ImageTransform, decode_image
from dew.data.sources import av_utils
from dew.data.sources.audio_utils import _read_wav_mono, read_audio_ffmpeg
from dew.data.sources.av_utils import choose_clip_start
from dew.registry import datasets

REPO_ROOT = Path(__file__).resolve().parents[1]

WORKERS = dict(loading=Loading(workers=0, threads=1, read_buffer=1, worker_buffer=1))


# ---------------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------------

def test_every_registered_dataset_is_a_frozen_spec_with_a_loader():
    """The registry is the one table a run picks a dataset from."""
    assert datasets["oxford_flowers102"] is OxfordFlowers is datasets.OxfordFlowers
    assert datasets["voxceleb2"] is VoxCeleb2 and datasets["token_windows"] is TokenWindows
    for name in datasets:
        spec = datasets[name]
        assert issubclass(spec, DatasetSpec) and dataclasses.is_dataclass(spec)
        assert spec.__dataclass_params__.frozen, name
        assert callable(spec.load)
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


def test_a_dataset_records_no_augmentation_mode_in_the_environment():
    """Augmentation is a field of the spec, read by the transform it builds."""
    assert OxfordFlowers().augmentation == "flip_jitter"
    with pytest.raises(ValueError, match="not one of none, flip_only, flip_jitter"):
        images.image_augmentations("jitter")


# ---------------------------------------------------------------------------------
# Lazy imports: the data layer must not drag in HF datasets / opencv / decord
# ---------------------------------------------------------------------------------

def test_importing_dew_data_pulls_in_no_heavy_dependencies():
    """`import dew.data` registers every dataset and must not reach the
    streaming stack, cv2, albumentations or tensorflow_datasets; a run that
    only reads token files pays for none of them. The hub source has the same
    duty: naming a hub dataset resolves without the streaming extra, only
    reading one needs it. Nothing from dew.inputs or dew.diffusion either:
    dew.config imports this package for the registry's union."""
    probe = (
        "import sys, dew.data;"
        "heavy = [m for m in ('datasets', 'cv2', 'albumentations', 'tensorflow_datasets',"
        " 'decord', 'transformers', 'dew.data.online_loader', 'dew.inputs', 'dew.diffusion',"
        " 'dew.sampling', 'wandb') if m in sys.modules];"
        "assert not heavy, heavy;"
        "from dew.registry import datasets;"
        "assert 'oxford_flowers102' in datasets and 'packed_tokens' in datasets;"
        "dew.data.HFDatasetSource(name='acme/pets');"
        "assert 'datasets' not in sys.modules"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), JAX_PLATFORMS="cpu")
    result = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT,
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr


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
    already_imported = "dew.data.online_loader" in sys.modules
    with pytest.raises(ValueError, match="sources="):
        dew.data.OnlineImages().load(batch=4)
    if not already_imported:
        assert "dew.data.online_loader" not in sys.modules


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


class _Endless:
    def __getitem__(self, index):
        return {"index": index, "image": np.zeros((4, 4, 3), np.uint8)}


@dataclasses.dataclass(frozen=True)
class Unsized(ImageDataset):
    def source(self):
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
    seen = _indices(first, 2)
    state = first.get_state()
    rest = _indices(first, 2)

    resumed = data.train()
    resumed.set_state(state)
    assert _indices(resumed, 2) == rest
    assert sorted(i for batch in seen + rest for i in batch) == list(range(32))


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

    assert len(VoxCeleb2(path=str(tmp_path), split="train").source()) == 4
    assert len(VoxCeleb2(path=str(tmp_path), split="test").source()) == 2


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
    """End to end with the AV reader and audio model stubbed out: the parts that
    need decord/pyav/wav2vec2 weights are exactly the parts we fake."""
    _voxceleb_tree(tmp_path)
    spec = VoxCeleb2(path=str(tmp_path), prompt_template="a video of {identity}",
                     frame_size=32, frames=4, audio_padding=1)
    records = spec.source()

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

    monkeypatch.setattr(av_utils, "read_av_random_clip", fake_read_av_random_clip)
    monkeypatch.setattr(video, "AutoAudioProcessor", FakeAudioProcessor)

    batch = video.AudioVideoTransform(spec).random_map(records[0], np.random.default_rng(0))

    assert seen["video_path"] == records[0]["video_path"]
    assert seen["num_frames"] == 4 and seen["audio_frame_padding"] == 1
    assert batch["video"].shape == (4, 32, 32, 3)          # resized to frame_size
    assert batch["caption"] == "a video of id00012"
    assert batch["audio"]["full_audio"].shape == (6, frame_samples)
    assert batch["audio"]["framewise_audio"].shape == (1, 4, 1, frame_samples)
    assert batch["audio"]["input_values"].shape == (6, frame_samples)


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
# The streaming collate
# ---------------------------------------------------------------------------------

def keep_captions(captions):
    """A caption reader that hands the words back, so a test reads what the
    dataset wrote before a run's encoder tokenizes it."""
    return {"caption": np.asarray(captions)}


def test_image_collate_resizes_mixed_shapes_to_the_largest(monkeypatch):
    """cv2.resize takes (width, height); passing (height, width) transposed every
    non-square image, np.stack failed, and the except branch fed zero images on."""
    collate = online_loader.generate_collate_fn("image")
    batch = [
        {"image": np.full((16, 24, 3), 200, np.uint8), "caption": "wide"},
        {"image": np.full((20, 16, 3), 100, np.uint8), "caption": "tall"},
    ]
    out = collate(batch)
    assert out["image"].shape == (2, 20, 24, 3)
    assert out["image"][0].min() == 200 and out["image"][1].min() == 100
    assert list(out["caption"]) == ["wide", "tall"]


def test_image_collate_raises_on_a_malformed_sample(monkeypatch):
    """The whole-batch try/except returned zeros captioned "Error processing
    image" for any failure, and a batch of zeros trains as data."""
    collate = online_loader.generate_collate_fn("image")
    batch = [
        {"image": np.full((16, 16, 3), 200, np.uint8), "caption": "fine"},
        {"image": "not an array", "caption": "broken"},
    ]

    with pytest.raises(AttributeError):
        collate(batch)


def test_collate_raises_on_a_sample_without_a_caption(monkeypatch):
    """A sample without a caption raises instead of collating as the empty string."""
    image_collate = online_loader.generate_collate_fn("image")
    video_collate = online_loader.generate_collate_fn("video")

    with pytest.raises(KeyError, match="caption"):
        image_collate([{"image": np.zeros((8, 8, 3), np.uint8)}])
    with pytest.raises(KeyError, match="caption"):
        video_collate([{"video": np.zeros((2, 8, 8, 3), np.uint8)}])


def test_video_collate_raises_on_a_malformed_sample(monkeypatch):
    collate = online_loader.generate_collate_fn("video")
    batch = [
        {"video": np.zeros((2, 8, 8, 3), np.uint8), "caption": "fine"},
        {"video": None, "caption": "broken"},
    ]

    with pytest.raises(AttributeError):
        collate(batch)


def test_image_collate_stacks_a_batch_of_one(monkeypatch):
    collate = online_loader.generate_collate_fn("image")
    out = collate([{"image": np.zeros((8, 8, 3), np.uint8), "caption": "one"}])
    assert out["image"].shape == (1, 8, 8, 3)
    assert list(out["caption"]) == ["one"]


def test_image_collate_raises_when_a_grayscale_record_meets_colour_ones(monkeypatch):
    """Resizing to the largest shape cannot rescue a record with no channels,
    and stacking it silently would be worse."""
    collate = online_loader.generate_collate_fn("image")
    batch = [
        {"image": np.zeros((8, 8, 3), np.uint8), "caption": "colour"},
        {"image": np.zeros((8, 8), np.uint8), "caption": "gray"},
    ]
    with pytest.raises(ValueError):
        collate(batch)


# ---------------------------------------------------------------------------------
# Determinism across worker counts, and a restart mid-epoch
# ---------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Augmenting(ImageDataset):
    """Deterministic images and class captions, addressed by index, through
    the real image transform: resize, flip, jitter and the prompt template
    all draw from grain's per-record rng, which is what a worker count could
    move. The captions are read back as text, so one stays comparable after a
    trip through a worker process."""

    length: int = 16

    def source(self):
        return _Images(self.length)

    def record(self, element, rng):
        template = images.PROMPT_TEMPLATES[int(rng.integers(len(images.PROMPT_TEMPLATES)))]
        name = ["rose", "tulip", "lotus", "orchid", "marigold"][element["index"] % 5]
        return element["image"], template.format(name), element["index"]


class _Images:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
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


def test_augmentation_really_moves_the_pixels():
    """Guards the test above: identical records at every worker count would
    also be true of a pipeline that augmented nothing."""
    records = _augmented(0)
    source = _Images(16)

    assert any(records[i][0] != cv2.resize(source[i]["image"], (8, 8),
                                           interpolation=cv2.INTER_AREA).tobytes()
               for i in records)
    assert len({caption for _, caption in records.values()}) > 1


@pytest.mark.parametrize("worker_count", [0, pytest.param(2, marks=pytest.mark.slow)])
def test_an_interrupted_epoch_resumes_on_exactly_the_records_it_had_not_seen(
        worker_count):
    """The trainer saves the iterator's position in its checkpoint, so a
    restored run owes the epoch its unseen records, no more and no fewer.

    The loader is built again from a source object of its own, which is what a
    resumed process has: grain validates a saved position against
    `repr(source)` and will not restore one it cannot match.

    Eight training records over two workers is two whole batches, since each
    worker batches its own slice and drops what is left over.
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

    assert "object at 0x" not in json.loads(state)["data_source"], (
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
    threads, and albumentations keeps the generators a call draws from on the
    pipeline itself, so a pipeline shared by those threads had one record's
    seed applied to another record's pixels: 75 to 82 of these 256 records
    differed between a 32-thread pass and a serial one, pass to pass, before
    each thread got a copy of its own. The captions come from the per-record
    rng directly and never moved."""
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
    to the training size would be a black record."""
    whole = np.random.RandomState(0).randint(0, 256, (16, 16, 3), np.uint8)
    encoded, buffer = cv2.imencode(".jpg", whole)
    assert encoded
    truncated = buffer.tobytes()[:len(buffer) // 2]

    with pytest.raises(cv2.error):
        decode_image(truncated)


def test_the_image_transform_resizes_augments_and_captions_one_record():
    """What every image dataset hands the loader: the resized uint8 image,
    the caption text and, for a class-labelled record, its index."""
    spec = Augmenting(image_size=8, augmentation="none")
    element = _Images(16)[5]

    out = ImageTransform(spec).random_map(element, np.random.default_rng(0))

    np.testing.assert_array_equal(
        out["image"], cv2.resize(element["image"], (8, 8), interpolation=cv2.INTER_AREA))
    assert out["caption"] == "A photo of a rose flower" and out["label"] == 5
    assert out["label"].dtype == np.int32


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
    assert [str(caption) for caption in captions] and captions.shape == (4,)

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
