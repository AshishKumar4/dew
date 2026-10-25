"""The clip stages, on clips whose frames and sound carry their own index.

Every frame of the synthesized clip is a flat grey of five times its index,
and the sound under frame f is the constant sample 100 f, so a record says
which frames it holds and which frames its sound is under. The clips are
muxed losslessly in AVI (ffv1 video, PCM sound at the reader's rate), where
ffmpeg's seek into the PCM track is sample-exact, so the sound comes back
sample for sample, beyond aligned. The readers need moviepy, the `av` extra,
and its bundled ffmpeg writes the fixtures, so nothing here depends on a
binary on PATH.
"""

import subprocess

import moviepy.config as moviepy_config
import numpy as np
import pytest

from dew.data import Memory, audio, video

FPS = 25
SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
CLIP_FRAMES = 50
# Small enough that a frame's grey survives ffv1 exactly, and an index
# recovered as round(grey / 5) is never ambiguous.
GREY_STEP = 5
SAMPLE_STEP = 100


def _mux(path, *, frames=CLIP_FRAMES, sound=CLIP_FRAMES, delay=0.0):
    """A clip of `frames` frames over `sound` frames' worth of sound starting `delay` seconds in;
    either count may be 0."""
    inputs, codecs = [], []
    if frames:
        pixels = np.stack([np.full((32, 32, 3), GREY_STEP * index, np.uint8) for index in range(frames)])
        path.with_suffix(".rgb").write_bytes(pixels.tobytes())
        inputs += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "32x32", "-r", str(FPS),
                   "-i", str(path.with_suffix(".rgb"))]
        codecs += ["-c:v", "ffv1"]
    if sound:
        samples = np.repeat(np.arange(sound, dtype=np.int16) * SAMPLE_STEP, SAMPLES_PER_FRAME)
        path.with_suffix(".pcm").write_bytes(samples.tobytes())
        inputs += ["-itsoffset", str(delay), "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
                   "-i", str(path.with_suffix(".pcm"))]
        codecs += ["-c:a", "pcm_s16le"]
    subprocess.run([moviepy_config.FFMPEG_BINARY, "-y", "-loglevel", "error", *inputs, *codecs, str(path)],
                   check=True)
    return str(path)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    root = tmp_path_factory.mktemp("av")
    return {"clip": _mux(root / "clip.avi"), "sound": _mux(root / "sound.wav", frames=0),
            "silent": _mux(root / "silent.avi", sound=0), "short": _mux(root / "short.avi", sound=20),
            "exact": _mux(root / "exact.avi", frames=12, sound=12),
            # Matroska keeps a stream's start, where AVI starts every stream with the file.
            "late": _mux(root / "late.mkv", delay=0.5)}


def _records(chain):
    return list(chain.repeat(1))


def _frame_indices(frames):
    return [round(int(frame[16, 16, 0]) / GREY_STEP) for frame in frames]


def _sound(first, count):
    """The fixture's samples from `first` on, `count` of them, as the reader returns them."""
    index = np.arange(first, first + count)
    return (index // SAMPLES_PER_FRAME * SAMPLE_STEP).astype(np.float32) / 32768


def _sound_from(wave, first):
    """Whether `wave` is the fixture's sound from sample `first`, to ffmpeg's rounding of the seek to
    the nearest sample (one sample is 62.5 microseconds)."""
    return any(np.array_equal(wave, _sound(first + shift, len(wave))) for shift in (-1, 0, 1))


def test_a_clip_s_frames_and_sound_come_from_the_window_one_draw_picks(media):
    """Every record draws its own window two frames clear of either end; its 8 frames are the
    frames at the window's start and the 7 after, and its sound is the track from 2 frames before
    the window to 2 after it, sample for sample. The clip arrives as bytes or as a path."""
    with open(media["clip"], "rb") as file:
        clip = file.read()
    rows = [{"mp4": clip if index % 2 else media["clip"], "id": index} for index in range(8)]
    chain = (Memory(rows).random_filter_map(video.PickClip(8 / FPS, margin=2 / FPS))
             .filter_map(video.DecodeVideo(8, fps=FPS, size=32))
             .filter_map(audio.DecodeAudio(SAMPLE_RATE, context=2 / FPS))
             .select_columns("clip", "video", "audio", "audio_valid"))

    records = _records(chain)

    assert len(records) == 8
    assert len({float(record["clip"][0]) for record in records}) == 8
    for record in records:
        start, stop = record["clip"]
        # The window is written as [start, start + its length] seconds in float64.
        assert start >= 2 / FPS and stop <= (CLIP_FRAMES - 2) / FPS and stop == start + 8 / FPS
        assert record["video"].shape == (8, 32, 32, 3) and record["video"].dtype == np.uint8
        assert _frame_indices(record["video"]) == [int(FPS * (start + i / FPS) + 1e-5) for i in range(8)]
        assert record["audio"].shape == (12 * SAMPLES_PER_FRAME,) and record["audio"].dtype == np.float32
        assert _sound_from(record["audio"], round((start - 2 / FPS) * SAMPLE_RATE))
        assert record["audio_valid"].all()


def test_a_sound_file_reads_through_the_same_stages(media):
    chain = (Memory([{"wav": media["sound"], "id": index} for index in range(4)])
             .random_filter_map(video.PickClip(0.5, field="wav"))
             .filter_map(audio.DecodeAudio(SAMPLE_RATE, field="wav")))

    for record in _records(chain):
        assert record["audio"].shape == (SAMPLE_RATE // 2,) and record["audio_valid"].all()
        assert _sound_from(record["audio"], round(float(record["clip"][0]) * SAMPLE_RATE))
        assert "video" not in record


def test_sound_a_window_lacks_is_rejected_or_silence_marked_invalid(media):
    """A clip with no track, one whose track ends inside the window, a window whose context opens
    before the file and a track that starts after its video: rejected, or under
    `missing="silence"` held as silence that `audio_valid` marks, the sound there sample for sample
    where it is."""
    window = np.asarray([0.6, 0.92])
    rows = [{"mp4": media["silent"], "clip": window, "id": 0},
            {"mp4": media["short"], "clip": window, "id": 1},
            {"mp4": media["clip"], "clip": np.asarray([0.0, 0.32]), "id": 2},
            {"mp4": media["clip"], "clip": window, "id": 3},
            {"mp4": media["late"], "clip": window - 0.2, "id": 4}]
    context = 2 / FPS

    def read(missing):
        chain = Memory(rows).filter_map(audio.DecodeAudio(SAMPLE_RATE, context=context, missing=missing))
        return {int(record["id"]): record for record in _records(chain)}

    assert list(read("reject")) == [3]
    silence = read("silence")
    silent, short, early, whole, late = (silence[index] for index in range(5))
    samples = round((0.32 + 2 * context) * SAMPLE_RATE)
    assert not silent["audio_valid"].any() and not silent["audio"].any()
    # The short clip's sound ends at 0.8 seconds, 0.28 seconds into the widened window.
    held = round(0.28 * SAMPLE_RATE)
    assert short["audio_valid"].tolist() == [True] * held + [False] * (samples - held)
    assert _sound_from(short["audio"][:held], round(0.52 * SAMPLE_RATE)) and not short["audio"][held:].any()
    # The early window's context opens 2 frames before the file does.
    lead = 2 * SAMPLES_PER_FRAME
    assert early["audio_valid"].tolist() == [False] * lead + [True] * (samples - lead)
    assert not early["audio"][:lead].any() and _sound_from(early["audio"][lead:], 0)
    assert whole["audio_valid"].all() and _sound_from(whole["audio"], round(0.52 * SAMPLE_RATE))
    # The late track starts half a second in, 0.18 seconds into the widened window from 0.32.
    held = round(0.18 * SAMPLE_RATE)
    assert late["audio_valid"].tolist() == [False] * held + [True] * (samples - held)
    assert not late["audio"][:held].any() and _sound_from(late["audio"][held:], 0)


def test_media_too_short_or_unreadable_is_dropped_and_frames_past_their_window_raise(media):
    rows = [{"mp4": media["clip"], "id": 0}, {"mp4": b"not a clip" * 64, "id": 1}]

    assert _records(Memory(rows).random_filter_map(video.PickClip(3.0))) == []
    with pytest.raises(ValueError):
        video.PickClip(3.0, rejects=False).random_map(rows[0], np.random.default_rng(0))
    picked = _records(Memory(rows).random_filter_map(video.PickClip(1.0)))
    assert [int(record["id"]) for record in picked] == [0]
    with pytest.raises(ValueError):
        video.DecodeVideo(10, fps=FPS).map({"mp4": media["clip"], "clip": np.asarray([0.0, 8 / FPS])})


def test_a_clip_that_fits_its_window_exactly_is_read(media):
    """Twelve frames hold an eight-frame window two frames clear of either end in one place only."""
    chain = (Memory([{"mp4": media["exact"], "id": 0}])
             .random_filter_map(video.PickClip(8 / FPS, margin=2 / FPS))
             .filter_map(video.DecodeVideo(8, fps=FPS, size=32))
             .filter_map(audio.DecodeAudio(SAMPLE_RATE, context=2 / FPS)))

    (record,) = _records(chain)

    # Twelve frames hold one eight-frame window two frames clear of either end: it starts at the margin.
    assert record["clip"].tolist() == [2 / FPS, 2 / FPS + 8 / FPS]
    assert _frame_indices(record["video"]) == list(range(2, 10))
    assert record["audio_valid"].all() and _sound_from(record["audio"], 0)


def test_a_window_is_the_record_s_own_draw(media):
    """The same record draws the same window in any read, without the global generator."""
    np.random.seed(1234)
    state = np.random.get_state()[1].copy()
    pick = video.PickClip(0.4, margin=0.1)

    def draws():
        record = {"mp4": media["clip"]}
        return [pick.random_map(record, np.random.default_rng(seed))["clip"] for seed in range(16)]

    first, again = draws(), draws()

    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
    starts = np.asarray([clip[0] for clip in first])
    assert starts.min() >= 0.1 and starts.max() <= 2.0 - 0.4 - 0.1 and len(set(starts.tolist())) == 16
    assert np.array_equal(state, np.random.get_state()[1])


def test_the_clips_preset_batches_a_clip_and_its_sound_by_the_schema_and_passes_other_fields(media):
    """presets.clips is the public steps written out: each record's window, its frames and its
    sound, the media read and gone, other fields stacked beside them; a silent clip keeps its
    frames with its sound marked invalid, and media that does not read is dropped."""
    from dew.data import presets

    rows = [{"mp4": media["clip"], "id": 0}, {"mp4": media["silent"], "id": 1},
            {"mp4": b"not media", "id": 2}, {"mp4": media["clip"], "id": 3}]
    written = (Memory(rows).random_filter_map(video.PickClip(8 / FPS))
               .filter_map(video.DecodeVideo(8, fps=FPS, size=32))
               .filter_map(audio.DecodeAudio(SAMPLE_RATE, missing="silence"))
               .remove_columns("mp4", "clip").batch(2))
    batched = Memory(rows).pipe(presets.clips, 8 / FPS, frames=8, fps=FPS, size=32, batch=2,
                                sample_rate=SAMPLE_RATE)
    batches, expected = _records(batched), _records(written)
    assert len(batches) == len(expected) == 1
    (batch,), (reference,) = batches, expected
    assert set(batch) == {"video", "audio", "audio_valid", "id"}
    for name in batch:
        np.testing.assert_array_equal(batch[name], reference[name])
    assert batch["video"].shape == (2, 8, 32, 32, 3) and batch["video"].dtype == np.uint8
    assert batch["audio"].shape == (2, 8 * SAMPLES_PER_FRAME) and batch["audio"].dtype == np.float32
    assert batch["id"].tolist() == [0, 1]
    assert batch["audio_valid"][0].all() and not batch["audio_valid"][1].any()
    silent = _records(Memory(rows[:2]).pipe(presets.clips, 8 / FPS, frames=8, fps=FPS, size=32, batch=2))
    assert set(silent[0]) == {"video", "id"}
