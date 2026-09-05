"""The audio-video clip reader, on a clip whose frames and audio carry
their own index.

Every frame of the synthesized clip is a flat grey of five times its index,
and the audio under frame f is the constant sample 100 f, so a clip that
comes back says which frames it holds and which frames its audio is under.
The clip is muxed losslessly (ffv1 video, PCM audio at the reader's rate),
so the audio has to come back sample-exact, not just aligned. The reader
needs moviepy, which is the `av` extra, and its bundled ffmpeg writes the
fixture, so nothing here depends on a binary on PATH.
"""

import subprocess

import numpy as np
import pytest

from dew.data.sources.av_utils import audio_window, choose_clip_start, read_av_random_clip

moviepy_config = pytest.importorskip("moviepy.config", reason="needs the av extra")

FPS = 25
SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
CLIP_FRAMES = 50
# Small enough that a frame's cubic-free grey survives ffv1 exactly, and an
# index recovered as round(grey / 5) is never ambiguous.
GREY_STEP = 5
SAMPLE_STEP = 100


def _clip(directory, name="clip.mkv", audio=True):
    frames = np.stack([np.full((32, 32, 3), GREY_STEP * index, np.uint8)
                       for index in range(CLIP_FRAMES)])
    samples = np.repeat(np.arange(CLIP_FRAMES, dtype=np.int16) * SAMPLE_STEP, SAMPLES_PER_FRAME)
    (directory / "frames.raw").write_bytes(frames.tobytes())
    (directory / "audio.raw").write_bytes(samples.tobytes())
    path = directory / name
    video = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "32x32", "-r", str(FPS),
             "-i", str(directory / "frames.raw")]
    sound = (["-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", str(directory / "audio.raw"),
              "-c:a", "pcm_s16le"] if audio else ["-an"])
    subprocess.run([moviepy_config.FFMPEG_BINARY, "-y", "-loglevel", "error", *video, *sound,
                    "-c:v", "ffv1", str(path)], check=True)
    return str(path)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    return _clip(tmp_path_factory.mktemp("av"))


def _frame_indices(frames):
    return [int(round(int(frame[16, 16, 0]) / GREY_STEP)) for frame in frames]


def _audio_indices(audio):
    return [int(round(float(np.median(row)) * 32768 / SAMPLE_STEP)) for row in audio]


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_a_clip_holds_the_frames_the_seed_picks_and_the_audio_under_them(clip, seed):
    """Frames [start, start + N), audio rows [start - P, start + N + P), and
    every audio row bit-exact: the clip is lossless and the reader asks
    ffmpeg for the track at its own rate. Read through moviepy's audio
    reader the rows came back scaled by 0.707 (its stereo upmix of a mono
    track) and sampled off a 44.1 kHz stream, so no row was exact."""
    start = choose_clip_start(CLIP_FRAMES, 8, 2, np.random.default_rng(seed))

    frames, audio = read_av_random_clip(clip, num_frames=8, audio_padding=2, seed=seed)

    assert frames.shape == (8, 32, 32, 3) and frames.dtype == np.uint8
    assert audio.shape == (12, SAMPLES_PER_FRAME) and audio.dtype == np.float32
    assert _frame_indices(frames) == list(range(start, start + 8))
    assert _audio_indices(audio) == list(range(start - 2, start + 10))
    expected = np.arange(start - 2, start + 10, dtype=np.float32)[:, None] * SAMPLE_STEP / 32768
    assert np.array_equal(audio, np.broadcast_to(expected, audio.shape))


def test_the_same_seed_reads_the_same_clip_without_touching_the_global_rng(clip):
    np.random.seed(99)
    global_state = np.random.get_state()[1].copy()

    first = read_av_random_clip(clip, num_frames=8, audio_padding=1, seed=3)
    second = read_av_random_clip(clip, num_frames=8, audio_padding=1, seed=3)
    other = read_av_random_clip(clip, num_frames=8, audio_padding=1, seed=4)

    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    assert not np.array_equal(first[0], other[0])
    assert np.array_equal(global_state, np.random.get_state()[1])


def test_an_audio_window_is_the_samples_between_its_times(clip):
    """The window is cut by time, mono at the rate asked for; a clip a few
    frames in starts at exactly that frame's first sample."""
    window = audio_window(clip, 3 / FPS, 2 / FPS, SAMPLE_RATE)

    assert window.shape == (2 * SAMPLES_PER_FRAME,) and window.dtype == np.float32
    assert np.array_equal(window[:SAMPLES_PER_FRAME], np.full(SAMPLES_PER_FRAME, 3 * SAMPLE_STEP / 32768, np.float32))
    assert np.array_equal(window[SAMPLES_PER_FRAME:], np.full(SAMPLES_PER_FRAME, 4 * SAMPLE_STEP / 32768, np.float32))


def test_a_clip_longer_than_the_video_is_refused(clip):
    with pytest.raises(ValueError, match="needs 54"):
        read_av_random_clip(clip, num_frames=50, audio_padding=2, seed=0)


def test_a_rate_that_is_not_whole_samples_per_frame_is_refused(clip):
    with pytest.raises(ValueError, match="whole number"):
        read_av_random_clip(clip, num_frames=8, audio_padding=2, seed=0, fps=30.0)


def test_a_video_without_audio_is_refused(tmp_path):
    silent = _clip(tmp_path, name="silent.mkv", audio=False)
    with pytest.raises(ValueError, match="does not contain any stream"):
        read_av_random_clip(silent, num_frames=8, audio_padding=2, seed=0)
