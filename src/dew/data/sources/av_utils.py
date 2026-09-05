"""One random clip of frames and aligned audio out of a video file.

moviepy is the reader because it is what the `av` extra installs and it
reads from wheels alone; it drives ffmpeg, so nothing here holds a decoder
open across records.
"""

from __future__ import annotations

import subprocess

import numpy as np


def choose_clip_start(total_frames: int, num_frames: int, padding: int,
                      rng: np.random.Generator) -> int:
    """A start for a clip of `num_frames` that leaves `padding` frames free
    on either side, drawn from `rng`, or the only start that fits.

    The generator is an argument because np.random.seed() inside a
    data-loading worker reseeds the process-global RNG and perturbs every
    other consumer of np.random.
    """
    latest = total_frames - num_frames - padding
    if latest <= padding:
        return padding
    return int(rng.integers(padding, latest))


def audio_window(path: str, start: float, duration: float, sample_rate: int) -> np.ndarray:
    """The sound of `path` from `start` for `duration` seconds, mono at
    `sample_rate`, float32 in [-1, 1].

    ffmpeg does the seek, the resampling and the downmix, through the binary
    moviepy drives. moviepy's own audio reader asks for stereo at 44.1 kHz
    whatever the track is, and ffmpeg's mono-to-stereo upmix takes 3 dB off
    a mono track; sampling that stream at another rate then picks the
    nearest 44.1 kHz sample, which aliases and is never sample-exact.
    """
    from moviepy.config import FFMPEG_BINARY

    done = subprocess.run(
        [FFMPEG_BINARY, "-loglevel", "error", "-nostdin", "-ss", f"{start:.6f}",
         "-t", f"{duration:.6f}", "-i", path, "-vn", "-f", "s16le", "-ac", "1",
         "-ar", str(sample_rate), "-"],
        capture_output=True)
    if done.returncode:
        raise ValueError(f"ffmpeg could not read the audio of {path}: "
                         f"{done.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(done.stdout, np.int16).astype(np.float32) / 32768.0


def read_av_random_clip(path: str, *, num_frames: int, audio_padding: int, seed: int,
                        sample_rate: int = 16000, fps: float = 25.0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """`num_frames` consecutive frames of `path` from a start `seed` picks,
    and the audio under them with `audio_padding` frames more on each side.

    Frames are sampled at `fps` and come back as uint8 `[num_frames, H, W, 3]`
    RGB. The audio is decoded mono at `sample_rate` and cut into one row per
    video frame, float32 in [-1, 1] of shape `[num_frames + 2 * audio_padding,
    sample_rate / fps]`, so row `audio_padding + i` is the sound under frame
    `i`; the rate therefore has to be a whole number of samples per frame.
    """
    from moviepy import VideoFileClip

    samples_per_frame = sample_rate / fps
    if not samples_per_frame.is_integer():
        raise ValueError(
            f"{sample_rate} Hz at {fps} fps is {samples_per_frame} audio samples a "
            "frame, and a row of audio per frame needs a whole number")
    samples_per_frame = int(samples_per_frame)
    padded_frames = num_frames + 2 * audio_padding

    # The audio is not opened here: a file without a track fails in
    # audio_window with ffmpeg's own words for it.
    with VideoFileClip(path, audio=False) as video:
        if video.duration is None:
            raise ValueError(f"{path} reports no duration, so no clip can be cut from it")
        total_frames = int(video.duration * fps)
        if total_frames < padded_frames:
            raise ValueError(
                f"{path} has {total_frames} frames at {fps} fps and a clip of "
                f"{num_frames} with {audio_padding} of padding needs {padded_frames}")
        start = choose_clip_start(total_frames, num_frames, audio_padding,
                                  np.random.default_rng(seed))
        # One frame per index, asked for by its own time: the reader seeks
        # to the first and steps to the rest, where a subclip's iterator
        # enumerates times from a float duration and can come up one short.
        frames = np.stack([video.get_frame((start + index) / fps) for index in range(num_frames)])

    samples = audio_window(path, (start - audio_padding) / fps, padded_frames / fps, sample_rate)
    needed = padded_frames * samples_per_frame
    if len(samples) < needed:
        raise ValueError(
            f"{path} gave {len(samples)} audio samples where the clip needs {needed}")
    return frames, samples[:needed].reshape(padded_frames, samples_per_frame)
