"""Frames and sound of a media file over a time window, which `video.pick_clip` draws once for both.

moviepy is the reader because it is what the `av` extra installs and it
reads from wheels alone; it drives ffmpeg, so nothing here holds a decoder
open across records.
"""

from __future__ import annotations

import subprocess

import numpy as np

FPS = 25.0
"""The rate `VideoData` samples clips at."""


def media_info(path: str) -> tuple[float, bool, bool]:
    """`path`'s duration in seconds and whether it holds a video and an audio stream, from ffmpeg's
    reading of the container, so a sound file answers too."""
    from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos

    try:
        probed = ffmpeg_parse_infos(path, check_duration=True)
    except (OSError, IndexError) as error:
        raise ValueError(f"ffmpeg could not read {path}: {error}") from error
    duration = probed.get("duration")
    if not duration:
        raise ValueError(f"{path} reports no duration, so no clip can be cut from it")
    return float(duration), bool(probed.get("video_found")), bool(probed.get("audio_found"))


def audio_window(path: str, start: float, duration: float, sample_rate: int) -> np.ndarray:
    """The sound of `path` from `start` for `duration` seconds, mono at
    `sample_rate`, float32 in [-1, 1]; shorter where the track ends first.

    ffmpeg does the seek, the resampling and the downmix, through the binary
    moviepy drives. moviepy's own audio reader asks for stereo at 44.1 kHz
    whatever the track is, and ffmpeg's mono-to-stereo upmix takes 3 dB off
    a mono track; sampling that stream at another rate then picks the
    nearest 44.1 kHz sample, which aliases and is never sample-exact.

    The seek lands a second early and decodes up to `start`: seeking straight
    to it, ffmpeg started an AVI's track at the next packet, 73 or 261
    samples late in 2 of 80 windows, where this read was within a sample of
    the asked start in all 80.
    """
    from moviepy.config import FFMPEG_BINARY

    early = max(start - 1.0, 0.0)
    done = subprocess.run(
        [FFMPEG_BINARY, "-loglevel", "error", "-nostdin", "-ss", f"{early:.6f}", "-i", path,
         "-ss", f"{start - early:.6f}", "-t", f"{duration:.6f}", "-vn", "-f", "s16le", "-ac", "1",
         "-ar", str(sample_rate), "-"],
        capture_output=True)
    if done.returncode:
        raise ValueError(f"ffmpeg could not read the audio of {path}: "
                         f"{done.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(done.stdout, np.int16).astype(np.float32) / 32768.0


def read_frames(path: str, start: float, count: int, fps: float) -> np.ndarray:
    """`count` frames of `path` at `start + i / fps` seconds, as uint8 `[count, H, W, 3]` RGB.

    Each frame is asked for by its own time: the reader seeks to the first
    and steps to the rest, where a subclip's iterator enumerates times from a
    float duration and can come up one short.
    """
    from moviepy import VideoFileClip

    with VideoFileClip(path, audio=False) as video:
        decoded: list[np.ndarray] = []
        for index in range(count):
            timestamp = start + index / fps
            frame = video.get_frame(timestamp)
            if frame is None:
                raise ValueError(f"{path} returned no video frame at {timestamp} seconds")
            decoded.append(frame)
    return np.stack(decoded)
