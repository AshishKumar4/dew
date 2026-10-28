"""Frames and sound of a media file over a time window, which `video.PickClip` draws once for both.

moviepy is the reader because it is what the `av` extra installs and it
reads from wheels alone; it drives ffmpeg, so nothing here holds a decoder
open across records.
"""

from __future__ import annotations

import re
import subprocess

import numpy as np

FPS = 25.0
"""The frame rate a clip is read at where none is asked for (`video.DecodeVideo`)."""


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


def audio_window(path: str, start: float, duration: float, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """The sound of `path` from `start` for `duration` seconds, mono at `sample_rate`, float32 in
    [-1, 1], and whether the track holds each sample: silence and False before the track begins,
    whether `start` is before the file's or the track starts late, and after it ends.

    ffmpeg does the seek, the resampling and the downmix, through the binary
    moviepy drives. moviepy's own audio reader asks for stereo at 44.1 kHz
    whatever the track is, and ffmpeg's mono-to-stereo upmix takes 3 dB off
    a mono track; sampling that stream at another rate then picks the
    nearest 44.1 kHz sample, which aliases and is never sample-exact.

    The samples are placed by the timestamp of the first one decoded, read a
    second before `start`: cut by ffmpeg's own seek instead, an AVI's track
    started 73 or 261 samples late in 2 of 80 windows and a Matroska one up
    to 464 samples off, and a track that starts after the video came back
    shifted to the window's start. Placed, all 80 were sample-exact.
    """
    from moviepy.config import FFMPEG_BINARY

    done = subprocess.run(
        [FFMPEG_BINARY, "-loglevel", "info", "-nostdin", "-copyts", "-start_at_zero",
         "-ss", f"{max(start - 1.0, 0.0):.6f}", "-to", f"{start + duration + 0.01:.6f}", "-i", path,
         "-map", "0:a:0", "-af", "ashowinfo", "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-"],
        capture_output=True)
    if done.returncode:
        raise ValueError(f"ffmpeg could not read the audio of {path}: "
                         f"{done.stderr.decode(errors='replace').strip()[-2000:]}")
    count = round(duration * sample_rate)
    wave, held = np.zeros(count, np.float32), np.zeros(count, bool)
    first = re.search(rb"pts_time:(\S+)", done.stderr)
    if first is None:
        return wave, held
    decoded = np.frombuffer(done.stdout, np.int16).astype(np.float32) / 32768.0
    # The first decoded sample's place in the window; before it where the read began early.
    at = round((float(first.group(1)) - start) * sample_rate)
    lead = min(max(at, 0), count)
    kept = decoded[max(-at, 0):][:count - lead]
    wave[lead:lead + len(kept)] = kept
    held[lead:lead + len(kept)] = True
    return wave, held


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
