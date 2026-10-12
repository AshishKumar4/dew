"""Audio-video datasets: a directory tree of clips, one random clip per record.

A record is a video file and a caption. The transform reads `frames` frames
from a random offset with the audio around them, resizes the frames and
featurises the audio for the audio model. Records leave as
`{"video": uint8 [frames, size, size, 3], "caption": str, "audio": {...}}`,
where the audio dict holds the audio model's own feature keys next to
`full_audio`, the padded waveform cut into one row per frame.
`load(tokenize=)` is where a run's condition reads the captions. The AV
reader and the audio processor are imported on use.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dew.data.sources.urls import Fetch
    from dew.data.steps.video import Audio, DecodeClip


def video_paths(root: str, extensions: tuple[str, ...]) -> list[str]:
    """Every video file under `root`, in one deterministic order."""
    suffixes = tuple(ext.lower() if ext.startswith('.') else f'.{ext}'.lower()
                     for ext in extensions)
    paths = []
    for directory, _, files in os.walk(root):
        paths += [os.path.join(directory, name) for name in files
                  if os.path.splitext(name)[1].lower() in suffixes]
    return sorted(paths)


def fit_frames(frames: np.ndarray, size: int) -> np.ndarray:
    """`frames` `[T, H, W, 3]` resized to `size` squares by area interpolation."""
    if frames.shape[1] == size and frames.shape[2] == size:
        return frames
    import cv2
    return np.stack([cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
                     for frame in frames])


def fetch(url: str = "url", to: str = "mp4", *, timeout: float = 30.0, max_bytes: int = 256 * 2**20) -> Fetch:
    """A record step that fetches the clip at the record's `url` field into
    `to` as bytes, as `images.fetch` does an image."""
    from dew.data.sources.urls import Fetch

    return Fetch(url, to, timeout, max_bytes)


def decode_clip(frames: int = 16, *, size: int = 256, fps: float = 25.0, field: str = "mp4",
                to: str = "video", audio_padding: int | None = None, sample_rate: int = 16_000) -> DecodeClip:
    """A random record step that decodes `frames` frames at `fps` from a
    random start of the clip in `field`, bytes or a path, each fitted into a
    `size` square, under `to`; with `audio_padding`, the sound under them
    too (`DecodeClip`). A clip that does not decode is counted and dropped."""
    from dew.data.steps.video import DecodeClip

    return DecodeClip(frames, size, fps, field, to, audio_padding, sample_rate)


def audio(features: str = "facebook/wav2vec2-base-960h") -> Audio:
    """A record step that featurizes a decoded clip's "audio" for the audio model `features`."""
    from dew.data.steps.video import Audio

    return Audio(features)
