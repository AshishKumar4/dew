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
    from dew.data.steps.video import DecodeFrames


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


def fetch(url: str = "url", to: str = "mp4", *, timeout: float = 30.0, retries: int = 2,
          max_bytes: int = 256 * 2**20) -> Fetch:
    """A record step that fetches the clip at the record's `url` field into
    `to` as bytes, as `images.fetch` does an image."""
    from dew.data.sources.urls import Fetch

    return Fetch(url, to, timeout, retries, max_bytes)


def decode(field: str = "mp4", to: str = "video", *, frames: int = 16, size: int = 256) -> DecodeFrames:
    """A random record step that decodes `frames` frames of the clip in
    `field`, from a random start, each fitted into a `size` square, under
    `to`, without audio; a clip that does not decode or is too short is
    counted and dropped."""
    from dew.data.steps.video import DecodeFrames

    return DecodeFrames(frames, size, field, to)
