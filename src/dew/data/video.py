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

import numpy as np


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


