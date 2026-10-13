"""Video records: a directory tree of clips, a clip window drawn once, and its frames.

`pick_clip` draws a record's time window, `decode` reads its frames and
`dew.data.audio.decode` its sound over the same window, so frames and audio
come from one draw. The AV reader is imported on use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dew.data.sources.urls import Fetch
    from dew.data.steps.video import DecodeVideo, PickClip


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


def pick_clip(seconds: float, *, margin: float = 0.0, field: str = "mp4", to: str = "clip") -> PickClip:
    """A random record step that draws a `seconds`-long window of the media in `field`, bytes or a
    path, `margin` seconds clear of either end, under `to` as `[start, stop]` seconds
    (`PickClip`). Media shorter than the window, or that does not read, is counted and dropped."""
    from dew.data.steps.video import PickClip

    return PickClip(seconds, margin, field, to)


def decode(frames: int = 16, *, fps: float = 25.0, size: int = 256, field: str = "mp4", clip: str = "clip",
           to: str = "video") -> DecodeVideo:
    """A record step that decodes `frames` frames at `fps` from the start of the window `pick_clip`
    wrote under `clip`, each fitted into a `size` square, uint8 under `to` (`DecodeVideo`). A clip
    that does not decode is counted and dropped."""
    from dew.data.steps.video import DecodeVideo

    return DecodeVideo(frames, fps, size, field, clip, to)
