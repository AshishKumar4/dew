"""Video records: a directory tree of clips, a clip window drawn once, and its frames.

`PickClip` draws a record's time window, `DecodeVideo` reads its frames and
`dew.data.audio.DecodeAudio` its sound over the same window, so frames and
audio come from one draw. The AV reader is imported on use.
"""

from __future__ import annotations

import contextlib
import dataclasses
import tempfile
from collections.abc import Iterator
from typing import TYPE_CHECKING

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch
from dew.data.pipeline import Rejected

if TYPE_CHECKING:
    from dew.data.sources.urls import Fetch


def fit_frames(frames: np.ndarray, size: int) -> np.ndarray:
    """`frames` `[T, H, W, 3]` resized to `size` squares by area interpolation."""
    if frames.shape[1] == size and frames.shape[2] == size:
        return frames
    import cv2
    return np.stack([cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
                     for frame in frames])


def fetch(url: str = "url", to: str = "mp4", *, timeout: float = 30.0, max_bytes: int = 256 * 2**20,
          threads: int | None = None) -> Fetch:
    """A record step that fetches the clip at the record's `url` field into
    `to` as bytes, `threads` at once where given, as `images.fetch` does an image."""
    from dew.data.sources.urls import Fetch

    return Fetch(url, to, timeout, max_bytes, threads)


@contextlib.contextmanager
def local_file(value: bytes | str) -> Iterator[str]:
    """A local file a decoder reads for a record's media `value`, its bytes or a local or remote path,
    kept while the context is open."""
    if isinstance(value, bytes):
        with tempfile.NamedTemporaryFile() as file:
            file.write(value)
            file.flush()
            yield file.name
        return
    path = str(value)
    if "://" in path and not path.startswith("file://"):
        from dew.data.storage import Store

        fs, target = Store(path if "::" in path else "dewcache::" + path).resolve()
        with fs.open(target, "rb") as cached:
            yield cached.name
        return
    yield path.removeprefix("file://")


def window(element: Batch, clip: str) -> tuple[float, float]:
    """The `(start, stop)` seconds of the window `PickClip` wrote under `clip`."""
    start, stop = np.asarray(element[clip], np.float64)
    return float(start), float(stop)


@dataclasses.dataclass(frozen=True)
class PickClip(grain.RandomMapTransform):
    """A window of `seconds` drawn from the record's rng within the media in `field` (a video or a sound
    file, its bytes or a path), `margin` seconds clear of either end, written under `to` as float64
    `[start, stop]` seconds for every decoder of the record to read; it stacks as any field does. Media
    too short for the window, or that does not read, is counted and dropped."""

    seconds: float
    _: dataclasses.KW_ONLY
    margin: float = 0.0
    field: str = "mp4"
    to: str = "clip"
    rejects: bool = True
    """Whether media that does not read or is too short is dropped (`filter_map`) rather than an error."""

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch | Rejected:
        from dew.data.sources.av_utils import media_info

        try:
            with local_file(element[self.field]) as path:
                duration, _, _ = media_info(path)
        except (OSError, ValueError):
            if not self.rejects:
                raise
            return Rejected("undecodable")
        latest = duration - self.seconds - self.margin
        # A nanosecond of slack: a clip that fits exactly, 0.48 - 0.32 - 0.08 seconds, is short by roundoff.
        if latest < self.margin - 1e-9:
            if not self.rejects:
                raise ValueError(f"{duration}-second media holds no {self.seconds}-second clip {self.margin} "
                                 "seconds clear of either end")
            return Rejected("shorter than the clip")
        start = float(rng.uniform(self.margin, max(latest, self.margin)))
        return {**element, self.to: np.asarray([start, start + self.seconds])}


@dataclasses.dataclass(frozen=True)
class DecodeVideo(grain.MapTransform):
    """`frames` frames at `fps` from the start of the window under `clip`, read from the video in
    `field`, each fitted into a `size` square, under `to`. The frames must fit the window: one shorter
    than `frames / fps` raises, as no record could pass. A clip that does not decode is counted and
    dropped."""

    frames: int = 16
    _: dataclasses.KW_ONLY
    fps: float = 25.0
    size: int = 256
    field: str = "mp4"
    clip: str = "clip"
    to: str = "video"
    rejects: bool = True
    """Whether a clip that does not decode is dropped (`filter_map`) rather than an error."""

    def map(self, element: Batch) -> Batch | Rejected:
        from dew.data.sources.av_utils import read_frames

        start, stop = window(element, self.clip)
        if start + self.frames / self.fps > stop + 1e-9:
            raise ValueError(f"{self.frames} frames at {self.fps} fps run past the {stop - start}-second "
                             f"clip under {self.clip!r}; pick a longer clip or fewer frames")
        try:
            with local_file(element[self.field]) as path:
                frames = read_frames(path, start, self.frames, self.fps)
        except (OSError, ValueError):
            if not self.rejects:
                raise
            return Rejected("undecodable")
        return {**element, self.to: fit_frames(frames, self.size)}
