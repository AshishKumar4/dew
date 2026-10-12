"""A clip's time window, drawn once, and its frames as record transforms."""

from __future__ import annotations

import contextlib
import dataclasses
import tempfile
from collections.abc import Iterator

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch
from dew.data.pipeline import Rejected
from dew.data.video import fit_frames


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
    """The `(start, stop)` seconds of the window `pick_clip` wrote under `clip`."""
    start, stop = np.asarray(element[clip], np.float64)
    return float(start), float(stop)


@dataclasses.dataclass(frozen=True)
class PickClip(grain.RandomMapTransform):
    """A window of `seconds` drawn from the record's rng within the media in `field` (a video or a sound
    file, its bytes or a path), `margin` seconds clear of either end, written under `to` as float64
    `[start, stop]` seconds for every decoder of the record to read; it stacks as any field does. Media
    too short for the window, or that does not read, is rejected."""

    seconds: float
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
    than `frames / fps` raises, as no record could pass. A clip that does not decode is rejected."""

    frames: int = 16
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
