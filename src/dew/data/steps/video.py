"""A clip's time window, drawn once, and its frames as record transforms."""

from __future__ import annotations

import contextlib
import dataclasses
import tempfile

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch
from dew.data.pipeline import Rejected
from dew.data.video import fit_frames


def local_file(value: bytes | str, held: contextlib.ExitStack) -> str:
    """A local file a decoder reads for a record's media `value`, its bytes or a local or remote path,
    which `held` keeps until it closes."""
    if isinstance(value, bytes):
        file = held.enter_context(tempfile.NamedTemporaryFile())  # noqa: SIM115
        file.write(value)
        file.flush()
        return file.name
    path = str(value)
    if "://" in path and not path.startswith("file://"):
        from dew.data.storage import Store

        fs, target = Store(path if "::" in path else "dewcache::" + path).resolve()
        return held.enter_context(fs.open(target, "rb")).name
    return path.removeprefix("file://")


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
    rejects = True

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch | Rejected:
        from dew.data.sources.av_utils import media_info

        try:
            with contextlib.ExitStack() as held:
                duration, _, _ = media_info(local_file(element[self.field], held))
        except (OSError, ValueError):
            return Rejected("undecodable")
        latest = duration - self.seconds - self.margin
        # A nanosecond of slack: a clip that fits exactly, 0.48 - 0.32 - 0.08 seconds, is short by roundoff.
        if latest < self.margin - 1e-9:
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
    rejects = True

    def map(self, element: Batch) -> Batch | Rejected:
        from dew.data.sources.av_utils import read_frames

        start, stop = window(element, self.clip)
        if start + self.frames / self.fps > stop + 1e-9:
            raise ValueError(f"{self.frames} frames at {self.fps} fps run past the {stop - start}-second "
                             f"clip under {self.clip!r}; pick a longer clip or fewer frames")
        try:
            with contextlib.ExitStack() as held:
                frames = read_frames(local_file(element[self.field], held), start, self.frames, self.fps)
        except (OSError, ValueError):
            return Rejected("undecodable")
        return {**element, self.to: fit_frames(frames, self.size)}
