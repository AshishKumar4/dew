"""A clip's sound over the window `video.pick_clip` drew, as a record transform."""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Literal

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch
from dew.data.pipeline import Rejected
from dew.data.steps.video import local_file, window


@dataclasses.dataclass(frozen=True)
class DecodeAudio(grain.MapTransform):
    """The sound of the media in `field` over the window under `clip`, widened by `context`
    seconds on either side, mono float32 in [-1, 1] at `sample_rate` under `to`, with `{to}_valid`
    marking the samples the track holds. A file with no sound, or less than the window, is rejected
    under `missing="reject"` and filled with silence, marked invalid, under `missing="silence"`; a
    file that does not read is rejected under both."""

    sample_rate: int = 16_000
    context: float = 0.0
    field: str = "mp4"
    clip: str = "clip"
    to: str = "audio"
    missing: Literal["reject", "silence"] = "reject"
    rejects = True

    def __post_init__(self) -> None:
        if self.missing not in ("reject", "silence"):
            raise ValueError(f"missing audio is rejected or filled with silence, not {self.missing!r}")

    def map(self, element: Batch) -> Batch | Rejected:
        from dew.data.sources.av_utils import audio_window, media_info

        start, stop = window(element, self.clip)
        first = start - self.context
        samples = round((stop - start + 2 * self.context) * self.sample_rate)
        try:
            with contextlib.ExitStack() as held:
                path = local_file(element[self.field], held)
                _, _, sound = media_info(path)
                wave = (audio_window(path, max(first, 0.0), (stop + self.context) - max(first, 0.0),
                                     self.sample_rate) if sound else np.zeros(0, np.float32))
        except (OSError, ValueError):
            return Rejected("undecodable")
        # A window that opens before the file's start reads from it, the samples before it absent.
        lead = min(samples, round(max(-first, 0.0) * self.sample_rate))
        wave = wave[:samples - lead]
        valid = np.zeros(samples, bool)
        valid[lead:lead + len(wave)] = True
        if not valid.all() and self.missing == "reject":
            return Rejected("no audio" if not sound else "audio shorter than the clip")
        full = np.zeros(samples, np.float32)
        full[lead:lead + len(wave)] = wave
        return {**element, self.to: full, f"{self.to}_valid": valid}
