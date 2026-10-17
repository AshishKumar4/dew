"""Sound records: a media file's audio over the window `video.PickClip` drew.

A sound file reads as a video's track does, so a pure-audio dataset is
`video.PickClip` then `DecodeAudio`. The reader is imported on use.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch
from dew.data.pipeline import Rejected
from dew.data.video import local_file, window


@dataclasses.dataclass(frozen=True)
class DecodeAudio(grain.MapTransform):
    """The sound of the media in `field` over the window under `clip`, widened by `context`
    seconds on either side, mono float32 in [-1, 1] at `sample_rate` under `to`, with `{to}_valid`
    marking the samples the track holds. A file with no sound, or less than the window, is rejected
    under `missing="reject"` and filled with silence, marked invalid, under `missing="silence"`; a
    file that does not read is rejected under both."""

    sample_rate: int = 16_000
    _: dataclasses.KW_ONLY
    context: float = 0.0
    field: str = "mp4"
    clip: str = "clip"
    to: str = "audio"
    missing: Literal["reject", "silence"] = "reject"
    rejects: bool = True
    """Whether a file that does not read, or under `missing="reject"` lacks sound, is dropped
    (`filter_map`) rather than an error."""

    def __post_init__(self) -> None:
        if self.missing not in ("reject", "silence"):
            raise ValueError(f"missing audio is rejected or filled with silence, not {self.missing!r}")

    def map(self, element: Batch) -> Batch | Rejected:
        from dew.data.sources.av_utils import audio_window, media_info

        start, stop = window(element, self.clip)
        samples = round((stop - start + 2 * self.context) * self.sample_rate)
        try:
            with local_file(element[self.field]) as path:
                _, _, sound = media_info(path)
                wave, valid = (audio_window(path, start - self.context, stop - start + 2 * self.context,
                                            self.sample_rate) if sound else
                               (np.zeros(samples, np.float32), np.zeros(samples, bool)))
        except (OSError, ValueError):
            if not self.rejects:
                raise
            return Rejected("undecodable")
        if not valid.all() and self.missing == "reject":
            reason = "no audio" if not sound else "audio shorter than the clip"
            if not self.rejects:
                raise ValueError(f"the media in {self.field!r} has {reason}")
            return Rejected(reason)
        return {**element, self.to: wave, f"{self.to}_valid": valid}
