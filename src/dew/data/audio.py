"""Sound records: a media file's audio over the window `video.pick_clip` drew.

A sound file reads as a video's track does, so a pure-audio dataset is
`video.pick_clip` then `decode`. The reader is imported on use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from dew.data.steps.audio import DecodeAudio


def decode(sample_rate: int = 16_000, *, context: float = 0.0, field: str = "mp4", clip: str = "clip",
           to: str = "audio", missing: Literal["reject", "silence"] = "reject") -> DecodeAudio:
    """A record step that decodes the sound of the media in `field` over the window under `clip`,
    `context` seconds wider on either side, mono float32 at `sample_rate` under `to`, with
    `{to}_valid` marking the samples the track holds (`DecodeAudio`). Media with no sound or too
    little is dropped, or under `missing="silence"` filled with silence marked invalid."""
    from dew.data.steps.audio import DecodeAudio

    return DecodeAudio(sample_rate, context, field, clip, to, missing)
