"""Clip decoding and audio features as record transforms."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import tempfile

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch
from dew.data.pipeline import Rejected
from dew.data.processors import AutoAudioProcessor
from dew.data.video import fit_frames


@dataclasses.dataclass(frozen=True)
class VideoPath(grain.MapTransform):
    caption: str = ""

    def map(self, element: Batch) -> Batch:
        from dew.records import text

        path = text(element.get("video_path", element.get("path")), "video_path")
        return {"video_path": path, "caption": element.get("caption", self.caption)}


@dataclasses.dataclass(frozen=True)
class DecodeClip(grain.RandomMapTransform):
    frames: int = 16
    size: int = 256
    audio_padding: int = 3
    sample_rate: int = 16_000

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        from dew.data.sources.av_utils import read_av_random_clip

        with contextlib.ExitStack() as pins:
            path = str(element["video_path"])
            if "://" in path and not path.startswith("file://"):
                from dew.data.storage import Store

                fs, target = Store(path if "::" in path else "dewcache::" + path).resolve()
                held = pins.enter_context(fs.open(target, "rb"))
                path = held.name
            frames, audio = read_av_random_clip(
                path.removeprefix("file://"),
                num_frames=self.frames,
                audio_padding=self.audio_padding,
                seed=int(rng.integers(0, 2**32 - 1)),
                sample_rate=self.sample_rate,
            )
        return {"video": fit_frames(frames, self.size), "caption": element.get("caption", ""), "audio": audio}


@dataclasses.dataclass(frozen=True)
class DecodeFrames(grain.RandomMapTransform):
    """`frames` consecutive frames from a random start of the clip in `field`,
    its bytes or a local path, each fitted into a `size` square, under `to`,
    without its audio. A clip that does not decode or is shorter than
    `frames` is rejected."""

    frames: int = 16
    size: int = 256
    field: str = "mp4"
    to: str = "video"
    rejects = True

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch | Rejected:
        from dew.data.sources.av_utils import read_random_frames

        clip, seed = element[self.field], int(rng.integers(0, 2**32 - 1))
        try:
            with contextlib.ExitStack() as held:
                if isinstance(clip, bytes):
                    # The decoder reads a file.
                    file = held.enter_context(tempfile.NamedTemporaryFile())
                    file.write(clip)
                    file.flush()
                    clip = file.name
                frames, _ = read_random_frames(str(clip).removeprefix("file://"), num_frames=self.frames,
                                               padding=0, seed=seed)
        except (OSError, ValueError):
            return Rejected("undecodable")
        return {**element, self.to: fit_frames(frames, self.size)}


@dataclasses.dataclass(frozen=True)
class FixedClip(grain.MapTransform):
    decoder: DecodeClip
    seed: int = 0

    def map(self, element: Batch) -> Batch:
        return self.decoder.random_map(element, np.random.default_rng(self.seed))


@dataclasses.dataclass(frozen=True)
class Audio(grain.MapTransform):
    model: str = "facebook/wav2vec2-base-960h"

    @functools.cached_property
    def processor(self) -> AutoAudioProcessor:
        return AutoAudioProcessor(tensor_type="np", modelname=self.model)

    def map(self, element: Batch) -> Batch:
        audio = np.asarray(element["audio"])
        features = self.processor(audio.reshape(-1))
        return {
            **element,
            "audio": {**{key: value[0] for key, value in features.items()}, "full_audio": audio},
        }
