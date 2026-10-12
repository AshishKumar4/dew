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
class DecodeClip(grain.RandomMapTransform):
    """`frames` consecutive frames at `fps` from a random start of the clip in
    `field` (its bytes, or a local or remote path), each fitted into a `size`
    square, under `to`; with `audio_padding`, the sound under them and
    `audio_padding` frames either side as one row a frame under "audio", at
    `sample_rate`. A clip that does not decode, is too short, or has no sound
    where it is asked for is rejected."""

    frames: int = 16
    size: int = 256
    fps: float = 25.0
    field: str = "mp4"
    to: str = "video"
    audio_padding: int | None = None
    sample_rate: int = 16_000
    rejects = True

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch | Rejected:
        from dew.data.sources.av_utils import read_av_random_clip, read_random_frames

        clip, seed = element[self.field], int(rng.integers(0, 2**32 - 1))
        try:
            with contextlib.ExitStack() as held:
                path = self._path(clip, held)
                if self.audio_padding is None:
                    frames, _ = read_random_frames(path, num_frames=self.frames, padding=0, seed=seed,
                                                   fps=self.fps)
                    sound = {}
                else:
                    frames, audio = read_av_random_clip(path, num_frames=self.frames,
                                                        audio_padding=self.audio_padding, seed=seed,
                                                        sample_rate=self.sample_rate, fps=self.fps)
                    sound = {"audio": audio}
        except (OSError, ValueError):
            return Rejected("undecodable")
        return {**element, self.to: fit_frames(frames, self.size), **sound}

    @staticmethod
    def _path(clip: bytes | str, held: contextlib.ExitStack) -> str:
        """A local file the decoder reads, which `held` keeps until it closes."""
        if isinstance(clip, bytes):
            file = held.enter_context(tempfile.NamedTemporaryFile())  # noqa: SIM115
            file.write(clip)
            file.flush()
            return file.name
        path = str(clip)
        if "://" in path and not path.startswith("file://"):
            from dew.data.storage import Store

            fs, target = Store(path if "::" in path else "dewcache::" + path).resolve()
            return held.enter_context(fs.open(target, "rb")).name
        return path.removeprefix("file://")


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
