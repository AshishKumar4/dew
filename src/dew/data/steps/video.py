"""Clip decoding and audio features as record transforms."""

from __future__ import annotations

import dataclasses
import functools

import grain.python as grain
import numpy as np

from dew.data.dataset import Batch
from dew.data.processors import AutoAudioProcessor
from dew.data.video import fit_frames


@dataclasses.dataclass(frozen=True)
class DecodeClip(grain.RandomMapTransform):
    frames: int = 16
    size: int = 256
    audio_padding: int = 3
    sample_rate: int = 16_000

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        from dew.data.sources.av_utils import read_av_random_clip

        frames, audio = read_av_random_clip(element["video_path"], num_frames=self.frames,
                                            audio_padding=self.audio_padding,
                                            seed=int(rng.integers(0, 2**32 - 1)), sample_rate=self.sample_rate)
        return {"video": fit_frames(frames, self.size), "caption": element.get("caption", ""), "audio": audio}


@dataclasses.dataclass(frozen=True)
class FixedClip(grain.MapTransform):
    decoder: DecodeClip
    seed: int = 0

    def map(self, element: Batch) -> Batch:
        return self.decoder.random_map(element, np.random.default_rng(self.seed))


@dataclasses.dataclass(frozen=True)
class Audio(grain.MapTransform):
    model: str = "facebook/wav2vec2-base-960h"

    @functools.cache
    def _processor(self) -> AutoAudioProcessor:
        return AutoAudioProcessor(tensor_type="np", modelname=self.model)

    def map(self, element: Batch) -> Batch:
        audio = np.asarray(element["audio"])
        features = self._processor()(audio.reshape(-1))
        return {**element, "audio": {**{key: value[0] for key, value in features.items()}, "full_audio": audio}}
