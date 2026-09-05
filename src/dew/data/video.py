"""Audio-video datasets: a directory tree of clips, one random clip per record.

A record is a video file and a caption; the transform reads `frames` frames
from a random offset with the audio around them, resizes the frames and
featurises the audio for the audio model. Records leave as
`{"video": uint8 [frames, size, size, 3], "caption": str, "audio": {...}}`,
where the audio dict holds the audio model's own feature keys next to
`full_audio`, the padded waveform cut into one row per frame;
`load(tokenize=)` is where a run's condition reads the captions. The AV
reader and the audio processor are imported on use.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import grain.python as pygrain
import numpy as np

from dew.registry import datasets

from .dataset import (CAPTION, Dataset, DatasetSpec, Loading, hold_out, local_batch, tokenized,
                      train_stream, validation_pass)
from .processors import AutoAudioProcessor


def video_paths(root: str, extensions: tuple[str, ...]) -> list[str]:
    """Every video file under `root`, in one deterministic order."""
    suffixes = tuple(ext.lower() if ext.startswith('.') else f'.{ext}'.lower()
                     for ext in extensions)
    paths = []
    for directory, _, files in os.walk(root):
        paths += [os.path.join(directory, name) for name in files
                  if os.path.splitext(name)[1].lower() in suffixes]
    return sorted(paths)


class AudioVideoTransform(pygrain.RandomMapTransform):
    """One clip per record: frames, their audio, and the record's caption."""

    def __init__(self, spec: "VideoDataset"):
        self.spec = spec
        self.audio = AutoAudioProcessor(tensor_type="np", modelname=spec.audio_model)

    def random_map(self, element: Any, rng: np.random.Generator) -> dict[str, Any]:
        # moviepy comes in with the reader, on the first record rather than on import.
        from .sources.av_utils import read_av_random_clip
        frames, audio = read_av_random_clip(
            element["video_path"], num_frames=self.spec.frames,
            audio_padding=self.spec.audio_padding, seed=int(rng.integers(0, 2**32 - 1)))
        size = self.spec.frame_size
        if frames.shape[1] != size or frames.shape[2] != size:
            import cv2
            frames = np.stack([cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
                               for frame in frames])
        # The extractor takes one waveform and hands back a batch of one;
        # given the rows it would read each frame's 640 samples as a clip
        # of its own. Key names differ per audio model, so its output
        # passes through untouched.
        features = self.audio(audio.reshape(-1))
        return {
            "video": frames,
            CAPTION: element["caption"],
            "audio": {**{key: value[0] for key, value in features.items()},
                      "full_audio": audio},
        }


@dataclasses.dataclass(frozen=True)
class VideoDataset(DatasetSpec):
    """Clips of `frames` frames at `frame_size`, with their audio, through grain.

    `val_batches` batches of records are held out of the head of the source,
    in canonical order, as the validation split; None or 0 holds nothing out.
    `count` uses that many records from the head of the source.
    """

    frame_size: int = 256
    frames: int = 16
    audio_padding: int = 3
    """Extra audio frames kept on either side of the sampled clip."""
    audio_model: str = "facebook/wav2vec2-base-960h"
    """HF audio model whose feature extractor prepares the audio inputs."""
    val_batches: int | None = 4
    count: int | None = None
    seed: int = 0
    loading: Loading = Loading()

    def source(self) -> list[dict[str, str]]:
        """One `{"video_path", "caption"}` record per clip, in a fixed order."""
        raise NotImplementedError

    def load(self, *, batch: int, tokenize=None) -> Dataset:
        source = self.source()
        name = type(self).__name__
        records = len(source) if self.count is None else self.count
        if records > len(source):
            raise ValueError(f"count {self.count} is more than the {len(source)} records of {name}")
        train, val = hold_out(source, records, (self.val_batches or 0) * batch, name)
        return Dataset(
            train=tokenized(train_stream(train, [AudioVideoTransform(self)], batch=local_batch(batch), seed=self.seed, loading=self.loading), tokenize),
            val=None if val is None else tokenized(
                validation_pass(val, [AudioVideoTransform(self)], batch=local_batch(batch), seed=self.seed, loading=self.loading), tokenize),
            records=len(train),
            batch=batch,
        )


@datasets("voxceleb2")
@dataclasses.dataclass(frozen=True)
class VoxCeleb2(VideoDataset):
    """A VoxCeleb2 tree, `<path>/<split>/<identity>/<clip>/<utterance>.mp4`,
    scanned recursively so extra nesting is tolerated. The caption is
    `prompt_template` with `{identity}` replaced by the speaker directory."""

    path: str | None = None
    split: str = "train"
    extensions: tuple[str, ...] = (".mp4", ".avi")
    prompt_template: str = "a video of a person speaking"

    def source(self):
        if not self.path:
            raise ValueError("VoxCeleb2 needs path= set to the dataset root, the "
                             "directory holding the split directories")
        split_root = os.path.join(self.path, self.split)
        if not os.path.isdir(split_root):
            raise ValueError(
                f"VoxCeleb2 split {self.split!r} not found: {split_root!r} is not a "
                "directory. Expected <root>/<split>/<identity>/<clip>/<utterance>.mp4.")
        records = []
        for video_path in video_paths(split_root, self.extensions):
            identity = os.path.relpath(video_path, split_root).split(os.sep)[0]
            try:
                caption = self.prompt_template.format(identity=identity)
            except (KeyError, IndexError, ValueError):
                raise ValueError(
                    f"prompt_template {self.prompt_template!r} may use {{identity}} "
                    "and no other placeholder") from None
            records.append({"video_path": video_path, "caption": caption})
        return records


@datasets("local_videos")
@dataclasses.dataclass(frozen=True)
class LocalVideos(VideoDataset):
    """Every video file under `path`, captioned with `caption`."""

    path: str | None = None
    extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".webm")
    caption: str = ""

    def source(self):
        if not self.path:
            raise ValueError("LocalVideos needs path= set to the directory of video files")
        return [{"video_path": video_path, "caption": self.caption}
                for video_path in video_paths(self.path, self.extensions)]
