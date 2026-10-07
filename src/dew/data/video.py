"""Audio-video datasets: a directory tree of clips, one random clip per record.

A record is a video file and a caption. The transform reads `frames` frames
from a random offset with the audio around them, resizes the frames and
featurises the audio for the audio model. Records leave as
`{"video": uint8 [frames, size, size, 3], "caption": str, "audio": {...}}`,
where the audio dict holds the audio model's own feature keys next to
`full_audio`, the padded waveform cut into one row per frame.
`load(tokenize=)` is where a run's condition reads the captions. The AV
reader and the audio processor are imported on use.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping

import grain.python as pygrain
import numpy as np

from .dataset import (
    CAPTION,
    Batch,
    Dataset,
    DatasetSpec,
    Tokenize,
    checked_count,
    hold_out,
    tokenized,
    train_stream,
    validation_pass,
)
from .images import import_opencv
from .processors import AutoAudioProcessor
from .sources.av_utils import FPS


def video_paths(root: str, extensions: tuple[str, ...]) -> list[str]:
    """Every video file under `root`, in one deterministic order."""
    suffixes = tuple(ext.lower() if ext.startswith('.') else f'.{ext}'.lower()
                     for ext in extensions)
    paths = []
    for directory, _, files in os.walk(root):
        paths += [os.path.join(directory, name) for name in files
                  if os.path.splitext(name)[1].lower() in suffixes]
    return sorted(paths)


def fit_frames(frames: np.ndarray, size: int) -> np.ndarray:
    """`frames` `[T, H, W, 3]` resized to `size` squares by area interpolation."""
    if frames.shape[1] == size and frames.shape[2] == size:
        return frames
    import cv2
    return np.stack([cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
                     for frame in frames])


class AudioVideoTransform(pygrain.RandomMapTransform):
    """Reads one clip per record: its frames, their audio, and its caption.

    It is built where its loader opens and unpickled where a spawned worker
    starts, both before any reader thread runs, and both import OpenCV
    (`dew.data.images.import_opencv`).
    """

    def __init__(self, spec: VideoDataset):
        import_opencv()
        self.spec = spec
        self.audio = AutoAudioProcessor(tensor_type="np", modelname=spec.audio_model)

    def __setstate__(self, state: Mapping[str, object]) -> None:
        import_opencv()
        self.__dict__.update(state)

    def random_map(self, element: Batch, rng: np.random.Generator) -> Batch:
        # moviepy is imported on the first record, so importing this module needs no `av` extra.
        from .sources.av_utils import read_av_random_clip
        # The extractor is told the waveform is at its own rate, so it is read at that rate.
        frames, audio = read_av_random_clip(
            element["video_path"], num_frames=self.spec.frames,
            audio_padding=self.spec.audio_padding, seed=int(rng.integers(0, 2**32 - 1)),
            sample_rate=self.audio.sampling_rate)
        frames = fit_frames(frames, self.spec.frame_size)
        # The extractor takes one waveform and hands back a batch of one;
        # given the rows it would read each frame's samples as a clip of
        # its own. Key names differ per audio model, so its output passes
        # through untouched.
        features = self.audio(audio.reshape(-1))
        return {
            "video": frames,
            CAPTION: element["caption"],
            "audio": {**{key: value[0] for key, value in features.items()},
                      "full_audio": audio},
        }


@dataclasses.dataclass(frozen=True)
class VideoDataset(DatasetSpec):
    """Reads clips of `frames` frames at `frame_size`, with their audio.

    Each record comes out as
    `{"video": uint8 [frames, frame_size, frame_size, 3], "caption": str, "audio": {...}}`.
    A subclass defines `source`. `val_batches` batches of records are held
    out of the head of the source, in canonical order, as the validation
    split; None or 0 holds nothing out. `count` uses only that many records
    from the head of the source.
    """

    frame_size: int = 256
    frames: int = 16
    audio_padding: int = 3
    """Extra audio frames kept on either side of the sampled clip."""
    audio_model: str = "facebook/wav2vec2-base-960h"
    """The HF audio model whose feature extractor prepares the audio inputs."""
    val_batches: int | None = 4
    count: int | None = None

    @property
    def audio_seconds(self) -> float:
        """The length in seconds of the waveform under a clip: its frames plus the
        padding on each side, at the 25 fps that clips are sampled at."""
        return (self.frames + 2 * self.audio_padding) / FPS

    def source(self) -> list[dict[str, str]]:
        """Return one `{"video_path", "caption"}` record per clip, in a fixed order."""
        raise NotImplementedError

    def load(self, *, batch: int, tokenize: Tokenize | None = None) -> Dataset:
        source = self.source()
        name = type(self).__name__
        records = (len(source) if self.count is None
                   else checked_count(self.count, len(source), name))
        held_out = (self.val_batches or 0) * batch
        train, validation = hold_out(source, records, held_out, name)
        return Dataset(
            train=tokenized(
                train_stream(
                    train, [AudioVideoTransform(self)], batch=batch, seed=self.seed, loading=self.loading
                ),
                tokenize,
            ),
            val=None
            if validation is None
            else tokenized(
                validation_pass(
                    validation, [AudioVideoTransform(self)], batch=batch, seed=self.seed, loading=self.loading
                ),
                tokenize,
            ),
            records=len(train),
            batch=batch,
            held_out=held_out,
        )


@dataclasses.dataclass(frozen=True)
class LocalVideos(VideoDataset):
    """Reads every video file under `path`, captioned with `caption`.

    It reads the files whose suffix is in `extensions`, in sorted path order.
    """

    path: str | None = None
    extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".webm")
    caption: str = ""

    def source(self):
        if not self.path:
            raise ValueError("LocalVideos needs path= set to the directory of video files")
        return [{"video_path": video_path, "caption": self.caption}
                for video_path in video_paths(self.path, self.extensions)]
