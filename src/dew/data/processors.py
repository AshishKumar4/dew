"""Turns raw audio into the arrays a batch carries.

This runs inside grain's workers, so the device only ever sees ready tensors.
`transformers` is imported on construction, not on import. Captions are
tokenized by the condition encoder that reads them (`dew.inputs`).
"""

import dataclasses
import inspect
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Sampled(Protocol):
    """States the sampling rate a feature extractor expects its audio at.

    Whisper's and wav2vec2's extractors do. One that states none is read at
    16 kHz, the rate every released speech model here is trained on.
    """

    sampling_rate: int


class AutoAudioProcessor:
    """Runs a Hugging Face audio feature extractor and returns its arrays
    unchanged, such as `input_values` for wav2vec2 or `input_features` for
    Whisper.

    `modelname` names the model whose extractor is loaded. The audio is read
    at `sampling_rate`, which defaults to the rate the extractor states, or
    16 kHz when it states none. An extractor that pads to a fixed window by
    default, as Whisper's pads to 30 seconds, keeps that padding. One that
    does not pad by default, such as wav2vec2's, pads a batch to its longest
    waveform.
    """

    def __init__(self, tensor_type="np", modelname="facebook/wav2vec2-base-960h",
                 sampling_rate=None):
        from transformers import AutoFeatureExtractor
        extractor = AutoFeatureExtractor.from_pretrained(modelname)
        self.processor = extractor
        self.tensor_type = tensor_type
        # An extractor that pads to a fixed window by default (Whisper's 30
        # seconds, which its encoder is built for) keeps that; one that pads
        # nothing by default (wav2vec2's) pads a batch to its longest
        # waveform, so rows of several lengths still stack.
        padding = inspect.signature(extractor.__call__).parameters.get("padding")
        stated = extractor.sampling_rate if isinstance(extractor, Sampled) else 16000
        self.sampling_rate = sampling_rate or stated
        self.padding = {} if padding is None else {"padding": padding.default or True}

    def __call__(self, audio):
        """Return the extractor's arrays for one waveform or a batch of them, as a dict."""
        features = self.processor(audio, sampling_rate=self.sampling_rate,
                                  return_tensors=self.tensor_type, **self.padding)
        return dict(features)

    def __repr__(self):
        return self.__class__.__name__ + '()'


@dataclasses.dataclass(frozen=True)
class LogMel:
    """A waveform's log mel frames `[frames, n_mels]` float32, from transformers' shared STFT and mel
    filters (`audio_utils`, which its feature extractors compute with): centred frames of a
    reflect-padded waveform under a periodic Hann window, HTK mel filters without a norm, the
    magnitude to `power`, floored at `floor` before the natural log. That is torchaudio's
    MelSpectrogram at its defaults and `power`, as MiMo-V2.6's audio tokenizer reads it at 24 kHz and
    128 mels (MiMo-Audio-Tokenizer's `mel_spectrogram`, `MiMoAudioTokenizer.mel`)."""

    sampling_rate: int
    n_fft: int
    hop_length: int
    win_length: int
    n_mels: int
    f_min: float = 0.0
    f_max: float | None = None
    power: float = 1.0
    floor: float = 1e-7

    def __call__(self, waveform) -> np.ndarray:
        from transformers.audio_utils import mel_filter_bank, spectrogram, window_function

        filters = mel_filter_bank(
            num_frequency_bins=self.n_fft // 2 + 1, num_mel_filters=self.n_mels, min_frequency=self.f_min,
            max_frequency=self.sampling_rate / 2 if self.f_max is None else self.f_max,
            sampling_rate=self.sampling_rate, norm=None, mel_scale="htk")
        frames = spectrogram(
            np.asarray(waveform, np.float32), window_function(self.win_length, "hann", periodic=True,
                                                              frame_length=self.n_fft),
            frame_length=self.n_fft, hop_length=self.hop_length, fft_length=self.n_fft, power=self.power,
            center=True, pad_mode="reflect", mel_filters=filters, mel_floor=self.floor, log_mel="log")
        return np.ascontiguousarray(frames.T, np.float32)
