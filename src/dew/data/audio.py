"""Host-side Gemma audio preprocessing using the model owner's feature extractor.

Transformers' Gemma 3n and Gemma 4 extractors differ in framing, preemphasis,
mel floor and padding-mask semantics. Calling those NumPy implementations
keeps speech normalization in one place; the native encoders consume their
float32 input_features and boolean input_features_mask (True for valid).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


class AudioProcessor:
    """Construct a Gemma audio feature extractor without downloading any files.

    ``config`` is the feature-extractor record (preprocessor_config.json),
    while ``model_type`` identifies the encoder family. Waveforms are mono
    float samples at the supplied sampling rate. This boundary never resamples.
    """

    def __init__(self, model_type: str, config: Mapping[str, object] | None = None):
        from transformers import Gemma3nAudioFeatureExtractor, Gemma4AudioFeatureExtractor

        if model_type == "gemma3n_audio":
            cls = Gemma3nAudioFeatureExtractor
        elif model_type == "gemma4_audio":
            cls = Gemma4AudioFeatureExtractor
        else:
            raise ValueError(f"audio model_type {model_type!r} is not supported")
        extractor = cls.from_dict(dict(config or {}))
        if not isinstance(extractor, (Gemma3nAudioFeatureExtractor, Gemma4AudioFeatureExtractor)):
            raise TypeError("audio configuration did not produce a feature extractor")
        self.extractor = extractor
        self.sampling_rate = self.extractor.sampling_rate

    def __call__(self, waveforms: np.ndarray | Sequence[np.ndarray] | Sequence[float], *,
                 sampling_rate: int, max_length: int | None = 480000,
                 truncation: bool = True, pad_to_multiple_of: int | None = 128) -> dict[str, np.ndarray]:
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"audio sampling_rate must be {self.sampling_rate}, got {sampling_rate}")
        if len(waveforms) == 0:
            raise ValueError("audio needs at least one nonempty mono waveform")
        first = waveforms[0]
        batch = waveforms if isinstance(first, (np.ndarray, list, tuple)) else [waveforms]
        arrays = [np.asarray(waveform, dtype=np.float32) for waveform in batch]
        if any(array.ndim != 1 or not array.size for array in arrays):
            raise ValueError("audio waveforms must be nonempty mono arrays")
        result = self.extractor(
            arrays, padding="longest", max_length=max_length, truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of, return_tensors="np",
            return_attention_mask=True)
        return {"input_features": np.asarray(result["input_features"], dtype=np.float32),
                "input_features_mask": np.asarray(result["input_features_mask"], dtype=np.bool_)}
