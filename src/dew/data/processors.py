"""Turns raw audio into the arrays a batch carries.

This runs inside grain's workers, so the device only ever sees ready tensors.
`transformers` is imported on construction, not on import. Captions are
tokenized by the condition encoder that reads them (`dew.inputs`).
"""

import inspect
from typing import Protocol, runtime_checkable


@runtime_checkable
class Sampled(Protocol):
    """States the sampling rate a feature extractor expects its audio at.

    Whisper's and wav2vec2's extractors do. One that states none is read at
    16 kHz, the rate every released speech model here is trained on.
    """

    sampling_rate: int


class AutoAudioProcessor:
    """Runs a Hugging Face audio feature extractor.

    Returns its arrays unchanged: `input_values` for wav2vec2,
    `input_features` for Whisper.
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
        """The extractor's arrays for one waveform or a batch of them."""
        features = self.processor(audio, sampling_rate=self.sampling_rate,
                                  return_tensors=self.tensor_type, **self.padding)
        return dict(features)

    def __repr__(self):
        return self.__class__.__name__ + '()'
