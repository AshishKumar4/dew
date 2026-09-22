"""Turns raw text and audio into the arrays a batch carries.

These run inside grain's workers, so the device only ever sees ready tensors.
`transformers` is imported on construction, not on import.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Sampled(Protocol):
    """States the sampling rate a feature extractor expects its audio at.

    Whisper's and wav2vec2's extractors do. One that states none is read at
    16 kHz, the rate every released speech model here is trained on.
    """

    sampling_rate: int


class AutoTextTokenizer:
    """Tokenizes captions, padded and truncated to the text model's context.

    `tensor_type` is what the tokenizer returns its arrays as; "np" is what
    every caller here asks for, since nothing downstream reads torch.
    """

    def __init__(self, tensor_type="np", modelname="openai/clip-vit-large-patch14"):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.tensor_type = tensor_type

    def __call__(self, inputs):
        tokens = self.tokenizer(inputs, padding="max_length", max_length=self.tokenizer.model_max_length,
                                truncation=True, return_tensors=self.tensor_type)
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "caption": inputs,
        }

    def __repr__(self):
        return self.__class__.__name__ + '()'


class AutoAudioProcessor:
    """Turns raw waveforms into the inputs of any HF audio model.

    Whatever keys the model's feature extractor emits (`input_values` for
    wav2vec2/HuBERT, `input_features` for Whisper/AST, ...) pass through
    unchanged, so switching audio models needs no change here.
    """

    def __init__(self, tensor_type="np", modelname="facebook/wav2vec2-base-960h",
                 sampling_rate=None):
        from transformers import AutoFeatureExtractor
        self.processor = AutoFeatureExtractor.from_pretrained(modelname)
        self.tensor_type = tensor_type
        stated = self.processor.sampling_rate if isinstance(self.processor, Sampled) else 16000
        self.sampling_rate = sampling_rate or stated

    def __call__(self, audio):
        features = self.processor(audio, sampling_rate=self.sampling_rate,
                                  padding=True, return_tensors=self.tensor_type)
        return dict(features)

    def __repr__(self):
        return self.__class__.__name__ + '()'
