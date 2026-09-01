"""CPU-side tokenizers and feature extractors that turn raw text and audio into model inputs."""

from .encoders import CLIPTextEncoder


class AutoTextTokenizer:
    def __init__(self, tensor_type="pt", modelname="openai/clip-vit-large-patch14"):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        self.tensor_type = tensor_type

    def __call__(self, inputs):
        # print(caption)
        tokens = self.tokenizer(inputs, padding="max_length", max_length=self.tokenizer.model_max_length,
                                truncation=True, return_tensors=self.tensor_type)
        # print(tokens.keys())
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "caption": inputs,
        }

    def __repr__(self):
        return self.__class__.__name__ + '()'


class AutoAudioProcessor:
    """Turn raw audio waveforms into model inputs, for any HF audio model.

    The audio counterpart of AutoTextTokenizer: it runs on CPU in the grain
    workers so the device only sees ready tensors. Whatever keys the model's
    feature extractor emits (`input_values` for wav2vec2/HuBERT,
    `input_features` for Whisper/AST, ...) are passed through unchanged, so
    switching audio models needs no change here.
    """
    def __init__(self, tensor_type="np", modelname="facebook/wav2vec2-base-960h",
                 sampling_rate=None):
        from transformers import AutoFeatureExtractor
        self.processor = AutoFeatureExtractor.from_pretrained(modelname)
        self.tensor_type = tensor_type
        self.modelname = modelname
        # The processor knows the rate its model was trained at
        self.sampling_rate = sampling_rate or getattr(self.processor, "sampling_rate", 16000)

    def __call__(self, audio):
        features = self.processor(audio, sampling_rate=self.sampling_rate,
                                  padding=True, return_tensors=self.tensor_type)
        return dict(features)

    def __repr__(self):
        return self.__class__.__name__ + '()'

def defaultTextEncodeModel(modelname = "openai/clip-vit-large-patch14", backend="jax"):
    """Default text encoder model."""
    return CLIPTextEncoder.from_modelname(modelname=modelname, backend=backend)
