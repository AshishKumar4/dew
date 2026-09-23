"""Inference tasks: native generation bound to weights, and external engines."""

from dew.sampling.pipelines import DenoisingInputs, Images, TextToImage

from .banks import CheckpointBanks, HeldBanks, LayerBanks, host_banked
from .clients import Completion, OllamaCompletion, OpenAICompletion, Usage
from .pipeline import RunProcessor, pipeline
from .rollouts import (
           Draw,
           NativeRolloutServer,
           OpenAIRolloutServer,
           Publication,
           RolloutServer,
           SafetensorsReload,
           VLLMGenerateServer,
)
from .serving import Server
from .tasks import BlockGeneration, MaskedGeneration, Processor, TextGeneration

__all__ = ["BlockGeneration", "CheckpointBanks", "Completion", "DenoisingInputs", "Draw", "HeldBanks",
           "Images", "LayerBanks", "MaskedGeneration", "NativeRolloutServer", "OllamaCompletion",
           "OpenAICompletion", "OpenAIRolloutServer", "Processor", "Publication", "RolloutServer", "RunProcessor",
           "SafetensorsReload", "Server", "TextGeneration", "TextToImage", "Usage", "VLLMGenerateServer", "host_banked",
           "pipeline"]
