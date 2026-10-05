"""Inference tasks: native generation with a model and its weights, and clients for external engines."""

from dew.sampling.pipelines import DenoisingInputs, Images, TextToImage

from .banks import CheckpointBanks, HeldBanks, LayerBanks, SafetensorsBanks
from .clients import Completion, OllamaCompletion, OpenAICompletion, Usage
from .nccl import NCCLPush
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

__all__ = [
    "BlockGeneration",
    "CheckpointBanks",
    "Completion",
    "DenoisingInputs",
    "Draw",
    "HeldBanks",
    "Images",
    "LayerBanks",
    "MaskedGeneration",
    "NCCLPush",
    "NativeRolloutServer",
    "OllamaCompletion",
    "OpenAICompletion",
    "OpenAIRolloutServer",
    "Processor",
    "Publication",
    "RolloutServer",
    "RunProcessor",
    "SafetensorsBanks",
    "SafetensorsReload",
    "Server",
    "TextGeneration",
    "TextToImage",
    "Usage",
    "VLLMGenerateServer",
    "pipeline",
]
