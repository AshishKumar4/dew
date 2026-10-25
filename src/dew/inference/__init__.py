"""Inference tasks: native generation with a model and its weights, and clients for external engines."""

from .banks import CheckpointBanks, HeldBanks, LayerBanks, SafetensorsBanks
from .clients import Completion, OllamaCompletion, OpenAICompletion, Usage
from .engine import Engine, Kind, Timed
from .images import DenoisingInputs, Images, TextToImage
from .nccl import NCCLPush
from .pipeline import RunProcessor, pipeline
from .rollouts import (
           Draw,
           InfrastructureError,
           NativeRolloutServer,
           OpenAIRolloutServer,
           Publication,
           RolloutServer,
           SafetensorsReload,
           VLLMGenerateServer,
)
from .scoring import Scored, Scoring, ScoringServer
from .serving import Server
from .tasks import BlockGeneration, MaskedGeneration, Processor, TextGeneration

__all__ = [
    "BlockGeneration",
    "CheckpointBanks",
    "Completion",
    "DenoisingInputs",
    "Draw",
    "Engine",
    "HeldBanks",
    "Images",
    "InfrastructureError",
    "Kind",
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
    "Scored",
    "Scoring",
    "ScoringServer",
    "Server",
    "TextGeneration",
    "TextToImage",
    "Timed",
    "Usage",
    "VLLMGenerateServer",
    "pipeline",
]
