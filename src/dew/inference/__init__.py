"""Inference tasks: native generation with a model and its weights, and clients for external engines,
each imported only when you use its API.

So a text-to-image task does not import the run loader's checkpointing, the data pipeline or the
rollout servers.
"""

from typing import TYPE_CHECKING

from dew._exports import lookup as _lookup

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew.inference.banks import CheckpointBanks, HeldBanks, LayerBanks, SafetensorsBanks
    from dew.inference.clients import Completion, OllamaCompletion, OpenAICompletion, Usage
    from dew.inference.engine import Engine, Kind, Timed
    from dew.inference.images import DenoisingInputs, Images, TextToImage
    from dew.inference.loading import RunProcessor, pipeline
    from dew.inference.nccl import NCCLPush
    from dew.inference.rollouts import (
        Draw,
        InfrastructureError,
        NativeRolloutServer,
        OpenAIRolloutServer,
        Publication,
        RolloutServer,
        SafetensorsReload,
        VLLMGenerateServer,
    )
    from dew.inference.scoring import Scored, Scoring, ScoringServer
    from dew.inference.serving import Server
    from dew.inference.tasks import BlockGeneration, MaskedGeneration, Processor, TextGeneration

_EXPORTS = {
    "CheckpointBanks": "dew.inference.banks", "HeldBanks": "dew.inference.banks",
    "LayerBanks": "dew.inference.banks", "SafetensorsBanks": "dew.inference.banks",
    "Completion": "dew.inference.clients", "OllamaCompletion": "dew.inference.clients",
    "OpenAICompletion": "dew.inference.clients", "Usage": "dew.inference.clients",
    "Engine": "dew.inference.engine", "Kind": "dew.inference.engine", "Timed": "dew.inference.engine",
    "DenoisingInputs": "dew.inference.images", "Images": "dew.inference.images",
    "TextToImage": "dew.inference.images",
    "RunProcessor": "dew.inference.loading", "pipeline": "dew.inference.loading",
    "NCCLPush": "dew.inference.nccl",
    "Draw": "dew.inference.rollouts", "InfrastructureError": "dew.inference.rollouts",
    "NativeRolloutServer": "dew.inference.rollouts", "OpenAIRolloutServer": "dew.inference.rollouts",
    "Publication": "dew.inference.rollouts", "RolloutServer": "dew.inference.rollouts",
    "SafetensorsReload": "dew.inference.rollouts", "VLLMGenerateServer": "dew.inference.rollouts",
    "Scored": "dew.inference.scoring", "Scoring": "dew.inference.scoring",
    "ScoringServer": "dew.inference.scoring",
    "Server": "dew.inference.serving",
    "BlockGeneration": "dew.inference.tasks", "MaskedGeneration": "dew.inference.tasks",
    "Processor": "dew.inference.tasks", "TextGeneration": "dew.inference.tasks",
}


__getattr__ = _lookup(__name__, _EXPORTS)


def __dir__() -> list[str]:
    return list(__all__)

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
