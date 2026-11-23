"""Inference tasks: native generation with a model and its weights, and clients for external engines,
each imported only when you use its API. An engine's own drivers stay in their modules: the weight
pushes (`dew.inference.nccl.NCCLPush`, `dew.inference.rollouts.SafetensorsReload`) and the layer banks'
adapters (`dew.inference.banks`).

So a text-to-image task does not import the run loader's checkpointing, the data pipeline or the
rollout servers.
"""

from typing import TYPE_CHECKING

from dew._exports import lookup as _lookup

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew.inference.clients import Completion, OllamaCompletion, OpenAICompletion, Usage
    from dew.inference.engine import Engine, Kind, Timed
    from dew.inference.images import DenoisingInputs, ImageRequest, Images, TextToImage
    from dew.inference.loading import pipeline
    from dew.inference.models import ModelSettings, SavedModel
    from dew.inference.rollouts import (
        Draw,
        InfrastructureError,
        NativeRolloutServer,
        OpenAIRolloutServer,
        Publication,
        RolloutServer,
        VLLMGenerateServer,
    )
    from dew.inference.scoring import Scoring, ScoringRequest, ScoringServer
    from dew.inference.serving import DenseKV, PagedKV, Server
    from dew.inference.tasks import BlockGeneration, MaskedGeneration, Processor, RunProcessor, TextGeneration

_EXPORTS = {
    "Completion": "dew.inference.clients", "OllamaCompletion": "dew.inference.clients",
    "OpenAICompletion": "dew.inference.clients", "Usage": "dew.inference.clients",
    "Engine": "dew.inference.engine", "Kind": "dew.inference.engine", "Timed": "dew.inference.engine",
    "DenoisingInputs": "dew.inference.images", "ImageRequest": "dew.inference.images",
    "Images": "dew.inference.images",
    "TextToImage": "dew.inference.images",
    "ModelSettings": "dew.inference.models", "SavedModel": "dew.inference.models",
    "RunProcessor": "dew.inference.tasks", "pipeline": "dew.inference.loading",
    "Draw": "dew.inference.rollouts", "InfrastructureError": "dew.inference.rollouts",
    "NativeRolloutServer": "dew.inference.rollouts", "OpenAIRolloutServer": "dew.inference.rollouts",
    "Publication": "dew.inference.rollouts", "RolloutServer": "dew.inference.rollouts",
    "VLLMGenerateServer": "dew.inference.rollouts",
    "Scoring": "dew.inference.scoring", "ScoringRequest": "dew.inference.scoring",
    "ScoringServer": "dew.inference.scoring",
    "DenseKV": "dew.inference.serving", "PagedKV": "dew.inference.serving", "Server": "dew.inference.serving",
    "BlockGeneration": "dew.inference.tasks", "MaskedGeneration": "dew.inference.tasks",
    "Processor": "dew.inference.tasks", "TextGeneration": "dew.inference.tasks",
}


__getattr__, __dir__ = _lookup(__name__, _EXPORTS)

__all__ = [
    "BlockGeneration",
    "Completion",
    "DenoisingInputs",
    "DenseKV",
    "Draw",
    "Engine",
    "ImageRequest",
    "Images",
    "InfrastructureError",
    "Kind",
    "MaskedGeneration",
    "ModelSettings",
    "NativeRolloutServer",
    "OllamaCompletion",
    "OpenAICompletion",
    "OpenAIRolloutServer",
    "PagedKV",
    "Processor",
    "Publication",
    "RolloutServer",
    "RunProcessor",
    "SavedModel",
    "Scoring",
    "ScoringRequest",
    "ScoringServer",
    "Server",
    "TextGeneration",
    "TextToImage",
    "Timed",
    "Usage",
    "VLLMGenerateServer",
    "pipeline",
]
