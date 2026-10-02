"""Checkpoint readers and translators, imported when their API is used.

Reading Hub metadata or a safetensors file needs neither the decoder
translators nor the diffusion pipeline loader.
"""

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hf_decoders import translate_config, translate_weights
    from .pretrained import (
        Pretrained,
        PretrainedBlockDecoder,
        PretrainedDecoder,
        PretrainedFallback,
        PretrainedMaskedDecoder,
        PretrainedPipeline,
        Processor,
        split_revision,
    )
    from .safetensors_io import load_params, save_hf_layout, save_params

_EXPORTS = {
    **dict.fromkeys(("Pretrained", "PretrainedBlockDecoder", "PretrainedDecoder", "PretrainedFallback",
                     "PretrainedMaskedDecoder", "PretrainedPipeline", "Processor", "split_revision"),
                    "pretrained"),
    "translate_config": "hf_decoders",
    "translate_weights": "hf_decoders",
    "load_params": "safetensors_io", "save_hf_layout": "safetensors_io", "save_params": "safetensors_io",
}


def __getattr__(name: str) -> type | Callable:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{module}"), name)


def __dir__() -> list[str]:
    return list(__all__)

__all__ = [
    "Pretrained",
    "PretrainedBlockDecoder",
    "PretrainedDecoder",
    "PretrainedFallback",
    "PretrainedMaskedDecoder",
    "PretrainedPipeline",
    "Processor",
    "load_params",
    "save_hf_layout",
    "save_params",
    "split_revision",
    "translate_config",
    "translate_weights",
]
