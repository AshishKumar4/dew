"""Checkpoint readers and translators, each imported only when you use its API.

So reading Hub metadata or a safetensors file does not import the decoder
translators or the diffusion pipeline loader.
"""

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew.interop.hf_decoders import translate_config, translate_weights
    from dew.interop.pretrained import (
        Pretrained,
        PretrainedBlockDecoder,
        PretrainedDecoder,
        PretrainedFallback,
        PretrainedMaskedDecoder,
        PretrainedPipeline,
        split_revision,
    )
    from dew.interop.processors import Processor
    from dew.interop.safetensors_io import load_params, save_hf_layout, save_params

_EXPORTS = {
    "Pretrained": "dew.interop.pretrained", "PretrainedBlockDecoder": "dew.interop.pretrained",
    "PretrainedDecoder": "dew.interop.pretrained", "PretrainedFallback": "dew.interop.pretrained",
    "PretrainedMaskedDecoder": "dew.interop.pretrained", "PretrainedPipeline": "dew.interop.pretrained",
    "split_revision": "dew.interop.pretrained",
    "Processor": "dew.interop.processors",
    "translate_config": "dew.interop.hf_decoders", "translate_weights": "dew.interop.hf_decoders",
    "load_params": "dew.interop.safetensors_io", "save_hf_layout": "dew.interop.safetensors_io",
    "save_params": "dew.interop.safetensors_io",
}


def __getattr__(name: str) -> type | Callable:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)


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
