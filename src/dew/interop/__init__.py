"""Checkpoint readers and translators, each imported only when you use its API.

So reading Hub metadata or a safetensors file does not import the decoder
translators or the diffusion pipeline loader.
"""

from typing import TYPE_CHECKING

from dew._exports import lookup as _lookup

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew.decision.sources import PretrainedDecision
    from dew.interop.hf_decoders import translate_config, translate_weights
    from dew.interop.pretrained import (
        Pretrained,
        PretrainedBlockDecoder,
        PretrainedDecoder,
        PretrainedFallback,
        PretrainedMaskedDecoder,
        PretrainedPipeline,
    )
    from dew.interop.processors import Processor
    from dew.interop.safetensors_io import load_params, save_hf_layout, save_params
    from dew.interop.sources import SourceReference

_EXPORTS = {
    "PretrainedDecision": "dew.decision.sources",
    "Pretrained": "dew.interop.pretrained", "PretrainedBlockDecoder": "dew.interop.pretrained",
    "PretrainedDecoder": "dew.interop.pretrained", "PretrainedFallback": "dew.interop.pretrained",
    "PretrainedMaskedDecoder": "dew.interop.pretrained", "PretrainedPipeline": "dew.interop.pretrained",
    "SourceReference": "dew.interop.sources",
    "Processor": "dew.interop.processors",
    "translate_config": "dew.interop.hf_decoders", "translate_weights": "dew.interop.hf_decoders",
    "load_params": "dew.interop.safetensors_io", "save_hf_layout": "dew.interop.safetensors_io",
    "save_params": "dew.interop.safetensors_io",
}


__getattr__ = _lookup(__name__, _EXPORTS)


def __dir__() -> list[str]:
    return list(__all__)

__all__ = [
    "Pretrained",
    "PretrainedBlockDecoder",
    "PretrainedDecision",
    "PretrainedDecoder",
    "PretrainedFallback",
    "PretrainedMaskedDecoder",
    "PretrainedPipeline",
    "Processor",
    "SourceReference",
    "load_params",
    "save_hf_layout",
    "save_params",
    "translate_config",
    "translate_weights",
]
