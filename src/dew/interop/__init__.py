"""Checkpoint readers and translators, imported when their API is used.

Reading Hub metadata or a safetensors file needs neither the decoder
translators nor the diffusion pipeline loader.
"""

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .codecs import dequantize_fp8_blocks, fp8_format
    from .export import export_run
    from .hf_decoders import save_pretrained_decoder, translate_config, translate_weights
    from .hub import pull_from_hub, push_to_hub
    from .pretrained import Pretrained, Processor, load_pretrained, split_revision
    from .safetensors_io import load_params, save_hf_layout, save_params

_EXPORTS = {
    "Pretrained": "pretrained", "Processor": "pretrained", "load_pretrained": "pretrained",
    "split_revision": "pretrained",
    "dequantize_fp8_blocks": "codecs", "fp8_format": "codecs",
    "export_run": "export",
    "pull_from_hub": "hub", "push_to_hub": "hub",
    "save_pretrained_decoder": "hf_decoders", "translate_config": "hf_decoders",
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
    "Processor",
    "dequantize_fp8_blocks",
    "export_run",
    "fp8_format",
    "load_params",
    "load_pretrained",
    "pull_from_hub",
    "push_to_hub",
    "save_hf_layout",
    "save_params",
    "save_pretrained_decoder",
    "split_revision",
    "translate_config",
    "translate_weights",
]
