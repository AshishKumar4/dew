"""Vision towers, projectors and reference checkpoint maps.

transformers 5 ships no Flax classes, so the towers are vendored the way
`dew/nn/text_encoders.py` vendors CLIP, in the reference layout, with each
weight read from the checkpoint's safetensors under its reference tensor name.
The operation order follows transformers 5.16.1
`models/siglip/modeling_siglip.py`, `models/llama4/modeling_llama4.py`,
`models/gemma4/modeling_gemma4.py` and `models/qwen3_5/modeling_qwen3_5.py`.

The SigLIP trunk is patch convolution with bias, learned position embeddings
with no class token, pre-norm encoder blocks and a post layer norm. Its block
shares both halves with CLIP's: the attention and the feed-forward
(`dew.nn.text_encoders`), the MLP carrying the config's activation. The Llama 4
trunk is MetaCLIP-style: an unfold patch embedding without bias, a class token
appended after the patches, learned positions, a pre norm, full-attention
blocks with a complex rotary over the patch grid, a post norm, the class
token dropped, and the pixel-shuffle MLP inside the tower where the reference
keeps it. The Gemma 4 trunk is patch pixels scaled to [-1, 1] through a
bias-free map, summed 2D position tables, RMS-normed blocks with a 2D rotary
and gated feed-forwards, and a position pooler with standardization. The
Qwen 3.5 trunk is a NaViT-style patchify with the still frame repeated along
time, interpolated learned positions with a 2D rotary, full-attention blocks
and the merger MLP as its projector. Each tower's projector is a registered
value beside it: Gemma 3's averages each patch block, norms and maps to the
decoder width, Llama 4's maps the shuffled output to the decoder width,
Gemma 4's norms without a scale and maps, and Qwen 3.5's is the merger.
Gemma 3n uses the MobileNet-v5 encoder in ``dew.nn.mobilenet`` and its own hard/soft
vision embedder. DeepSeek-V4.1's trunk (the release's
`inference/vision.py`) is a biased patch map, RMS-normed blocks with the
same 2D rotary as Qwen 3.5's and a SwiGLU feed-forward, and a final norm; its
projector is the aligner, which groups squares of the patch grid through an
exact-GELU MLP and lays the image out as its decoder reads it.

Each family lives in its own module beside this one (`vision_siglip`,
`vision_llama4`, `vision_gemma4`, `vision_qwen35`, `vision_deepseek_v41`,
`vision_gemma3n`) over the bases in `vision_common`. This module re-exports
their names and keeps where each kind's tensors sit and which map reads them.
"""

from collections.abc import Callable, Mapping

import numpy as np

from dew.nn.vision_common import (
    PIXEL_VALUES_KEY as PIXEL_VALUES_KEY,
    ProjectorBase as ProjectorBase,
    TowerBase as TowerBase,
    TowerGeometry as TowerGeometry,
    projector_weight_path as projector_weight_path,
)
from dew.nn.vision_deepseek_v41 import (
    DeepseekV41Projector as DeepseekV41Projector,
    DeepseekV41ProjectorModule as DeepseekV41ProjectorModule,
    DeepseekV41Vision as DeepseekV41Vision,
    DeepseekV41VisionBlock as DeepseekV41VisionBlock,
    DeepseekV41VisionTransformer as DeepseekV41VisionTransformer,
    deepseek_v41_vision_path as deepseek_v41_vision_path,
    translate_deepseek_v41_projector_config as translate_deepseek_v41_projector_config,
    translate_deepseek_v41_projector_weights as translate_deepseek_v41_projector_weights,
    translate_deepseek_v41_vision_config as translate_deepseek_v41_vision_config,
    translate_deepseek_v41_vision_weights as translate_deepseek_v41_vision_weights,
)
from dew.nn.vision_gemma3n import (
    Gemma3nProjector as Gemma3nProjector,
    Gemma3nProjectorModule as Gemma3nProjectorModule,
    Gemma3nVision as Gemma3nVision,
    gemma3n_vision_path as gemma3n_vision_path,
    translate_gemma3n_projector_config as translate_gemma3n_projector_config,
    translate_gemma3n_projector_weights as translate_gemma3n_projector_weights,
    translate_gemma3n_vision_config as translate_gemma3n_vision_config,
    translate_gemma3n_vision_weights as translate_gemma3n_vision_weights,
)
from dew.nn.vision_gemma4 import (
    _GEMMA4_VISION_TENSORS as _GEMMA4_VISION_TENSORS,
    Gemma4ClippableLinear as Gemma4ClippableLinear,
    Gemma4Projector as Gemma4Projector,
    Gemma4ProjectorModule as Gemma4ProjectorModule,
    Gemma4Vision as Gemma4Vision,
    Gemma4VisionAttention as Gemma4VisionAttention,
    Gemma4VisionEncoderLayer as Gemma4VisionEncoderLayer,
    Gemma4VisionMLP as Gemma4VisionMLP,
    Gemma4VisionTransformer as Gemma4VisionTransformer,
    export_gemma4_vision_config as export_gemma4_vision_config,
    gemma4_vision_path as gemma4_vision_path,
    translate_gemma4_projector_config as translate_gemma4_projector_config,
    translate_gemma4_projector_weights as translate_gemma4_projector_weights,
    translate_gemma4_vision_config as translate_gemma4_vision_config,
    translate_gemma4_vision_weights as translate_gemma4_vision_weights,
)
from dew.nn.vision_llama4 import (
    Llama4Projector as Llama4Projector,
    Llama4ProjectorModule as Llama4ProjectorModule,
    Llama4Vision as Llama4Vision,
    Llama4VisionAdapter as Llama4VisionAdapter,
    Llama4VisionAdapterMLP as Llama4VisionAdapterMLP,
    Llama4VisionAttention as Llama4VisionAttention,
    Llama4VisionEncoderLayer as Llama4VisionEncoderLayer,
    Llama4VisionTransformer as Llama4VisionTransformer,
    llama4_vision_path as llama4_vision_path,
    pixel_shuffle as pixel_shuffle,
    translate_llama4_projector_config as translate_llama4_projector_config,
    translate_llama4_projector_weights as translate_llama4_projector_weights,
    translate_llama4_vision_config as translate_llama4_vision_config,
    translate_llama4_vision_weights as translate_llama4_vision_weights,
)
from dew.nn.vision_qwen35 import (
    Qwen35Projector as Qwen35Projector,
    Qwen35ProjectorModule as Qwen35ProjectorModule,
    Qwen35Vision as Qwen35Vision,
    Qwen35VisionAttention as Qwen35VisionAttention,
    Qwen35VisionBlock as Qwen35VisionBlock,
    Qwen35VisionTransformer as Qwen35VisionTransformer,
    qwen35_vision_path as qwen35_vision_path,
    translate_qwen35_projector_config as translate_qwen35_projector_config,
    translate_qwen35_projector_weights as translate_qwen35_projector_weights,
    translate_qwen35_vision_config as translate_qwen35_vision_config,
    translate_qwen35_vision_weights as translate_qwen35_vision_weights,
)
from dew.nn.vision_siglip import (
    GemmaProjector as GemmaProjector,
    GemmaProjectorModule as GemmaProjectorModule,
    SiglipVision as SiglipVision,
    SiglipVisionTransformer as SiglipVisionTransformer,
    siglip_vision_path as siglip_vision_path,
    translate_gemma_projector_config as translate_gemma_projector_config,
    translate_gemma_projector_weights as translate_gemma_projector_weights,
    translate_siglip_vision_config as translate_siglip_vision_config,
    translate_siglip_vision_weights as translate_siglip_vision_weights,
)
from dew.objectives.base import Variables

# Where each tower and projector kind's tensors sit in a media checkpoint, and
# the translators that read them.
TOWER_PREFIX = {"siglip": "vision_tower.", "llama4": "vision_model.", "gemma4": "vision_tower.",
                "qwen3_5": "visual.", "gemma3n": "vision_tower.", "deepseek_v41": "vision."}
PROJECTOR_PREFIX = {"gemma": "multi_modal_projector.", "llama4": "multi_modal_projector.",
                    "gemma4": "embed_vision.", "qwen3_5": "visual.merger.", "gemma3n": "embed_vision.",
                    "deepseek_v41": "aligner."}
TOWER_PATHS: dict[str, Callable[[str], tuple[str, ...] | None]] = {
    "siglip": siglip_vision_path, "llama4": llama4_vision_path, "gemma4": gemma4_vision_path,
    "qwen3_5": qwen35_vision_path, "gemma3n": gemma3n_vision_path, "deepseek_v41": deepseek_v41_vision_path}
# Gemma 4 is absent: its tower's map returns whole collections, not one params tree.
_TOWER_WEIGHTS = {"siglip": translate_siglip_vision_weights, "llama4": translate_llama4_vision_weights,
                  "qwen3_5": translate_qwen35_vision_weights, "gemma3n": translate_gemma3n_vision_weights,
                  "deepseek_v41": translate_deepseek_v41_vision_weights}
_PROJECTOR_WEIGHTS = {
    "gemma": translate_gemma_projector_weights,
    "llama4": translate_llama4_projector_weights,
    "gemma4": translate_gemma4_projector_weights,
    "qwen3_5": translate_qwen35_projector_weights,
    "gemma3n": translate_gemma3n_projector_weights,
    "deepseek_v41": translate_deepseek_v41_projector_weights,
}


def tower_variables(kind: str, hf_tensors: Mapping[str, np.ndarray], param_dtype: str) -> Variables:
    """One vision tower's variables, in the requested storage."""
    if kind == "gemma4":
        return translate_gemma4_vision_weights(hf_tensors, param_dtype=param_dtype)
    if kind not in _TOWER_WEIGHTS:
        raise ValueError(f"tower kind {kind!r} has no weight map here")
    return {"params": _TOWER_WEIGHTS[kind](hf_tensors, param_dtype=param_dtype)}


def projector_variables(kind: str, hf_tensors: Mapping[str, np.ndarray], param_dtype: str) -> Variables:
    """One projector kind's tensors, in the requested storage."""
    if kind not in _PROJECTOR_WEIGHTS:
        raise ValueError(f"projector kind {kind!r} has no weight map here")
    return _PROJECTOR_WEIGHTS[kind](hf_tensors, param_dtype=param_dtype)
