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
and the merger MLP as its projector. Each tower's projector is a
value beside it: Gemma 3's averages each patch block, norms and maps to the
decoder width, Llama 4's maps the shuffled output to the decoder width,
Gemma 4's norms without a scale and maps, and Qwen 3.5's is the merger.
Gemma 3n uses the MobileNet-v5 encoder in ``dew.nn.mobilenet`` and its own hard/soft
vision embedder. DeepSeek-V4.1's trunk (the release's
`inference/vision.py`) is a biased patch map, RMS-normed blocks with the
same 2D rotary as Qwen 3.5's and a SwiGLU feed-forward, and a final norm; its
projector is the aligner, which groups squares of the patch grid through an
exact-GELU MLP and lays the image out as its decoder reads it.

Each family lives in its own module of this package (`siglip`, `llama4`,
`gemma4`, `qwen35`, `deepseek_v41`, `gemma3n`) over the bases in `common`,
with its path map and its config and weight translators. The package exports
the classes a caller constructs or subclasses, and keeps where each kind's
tensors sit and which map reads them.
"""

import functools
from collections.abc import Callable, Mapping

import numpy as np

from dew.interop.weights import translate_parameters
from dew.objectives.base import Variables

from .common import (
    ProjectorBase as ProjectorBase,
    TowerBase as TowerBase,
    TowerGeometry as TowerGeometry,
    projector_weight_path,
)
from .deepseek_v41 import (
    DeepseekV41Projector as DeepseekV41Projector,
    DeepseekV41ProjectorModule as DeepseekV41ProjectorModule,
    DeepseekV41Vision as DeepseekV41Vision,
    DeepseekV41VisionTransformer as DeepseekV41VisionTransformer,
    deepseek_v41_vision_path,
)
from .gemma3n import (
    Gemma3nProjector as Gemma3nProjector,
    Gemma3nProjectorModule as Gemma3nProjectorModule,
    Gemma3nVision as Gemma3nVision,
    gemma3n_vision_path,
    translate_gemma3n_projector_weights,
)
from .gemma4 import (
    Gemma4Projector as Gemma4Projector,
    Gemma4ProjectorModule as Gemma4ProjectorModule,
    Gemma4Vision as Gemma4Vision,
    Gemma4VisionTransformer as Gemma4VisionTransformer,
    gemma4_vision_path,
    translate_gemma4_projector_weights,
)
from .llama4 import (
    Llama4Projector as Llama4Projector,
    Llama4ProjectorModule as Llama4ProjectorModule,
    Llama4Vision as Llama4Vision,
    Llama4VisionTransformer as Llama4VisionTransformer,
    llama4_vision_path,
)
from .qwen35 import (
    Qwen35Projector as Qwen35Projector,
    Qwen35ProjectorModule as Qwen35ProjectorModule,
    Qwen35Vision as Qwen35Vision,
    Qwen35VisionTransformer as Qwen35VisionTransformer,
    qwen35_vision_path,
    translate_qwen35_vision_weights,
)
from .siglip import (
    GemmaProjector as GemmaProjector,
    GemmaProjectorModule as GemmaProjectorModule,
    SiglipVision as SiglipVision,
    SiglipVisionTransformer as SiglipVisionTransformer,
    siglip_vision_path,
    translate_gemma_projector_weights,
)

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
# Qwen 3.5's tower reshapes its patch convolution, and three projectors check
# their tensors; every other map reads one tensor to one leaf.
_PROJECTOR_WEIGHTS: dict[str, Callable[..., Variables]] = {
    "gemma": translate_gemma_projector_weights,
    "gemma4": translate_gemma4_projector_weights,
    "gemma3n": translate_gemma3n_projector_weights,
}


def tower_variables(kind: str, hf_tensors: Mapping[str, np.ndarray], param_dtype: str) -> Variables:
    """One vision tower's variables, in the requested storage."""
    if kind == "qwen3_5":
        return {"params": translate_qwen35_vision_weights(hf_tensors, param_dtype=param_dtype)}
    if kind not in TOWER_PATHS:
        raise ValueError(f"tower kind {kind!r} has no weight map here")
    tree = translate_parameters(hf_tensors, TOWER_PATHS[kind], param_dtype)
    # Gemma 4's map names whole collections, its frozen buffers beside its params.
    return tree if kind == "gemma4" else {"params": tree}


def projector_variables(kind: str, hf_tensors: Mapping[str, np.ndarray], param_dtype: str) -> Variables:
    """One projector kind's tensors, in the requested storage."""
    if kind in _PROJECTOR_WEIGHTS:
        return _PROJECTOR_WEIGHTS[kind](hf_tensors, param_dtype=param_dtype)
    if kind not in PROJECTOR_PREFIX:
        raise ValueError(f"projector kind {kind!r} has no weight map here")
    return translate_parameters(hf_tensors, functools.partial(projector_weight_path, kind), param_dtype)
