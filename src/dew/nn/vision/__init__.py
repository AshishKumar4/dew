"""Vision towers, projectors and reference checkpoint maps.

transformers 5 ships no Flax classes, so the towers are vendored the way
`dew/nn/text_encoders.py` vendors CLIP, in the reference layout, with each
weight read from the checkpoint's safetensors under its reference tensor name.
The operation order follows transformers 5.16.1
`models/siglip/modeling_siglip.py`, `models/llama4/modeling_llama4.py`,
`models/gemma4/modeling_gemma4.py`, `models/qwen3_5/modeling_qwen3_5.py` and
`models/muse_glimmer/modeling_muse_glimmer.py`, and MiMo-V2.6's `modeling_mimo_v2.py`.

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
Qwen 3.5, MiMo-V2.6 and Muse Glimmer trunks are configs of one windowed and
full vision transformer (`vit`): a NaViT-style patchify with the still frame
repeated along time, an optional resampled position table, decoder blocks
with a 2D rotary that attend over a frame, a band of keys or a window of
patches, each layer in its own token order, and the merge-block projector. Each tower's projector is a
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
`gemma4`, `qwen35`, `mimo`, `muse_glimmer`, `deepseek_v41`, `gemma3n`) over the
bases in `common` and the shared trunk in `vit`,
with its path map and its config and weight translators. The package exports
the classes a caller constructs or subclasses, and keeps where each kind's
tensors sit and which map reads them.
"""

import functools
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np

from dew.interop.weights import nested, placed
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
from .mimo import MiMoProjector as MiMoProjector, MiMoVision as MiMoVision, mimo_vision_path
from .muse_glimmer import (
    MuseGlimmerProjector as MuseGlimmerProjector,
    MuseGlimmerProjectorModule as MuseGlimmerProjectorModule,
    MuseGlimmerVision as MuseGlimmerVision,
    muse_glimmer_vision_path,
)
from .qwen35 import Qwen35Projector as Qwen35Projector, Qwen35Vision as Qwen35Vision, qwen35_vision_path
from .siglip import (
    GemmaProjector as GemmaProjector,
    GemmaProjectorModule as GemmaProjectorModule,
    SiglipVision as SiglipVision,
    SiglipVisionTransformer as SiglipVisionTransformer,
    siglip_vision_path,
    translate_gemma_projector_weights,
)
from .vit import (
    MergeProjector as MergeProjector,
    VisionTransformer as VisionTransformer,
    ViTLayout as ViTLayout,
    translate_vit_weights,
)

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout


# Where each tower and projector kind's tensors sit in a media checkpoint, and
# the translators that read them.
TOWER_PREFIX = {"siglip": "vision_tower.", "llama4": "vision_model.", "gemma4": "vision_tower.",
                "qwen3_5": "visual.", "gemma3n": "vision_tower.", "deepseek_v41": "vision.",
                "mimo": "visual.", "muse_glimmer": "vision_tower."}
PROJECTOR_PREFIX: dict[str, str | tuple[str, ...]] = {
    "gemma": "multi_modal_projector.", "llama4": "multi_modal_projector.", "gemma4": "embed_vision.",
    "qwen3_5": "visual.merger.", "gemma3n": "embed_vision.", "deepseek_v41": "aligner.",
    "mimo": "visual.merger.", "muse_glimmer": ("vision_adapter.", "vision_projection.")}
"""Where each projector kind's tensors sit: one prefix, or several whose tensors each keep the name
past their own."""
TOWER_PATHS: dict[str, Callable[[str], tuple[str, ...] | None]] = {
    "siglip": siglip_vision_path, "llama4": llama4_vision_path, "gemma4": gemma4_vision_path,
    "qwen3_5": qwen35_vision_path, "gemma3n": gemma3n_vision_path, "deepseek_v41": deepseek_v41_vision_path,
    "mimo": mimo_vision_path, "muse_glimmer": muse_glimmer_vision_path}
# The Qwen-style towers reshape their patch convolution (`translate_vit_weights`), and three projectors check
# their tensors; every other map reads one tensor to one leaf.
_PROJECTOR_WEIGHTS: dict[str, Callable[..., tuple[Variables, tuple[WeightLayout, ...]]]] = {
    "gemma": translate_gemma_projector_weights,
    "gemma4": translate_gemma4_projector_weights,
    "gemma3n": translate_gemma3n_projector_weights,
}


def tower_variables(kind: str, hf_tensors: Mapping[str, np.ndarray], param_dtype: str
                    ) -> tuple[Variables, tuple[WeightLayout, ...]]:
    """One vision tower's variables, in the requested storage, and the layouts that write each
    tensor it placed back from those variables (`placed`); a tensor the tower does not compute
    with, as SigLIP's pooling head, has none."""
    if kind in ("qwen3_5", "mimo"):
        params, layouts = translate_vit_weights(hf_tensors, TOWER_PATHS[kind], param_dtype)
        return {"params": params}, nested(layouts, 0, "params")
    if kind not in TOWER_PATHS:
        raise ValueError(f"tower kind {kind!r} has no weight map here")
    tree, layouts = placed(hf_tensors, TOWER_PATHS[kind], param_dtype)
    # Gemma 4's map names whole collections, its frozen buffers beside its params.
    return (tree, layouts) if kind == "gemma4" else ({"params": tree}, nested(layouts, 0, "params"))


def projector_variables(kind: str, hf_tensors: Mapping[str, np.ndarray], param_dtype: str
                        ) -> tuple[Variables, tuple[WeightLayout, ...]]:
    """One projector kind's parameters, in the requested storage, and the layouts that write
    its tensors back from them."""
    if kind in _PROJECTOR_WEIGHTS:
        return _PROJECTOR_WEIGHTS[kind](hf_tensors, param_dtype=param_dtype)
    if kind not in PROJECTOR_PREFIX:
        raise ValueError(f"projector kind {kind!r} has no weight map here")
    return placed(hf_tensors, functools.partial(projector_weight_path, kind), param_dtype)
