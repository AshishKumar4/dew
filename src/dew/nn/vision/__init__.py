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

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np

from dew.batch import Variables
from dew.interop.weights import placed

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


@dataclasses.dataclass(frozen=True)
class TowerLayout:
    """Where one tower kind's tensors sit in a media checkpoint and the leaves they fill: `prefix`
    before every name, and `path` from a name past it to its leaf, rooted at its collection
    (`params`, and Gemma 4's `constants` beside it). `patch_dense` where the tower runs its patch
    convolution as one Dense map (`translate_vit_weights`). Loading (`variables`), routing a
    checkpoint's names (`hf_decoders`) and binding an export (`interop.pretrained`) read it alike."""

    prefix: str
    path: Callable[[str], tuple[str, ...] | None]
    patch_dense: bool = False

    def variables(self, hf_tensors: Mapping[str, np.ndarray], param_dtype: str
                  ) -> tuple[Variables, tuple[WeightLayout, ...]]:
        """The tower's variables from its tensors, in the requested storage, and the layouts that
        write each tensor it placed back from them (`placed`); a tensor the tower does not compute
        with, as SigLIP's pooling head, has none."""
        translate = translate_vit_weights if self.patch_dense else placed
        return translate(hf_tensors, self.path, param_dtype)


def _in_params(path: Callable[[str], tuple[str, ...] | None]) -> Callable[[str], tuple[str, ...] | None]:
    """`path`, whose leaves are parameters, rooted at `params`."""
    def rooted(name: str) -> tuple[str, ...] | None:
        leaf = path(name)
        return None if leaf is None else ("params", *leaf)
    return rooted


TOWERS = {
    "siglip": TowerLayout("vision_tower.", _in_params(siglip_vision_path)),
    "llama4": TowerLayout("vision_model.", _in_params(llama4_vision_path)),
    "gemma4": TowerLayout("vision_tower.", gemma4_vision_path),
    "qwen3_5": TowerLayout("visual.", _in_params(qwen35_vision_path), patch_dense=True),
    "gemma3n": TowerLayout("vision_tower.", _in_params(gemma3n_vision_path)),
    "deepseek_v41": TowerLayout("vision.", _in_params(deepseek_v41_vision_path)),
    "mimo": TowerLayout("visual.", _in_params(mimo_vision_path), patch_dense=True),
    "muse_glimmer": TowerLayout("vision_tower.", _in_params(muse_glimmer_vision_path)),
}
"""Each tower kind's checkpoint layout."""
PROJECTOR_PREFIX: dict[str, str | tuple[str, ...]] = {
    "gemma": "multi_modal_projector.", "llama4": "multi_modal_projector.", "gemma4": "embed_vision.",
    "qwen3_5": "visual.merger.", "gemma3n": "embed_vision.", "deepseek_v41": "aligner.",
    "mimo": "visual.merger.", "muse_glimmer": ("vision_adapter.", "vision_projection.")}
"""Where each projector kind's tensors sit: one prefix, or several whose tensors each keep the name
past their own."""
# Three projectors check their tensors; every other map reads one tensor to one leaf.
_PROJECTOR_WEIGHTS: dict[str, Callable[..., tuple[Variables, tuple[WeightLayout, ...]]]] = {
    "gemma": translate_gemma_projector_weights,
    "gemma4": translate_gemma4_projector_weights,
    "gemma3n": translate_gemma3n_projector_weights,
}


def tower_variables(kind: str, hf_tensors: Mapping[str, np.ndarray], param_dtype: str
                    ) -> tuple[Variables, tuple[WeightLayout, ...]]:
    """One vision tower's variables, in the requested storage (`TOWERS`), and the layouts that
    write its tensors back from them."""
    if kind not in TOWERS:
        raise ValueError(f"tower kind {kind!r} has no weight map here")
    return TOWERS[kind].variables(hf_tensors, param_dtype)


def projector_variables(kind: str, hf_tensors: Mapping[str, np.ndarray], param_dtype: str
                        ) -> tuple[Variables, tuple[WeightLayout, ...]]:
    """One projector kind's parameters, in the requested storage, and the layouts that write
    its tensors back from them."""
    if kind in _PROJECTOR_WEIGHTS:
        return _PROJECTOR_WEIGHTS[kind](hf_tensors, param_dtype=param_dtype)
    if kind not in PROJECTOR_PREFIX:
        raise ValueError(f"projector kind {kind!r} has no weight map here")
    return placed(hf_tensors, functools.partial(projector_weight_path, kind), param_dtype)
