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
with its path map and its config and weight translators, and each family's
classes say where their tensors sit (`TowerLayout`, `ProjectorLayout`), so the
kinds `dew.registry` names are the one list of them. The package exports the
classes a caller constructs or subclasses, and reads a kind's tensors through
its class's layout (`tower_variables`, `projector_variables`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from dew.objectives.base import Variables

from .common import (
    ProjectorBase as ProjectorBase,
    ProjectorLayout as ProjectorLayout,
    TowerBase as TowerBase,
    TowerGeometry as TowerGeometry,
    TowerLayout as TowerLayout,
    projector_layout as projector_layout,
    tower_layout as tower_layout,
)
from .deepseek_v41 import DeepseekV41Projector as DeepseekV41Projector, DeepseekV41Vision as DeepseekV41Vision
from .gemma3n import Gemma3nProjector as Gemma3nProjector, Gemma3nVision as Gemma3nVision
from .gemma4 import (
    Gemma4Projector as Gemma4Projector,
    Gemma4Vision as Gemma4Vision,
    Gemma4VisionTransformer as Gemma4VisionTransformer,
)
from .llama4 import (
    Llama4Projector as Llama4Projector,
    Llama4Vision as Llama4Vision,
    Llama4VisionTransformer as Llama4VisionTransformer,
)
from .mimo import MiMoProjector as MiMoProjector, MiMoVision as MiMoVision
from .muse_glimmer import MuseGlimmerProjector as MuseGlimmerProjector, MuseGlimmerVision as MuseGlimmerVision
from .qwen35 import Qwen35Projector as Qwen35Projector, Qwen35Vision as Qwen35Vision
from .siglip import GemmaProjector as GemmaProjector, SiglipVision as SiglipVision
from .vit import (
    MergeProjector as MergeProjector,
    VisionTransformer as VisionTransformer,
    ViTLayout as ViTLayout,
)

if TYPE_CHECKING:
    from dew.interop.streaming import WeightLayout


def tower_variables(kind: str, hf_tensors: Mapping[str, np.ndarray], param_dtype: str
                    ) -> tuple[Variables, tuple[WeightLayout, ...]]:
    """One vision tower's variables, in the requested storage (`tower_layout`), and the layouts
    that write its tensors back from them."""
    return tower_layout(kind).variables(hf_tensors, param_dtype)


def projector_variables(kind: str, hf_tensors: Mapping[str, np.ndarray], param_dtype: str
                        ) -> tuple[Variables, tuple[WeightLayout, ...]]:
    """One projector kind's parameters, in the requested storage (`projector_layout`), and the
    layouts that write its tensors back from them."""
    return projector_layout(kind).variables(hf_tensors, param_dtype)
