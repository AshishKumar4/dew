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
its class's layout (`tower_variables`, `projector_variables`). Each name is imported from its own module
when first read, so a caller of one family's tower reads no other family's.
"""

from typing import TYPE_CHECKING

from dew._exports import lookup as _lookup

if TYPE_CHECKING:  # written from _EXPORTS (tools/gen_exports.py)
    from dew.nn.vision.common import (
        ProjectorBase,
        ProjectorLayout,
        TowerBase,
        TowerGeometry,
        TowerLayout,
        projector_layout,
        projector_variables,
        tower_layout,
        tower_variables,
    )
    from dew.nn.vision.deepseek_v41 import DeepseekV41Projector, DeepseekV41Vision
    from dew.nn.vision.gemma3n import Gemma3nProjector, Gemma3nVision
    from dew.nn.vision.gemma4 import Gemma4Projector, Gemma4Vision, Gemma4VisionTransformer
    from dew.nn.vision.llama4 import Llama4Projector, Llama4Vision, Llama4VisionTransformer
    from dew.nn.vision.mimo import MiMoProjector, MiMoVision
    from dew.nn.vision.muse_glimmer import MuseGlimmerProjector, MuseGlimmerVision
    from dew.nn.vision.qwen35 import Qwen35Projector, Qwen35Vision
    from dew.nn.vision.siglip import GemmaProjector, SiglipVision
    from dew.nn.vision.vit import MergeProjector, VisionTransformer, ViTLayout

_EXPORTS = {
    "ProjectorBase": "dew.nn.vision.common",
    "ProjectorLayout": "dew.nn.vision.common",
    "TowerBase": "dew.nn.vision.common",
    "TowerGeometry": "dew.nn.vision.common",
    "TowerLayout": "dew.nn.vision.common",
    "projector_layout": "dew.nn.vision.common",
    "projector_variables": "dew.nn.vision.common",
    "tower_layout": "dew.nn.vision.common",
    "tower_variables": "dew.nn.vision.common",
    "DeepseekV41Projector": "dew.nn.vision.deepseek_v41",
    "DeepseekV41Vision": "dew.nn.vision.deepseek_v41",
    "Gemma3nProjector": "dew.nn.vision.gemma3n",
    "Gemma3nVision": "dew.nn.vision.gemma3n",
    "Gemma4Projector": "dew.nn.vision.gemma4",
    "Gemma4Vision": "dew.nn.vision.gemma4",
    "Gemma4VisionTransformer": "dew.nn.vision.gemma4",
    "Llama4Projector": "dew.nn.vision.llama4",
    "Llama4Vision": "dew.nn.vision.llama4",
    "Llama4VisionTransformer": "dew.nn.vision.llama4",
    "MiMoProjector": "dew.nn.vision.mimo",
    "MiMoVision": "dew.nn.vision.mimo",
    "MuseGlimmerProjector": "dew.nn.vision.muse_glimmer",
    "MuseGlimmerVision": "dew.nn.vision.muse_glimmer",
    "Qwen35Projector": "dew.nn.vision.qwen35",
    "Qwen35Vision": "dew.nn.vision.qwen35",
    "GemmaProjector": "dew.nn.vision.siglip",
    "SiglipVision": "dew.nn.vision.siglip",
    "MergeProjector": "dew.nn.vision.vit",
    "VisionTransformer": "dew.nn.vision.vit",
    "ViTLayout": "dew.nn.vision.vit",
}

__getattr__ = _lookup(__name__, _EXPORTS)


def __dir__() -> list[str]:
    return list(__all__)

__all__ = [
    "DeepseekV41Projector",
    "DeepseekV41Vision",
    "Gemma3nProjector",
    "Gemma3nVision",
    "Gemma4Projector",
    "Gemma4Vision",
    "Gemma4VisionTransformer",
    "GemmaProjector",
    "Llama4Projector",
    "Llama4Vision",
    "Llama4VisionTransformer",
    "MergeProjector",
    "MiMoProjector",
    "MiMoVision",
    "MuseGlimmerProjector",
    "MuseGlimmerVision",
    "ProjectorBase",
    "ProjectorLayout",
    "Qwen35Projector",
    "Qwen35Vision",
    "SiglipVision",
    "TowerBase",
    "TowerGeometry",
    "TowerLayout",
    "ViTLayout",
    "VisionTransformer",
    "projector_layout",
    "projector_variables",
    "tower_layout",
    "tower_variables",
]
