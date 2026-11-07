"""DeepSeek V4.1's vision tower and projector as a release stores them: their tensors' names and their
config's fields, both ways (`dew.interop.towers.common`), for the models
`dew.nn.vision.deepseek_v41` computes."""

from collections.abc import Mapping

from dew import _records
from dew.interop.layouts import Table
from dew.nn.vision.deepseek_v41 import DeepseekV41Projector, DeepseekV41Vision
from dew.registry import Record

from .common import ProjectorLayout, TowerLayout, _vision_section, block_routes, rope_base, tensor_routes

_BLOCK_MAPS = {("attn", "wqkv"): ("self_attn", "qkv_proj"), ("attn", "wo"): ("self_attn", "o_proj"),
               ("mlp", "w1"): ("mlp", "gate_up_proj"), ("mlp", "w2"): ("mlp", "down_proj")}


_DEEPSEEK_V41_VISION_TENSORS = {
    "patch_embed.proj.weight": ("patch_embed", "kernel"),
    "patch_embed.proj.bias": ("patch_embed", "bias"),
    "norm.weight": ("post_norm", "scale"),
}


_DEEPSEEK_V41_ROUTES = Table(*tensor_routes(_DEEPSEEK_V41_VISION_TENSORS),
                             block_routes(_BLOCK_MAPS, norm_bias=False))
"""A DeepSeek-V4.1 ViT's tensor names past `vision.`, both ways."""


def translate_deepseek_v41_vision_config(hf_config: Mapping[str, object]) -> Record:
    """A DeepSeek-V4.1 vision_config into a DeepseekV41Vision value's fields.

    The image-size fields (max_image_tokens, min_pixels, max_wh_ratio) plan
    the processor's resize and the downsample ratio is the aligner's, so the
    trunk reads neither.
    """
    vision = _vision_section(hf_config)
    if vision.get("model_type", "deepseek_v41_vision") != "deepseek_v41_vision":
        raise ValueError(f"vision model_type {vision.get('model_type')!r} is not DeepSeek-V4.1's ViT")
    width = _records.integer(vision["hidden_size"], "hidden_size")
    heads = _records.integer(vision["num_attention_heads"], "num_attention_heads")
    if width % heads or (width // heads) % 4:
        raise ValueError(f"hidden_size {width} over {heads} heads leaves no head width the 2D rotary "
                         "splits into height and width pairs")
    return {
        "class": "deepseek_v41", "fields": {
        "num_hidden_layers": _records.integer(vision["num_hidden_layers"], "num_hidden_layers"),
        "hidden_size": width,
        "num_attention_heads": heads,
        "intermediate_size": _records.integer(vision["intermediate_size"], "intermediate_size"),
        "patch_size": _records.integer(vision["patch_size"], "patch_size"),
        "rope_theta": rope_base(vision, 10000.0),
    }}


def translate_deepseek_v41_projector_config(hf_config: Mapping[str, object],
                                            text_width: int) -> Record:
    """DeepSeek-V4.1's aligner fields: the ViT width, its downsample ratio and
    the decoder width its rows and span vectors enter."""
    vision = _vision_section(hf_config)
    return {
        "class": "deepseek_v41", "fields": {
        "vision_width": _records.integer(vision["hidden_size"], "hidden_size"),
        "downsample_ratio": _records.integer(vision.get("downsample_ratio", 3), "downsample_ratio"),
        "out_width": int(text_width),
    }}


DEEPSEEK_V41_VISION = TowerLayout("vision.", _DEEPSEEK_V41_ROUTES)


DEEPSEEK_V41_PROJECTOR = ProjectorLayout("aligner.", {
    "w1.weight": ("w1", "kernel"), "w1.bias": ("w1", "bias"),
    "w2.weight": ("w2", "kernel"), "w2.bias": ("w2", "bias"),
    "image_start": ("image_start",), "image_newline": ("image_newline",), "image_end": ("image_end",),
})

TOWERS = {DeepseekV41Vision: DEEPSEEK_V41_VISION}
"""Where each of these tower kinds' tensors sit in a media checkpoint (`dew.interop.towers.tower_layout`)."""


PROJECTORS = {DeepseekV41Projector: DEEPSEEK_V41_PROJECTOR}
"""Where each of these projector kinds' tensors sit in a media checkpoint
(`dew.interop.towers.projector_layout`)."""
