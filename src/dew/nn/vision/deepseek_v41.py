"""DeepSeek-V4.1's vision trunk and aligner, with their checkpoint maps
(the release's `inference/vision.py`).
"""

import dataclasses
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew import records
from dew.interop.weights import in_params
from dew.nn.activations import gelu_exact
from dew.nn.mlp import GatedMLP
from dew.registry import Record

from .common import (
    ProjectorBase,
    ProjectorLayout,
    TowerBase,
    TowerGeometry,
    TowerLayout,
    _vision_section,
    block_tensor_path,
    rope_base,
)
from .vit import VisionTransformer, ViTLayout


@dataclasses.dataclass(frozen=True)
class DeepseekV41Vision(TowerBase):
    """DeepSeek-V4.1's ViT geometry, under its vision_config's names (vision.py:87-103).

    Its layout (`ViTLayout`): each `patch_size` square's pixels, channel-major, through one biased
    map; blocks of full attention whose queries, keys and values come from one biased map, rotated by
    the grid's rows then columns in fp32 as Qwen 3.5's vision rope turns them (vision.py:8-21), then
    a SwiGLU whose gate and up maps are one, each after an RMS norm (vision.py:46-84); a final RMS
    norm, and the trunk returned as its `[images, rows, columns, hidden_size]` grid, which the
    aligner groups.
    """

    num_hidden_layers: int = 32
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 2816
    patch_size: int = 14
    rope_theta: float = 10000.0

    def layout(self) -> ViTLayout:
        depth, heads = self.num_hidden_layers, self.num_attention_heads
        return ViTLayout(
            depth=depth, hidden_size=self.hidden_size, num_heads=heads, num_kv_heads=heads,
            head_dim=self.hidden_size // heads, intermediate_size=self.intermediate_size, activation="swiglu",
            in_channels=3, patch_size=self.patch_size, temporal_patch_size=1, spatial_merge_size=1,
            norm_type="rms", norm_eps=1e-6, norm_bias=False, mlp_bias=False, packed=True, mlp_packed=True,
            patch_bias=True, attention=("full",) * depth, order=("patches",) * depth, sinks=(False,) * depth,
            post_norm="rms", outer_norm_eps=1e-6, rope_theta=self.rope_theta, output="patch_grid")

    def build(self) -> nn.Module:
        return VisionTransformer(self.layout())

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(patch_size=self.patch_size, channels=3)


class DeepseekV41Projector(ProjectorBase):
    """DeepSeek-V4.1's aligner and image span (vision.py:106-119,
    model.py:1228-1239).

    The patch grid is zero-padded at its bottom and right to whole
    `downsample_ratio` squares, each square's features concatenate
    channel-major (unfold's order), and a two-layer exact-GELU map takes
    them to the decoder width. The span the decoder reads is a learned start
    vector, each row of aligned features followed by a learned newline
    vector, and a learned end vector: `rows * (columns + 1) + 2` positions.
    """

    vision_width: int
    downsample_ratio: int
    out_width: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, image_features) -> jax.Array:
        ratio = self.downsample_ratio
        images, rows, columns, _ = image_features.shape
        padded = jnp.pad(image_features, ((0, 0), (0, -rows % ratio), (0, -columns % ratio), (0, 0)))
        high, wide = padded.shape[1] // ratio, padded.shape[2] // ratio
        squares = padded.reshape(images, high, ratio, wide, ratio, self.vision_width).transpose(
            0, 1, 3, 5, 2, 4
        )
        grouped = squares.reshape(images, high, wide, -1)
        mlp = GatedMLP(self.out_width, self.out_width, in_features=grouped.shape[-1],
                        activation=gelu_exact, use_bias=True, names=("w1", "w2"), dtype=self.dtype,
                        precision=self.precision)
        nn.share_scope(self, mlp)
        aligned = mlp(grouped)
        start, newline, end = (
            self.param(name, nn.initializers.normal(1.0), (self.out_width,), jnp.float32).astype(
                aligned.dtype
            )
            for name in ("image_start", "image_newline", "image_end")
        )
        lines = jnp.concatenate(
            [aligned, jnp.broadcast_to(newline, (images, high, 1, self.out_width))], axis=2
        )
        return jnp.concatenate([jnp.broadcast_to(start, (images, 1, self.out_width)),
                                lines.reshape(images, high * (wide + 1), self.out_width),
                                jnp.broadcast_to(end, (images, 1, self.out_width))], axis=1)


# A block's maps as the decoder block names them (`block_tensor_path`).
_BLOCK_MAPS = {("attn", "wqkv"): ("self_attn", "qkv_proj"), ("attn", "wo"): ("self_attn", "o_proj"),
               ("mlp", "w1"): ("mlp", "gate_up_proj"), ("mlp", "w2"): ("mlp", "down_proj")}
_DEEPSEEK_V41_VISION_TENSORS = {
    "patch_embed.proj.weight": ("patch_embed", "kernel"),
    "patch_embed.proj.bias": ("patch_embed", "bias"),
    "norm.weight": ("post_norm", "scale"),
}


def deepseek_v41_vision_path(hf_name: str) -> tuple[str, ...]:
    """One DeepSeek-V4.1 ViT tensor name, its `vision.` prefix off, into its
    path in the trunk tree; anything else raises ValueError."""
    path = (_DEEPSEEK_V41_VISION_TENSORS.get(hf_name)
            or block_tensor_path(hf_name.split("."), _BLOCK_MAPS, norm_bias=False))
    if path is None:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    return path


def translate_deepseek_v41_vision_config(hf_config: Mapping[str, object]) -> Record:
    """A DeepSeek-V4.1 vision_config into a DeepseekV41Vision value's fields.

    The image-size fields (max_image_tokens, min_pixels, max_wh_ratio) plan
    the processor's resize and the downsample ratio is the aligner's, so the
    trunk reads neither.
    """
    vision = _vision_section(hf_config)
    if vision.get("model_type", "deepseek_v41_vision") != "deepseek_v41_vision":
        raise ValueError(f"vision model_type {vision.get('model_type')!r} is not DeepSeek-V4.1's ViT")
    width = records.integer(vision["hidden_size"], "hidden_size")
    heads = records.integer(vision["num_attention_heads"], "num_attention_heads")
    if width % heads or (width // heads) % 4:
        raise ValueError(f"hidden_size {width} over {heads} heads leaves no head width the 2D rotary "
                         "splits into height and width pairs")
    return {
        "class": "deepseek_v41", "fields": {
        "num_hidden_layers": records.integer(vision["num_hidden_layers"], "num_hidden_layers"),
        "hidden_size": width,
        "num_attention_heads": heads,
        "intermediate_size": records.integer(vision["intermediate_size"], "intermediate_size"),
        "patch_size": records.integer(vision["patch_size"], "patch_size"),
        "rope_theta": rope_base(vision, 10000.0),
    }}


def translate_deepseek_v41_projector_config(hf_config: Mapping[str, object],
                                            text_width: int) -> Record:
    """DeepSeek-V4.1's aligner fields: the ViT width, its downsample ratio and
    the decoder width its rows and span vectors enter."""
    vision = _vision_section(hf_config)
    return {
        "class": "deepseek_v41", "fields": {
        "vision_width": records.integer(vision["hidden_size"], "hidden_size"),
        "downsample_ratio": records.integer(vision.get("downsample_ratio", 3), "downsample_ratio"),
        "out_width": int(text_width),
    }}


DeepseekV41Vision.checkpoint = TowerLayout("vision.", in_params(deepseek_v41_vision_path))
# The aligner's maps under its `aligner.` prefix, and the span's learned vectors, which the release
# keeps at its top level (model.py:1215-1222).
DeepseekV41Projector.checkpoint = ProjectorLayout("aligner.", {
    "w1.weight": ("w1", "kernel"), "w1.bias": ("w1", "bias"),
    "w2.weight": ("w2", "kernel"), "w2.bias": ("w2", "bias"),
    "image_start": ("image_start",), "image_newline": ("image_newline",), "image_end": ("image_end",),
})
