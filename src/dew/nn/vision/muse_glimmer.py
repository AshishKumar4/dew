"""Muse Glimmer's vision trunk and projector, with their checkpoint maps
(transformers 5.16.1 `models/muse_glimmer/modeling_muse_glimmer.py`).
"""

import dataclasses
from collections.abc import Mapping

import jax
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike

from dew import _records
from dew.interop.weights import in_params
from dew.nn.activations import HIDDEN_ACT
from dew.nn.attention import RMSNorm
from dew.nn.dense import Dense
from dew.nn.precision import float32_dot_general
from dew.nn.text_encoders import TOWER_ACTIVATIONS
from dew.registry import Record

from .common import (
    ProjectorBase,
    ProjectorLayout,
    TowerBase,
    TowerGeometry,
    TowerLayout,
    _vision_section,
    activation_name,
    block_tensor_path,
    rope_base,
)
from .vit import LearnedPositions, MergeProjector, VisionTransformer, ViTLayout


def default_layer_types(layers: int) -> tuple[str, ...]:
    """Muse Glimmer's attention cadence over `layers` layers: full attention every fourth layer and
    at the last, window attention elsewhere (MuseGlimmerVisionConfig's default `layer_types`)."""
    return tuple("full_attention" if (i + 1) % 4 == 0 or i == layers - 1 else "window_attention"
                 for i in range(layers))


@dataclasses.dataclass(frozen=True)
class MuseGlimmerVision(TowerBase):
    """Muse Glimmer's trunk geometry under its vision_config's names.

    Its layout (`ViTLayout`): a bias-free map of each raster patch's time-then-channel pixels, the
    learned `pos_emb_height` x `pos_emb_width` table resampled with half-pixel centres and zero
    padding (MuseGlimmerVisionPatchEmbedder), a layer norm before the blocks and one after, and
    CLIP-style blocks (layer norms at 1e-5, biased query, key and value maps, an ungated
    feed-forward) rotated by the patch grid, columns then rows, each counted from one. The tokens run
    window by window, `pos_emb_height` patches square; a `window_attention` layer attends within its
    window, a `full_attention` one over its frame.
    """

    num_hidden_layers: int = 50
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_attention_heads: int = 16
    hidden_act: str = "gelu"
    patch_size: int = 14
    patch_temporal: int = 2
    merge_size: int = 2
    pos_emb_height: int = 32
    pos_emb_width: int = 32
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    layer_types: tuple[str, ...] = default_layer_types(50)

    def __post_init__(self):
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(f"layer_types names {len(self.layer_types)} layers of {self.num_hidden_layers}")
        if self.pos_emb_height != self.pos_emb_width:
            raise ValueError("the reference resamples a square position table only")

    def layout(self) -> ViTLayout:
        heads = self.num_attention_heads
        return ViTLayout(
            depth=self.num_hidden_layers, hidden_size=self.hidden_size, num_heads=heads, num_kv_heads=heads,
            head_dim=self.hidden_size // heads, intermediate_size=self.intermediate_size,
            activation=TOWER_ACTIVATIONS[self.hidden_act], in_channels=3, patch_size=self.patch_size,
            temporal_patch_size=self.patch_temporal, spatial_merge_size=self.merge_size, norm_type="layer",
            norm_eps=1e-5, norm_bias=True, mlp_bias=True, packed=False, patch_bias=False,
            attention=tuple("window" if kind == "window_attention" else "full" for kind in self.layer_types),
            order=("windows",) * self.num_hidden_layers, sinks=(False,) * self.num_hidden_layers,
            positions=LearnedPositions(self.pos_emb_height, align_corners=False), pre_norm=True,
            post_norm="layer", outer_norm_eps=self.layer_norm_eps, rope_theta=self.rope_theta,
            rotary_columns_first=True, rotary_origin=1, window=self.pos_emb_height, merge_blocks=False,
            channels_first=False)

    def build(self) -> nn.Module:
        return VisionTransformer(self.layout())

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(patch_size=self.patch_size, block_size=self.merge_size, channels=3)


class MuseGlimmerProjector(ProjectorBase):
    """The merged image tokens as the decoder reads them (MuseGlimmerModel.get_image_features): each
    merge block's features channel by channel through the adapter's two bias-free maps, each followed
    by the activation (`MergeProjector`), a bias-free map to the decoder width, and a scale-free RMS
    norm (`perception_emb_norm`)."""

    vision_width: int
    merge_size: int
    adapter_width: int
    text_width: int
    activation: str = "gelu"
    norm_eps: float = 1e-6
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, image_features) -> jax.Array:
        adapted = MergeProjector(hidden_size=self.vision_width, spatial_merge_size=self.merge_size,
                                 width=self.adapter_width, out_width=self.adapter_width, norm=False,
                                 channel_major=True, bias=False, activation=self.activation,
                                 activation_after=True, dtype=self.dtype, precision=self.precision,
                                 name="adapter")(image_features)
        projected = Dense(self.text_width, use_bias=False, dtype=self.dtype, precision=self.precision,
                          name="projection", dot_general=float32_dot_general)(adapted)
        return RMSNorm(epsilon=self.norm_eps, with_scale=False, dtype=self.dtype, name="norm")(projected)


_TENSORS = {
    "patch_embedder.patch_embedding.weight": ("patch_embed", "kernel"),
    "patch_embedder.position_embedding_table.weight": ("position_table", "embedding"),
    "ln_pre.weight": ("pre_norm", "scale"), "ln_pre.bias": ("pre_norm", "bias"),
    "ln_post.weight": ("post_norm", "scale"), "ln_post.bias": ("post_norm", "bias"),
}
_BLOCK_MAPS = {("attn", "q_proj"): ("self_attn", "q_proj"), ("attn", "k_proj"): ("self_attn", "k_proj"),
               ("attn", "v_proj"): ("self_attn", "v_proj"), ("attn", "proj"): ("self_attn", "o_proj"),
               ("mlp", "fc1"): ("mlp", "up_proj"), ("mlp", "fc2"): ("mlp", "down_proj")}


def muse_glimmer_vision_path(hf_name: str) -> tuple[str, ...]:
    """One Muse Glimmer trunk tensor name, its `vision_tower.` prefix off, into its path in the trunk
    tree; anything else raises ValueError."""
    path = _TENSORS.get(hf_name) or block_tensor_path(hf_name.split("."), _BLOCK_MAPS, prefix="layers")
    if path is None:
        raise ValueError(f"unknown tensor name {hf_name!r}")
    return path


def translate_muse_glimmer_vision_config(hf_config: Mapping[str, object]) -> Record:
    """A MuseGlimmerVisionConfig into a MuseGlimmerVision value's fields; the rotary takes the
    default rope only, and the feed-forward the activations the ungated MLP runs."""
    vision = _vision_section(hf_config)
    if vision.get("model_type", "muse_glimmer_vision") != "muse_glimmer_vision":
        raise ValueError(f"vision model_type {vision.get('model_type')!r} is not Muse Glimmer's trunk")
    layers = _records.integer(vision["num_hidden_layers"], "num_hidden_layers")
    kinds = _records.strings(vision.get("layer_types") or default_layer_types(layers), "layer_types")
    unknown = sorted(set(kinds) - {"full_attention", "window_attention"})
    if unknown:
        raise ValueError(f"layer_types {unknown} are neither window_attention nor full_attention")
    return {"class": "muse_glimmer", "fields": {
        "num_hidden_layers": layers,
        "hidden_size": _records.integer(vision["hidden_size"], "hidden_size"),
        "intermediate_size": _records.integer(vision["intermediate_size"], "intermediate_size"),
        "num_attention_heads": _records.integer(vision["num_attention_heads"], "num_attention_heads"),
        "hidden_act": activation_name(vision, TOWER_ACTIVATIONS, "gelu"),
        "patch_size": _records.integer(vision["patch_size"], "patch_size"),
        "patch_temporal": _records.integer(vision.get("patch_temporal", 2), "patch_temporal"),
        "merge_size": _records.integer(vision.get("merge_size", 2), "merge_size"),
        "pos_emb_height": _records.integer(vision.get("pos_emb_height", 32), "pos_emb_height"),
        "pos_emb_width": _records.integer(vision.get("pos_emb_width", 32), "pos_emb_width"),
        "layer_norm_eps": _records.number(vision.get("layer_norm_eps", 1e-5), "layer_norm_eps"),
        # Transformers 5.17 names its grid rotary 'axial' (MuseGlimmerVisionConfig.default_rope_type).
        "rope_theta": rope_base(vision, 10000.0, axial=True),
        "layer_types": kinds,
    }}


def translate_muse_glimmer_projector_config(hf_config: Mapping[str, object]) -> Record:
    """A MuseGlimmerConfig's projector fields: the adapter reads the trunk's merge blocks, which its
    `out_hidden_size` must be, and the projection writes the text width."""
    vision = _vision_section(hf_config)
    text = _records.record(hf_config.get("text_config", {}), "text_config")
    width = _records.integer(vision["hidden_size"], "hidden_size")
    merge = _records.integer(vision.get("merge_size", 2), "merge_size")
    merged = _records.integer(hf_config.get("out_hidden_size", width * merge ** 2), "out_hidden_size")
    if merged != width * merge ** 2:
        raise ValueError(f"out_hidden_size ({merged}) is not the trunk's merge blocks ({width} x {merge}^2)")
    activation = activation_name(hf_config, HIDDEN_ACT, "gelu", ("projector_hidden_act",))
    return {"class": "muse_glimmer", "fields": {
        "vision_width": width,
        "merge_size": merge,
        "adapter_width": _records.integer(hf_config.get("projector_hidden_size", 4096),
                                         "projector_hidden_size"),
        "text_width": _records.integer(text["hidden_size"], "hidden_size"),
        "activation": activation,
        "norm_eps": _records.number(text.get("rms_norm_eps", 1e-6), "rms_norm_eps"),
    }}


MuseGlimmerVision.checkpoint = TowerLayout("vision_tower.", in_params(muse_glimmer_vision_path))
# The adapter and the projection, each name its prefix off.
MuseGlimmerProjector.checkpoint = ProjectorLayout(("vision_adapter.", "vision_projection."), {
    "fc1.weight": ("adapter", "fc1", "kernel"), "fc2.weight": ("adapter", "fc2", "kernel"),
    "weight": ("projection", "kernel"),
})
