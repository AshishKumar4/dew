"""Kimi K2.5's and Kimi K3's vision trunk (MoonViT) and projector, with their checkpoint maps
(moonshotai/Kimi-K2.5@4d01dfe `modeling_kimi_k25.py`: MoonViT3dPretrainedModel, PatchMergerMLP;
moonshotai/Kimi-K3@f831ab6 `modeling_kimi_k3.py`, its options: RMS norms, bias-free maps, a wider
attention, PatchMergerMLPV2).
"""

import dataclasses
from collections.abc import Callable, Mapping
from typing import Literal

import numpy as np
from flax import linen as nn

from dew import _records
from dew.interop.layouts import Route, Table
from dew.interop.unread import Stored
from dew.interop.weights import Tree, insert
from dew.registry import Record

from .common import (
    ProjectorBase,
    ProjectorLayout,
    TowerBase,
    TowerGeometry,
    TowerLayout,
    _vision_section,
    head_width,
    tensor_routes,
)
from .vit import (
    PATCH_CONVOLUTION,
    LearnedPositions,
    MergeProjector,
    VisionTransformer,
    ViTLayout,
    translate_vit_weights,
)


@dataclasses.dataclass(frozen=True)
class MoonViT(TowerBase):
    """MoonViT's geometry under its vision_config's names.

    Its layout (`ViTLayout`): a convolution of each raster 14x14 patch, its channels then its pixels
    (`MoonVision3dPatchEmbed`), plus the learned `init_pos_emb_height` square table resampled to the
    frame's patch grid as `F.interpolate` resamples it (`pos_emb_interpolation_mode`, half-pixel centres,
    edges held) and, for an item of more than one frame, a fixed sinusoidal table of
    `init_pos_emb_time` frames (`Learnable2DInterpPosEmbDivided_fixed`); pre-norm blocks over every patch
    of an item, its frames together (`spatial_temporal`), their queries, keys and values one map
    rotated in adjacent pairs by the patch grid, a column's pair then a row's (`Rope2DPosEmbRepeated`),
    and an ungated tanh-GELU feed-forward (`MLP2`); a norm after the blocks; then each merge block's
    patches averaged over the item's frames (`tpool_patch_merger`, `sd2_tpool`). K2.5's norms are layer
    norms and its maps biased; K3 states RMS norms (`norm_type`), each at `rms_eps`, by default (None) the
    epsilon of the dtype it accumulates in as torch's RMSNorm(eps=None) takes it (float32's, float64's in
    float64), unbiased maps and a wider attention (`qkv_hidden_size`).
    """

    vt_hidden_size: int = 1152
    vt_intermediate_size: int = 4304
    vt_num_attention_heads: int = 16
    vt_num_hidden_layers: int = 27
    patch_size: int = 14
    merge_kernel_size: int = 2
    init_pos_emb_height: int = 64
    init_pos_emb_time: int = 4
    pos_emb_interpolation_mode: Literal["bicubic", "bilinear"] = "bicubic"
    qkv_hidden_size: int | None = None
    attn_bias: bool = True
    linear_bias: bool = True
    patch_embed_proj_bias: bool = True
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    rms_eps: float | None = None

    def __post_init__(self):
        if self.pos_emb_interpolation_mode not in ("bicubic", "bilinear"):
            raise ValueError(f"pos_emb_interpolation_mode {self.pos_emb_interpolation_mode!r} is neither "
                             "bicubic nor bilinear")
        if self.norm_type not in ("layernorm", "rmsnorm"):
            raise ValueError(f"norm_type {self.norm_type!r} is neither layernorm nor rmsnorm")

    def layout(self) -> ViTLayout:
        depth, heads = self.vt_num_hidden_layers, self.vt_num_attention_heads
        width = self.vt_hidden_size if self.qkv_hidden_size is None else self.qkv_hidden_size
        rms = self.norm_type == "rmsnorm"
        # torch's LayerNorm at its default epsilon (K2.5), or its RMSNorm at its accumulating dtype's (K3).
        eps = self.rms_eps if rms else 1e-5
        return ViTLayout(
            depth=depth, hidden_size=self.vt_hidden_size, num_heads=heads, num_kv_heads=heads,
            head_dim=head_width(width, heads), intermediate_size=self.vt_intermediate_size,
            activation="gelu", in_channels=3, patch_size=self.patch_size, temporal_patch_size=1,
            spatial_merge_size=self.merge_kernel_size, norm_type="rms" if rms else "layer", norm_eps=eps,
            norm_bias=not rms,
            mlp_bias=self.linear_bias, packed=True, patch_bias=self.patch_embed_proj_bias,
            attention=("item",) * depth, order=("patches",) * depth, sinks=(False,) * depth,
            positions=LearnedPositions(self.init_pos_emb_height, mode=self.pos_emb_interpolation_mode,
                                       align_corners=False, border=True,
                                       initializer=nn.initializers.normal(1.0)),
            rotary="adjacent", rotary_interleaved=True, rotary_columns_first=True,
            post_norm="rms" if rms else "layer", outer_norm_eps=eps, attention_bias=self.attn_bias,
            merge_blocks=False,
            frame_positions=self.init_pos_emb_time, pool_frames=True)

    def build(self) -> nn.Module:
        return VisionTransformer(self.layout())

    def geometry(self) -> TowerGeometry:
        return TowerGeometry(patch_size=self.patch_size, block_size=self.merge_kernel_size, channels=3,
                             pools_frames=True)


class MoonViTProjector(MergeProjector, ProjectorBase):
    """MoonViT's projector (`PatchMergerMLP`): a layer norm of each trunk token at `projector_ln_eps`,
    each merge block's tokens joined unit by unit, and two biased maps around an exact GELU, the first
    as wide as the block, the second to the decoder width."""


class MoonViTProjectorV2(MergeProjector, ProjectorBase):
    """Kimi K3's projector (`PatchMergerMLPV2`): each merge block's trunk tokens joined unit by unit, two
    unbiased maps around an exact GELU, the first as wide as the block, the second to the decoder width,
    and an RMS norm after them at `projector_ln_eps`; no norm before."""


_POSITIONS = "patch_embed.pos_emb.weight"
"""The learned position table, `[height, width, features]` as the release stores it."""

_TENSORS = {"patch_embed.proj.bias": ("patch_embed", "bias"),
            "encoder.final_layernorm.weight": ("post_norm", "scale"),
            "encoder.final_layernorm.bias": ("post_norm", "bias")}
_BLOCK_LINEARS = {"wqkv": ("self_attn", "qkv_proj"), "wo": ("self_attn", "o_proj"),
                  "norm0": ("input_layernorm",), "norm1": ("post_attention_layernorm",)}
_BLOCK_MLP = {"fc0": ("mlp", "up_proj"), "fc1": ("mlp", "down_proj")}


_BLOCK = "encoder.blocks.{index}."
_MOONVIT_ROUTES = Table(
    Route(PATCH_CONVOLUTION, "params/patch_embed/kernel"),
    Route(_POSITIONS, "params/position_table/embedding"),
    *tensor_routes(_TENSORS),
    *(Route(_BLOCK + f"{name}.{tensor}", "params/blocks_{index}/" + "/".join(
        (*path, ("scale" if name.startswith("norm") else "kernel") if tensor == "weight" else "bias")))
      for name, path in _BLOCK_LINEARS.items() for tensor in ("weight", "bias")),
    *(Route(_BLOCK + f"mlp.{name}.{tensor}", "params/blocks_{index}/" + "/".join(
        (*path, "kernel" if tensor == "weight" else "bias")))
      for name, path in _BLOCK_MLP.items() for tensor in ("weight", "bias")))
"""A MoonViT trunk's tensor names past `vision_tower.`, both ways; the patch convolution and the position
table read and write folded (`_read_moonvit`, `_folded`)."""


def _folded(tower: TowerBase) -> Mapping[str, tuple[int, ...]]:
    """The patch convolution `[width, 3, patch, patch]` and the position table `[height, width, features]`
    as the release stores them."""
    assert isinstance(tower, MoonViT)
    side = tower.init_pos_emb_height
    return {PATCH_CONVOLUTION: (tower.vt_hidden_size, 3, tower.patch_size, tower.patch_size),
            _POSITIONS: (side, side, tower.vt_hidden_size)}


def _read_moonvit(tensors: Mapping[str, Stored], path_of: Callable[[str], tuple[str, ...] | None],
                  param_dtype: str) -> tuple[Tree[Stored], tuple]:
    """The trunk's variables and their layouts (`translate_vit_weights`), the position table held as
    the rows `VisionTransformer` reads, row by row of the release's square."""
    from dew.interop.streaming import WeightLayout
    from dew.interop.unread import held
    from dew.interop.weights import checkpoint_array

    variables, layouts = translate_vit_weights({name: tensor for name, tensor in tensors.items()
                                                if name != _POSITIONS}, path_of, param_dtype)
    if _POSITIONS in tensors:
        table = checkpoint_array(tensors[_POSITIONS], param_dtype)
        rows = held(np.reshape(table, (-1, table.shape[-1])))
        insert(variables, ("params", "position_table", "embedding"), rows, _POSITIONS)
        layouts += (WeightLayout(_POSITIONS, (("params", "position_table", "embedding"),), table.shape),)
    return variables, layouts


_SPELLINGS = {"vt_hidden_size": "hidden_size", "vt_intermediate_size": "intermediate_size",
              "vt_num_attention_heads": "num_attention_heads", "vt_num_hidden_layers": "num_hidden_layers",
              "init_pos_emb_height": "pos_emb_height", "init_pos_emb_width": "pos_emb_width",
              "init_pos_emb_time": "pos_emb_time", "activation_func": "hidden_act"}
"""MoonViT's vision_config fields under the release's names and, where transformers' port
(`Kimi_K25VisionConfig`) spells one otherwise, under its; either reads the same checkpoint."""


def _spelled(vision: Mapping[str, object], name: str) -> str:
    """The name a vision_config states a field under: the release's, else transformers' where it states
    that (`_SPELLINGS`), else the release's, which reads the field's default."""
    other = _SPELLINGS.get(name, name)
    return other if name not in vision and other in vision else name


def translate_moonvit_config(hf_config: Mapping[str, object]) -> Record:
    """A Kimi K2.5 or K3 vision_config, the release's or transformers' (`_SPELLINGS`), into a MoonViT
    value's fields: a square position table, the 2x2 temporally pooled merge (`sd2_tpool`), divided
    fixed positions and spatio-temporal attention, the tanh-GELU feed-forward (`mlp2`), layer or RMS
    norms and the rotary's base of 10000, as the releases state them; any other is refused."""
    vision = _vision_section(hf_config)
    stated = {"merge_type": "sd2_tpool", "pos_emb_type": "divided_fixed",
              "video_attn_type": "spatial_temporal", "mlp_type": "mlp2"}
    for field, value in stated.items():
        if vision.get(field, value) != value:
            raise ValueError(f"vision_config {field} {vision[field]!r} is not MoonViT's {value!r} here")
    activation = vision.get(_spelled(vision, "activation_func"), "gelu_pytorch_tanh")
    if activation != "gelu_pytorch_tanh":
        raise ValueError(f"vision_config activation_func {activation!r} is not MoonViT's tanh GELU")
    rope = vision.get("rope_parameters")
    theta = 10000.0 if rope is None else _records.number(_records.record(rope, "rope_parameters").get(
        "rope_theta", 10000.0), "rope_theta")
    if theta != 10000.0:
        raise ValueError(f"vision_config rope_theta {theta} is not MoonViT's 10000")
    height = _records.integer(vision.get(_spelled(vision, "init_pos_emb_height"), 64), "init_pos_emb_height")
    width = vision.get(_spelled(vision, "init_pos_emb_width"), height)
    if _records.integer(width, "init_pos_emb_width") != height:
        raise ValueError("MoonViT's position table resamples here as a square only")
    merge = _records.integers(vision.get("merge_kernel_size", [2, 2]), "merge_kernel_size")
    if len(merge) != 2 or merge[0] != merge[1]:
        raise ValueError(f"merge_kernel_size {merge} is no square merge block")
    qkv = vision.get("qkv_hidden_size")
    norm = _records.text(vision.get("norm_type", "layernorm"), "norm_type")
    if norm not in ("layernorm", "rmsnorm"):
        raise ValueError(f"vision_config norm_type {norm!r} is neither of MoonViT's, layernorm or rmsnorm")
    return {"class": "moonvit", "fields": {
        **{name: _records.integer(vision.get(_spelled(vision, name)), name)
           for name in ("vt_hidden_size", "vt_intermediate_size", "vt_num_attention_heads",
                        "vt_num_hidden_layers")},
        "patch_size": _records.integer(vision.get("patch_size", 14), "patch_size"),
        "merge_kernel_size": merge[0],
        "init_pos_emb_height": height,
        "init_pos_emb_time": _records.integer(vision.get(_spelled(vision, "init_pos_emb_time"), 4),
                                              "init_pos_emb_time"),
        "pos_emb_interpolation_mode": _records.text(vision.get("pos_emb_interpolation_mode", "bicubic"),
                                                    "pos_emb_interpolation_mode"),
        "qkv_hidden_size": None if qkv is None else _records.integer(qkv, "qkv_hidden_size"),
        "attn_bias": _records.boolean(vision.get("attn_bias", True), "attn_bias"),
        "linear_bias": _records.boolean(vision.get("linear_bias", True), "linear_bias"),
        "patch_embed_proj_bias": _records.boolean(vision.get("patch_embed_proj_bias", True),
                                                  "patch_embed_proj_bias"),
        "norm_type": norm,
    }}


def translate_moonvit_projector_config(hf_config: Mapping[str, object], text_width: int) -> Record:
    """A Kimi K2.5 or K3 wrapper's projector fields (`mm_projector_type` 'patchmerger', K2.5's
    `PatchMergerMLP`, or 'patchmergerv2', K3's `PatchMergerMLPV2`): the merge block's width in and out of
    its first map, the decoder's out of its second, as the release's vision_config states them
    (`mm_hidden_size`, `projector_ln_eps`) or transformers' wrapper config (`projection_hidden_size`,
    `projection_layer_norm_eps`)."""
    vision = _vision_section(hf_config)
    kind = vision.get("mm_projector_type", "patchmerger")
    if kind not in ("patchmerger", "patchmergerv2"):
        raise ValueError(f"mm_projector_type {kind!r} is neither patch merger read here")
    width = _records.integer(vision.get("mm_hidden_size", hf_config.get("projection_hidden_size")),
                             "mm_hidden_size")
    merge = _records.integers(vision.get("merge_kernel_size", [2, 2]), "merge_kernel_size")[0]
    if _records.integer(vision.get("text_hidden_size", text_width), "text_hidden_size") != text_width:
        raise ValueError(f"vision_config text_hidden_size is not the decoder's width {text_width}")
    eps = vision.get("projector_ln_eps", hf_config.get("projection_layer_norm_eps", 1e-5))
    # K3's merger norms after its unbiased maps, and not before them (`MoonViTProjectorV2`).
    after = {"norm": False, "bias": False, "post_norm": True} if kind == "patchmergerv2" else {}
    return {"class": "moonvit_v2" if after else "moonvit", "fields": {
        "hidden_size": width, "spatial_merge_size": merge, "width": width * merge ** 2,
        "out_width": text_width, "norm_eps": _records.number(eps, "projector_ln_eps"), "activation": "gelu",
        **after}}


MoonViT.checkpoint = TowerLayout("vision_tower.", _MOONVIT_ROUTES, read=_read_moonvit, shapes=_folded)
MoonViTProjector.checkpoint = ProjectorLayout("mm_projector.", {
    "pre_norm.weight": ("norm", "scale"), "pre_norm.bias": ("norm", "bias"),
    "proj.0.weight": ("fc1", "kernel"), "proj.0.bias": ("fc1", "bias"),
    "proj.2.weight": ("fc2", "kernel"), "proj.2.bias": ("fc2", "bias"),
})
MoonViTProjectorV2.checkpoint = ProjectorLayout("mm_projector.", {
    "proj.0.weight": ("fc1", "kernel"), "proj.2.weight": ("fc2", "kernel"),
    "post_norm.weight": ("post_norm", "scale"),
})
