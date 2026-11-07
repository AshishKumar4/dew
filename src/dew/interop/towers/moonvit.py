"""MoonViT's vision tower and projector as a release stores them: their tensors' names and their
config's fields, both ways (`dew.interop.towers.common`), for the models `dew.nn.vision.moonvit` computes."""

from collections.abc import Callable, Mapping

import numpy as np

from dew import _records
from dew.interop.layouts import Route, Table
from dew.interop.unread import Stored
from dew.interop.weights import Tree, insert
from dew.nn.vision.moonvit import MoonViT, MoonViTProjector, MoonViTProjectorV2
from dew.registry import Record

from .common import ProjectorLayout, TowerBase, TowerLayout, _vision_section, tensor_routes
from .vit import PATCH_CONVOLUTION, translate_vit_weights

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


MOONVIT = TowerLayout("vision_tower.", _MOONVIT_ROUTES, read=_read_moonvit, shapes=_folded)


MOONVIT_PROJECTOR = ProjectorLayout("mm_projector.", {
    "pre_norm.weight": ("norm", "scale"), "pre_norm.bias": ("norm", "bias"),
    "proj.0.weight": ("fc1", "kernel"), "proj.0.bias": ("fc1", "bias"),
    "proj.2.weight": ("fc2", "kernel"), "proj.2.bias": ("fc2", "bias"),
})


MOONVIT_PROJECTOR_V2 = ProjectorLayout("mm_projector.", {
    "proj.0.weight": ("fc1", "kernel"), "proj.2.weight": ("fc2", "kernel"),
    "post_norm.weight": ("post_norm", "scale"),
})

TOWERS = {MoonViT: MOONVIT}
"""Where each of these tower kinds' tensors sit in a media checkpoint (`dew.interop.towers.tower_layout`)."""


PROJECTORS = {MoonViTProjector: MOONVIT_PROJECTOR, MoonViTProjectorV2: MOONVIT_PROJECTOR_V2}
"""Where each of these projector kinds' tensors sit in a media checkpoint
(`dew.interop.towers.projector_layout`)."""
