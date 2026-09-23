"""Translate published latent diffusion checkpoints into native model values.

Source configurations and safetensors are read as data. No external model or
scheduler implementation is imported by this module.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from flax.typing import Dtype
from jax.typing import DTypeLike

from dew import records
from dew.interop.safetensors_io import read_weights
from dew.nn.backbones.unet_condition import UNet2DCondition, UNetStage
from dew.nn.text_encoders import checkpoint_array
from dew.registry import resolve_dtype

if TYPE_CHECKING:
    from dew.interop.pretrained import WeightLayout

type TensorTree = dict[str, np.ndarray | TensorTree]


def _insert(tree: TensorTree, path: tuple[str, ...], value: np.ndarray) -> None:
    node = tree
    for key in path[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"Tensor path crosses an existing leaf: {path}")
        node = child
    held = node.get(path[-1])
    # A tied tensor a checkpoint stores under two names is one parameter,
    # and each name is still bound for export. Two different arrays under
    # one path are two parameters and one of them would be lost.
    if isinstance(held, np.ndarray) and np.array_equal(held, value):
        return
    if held is not None:
        raise ValueError(f"Two source tensors map to {path}")
    node[path[-1]] = value


def _source_alias(tensors: Mapping[str, np.ndarray], owners: dict[tuple[str, ...], str],
                  path: tuple[str, ...], name: str) -> None:
    """Record `name` as the owner of `path`, and raise if another name differs there.

    The check runs on the source values, before the storage cast, because two
    different values can round to the same BF16 leaf.
    """
    previous = owners.setdefault(path, name)
    if previous != name and not np.array_equal(tensors[previous], tensors[name]):
        raise ValueError(f"Two different source tensors {previous!r} and {name!r} map to {path}")



class UNetFields(TypedDict):
    stages: tuple[UNetStage, ...]
    in_channels: int
    out_channels: int
    blocks_per_level: int
    linear_projection: bool
    additional_time_features: int
    middle_attention: bool
    frequency_shift: float
    cosine_first: bool
    dropout: float
    norm_groups: int
    norm_epsilon: float
    attention_norm_epsilon: float
    approximate_gelu: bool
    dtype: Dtype
    attention_impl: str


def unet_fields(config: Mapping[str, object], *, dtype: DTypeLike | None = "float32",
                attention_impl="auto") -> UNetFields:
    """Read a Diffusers UNet config into the fields `UNet2DCondition` takes.

    A control whose active value this UNet cannot compute raises rather than
    being dropped, so a checkpoint cannot load as a model it does not describe.
    """
    raw_widths = config["block_out_channels"]
    if not isinstance(raw_widths, (tuple, list)) or not raw_widths:
        raise ValueError("block_out_channels must be a nonempty sequence")
    widths = tuple(records.integer(value, "block_out_channels") for value in raw_widths)
    count = len(widths)
    def per_stage(value, name):
        values = tuple(value) if isinstance(value, (list, tuple)) else (value,) * count
        if len(values) != count:
            raise ValueError(f"{name} must have one entry per UNet stage")
        return values
    heads = tuple(records.integer(value, "attention heads") for value in per_stage(
        config.get("num_attention_heads") or config.get("attention_head_dim", 8), "attention heads"))
    depths = tuple(records.integer(value, "transformer depth") for value in per_stage(config.get("transformer_layers_per_block", 1), "transformer depth"))
    only_cross = tuple(records.boolean(value, "only_cross_attention") for value in per_stage(config.get("only_cross_attention", False), "only_cross_attention"))
    down, up = config["down_block_types"], config["up_block_types"]
    if not isinstance(down, (list, tuple)) or not isinstance(up, (list, tuple)) or len(down) != count or len(up) != count:
        raise ValueError("UNet down/up blocks must have one entry per stage")
    if any(kind not in ("DownBlock2D", "CrossAttnDownBlock2D") for kind in down):
        raise ValueError("Unsupported conditional UNet downsampling stage")
    if any(kind not in ("UpBlock2D", "CrossAttnUpBlock2D") for kind in up):
        raise ValueError("Unsupported conditional UNet upsampling stage")
    cross = tuple(kind == "CrossAttnDownBlock2D" for kind in down)
    if tuple(kind == "CrossAttnUpBlock2D" for kind in up) != cross[::-1]:
        raise ValueError("UNet decoder attention must mirror its encoder stages")
    addition = config.get("addition_embed_type")
    if addition not in (None, "text_time"):
        raise ValueError(f"Unsupported additional UNet condition: {addition}")
    middle = config.get("mid_block_type", "UNetMidBlock2DCrossAttn")
    if middle not in (None, "UNetMidBlock2DCrossAttn"):
        raise ValueError(f"Unsupported UNet middle block: {middle}")
    fixed = {
        "act_fn": "silu", "center_input_sample": False, "downsample_padding": 1,
        "mid_block_scale_factor": 1, "resnet_out_scale_factor": 1.0,
        "resnet_skip_time_act": False, "resnet_time_scale_shift": "default",
        "time_embedding_type": "positional", "time_embedding_act_fn": None,
        "timestep_post_act": None, "time_cond_proj_dim": None,
        "class_embed_type": None, "num_class_embeds": None, "class_embeddings_concat": False,
        "encoder_hid_dim": None, "encoder_hid_dim_type": None, "cross_attention_norm": None,
        "attention_type": "default", "dual_cross_attention": False,
        "reverse_transformer_layers_per_block": None, "upcast_attention": False,
        "conv_in_kernel": 3, "conv_out_kernel": 3,
    }
    for name, expected in fixed.items():
        if name in config and config[name] != expected:
            raise ValueError(f"Native UNet cannot honor active {name}={config[name]!r}")
    if config.get("time_embedding_dim") not in (None, widths[0] * 4):
        raise ValueError("Native UNet requires a time embedding four times its first width")
    groups = records.integer(config.get("norm_num_groups", 32), "norm_num_groups")
    epsilon = records.number(config.get("norm_eps", 1e-5), "norm_eps")
    if groups < 1 or epsilon <= 0 or any(width < 1 or width % groups for width in widths):
        raise ValueError("UNet normalization requires positive epsilon and widths divisible by its group count")
    if any(head < 1 or width % head or depth < 1 for width, head, depth in zip(widths, heads, depths, strict=True)):
        raise ValueError("UNet stage heads must divide their width and transformer depth must be positive")
    source_name = config.get("_class_name", "UNet2DConditionModel")
    if source_name not in ("UNet2DConditionModel", "FlaxUNet2DConditionModel"):
        raise ValueError(f"Unsupported UNet source class: {source_name}")
    flax_semantics = source_name == "FlaxUNet2DConditionModel"
    if flax_semantics and (groups != 32 or epsilon != 1e-5):
        raise ValueError("The published Flax UNet has fixed normalization groups and epsilon")
    return UNetFields(
        stages=tuple(UNetStage(width, head, depth, attended, cross_only)
                     for width, head, depth, attended, cross_only in zip(widths, heads, depths, cross, only_cross, strict=True)),
        in_channels=records.integer(config["in_channels"], "in_channels"), out_channels=records.integer(config["out_channels"], "out_channels"),
        blocks_per_level=records.integer(config.get("layers_per_block", 2), "layers_per_block"),
        linear_projection=records.boolean(config.get("use_linear_projection", False), "use_linear_projection"),
        additional_time_features=records.integer(config["addition_time_embed_dim"], "addition_time_embed_dim") if addition else 0,
        middle_attention=middle is not None, frequency_shift=records.number(config.get("freq_shift", 0), "freq_shift"),
        cosine_first=records.boolean(config.get("flip_sin_to_cos", True), "flip_sin_to_cos"),
        dropout=records.number(config.get("dropout", 0), "dropout"), norm_groups=groups, norm_epsilon=epsilon,
        attention_norm_epsilon=1e-5 if flax_semantics else 1e-6, approximate_gelu=flax_semantics,
        dtype=resolve_dtype(dtype), attention_impl=attention_impl)


def _unet_path(name: str, rank: int) -> tuple[str, ...]:
    """Return the `UNet2DCondition` parameter path for a Diffusers UNet tensor.

    `parts` holds the dotted name segments still to read and shrinks as each
    one is consumed; `path` holds the native path built so far; `leaf` is the
    trailing `weight` or `bias`. `rank` picks `scale` over `kernel` for a
    one-dimensional weight. An unknown name raises with that name.
    """
    parts = name.split(".")
    leaf = parts.pop()
    if leaf not in ("weight", "bias"):
        raise ValueError(f"Unknown UNet tensor {name}")
    head = parts.pop(0)
    roots = {"conv_in": "input", "conv_out": "output", "conv_norm_out": "output_norm",
             "time_embedding": "time", "add_embedding": "additional_time"}
    if head in roots:
        path = [roots[head]]
    elif head in ("down_blocks", "up_blocks"):
        path = [f"{'down' if head == 'down_blocks' else 'up'}_{parts.pop(0)}"]
        kind, index = parts.pop(0), parts.pop(0)
        if kind == "resnets":
            path.append(f"residual_{index}")
        elif kind == "attentions":
            path.append(f"attention_{index}")
        elif kind in ("downsamplers", "upsamplers") and index == "0" and parts == ["conv"]:
            path.append("resize")
            parts = []
        else:
            raise ValueError(f"Unknown UNet stage tensor {name}")
    elif head == "mid_block":
        kind, index = parts.pop(0), parts.pop(0)
        if kind == "resnets" and index in ("0", "1"):
            path = ["middle_in" if index == "0" else "middle_out"]
        elif kind == "attentions" and index == "0":
            path = ["middle_attention"]
        else:
            raise ValueError(f"Unknown UNet middle tensor {name}")
    else:
        raise ValueError(f"Unknown UNet tensor {name}")
    attention = any(part.startswith("attention_") for part in path) or path[0] == "middle_attention"
    if attention and parts[:1] == ["transformer_blocks"]:
        path.append(f"layer_{parts[1]}")
        parts = parts[2:]
        kind = parts.pop(0)
        if kind in ("attn1", "attn2"):
            path.append("self_attention" if kind == "attn1" else "cross_attention")
            projection = parts.pop(0)
            if projection == "to_out" and parts == ["0"]:
                path.append("output")
                parts = []
            elif projection in ("to_q", "to_k", "to_v"):
                path.append(projection[-1])
            else:
                raise ValueError(f"Unknown UNet attention tensor {name}")
        elif kind == "ff":
            path.append("feed_forward")
            if parts[:2] == ["net", "0"]:
                path.append("net_0")
                parts = parts[2:]
            elif parts == ["net", "2"]:
                path.append("net_2")
                parts = []
            else:
                raise ValueError(f"Unknown UNet feed-forward tensor {name}")
        elif kind in ("norm1", "norm2", "norm3"):
            path.append({"norm1": "self_norm", "norm2": "cross_norm", "norm3": "ff_norm"}[kind])
        else:
            raise ValueError(f"Unknown UNet transformer tensor {name}")
    names = {"time_emb_proj": "temb_projection", "conv_shortcut": "residual_conv",
             "linear_1": "in_proj", "linear_2": "out_proj", "proj_in": "input", "proj_out": "output"}
    path.extend(names.get(part, part) for part in parts)
    path.append("bias" if leaf == "bias" else "scale" if rank == 1 else "kernel")
    return tuple(path)


def translate_unet_weights(tensors: Mapping[str, np.ndarray], model: UNet2DCondition, *,
                           param_dtype: str = "float32") -> tuple[TensorTree, tuple[WeightLayout, ...]]:
    """Map UNet tensors into a parameter tree and the layouts that invert it.

    Each leaf is cast to `param_dtype` before its transpose. Attention kernels
    are also reshaped to per-head axes, which is why this does not go through
    `record_layouts`.
    """
    from dew.interop.pretrained import WeightLayout
    parameters: TensorTree = {}
    layouts = []
    owners: dict[tuple[str, ...], str] = {}
    for name, tensor in tensors.items():
        path = _unet_path(name, tensor.ndim)
        _source_alias(tensors, owners, path, name)
        tensor = checkpoint_array(tensor, param_dtype)
        value, transpose = tensor, None
        if path[-1] == "kernel":
            value = tensor.transpose(2, 3, 1, 0) if tensor.ndim == 4 else tensor.T
            transpose = (3, 2, 0, 1) if tensor.ndim == 4 else (1, 0)
            if "self_attention" in path or "cross_attention" in path:
                root = path[0]
                stage = (model.stages[int(root.removeprefix("down_"))] if root.startswith("down_") else
                         model.stages[::-1][int(root.removeprefix("up_"))] if root.startswith("up_") else model.stages[-1])
                if path[-2] in ("q", "k", "v"):
                    value = value.reshape(value.shape[0], stage.heads, stage.features // stage.heads)
                    transpose = (1, 2, 0)
                else:
                    value = value.reshape(stage.heads, stage.features // stage.heads, stage.features)
                    transpose = (2, 0, 1)
        _insert(parameters, path, value)
        layouts.append(WeightLayout("unet/" + name, (("params", *path),), tensor.shape, transpose))
    return parameters, tuple(layouts)



class SD3Fields(TypedDict):
    patch_size: int
    in_channels: int
    out_channels: int
    num_layers: int
    heads: int
    head_dim: int
    joint_attention_dim: int
    caption_projection_dim: int
    pooled_projection_dim: int
    sample_size: int
    pos_embed_max_size: int
    dual_attention_layers: tuple[int, ...]
    qk_norm: str | None
    dtype: object
    attention_impl: str


def sd3_fields(config: Mapping[str, object], *, dtype: DTypeLike | None = "float32",
               attention_impl="auto") -> SD3Fields:
    """Read a published `SD3Transformer2DModel` config into native model fields.

    Every geometry control the source declares is read. A control whose active
    meaning this model does not carry is refused rather than dropped, so a
    checkpoint that means something else cannot load as if it did not.
    """
    from dew.interop.pretrained import resolve_dtype

    heads = records.integer(config["num_attention_heads"], "num_attention_heads")
    head_dim = records.integer(config["attention_head_dim"], "attention_head_dim")
    channels = records.integer(config["in_channels"], "in_channels")
    out_channels = config.get("out_channels")
    dual = config.get("dual_attention_layers") or ()
    if not isinstance(dual, (list, tuple)) or any(type(index) is not int for index in dual):
        raise ValueError("dual_attention_layers must be a sequence of block indices")
    qk_norm = config.get("qk_norm")
    if qk_norm not in (None, "rms_norm"):
        raise ValueError(f"Native SD3 implements qk_norm 'rms_norm', not {qk_norm!r}")
    return SD3Fields(
        patch_size=records.integer(config["patch_size"], "patch_size"), in_channels=channels,
        out_channels=channels if out_channels is None else records.integer(out_channels, "out_channels"),
        num_layers=records.integer(config["num_layers"], "num_layers"), heads=heads, head_dim=head_dim,
        joint_attention_dim=records.integer(config["joint_attention_dim"], "joint_attention_dim"),
        caption_projection_dim=records.integer(config["caption_projection_dim"], "caption_projection_dim"),
        pooled_projection_dim=records.integer(config["pooled_projection_dim"], "pooled_projection_dim"),
        sample_size=records.integer(config["sample_size"], "sample_size"),
        pos_embed_max_size=records.integer(config["pos_embed_max_size"], "pos_embed_max_size"),
        dual_attention_layers=tuple(dual), qk_norm=qk_norm, dtype=resolve_dtype(dtype),
        attention_impl=attention_impl)


_SD3_EMBEDDERS = {
    "pos_embed.proj": ("pos_embed_proj",),
    "context_embedder": ("context_embedder",),
    "proj_out": ("proj_out",),
    "norm_out.linear": ("norm_out", "linear"),
    "time_text_embed.timestep_embedder.linear_1": ("timestep_embedder_linear_1",),
    "time_text_embed.timestep_embedder.linear_2": ("timestep_embedder_linear_2",),
    "time_text_embed.text_embedder.linear_1": ("text_embedder_linear_1",),
    "time_text_embed.text_embedder.linear_2": ("text_embedder_linear_2",),
}
# The MM-DiT tensors SD3 and Flux share: the modulation linear of each
# stream, the joint attention's projections, output projections and per-head
# norms, and the two feed-forwards. The attention's own name is the caller's,
# since SD3.5's dual-attention blocks carry a second one.
_JOINT_ATTENTION = ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj",
                    "to_add_out", "norm_q", "norm_k", "norm_added_q", "norm_added_k")
_SINGLE_ATTENTION = ("to_q", "to_k", "to_v", "norm_q", "norm_k")


def _dit_leaf(leaf: str) -> str:
    if leaf == "weight":
        return "kernel"
    if leaf != "bias":
        raise ValueError(f"unknown tensor leaf {leaf!r}")
    return "bias"


def _dit_attention(block: tuple[str, ...], attention: str, inner: list[str], leaf: str,
                   name: str, *, joint: bool) -> tuple[str, ...]:
    """Return the path of one attention tensor inside `block`.

    The tensor is a projection, the output projection a joint attention holds,
    or a per-head norm's scale. `inner` is the name's segments below the
    attention.
    """
    if joint and inner == ["to_out", "0"]:
        return (*block, attention, "to_out_0", _dit_leaf(leaf))
    if len(inner) == 1 and inner[0] in (_JOINT_ATTENTION if joint else _SINGLE_ATTENTION):
        if inner[0].startswith("norm"):
            if leaf != "weight":
                raise ValueError(f"unknown tensor name {name!r}")
            return (*block, attention, inner[0], "scale")
        return (*block, attention, inner[0], _dit_leaf(leaf))
    raise ValueError(f"unknown tensor name {name!r}")


def _dit_block(block: tuple[str, ...], rest: list[str], leaf: str, name: str, *,
               attentions: tuple[str, ...]) -> tuple[str, ...]:
    """Return the path of one tensor inside a joint block.

    The tensor is either stream's modulation, one of the block's attentions,
    or either stream's feed-forward. `rest` is the name's segments below the
    block.
    """
    if rest in (["norm1", "linear"], ["norm1_context", "linear"]):
        return (*block, rest[0], "linear", _dit_leaf(leaf))
    if len(rest) > 1 and rest[0] in attentions:
        return _dit_attention(block, rest[0], rest[1:], leaf, name, joint=True)
    if len(rest) > 1 and rest[0] in ("ff", "ff_context"):
        if rest[1:] == ["net", "0", "proj"]:
            return (*block, rest[0], "net_0_proj", _dit_leaf(leaf))
        if rest[1:] == ["net", "2"]:
            return (*block, rest[0], "net_2", _dit_leaf(leaf))
    raise ValueError(f"unknown tensor name {name!r}")


def _sd3_path(name: str) -> tuple[str, ...] | None:
    """Return the `SD3Transformer` path for a published SD3 tensor name.

    The position buffer is not a parameter and comes back as None. Every other
    declared tensor maps, and an unknown name raises with that name so a
    checkpoint carrying something else cannot load silently.
    """
    if name == "pos_embed.pos_embed":
        return None
    parts = name.split(".")
    leaf, stem = parts[-1], ".".join(parts[:-1])
    if stem in _SD3_EMBEDDERS:
        return (*_SD3_EMBEDDERS[stem], _dit_leaf(leaf))
    if parts[0] == "transformer_blocks" and parts[1].isdigit():
        return _dit_block((f"transformer_blocks_{parts[1]}",), parts[2:-1], leaf, name,
                          attentions=("attn", "attn2"))
    raise ValueError(f"unknown tensor name {name!r}")


class FluxFields(TypedDict):
    patch_size: int
    in_channels: int
    out_channels: int
    num_layers: int
    num_single_layers: int
    heads: int
    head_dim: int
    joint_attention_dim: int
    pooled_projection_dim: int
    guidance_embeds: bool
    axes_dims_rope: tuple[int, ...]
    dtype: object
    attention_impl: str


def flux_fields(config: Mapping[str, object], *, dtype: DTypeLike | None = "float32",
                attention_impl="auto") -> FluxFields:
    """Read a published `FluxTransformer2DModel` config into native model fields.

    Every geometry control the source declares is read, including whether it
    embeds its distilled guidance, which changes what the model takes as an
    input rather than only which tensors it holds.
    """
    from dew.interop.pretrained import resolve_dtype

    channels = records.integer(config["in_channels"], "in_channels")
    out_channels = config.get("out_channels")
    axes = config.get("axes_dims_rope", (16, 56, 56))
    if not isinstance(axes, (list, tuple)) or any(type(size) is not int for size in axes):
        raise ValueError("axes_dims_rope must be a sequence of channel counts")
    heads = records.integer(config["num_attention_heads"], "num_attention_heads")
    head_dim = records.integer(config["attention_head_dim"], "attention_head_dim")
    if sum(axes) != head_dim:
        raise ValueError(f"axes_dims_rope {tuple(axes)} must cover the {head_dim} head channels")
    if any(size % 2 for size in axes):
        raise ValueError(f"axes_dims_rope {tuple(axes)} rotates channel pairs, so each is even")
    return FluxFields(
        patch_size=records.integer(config.get("patch_size", 1), "patch_size"), in_channels=channels,
        out_channels=channels if out_channels is None else records.integer(out_channels, "out_channels"),
        num_layers=records.integer(config["num_layers"], "num_layers"),
        num_single_layers=records.integer(config["num_single_layers"], "num_single_layers"),
        heads=heads, head_dim=head_dim,
        joint_attention_dim=records.integer(config["joint_attention_dim"], "joint_attention_dim"),
        pooled_projection_dim=records.integer(config["pooled_projection_dim"], "pooled_projection_dim"),
        guidance_embeds=records.boolean(config.get("guidance_embeds", False), "guidance_embeds"),
        axes_dims_rope=tuple(axes), dtype=resolve_dtype(dtype),
        attention_impl=attention_impl)


_FLUX_EMBEDDERS = {
    "x_embedder": ("x_embedder",),
    "context_embedder": ("context_embedder",),
    "proj_out": ("proj_out",),
    "norm_out.linear": ("norm_out", "linear"),
    "time_text_embed.timestep_embedder.linear_1": ("timestep_embedder_linear_1",),
    "time_text_embed.timestep_embedder.linear_2": ("timestep_embedder_linear_2",),
    "time_text_embed.guidance_embedder.linear_1": ("guidance_embedder_linear_1",),
    "time_text_embed.guidance_embedder.linear_2": ("guidance_embedder_linear_2",),
    "time_text_embed.text_embedder.linear_1": ("text_embedder_linear_1",),
    "time_text_embed.text_embedder.linear_2": ("text_embedder_linear_2",),
}


def _flux_path(name: str) -> tuple[str, ...]:
    """Return the `FluxTransformer` path for a published Flux tensor name.

    A single-stream block's fused output projection is `proj_out` in the
    source, inside its own block; here it is `proj_fused`, since the model's
    own `proj_out` runs the other way round and one name carries one pair of
    axes. Its attention is `pre_only`: it holds neither the context
    projections nor any output of its own.
    """
    parts = name.split(".")
    leaf, stem = parts[-1], ".".join(parts[:-1])
    if stem in _FLUX_EMBEDDERS:
        return (*_FLUX_EMBEDDERS[stem], _dit_leaf(leaf))
    if parts[0] == "transformer_blocks" and parts[1].isdigit():
        return _dit_block((f"transformer_blocks_{parts[1]}",), parts[2:-1], leaf, name,
                          attentions=("attn",))
    if parts[0] == "single_transformer_blocks" and parts[1].isdigit():
        block = (f"single_transformer_blocks_{parts[1]}",)
        rest = parts[2:-1]
        if rest == ["norm", "linear"]:
            return (*block, "norm", "linear", _dit_leaf(leaf))
        if rest == ["proj_mlp"]:
            return (*block, "proj_mlp", _dit_leaf(leaf))
        if rest == ["proj_out"]:
            return (*block, "proj_fused", _dit_leaf(leaf))
        if len(rest) > 1 and rest[0] == "attn":
            return _dit_attention(block, "attn", rest[1:], leaf, name, joint=False)
    raise ValueError(f"unknown tensor name {name!r}")


def translate_flux_weights(tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
                           ) -> tuple[TensorTree, tuple[WeightLayout, ...]]:
    """Map Flux tensors into a parameter tree and the layouts that invert it.

    Rotary tables are computed from input ids, so Flux stores no positional
    buffer and there is nothing to place outside `params`.
    """
    return record_layouts("transformer", tensors, _flux_path, ("params",), param_dtype=param_dtype)


def translate_sd3_weights(tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
                          ) -> tuple[TensorTree, TensorTree, tuple[WeightLayout, ...]]:
    """Map SD3 tensors into a parameter tree, its buffers and the inverting layouts.

    The source's position embedding is a persistent sin/cos-initialized
    buffer, so its stored values land in the `buffers` collection: an
    optimizer and an EMA see only `params`, and export writes the stored
    array back unchanged.
    """
    from dew.interop.pretrained import WeightLayout

    parameters, layouts = record_layouts(
        "transformer", tensors, _sd3_path, ("params",), param_dtype=param_dtype)
    buffers: TensorTree = {}
    position = tensors.get("pos_embed.pos_embed")
    if position is None:
        raise ValueError("An SD3 transformer stores its position embedding buffer")
    array = np.asarray(position, np.float32)
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError(f"The position buffer must be [1, tokens, width], got {array.shape}")
    buffers["pos_embed"] = np.ascontiguousarray(array)
    layouts += (WeightLayout("transformer/pos_embed.pos_embed", (("buffers", "pos_embed"),),
                             tuple(position.shape), None),)
    return parameters, buffers, layouts


class QwenImageFields(TypedDict):
    in_channels: int
    out_channels: int
    num_layers: int
    heads: int
    head_dim: int
    context_in_dim: int
    mlp_ratio: int
    axes_dims_rope: tuple[int, ...]
    eps: float
    causal_condition: bool
    dtype: object
    attention_impl: str


def qwen_image_fields(config: Mapping[str, object], *, dtype: DTypeLike | None = "float32",
                      attention_impl="auto") -> QwenImageFields:
    """Read a published `QwenImage21Transformer2DModel` config into native fields.

    Every control the class declares is read. Its pipeline hands the latent
    over unpatched, so a patch size other than one is a checkpoint no
    published pipeline drives and is refused rather than folded.
    """
    from dew.interop.pretrained import resolve_dtype

    if records.integer(config.get("patch_size", 1), "patch_size") != 1:
        raise ValueError("Qwen-Image 2.1's pipeline reads its latent unpatched; patch_size must be 1")
    channels = records.integer(config.get("in_channels", 64), "in_channels")
    out_channels = config.get("out_channels", 64)
    axes = config.get("axes_dims_rope", (16, 56, 56))
    if (not isinstance(axes, (list, tuple)) or len(axes) != 3
            or any(type(size) is not int or size % 2 for size in axes)):
        raise ValueError("axes_dims_rope must be three even channel counts")
    head_dim = records.integer(config.get("attention_head_dim", 128), "attention_head_dim")
    if sum(axes) != head_dim:
        raise ValueError(f"axes_dims_rope {tuple(axes)} must cover the {head_dim} head channels")
    return QwenImageFields(
        in_channels=channels,
        out_channels=channels if out_channels is None else records.integer(out_channels, "out_channels"),
        num_layers=records.integer(config.get("num_layers", 32), "num_layers"),
        heads=records.integer(config.get("num_attention_heads", 32), "num_attention_heads"),
        head_dim=head_dim,
        context_in_dim=records.integer(config.get("context_in_dim", 4096), "context_in_dim"),
        mlp_ratio=records.integer(config.get("mlp_ratio", 3), "mlp_ratio"),
        axes_dims_rope=tuple(axes), eps=records.number(config.get("eps", 1e-6), "eps"),
        causal_condition=records.boolean(config.get("causal_condition", True), "causal_condition"),
        dtype=resolve_dtype(dtype), attention_impl=attention_impl)


_QWEN_IMAGE_EMBEDDERS = {
    "img_in": ("img_in",),
    "proj_out": ("proj_out",),
    "modulation.1": ("modulation",),
    "norm_out.linear": ("norm_out_linear",),
    "txt_in.in_layer": ("txt_in", "in_layer"),
    "txt_in.out_layer": ("txt_in", "out_layer"),
    "time_text_embed.timestep_embedder.linear_1": ("timestep_embedder_linear_1",),
    "time_text_embed.timestep_embedder.linear_2": ("timestep_embedder_linear_2",),
}


def _qwen_image_path(name: str) -> tuple[str, ...]:
    """Return the `QwenImageTransformer` path for a published Qwen-Image 2.1 tensor name.

    The class holds no bias anywhere, so a bias is an unknown name. The text
    projection's norm stores its scale zero-centred, which the native norm
    reads the same way.
    """
    parts = name.split(".")
    leaf, stem = parts[-1], ".".join(parts[:-1])
    if leaf != "weight":
        raise ValueError(f"unknown tensor name {name!r}")
    if stem in _QWEN_IMAGE_EMBEDDERS:
        return (*_QWEN_IMAGE_EMBEDDERS[stem], "kernel")
    if stem == "txt_in.text_norm":
        return ("txt_in", "text_norm", "scale")
    if parts[0] == "transformer_blocks" and parts[1].isdigit():
        block = (f"transformer_blocks_{parts[1]}",)
        rest = parts[2:-1]
        if rest == ["attn", "to_out", "0"]:
            return (*block, "attn", "to_out_0", "kernel")
        if len(rest) == 2 and rest[0] == "img_mlp" and rest[1] in ("proj", "gate_layer", "out"):
            return (*block, "img_mlp", rest[1], "kernel")
        if len(rest) > 1 and rest[0] == "attn":
            return _dit_attention(block, "attn", rest[1:], leaf, name, joint=False)
    raise ValueError(f"unknown tensor name {name!r}")


def translate_qwen_image_weights(tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32"
                                 ) -> tuple[TensorTree, tuple[WeightLayout, ...]]:
    """Map Qwen-Image 2.1 transformer tensors into a parameter tree and the
    layouts that invert it. Its rotary table is computed from positions, so
    it stores no buffer."""
    return record_layouts("transformer", tensors, _qwen_image_path, ("params",),
                          param_dtype=param_dtype)


def component_tensors(directory: Path, component: str) -> dict[str, np.ndarray]:
    """Read one published component's weights: the shards its index names or
    its one weights file, never a precision variant beside them."""
    return read_weights(directory / component)


def record_layouts(component: str, tensors: Mapping[str, np.ndarray],
                   path_of: Callable[[str], tuple[str, ...] | None], prefix: tuple[str, ...], *,
                   param_dtype: str = "float32") -> tuple[TensorTree, tuple[WeightLayout, ...]]:
    """Map a component's tensors into a parameter tree and the layouts that invert it.

    `path_of` gives each tensor's tree path, or None to skip it. Kernels are
    transposed and the transpose is recorded in the layout, so `WeightLayout.export`
    writes the tensor back unchanged. Each leaf is cast to `param_dtype` before
    the transpose; buffers and scoring state are the caller's to keep in FP32.
    """
    from dew.interop.pretrained import WeightLayout
    parameters: TensorTree = {}
    layouts = []
    owners: dict[tuple[str, ...], str] = {}
    for name, tensor in tensors.items():
        path = path_of(name)
        if path is None:
            continue
        _source_alias(tensors, owners, path, name)
        array = checkpoint_array(tensor, param_dtype)
        transpose = None
        if path[-1] == "kernel":
            transpose = (3, 2, 0, 1) if array.ndim == 4 else (1, 0)
            array = array.transpose(2, 3, 1, 0) if array.ndim == 4 else array.T
        _insert(parameters, path, np.ascontiguousarray(array))
        layouts.append(WeightLayout(f"{component}/{name}", ((*prefix, *path),), tensor.shape, transpose))
    return parameters, tuple(layouts)


def flax_component_parameters(component: str, tensors: Mapping[str, np.ndarray]) -> TensorTree:
    """Map canonical tensor names into the storage tree a Flax checkpoint declares.

    This translates files only; no foreign model implementation is imported.
    """
    parameters: TensorTree = {}
    for name, tensor in tensors.items():
        array = np.asarray(tensor)
        parts = name.split(".")
        leaf = parts[-1]
        if component == "vae":
            from dew.nn.autoencoders.vae import _vae_path
            path = _vae_path(name, array.ndim)
        else:
            if leaf in ("weight", "bias"):
                parts.pop()
                leaf = ("bias" if leaf == "bias" else "embedding" if parts[-1] in ("token_embedding", "position_embedding")
                        else "scale" if array.ndim == 1 else "kernel")
            else:
                parts.pop()
            if component == "unet":
                combined = []
                index = 0
                while index < len(parts):
                    if index + 1 < len(parts) and parts[index + 1].isdigit():
                        combined.append(parts[index] + "_" + parts[index + 1])
                        index += 2
                    else:
                        combined.append(parts[index])
                        index += 1
                parts = combined
            path = (*parts, leaf)
        if path[-1] == "kernel":
            array = array.transpose(2, 3, 1, 0) if array.ndim == 4 else array.T
        _insert(parameters, path, np.ascontiguousarray(array))
    return parameters


def write_flax_component(directory: Path, component: str, tensors: Mapping[str, np.ndarray]) -> None:
    """Write `tensors` as the msgpack file a Flax component of `component` ships.

    A source that declares a Flax class keeps that format beside the
    safetensors, so a reader of either one finds what it expects.
    """
    from flax.serialization import to_bytes
    filename = "diffusion_flax_model.msgpack" if component in ("unet", "vae") else "flax_model.msgpack"
    (directory / component / filename).write_bytes(to_bytes(flax_component_parameters(component, tensors)))


def save_source(source, values, destination: Path) -> None:
    """Write a native diffusion bundle back to its published directory layout."""
    import shutil

    from dew.inputs.diffusion import QwenImageConditioner
    from dew.interop.safetensors_io import write_file
    destination.mkdir(parents=True, exist_ok=True)
    config = dict(source.config)
    index = dict(config.pop("model_index"))
    if source.inputs is not None:
        height, width = source.inputs.sample.shape[-3:-1]
        index.update(dew_height=height, dew_width=width)
    (destination / "model_index.json").write_text(json.dumps(index, indent=2))
    component_configs = {name: value for name, value in config.items() if isinstance(value, Mapping)}
    for name, component_config in component_configs.items():
        folder = destination / name
        folder.mkdir(exist_ok=True)
        file = {"scheduler": "scheduler_config.json", "feature_extractor": "preprocessor_config.json"}.get(name, "config.json")
        (folder / file).write_text(json.dumps(component_config, indent=2))
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for layout in source.weight_layouts:
        component, _, name = layout.name.partition("/")
        grouped.setdefault(component, {})[name] = layout.export(values)
    for component, tensors in grouped.items():
        weights = "diffusion_pytorch_model" if component in ("unet", "vae", "transformer") else "model"
        write_file(tensors, destination / component / f"{weights}.safetensors", metadata={"format": "pt"})
        declared = index.get(component)
        if isinstance(declared, (list, tuple)) and len(declared) == 2 and isinstance(declared[1], str) and declared[1].startswith("Flax"):
            write_flax_component(destination, component, tensors)
    encoder = source.inputs.conditions["conditioning"].encoder
    if isinstance(encoder, QwenImageConditioner):
        # The processor is read, never trained: its files go out as they came.
        shutil.copytree(Path(source.source) / "processor", destination / "processor",
                        dirs_exist_ok=True)
        return
    for name, tokenizer in zip(encoder.names, encoder.tokenizers, strict=True):
        folder = destination / ("tokenizer" + name.removeprefix("text_encoder"))
        tokenizer.save_pretrained(folder)
        # A CLIP tokenizer's own vocabulary and merges beside its config.
        tokenizer.backend_tokenizer.model.save(str(folder))
    if encoder.t5 is not None:
        # The T5 tokenizer ships one file, which `save_pretrained` writes, in
        # the slot its own family keeps it.
        encoder.t5.tokenizer.save_pretrained(
            destination / ("tokenizer" + encoder.t5.name.removeprefix("text_encoder")))
