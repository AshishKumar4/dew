"""Translate published latent diffusion checkpoints into native model values.

Source configurations and safetensors are read as data. No external model or
scheduler implementation is imported by this module.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jax.typing import DTypeLike

from dew import records
from dew.interop import weights
from dew.interop.config_records import NativeFields, native_fields
from dew.interop.safetensors_io import read_weights
from dew.interop.weights import ParamTree, checkpoint_dtype, insert, source_alias
from dew.nn.backbones.flux import FluxTransformer
from dew.nn.backbones.flux2 import Flux2Transformer
from dew.nn.backbones.qwen_image import QwenImageTransformer
from dew.nn.backbones.sd3 import SD3Transformer
from dew.nn.backbones.unet_condition import UNet2DCondition, UNetStage
from dew.nn.backbones.wan import WanTransformer
from dew.nn.backbones.z_image import ZImageTransformer
from dew.registry import resolve_dtype

if TYPE_CHECKING:
    from dew.interop.streaming import LazyTree, WeightLayout


def flag(config: Mapping[str, object], name: str, *, default: bool) -> bool:
    """One boolean diffusers config field, read as diffusers reads it: a
    missing field is the class's default, and a null one is False, since
    diffusers only tests the value's truth (a null `flip_sin_to_cos` embeds
    sin first although the default is True). SDXL's UNets store
    `upcast_attention: null`."""
    if name not in config:
        return default
    value = config[name]
    return False if value is None else records.boolean(value, name)


def unet_fields(config: Mapping[str, object], *, dtype: DTypeLike | None = "float32",
                attention_impl="auto") -> NativeFields[UNet2DCondition]:
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
    depths = tuple(
        records.integer(value, "transformer depth")
        for value in per_stage(config.get("transformer_layers_per_block", 1), "transformer depth")
    )
    only_cross = tuple(
        records.boolean(value, "only_cross_attention")
        for value in per_stage(config.get("only_cross_attention", False), "only_cross_attention")
    )
    down, up = config["down_block_types"], config["up_block_types"]
    if (
        not isinstance(down, (list, tuple))
        or not isinstance(up, (list, tuple))
        or len(down) != count
        or len(up) != count
    ):
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
        # A null flag is False to diffusers, the value the UNet computes.
        value = (
            flag(config, name, default=expected) if isinstance(expected, bool) else config.get(name, expected)
        )
        if value != expected:
            raise ValueError(f"Native UNet cannot honor active {name}={config[name]!r}")
    if config.get("time_embedding_dim") not in (None, widths[0] * 4):
        raise ValueError("Native UNet requires a time embedding four times its first width")
    groups = records.integer(config.get("norm_num_groups", 32), "norm_num_groups")
    epsilon = records.number(config.get("norm_eps", 1e-5), "norm_eps")
    if groups < 1 or epsilon <= 0 or any(width < 1 or width % groups for width in widths):
        raise ValueError(
            "UNet normalization requires positive epsilon and widths divisible by its group count"
        )
    if any(
        head < 1 or width % head or depth < 1
        for width, head, depth in zip(widths, heads, depths, strict=True)
    ):
        raise ValueError("UNet stage heads must divide their width and transformer depth must be positive")
    source_name = config.get("_class_name", "UNet2DConditionModel")
    if source_name not in ("UNet2DConditionModel", "FlaxUNet2DConditionModel"):
        raise ValueError(f"Unsupported UNet source class: {source_name}")
    flax_semantics = source_name == "FlaxUNet2DConditionModel"
    if flax_semantics and (groups != 32 or epsilon != 1e-5):
        raise ValueError("The published Flax UNet has fixed normalization groups and epsilon")
    return native_fields(UNet2DCondition)(
        stages=tuple(
            UNetStage(width, head, depth, attended, cross_only)
            for width, head, depth, attended, cross_only in zip(
                widths, heads, depths, cross, only_cross, strict=True
            )
        ),
        in_channels=records.integer(config["in_channels"], "in_channels"),
        out_channels=records.integer(config["out_channels"], "out_channels"),
        blocks_per_level=records.integer(config.get("layers_per_block", 2), "layers_per_block"),
        linear_projection=flag(config, "use_linear_projection", default=False),
        additional_time_features=records.integer(config["addition_time_embed_dim"], "addition_time_embed_dim")
        if addition
        else 0,
        middle_attention=middle is not None,
        frequency_shift=records.number(config.get("freq_shift", 0), "freq_shift"),
        cosine_first=flag(config, "flip_sin_to_cos", default=True),
        dropout=records.number(config.get("dropout", 0), "dropout"),
        norm_groups=groups,
        norm_epsilon=epsilon,
        attention_norm_epsilon=1e-5 if flax_semantics else 1e-6,
        approximate_gelu=flax_semantics,
        dtype=resolve_dtype(dtype),
        attention_impl=attention_impl,
    )


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
                path.append("to_out_0")
                parts = []
            elif projection in ("to_q", "to_k", "to_v"):
                path.append(projection)
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
                           param_dtype: str = "float32", lazy: bool = False
                           ) -> tuple[LazyTree, tuple[WeightLayout, ...]]:
    """Map UNet tensors into a parameter tree and the layouts that invert it.

    Each leaf is cast to `param_dtype` before its transpose. Attention kernels
    are also reshaped to per-head axes, which is why this does not go through
    `weights.record_layouts`. Read eagerly, a transposed kernel is a view of the
    stored tensor. With `lazy` every other linear kernel and every
    untransposed leaf is a `SourceLeaf`, read when it is placed; a reshaped
    attention kernel and a convolution are read whole either way.
    """
    from dew.interop.streaming import SourceLeaf, WeightLayout
    parameters: LazyTree = {}
    layouts = []
    owners: dict[tuple[str, ...], str] = {}
    for name, tensor in tensors.items():
        path = _unet_path(name, tensor.ndim)
        source_alias(tensors, owners, path, name)
        stored = np.asarray(tensor)
        dtype = checkpoint_dtype(stored.dtype, param_dtype)
        kernel = path[-1] == "kernel"
        attention = "self_attention" in path or "cross_attention" in path
        if lazy and not (kernel and (stored.ndim == 4 or attention)):
            layouts.append(WeightLayout("unet/" + name, (("params", *path),), stored.shape,
                                        (1, 0) if kernel else None))
            if owners[path] == name:
                # Equal source aliases add export names without another leaf.
                insert(parameters, path, SourceLeaf((stored,), dtype, transposed=kernel), name)
            continue
        value = stored.astype(dtype, copy=False)
        transpose = None
        if kernel:
            value = value.transpose(2, 3, 1, 0) if value.ndim == 4 else value.T
            transpose = (3, 2, 0, 1) if stored.ndim == 4 else (1, 0)
            if attention:
                root = path[0]
                stage = (
                    model.stages[int(root.removeprefix("down_"))]
                    if root.startswith("down_")
                    else model.stages[::-1][int(root.removeprefix("up_"))]
                    if root.startswith("up_")
                    else model.stages[-1]
                )
                if path[-2] in ("to_q", "to_k", "to_v"):
                    value = value.reshape(value.shape[0], stage.heads, stage.features // stage.heads)
                    transpose = (1, 2, 0)
                else:
                    value = value.reshape(stage.heads, stage.features // stage.heads, stage.features)
                    transpose = (2, 0, 1)
        insert(parameters, path, value, name)
        layouts.append(WeightLayout("unet/" + name, (("params", *path),), stored.shape, transpose))
    return parameters, tuple(layouts)



def sd3_fields(config: Mapping[str, object], *, dtype: DTypeLike | None = "float32",
               attention_impl="auto") -> NativeFields[SD3Transformer]:
    """Read a published `SD3Transformer2DModel` config into native model fields.

    Every geometry control the source declares is read. A control whose active
    meaning this model does not carry is refused rather than dropped, so a
    checkpoint that means something else cannot load as if it did not.
    """
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
    return native_fields(SD3Transformer)(
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
            return (*block, rest[0], "up_proj", _dit_leaf(leaf))
        if rest[1:] == ["net", "2"]:
            return (*block, rest[0], "down_proj", _dit_leaf(leaf))
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


def flux_fields(config: Mapping[str, object], *, dtype: DTypeLike | None = "float32",
                attention_impl="auto") -> NativeFields[FluxTransformer]:
    """Read a published `FluxTransformer2DModel` config into native model fields.

    Every geometry control the source declares is read, including whether it
    embeds its distilled guidance, which changes what the model takes as an
    input rather than only which tensors it holds.
    """
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
    return native_fields(FluxTransformer)(
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



def flux2_fields(config: Mapping[str, object], *, dtype: DTypeLike | None = "float32",
                 attention_impl="auto") -> NativeFields[Flux2Transformer]:
    """Read a published `Flux2Transformer2DModel` config into native model
    fields, refusing a patch size the pipeline does not use."""
    if records.integer(config.get("patch_size", 1), "patch_size") != 1:
        raise ValueError("FLUX.2's pipeline folds its latent 2x2 itself; the transformer's patch_size is 1")
    channels = records.integer(config.get("in_channels", 128), "in_channels")
    out_channels = config.get("out_channels")
    axes = config.get("axes_dims_rope", (32, 32, 32, 32))
    heads = records.integer(config.get("num_attention_heads", 48), "num_attention_heads")
    head_dim = records.integer(config.get("attention_head_dim", 128), "attention_head_dim")
    if not isinstance(axes, (list, tuple)) or any(type(size) is not int or size % 2 for size in axes):
        raise ValueError("axes_dims_rope must be a sequence of even channel counts")
    if sum(axes) != head_dim or len(axes) != 4:
        raise ValueError(f"axes_dims_rope {tuple(axes)} must split the {head_dim} head channels over the "
                         f"four axes the pipeline's ids carry")
    return native_fields(Flux2Transformer)(
        in_channels=channels,
        out_channels=channels if out_channels is None else records.integer(out_channels, "out_channels"),
        num_layers=records.integer(config.get("num_layers", 8), "num_layers"),
        num_single_layers=records.integer(config.get("num_single_layers", 48), "num_single_layers"),
        heads=heads, head_dim=head_dim,
        joint_attention_dim=records.integer(config.get("joint_attention_dim", 15360), "joint_attention_dim"),
        timestep_guidance_channels=records.integer(config.get("timestep_guidance_channels", 256),
                                                   "timestep_guidance_channels"),
        mlp_ratio=records.number(config.get("mlp_ratio", 3.0), "mlp_ratio"), axes_dims_rope=tuple(axes),
        rope_theta=records.number(config.get("rope_theta", 2000), "rope_theta"),
        eps=records.number(config.get("eps", 1e-6), "eps"),
        guidance_embeds=records.boolean(config.get("guidance_embeds", True), "guidance_embeds"),
        dtype=resolve_dtype(dtype), attention_impl=attention_impl)


_FLUX2_EMBEDDERS = {
    "x_embedder": ("x_embedder",),
    "context_embedder": ("context_embedder",),
    "proj_out": ("proj_out",),
    "norm_out.linear": ("norm_out", "linear"),
    "double_stream_modulation_img.linear": ("double_stream_modulation_img", "linear"),
    "double_stream_modulation_txt.linear": ("double_stream_modulation_txt", "linear"),
    "single_stream_modulation.linear": ("single_stream_modulation", "linear"),
    "time_guidance_embed.timestep_embedder.linear_1": ("timestep_embedder_linear_1",),
    "time_guidance_embed.timestep_embedder.linear_2": ("timestep_embedder_linear_2",),
    "time_guidance_embed.guidance_embedder.linear_1": ("guidance_embedder_linear_1",),
    "time_guidance_embed.guidance_embedder.linear_2": ("guidance_embedder_linear_2",),
}


def _flux2_path(name: str) -> tuple[str, ...]:
    """Return the `Flux2Transformer` path for a published FLUX.2 tensor name."""
    parts = name.split(".")
    leaf, stem = parts[-1], ".".join(parts[:-1])
    if stem in _FLUX2_EMBEDDERS:
        return (*_FLUX2_EMBEDDERS[stem], _dit_leaf(leaf))
    if len(parts) > 3 and parts[1].isdigit():
        block, rest = (f"{parts[0]}_{parts[1]}",), parts[2:-1]
        if parts[0] == "transformer_blocks":
            if len(rest) == 2 and rest[0] in ("ff", "ff_context") and rest[1] in ("linear_in", "linear_out"):
                return (*block, rest[0], "gate_up_proj" if rest[1] == "linear_in" else "down_proj",
                        _dit_leaf(leaf))
            if len(rest) > 1 and rest[0] == "attn":
                return _dit_attention(block, "attn", rest[1:], leaf, name, joint=True)
        # A single block's fused maps and head norms sit under `attn` in the
        # source, whose parallel attention holds the feed-forward too; here
        # they are the block's own, the output map named as Flux's is.
        if parts[0] == "single_transformer_blocks" and rest[:1] == ["attn"] and len(rest) == 2:
            if rest[1] in ("to_qkv_mlp_proj", "to_out"):
                return (*block, "proj_fused" if rest[1] == "to_out" else rest[1], _dit_leaf(leaf))
            if rest[1] in ("norm_q", "norm_k") and leaf == "weight":
                return (*block, rest[1], "scale")
    raise ValueError(f"unknown tensor name {name!r}")


def translate_flux2_weights(tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32",
                            lazy: bool = False
                            ) -> tuple[LazyTree, tuple[WeightLayout, ...]]:
    """Map FLUX.2 tensors into a parameter tree and the layouts that invert
    it; its rotary tables are computed from ids, so it stores no buffer."""
    return weights.record_layouts("transformer", tensors, _flux2_path, ("params",), param_dtype=param_dtype,
                          lazy=lazy)


def z_image_fields(config: Mapping[str, object], *, dtype: DTypeLike | None = "float32",
                   attention_impl="auto") -> NativeFields[ZImageTransformer]:
    """Read a published `ZImageTransformer2DModel` config into native model
    fields, refusing what the port does not compute: another patch size,
    grouped keys and values, no query and key norms, or the Omni model's
    SigLIP stream."""
    if records.integers(config.get("all_patch_size", (2,)), "all_patch_size") != (2,) or records.integers(
            config.get("all_f_patch_size", (1,)), "all_f_patch_size") != (1,):
        raise ValueError("the port computes Z-Image's 2x2 patches of one frame")
    heads = records.integer(config.get("n_heads", 30), "n_heads")
    if records.integer(config.get("n_kv_heads", heads), "n_kv_heads") != heads:
        raise ValueError("the source's attention has as many key and value heads as query heads")
    if not records.boolean(config.get("qk_norm", True), "qk_norm"):
        raise ValueError("the port computes Z-Image with its query and key norms")
    if config.get("siglip_feat_dim") is not None:
        raise ValueError("the SigLIP stream belongs to Z-Image Omni, which the port does not compute")
    dim = records.integer(config.get("dim", 3840), "dim")
    axes = records.integers(config.get("axes_dims", (32, 48, 48)), "axes_dims")
    lengths = records.integers(config.get("axes_lens", (1024, 512, 512)), "axes_lens")
    if sum(axes) != dim // heads or len(axes) != 3 or len(lengths) != 3:
        raise ValueError(f"axes_dims {axes} must split the {dim // heads} head channels over three axes")
    return native_fields(ZImageTransformer)(
        in_channels=records.integer(config.get("in_channels", 16), "in_channels"),
        dim=dim,
        n_layers=records.integer(config.get("n_layers", 30), "n_layers"),
        n_refiner_layers=records.integer(config.get("n_refiner_layers", 2), "n_refiner_layers"),
        n_heads=heads,
        norm_eps=records.number(config.get("norm_eps", 1e-5), "norm_eps"),
        cap_feat_dim=records.integer(config.get("cap_feat_dim", 2560), "cap_feat_dim"),
        rope_theta=records.number(config.get("rope_theta", 256.0), "rope_theta"),
        t_scale=records.number(config.get("t_scale", 1000.0), "t_scale"),
        axes_dims=axes,
        axes_lens=lengths,
        dtype=resolve_dtype(dtype),
        attention_impl=attention_impl,
    )


_Z_IMAGE_MODULES = {
    "all_x_embedder.2-1": ("x_embedder",), "all_final_layer.2-1.linear": ("final_linear",),
    "all_final_layer.2-1.adaLN_modulation.1": ("final_modulation",), "t_embedder.mlp.0": ("t_embedder_1",),
    "t_embedder.mlp.2": ("t_embedder_2",), "cap_embedder.1": ("cap_embedder",),
}
_Z_IMAGE_BLOCK = {
    "attention.to_q": ("attention", "to_q"), "attention.to_k": ("attention", "to_k"),
    "attention.to_v": ("attention", "to_v"), "attention.to_out.0": ("attention", "to_out_0"),
    "attention.norm_q": ("attention", "norm_q"), "attention.norm_k": ("attention", "norm_k"),
    "feed_forward.w1": ("feed_forward", "gate_proj"), "feed_forward.w2": ("feed_forward", "down_proj"),
    "feed_forward.w3": ("feed_forward", "up_proj"), "adaLN_modulation.0": ("modulation",),
    **{norm: (norm,) for norm in ("attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2")},
}


def _z_image_path(name: str, ndim: int) -> tuple[str, ...]:
    """Return the `ZImageTransformer` path for a published Z-Image tensor
    name; a one-axis weight is a norm's scale."""
    if name in ("x_pad_token", "cap_pad_token"):
        return (name,)
    if name == "cap_embedder.0.weight":
        return ("cap_norm", "scale")
    stem, _, leaf = name.rpartition(".")
    if leaf == "weight":
        leaf = "kernel" if ndim == 2 else "scale"
    elif leaf != "bias":
        raise ValueError(f"unknown tensor name {name!r}")
    if stem in _Z_IMAGE_MODULES:
        return (*_Z_IMAGE_MODULES[stem], leaf)
    stack, _, rest = stem.partition(".")
    index, _, inner = rest.partition(".")
    if (
        stack in ("noise_refiner", "context_refiner", "layers")
        and index.isdigit()
        and inner in _Z_IMAGE_BLOCK
    ):
        return (f"{stack}_{index}", *_Z_IMAGE_BLOCK[inner], leaf)
    raise ValueError(f"unknown tensor name {name!r}")


def translate_z_image_weights(tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32",
                              lazy: bool = False
                              ) -> tuple[LazyTree, tuple[WeightLayout, ...]]:
    """Map Z-Image tensors into a parameter tree and the layouts that invert
    it; its rotary table is computed, so it stores no buffer."""
    return weights.record_layouts(
        "transformer", tensors, lambda name: _z_image_path(name, np.ndim(tensors[name])),
        ("params",), param_dtype=param_dtype, lazy=lazy)


def wan_fields(config: Mapping[str, object], *, dtype: DTypeLike | None = "float32",
               attention_impl="auto") -> NativeFields[WanTransformer]:
    """Read a published `WanTransformer3DModel` config into native model
    fields, refusing what the text-to-video port does not compute: the
    image-to-video models' image embedder and added key and value
    projections, and a query and key norm other than across all heads."""
    for name in ("image_dim", "added_kv_proj_dim", "pos_embed_seq_len"):
        if config.get(name) is not None:
            raise ValueError(f"{name} belongs to Wan's image-to-video models, which the port does not "
                             "compute")
    qk_norm = config.get("qk_norm", "rms_norm_across_heads")
    if qk_norm not in (None, "rms_norm_across_heads"):
        raise ValueError(f"the port computes Wan's qk_norm 'rms_norm_across_heads', not {qk_norm!r}")
    patch = records.integers(config.get("patch_size", (1, 2, 2)), "patch_size")
    head_dim = records.integer(config.get("attention_head_dim", 128), "attention_head_dim")
    if len(patch) != 3 or head_dim % 2:
        raise ValueError("Wan's patches are (frames, rows, columns) and its heads rotate channel pairs")
    channels = records.integer(config.get("in_channels", 16), "in_channels")
    out_channels = config.get("out_channels")
    return native_fields(WanTransformer)(
        patch_size=patch,
        num_attention_heads=records.integer(config.get("num_attention_heads", 40), "num_attention_heads"),
        attention_head_dim=head_dim,
        in_channels=channels,
        out_channels=channels if out_channels is None else records.integer(out_channels, "out_channels"),
        text_dim=records.integer(config.get("text_dim", 4096), "text_dim"),
        freq_dim=records.integer(config.get("freq_dim", 256), "freq_dim"),
        ffn_dim=records.integer(config.get("ffn_dim", 13824), "ffn_dim"),
        num_layers=records.integer(config.get("num_layers", 40), "num_layers"),
        cross_attn_norm=records.boolean(config.get("cross_attn_norm", True), "cross_attn_norm"),
        qk_norm=qk_norm,
        eps=records.number(config.get("eps", 1e-6), "eps"),
        rope_max_seq_len=records.integer(config.get("rope_max_seq_len", 1024), "rope_max_seq_len"),
        dtype=resolve_dtype(dtype),
        attention_impl=attention_impl,
    )


_WAN_MODULES = {
    "patch_embedding": ("patch_embedding_3d",), "proj_out": ("proj_out",),
    "condition_embedder.time_embedder.linear_1": ("time_embedder_linear_1",),
    "condition_embedder.time_embedder.linear_2": ("time_embedder_linear_2",),
    "condition_embedder.time_proj": ("time_proj", "linear"),
    "condition_embedder.text_embedder.linear_1": ("text_embedder_linear_1",),
    "condition_embedder.text_embedder.linear_2": ("text_embedder_linear_2",),
}


def _wan_path(name: str) -> tuple[str, ...]:
    """Return the `WanTransformer` path for a published Wan tensor name."""
    parts = name.split(".")
    if name == "scale_shift_table":
        return (name,)
    if parts[0] == "blocks" and len(parts) > 2 and parts[1].isdigit():
        block, rest = (f"blocks_{parts[1]}",), parts[2:]
        if rest == ["scale_shift_table"]:
            return (*block, *rest)
        if rest[0] in ("attn1", "attn2") and len(rest) > 2:
            return _dit_attention(block, rest[0], rest[1:-1], rest[-1], name, joint=True)
        if rest[0] == "norm2" and len(rest) == 2:
            return (*block, "norm2", "scale" if rest[1] == "weight" else _dit_leaf(rest[1]))
        if rest[:2] == ["ffn", "net"] and rest[2:-1] in (["0", "proj"], ["2"]):
            return (*block, "ffn", "up_proj" if rest[2] == "0" else "down_proj", _dit_leaf(rest[-1]))
    stem, _, leaf = name.rpartition(".")
    if stem in _WAN_MODULES:
        return (*_WAN_MODULES[stem], _dit_leaf(leaf))
    raise ValueError(f"unknown tensor name {name!r}")


def translate_wan_weights(tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32",
                          lazy: bool = False
                          ) -> tuple[LazyTree, tuple[WeightLayout, ...]]:
    """Map Wan transformer tensors into a parameter tree and the layouts that
    invert it; its rotary table is computed, so it stores no buffer."""
    return weights.record_layouts("transformer", tensors, _wan_path, ("params",),
                                  param_dtype=param_dtype, lazy=lazy)

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


def translate_flux_weights(tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32",
                           lazy: bool = False
                           ) -> tuple[LazyTree, tuple[WeightLayout, ...]]:
    """Map Flux tensors into a parameter tree and the layouts that invert it.

    Rotary tables are computed from input ids, so Flux stores no positional
    buffer and there is nothing to place outside `params`.
    """
    return weights.record_layouts("transformer", tensors, _flux_path, ("params",),
                                  param_dtype=param_dtype, lazy=lazy)


def translate_sd3_weights(tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32",
                          lazy: bool = False
                          ) -> tuple[LazyTree, ParamTree, tuple[WeightLayout, ...]]:
    """Map SD3 tensors into a parameter tree, its buffers and the inverting layouts.

    The source's position embedding is a persistent sin/cos-initialized
    buffer, so its stored values land in the `buffers` collection: an
    optimizer and an EMA see only `params`, and export writes the stored
    array back unchanged.
    """
    from dew.interop.streaming import WeightLayout

    parameters, layouts = weights.record_layouts(
        "transformer", tensors, _sd3_path, ("params",), param_dtype=param_dtype, lazy=lazy)
    buffers: ParamTree = {}
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


def qwen_image_fields(config: Mapping[str, object], *, dtype: DTypeLike | None = "float32",
                      attention_impl="auto") -> NativeFields[QwenImageTransformer]:
    """Read a published `QwenImage21Transformer2DModel` config into native fields.

    Every control the class declares is read. Its pipeline hands the latent
    over unpatched, so a patch size other than one is a checkpoint no
    published pipeline drives and is refused rather than folded.
    """
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
    return native_fields(QwenImageTransformer)(
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


# `QwenImage21SwiGLUFeedForward` is `out(silu(gate_layer(x)) * proj(x))`, bias
# free: GatedMLP's swiglu under its own projection names.
_QWEN_IMAGE_MLP = {"gate_layer": "gate_proj", "proj": "up_proj", "out": "down_proj"}


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
        if len(rest) == 2 and rest[0] == "img_mlp" and rest[1] in _QWEN_IMAGE_MLP:
            return (*block, "img_mlp", _QWEN_IMAGE_MLP[rest[1]], "kernel")
        if len(rest) > 1 and rest[0] == "attn":
            return _dit_attention(block, "attn", rest[1:], leaf, name, joint=False)
    raise ValueError(f"unknown tensor name {name!r}")


def translate_qwen_image_weights(tensors: Mapping[str, np.ndarray], *, param_dtype: str = "float32",
                                 lazy: bool = False
                                 ) -> tuple[LazyTree, tuple[WeightLayout, ...]]:
    """Map Qwen-Image 2.1 transformer tensors into a parameter tree and the
    layouts that invert it. Its rotary table is computed from positions, so
    it stores no buffer."""
    return weights.record_layouts("transformer", tensors, _qwen_image_path, ("params",),
                          param_dtype=param_dtype, lazy=lazy)


def component_tensors(directory: Path, component: str) -> dict[str, np.ndarray]:
    """Read one published component's weights: the shards its index names or
    its one weights file, never a precision variant beside them."""
    return read_weights(directory / component)


def flax_component_parameters(component: str, tensors: Mapping[str, np.ndarray]) -> ParamTree:
    """Map canonical tensor names into the storage tree a Flax checkpoint declares.

    This translates files only; no foreign model implementation is imported.
    """
    parameters: ParamTree = {}
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
                leaf = (
                    "bias"
                    if leaf == "bias"
                    else "embedding"
                    if parts[-1] in ("token_embedding", "position_embedding")
                    else "scale"
                    if array.ndim == 1
                    else "kernel"
                )
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
        insert(parameters, path, np.ascontiguousarray(array), name)
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
    from dew.interop.safetensors_io import write_file
    destination.mkdir(parents=True, exist_ok=True)
    config = dict(source.config)
    index = dict(config.pop("model_index"))
    if source.inputs is not None:
        shape = source.inputs.sample.shape
        index.update(dew_height=shape[-3], dew_width=shape[-2])
        if len(shape) == 4:
            index.update(dew_frames=shape[0])
    (destination / "model_index.json").write_text(json.dumps(index, indent=2))
    component_configs = {name: value for name, value in config.items() if isinstance(value, Mapping)}
    for name, component_config in component_configs.items():
        folder = destination / name
        folder.mkdir(exist_ok=True)
        file = {"scheduler": "scheduler_config.json", "feature_extractor": "preprocessor_config.json"}.get(
            name, "config.json"
        )
        (folder / file).write_text(json.dumps(component_config, indent=2))
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for layout in source.weight_layouts:
        component, _, name = layout.name.partition("/")
        grouped.setdefault(component, {})[name] = layout.export(values)
    for component, tensors in grouped.items():
        weights = "diffusion_pytorch_model" if component in ("unet", "vae", "transformer") else "model"
        write_file(tensors, destination / component / f"{weights}.safetensors", metadata={"format": "pt"})
        declared = index.get(component)
        if (
            isinstance(declared, (list, tuple))
            and len(declared) == 2
            and isinstance(declared[1], str)
            and declared[1].startswith("Flax")
        ):
            write_flax_component(destination, component, tensors)
    source.inputs.conditions["conditioning"].encoder.save_assets(destination)
