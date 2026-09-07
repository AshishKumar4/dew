"""Checkpoint-file translation for native latent diffusion models.

Source configurations and safetensors are data. No external model or scheduler
implementation is imported by this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import json
import numpy as np

from dew.nn.backbones.unet_condition import UNet2DCondition, UNetStage
from dew.registry import resolve_dtype
from dew.interop.safetensors_io import load_params


def unet_fields(config: Mapping, *, dtype="float32", attention_impl="auto") -> dict:
    widths = tuple(config["block_out_channels"])
    count = len(widths)
    def stages(value):
        return (value,) * count if isinstance(value, (int, bool)) else tuple(value)
    heads = stages(config.get("num_attention_heads") or config.get("attention_head_dim", 8))
    depths = stages(config.get("transformer_layers_per_block", 1))
    only_cross = stages(config.get("only_cross_attention", False))
    down = config["down_block_types"]
    up = config["up_block_types"]
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
    return dict(stages=tuple(UNetStage(int(width), int(head), int(depth), attended, bool(cross_only))
                            for width, head, depth, attended, cross_only in zip(widths, heads, depths, cross, only_cross)),
                in_channels=int(config["in_channels"]), out_channels=int(config["out_channels"]),
                blocks_per_level=int(config.get("layers_per_block", 2)),
                linear_projection=bool(config.get("use_linear_projection", False)),
                additional_time_features=int(config["addition_time_embed_dim"]) if addition else 0,
                middle_attention=config.get("mid_block_type", "UNetMidBlock2DCrossAttn") is not None,
                frequency_shift=float(config.get("freq_shift", 0)), cosine_first=bool(config.get("flip_sin_to_cos", True)),
                dropout=float(config.get("dropout", 0)), dtype=resolve_dtype(dtype),
                attention_impl=None if attention_impl == "reference" else attention_impl)


def _unet_path(name: str, rank: int) -> tuple[str, ...]:
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


def translate_unet_weights(tensors: Mapping[str, np.ndarray], model: UNet2DCondition):
    """Native parameters and reversible source layouts, without random leaves."""
    from dew.interop.pretrained import WeightLayout
    parameters, layouts = {}, []
    for name, tensor in tensors.items():
        tensor = np.asarray(tensor)
        path = _unet_path(name, tensor.ndim)
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
        node = parameters
        for key in path[:-1]:
            node = node.setdefault(key, {})
        if path[-1] in node:
            raise ValueError(f"Two source tensors map to {'/'.join(path)}")
        node[path[-1]] = value
        layouts.append(WeightLayout("unet/" + name, (("params", *path),), tensor.shape, transpose))
    return parameters, tuple(layouts)


def component_tensors(directory: Path, component: str) -> dict:
    """Read published safetensors, including sharded component directories."""
    folder = directory / component
    weights = "diffusion_pytorch_model" if component in ("unet", "vae") else "model"
    index = folder / f"{weights}.safetensors.index.json"
    if index.is_file():
        record = json.loads(index.read_text())
        shards = sorted(set(record["weight_map"].values()))
        result = {}
        for name in shards:
            values = load_params(folder / name)
            if result.keys() & values.keys():
                raise ValueError(f"Duplicate tensors across {component} shards")
            result.update(values)
        return result
    return load_params(folder / f"{weights}.safetensors")


def record_layouts(component: str, tensors: Mapping[str, np.ndarray], path_of, prefix: tuple[str, ...]):
    """Translate one component's published tensors and record their inverse layouts."""
    from dew.interop.pretrained import WeightLayout
    parameters: dict = {}
    layouts = []
    for name, tensor in tensors.items():
        path = path_of(name)
        if path is None:
            continue
        array = np.asarray(tensor, np.float32)
        transpose = None
        if path[-1] == "kernel":
            transpose = (3, 2, 0, 1) if array.ndim == 4 else (1, 0)
            array = array.transpose(2, 3, 1, 0) if array.ndim == 4 else array.T
        node = parameters
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = np.ascontiguousarray(array)
        layouts.append(WeightLayout(f"{component}/{name}", ((*prefix, *path),), tensor.shape, transpose))
    return parameters, tuple(layouts)


def save_source(source, values, destination: Path) -> None:
    """Write a native diffusion bundle back to its published directory layout."""
    from dew.interop.safetensors_io import _safetensors
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
        weights = "diffusion_pytorch_model" if component in ("unet", "vae") else "model"
        _safetensors().save_file(tensors, destination / component / f"{weights}.safetensors")
    encoder = source.inputs.conditions["conditioning"].encoder
    for name, tokenizer in zip(encoder.names, encoder.tokenizers):
        tokenizer.save_pretrained(destination / ("tokenizer" + name.removeprefix("text_encoder")))
